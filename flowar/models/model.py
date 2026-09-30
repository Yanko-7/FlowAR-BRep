from dataclasses import dataclass, field
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import RopeParameters

from flowar.data.sequence import BREP_VOCAB_SIZE
from flowar.models.embeddings import MLPProjector
from flowar.models.flow_head import MlpDiffHead
from flowar.models.transformer import Transformer


@dataclass
class FlowARBRepConfig:
    """Architecture fields are explicit; misspelled/unknown options fail at construction."""

    model_type: ClassVar[str] = "flowar_brep"
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    hidden_size: int = 768
    dropout: float = 0.1
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    num_hidden_layers: int = 24
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    vocab_size: int = BREP_VOCAB_SIZE
    surface_latent_dim: int = 64
    curve_latent_dim: int = 16
    diff_depth: int = 6
    diff_adaln_depth: int = 2
    diff_model_dim: int = 768
    diff_time_shift: float = 1.0
    diff_P_mean: float = 0.0
    diff_P_std: float = 1.0
    diff_backbone: str = "mlp"
    diff_prediction_type: str = "x_prev"
    diff_batch_mul: int = 1
    diff_surface_batch_mul: int | None = None
    diff_curve_batch_mul: int | None = None
    cond_type: str | None = None
    cond_num_tokens: int = 257
    cond_freeze_encoder: bool = True
    cond_pc_ckpt: str = ""
    cond_clip_model: str = "openai/clip-vit-large-patch14"
    cond_dino_model: str = ""
    rope_parameters: dict = field(init=False)

    def __post_init__(self):
        if self.diff_backbone != "mlp" or self.diff_prediction_type != "x_prev":
            raise ValueError("The reference model uses an MLP flow head with x-prediction")
        if self.num_attention_heads < 1 or self.num_key_value_heads < 1:
            raise ValueError("Attention head counts must be positive")
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Hidden size and attention head counts are incompatible")
        if self.cond_type not in (None, "pointcloud", "clip_text", "dino_image"):
            raise ValueError(f"Unknown conditioning modality: {self.cond_type}")
        if self.cond_type is not None and self.cond_num_tokens != 257:
            raise ValueError("Condition encoders produce exactly 257 prefix tokens")
        self.vocab_size = (self.vocab_size + 127) // 128 * 128
        self.rope_parameters = RopeParameters(rope_theta=100000.0, rope_type="default")
        self.diff_batch_mul = max(1, int(self.diff_batch_mul))
        self.diff_surface_batch_mul = max(
            1, int(self.diff_surface_batch_mul or self.diff_batch_mul)
        )
        self.diff_curve_batch_mul = max(1, int(self.diff_curve_batch_mul or self.diff_batch_mul))


def _compute_ce_loss(head, states, labels, drop_out=0.2):
    dropped_states = F.dropout(states, p=drop_out, training=head.training)
    return F.cross_entropy(head(dropped_states), labels, reduction="none")


def _compute_diff_loss(head, x, cond, batch_mul: int):
    if x.shape[0] == 0 or batch_mul <= 1:
        return head(x, cond)
    # PyTorch native repeat_interleave is cleaner and faster than expand+reshape
    loss_all = head(
        x.repeat_interleave(batch_mul, dim=0),
        cond.repeat_interleave(batch_mul, dim=0),
    )
    return loss_all.view(x.shape[0], batch_mul, *loss_all.shape[1:]).mean(dim=1)


class FlowARBRep(nn.Module):
    def __init__(self, config: FlowARBRepConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.transformer = Transformer(config)
        padded_vocab_size = (BREP_VOCAB_SIZE + 7) // 8 * 8

        self.text_embed = nn.Embedding(padded_vocab_size, self.hidden_size)
        self.text_head = nn.Linear(self.hidden_size, padded_vocab_size, bias=False)

        self.surface_embed = MLPProjector(
            config.surface_latent_dim, self.hidden_size, self.hidden_size * 2
        )
        self.curve_embed = MLPProjector(
            config.curve_latent_dim, self.hidden_size, self.hidden_size * 2
        )

        DiffCls = MlpDiffHead
        diff_kwargs = dict(
            ch_latent=config.diff_model_dim,
            depth=config.diff_depth,
            depth_adaln=config.diff_adaln_depth,
            time_shift=config.diff_time_shift,
            P_mean=config.diff_P_mean,
            P_std=config.diff_P_std,
            prediction_type=config.diff_prediction_type,
        )
        self.surface_diff_head = DiffCls(
            ch_target=config.surface_latent_dim, ch_cond=self.hidden_size, **diff_kwargs
        )
        self.curve_diff_head = DiffCls(
            ch_target=config.curve_latent_dim, ch_cond=self.hidden_size, **diff_kwargs
        )

        self.cond_adapter = None
        if config.cond_type == "pointcloud":
            from flowar.models.conditioning import PointCloudCondAdapter

            self.cond_adapter = PointCloudCondAdapter(
                hidden_size=self.hidden_size,
                ckpt_path=config.cond_pc_ckpt,
                freeze_encoder=config.cond_freeze_encoder,
            )
        elif config.cond_type == "clip_text":
            from flowar.models.conditioning import CLIPCondAdapter

            self.cond_adapter = CLIPCondAdapter(
                hidden_size=self.hidden_size,
                clip_model=config.cond_clip_model,
                modality="text",
                freeze_encoder=config.cond_freeze_encoder,
            )
        elif config.cond_type == "dino_image":
            from flowar.models.conditioning import DINOv3CondAdapter

            self.cond_adapter = DINOv3CondAdapter(
                hidden_size=self.hidden_size,
                dino_model=config.cond_dino_model,
                freeze_encoder=config.cond_freeze_encoder,
            )

        self._init_weights()

    def forward_packed(
        self,
        total_len: int,
        max_seqlen: int,
        cu_seqlens: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_surface_vectors: torch.Tensor | None = None,
        packed_surface_geom_indexes: torch.Tensor | None = None,
        packed_curve_vectors: torch.Tensor | None = None,
        packed_curve_geom_indexes: torch.Tensor | None = None,
        ce_loss_indexes: torch.Tensor | None = None,
        packed_label_ids: torch.Tensor | None = None,
        surface_loss_indexes: torch.Tensor | None = None,
        curve_loss_indexes: torch.Tensor | None = None,
        packed_condition_embed: torch.Tensor | None = None,
        packed_condition_indexes: torch.Tensor | None = None,
        condition_inputs: dict[str, torch.Tensor] | None = None,
    ):
        if condition_inputs is not None:
            if self.cond_adapter is None or packed_condition_embed is not None:
                raise ValueError(
                    "Raw conditioning requires an adapter and no precomputed embedding"
                )
            # Run the trainable projection inside forward so DDP tracks its gradients.
            packed_condition_embed = self.cond_adapter(**condition_inputs).flatten(0, 1)
        device = packed_text_ids.device
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32

        packed_sequence = torch.zeros((total_len, self.hidden_size), device=device, dtype=dtype)

        # The dataset already handles excluding <BOS> from the loss
        if packed_condition_embed is not None and packed_condition_embed.numel() > 0:
            packed_sequence[packed_condition_indexes] = packed_condition_embed.to(dtype)

        packed_sequence[packed_text_indexes] = self.text_embed(packed_text_ids).to(dtype)

        if packed_surface_vectors is not None and packed_surface_vectors.numel() > 0:
            x_s = packed_surface_vectors.flatten(1)
            surf_emb = self.surface_embed(x_s)
            packed_sequence[packed_surface_geom_indexes] = surf_emb

        if packed_curve_vectors is not None and packed_curve_vectors.numel() > 0:
            x_c = packed_curve_vectors.flatten(1)
            curv_emb = self.curve_embed(x_c)
            packed_sequence[packed_curve_geom_indexes] = curv_emb

        hidden_states = self.transformer.forward_train(
            packed_sequence.unsqueeze(0),
            packed_position_ids.unsqueeze(0),
            cu_seqlens,
            max_seqlen,
        )[0]

        ce, surface_loss, curve_loss = None, None, None

        if ce_loss_indexes is not None:
            ce = _compute_ce_loss(self.text_head, hidden_states[ce_loss_indexes], packed_label_ids)

        if (
            surface_loss_indexes is not None
            and packed_surface_vectors is not None
            and surface_loss_indexes.numel() > 0
        ):
            surface_loss = _compute_diff_loss(
                self.surface_diff_head,
                packed_surface_vectors.flatten(1),
                hidden_states[surface_loss_indexes],
                self.config.diff_surface_batch_mul,
            )

        if (
            curve_loss_indexes is not None
            and packed_curve_vectors is not None
            and curve_loss_indexes.numel() > 0
        ):
            curve_loss = _compute_diff_loss(
                self.curve_diff_head,
                packed_curve_vectors.flatten(1),
                hidden_states[curve_loss_indexes],
                self.config.diff_curve_batch_mul,
            )
        return ce, surface_loss, curve_loss

    def forward(self, **kwargs):
        return self.forward_packed(**kwargs)

    def _init_weights(self):
        std = self.config.initializer_range
        nn.init.normal_(self.text_embed.weight, std=std)
        nn.init.normal_(self.text_head.weight, std=std)
        nn.init.normal_(self.surface_embed.net[0].weight, std=std)
        nn.init.normal_(self.curve_embed.net[0].weight, std=std)
        # Zero-initialize the last layer
        nn.init.zeros_(self.surface_embed.net[-1].weight)
        nn.init.zeros_(self.curve_embed.net[-1].weight)
        for module in [self.surface_embed, self.curve_embed]:
            for layer in module.net:
                if hasattr(layer, "bias") and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
