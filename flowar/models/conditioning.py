from __future__ import annotations

import contextlib
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flowar.models.embeddings import MLPProjector

_COND_NUM_TOKENS = 257  # 1 CLS + 256 patch tokens


def _load_point_encoder_weights(encoder: nn.Module, path: str) -> None:
    """Load a complete encoder from a bare state dict or a wrapped Uni3D checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for field in ("module", "state_dict", "model", "model_state_dict"):
            if isinstance(state.get(field), dict):
                state = state[field]
                break
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError(f"Invalid point-cloud checkpoint {path}: expected a state dictionary")

    normalized = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        if key in normalized:
            raise ValueError(f"Ambiguous point-cloud checkpoint {path}: duplicate key {key}")
        normalized[key] = value

    prefix = "point_encoder."
    if any(key.startswith(prefix) for key in normalized):
        normalized = {
            key[len(prefix) :]: value for key, value in normalized.items() if key.startswith(prefix)
        }
    expected = encoder.state_dict()
    missing = sorted(expected.keys() - normalized.keys())
    if missing:
        raise ValueError(f"Incomplete point-cloud checkpoint {path}: missing {missing[:5]}")
    encoder.load_state_dict({key: normalized[key] for key in expected}, strict=True)


class FrozenEncoderAdapter(nn.Module):
    """Train the projection without changing frozen encoder statistics or dropout."""

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            encoder = self.pc_encoder if hasattr(self, "pc_encoder") else self.encoder
            encoder.eval()
        return self


class PointCloudCondAdapter(FrozenEncoderAdapter):
    """Uni3D point cloud encoder → fixed 257-token continuous condition prefix via 2-layer MLP.

    With num_group=256 the encoder produces [B, 257, pc_feat_dim] (1 CLS + 256 patches).
    A shared per-token MLPProjector maps pc_feat_dim → hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        ckpt_path: str = "",
        freeze_encoder: bool = True,
        pc_model: str = "eva02_base_patch14_448",
        pc_feat_dim: int = 768,
        embed_dim: int = 1024,
        group_size: int = 32,
        num_group: int = 256,
        pc_encoder_dim: int = 512,
        patch_dropout: float = 0.0,
    ):
        super().__init__()
        if freeze_encoder and not ckpt_path:
            raise ValueError(
                "A pretrained checkpoint is required to freeze the point-cloud encoder"
            )
        import timm

        from flowar.models.point_encoder import PointcloudEncoder

        args = types.SimpleNamespace(
            pc_feat_dim=pc_feat_dim,
            embed_dim=embed_dim,
            group_size=group_size,
            num_group=num_group,
            pc_encoder_dim=pc_encoder_dim,
            patch_dropout=patch_dropout,
        )
        transformer = timm.create_model(pc_model, drop_path_rate=0.0)
        self.pc_encoder = PointcloudEncoder(transformer, args)
        if ckpt_path:
            _load_point_encoder_weights(self.pc_encoder, ckpt_path)

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.pc_encoder.requires_grad_(False).eval()

        self.proj = MLPProjector(pc_feat_dim, hidden_size, hidden_size * 2, nn.GELU)
        nn.init.xavier_normal_(self.proj.net[0].weight)
        nn.init.zeros_(self.proj.net[0].bias)
        nn.init.normal_(self.proj.net[2].weight, std=0.02)
        nn.init.zeros_(self.proj.net[2].bias)

    def _encode_patches(self, pts: Tensor, colors: Tensor) -> Tensor:
        enc = self.pc_encoder
        _, center, features = enc.group_divider(pts, colors)
        x = enc.encoder2trans(enc.encoder(features))  # [B, G, pc_feat_dim]

        cls = enc.cls_token.expand(x.size(0), -1, -1)
        pos = torch.cat((enc.cls_pos.expand(x.size(0), -1, -1), enc.pos_embed(center)), dim=1)
        x = enc.patch_dropout(torch.cat((cls, x), dim=1) + pos)
        x = enc.visual.pos_drop(x)
        for blk in enc.visual.blocks:
            x = blk(x)
        return enc.visual.norm(x)  # [B, G+1, pc_feat_dim]

    def forward(self, pts: Tensor, colors: Optional[Tensor] = None) -> Tensor:
        """pts: [B, N, 3]; colors: [B, N, 3] or None. Returns [B, 257, hidden_size]."""
        if colors is None:
            colors = torch.zeros_like(pts)
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            patches = self._encode_patches(pts, colors)  # [B, 257, pc_feat_dim]
        return self.proj(patches)  # [B, 257, hidden_size]


class CLIPCondAdapter(FrozenEncoderAdapter):
    """CLIP image or text encoder → fixed 257-token continuous condition prefix via 2-layer MLP.

    Image (ViT-L/14 on 224×224): last_hidden_state is [B, 257, 1024] — naturally 257 tokens.
    Text: last_hidden_state is [B, ≤77, 768] — zero-padded to [B, 257, 768] before projection.
    A shared per-token MLPProjector maps enc_dim → hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        clip_model: str = "openai/clip-vit-large-patch14",
        modality: str = "image",
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained(clip_model)
        if modality == "image":
            self.encoder = clip.vision_model
            enc_dim = clip.config.vision_config.hidden_size
        elif modality == "text":
            self.encoder = clip.text_model
            enc_dim = clip.config.text_config.hidden_size
        else:
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")

        self.modality = modality
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False).eval()

        self.proj = MLPProjector(enc_dim, hidden_size, hidden_size * 2, nn.GELU)
        nn.init.xavier_normal_(self.proj.net[0].weight)
        nn.init.zeros_(self.proj.net[0].bias)
        nn.init.normal_(self.proj.net[2].weight, std=0.02)

    def forward(self, **kwargs) -> Tensor:
        """
        image: pixel_values=[B, 3, H, W]
        text:  input_ids=[B, seq_len], attention_mask=[B, seq_len]
        Returns: [B, 257, hidden_size]
        """
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            tokens = self.encoder(**kwargs).last_hidden_state  # [B, T, enc_dim]
        if tokens.size(1) < _COND_NUM_TOKENS:
            tokens = F.pad(tokens, (0, 0, 0, _COND_NUM_TOKENS - tokens.size(1)))
        return self.proj(tokens)  # [B, 257, hidden_size]


class DINOv3CondAdapter(FrozenEncoderAdapter):
    """DINOv3 ViT-Base image encoder → fixed 257-token continuous condition prefix via 2-layer MLP.

    Loaded via HuggingFace AutoModel from a local model directory.
    Input 256×256: 16×16 = 256 patch tokens.  last_hidden_state layout:
      [CLS, num_register_tokens, patch_0 ... patch_255] → [B, 1+R+256, 768]
    Register tokens are discarded; only CLS (1) + patches (256) = 257 tokens are kept.
    enc_dim=768 (ViT-Base hidden_size) → MLPProjector → hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        dino_model: str,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(dino_model)
        cfg = self.encoder.config
        enc_dim = cfg.hidden_size  # 768
        self.n_register = getattr(cfg, "num_register_tokens", 0)  # 4 for vitb16

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False).eval()

        self.proj = MLPProjector(enc_dim, hidden_size, hidden_size * 2, nn.GELU)
        nn.init.xavier_normal_(self.proj.net[0].weight)
        nn.init.zeros_(self.proj.net[0].bias)
        nn.init.normal_(self.proj.net[2].weight, std=0.02)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """pixel_values: [B, 3, 256, 256]. Returns [B, 257, hidden_size]."""
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            tokens = self.encoder(pixel_values=pixel_values).last_hidden_state
        # tokens: [B, 1 + n_register + N_patches, enc_dim]
        cls = tokens[:, 0:1, :]  # [B, 1, enc_dim]
        patches = tokens[:, 1 + self.n_register :, :]  # [B, 256, enc_dim]
        tokens = torch.cat([cls, patches], dim=1)  # [B, 257, enc_dim]
        return self.proj(tokens)  # [B, 257, hidden_size]
