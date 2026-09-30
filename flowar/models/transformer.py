import math

# from sageattention import sageattn
from typing import Optional, Tuple

import torch
import torch.library
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_interface
    from flash_attn import flash_attn_with_kvcache as _fa2_flash_attn_with_kvcache
except ImportError:
    flash_attn_interface = None
    _fa2_flash_attn_with_kvcache = None
from torch.nn import RMSNorm
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from flowar.models.embeddings import KVCache
from flowar.models.layers import SwiGLUFFN


@torch.library.custom_op("flowar::fa2_kvcache", mutates_args={"k_cache", "v_cache"})
def _fa2_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    return _fa2_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        causal=causal,
        window_size=(window_left, window_right),
    )


@_fa2_kvcache.register_fake
def _fa2_kvcache_fake(q, k_cache, v_cache, k, v, cache_seqlens, causal, window_left, window_right):
    return torch.empty_like(q)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    causal=False,
    window_size=(-1, -1),
):
    """
    Flash Attention with KV cache for inference.


    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0]  # assume uniform position across batch

    # Insert new k, v into cache (in-place)
    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    end_pos = pos + T_new

    if T_new == 1:
        # ── Decode path: use full pre-allocated cache so all tensor shapes are
        # static → a single CUDA graph covers every decode step.
        # A boolean validity mask (fixed shape) replaces the dynamic slice.
        max_seq_len = k_cache.shape[1]
        col = torch.arange(max_seq_len, device=q.device)  # (max_seq_len,) — fixed
        valid = col < end_pos  # (max_seq_len,) — fixed
        window = window_size[0]
        if window >= 0:
            valid = valid & (col >= end_pos - (window + 1))
        attn_mask = valid.view(1, 1, 1, max_seq_len)  # (1,1,1,max_seq_len) — fixed

        q_sdpa = q.transpose(1, 2)  # (B, H, 1, D)
        k_sdpa = k_cache.transpose(1, 2)  # (B, Hkv, max_seq_len, D) — fixed!
        v_sdpa = v_cache.transpose(1, 2)
        enable_gqa = H != k_sdpa.size(1)
        y = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, enable_gqa=enable_gqa
        )
        return y.transpose(1, 2)
    else:
        # ── Prefill path: slice to actual length.
        # Only called from the uncompiled forward_inference, dynamic shapes are fine.
        k_full = k_cache[:, :end_pos, :, :]
        v_full = v_cache[:, :end_pos, :, :]
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)
        enable_gqa = H != k_sdpa.size(1)
        y = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
        return y.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.inner_dim = self.num_heads * self.head_dim

        inner_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, inner_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kv_cache: KVCache | None = None,
    ):
        B, T, C = x.size()
        q = self.q_norm(self.q_proj(x).view(B, T, self.num_heads, self.head_dim)).to(dtype=x.dtype)
        k = self.k_norm(self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)).to(
            dtype=x.dtype
        )
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, *cos_sin, unsqueeze_dim=2)
        if kv_cache is None:
            # Trainning
            if cu_seqlens is not None and q.is_cuda and flash_attn_interface is not None:
                out = flash_attn_interface.flash_attn_varlen_func(
                    q[0],
                    k[0],
                    v[0],
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=0.1,
                    causal=True,
                )
            elif cu_seqlens is not None:
                # Apply attention to each packed document separately; never leak across samples.
                segments = []
                boundaries = cu_seqlens.tolist()
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    if start == end:
                        continue
                    segments.append(
                        F.scaled_dot_product_attention(
                            q[:, start:end].transpose(1, 2),
                            k[:, start:end].transpose(1, 2),
                            v[:, start:end].transpose(1, 2),
                            is_causal=True,
                            enable_gqa=self.num_heads != self.num_kv_heads,
                            dropout_p=0.1 if self.training else 0.0,
                        ).transpose(1, 2)
                    )
                out = torch.cat(segments, dim=1)
            else:
                out = flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
            out = out.contiguous().view(B, T, -1)
            gate = self.gate_proj(x).sigmoid()
            return self.o_proj(out * gate)
        else:
            # Inference
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            if q.is_cuda and _fa2_flash_attn_with_kvcache is not None:
                y = _fa2_kvcache(q, k_cache, v_cache, k, v, kv_cache.cache_seqlens, True, -1, 0)
            else:
                y = flash_attn_with_kvcache(
                    q, k_cache, v_cache, k, v, kv_cache.cache_seqlens, True, (-1, 0)
                )
            y = y.contiguous().view(B, T, -1)
            if self.layer_idx == kv_cache.n_layers - 1 and not kv_cache.advance_disabled:
                kv_cache.advance(T)
            gate = self.gate_proj(x).sigmoid()
        y = self.o_proj(y * gate)
        return y


class Block(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        drop = config.dropout
        self.resid_drop = nn.Dropout(drop)
        self.mlp = SwiGLUFFN(config.hidden_size, config.intermediate_size, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kv_cache: KVCache | None = None,
    ):
        x = x + self.resid_drop(
            self.self_attn(
                self.input_layernorm(x).to(x.dtype),
                cos_sin,
                cu_seqlens,
                max_seqlen,
                kv_cache,
            )
        )
        x = x + self.resid_drop(self.mlp(self.post_attention_layernorm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rotary_seq_len = config.max_position_embeddings * 10
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.config = config
        self.layers = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_drop = nn.Dropout(config.dropout)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        std = getattr(self.config, "initializer_range", 0.02)
        scale = std / math.sqrt(2 * self.config.num_hidden_layers)

        for block in self.layers:
            nn.init.normal_(block.self_attn.q_proj.weight, std=std)
            nn.init.normal_(block.self_attn.k_proj.weight, std=std)
            nn.init.normal_(block.self_attn.v_proj.weight, std=std)
            nn.init.normal_(block.self_attn.o_proj.weight, std=scale)
            nn.init.normal_(block.self_attn.gate_proj.weight, std=std)
            nn.init.normal_(block.mlp.w12.weight, std=std)
            nn.init.normal_(block.mlp.w3.weight, std=scale)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kv_cache: KVCache | None = None,
    ):
        B, T, C = x.size()

        if position_ids is not None:
            cos, sin = self.rotary_emb(x, position_ids)
        else:
            T0 = kv_cache.get_pos()
            base_positions = torch.arange(T, device=x.device)
            position_ids = (base_positions + T0).unsqueeze(0)
            cos, sin = self.rotary_emb(x, position_ids)

        cos_sin = (cos, sin)
        for block in self.layers:
            x = block(x, cos_sin, cu_seqlens, max_seqlen, kv_cache)
        return self.norm(x).to(x.dtype)

    def forward_train(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        B, T, C = x.size()
        x = self.embed_drop(x)
        cos, sin = self.rotary_emb(x, position_ids)
        cos_sin = (cos, sin)
        for block in self.layers:
            x = block(x, cos_sin, cu_seqlens, max_seqlen, None)
        return self.norm(x).to(x.dtype)

    def forward_inference(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        kv_cache: KVCache | None = None,
    ):
        B, T, C = x.size()

        if position_ids is not None:
            cos, sin = self.rotary_emb(x, position_ids)
        else:
            T0 = kv_cache.get_pos()
            base_positions = torch.arange(T, device=x.device)
            position_ids = (base_positions + T0).unsqueeze(0)
            cos, sin = self.rotary_emb(x, position_ids)

        cos_sin = (cos, sin)
        for block in self.layers:
            x = block(x, cos_sin, cu_seqlens, max_seqlen, kv_cache)
        return self.norm(x).to(x.dtype)
