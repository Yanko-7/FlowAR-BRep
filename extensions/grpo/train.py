import argparse
import copy
import math
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from flowar.checkpoint import load_model_from_checkpoint
from flowar.data.sequence import BRepTokenType
from flowar.generation.constraints import TokenValidationStatus, validate_ids
from flowar.generation.engine import Engine
from flowar.generation.prefix import SequencePrefix
from flowar.generation.results import collect_samples
from flowar.models.flow_head import (
    sde_step_forward_with_logprob,
    time_shift_func,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO refinement of FlowAR-BRep")
    parser.add_argument("--run", type=str, default="wandb", help="wandb run name")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Micro-batch size per forward pass",
    )
    parser.add_argument(
        "--rollout-size",
        type=int,
        default=512,
        help="Number of samples generated per rollout",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--device_type", type=str, default="")
    parser.add_argument("--model_tag", type=str, default="")
    parser.add_argument("--gen-max-batch-size", type=int, default=256)
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=10,
        help="Diffusion sampling steps per geometry token",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/model.pt",
        help="Path to pretrained checkpoint to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--reward-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for BRep build in reward_fn",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Linear LR warmup steps at training start",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=31,
        help="Minimum surface count for positive reward; builds below this threshold are treated as failures (use 0 to disable)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.05,
        help="KL penalty coefficient against frozen reference model (0 to disable)",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=5.0,
        help="Clip normalized advantages to [-clip, +clip] (0 to disable)",
    )
    return parser.parse_args()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def _all_reduce_grads(model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return
    present = torch.tensor(
        [p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    for parameter, used in zip(parameters, present.tolist()):
        if used:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())


@dataclass
class Episode:
    seq_len: int  # total generated sequence length (no padding)
    text_ids: np.ndarray  # [N_text]
    text_indexes: np.ndarray  # [N_text]  sequence positions of text tokens
    surface_vecs_trajectory: (
        np.ndarray
    )  # [T+1, N_surf, surf_dim]  diffusion trajectory; [-1] = final
    surface_geom_indexes: np.ndarray  # [N_surf]  sequence positions of surface tokens
    curve_vecs_trajectory: np.ndarray  # [T+1, N_curv, curv_dim]  diffusion trajectory; [-1] = final
    curve_geom_indexes: np.ndarray  # [N_curv]  sequence positions of curve tokens
    reward: float
    reward_info: dict[str, float] = field(default_factory=dict)
    invalid: bool = False


def _reward_worker(ids, surf_vecs, curv_vecs, ordering: str = "interleaved") -> bool:
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    from flowar.geometry.builder import BRepShapeBuilder, convert_generated_to_brep_data

    brepdata = convert_generated_to_brep_data(
        ids=ids, surf_vecs=surf_vecs, curv_vecs=curv_vecs, ordering=ordering
    )
    builder = BRepShapeBuilder(brepdata)
    shape = builder.build()
    return shape is not None


def reward_fn(
    ep: Episode,
    pool,
    timeout: float = 30.0,
    min_faces: int = 0,
    ordering: str = "interleaved",
):
    """Returns (reward, info_dict).

    Reward scale:
      -1.0  : token-sequence validation failure
      -0.5  : valid sequence but BRep build failed / timed out
      faces / 20.0 : successful build meeting the configured minimum face count
    """
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    seq = np.zeros(ep.seq_len, dtype=np.int32)
    seq[ep.text_indexes] = ep.text_ids
    seq[ep.surface_geom_indexes] = BRepTokenType.SURFACE_GEOM
    seq[ep.curve_geom_indexes] = BRepTokenType.CURVE_GEOM
    ids = seq.tolist()
    status, _reason, _err_idx = validate_ids(ids, ordering=ordering)
    if ep.invalid or status != TokenValidationStatus.SUCCESS:
        return -1.0, {"valid": 0.0, "built": 0.0}

    if len(ep.surface_geom_indexes) < min_faces:
        return -0.5, {"valid": 1.0, "built": 0.0}

    surf_vecs = ep.surface_vecs_trajectory[-1]
    curv_vecs = ep.curve_vecs_trajectory[-1]

    future = pool.schedule(
        _reward_worker, args=(ids, surf_vecs, curv_vecs, ordering), timeout=timeout
    )
    try:
        ok = future.result()
    except Exception:
        return -0.5, {"valid": 1.0, "built": 0.0}
    if not ok:
        return -0.5, {"valid": 1.0, "built": 0.0}
    n_surf = len(ep.surface_geom_indexes)
    reward = n_surf / 20.0
    return reward, {"valid": 1.0, "built": 1.0}


def build_optimizer(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def rollout(model, rollout_size, reward_fn, args, device, rank) -> list[Episode]:
    """Record the same prefix and mixed sequence used by the sampling policy."""
    if rollout_size < 1 or args.gen_max_batch_size < 1:
        raise ValueError("Rollout and generation batch sizes must be positive")
    model.eval()
    engine = Engine(model)
    prefix = SequencePrefix.from_tokens([BRepTokenType.BOS, BRepTokenType.COMPLEXITY_L3])
    results = []
    for batch_index, offset in enumerate(range(0, rollout_size, args.gen_max_batch_size)):
        count = min(args.gen_max_batch_size, rollout_size - offset)
        steps = engine.generate(
            prefix.ids,
            num_samples=count,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=hash((batch_index, rank)) & 0x7FFFFFFF,
            use_validation=True,
            return_with_trajectory=True,
            num_sampling_steps=args.num_sampling_steps,
        )
        for sample in collect_samples(steps, count, prefix):
            ids = np.asarray(sample.ids, dtype=np.int64)
            surface = ids == BRepTokenType.SURFACE_GEOM
            curve = ids == BRepTokenType.CURVE_GEOM
            text = ~(surface | curve)

            def trajectories(values, width):
                if values:
                    return np.stack(values).transpose(1, 0, 2)
                return np.empty((args.num_sampling_steps + 2, 0, width), dtype=np.float32)

            episode = Episode(
                seq_len=len(ids),
                text_ids=ids[text],
                text_indexes=np.flatnonzero(text),
                surface_vecs_trajectory=trajectories(
                    sample.surf_trajs, model.config.surface_latent_dim
                ),
                surface_geom_indexes=np.flatnonzero(surface),
                curve_vecs_trajectory=trajectories(
                    sample.curv_trajs, model.config.curve_latent_dim
                ),
                curve_geom_indexes=np.flatnonzero(curve),
                reward=0.0,
                invalid=sample.early_stopped,
            )
            episode.reward, episode.reward_info = reward_fn(episode)
            results.append(episode)
    return results


@dataclass
class EpisodeBatch:
    text_ids: torch.Tensor  # (B, max_text_len)
    label_ids: torch.Tensor  # (B, max_ce_len)
    text_mask: torch.BoolTensor  # (B, max_text_len) - True for valid data, False for padding

    surface_vecs_trajectory: torch.Tensor  # (B, max_surf_len, T+1, vector_dim)
    surface_mask: torch.BoolTensor  # (B, max_surf_len)

    curve_vecs_trajectory: torch.Tensor  # (B, max_curve_len, T+1, vector_dim)
    curve_mask: torch.BoolTensor  # (B, max_curve_len)

    seq_len: torch.Tensor  # (B,)
    text_indexes: torch.Tensor  # (B, max_text_len)
    surface_geom_indexes: torch.Tensor  # (B, max_surf_len)
    curve_geom_indexes: torch.Tensor  # (B, max_curve_len)
    ce_loss_indexes: torch.Tensor  # (B, max_ce_len)
    ce_loss_mask: torch.BoolTensor  # (B, max_ce_len)
    reward: torch.Tensor  # (B,)
    reward_info: dict[str, torch.Tensor]


def collate_episodes_with_padding(episodes: list[Episode]):
    def to_ts(arr, dtype=torch.long):
        return (
            torch.tensor([], dtype=dtype)
            if arr is None or len(arr) == 0
            else torch.from_numpy(np.asarray(arr)).to(dtype)
        )

    text_ids, text_indexes = [], []
    surf_trajs, surf_geom = [], []
    curve_trajs, curve_geom = [], []
    ce_loss_idx, label_ids = [], []

    for ep in episodes:
        text_ids.append(to_ts(ep.text_ids))
        text_indexes.append(to_ts(ep.text_indexes))

        # surface_vecs_trajectory shape: [T+1, N_surf, D]
        # Transpose to [N_surf, T+1, D] so pad_sequence pads over N_surf -> [B, max_surf, T+1, D]
        sv = ep.surface_vecs_trajectory  # [T+1, N_surf, D]
        if sv.shape[1] > 0:
            surf_trajs.append(torch.from_numpy(sv.transpose(1, 0, 2)).float())  # [N_surf, T+1, D]
        else:
            surf_trajs.append(torch.empty((0, sv.shape[0], sv.shape[2]), dtype=torch.float32))
        surf_geom.append(to_ts(ep.surface_geom_indexes))

        cv = ep.curve_vecs_trajectory  # [T+1, N_curv, D]
        if cv.shape[1] > 0:
            curve_trajs.append(torch.from_numpy(cv.transpose(1, 0, 2)).float())  # [N_curv, T+1, D]
        else:
            curve_trajs.append(torch.empty((0, cv.shape[0], cv.shape[2]), dtype=torch.float32))
        curve_geom.append(to_ts(ep.curve_geom_indexes))

        NUM_PREFIX = (
            2  # BOS + COMPLEXITY_L3 are fixed prompt tokens; skip their positions from GRPO loss
        )
        if len(ep.text_indexes) > NUM_PREFIX:
            pred_idx = ep.text_indexes[NUM_PREFIX:] - 1
            is_valid_src = np.zeros(ep.seq_len + 1, dtype=bool)
            is_valid_src[ep.text_indexes] = True
            if len(ep.surface_geom_indexes) > 0:
                is_valid_src[ep.surface_geom_indexes] = True
            if len(ep.curve_geom_indexes) > 0:
                is_valid_src[ep.curve_geom_indexes] = True

            mask = is_valid_src[pred_idx]
            ce_loss_idx.append(to_ts(pred_idx[mask]))
            label_ids.append(to_ts(ep.text_ids[NUM_PREFIX:][mask]))
        else:
            ce_loss_idx.append(torch.tensor([], dtype=torch.long))
            label_ids.append(torch.tensor([], dtype=torch.long))

    def pad_and_mask(tensor_list, pad_val=0):
        if all(t.numel() == 0 for t in tensor_list):
            return None, None
        padded = pad_sequence(tensor_list, batch_first=True, padding_value=pad_val)
        lens = torch.tensor([len(t) for t in tensor_list], dtype=torch.long)
        mask = torch.arange(padded.size(1)).unsqueeze(0) < lens.unsqueeze(1)
        return padded, mask

    p_text_ids, text_mask = pad_and_mask(text_ids)
    p_text_idx, _ = pad_and_mask(text_indexes)
    p_surf_trajs, surf_mask = pad_and_mask(surf_trajs, pad_val=0.0)  # [B, max_surf, T+1, D]
    p_surf_geom, _ = pad_and_mask(surf_geom)
    p_curve_trajs, curve_mask = pad_and_mask(curve_trajs, pad_val=0.0)  # [B, max_curv, T+1, D]
    p_curve_geom, _ = pad_and_mask(curve_geom)

    p_ce_loss_idx, ce_loss_mask = pad_and_mask(ce_loss_idx)
    p_label_ids, _ = pad_and_mask(label_ids, pad_val=-100)

    return EpisodeBatch(
        seq_len=torch.tensor([ep.seq_len for ep in episodes], dtype=torch.long),
        text_ids=p_text_ids,
        text_indexes=p_text_idx,
        text_mask=text_mask,
        surface_vecs_trajectory=p_surf_trajs,
        surface_geom_indexes=p_surf_geom,
        surface_mask=surf_mask,
        curve_vecs_trajectory=p_curve_trajs,
        curve_geom_indexes=p_curve_geom,
        curve_mask=curve_mask,
        ce_loss_indexes=p_ce_loss_idx,
        ce_loss_mask=ce_loss_mask,
        label_ids=p_label_ids,
        reward=torch.tensor([ep.reward for ep in episodes], dtype=torch.float32),
        reward_info={},
    )


def get_last_hidden_states(
    model,
    text_ids: torch.Tensor,  # (B, max_text)
    text_indexes: torch.Tensor,  # (B, max_text)
    text_mask: torch.BoolTensor,  # (B, max_text)
    seq_len: torch.Tensor,  # (B,)
    surface_vectors: Optional[torch.Tensor] = None,  # (B, max_surf, ...)
    surface_geom_indexes: Optional[torch.Tensor] = None,  # (B, max_surf)
    surface_mask: Optional[torch.BoolTensor] = None,  # (B, max_surf)
    curve_vectors: Optional[torch.Tensor] = None,  # (B, max_curve, ...)
    curve_geom_indexes: Optional[torch.Tensor] = None,  # (B, max_curve)
    curve_mask: Optional[torch.BoolTensor] = None,  # (B, max_curve)
    ce_loss_indexes: Optional[torch.Tensor] = None,  # (B, max_ce)
    ce_loss_mask: Optional[torch.BoolTensor] = None,  # (B, max_ce)
):
    B = text_ids.size(0)
    T = seq_len.max().item()
    device = text_ids.device
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else torch.float32

    sequence = torch.zeros((B, T, model.hidden_size), device=device, dtype=dtype)

    def _scatter(embeds, indexes, mask):
        if embeds is not None and mask is not None and mask.any():
            b_idx, i_idx = torch.where(mask)
            sequence[b_idx, indexes[b_idx, i_idx]] = embeds[b_idx, i_idx].to(dtype)

    _scatter(model.text_embed(text_ids), text_indexes, text_mask)
    if surface_vectors is not None:
        _scatter(
            model.surface_embed(surface_vectors.flatten(2)),
            surface_geom_indexes,
            surface_mask,
        )
    if curve_vectors is not None:
        _scatter(model.curve_embed(curve_vectors.flatten(2)), curve_geom_indexes, curve_mask)

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    hidden_states = model.transformer(sequence, position_ids=position_ids)

    def _gather(indexes, mask):
        if indexes is not None and mask is not None and mask.any():
            b_idx, i_idx = torch.where(mask)
            return hidden_states[b_idx, indexes[b_idx, i_idx]], b_idx, i_idx
        return None, None, None

    h_ce, b_ce, i_ce = _gather(ce_loss_indexes, ce_loss_mask)
    text_logits = model.text_head(h_ce) if h_ce is not None else None

    cond_h_surf, b_surf, _ = _gather(
        torch.clamp(surface_geom_indexes - 1, min=0) if surface_geom_indexes is not None else None,
        surface_mask,
    )

    cond_h_curve, b_curve, _ = _gather(
        torch.clamp(curve_geom_indexes - 1, min=0) if curve_geom_indexes is not None else None,
        curve_mask,
    )
    return (text_logits, b_ce, i_ce), (cond_h_surf, b_surf), (cond_h_curve, b_curve)


def compute_log_prob(
    diff_head,
    trajectory: torch.Tensor,  # [N, T+1, D]  T+1 = num_sampling_steps + 2
    cond: torch.Tensor,  # [N, H]
    num_sampling_steps: int = 20,
    last_step_size: float = 0.04,
    sde_type: str = "cps",
) -> torch.Tensor:  # [N] — accumulated log_prob across all SDE steps
    """Recompute SDE log_prob for recorded trajectories under the current policy.

    Replays the exact time schedule of euler_maruyama_with_logprob so that
    log p(x_{t+1} | x_t) is evaluated at the *stored* next states rather than
    fresh samples, which gives the policy-gradient signal without re-sampling.
    """
    device = trajectory.device
    N = trajectory.shape[0]

    t_all = torch.linspace(0, 1.0 - last_step_size, num_sampling_steps + 1, device=device)
    if hasattr(diff_head, "time_shift") and diff_head.time_shift != 1.0:
        t_all = time_shift_func(t_all, diff_head.time_shift)
    dt = t_all[1:] - t_all[:-1]  # [num_sampling_steps]

    t = torch.tensor(0.0, device=device)
    t_batch = torch.zeros(N, device=device)
    total_log_prob = torch.zeros(N, device=device, dtype=torch.float32)

    for i in range(num_sampling_steps):
        t_batch.fill_(t.item())
        x_t = trajectory[:, i, :]  # [N, D]
        x_next = trajectory[:, i + 1, :]  # [N, D]

        output = diff_head.net(x_t, t_batch, cond)  # [N, D]
        denom = (1.0 - t_batch).view(-1, 1).clamp_min(1e-5)
        v = (output - x_t) / denom

        _, log_prob, _, _ = sde_step_forward_with_logprob(
            v, t, t + dt[i], dt[i], x_t, x_next_target=x_next, sde_type=sde_type
        )
        total_log_prob = total_log_prob + log_prob.float()
        t = t + dt[i]

    # Last step: t -> 1.0
    t_last = torch.tensor(1.0 - last_step_size, device=device)
    t_batch.fill_(t_last.item())
    x_t = trajectory[:, num_sampling_steps, :]
    x_next = trajectory[:, num_sampling_steps + 1, :]

    output = diff_head.net(x_t, t_batch, cond)
    denom = (1.0 - t_batch).view(-1, 1).clamp_min(1e-5)
    v = (output - x_t) / denom

    _, log_prob, _, _ = sde_step_forward_with_logprob(
        v,
        t_last,
        t_last + last_step_size,
        last_step_size,
        x_t,
        x_next_target=x_next,
        sde_type=sde_type,
    )
    total_log_prob = total_log_prob + log_prob.float()

    return total_log_prob


def _compute_pg_loss(
    model,
    episodes: list[Episode],
    advantages: torch.Tensor,
    device,
    sde_type: str = "cps",
    num_sampling_steps: int = 20,
    ref_model=None,
    kl_coeff: float = 0.0,
) -> Optional[torch.Tensor]:
    batch = collate_episodes_with_padding(episodes)

    def _to(t):
        return t.to(device) if t is not None else None

    with torch.autocast(
        torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
    ):
        surf_vecs_final = (
            batch.surface_vecs_trajectory[:, :, -1, :].to(device)
            if batch.surface_vecs_trajectory is not None
            else None
        )
        curv_vecs_final = (
            batch.curve_vecs_trajectory[:, :, -1, :].to(device)
            if batch.curve_vecs_trajectory is not None
            else None
        )

        (text_logits, b_ce, i_ce), (cond_h_surf, b_surf), (cond_h_curve, b_curve) = (
            get_last_hidden_states(
                model,
                text_ids=_to(batch.text_ids),
                text_indexes=_to(batch.text_indexes),
                text_mask=_to(batch.text_mask),
                seq_len=_to(batch.seq_len),
                surface_vectors=surf_vecs_final,
                surface_geom_indexes=_to(batch.surface_geom_indexes),
                surface_mask=_to(batch.surface_mask),
                curve_vectors=curv_vecs_final,
                curve_geom_indexes=_to(batch.curve_geom_indexes),
                curve_mask=_to(batch.curve_mask),
                ce_loss_indexes=_to(batch.ce_loss_indexes),
                ce_loss_mask=_to(batch.ce_loss_mask),
            )
        )

        loss = None

        if text_logits is not None:
            labels_ce = _to(batch.label_ids)[b_ce, i_ce]
            ce = F.cross_entropy(text_logits.float(), labels_ce, reduction="none")
            term = (ce * advantages[b_ce].detach()).mean()
            loss = term if loss is None else loss + term

            if ref_model is not None and kl_coeff > 0.0:
                with torch.no_grad():
                    (ref_logits, b_ce_ref, i_ce_ref), _, _ = get_last_hidden_states(
                        ref_model,
                        text_ids=_to(batch.text_ids),
                        text_indexes=_to(batch.text_indexes),
                        text_mask=_to(batch.text_mask),
                        seq_len=_to(batch.seq_len),
                        surface_vectors=surf_vecs_final,
                        surface_geom_indexes=_to(batch.surface_geom_indexes),
                        surface_mask=_to(batch.surface_mask),
                        curve_vectors=curv_vecs_final,
                        curve_geom_indexes=_to(batch.curve_geom_indexes),
                        curve_mask=_to(batch.curve_mask),
                        ce_loss_indexes=_to(batch.ce_loss_indexes),
                        ce_loss_mask=_to(batch.ce_loss_mask),
                    )
                if ref_logits is not None:
                    ce_ref = F.cross_entropy(ref_logits.float(), labels_ce, reduction="none")
                    kl = (-ce + ce_ref.detach()).mean()  # log π_current - log π_ref
                    loss = loss + kl_coeff * kl

    return loss


def update_policy(
    model,
    episodes: list[Episode],
    optimizer,
    micro_batch_size: int,
    device,
    max_grad_norm: float = 1.0,
    num_sampling_steps: int = 20,
    ref_model=None,
    kl_coeff: float = 0.0,
    advantage_clip: float = 0.0,
) -> tuple[float, float]:
    if not episodes or micro_batch_size < 1:
        raise ValueError("Policy updates require episodes and a positive micro-batch size")
    model.train()

    rewards = torch.tensor([ep.reward for ep in episodes], dtype=torch.float32, device=device)
    if dist.is_initialized():
        n = torch.tensor(float(len(episodes)), device=device)
        s, ss = rewards.sum(), (rewards**2).sum()
        dist.all_reduce(n)
        dist.all_reduce(s)
        dist.all_reduce(ss)
        mean = s / n
        std = ((ss / n - mean**2).clamp_min(0) + 1e-8).sqrt()
    else:
        mean, std = rewards.mean(), rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean) / std
    if advantage_clip > 0.0:
        advantages = advantages.clamp(-advantage_clip, advantage_clip)

    sorted_pairs = sorted(enumerate(episodes), key=lambda x: x[1].seq_len)
    sorted_idxs = [p[0] for p in sorted_pairs]
    sorted_eps = [p[1] for p in sorted_pairs]
    adv_sorted = advantages[sorted_idxs]

    optimizer.zero_grad()
    target_count = sum(max(len(ep.text_ids) - 2, 0) for ep in sorted_eps)
    total_loss = 0.0
    for i in range(0, len(sorted_eps), micro_batch_size):
        batch_eps = sorted_eps[i : i + micro_batch_size]
        batch_adv = adv_sorted[i : i + micro_batch_size]
        loss = _compute_pg_loss(
            model,
            batch_eps,
            batch_adv,
            device,
            num_sampling_steps=num_sampling_steps,
            ref_model=ref_model,
            kl_coeff=kl_coeff,
        )
        if loss is not None:
            targets = sum(max(len(ep.text_ids) - 2, 0) for ep in batch_eps)
            weight = targets / max(target_count, 1)
            (loss * weight).backward()
            total_loss += loss.item() * weight

    if dist.is_initialized():
        _all_reduce_grads(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return total_loss, grad_norm.item()


def main():
    torch.set_float32_matmul_precision("high")
    args = parse_args()

    ddp = int(os.environ.get("LOCAL_RANK", -1)) != -1
    if ddp:
        setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        args.device_type
        if args.device_type
        else (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    )

    if not args.checkpoint:
        print("Error: --checkpoint is required.", file=sys.stderr)
        sys.exit(1)

    if is_main():
        print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_model_from_checkpoint(args.checkpoint, use_ema=True)
    model = model.to(device)

    if args.kl_coeff > 0.0:
        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        if is_main():
            print(f"Reference model frozen for KL penalty (coeff={args.kl_coeff})")
    else:
        ref_model = None

    total_params = sum(p.numel() for p in model.parameters())
    if is_main():
        print(f"Model params: {total_params:,} | device: {device}")

    optimizer = build_optimizer(model, args)
    _warmup = args.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, _warmup)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = WANDB_AVAILABLE and args.run and is_main()
    if use_wandb:
        try:
            wandb.init(project="grpo-brep", name=args.run, config=vars(args))
        except Exception as e:
            print(f"wandb init failed: {e}")
            use_wandb = False

    from pebble import ProcessPool

    pool_ctx = multiprocessing.get_context("spawn")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_rank_rollout = math.ceil(args.rollout_size / world_size)

    global_step = 0
    best_reward = float("-inf")
    with ProcessPool(max_workers=4, context=pool_ctx) as pool:
        for step in range(args.total_steps):
            global_step = step + 1

            episodes = rollout(
                model,
                per_rank_rollout,
                lambda ep: reward_fn(
                    ep, pool=pool, timeout=args.reward_timeout, min_faces=args.min_faces
                ),
                args,
                device,
                rank=rank,
            )
            rewards = np.array([ep.reward for ep in episodes], dtype=np.float32)
            mean_reward = float(rewards.mean())
            max_reward = float(rewards.max())
            min_reward = float(rewards.min())
            success_rate = float((rewards > 0).mean())

            avg_loss, grad_norm = update_policy(
                model,
                episodes,
                optimizer,
                args.train_batch_size,
                device,
                num_sampling_steps=args.num_sampling_steps,
                ref_model=ref_model,
                kl_coeff=args.kl_coeff,
                advantage_clip=args.advantage_clip,
            )
            scheduler.step()
            avg_seq_len = float(np.mean([ep.seq_len for ep in episodes]))
            valid_rate = float(np.mean([ep.reward_info.get("valid", 0.0) for ep in episodes]))
            build_rate = float(np.mean([ep.reward_info.get("built", 0.0) for ep in episodes]))
            if is_main():
                print(
                    f"[{global_step}/{args.total_steps}] loss={avg_loss:.4f}  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"reward mean={mean_reward:.2f} max={max_reward:.2f} min={min_reward:.2f}  "
                    f"success={success_rate:.2%}  valid={valid_rate:.2%}  build={build_rate:.2%}  "
                    f"avg_len={avg_seq_len:.1f}"
                )
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm,
                        "train/reward_mean": mean_reward,
                        "train/reward_max": max_reward,
                        "train/reward_min": min_reward,
                        "train/success_rate": success_rate,
                        "train/valid_rate": valid_rate,
                        "train/build_rate": build_rate,
                        "train/avg_seq_len": avg_seq_len,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            if is_main() and mean_reward > best_reward:
                best_reward = mean_reward
                save_path = output_dir / "grpo_best.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"New best reward {best_reward:.4f} → Saved: {save_path}")

            if is_main() and global_step % args.save_every == 0:
                save_path = output_dir / f"grpo_step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"Saved: {save_path}")

    if is_main():
        save_path = output_dir / "grpo_final.pt"
        torch.save(
            {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            },
            save_path,
        )
        print(f"Training complete. Saved: {save_path}")
        if use_wandb:
            wandb.finish()
    if ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
