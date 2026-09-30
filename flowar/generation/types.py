"""Explicit contracts shared by the decoding engines and their callers."""

from dataclasses import dataclass

import numpy as np

MODE_TEXT = 0
MODE_AWAIT_EDGE = 1
MODE_BBOX_SURF = 2
MODE_BBOX_CURV = 3
MODE_DIFF_SURF = 4
MODE_DIFF_CURV = 5
MODE_DONE = 99


@dataclass
class StepResult:
    modes: list[int]
    text_ids: list[int]
    surf_vecs: np.ndarray
    curv_vecs: np.ndarray
    invalid_flags: list[bool]
    surf_trajs: np.ndarray | None = None  # [num_samples, T+1, surf_dim]
    curv_trajs: np.ndarray | None = None  # [num_samples, T+1, curv_dim]
    rollback_count: int = 0  # >0 means consumer should discard this many previously yielded tokens
    final_seq: list[int] | None = None  # authoritative seq at end
    final_surf_vecs: list[np.ndarray] | None = None
    final_curv_vecs: list[np.ndarray] | None = None
    geom_stats: dict | None = None  # rejection/backtrack counts (final yield only)


@dataclass(frozen=True)
class SamplingOptions:
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 78
    validate: bool = True
    constrain_tokens: bool = False
    steps: int = 20
    trajectory: bool = False

    def __post_init__(self):
        if self.max_new_tokens < 0 or self.steps < 1:
            raise ValueError("Token budget must be nonnegative and sampling steps positive")
        if self.temperature < 0 or (self.top_k is not None and self.top_k < 1):
            raise ValueError("Temperature must be nonnegative and top_k positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
