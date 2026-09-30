"""One result collector for regular, prefix, conditional and backtracking generation."""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from flowar.data.sequence import BRepTokenType
from flowar.generation.constraints import TokenValidationStatus, validate_ids
from flowar.generation.prefix import SequencePrefix
from flowar.generation.types import MODE_DONE, StepResult


@dataclass
class GeneratedSample:
    ids: list[int] = field(default_factory=list)
    surf_vecs: list[np.ndarray] = field(default_factory=list)
    curv_vecs: list[np.ndarray] = field(default_factory=list)
    surf_trajs: list[np.ndarray] = field(default_factory=list)
    curv_trajs: list[np.ndarray] = field(default_factory=list)
    early_stopped: bool = False
    geometry_stats: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.early_stopped and validate_ids(self.ids)[0] == TokenValidationStatus.SUCCESS

    def geometry(self) -> dict:
        """Conversion boundary for CAD routines; no metadata hidden among geometry fields."""
        return {"ids": self.ids, "surf_vecs": self.surf_vecs, "curv_vecs": self.curv_vecs}

    def export_data(self) -> dict:
        return {**self.geometry(), "surf_trajs": self.surf_trajs, "curv_trajs": self.curv_trajs}

    def truncate(self, count: int):
        if count > len(self.ids):
            raise ValueError("Rollback exceeds the collected sequence")
        for token in self.ids[len(self.ids) - count :]:
            if token == BRepTokenType.SURFACE_GEOM:
                self.surf_vecs.pop()
                if self.surf_trajs:
                    self.surf_trajs.pop()
            elif token == BRepTokenType.CURVE_GEOM:
                self.curv_vecs.pop()
                if self.curv_trajs:
                    self.curv_trajs.pop()
        if count:
            del self.ids[-count:]


def collect_samples(steps: Iterable[StepResult], count: int, prefix: SequencePrefix | None = None):
    if count < 1:
        raise ValueError("Sample count must be positive")
    samples = [GeneratedSample() for _ in range(count)]
    if prefix is not None:
        for sample in samples:
            sample.ids = list(prefix.ids)
            sample.surf_vecs = [v.copy() for v in prefix.surfaces]
            sample.curv_vecs = [v.copy() for v in prefix.curves]
    for step in steps:
        if len(step.text_ids) != count:
            raise ValueError("Decoder batch size changed during generation")
        if step.final_seq is not None:
            if count != 1:
                raise ValueError("Authoritative backtracking results must contain one sample")
            sample = samples[0]
            sample.ids = list(step.final_seq)
            sample.surf_vecs = list(step.final_surf_vecs or [])
            sample.curv_vecs = list(step.final_curv_vecs or [])
            sample.early_stopped = step.invalid_flags[0]
            sample.geometry_stats = dict(step.geom_stats or {})
            continue
        if step.rollback_count:
            if count != 1:
                raise ValueError("Rollback is only supported for single-sample decoding")
            samples[0].truncate(step.rollback_count)
            continue
        for i, sample in enumerate(samples):
            sample.early_stopped |= step.invalid_flags[i]
            if step.modes[i] == MODE_DONE:
                continue
            token = step.text_ids[i]
            sample.ids.append(token)
            if token == BRepTokenType.SURFACE_GEOM:
                sample.surf_vecs.append(step.surf_vecs[i].copy())
                if step.surf_trajs is not None:
                    sample.surf_trajs.append(step.surf_trajs[i].copy())
            elif token == BRepTokenType.CURVE_GEOM:
                sample.curv_vecs.append(step.curv_vecs[i].copy())
                if step.curv_trajs is not None:
                    sample.curv_trajs.append(step.curv_trajs[i].copy())
    return samples
