from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

from flowar.data.sequence import BBOX_THRESHOLD


@dataclass
class GeomValidationConfig:
    endpoint_tol: float = 0.01
    check_intersect: bool = False
    intersect_max_depth: int = 8
    bbox_min_span: float = BBOX_THRESHOLD
    budget_edge: int = 1
    budget_loop: int = 1
    budget_face: int = 20
    budget_total_face: int = 50


def bezier_endpoints(ctrl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the start and end 3-D endpoints of a rational cubic Bézier edge.

    Args:
        ctrl: ``(4, 4)`` array of control points ``(x, y, z, w)`` in world
              coordinates (already denormalized).

    Returns:
        ``(start, end)`` each of shape ``(3,)``.
    """
    return ctrl[0, :3].copy(), ctrl[3, :3].copy()


def endpoints_coincide(p1: np.ndarray, p2: np.ndarray, tol: float) -> bool:
    """Return True if two 3-D points are within *tol* Euclidean distance."""
    return float(np.linalg.norm(p1 - p2)) < tol


def bezier_aabb(ctrl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a Bézier edge's control points.

    Args:
        ctrl: ``(4, 4)`` array ``(x, y, z, w)``.

    Returns:
        ``(lo, hi)`` each of shape ``(3,)``.
    """
    pts = ctrl[:, :3]
    return pts.min(axis=0), pts.max(axis=0)


def _aabb_overlap(lo1: np.ndarray, hi1: np.ndarray, lo2: np.ndarray, hi2: np.ndarray) -> bool:
    return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1))


def _casteljau_split(ctrl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a rational cubic Bézier at t=0.5 using de Casteljau's algorithm.

    Control points are stored as ``(x, y, z, w)`` (Cartesian + weight).  The
    algorithm must be applied in homogeneous space ``(wx, wy, wz, w)`` and the
    result converted back, otherwise the subdivision does not converge to the
    actual rational curve.

    Args:
        ctrl: ``(4, 4)`` control points ``(x, y, z, w)``.

    Returns:
        ``(left, right)`` each ``(4, 4)`` in ``(x, y, z, w)`` form.
    """
    hom = ctrl.copy()
    hom[:, :3] *= hom[:, 3:4]  # (x,y,z,w) -> (wx,wy,wz,w)

    p0, p1, p2, p3 = hom[0], hom[1], hom[2], hom[3]
    p01 = 0.5 * (p0 + p1)
    p12 = 0.5 * (p1 + p2)
    p23 = 0.5 * (p2 + p3)
    p012 = 0.5 * (p01 + p12)
    p123 = 0.5 * (p12 + p23)
    p0123 = 0.5 * (p012 + p123)
    left_hom = np.stack([p0, p01, p012, p0123])
    right_hom = np.stack([p0123, p123, p23, p3])

    def _to_cartesian(h: np.ndarray) -> np.ndarray:
        out = h.copy()
        out[:, :3] /= out[:, 3:4]
        return out

    return _to_cartesian(left_hom), _to_cartesian(right_hom)


def bezier_intersect_fast(ctrl1: np.ndarray, ctrl2: np.ndarray, max_depth: int = 8) -> bool:
    """Fast intersection test for two rational cubic Bézier curves.

    Uses AABB overlap as a necessary condition and de Casteljau subdivision
    as refinement.  Only reports *existence* of intersection; no point is
    computed.

    Args:
        ctrl1: ``(4, 4)`` control points of the first curve.
        ctrl2: ``(4, 4)`` control points of the second curve.
        max_depth: Maximum subdivision depth.

    Returns:
        ``True`` if an intersection is detected, ``False`` otherwise.
    """
    lo1, hi1 = bezier_aabb(ctrl1)
    lo2, hi2 = bezier_aabb(ctrl2)
    if not _aabb_overlap(lo1, hi1, lo2, hi2):
        return False
    if max_depth == 0:
        return True
    l1, r1 = _casteljau_split(ctrl1)
    l2, r2 = _casteljau_split(ctrl2)
    return (
        bezier_intersect_fast(l1, l2, max_depth - 1)
        or bezier_intersect_fast(l1, r2, max_depth - 1)
        or bezier_intersect_fast(r1, l2, max_depth - 1)
        or bezier_intersect_fast(r1, r2, max_depth - 1)
    )


class GeomCheckResult(IntEnum):
    OK = 0
    ENDPOINT_MISMATCH = 1
    INTERSECTION_DETECTED = 2


class GeomChecker:
    def __init__(self, config: GeomValidationConfig):
        self._config = config
        self._loop_edges: List[np.ndarray] = []
        self._all_edges: List[np.ndarray] = []
        self._edge_by_token: dict = {}
        self._last_conflict_edge: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Loop lifecycle
    # ------------------------------------------------------------------

    def begin_loop(self) -> None:
        self._loop_edges.clear()

    def end_loop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_new_edge(self, ctrl: np.ndarray) -> GeomCheckResult:
        new_start, new_end = bezier_endpoints(ctrl)
        tol = self._config.endpoint_tol

        for existing in self._loop_edges:
            if np.allclose(ctrl, existing, atol=tol):
                self._last_conflict_edge = existing
                return GeomCheckResult.ENDPOINT_MISMATCH

        if self._loop_edges:
            prev = self._loop_edges[-1]
            prev_start, prev_end = bezier_endpoints(prev)
            shared = (
                endpoints_coincide(prev_end, new_start, tol)
                or endpoints_coincide(prev_end, new_end, tol)
                or endpoints_coincide(prev_start, new_start, tol)
                or endpoints_coincide(prev_start, new_end, tol)
            )
            if not shared:
                self._last_conflict_edge = prev
                return GeomCheckResult.ENDPOINT_MISMATCH

        if self._config.check_intersect:
            prev = self._loop_edges[-1] if self._loop_edges else None
            for existing in self._all_edges:
                if existing is prev:
                    continue
                if bezier_intersect_fast(ctrl, existing, self._config.intersect_max_depth):
                    return GeomCheckResult.INTERSECTION_DETECTED

        return GeomCheckResult.OK

    def validate_loop_closure(self) -> bool:
        """Check that the last edge shares an endpoint with the first edge."""
        if len(self._loop_edges) < 2:
            return True
        tol = self._config.endpoint_tol
        first_s, first_e = bezier_endpoints(self._loop_edges[0])
        last_s, last_e = bezier_endpoints(self._loop_edges[-1])
        return (
            endpoints_coincide(last_e, first_s, tol)
            or endpoints_coincide(last_e, first_e, tol)
            or endpoints_coincide(last_s, first_s, tol)
            or endpoints_coincide(last_s, first_e, tol)
        )

    def validate_ref_edge(self, ctrl: np.ndarray) -> GeomCheckResult:
        """Validate a referenced (EDGE_REF) edge for endpoint coincidence.

        EDGE_REF edges re-use already-committed geometry; they participate in
        the same endpoint-coincidence check as newly declared edges.
        """
        return self.validate_new_edge(ctrl)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def commit_edge(self, ctrl: np.ndarray) -> None:
        c = ctrl.copy()
        self._loop_edges.append(c)
        self._all_edges.append(c)

    def register_edge_token(self, token_id: int, ctrl: np.ndarray) -> None:
        """Associate a token ID with its control points for EDGE_REF lookup."""
        self._edge_by_token[token_id] = ctrl.copy()

    def get_edge_by_token(self, token_id: int) -> Optional[np.ndarray]:
        return self._edge_by_token.get(token_id)

    def release_edge_token(self, token_id: int) -> None:
        self._edge_by_token.pop(token_id, None)

    # ------------------------------------------------------------------
    # Snapshot / restore (for rejection sampling checkpoints)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "loop_edges": [e.copy() for e in self._loop_edges],
            "all_edges": [e.copy() for e in self._all_edges],
            "edge_by_token": {k: v.copy() for k, v in self._edge_by_token.items()},
        }

    def restore(self, snap: dict) -> None:
        self._loop_edges = [e.copy() for e in snap["loop_edges"]]
        self._all_edges = [e.copy() for e in snap["all_edges"]]
        self._edge_by_token = {k: v.copy() for k, v in snap["edge_by_token"].items()}
        self._last_conflict_edge = None
