"""Serialize interleaved topology tokens and continuous B-Rep geometry.

Geometry occupies one sequence position; edge IDs are recycled after reference.
See docs/sequence_format.md for the vocabulary and sequence layout.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from flowar.geometry.bezier import (
    eval_rational_bezier_curves,
    eval_rational_bezier_surfaces,
)

# Spatial resolution configuration

SPATIAL_RESOLUTION = 1024
EDGE_ID_COUNT = 980

# UV-grid geometry representation: resolution at which normalized rational Bézier
# faces/curves are evaluated before VAE compression (matches the pretrained
# BrepGen surface/edge VAEs).
GEOM_GRID_RES = 32

# Token vocabulary definition


class BRepTokenType(IntEnum):
    """BRep Token classification

    Token layout:
    - 0-31: special control tokens
    """

    # === special control tokens (0-31) ===
    PAD = 0
    BOS = 1
    EOS = 2

    # Face structure
    FACE_START = 3
    FACE_END = 4

    # BBox structure
    BBOX_START = 5
    BBOX_END = 6

    # Geometry placeholders
    SURFACE_GEOM = 7
    CURVE_GEOM = 8

    # Loop structure
    LOOP_START = 9
    LOOP_END = 10

    # Edge types
    EDGE_NEW = 11
    EDGE_REF = 12

    SEP = 13

    # Condition tokens
    ## Point cloud condition
    PC_COND = 14
    ## Image condition
    IMG_COND = 15

    # Complexity control tokens
    ## Level 1: 1-15 faces
    COMPLEXITY_L1 = 17
    ## Level 2: 16-30 faces
    COMPLEXITY_L2 = 18
    ## Level 3: 31+ faces
    COMPLEXITY_L3 = 19

    # IDs 20–31 remain reserved to preserve existing token offsets.


# === Complexity level configuration ===
COMPLEXITY_LEVEL_SIZE = 15  # faces per level
COMPLEXITY_NUM_LEVELS = 3  # total levels

# === Token range definitions ===
COORD_TOKEN_OFFSET = 32
COORD_TOKEN_MIN = COORD_TOKEN_OFFSET
COORD_TOKEN_MAX = COORD_TOKEN_OFFSET + SPATIAL_RESOLUTION - 1
BBOX_TOKEN_COUNT = 6  # 6 tokens for bbox (x_min, y_min, z_min, x_max, y_max, z_max)
EDGE_ID_OFFSET = COORD_TOKEN_MAX + 1
EDGE_ID_MIN = EDGE_ID_OFFSET
EDGE_ID_MAX = EDGE_ID_OFFSET + EDGE_ID_COUNT - 1

BREP_VOCAB_SIZE = EDGE_ID_MAX + 1


# coordinate encoding/decoding tools


def coord_to_token(value: float, resolution: int = SPATIAL_RESOLUTION) -> int:
    value = np.clip(value, -1.0, 1.0)
    value_normalized = (value + 1.0) / 2.0
    quantized = int(value_normalized * (resolution - 1) + 0.5)
    quantized = min(max(quantized, 0), resolution - 1)
    return quantized + COORD_TOKEN_OFFSET


def token_to_coord(token_id: int, resolution: int = SPATIAL_RESOLUTION) -> float:
    quantized = token_id - COORD_TOKEN_OFFSET
    value_normalized = quantized / (resolution - 1)
    return value_normalized * 2.0 - 1.0


def is_coord_token(token_id: int) -> bool:
    return COORD_TOKEN_MIN <= token_id <= COORD_TOKEN_MAX


def edge_id_to_token(edge_id: int) -> int:
    if edge_id < 0 or edge_id >= EDGE_ID_COUNT:
        raise ValueError(f"Edge ID {edge_id} out of range [0, {EDGE_ID_COUNT})")
    return edge_id + EDGE_ID_OFFSET


def token_to_edge_id(token_id: int) -> int:
    return token_id - EDGE_ID_OFFSET


def is_edge_id_token(token_id: int) -> bool:
    return EDGE_ID_MIN <= token_id <= EDGE_ID_MAX


def is_complexity_token(token_id: int) -> bool:
    return BRepTokenType.COMPLEXITY_L1 <= token_id <= BRepTokenType.COMPLEXITY_L3


def _parse_bboxes(ids: list, ordering: str = "interleaved") -> tuple[list, list]:
    """Walk the token sequence and collect face and edge bboxes in order."""
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    face_bboxes: list = []
    edge_bboxes: list = []
    idx, n = 0, len(ids)

    while idx < n:
        tok = ids[idx]
        if tok in (BRepTokenType.BOS, BRepTokenType.PAD) or is_complexity_token(tok):
            idx += 1
            continue
        if tok == BRepTokenType.EOS:
            break
        if tok == BRepTokenType.FACE_START:
            if idx + 7 <= n:
                face_bboxes.append(tokens_to_bbox(ids[idx + 1 : idx + 7]))
            idx += 8
        elif tok == BRepTokenType.FACE_END:
            idx += 1
        elif tok == BRepTokenType.LOOP_START:
            idx += 1
            while idx < n and ids[idx] != BRepTokenType.LOOP_END:
                ltok = ids[idx]
                if ltok == BRepTokenType.EDGE_NEW:
                    if idx + 8 <= n:
                        edge_bboxes.append(tokens_to_bbox(ids[idx + 2 : idx + 8]))
                    idx += 9
                elif ltok == BRepTokenType.EDGE_REF:
                    idx += 2
                else:
                    idx += 1
            idx += 1
        else:
            idx += 1
    return face_bboxes, edge_bboxes


def num_faces_to_complexity_token(num_faces: int) -> int:
    """Map a face count to the appropriate complexity level token."""
    level = min((num_faces - 1) // COMPLEXITY_LEVEL_SIZE, COMPLEXITY_NUM_LEVELS - 1)
    return int(BRepTokenType.COMPLEXITY_L1) + level


def bbox_to_tokens(bbox: np.ndarray, resolution: int = SPATIAL_RESOLUTION) -> list[int]:
    # Reorder [x1, y1, z1, x2, y2, z2] -> [z1, z2, y1, y2, x1, x2]
    v = np.asarray(bbox).ravel()[[2, 5, 1, 4, 0, 3]]

    # Vectorized implementation of coord_to_token logic
    # 1. Clip -> Normalize -> Scale -> Round (+0.5 then cast)
    v = (np.clip(v, -1.0, 1.0) + 1.0) * 0.5 * (resolution - 1) + 0.5

    # 2. Clip result -> Offset
    return (np.clip(v, 0, resolution - 1).astype(int) + COORD_TOKEN_OFFSET).tolist()


def tokens_to_bbox(tokens: list[int], resolution: int = SPATIAL_RESOLUTION) -> np.ndarray:
    if len(tokens) != 6:
        raise ValueError(f"Expected 6 tokens, got {len(tokens)}")

    # Decode tokens to values
    # Order is Z -> Y -> X: [z_min, z_max, y_min, y_max, x_min, x_max]
    vals = [token_to_coord(t, resolution) for t in tokens]
    z_min, z_max, y_min, y_max, x_min, x_max = vals

    # Return standard [x_min, y_min, z_min, x_max, y_max, z_max]
    return np.array(
        [x_min, y_min, z_min, x_max, y_max, z_max],
        dtype=np.float32,
    )


# sequence data structure


@dataclass
class BRepSequence:
    text_ids: list[int] = field(default_factory=list)
    text_indexes: list[int] = field(default_factory=list)

    surface_geom_vectors: list[np.ndarray] = field(default_factory=list)
    surface_geom_indexes: list[int] = field(default_factory=list)

    curve_geom_vectors: list[np.ndarray] = field(default_factory=list)
    curve_geom_indexes: list[int] = field(default_factory=list)

    position_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    total_length: int = 0

    condition_indexes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    condition_pc: "np.ndarray | None" = field(default=None)
    condition_pixel_values: "np.ndarray | None" = field(default=None)
    condition_text_ids: "np.ndarray | None" = field(default=None)  # [77] int32
    condition_text_mask: "np.ndarray | None" = field(default=None)  # [77] int32


# edge reference registry


class EdgeRegistry:
    def __init__(
        self,
        id_range: int = EDGE_ID_COUNT,
        reuse_ids: bool = True,
        random_start: bool = True,
    ):
        if not reuse_ids:
            raise ValueError("Edge-ID recycling is required")
        self._id_range = id_range
        self._edge_to_token: dict[int, int] = {}
        self._available: deque = deque()
        self._start: int = random.randint(0, id_range - 1) if random_start else 0
        self._allocated: int = 0

    def register(self, edge_id: int) -> tuple[bool, int]:
        if edge_id in self._edge_to_token:
            token_id = self._edge_to_token.pop(edge_id)
            self._available.append(token_id)
            return False, token_id

        if self._available:
            token_id = self._available.popleft()
        else:
            if self._allocated >= self._id_range:
                raise ValueError(f"Edge ID allocation exceeds {self._id_range}")
            token_id = (self._start + self._allocated) % self._id_range
            self._allocated += 1

        self._edge_to_token[edge_id] = token_id
        return True, token_id

    def reset(self):
        self._edge_to_token.clear()
        self._available.clear()
        self._start = random.randint(0, self._id_range - 1)
        self._allocated = 0


# arrays → sequence (direct, no intermediate objects)

BBOX_THRESHOLD = 1 / 2 ** (10 - 1)


def arrays_to_sequence(
    face_controls: np.ndarray,
    edge_controls: np.ndarray,
    outer_edge_indices: np.ndarray,
    face_outer_offsets: np.ndarray,
    inner_edge_indices: np.ndarray,
    inner_loop_offsets: np.ndarray,
    face_inner_offsets: np.ndarray,
    resolution: int = SPATIAL_RESOLUTION,
    edge_id_range: int = EDGE_ID_COUNT,
    complexity_token: int | None = None,
    bbox_threshold: float = BBOX_THRESHOLD,
    ordering: str = "interleaved",
    reuse_edge_ids: bool = True,
    random_edgeid_start: bool = True,
    geom_repr: str = "bezier",
) -> BRepSequence:
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    if not reuse_edge_ids:
        raise ValueError("Edge-ID recycling is required")
    num_faces = len(face_controls)
    num_edges = len(edge_controls)

    if geom_repr == "bezier":
        face_norms = [
            normalize_points_with_bbox(face_controls[i], bbox_threshold) for i in range(num_faces)
        ]
        edge_norms = [
            normalize_points_with_bbox(edge_controls[i], bbox_threshold) for i in range(num_edges)
        ]
    elif geom_repr == "uvgrid":
        # UV-grid representation: geom vector is a point grid (surface [G,G,3] / curve
        # [G,3]) sampled from the true surface/curve, normalized by *its own* AABB
        # (matches the pretrained VAE's training normalization). bbox = sample AABB.
        face_norms = [
            surface_controls_to_grid_bbox(face_controls[i], bbox_threshold=bbox_threshold)
            for i in range(num_faces)
        ]
        edge_norms = [
            curve_controls_to_grid_bbox(edge_controls[i], bbox_threshold=bbox_threshold)
            for i in range(num_edges)
        ]
    else:
        raise ValueError(f"Unknown geom_repr: {geom_repr!r}")

    seq = BRepSequence()
    registry = EdgeRegistry(
        id_range=edge_id_range,
        reuse_ids=reuse_edge_ids,
        random_start=random_edgeid_start,
    )
    idx = 0

    def emit(token):
        nonlocal idx
        seq.text_ids.append(token)
        seq.text_indexes.append(idx)
        idx += 1

    def emit_bbox(bbox):
        for tok in bbox_to_tokens(bbox, resolution):
            emit(tok)

    def emit_geom(geom_idx_list, vectors_list, pts):
        nonlocal idx
        vectors_list.append(pts.astype(np.float32, copy=False))
        geom_idx_list.append(idx)
        idx += 1

    def emit_loop(edge_ids):
        emit(BRepTokenType.LOOP_START)
        edge_ids = list(edge_ids)
        n = len(edge_ids)
        if n > 1:
            is_ref = [int(eid) in registry._edge_to_token for eid in edge_ids]
            if any(is_ref):
                best_start, best_run = 0, 0
                for i in range(n):
                    run = 0
                    while run < n and is_ref[(i + run) % n]:
                        run += 1
                    if run > best_run:
                        best_run = run
                        best_start = i
                edge_ids = edge_ids[best_start:] + edge_ids[:best_start]
        for eid in edge_ids:
            is_new, tid = registry.register(int(eid))
            if is_new:
                emit(BRepTokenType.EDGE_NEW)
                emit(edge_id_to_token(tid))
                if eid < num_edges:
                    norm_pts, bbox = edge_norms[eid]
                    emit_bbox(bbox)
                    emit_geom(
                        seq.curve_geom_indexes,
                        seq.curve_geom_vectors,
                        norm_pts,
                    )
            else:
                emit(BRepTokenType.EDGE_REF)
                emit(edge_id_to_token(tid))

        emit(BRepTokenType.LOOP_END)

    emit(BRepTokenType.BOS)

    if complexity_token is not None:
        emit(complexity_token)

    for fi in range(num_faces):
        norm_pts, bbox = face_norms[fi]

        emit(BRepTokenType.FACE_START)
        emit_bbox(bbox)
        emit_geom(
            seq.surface_geom_indexes,
            seq.surface_geom_vectors,
            norm_pts,
        )

        outer = outer_edge_indices[face_outer_offsets[fi] : face_outer_offsets[fi + 1]]
        if len(outer) > 0:
            emit_loop(outer)

        for j in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
            inner = inner_edge_indices[inner_loop_offsets[j] : inner_loop_offsets[j + 1]]
            if len(inner) > 0:
                emit_loop(inner)

        emit(BRepTokenType.FACE_END)

    emit(BRepTokenType.EOS)
    seq.total_length = idx

    seq.text_ids = np.array(seq.text_ids, dtype=np.int64)
    seq.text_indexes = np.array(seq.text_indexes, dtype=np.int64)
    seq.position_ids = np.arange(idx, dtype=np.int64)
    seq.surface_geom_indexes = np.array(seq.surface_geom_indexes, dtype=np.int64)
    seq.curve_geom_indexes = np.array(seq.curve_geom_indexes, dtype=np.int64)
    seq.surface_geom_vectors = (
        np.stack(seq.surface_geom_vectors)
        if seq.surface_geom_vectors
        else np.empty((0,), dtype=np.float32)
    )
    seq.curve_geom_vectors = (
        np.stack(seq.curve_geom_vectors)
        if seq.curve_geom_vectors
        else np.empty((0,), dtype=np.float32)
    )
    return seq


def sequence_to_arrays(seq: BRepSequence, ordering: str = "interleaved") -> tuple[np.ndarray, ...]:
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    face_points_list = []
    edge_points_list = []
    outer_edge_indices = []
    face_outer_offsets = [0]
    inner_edge_indices = []
    inner_loop_offsets = [0]
    face_inner_offsets = [0]

    token_iter = iter(seq.text_ids)
    tid_to_global_edge_idx: dict[int, int] = {}
    surface_geom_iter = iter(seq.surface_geom_vectors)
    curve_geom_iter = iter(seq.curve_geom_vectors)

    current_face_outer = []
    current_face_inners = []  # List[List[int]]
    current_loop_edges = []

    # Simple state machine
    def read_bbox():
        toks = [next(token_iter) for _ in range(6)]
        return tokens_to_bbox(toks)

    try:
        while True:
            token = next(token_iter)

            if token == BRepTokenType.FACE_START:
                bbox = read_bbox()
                norm_pts = next(surface_geom_iter)
                face_points_list.append(denormalize_points_with_bbox(norm_pts, bbox))
                current_face_outer = []
                current_face_inners = []

            elif token == BRepTokenType.LOOP_START:
                current_loop_edges = []

            elif token == BRepTokenType.EDGE_NEW:
                tid = next(token_iter)
                bbox = read_bbox()
                norm_pts = next(curve_geom_iter)
                global_idx = len(edge_points_list)
                edge_points_list.append(denormalize_points_with_bbox(norm_pts, bbox))
                tid_to_global_edge_idx[tid] = global_idx
                current_loop_edges.append(global_idx)

            elif token == BRepTokenType.EDGE_REF:
                tid = next(token_iter)
                global_idx = tid_to_global_edge_idx[tid]
                current_loop_edges.append(global_idx)

            elif token == BRepTokenType.LOOP_END:
                if not current_face_outer and len(current_face_inners) == 0:
                    current_face_outer = current_loop_edges
                else:
                    current_face_inners.append(current_loop_edges)

            elif token == BRepTokenType.FACE_END:
                # Commit the topology data for the current face
                # 1. Outer
                outer_edge_indices.extend(current_face_outer)
                face_outer_offsets.append(len(outer_edge_indices))

                # 2. Inner
                for inner_loop in current_face_inners:
                    inner_edge_indices.extend(inner_loop)
                    inner_loop_offsets.append(len(inner_edge_indices))

                # Face inner offsets index into inner_loop_offsets
                face_inner_offsets.append(len(inner_loop_offsets) - 1)

            elif token == BRepTokenType.EOS:
                break

    except StopIteration:
        pass

    # Convert to NumPy arrays
    return (
        np.array(face_points_list) if face_points_list else np.empty((0,), dtype=np.float32),
        np.array(edge_points_list) if edge_points_list else np.empty((0,), dtype=np.float32),
        np.array(outer_edge_indices, dtype=np.int64),
        np.array(face_outer_offsets, dtype=np.int64),
        np.array(inner_edge_indices, dtype=np.int64),
        np.array(inner_loop_offsets, dtype=np.int64),
        np.array(face_inner_offsets, dtype=np.int64),
    )


def id_to_string(token_id: int) -> str:

    if token_id in BRepTokenType._value2member_map_:
        return BRepTokenType(token_id).name
    elif is_coord_token(token_id):
        return f"<Coord:{token_to_coord(token_id):.3f}>"
    elif is_edge_id_token(token_id):
        return f"<EdgeID:{token_to_edge_id(token_id)}>"
    else:
        return f"<Unknown:{token_id}>"


def sequence_to_string(seq: BRepSequence) -> str:
    token_map = {
        BRepTokenType.BOS: "<BOS>",
        BRepTokenType.EOS: "<EOS>",
        BRepTokenType.FACE_START: "<FACE>",
        BRepTokenType.FACE_END: "</FACE>",
        BRepTokenType.LOOP_START: "<LOOP>",
        BRepTokenType.LOOP_END: "</LOOP>",
        BRepTokenType.CURVE_GEOM: "<CURVE_GEOM>",
    }

    result = []
    iter_tokens = iter(seq.text_ids)

    for token in iter_tokens:
        if token in token_map:
            result.append(token_map[token])
        else:
            if is_coord_token(token):
                result.append("<Coord>")
            elif is_edge_id_token(token):
                result.append(str(token_to_edge_id(token)))
            else:
                result.append("<Geom Vector>")

    return " ".join(result)


# normalize helper


def sort_points_by_corners(points: np.ndarray) -> np.ndarray:
    if points.ndim == 2:
        if tuple(points[0, :3]) > tuple(points[-1, :3]):
            points = points[::-1]

    elif points.ndim == 3:
        ops = [
            (points[0, 0, :3], points),
            (points[0, -1, :3], points[:, ::-1]),
            (points[-1, 0, :3], points[::-1, :]),
            (points[-1, -1, :3], points[::-1, ::-1]),
        ]
        points = min(ops, key=lambda x: tuple(x[0]))[1]

        if points.shape[0] == points.shape[1] and tuple(points[0, 1, :3]) > tuple(points[1, 0, :3]):
            points = points.transpose(1, 0, 2)

    return np.ascontiguousarray(points)


def normalize_points_with_bbox(
    points: np.ndarray, tol: float = BBOX_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    points = sort_points_by_corners(points)
    xyz, w = points[..., :3], points[..., 3:4]
    w_norm = (w - 1) / (w + 1)

    pts_flat = xyz.reshape(-1, 3)
    vmin, vmax = pts_flat.min(0), pts_flat.max(0)
    center = (vmin + vmax) / 2.0
    span = (vmax - vmin).max()
    if span < tol:
        raise ValueError(f"Degenerate geometry with span {span:.6f} detected during normalization")
    scale = 2.0 / span

    norm_xyz = (xyz - center) * scale
    return np.concatenate([norm_xyz, w_norm], axis=-1), np.concatenate([vmin, vmax])


def denormalize_points_with_bbox(norm_points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    vmin, vmax = bbox[:3], bbox[3:]
    center = (vmin + vmax) / 2.0
    span = (vmax - vmin).max()
    if span < BBOX_THRESHOLD:
        raise ValueError(
            f"Degenerate geometry with span {span:.6f} detected during denormalization"
        )
    scale = 2.0 / span
    xyz, w = norm_points[..., :3], norm_points[..., 3:4]
    orig_xyz = (xyz / scale) + center
    degenerate = (vmax - vmin) < BBOX_THRESHOLD
    if degenerate.any():
        orig_xyz[..., degenerate] = center[degenerate]
    w_pred_safe = np.clip(w, a_min=-0.999, a_max=0.999)
    orig_w = (1.0 + w_pred_safe) / (1.0 - w_pred_safe)
    return np.concatenate([orig_xyz, orig_w], axis=-1)


def _grid_and_bbox_from_samples(grid: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a world-space sample grid by *its own* AABB and return (norm, bbox).

    This matches how the pretrained VAE was trained (BrepGen ``surf_ncs`` /
    ``edge_ncs``): the surface/curve fills ``[-1, 1]`` on its longest axis.  The
    returned bbox is the sample AABB — it is what gets tokenized and later used
    by :func:`denormalize_grid_with_bbox`, so norm/denorm stay exact.
    """
    pts = grid.reshape(-1, 3)
    vmin, vmax = pts.min(0), pts.max(0)
    center = (vmin + vmax) / 2.0
    span = (vmax - vmin).max()
    if span < tol:
        raise ValueError(f"Degenerate geometry with span {span:.6f} during grid normalization")
    norm = (grid - center) * (2.0 / span)
    return norm.astype(np.float32), np.concatenate([vmin, vmax]).astype(np.float32)


def surface_controls_to_grid_bbox(
    world_ctrl: np.ndarray,
    res: int = GEOM_GRID_RES,
    bbox_threshold: float = BBOX_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """World rational-Bézier surface control points -> (normalized UV grid, bbox).

    Evaluates the true surface (original weights) on the corner-canonicalized
    control net, then normalizes by the surface's own AABB.
    """
    ctrl = sort_points_by_corners(np.asarray(world_ctrl, dtype=np.float64))
    grid = eval_rational_bezier_surfaces(ctrl[None], res=res)[0]  # [G, G, 3] world
    return _grid_and_bbox_from_samples(grid, bbox_threshold)


def curve_controls_to_grid_bbox(
    world_ctrl: np.ndarray,
    res: int = GEOM_GRID_RES,
    bbox_threshold: float = BBOX_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """World rational-Bézier curve control points -> (normalized point grid, bbox)."""
    ctrl = sort_points_by_corners(np.asarray(world_ctrl, dtype=np.float64))
    grid = eval_rational_bezier_curves(ctrl[None], res=res)[0]  # [G, 3] world
    return _grid_and_bbox_from_samples(grid, bbox_threshold)


def denormalize_grid_with_bbox(norm_grid: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """De-normalize an xyz point grid (no weight channel) back to world space.

    Counterpart of :func:`denormalize_points_with_bbox` for the UV-grid geometry
    representation, where the geometry vector is a sampled point grid
    (surface ``[U, V, 3]`` / curve ``[N, 3]``) rather than rational control
    points.  The affine matches the xyz branch of the control-point version.
    """
    vmin, vmax = bbox[:3], bbox[3:]
    center = (vmin + vmax) / 2.0
    span = (vmax - vmin).max()
    if span < BBOX_THRESHOLD:
        raise ValueError(
            f"Degenerate geometry with span {span:.6f} detected during denormalization"
        )
    scale = 2.0 / span
    orig_xyz = (norm_grid[..., :3] / scale) + center
    degenerate = (vmax - vmin) < BBOX_THRESHOLD
    if degenerate.any():
        orig_xyz[..., degenerate] = center[degenerate]
    return orig_xyz


# NPZ → BRepSequence loader


def load_brep_sequence_from_npz(
    path: str,
    resolution: int = SPATIAL_RESOLUTION,
    edge_id_range: int = EDGE_ID_COUNT,
    face_indices: "list[int] | None" = None,
    reuse_edge_ids: bool = True,
    geom_repr: str = "bezier",
) -> BRepSequence:
    """Load a dataset-format NPZ file and return a BRepSequence.

    Args:
        path: Path to the ``.npz`` file containing ``face_controls``,
            ``edge_controls``, ``outer_edge_indices``, ``face_outer_offsets``,
            ``inner_edge_indices``, ``inner_loop_offsets``, ``face_inner_offsets``.
        resolution: Spatial quantisation resolution (default: SPATIAL_RESOLUTION).
        edge_id_range: Number of edge ID tokens available (default: EDGE_ID_COUNT).
        face_indices: Optional list of face indices to extract a subset.  When
            ``None`` all faces are used.

    Returns:
        A :class:`BRepSequence` ending with an EOS token.
    """
    if not reuse_edge_ids:
        raise ValueError("Edge-ID recycling is required")
    with np.load(path) as data:
        f_ctrl = data["face_controls"]
        e_ctrl = data["edge_controls"]
        o_edge = data["outer_edge_indices"]
        f_o_off = data["face_outer_offsets"]
        i_edge = data["inner_edge_indices"]
        i_l_off = data["inner_loop_offsets"]
        f_i_off = data["face_inner_offsets"]

    if face_indices is not None:
        f_ctrl, e_ctrl, o_edge, f_o_off, i_edge, i_l_off, f_i_off = extract_face_subset(
            f_ctrl,
            e_ctrl,
            o_edge,
            f_o_off,
            i_edge,
            i_l_off,
            f_i_off,
            face_indices=face_indices,
        )

    return arrays_to_sequence(
        face_controls=f_ctrl,
        edge_controls=e_ctrl,
        outer_edge_indices=o_edge,
        face_outer_offsets=f_o_off,
        inner_edge_indices=i_edge,
        inner_loop_offsets=i_l_off,
        face_inner_offsets=f_i_off,
        resolution=resolution,
        edge_id_range=edge_id_range,
        complexity_token=BRepTokenType.COMPLEXITY_L3,
        reuse_edge_ids=reuse_edge_ids,
        geom_repr=geom_repr,
    )


def extract_face_subset(
    face_controls: np.ndarray,
    edge_controls: np.ndarray,
    outer_edge_indices: np.ndarray,
    face_outer_offsets: np.ndarray,
    inner_edge_indices: np.ndarray,
    inner_loop_offsets: np.ndarray,
    face_inner_offsets: np.ndarray,
    face_indices: list[int],
) -> tuple[np.ndarray, ...]:
    """Extract a subset of faces (and their referenced edges) from BRepData arrays.

    Edge indices are remapped to a dense contiguous range.  Face ordering in the
    output matches the order given in *face_indices*.

    Returns:
        Tuple of seven arrays with the same schema as the input.
    """
    face_indices = list(face_indices)

    # Collect which edges are referenced by the selected faces
    referenced_edges: list[int] = []
    seen_edges: dict = {}

    def _collect_loop(loop_edge_ids):
        for eid in loop_edge_ids:
            eid = int(eid)
            if eid not in seen_edges:
                seen_edges[eid] = len(referenced_edges)
                referenced_edges.append(eid)

    for fi in face_indices:
        # outer loop
        outer = outer_edge_indices[face_outer_offsets[fi] : face_outer_offsets[fi + 1]]
        _collect_loop(outer)
        # inner loops
        for li in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
            inner = inner_edge_indices[inner_loop_offsets[li] : inner_loop_offsets[li + 1]]
            _collect_loop(inner)

    # Build new arrays
    sub_face_controls = face_controls[face_indices]
    sub_edge_controls = (
        edge_controls[referenced_edges]
        if referenced_edges
        else np.empty((0,) + edge_controls.shape[1:], dtype=edge_controls.dtype)
    )

    sub_outer_edge_indices: list[int] = []
    sub_face_outer_offsets: list[int] = [0]

    sub_inner_edge_indices: list[int] = []
    sub_inner_loop_offsets: list[int] = [0]
    sub_face_inner_offsets: list[int] = [0]

    for fi in face_indices:
        outer = outer_edge_indices[face_outer_offsets[fi] : face_outer_offsets[fi + 1]]
        sub_outer_edge_indices.extend(seen_edges[int(e)] for e in outer)
        sub_face_outer_offsets.append(len(sub_outer_edge_indices))

        for li in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
            inner = inner_edge_indices[inner_loop_offsets[li] : inner_loop_offsets[li + 1]]
            sub_inner_edge_indices.extend(seen_edges[int(e)] for e in inner)
            sub_inner_loop_offsets.append(len(sub_inner_edge_indices))

        sub_face_inner_offsets.append(len(sub_inner_loop_offsets) - 1)

    return (
        sub_face_controls,
        sub_edge_controls,
        np.array(sub_outer_edge_indices, dtype=np.int64),
        np.array(sub_face_outer_offsets, dtype=np.int64),
        np.array(sub_inner_edge_indices, dtype=np.int64),
        np.array(sub_inner_loop_offsets, dtype=np.int64),
        np.array(sub_face_inner_offsets, dtype=np.int64),
    )
