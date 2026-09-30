"""Checks for the NPZ geometry and closed two-manifold training topology."""

import numpy as np


def validate_arrays(data, max_faces=50, max_edges=1000, max_face_edges=30):
    faces, edges = data["face_controls"], data["edge_controls"]
    for name, array, shape, limit in (
        ("face_controls", faces, (4, 4, 4), max_faces),
        ("edge_controls", edges, (4, 4), max_edges),
    ):
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape or not 0 < len(array) <= limit:
            raise ValueError(f"Invalid {name} shape: {array.shape}")
        if not np.isfinite(array).all() or np.any(array[..., 3] <= 0):
            raise ValueError(f"Nonfinite controls or nonpositive rational weights in {name}")
        if np.max(np.abs(array[..., :3])) > 1.00001:
            raise ValueError(f"Unnormalized coordinates in {name}")
        xyz = array[..., :3].reshape(len(array), -1, 3)
        if np.any(np.ptp(xyz, axis=1).max(axis=1) < 1 / 512):
            raise ValueError(f"Geometry smaller than training bbox threshold in {name}")

    def offsets(name, count, end, allow_empty):
        array = data[name]
        if array.shape != (count + 1,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Invalid {name} shape or dtype")
        if array[0] != 0 or array[-1] != end or np.any(np.diff(array) < (0 if allow_empty else 1)):
            raise ValueError(f"Invalid {name} boundaries")
        return array

    outer, inner = data["outer_edge_indices"], data["inner_edge_indices"]
    for array in (outer, inner):
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Edge indices must be one-dimensional integers")
        if np.any(array < 0) or np.any(array >= len(edges)):
            raise ValueError("Edge index outside edge_controls")
    outer_offsets = offsets("face_outer_offsets", len(faces), len(outer), False)
    num_inner_loops = len(data["inner_loop_offsets"]) - 1
    loop_offsets = offsets("inner_loop_offsets", num_inner_loops, len(inner), False)
    face_offsets = offsets("face_inner_offsets", len(faces), num_inner_loops, True)
    incidence = np.zeros(len(edges), dtype=np.int32)
    for face in range(len(faces)):
        ids = list(outer[outer_offsets[face] : outer_offsets[face + 1]])
        for loop in range(face_offsets[face], face_offsets[face + 1]):
            ids.extend(inner[loop_offsets[loop] : loop_offsets[loop + 1]])
        if len(ids) > max_face_edges or len(ids) != len(set(ids)):
            raise ValueError("Face has too many edges or repeated edge references")
        incidence[ids] += 1
    if np.any(incidence != 2):
        raise ValueError("Every edge must belong to exactly two distinct faces")
