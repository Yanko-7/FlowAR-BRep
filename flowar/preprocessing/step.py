"""Extract one training sample per solid from STEP geometry."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.ShapeUpgrade import (
    ShapeUpgrade_ShapeDivideAngle,
    ShapeUpgrade_ShapeDivideClosed,
    ShapeUpgrade_ShapeDivideClosedEdges,
)
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Extend.DataExchange import read_step_file

from flowar.preprocessing.geometry import (
    extract_or_fit_bicubic_patch,
    extract_or_fit_cubic_curve,
)
from flowar.preprocessing.validation import validate_arrays


@dataclass(frozen=True)
class PreprocessOptions:
    max_faces: int = 50
    max_edges: int = 1000
    max_face_edges: int = 30
    tolerance: float = 0.01


def prepare_shape(shape, tolerance: float):
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    shape = fixer.Shape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Invalid solid after shape repair")

    # Closed-shape splitting follows the occwl API pattern:
    # https://github.com/AutodeskAILab/occwl/blob/main/src/occwl/base.py
    for divider_type in (ShapeUpgrade_ShapeDivideClosed, ShapeUpgrade_ShapeDivideClosedEdges):
        divider = divider_type(shape)
        divider.SetPrecision(tolerance)
        divider.SetMinTolerance(tolerance * 0.1)
        divider.SetMaxTolerance(tolerance)
        divider.SetNbSplitPoints(1)
        if divider.Perform():
            shape = divider.Result()
        # Closed-face splitting can leave invalid seam/pole bookkeeping (for
        # example on spheres); repair it before splitting the resulting edges.
        if not BRepCheck_Analyzer(shape).IsValid():
            fixer = ShapeFix_Shape(shape)
            fixer.Perform()
            shape = fixer.Shape()
    divider = ShapeUpgrade_ShapeDivideAngle(math.pi, shape)
    if divider.Perform():
        shape = divider.Result()
    if not shape.IsNull() and not BRepCheck_Analyzer(shape).IsValid():
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Invalid solid after splitting closed faces and edges")
    return shape


def extract_arrays(shape, options: PreprocessOptions) -> dict[str, np.ndarray]:
    faces = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, faces)
    if not 0 < faces.Size() <= options.max_faces:
        raise ValueError(f"Face count {faces.Size()} exceeds range [1, {options.max_faces}]")

    edges = TopTools_IndexedMapOfShape()
    face_controls, edge_controls = [], []
    outer_edges, inner_edges = [], []
    face_outer_offsets, inner_loop_offsets, face_inner_offsets = [0], [0], [0]

    def extract_loop(wire, face):
        loop = []
        explorer = BRepTools_WireExplorer(wire, face)
        visited = 0
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            visited += 1
            if not BRep_Tool.Degenerated(edge):
                index = edges.FindIndex(edge)
                if index == 0:
                    index = edges.Add(edge)
                    if index > options.max_edges:
                        raise ValueError(f"Edge count exceeds {options.max_edges}")
                    edge_controls.append(extract_or_fit_cubic_curve(edge))
                loop.append(index - 1)
            explorer.Next()
        expected = TopExp_Explorer(wire, TopAbs_EDGE)
        count = 0
        while expected.More():
            count += 1
            expected.Next()
        if count != visited or not loop:
            raise ValueError("Empty or incompletely traversed boundary wire")
        if len(loop) != len(set(loop)):
            raise ValueError("A boundary wire references the same edge more than once")
        return loop

    for index in range(1, faces.Size() + 1):
        face = topods.Face(faces.FindKey(index))
        face_controls.append(extract_or_fit_bicubic_patch(face))
        outer = breptools.OuterWire(face)
        if outer.IsNull():
            raise ValueError("Face has no outer boundary wire")
        outer_edges.extend(extract_loop(outer, face))
        face_outer_offsets.append(len(outer_edges))
        wires = TopExp_Explorer(face, TopAbs_WIRE)
        while wires.More():
            wire = topods.Wire(wires.Current())
            if not wire.IsSame(outer):
                inner_edges.extend(extract_loop(wire, face))
                inner_loop_offsets.append(len(inner_edges))
            wires.Next()
        face_inner_offsets.append(len(inner_loop_offsets) - 1)

    face_controls = np.asarray(face_controls, dtype=np.float64)
    edge_controls = np.asarray(edge_controls, dtype=np.float64)
    if edge_controls.size == 0:
        raise ValueError("Solid has no nondegenerate edges")
    xyz = np.concatenate([face_controls.reshape(-1, 4), edge_controls.reshape(-1, 4)])[:, :3]
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    span = (high - low).max()
    if not np.isfinite(xyz).all() or not np.isfinite(span) or span <= 0:
        raise ValueError("Nonfinite or zero-size geometry")
    center, scale = (low + high) / 2, 2.0 / span
    for controls in (face_controls, edge_controls):
        controls[..., :3] = (controls[..., :3] - center) * scale
    data = {
        "face_controls": face_controls.astype(np.float32),
        "edge_controls": edge_controls.astype(np.float32),
        "outer_edge_indices": np.asarray(outer_edges, dtype=np.int32),
        "face_outer_offsets": np.asarray(face_outer_offsets, dtype=np.int32),
        "inner_edge_indices": np.asarray(inner_edges, dtype=np.int32),
        "inner_loop_offsets": np.asarray(inner_loop_offsets, dtype=np.int32),
        "face_inner_offsets": np.asarray(face_inner_offsets, dtype=np.int32),
        "center": center,
        "scale": np.asarray(scale),
        "f_count": np.asarray([len(face_controls)], dtype=np.int32),
        "e_count": np.asarray([len(edge_controls)], dtype=np.int32),
    }
    validate_arrays(data, options.max_faces, options.max_edges, options.max_face_edges)
    return data


def convert_file(path: str, output_dir: str, options: PreprocessOptions, overwrite=False):
    """Return per-solid results; the caller isolates this function in a timed worker."""
    path, output_dir = Path(path), Path(output_dir)
    with path.open("rb") as stream:
        signature = hashlib.file_digest(stream, "sha256")
    signature.update(json.dumps(asdict(options), sort_keys=True).encode())
    signature.update(b"FlowAR STEP preprocessing v1")
    signature = signature.hexdigest()
    shape = read_step_file(str(path), verbosity=False)
    solids = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_SOLID, solids)
    if not solids.Size():
        raise ValueError("STEP contains no solids; open shells are not training samples")
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, solids.Size() + 1):
        name = f"{path.stem}__solid{index - 1:03d}"
        dest = output_dir / f"{name}.npz"
        temporary = dest.with_suffix(".npz.tmp")
        result = {"source": path.name, "solid": index - 1, "output": dest.name}
        try:
            if dest.exists() and not overwrite:
                with np.load(dest, allow_pickle=False) as existing:
                    if str(existing.get("preprocessing_signature", "")) != signature:
                        raise ValueError(
                            "Existing output has different inputs/settings; use --overwrite"
                        )
                    validate_arrays(
                        existing, options.max_faces, options.max_edges, options.max_face_edges
                    )
                result["status"] = "existing"
            else:
                solid = topods.Solid(solids.FindKey(index))
                data = extract_arrays(prepare_shape(solid, options.tolerance), options)
                data.update(
                    source=np.asarray(path.name),
                    solid_index=np.asarray(index - 1),
                    preprocessing_signature=np.asarray(signature),
                )
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **data)
                temporary.replace(dest)
                result["status"] = "written"
        except Exception as exc:
            result.update(status="rejected", reason=str(exc))
        finally:
            temporary.unlink(missing_ok=True)
        results.append(result)
    return results
