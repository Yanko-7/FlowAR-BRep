from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import (
    ShapeFix_Edge,
    ShapeFix_Face,
    ShapeFix_Shape,
    ShapeFix_Wire,
)
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
)
from OCC.Core.TopExp import topexp
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Wire,
    topods,
)
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
)
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from flowar.data.sequence import (
    BRepTokenType,
    denormalize_grid_with_bbox,
    denormalize_points_with_bbox,
    is_complexity_token,
    token_to_edge_id,
    tokens_to_bbox,
)


@dataclass
class BRepData:
    face_controls: np.ndarray  # (N, 4, 4, 4) - x, y, z, weight
    edge_controls: np.ndarray  # (M, 4, 4) - x, y, z, weight
    outer_edge_indices: np.ndarray
    face_outer_offsets: np.ndarray
    inner_edge_indices: np.ndarray
    inner_loop_offsets: np.ndarray
    face_inner_offsets: np.ndarray

    @classmethod
    def load_npz(cls, path: str) -> "BRepData":
        with np.load(path) as data:
            return cls(**{k: data[k] for k in data.files})


def _bezier_surf_decode(vec: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return denormalize_points_with_bbox(vec.reshape(4, 4, 4), bbox)


def _bezier_curv_decode(vec: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return denormalize_points_with_bbox(vec.reshape(4, 4), bbox)


def _grid_surf_decode(grid: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return denormalize_grid_with_bbox(grid, bbox)


def _grid_curv_decode(grid: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return denormalize_grid_with_bbox(grid, bbox)


def convert_generated_to_brep_data(
    ids: list[int],
    surf_vecs: list[np.ndarray],
    curv_vecs: list[np.ndarray],
    ordering: str = "interleaved",
    surf_decode=_bezier_surf_decode,
    curv_decode=_bezier_curv_decode,
) -> BRepData:
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")

    surf_it, curv_it = iter(surf_vecs), iter(curv_vecs)
    face_ctrls, edge_ctrls = [], []
    active_edges: dict[int, int] = {}

    outer_edges, face_outer_offsets = [], [0]
    inner_edges, inner_loop_offsets, face_inner_offsets = [], [0], [0]

    idx, n = 0, len(ids)
    loop_idx = 0

    while idx < n:
        tok = ids[idx]

        if tok in (BRepTokenType.BOS, BRepTokenType.PAD) or is_complexity_token(tok):
            idx += 1
            continue
        if tok == BRepTokenType.EOS:
            break

        if tok == BRepTokenType.FACE_START:
            if idx + 7 > n:
                raise ValueError("Truncated bbox — treat as early EOS")
            bbox = tokens_to_bbox(ids[idx + 1 : idx + 7])
            face_ctrls.append(surf_decode(next(surf_it), bbox))
            loop_idx = 0
            idx += 8

        elif tok == BRepTokenType.FACE_END:
            face_outer_offsets.append(len(outer_edges))
            face_inner_offsets.append(len(inner_loop_offsets) - 1)
            idx += 1

        elif tok == BRepTokenType.LOOP_START:
            idx += 1
            curr_loop = []

            while idx < n and ids[idx] != BRepTokenType.LOOP_END:
                ltok = ids[idx]
                edge_id = token_to_edge_id(ids[idx + 1])

                if ltok == BRepTokenType.EDGE_NEW:
                    if idx + 8 > n:
                        raise ValueError("Truncated bbox — treat as early EOS")
                    bbox = tokens_to_bbox(ids[idx + 2 : idx + 8])
                    canonical_idx = len(edge_ctrls)
                    edge_ctrls.append(curv_decode(next(curv_it), bbox))
                    active_edges[edge_id] = canonical_idx
                    curr_loop.append(canonical_idx)
                    idx += 9

                elif ltok == BRepTokenType.EDGE_REF:
                    curr_loop.append(active_edges.pop(edge_id))
                    idx += 2
                else:
                    raise ValueError(f"Invalid loop token {ltok} at {idx}.")

            if loop_idx == 0:
                outer_edges.extend(curr_loop)
            else:
                inner_edges.extend(curr_loop)
                inner_loop_offsets.append(len(inner_edges))

            loop_idx += 1
            idx += 1

        else:
            raise ValueError(f"Unexpected token {tok} at index {idx}.")

    return BRepData(
        face_controls=np.asarray(face_ctrls, dtype=np.float32),
        edge_controls=np.asarray(edge_ctrls, dtype=np.float32),
        outer_edge_indices=np.asarray(outer_edges, dtype=np.int32),
        face_outer_offsets=np.asarray(face_outer_offsets, dtype=np.int32),
        inner_edge_indices=np.asarray(inner_edges, dtype=np.int32),
        inner_loop_offsets=np.asarray(inner_loop_offsets, dtype=np.int32),
        face_inner_offsets=np.asarray(face_inner_offsets, dtype=np.int32),
    )


class BRepShapeBuilder:
    def __init__(self, data: BRepData):
        self.data = data
        self._errors: List[str] = []
        self._tolerance = 0.01
        self._degree = 3

    @property
    def has_errors(self) -> bool:
        return len(self._errors) > 0

    def get_error_report(self) -> List[str]:
        return self._errors.copy()

    def build_trimmed_compound(self) -> Optional[TopoDS_Shape]:
        """Build a compound of trimmed faces without sewing or watertight validation.

        Suitable for partial face subsets that are not closed solids.
        Returns a TopoDS_Compound of properly trimmed faces, or None on failure.
        """
        try:
            unique_verts, edge_adj = self._extract_topology()
            snapped_edges = self._snap_edges(unique_verts, edge_adj)
            edges = [
                BRepBuilderAPI_MakeEdge(self._ctrls2curve(pts)).Edge() for pts in snapped_edges
            ]
            faces = self._build_faces(self.data.face_controls, edges)
            if not faces:
                return None
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for f in faces:
                builder.Add(compound, f)
            return compound
        except Exception as e:
            self._errors.append(f"build_trimmed_compound failed: {e}")
            return None

    def build(self) -> Optional[TopoDS_Shape]:
        self._errors.clear()
        try:
            unique_verts, edge_adj = self._extract_topology()
            snapped_edges = self._snap_edges(unique_verts, edge_adj)

            edges = [
                BRepBuilderAPI_MakeEdge(self._ctrls2curve(pts)).Edge() for pts in snapped_edges
            ]
            faces = self._build_faces(self.data.face_controls, edges)

            shape = self._sew_and_solidify(faces)
            self._validate_shape(shape)

            return shape if not self.has_errors else None
        except Exception as e:
            self._errors.append(f"Critical Build Exception: {str(e)}")
            return None

    def _extract_topology(self) -> Tuple[np.ndarray, np.ndarray]:
        num_edges = len(self.data.edge_controls)
        rows, cols = [], []

        def process_loop(edges_idx):
            N = len(edges_idx)
            if N == 0:
                return
            if N == 1:
                e = edges_idx[0]
                rows.extend([2 * e, 2 * e + 1])
                cols.extend([2 * e + 1, 2 * e])
                return

            for i in range(N):
                e1, e2 = edges_idx[i], edges_idx[(i + 1) % N]
                p1, p2 = (
                    self.data.edge_controls[e1][[0, -1], :3],
                    self.data.edge_controls[e2][[0, -1], :3],
                )
                dist = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
                idx1, idx2 = np.unravel_index(dist.argmin(), dist.shape)
                n1, n2 = 2 * e1 + idx1, 2 * e2 + idx2
                rows.extend([n1, n2])
                cols.extend([n2, n1])

        d = self.data
        for f in range(len(d.face_outer_offsets) - 1):
            process_loop(
                d.outer_edge_indices[d.face_outer_offsets[f] : d.face_outer_offsets[f + 1]]
            )
            for li in range(d.face_inner_offsets[f], d.face_inner_offsets[f + 1]):
                process_loop(
                    d.inner_edge_indices[d.inner_loop_offsets[li] : d.inner_loop_offsets[li + 1]]
                )

        adj_matrix = coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(2 * num_edges, 2 * num_edges),
        )
        n_comps, labels = connected_components(adj_matrix, directed=False)

        all_pts = d.edge_controls[:, [0, -1], :3].reshape(-1, 3)
        sums = np.zeros((n_comps, 3))
        np.add.at(sums, labels, all_pts)
        counts = np.bincount(labels, minlength=n_comps)[:, None]

        return sums / counts, labels.reshape(-1, 2)

    def _snap_edges(self, unique_vertices: np.ndarray, edge_vertex_adj: np.ndarray) -> np.ndarray:
        ep = self.data.edge_controls.copy()
        ep[:, 0, :3] = unique_vertices[edge_vertex_adj[:, 0]]
        ep[:, -1, :3] = unique_vertices[edge_vertex_adj[:, 1]]
        return ep

    def _build_faces(
        self, face_controls: np.ndarray, edges: List[TopoDS_Edge]
    ) -> List[TopoDS_Face]:
        faces = []
        d = self.data
        for f_idx, ctrls in enumerate(face_controls):
            out_wire = self._build_healed_wire(
                self._get_slice_indices(d.face_outer_offsets, f_idx, d.outer_edge_indices),
                edges,
            )
            in_wires = []

            for li in self._get_slice_indices(
                d.face_inner_offsets, f_idx, np.arange(len(d.inner_loop_offsets))
            ):
                if in_w := self._build_healed_wire(
                    self._get_slice_indices(d.inner_loop_offsets, li, d.inner_edge_indices),
                    edges,
                ):
                    in_wires.append(in_w)

            surf = self._ctrls2surface(ctrls)
            face_maker = BRepBuilderAPI_MakeFace(surf, out_wire)
            for w in in_wires:
                # w.Reverse()
                face_maker.Add(w)

            if not face_maker.IsDone():
                self._errors.append(f"MakeFace failed for face {f_idx}")
                continue

            raw_face = face_maker.Face()
            self._fix_wires(raw_face)
            self._add_pcurves(raw_face)
            self._fix_wires(raw_face)
            faces.append(self._fix_face(raw_face))

        return faces

    def _shape_to_solids(self, shape: TopoDS_Shape) -> List[TopoDS_Shape]:
        st = shape.ShapeType()
        if st == TopAbs_SOLID:
            return [shape]
        if st == TopAbs_SHELL:
            maker = BRepBuilderAPI_MakeSolid()
            maker.Add(topods.Shell(shape))
            maker.Build()
            if maker.IsDone():
                return [maker.Solid()]
            self._errors.append("MakeSolid failed for a shell")
            return []
        if st == TopAbs_COMPOUND:
            solids = []
            it = TopoDS_Iterator(shape)
            while it.More():
                solids.extend(self._shape_to_solids(it.Value()))
                it.Next()
            return solids
        self._errors.append(f"Cannot convert shape type {st} to solid, skipping")
        return []

    def _sew_and_solidify(self, faces: List[TopoDS_Face]) -> TopoDS_Shape:
        sewing = BRepBuilderAPI_Sewing(self._tolerance)
        for f in faces:
            sewing.Add(f)
        sewing.Perform()
        sewn_shape = sewing.SewedShape()

        if sewn_shape.IsNull():
            self._errors.append("Sewing produced a null shape")
            return sewn_shape

        solids = self._shape_to_solids(sewn_shape)
        if not solids:
            self._errors.append("No shells found in sewn shape")
            return sewn_shape

        if len(solids) == 1:
            compound = solids[0]
        else:
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for s in solids:
                builder.Add(compound, s)

        global_fixer = ShapeFix_Shape(compound)
        global_fixer.SetPrecision(self._tolerance)
        global_fixer.SetMaxTolerance(self._tolerance * 10)
        global_fixer.Perform()
        return global_fixer.Shape()

    def _validate_shape(self, shape: TopoDS_Shape):
        if not BRepCheck_Analyzer(shape).IsValid():
            self._errors.append(
                "Topology Validation Failed: Unclosed geometry, open shells, or self-intersections."
            )
        if not self._is_watertight(shape):
            self._errors.append(
                "Watertightness Check Failed: Detected edges belonging to fewer than 2 faces."
            )

    def _is_watertight(self, shape: TopoDS_Shape) -> bool:
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

        for i in range(1, edge_face_map.Size() + 1):
            if BRep_Tool.Degenerated(edge_face_map.FindKey(i)):
                continue
            if edge_face_map.FindFromIndex(i).Size() < 2:
                return False
        return True

    # def _build_healed_wire(
    #     self, edge_indices: np.ndarray, edges: List[TopoDS_Edge]
    # ) -> Optional[TopoDS_Wire]:
    #     if not len(edge_indices):
    #         return None
    #     wire_data = ShapeExtend_WireData()
    #     for idx in edge_indices:
    #         wire_data.Add(edges[idx])

    #     fixer = ShapeFix_Wire()
    #     fixer.Load(wire_data)
    #     fixer.SetPrecision(1e-3)
    #     fixer.SetMaxTolerance(1e-2)
    #     fixer.SetClosedWireMode(True)
    #     fixer.Perform()
    #     return fixer.Wire()

    def _build_healed_wire(
        self, edge_indices: np.ndarray, edges: List[TopoDS_Edge]
    ) -> Optional[TopoDS_Wire]:
        if not len(edge_indices):
            return None
        wire_builder = BRepBuilderAPI_MakeWire()
        for idx in edge_indices:
            wire_builder.Add(edges[idx])
        return wire_builder.Wire()

    def _fix_wires(self, face: TopoDS_Face):
        for wire in TopologyExplorer(face).wires():
            wire_fixer = ShapeFix_Wire(wire, face, self._tolerance)
            assert wire_fixer.IsReady()
            wire_fixer.Perform()

    def _add_pcurves(self, face: TopoDS_Face):
        edge_fixer = ShapeFix_Edge()
        for wire in TopologyExplorer(face).wires():
            for edge in WireExplorer(wire).ordered_edges():
                edge_fixer.FixAddPCurve(edge, face, False, 1e-3)

    def _fix_face(self, face: TopoDS_Face) -> TopoDS_Face:
        fixer = ShapeFix_Face(face)
        fixer.SetPrecision(self._tolerance)
        fixer.SetMaxTolerance(0.1)
        fixer.Perform()
        fixer.FixOrientation()
        return fixer.Face()

    def _get_slice_indices(
        self, offsets: np.ndarray, idx: int, target_array: np.ndarray
    ) -> np.ndarray:
        start = offsets[idx]
        end = offsets[idx + 1] if idx + 1 < len(offsets) else len(target_array)
        return target_array[start:end] if start != end else np.array([], dtype=int)

    # --- NURBS Construction ---

    def _create_knots_mults(
        self,
    ) -> Tuple[TColStd_Array1OfReal, TColStd_Array1OfInteger]:
        knots = TColStd_Array1OfReal(1, 2)
        knots.SetValue(1, 0.0)
        knots.SetValue(2, 1.0)
        mults = TColStd_Array1OfInteger(1, 2)
        mults.SetValue(1, self._degree + 1)
        mults.SetValue(2, self._degree + 1)
        return knots, mults

    def _ctrls2curve(self, controls: np.ndarray) -> Geom_BSplineCurve:
        poles = TColgp_Array1OfPnt(1, 4)
        weights = TColStd_Array1OfReal(1, 4)
        for i, (x, y, z, w) in enumerate(controls):
            poles.SetValue(i + 1, gp_Pnt(float(x), float(y), float(z)))
            weights.SetValue(i + 1, float(w))

        knots, mults = self._create_knots_mults()
        return Geom_BSplineCurve(poles, weights, knots, mults, self._degree)

    def _ctrls2surface(self, controls: np.ndarray) -> Geom_BSplineSurface:
        poles = TColgp_Array2OfPnt(1, 4, 1, 4)
        weights = TColStd_Array2OfReal(1, 4, 1, 4)
        for i in range(4):
            for j in range(4):
                x, y, z, w = controls[i, j]
                poles.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
                weights.SetValue(i + 1, j + 1, float(w))

        knots, mults = self._create_knots_mults()
        return Geom_BSplineSurface(
            poles, weights, knots, knots, mults, mults, self._degree, self._degree
        )


class BRepShapeBuilderUV(BRepShapeBuilder):
    """BRep builder for the UV-grid geometry representation.

    ``data.face_controls`` are world-space UV point grids ``[Nf, U, V, 3]`` and
    ``data.edge_controls`` are world-space polylines ``[Ne, P, 3]``.  All topology
    handling (vertex extraction / snapping / wires / sewing) is inherited from
    :class:`BRepShapeBuilder` unchanged — it only touches the ``[0, -1]`` endpoint
    rows, which are valid for grids.  Only the geometry fitting differs: BSpline
    approximation from sampled points (à la BrepGen ``construct_brep``) instead of
    rational Bézier control points.
    """

    # BSpline fitting parameters (match BrepGen construct_brep)
    _SURF_TOL = 5e-2
    _EDGE_TOLS = (5e-3, 8e-3, 5e-2)

    def _ctrls2curve(self, points: np.ndarray) -> Geom_BSplineCurve:
        pts = np.asarray(points)[:, :3].astype(float)
        # Drop near-coincident consecutive points (GeomAPI fails on duplicates).
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("Degenerate edge polyline (<2 distinct points)")
        arr = TColgp_Array1OfPnt(1, len(pts))
        for i, (x, y, z) in enumerate(pts):
            arr.SetValue(i + 1, gp_Pnt(float(x), float(y), float(z)))
        # C2 BSpline approximation (BrepGen's tolerance ladder). If it cannot fit,
        # fail honestly — no continuity/degree degradation.
        last_exc = None
        for tol in self._EDGE_TOLS:
            try:
                return GeomAPI_PointsToBSpline(arr, 0, 8, GeomAbs_C2, tol).Curve()
            except Exception as e:  # noqa: BLE001 - tolerance ladder
                last_exc = e
        raise last_exc

    def _ctrls2surface(self, points: np.ndarray) -> Geom_BSplineSurface:
        pts = np.asarray(points)[..., :3]
        U, V = pts.shape[0], pts.shape[1]
        arr = TColgp_Array2OfPnt(1, U, 1, V)
        for i in range(U):
            for j in range(V):
                x, y, z = pts[i, j]
                arr.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
        # Single C2 fit (matches BrepGen). Raises StdFail on failure → honest fail.
        return GeomAPI_PointsToBSplineSurface(arr, 3, 8, GeomAbs_C2, self._SURF_TOL).Surface()


def convert_generated_to_brep_data_uv(
    ids: list[int],
    surf_vecs: list[np.ndarray],
    curv_vecs: list[np.ndarray],
    vae=None,
    ordering: str = "interleaved",
) -> BRepData:
    """UV-grid counterpart of :func:`convert_generated_to_brep_data`.

    If *vae* is given, ``surf_vecs`` / ``curv_vecs`` are per-primitive VAE latents
    (48-d / 12-d) that are batch-decoded to normalized point grids. If *vae* is
    ``None``, they are assumed to already be normalized point grids (surface
    ``[32,32,3]`` / curve ``[32,3]``) — used when decoding happens upstream (e.g.
    on the GPU in the main process before dispatching CPU build workers).
    The shared token walk then de-normalizes each grid with its bbox.
    """
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    if vae is not None:
        import torch

        device = next(vae.parameters()).device
        if surf_vecs:
            z = torch.as_tensor(np.stack(surf_vecs), dtype=torch.float32, device=device)
            surf_grids = list(vae.decode_surf(z).cpu().numpy())
        else:
            surf_grids = []
        if curv_vecs:
            z = torch.as_tensor(np.stack(curv_vecs), dtype=torch.float32, device=device)
            edge_grids = list(vae.decode_edge(z).cpu().numpy())
        else:
            edge_grids = []
    else:
        surf_grids = list(surf_vecs)
        edge_grids = list(curv_vecs)

    return convert_generated_to_brep_data(
        ids,
        surf_grids,
        edge_grids,
        ordering=ordering,
        surf_decode=_grid_surf_decode,
        curv_decode=_grid_curv_decode,
    )
