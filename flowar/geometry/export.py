from __future__ import annotations

from pathlib import Path

import numpy as np
from OCC.Core.BRepTools import breptools
from OCC.Extend.DataExchange import write_step_file

from flowar.data.sequence import _parse_bboxes, denormalize_points_with_bbox
from flowar.generation.constraints import count_faces_edges
from flowar.geometry.bezier import (
    eval_rational_bezier_curves,
    eval_rational_bezier_surfaces,
)
from flowar.geometry.builder import (
    BRepShapeBuilder,
    BRepShapeBuilderUV,
    convert_generated_to_brep_data,
    convert_generated_to_brep_data_uv,
)


def sample_stem(rank: int, batch_index: int, sample_index: int, ids: list[int]) -> str:
    """One identifier for CAD, preview, raw data and trajectory files."""
    faces, edges = count_faces_edges(ids)
    return f"r{rank}_b{batch_index:05d}_s{sample_index:03d}_{faces}f_{edges}e"


def save_trajectory_npz(
    out_path,
    ids: list,
    surf_trajs_list: list,
    curv_trajs_list: list,
) -> None:
    """Save raw latent trajectories and decoded control-point trajectories to npz."""
    face_bboxes_raw, edge_bboxes_raw = _parse_bboxes(ids)

    # ── Raw latent arrays ────────────────────────────────────────────────────
    face_latent = (
        np.stack(surf_trajs_list, axis=0) if surf_trajs_list else np.empty((0,), dtype=np.float32)
    )
    edge_latent = (
        np.stack(curv_trajs_list, axis=0) if curv_trajs_list else np.empty((0,), dtype=np.float32)
    )

    face_bboxes_np = (
        np.stack(face_bboxes_raw, axis=0) if face_bboxes_raw else np.empty((0, 6), dtype=np.float32)
    )
    edge_bboxes_np = (
        np.stack(edge_bboxes_raw, axis=0) if edge_bboxes_raw else np.empty((0, 6), dtype=np.float32)
    )

    # ── Decoded control-point trajectories (Bézier geom only; skip for uvgrid) ──
    face_ctrl_trajs = []
    for fi, bbox in enumerate(face_bboxes_raw):
        if fi >= len(surf_trajs_list):
            break
        traj = surf_trajs_list[fi]  # [T+1, surf_dim]
        if traj.shape[-1] != 64:  # not rational-Bézier control points (e.g. VAE latent)
            break
        n_steps = traj.shape[0]
        decoded = np.stack(
            [denormalize_points_with_bbox(traj[t].reshape(4, 4, 4), bbox) for t in range(n_steps)],
            axis=0,
        )  # [T+1, 4, 4, 4]
        face_ctrl_trajs.append(decoded)

    edge_ctrl_trajs = []
    for ei, bbox in enumerate(edge_bboxes_raw):
        if ei >= len(curv_trajs_list):
            break
        traj = curv_trajs_list[ei]  # [T+1, curv_dim]
        if traj.shape[-1] != 16:  # not rational-Bézier control points
            break
        n_steps = traj.shape[0]
        decoded = np.stack(
            [denormalize_points_with_bbox(traj[t].reshape(4, 4), bbox) for t in range(n_steps)],
            axis=0,
        )  # [T+1, 4, 4]
        edge_ctrl_trajs.append(decoded)

    face_ctrl_np = (
        np.stack(face_ctrl_trajs, axis=0) if face_ctrl_trajs else np.empty((0,), dtype=np.float32)
    )
    edge_ctrl_np = (
        np.stack(edge_ctrl_trajs, axis=0) if edge_ctrl_trajs else np.empty((0,), dtype=np.float32)
    )

    np.savez_compressed(
        str(out_path),
        face_latent_trajs=face_latent,
        edge_latent_trajs=edge_latent,
        face_bboxes=face_bboxes_np,
        edge_bboxes=edge_bboxes_np,
        face_ctrl_trajs=face_ctrl_np,
        edge_ctrl_trajs=edge_ctrl_np,
    )


def save_trajectories_batch(
    results: list[dict],
    rank: int,
    batch_idx: int,
    out_dir: Path,
    error_log_path: Path,
) -> None:
    """Save per-sample denoising trajectory npz files for one generation batch."""
    for sample_idx, res in enumerate(results):
        raw_surf_trajs = res.get("surf_trajs", [])
        raw_curv_trajs = res.get("curv_trajs", [])
        if raw_surf_trajs or raw_curv_trajs:
            traj_stem = sample_stem(rank, batch_idx, sample_idx, res["ids"])
            try:
                save_trajectory_npz(
                    out_dir / f"{traj_stem}.traj.npz",
                    res["ids"],
                    raw_surf_trajs,
                    raw_curv_trajs,
                )
            except Exception as traj_exc:
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{traj_stem} | traj_save_error | {traj_exc}\n")


def save_brep_sample(
    res: dict,
    stem: str,
    out_dir: Path,
    output_format: str,
    save_res: bool,
    error_log_path: Path,
    save_png: bool = True,
    save_npz: bool = False,
    ordering: str = "interleaved",
    geom_repr: str = "bezier",
    geom_vae=None,
    raw_geometry: dict | None = None,
) -> bool:
    """Build and write one valid BRep sample to disk.

    Does not mutate the supplied sample. ``raw_geometry`` preserves VAE latents
    when *res* has already been decoded to grids by the parent process.

    When ``geom_repr == "uvgrid"`` the per-primitive vectors are VAE latents:
    they are decoded to UV point grids (via *geom_vae*) and the shape is built by
    BSpline-fitting (:class:`BRepShapeBuilderUV`).

    Returns:
        True on success, False on build failure or exception.
    """
    if ordering != "interleaved":
        raise ValueError("Only interleaved ordering is supported")
    try:
        if geom_repr == "uvgrid":
            brepdata = convert_generated_to_brep_data_uv(
                res["ids"],
                res["surf_vecs"],
                res["curv_vecs"],
                geom_vae,
                ordering=ordering,
            )
            builder = BRepShapeBuilderUV(brepdata)
        else:
            brepdata = convert_generated_to_brep_data(
                res["ids"], res["surf_vecs"], res["curv_vecs"], ordering=ordering
            )
            builder = BRepShapeBuilder(brepdata)
        shape = builder.build()
        if shape is None:
            errs = builder.get_error_report()
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{stem} | build_failed | {'; '.join(errs)}\n")
            return False
        if save_npz:
            np.savez_compressed(
                str(out_dir / f"{stem}.npz"),
                face_controls=brepdata.face_controls,
                edge_controls=brepdata.edge_controls,
                outer_edge_indices=brepdata.outer_edge_indices,
                face_outer_offsets=brepdata.face_outer_offsets,
                inner_edge_indices=brepdata.inner_edge_indices,
                inner_loop_offsets=brepdata.inner_loop_offsets,
                face_inner_offsets=brepdata.face_inner_offsets,
            )
        if output_format == "step":
            write_step_file(shape, str(out_dir / f"{stem}.step"))
        elif output_format == "brep":
            breptools.Write(shape, str(out_dir / f"{stem}.brep"))
        if save_png:
            from flowar.visualization.grids import PointGridVisualizer

            if geom_repr == "uvgrid":
                face_pts = brepdata.face_controls  # already world UV grids [N,32,32,3]
                edge_pts = brepdata.edge_controls  # [N,32,3]
            else:
                face_pts = eval_rational_bezier_surfaces(brepdata.face_controls)
                edge_pts = eval_rational_bezier_curves(brepdata.edge_controls)
            PointGridVisualizer().visualize(
                face_pts,
                edge_pts,
                save_path=str(out_dir / f"{stem}.png"),
                title=stem,
            )
        if save_res:
            raw = res if raw_geometry is None else raw_geometry
            np.savez_compressed(
                str(out_dir / f"{stem}.res.npz"),
                ids=np.asarray(raw["ids"], dtype=np.int32),
                surf_vecs=np.stack(raw["surf_vecs"], axis=0)
                if raw["surf_vecs"]
                else np.empty((0,), dtype=np.float32),
                curv_vecs=np.stack(raw["curv_vecs"], axis=0)
                if raw["curv_vecs"]
                else np.empty((0,), dtype=np.float32),
            )
        return True
    except Exception as e:
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"{stem} | exception | {e}\n")
        return False
