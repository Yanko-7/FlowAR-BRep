from typing import List, Optional

import numpy as np

from flowar.data.sequence import (
    BRepTokenType,
    denormalize_points_with_bbox,
    tokens_to_bbox,
)


def _plot_failed_state(
    seq: List[int],
    surf_vecs: List[np.ndarray],
    curv_vecs: List[np.ndarray],
    save_path: str,
    bad_edge_ctrl: Optional[np.ndarray] = None,
    conflict_edge_ctrl: Optional[np.ndarray] = None,
) -> None:
    """Decode and visualise all committed faces and edges at a failed generation state.

    Robustly handles incomplete / truncated token sequences by stopping on any
    indexing error rather than raising.  *bad_edge_ctrl* (world-space ``(4, 4)``
    control points) is rendered in red to highlight the offending edge.
    *conflict_edge_ctrl* (the existing committed edge it intersects) is rendered
    in orange.
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        from flowar.geometry.bezier import (
            eval_rational_bezier_curves,
            eval_rational_bezier_surfaces,
        )
    except Exception:
        return

    face_ctrls: List[np.ndarray] = []
    edge_ctrls: List[np.ndarray] = []
    surf_idx = 0
    curv_idx = 0
    n = len(seq)
    idx = 0

    try:
        while idx < n:
            tok = seq[idx]
            if tok == BRepTokenType.EOS:
                break
            if tok == BRepTokenType.FACE_START:
                if idx + 7 <= n and surf_idx < len(surf_vecs):
                    bbox = tokens_to_bbox(seq[idx + 1 : idx + 7])
                    face_ctrls.append(
                        denormalize_points_with_bbox(surf_vecs[surf_idx].reshape(4, 4, 4), bbox)
                    )
                    surf_idx += 1
                idx += 8
            elif tok == BRepTokenType.EDGE_NEW:
                if idx + 8 <= n and curv_idx < len(curv_vecs):
                    bbox = tokens_to_bbox(seq[idx + 2 : idx + 8])
                    edge_ctrls.append(
                        denormalize_points_with_bbox(curv_vecs[curv_idx].reshape(4, 4), bbox)
                    )
                    curv_idx += 1
                idx += 9
            else:
                idx += 1
    except Exception:
        pass

    if not face_ctrls and not edge_ctrls and bad_edge_ctrl is None and conflict_edge_ctrl is None:
        return

    face_pts = (
        eval_rational_bezier_surfaces(np.asarray(face_ctrls, dtype=np.float32))
        if face_ctrls
        else []
    )
    edge_pts = (
        eval_rational_bezier_curves(np.asarray(edge_ctrls, dtype=np.float32)) if edge_ctrls else []
    )

    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_points: List[np.ndarray] = []

    colors = cm.tab20(np.linspace(0, 1, max(len(face_pts), 1)))
    for i, face_grid in enumerate(face_pts):
        pts = face_grid.reshape(-1, 3)
        if len(pts) == 0:
            continue
        all_points.append(pts)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=[colors[i % 20]],
            s=5,
            alpha=0.6,
            edgecolors="none",
        )

    for edge_curve in edge_pts:
        pts = edge_curve.reshape(-1, 3)
        if len(pts) == 0:
            continue
        all_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=2.0, alpha=0.9)

    if bad_edge_ctrl is not None:
        try:
            bad_pts = eval_rational_bezier_curves(np.asarray([bad_edge_ctrl], dtype=np.float32))[0]
            all_points.append(bad_pts)
            ax.plot(
                bad_pts[:, 0],
                bad_pts[:, 1],
                bad_pts[:, 2],
                color="red",
                linewidth=3.5,
                alpha=1.0,
                label="failed edge",
                zorder=5,
            )
            for pt in (bad_pts[0], bad_pts[-1]):
                ax.scatter(
                    pt[0],
                    pt[1],
                    pt[2],
                    color="red",
                    s=80,
                    zorder=6,
                    edgecolors="darkred",
                )
        except Exception:
            pass

    if conflict_edge_ctrl is not None:
        try:
            conf_pts = eval_rational_bezier_curves(
                np.asarray([conflict_edge_ctrl], dtype=np.float32)
            )[0]
            all_points.append(conf_pts)
            ax.plot(
                conf_pts[:, 0],
                conf_pts[:, 1],
                conf_pts[:, 2],
                color="orange",
                linewidth=3.5,
                alpha=1.0,
                label="conflict edge",
                zorder=5,
            )
            for pt in (conf_pts[0], conf_pts[-1]):
                ax.scatter(
                    pt[0],
                    pt[1],
                    pt[2],
                    color="orange",
                    s=80,
                    zorder=6,
                    edgecolors="darkorange",
                )
        except Exception:
            pass

    if bad_edge_ctrl is not None or conflict_edge_ctrl is not None:
        ax.legend()

    if all_points:
        concat_pts = np.vstack(all_points)
        min_pt, max_pt = concat_pts.min(axis=0), concat_pts.max(axis=0)
        center = (max_pt + min_pt) / 2
        max_range = max((max_pt - min_pt).max() / 2.0, 1e-6)
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_box_aspect([1, 1, 1])

    ax.set_title(f"Failed state — {len(face_ctrls)} face(s), {len(edge_ctrls)} edge(s)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
