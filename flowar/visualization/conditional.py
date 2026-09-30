"""Side-by-side condition and generated geometry previews."""

import textwrap

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from flowar.geometry.bezier import (
    eval_rational_bezier_curves,
    eval_rational_bezier_surfaces,
)


def _render_brep_axes(ax, brep_data, surf_res: int = 20, curve_res: int = 50) -> list:
    """Render BRep surfaces and curves onto *ax*. Returns list of sampled point arrays."""
    all_pts = []
    fc = brep_data.face_controls
    ec = brep_data.edge_controls
    n_faces, n_edges = len(fc), len(ec)
    palette = cm.tab20(np.linspace(0, 1, max(n_faces, 1)))

    if n_faces > 0:
        try:
            surfaces = eval_rational_bezier_surfaces(fc, res=surf_res)
            for i, S in enumerate(surfaces):
                if not np.all(np.isfinite(S)):
                    continue
                ax.plot_surface(
                    S[..., 0],
                    S[..., 1],
                    S[..., 2],
                    color=palette[i % len(palette)],
                    alpha=0.40,
                    linewidth=0,
                    antialiased=True,
                )
                all_pts.append(S.reshape(-1, 3))
        except Exception as exc:
            print(f"    [warn] Surface rendering failed: {exc}")

    if n_edges > 0:
        try:
            curves = eval_rational_bezier_curves(ec, res=curve_res)
            for C in curves:
                if not np.all(np.isfinite(C)):
                    continue
                ax.plot(C[:, 0], C[:, 1], C[:, 2], "k-", linewidth=1.2, alpha=0.85)
                all_pts.append(C)
        except Exception as exc:
            print(f"    [warn] Curve rendering failed: {exc}")

    ax.set_title(f"Generated BRep  ({n_faces} faces, {n_edges} edges)", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    return all_pts


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    """Force equal aspect ratio on a 3D Axes given a point array [N, 3]."""
    if len(pts) == 0:
        return
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    center = (mn + mx) / 2.0
    r = max((mx - mn).max() / 2.0, 1e-3)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def save_conditional_preview(
    condition,
    modality: str,
    brep_data,
    output_path: str,
    stem: str = "",
    surf_res: int = 20,
    curve_res: int = 50,
) -> None:
    """Render one condition beside its generated B-Rep, closing the figure on failure."""
    if modality not in ("pointcloud", "dino_image", "clip_text"):
        raise ValueError(f"Unknown preview modality: {modality}")
    fig = plt.figure(figsize=(14, 6))
    try:
        fig.suptitle(stem, fontsize=11)
        projection = "3d" if modality == "pointcloud" else None
        ax = fig.add_subplot(1, 2, 1, projection=projection)
        if modality == "pointcloud":
            ax.scatter(*condition.T, s=3.0, alpha=0.8, c="steelblue", edgecolors="none")
            ax.set(title="Input Point Cloud", xlabel="X", ylabel="Y", zlabel="Z")
            ax.set_box_aspect([1, 1, 1])
            _set_axes_equal(ax, condition)
        elif modality == "dino_image":
            ax.imshow(condition)
            ax.set_title("Input Image", fontsize=10)
            ax.axis("off")
        else:
            ax.text(
                0.5,
                0.5,
                textwrap.fill(condition, width=40),
                ha="center",
                va="center",
                fontsize=11,
                fontfamily="monospace",
                linespacing=1.4,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
            )
            ax.set_title("Text Caption", fontsize=10)
            ax.axis("off")

        generated = fig.add_subplot(1, 2, 2, projection="3d")
        points = _render_brep_axes(generated, brep_data, surf_res, curve_res)
        if points:
            _set_axes_equal(generated, np.vstack(points))
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"    Saved: {output_path}")
