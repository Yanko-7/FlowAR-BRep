from dataclasses import dataclass

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_C1
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer


def points2surfs(pointss):
    """
    Args:
        points: List[32,32,3] array of 3D points
    """
    constructed_faces = []
    for points in pointss:
        num_u = points.shape[0]
        num_v = points.shape[1]
        uv_points_array = TColgp_Array2OfPnt(1, num_u, 1, num_v)
        for u_index in range(num_u):
            for v_index in range(num_v):
                p = gp_Pnt(
                    float(points[u_index, v_index, 0]),
                    float(points[u_index, v_index, 1]),
                    float(points[u_index, v_index, 2]),
                )
                uv_points_array.SetValue(u_index + 1, v_index + 1, p)

        approx_face = GeomAPI_PointsToBSplineSurface(
            uv_points_array, 3, 3, GeomAbs_C1, GeomAbs_C1, 5e-2
        ).Surface()
        constructed_faces.append(approx_face)
    return constructed_faces


def surface2tri(face):
    """
    Args:
        face: Geom_Surface
    Returns:
        vertices: List of vertices
        faces: List of faces (triangles)
    """
    # Create a BRep face from the Geom_Surface
    brep_face = BRepBuilderAPI_MakeFace(face, 1e-6).Face()

    # Fix the face to ensure it's valid
    shape_fix_face = ShapeFix_Face(brep_face)
    shape_fix_face.Perform()
    fixed_face = shape_fix_face.Face()

    # Triangulate the face

    mesh = BRepMesh_IncrementalMesh(fixed_face, 0.1)
    mesh.Perform()

    # Extract vertices and faces
    vertices = []
    faces = []

    topo_explorer = TopologyExplorer(fixed_face)
    for vertex in topo_explorer.vertices():
        pnt = vertex.Pnt()
        vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

    for face in topo_explorer.faces():
        wire_explorer = WireExplorer(face)
        for edge in wire_explorer.edges():
            edge_vertices = []
            edge_explorer = TopologyExplorer(edge)
            for vertex in edge_explorer.vertices():
                index = topo_explorer.vertices().index(vertex)
                edge_vertices.append(index)
            if len(edge_vertices) >= 3:
                for i in range(1, len(edge_vertices) - 1):
                    faces.append([edge_vertices[0], edge_vertices[i], edge_vertices[i + 1]])

    return vertices, faces


def plot_surfaces(surfaces):
    """
    Plot triangular meshes for multiple surfaces using Plotly

    Args:
        surfaces: List of Geom_Surface objects
    """

    import plotly.graph_objects as go

    fig = go.Figure()

    for i, surface in enumerate(surfaces):
        vertices, faces = surface2tri(surface)

        if not vertices or not faces:
            continue

        # Extract vertex coordinates
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        # Extract triangle indices
        i_idx = [f[0] for f in faces]
        j_idx = [f[1] for f in faces]
        k_idx = [f[2] for f in faces]

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                opacity=0.8,
                name=f"Surface {i}",
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        title="B-Spline Surfaces",
    )

    fig.show()


def plot_mesh(vertices, faces):
    """
    Plot a single triangular mesh using Plotly

    Args:
        vertices: List of [x, y, z] coordinates
        faces: List of [i, j, k] triangle indices
    """

    import plotly.graph_objects as go

    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]

    i_idx = [f[0] for f in faces]
    j_idx = [f[1] for f in faces]
    k_idx = [f[2] for f in faces]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                opacity=0.8,
                colorscale="Viridis",
                intensity=z,  # Color by z value
            )
        ]
    )

    fig.update_layout(scene=dict(aspectmode="data"), title="Mesh Visualization")

    fig.write_html("mesh_visualization.html")


def save_boxes_image(boxes, save_path="boxes_output1.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_x, all_y, all_z = [], [], []

    for box in boxes:
        x1, y1, z1, x2, y2, z2 = box
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])
        all_z.extend([z1, z2])

        # Define the eight vertices
        x = [x1, x2, x2, x1, x1, x2, x2, x1]
        y = [y1, y1, y2, y2, y1, y1, y2, y2]
        z = [z1, z1, z1, z1, z2, z2, z2, z2]

        # Twelve edges
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        for line in lines:
            ax.plot3D(
                [x[line[0]], x[line[1]]],
                [y[line[0]], y[line[1]]],
                [z[line[0]], z[line[1]]],
                "r-",  # Red lines
            )

    # Automatically adjust the view limits
    if all_x:
        max_range = (
            np.array(
                [
                    max(all_x) - min(all_x),
                    max(all_y) - min(all_y),
                    max(all_z) - min(all_z),
                ]
            ).max()
            / 2.0
        )
        mid_x = (max(all_x) + min(all_x)) * 0.5
        mid_y = (max(all_y) + min(all_y)) * 0.5
        mid_z = (max(all_z) + min(all_z)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Save the image
    plt.savefig(save_path, dpi=150)
    print(f"Image saved to: {save_path}")
    plt.close()  # Release memory


def plot_merged_batch(filename="merged_viz.html", face_points=None, edge_points=None, bboxes=None):
    import plotly.graph_objects as go

    data_traces = []

    # --- 1. Process the point cloud ---
    points_list, ids_list = [], []
    if face_points is not None:
        B = face_points.shape[0]
        points_list.append(face_points.reshape(-1, 3))
        ids_list.append(np.repeat(np.arange(B), face_points.size // (B * 3)))

    if edge_points is not None:
        B = edge_points.shape[0]
        points_list.append(edge_points.reshape(-1, 3))
        ids_list.append(np.repeat(np.arange(B), edge_points.size // (B * 3)))

    if points_list:
        xyz = np.concatenate(points_list, axis=0)
        batch_ids = np.concatenate(ids_list, axis=0)
        data_traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=batch_ids,
                    colorscale="Jet",
                    opacity=0.8,
                    colorbar=dict(title="Batch ID"),
                ),
                text=batch_ids,
                hovertemplate="Batch: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}",
            )
        )

    # --- 2. Process bounding boxes ---
    if bboxes is not None:
        bx, by, bz = [], [], []
        # Continuous cube traversal indices: bottom loop -> vertical edge -> top loop -> vertical edge...
        ix = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
        for b in bboxes.reshape(-1, 6):
            xm, ym, zm, xM, yM, zM = b
            c = np.array(
                [
                    [xm, ym, zm],
                    [xM, ym, zm],
                    [xM, yM, zm],
                    [xm, yM, zm],  # 0-3 bottom
                    [xm, ym, zM],
                    [xM, ym, zM],
                    [xM, yM, zM],
                    [xm, yM, zM],
                ]
            )  # 4-7 top
            bx.extend(c[ix, 0])
            by.extend(c[ix, 1])
            bz.extend(c[ix, 2])
            bx += [None]
            by += [None]
            bz += [None]  # Break the line segment

        data_traces.append(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                line=dict(color="red", width=3),
                name="BBox",
                hoverinfo="none",
            )
        )

    # --- 3. Plot ---
    if not data_traces:
        return

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.write_html(filename)
    print(f"✅ Save: {filename}")


def plot_point_cloud_batch(pc_data, filename="batch_pointcloud_viz.html"):
    """
    Plot a batch of point clouds with shape (N, 3, 32, 32).
    The generated HTML includes a slider for switching between samples.

    Args:
        pc_data: torch.Tensor, shape (N, 3, 32, 32)
        filename: Output HTML filename
    """

    import plotly.graph_objects as go

    points_batch = pc_data

    # Ensure four dimensions (N, C, H, W); add a batch dimension for (C, H, W) input
    if points_batch.ndim == 3:
        points_batch = points_batch[np.newaxis, ...]

    N = points_batch.shape[0]  # Batch Size

    # --- 2. Create the figure ---
    fig = go.Figure()

    # --- 3. Create a trace for each sample ---
    for i in range(N):
        # Extract sample i: (3, 32, 32) -> reshape -> (1024, 3)
        # points_batch[i] shape is (3, 32, 32)
        points = points_batch[i].reshape(3, -1).T

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Only sample 0 is visible by default (visible=True); all others are set to False
        is_visible = i == 0

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.8),
                visible=is_visible,
                name=f"Sample {i}",
            )
        )

    # --- 4. Create the slider ---
    steps = []
    for i in range(N):
        # Define the behavior of each slider step
        step = dict(
            method="update",
            args=[
                {"visible": [False] * N},  # First, hide all traces
                {"title": f"Batch Visualization - Sample {i}"},  # Update the title
            ],
            label=f"{i}",
        )
        # Then, make the trace at index i visible
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Sample Index: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    # --- 5. Configure the layout ---
    fig.update_layout(
        sliders=sliders,
        title="Batch Visualization - Sample 0",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    # --- 6. Save ---
    fig.write_html(filename)
    print(f"✅ Point cloud batch ({N} samples) saved to: {filename}")


def plot_point_cloud(pc_data, filename="pointcloud_viz.html"):
    """
    Plot a point cloud with shape (1, 3, 32, 32) or (3, 32, 32).

    Args:
        pc_data: torch.Tensor or np.array
        filename: Output HTML filename
    """

    import plotly.graph_objects as go

    points = pc_data.squeeze().reshape(3, -1).T

    # --- 2. Extract coordinates ---
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # --- 3. Plot ---
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=4,  # Point size
                    color=z,  # Color by Z-axis height to show depth
                    colorscale="Viridis",  # Color scheme
                    opacity=0.8,
                ),
            )
        ]
    )

    # Keep equal XYZ scaling to prevent distortion
    fig.update_layout(
        title=f"Point Cloud Visualization ({points.shape[0]} points)",
        scene=dict(aspectmode="data"),
    )

    # --- 4. Save ---
    fig.write_html(filename)
    print(f"✅ Point cloud saved to: {filename} (Download and open in a browser)")


"""
A simple class to visualize point grids using matplotlib.

We could try to avoid explicit dependency on
pytorch and Open Cascade in here
"""


@dataclass
class BRepData:
    face_points: np.ndarray
    edge_points: np.ndarray
    outer_edge_indices: np.ndarray
    face_outer_offsets: np.ndarray
    inner_edge_indices: np.ndarray
    inner_loop_offsets: np.ndarray
    face_inner_offsets: np.ndarray


class PointGridVisualizer:
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def visualize(
        self,
        face_points,
        edge_points,
        save_path: str = "brep_visualization.png",
        title="B-Rep Point Grid",
    ):
        fig = plt.figure(figsize=self.figsize)
        try:
            ax = fig.add_subplot(111, projection="3d")

            all_points = []

            # 1. Render faces
            num_faces = len(face_points)
            colors = cm.tab20(np.linspace(0, 1, max(num_faces, 1)))

            for i, face_grid in enumerate(face_points):
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

            # 2. Render edges
            for edge_curve in edge_points:
                pts = edge_curve.reshape(-1, 3)
                if len(pts) == 0:
                    continue

                all_points.append(pts)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=2.0, alpha=0.9)

            # 3. Use equal scaling for all axes
            if all_points:
                concat_pts = np.vstack(all_points)
                min_pt, max_pt = concat_pts.min(axis=0), concat_pts.max(axis=0)
                center = (max_pt + min_pt) / 2
                max_range = max(float((max_pt - min_pt).max()) / 2.0, 1e-3)

                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(center[2] - max_range, center[2] + max_range)
                ax.set_box_aspect([1, 1, 1])

            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.tight_layout()

            # Save a high-resolution image and close the figure
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
