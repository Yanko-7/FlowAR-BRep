from itertools import combinations

import numpy as np


def gen_random_rotation_matrix():
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def build_face_adjacency(
    outer_edges, face_outer_offsets, inner_edges, inner_loop_offsets, face_inner_offsets
):
    num_faces = len(face_outer_offsets) - 1
    if num_faces <= 0:
        return []

    num_edges = int(max(np.max(outer_edges, initial=-1), np.max(inner_edges, initial=-1))) + 1

    edge_to_faces = [[] for _ in range(num_edges)]

    for f_id in range(num_faces):
        e_ids = list(outer_edges[face_outer_offsets[f_id] : face_outer_offsets[f_id + 1]])

        for l_idx in range(face_inner_offsets[f_id], face_inner_offsets[f_id + 1]):
            e_ids.extend(inner_edges[inner_loop_offsets[l_idx] : inner_loop_offsets[l_idx + 1]])

        for e_id in set(e_ids):
            edge_to_faces[e_id].append(f_id)

    adj = [set() for _ in range(num_faces)]
    for faces in edge_to_faces:
        for u, v in combinations(faces, 2):
            adj[u].add(v)
            adj[v].add(u)

    return adj


def _bfs_reorder_faces(
    face_points,
    outer_edge_indices,
    face_outer_offsets,
    inner_edge_indices,
    inner_loop_offsets,
    face_inner_offsets,
    random_start_num=0,
):
    num_faces = len(face_points)
    if num_faces <= 1:
        return (
            face_points,
            outer_edge_indices,
            face_outer_offsets,
            inner_edge_indices,
            inner_loop_offsets,
            face_inner_offsets,
        )

    centroids = face_points.mean(axis=(1, 2))

    adj = build_face_adjacency(
        outer_edge_indices,
        face_outer_offsets,
        inner_edge_indices,
        inner_loop_offsets,
        face_inner_offsets,
    )

    sorted_indices = np.lexsort((centroids[:, 0], centroids[:, 1], centroids[:, 2]))

    def greedy_chain(layer_faces, entry_face=None, processed=None):
        """Order faces within a BFS layer, preferring adjacent faces at each step.

        Primary key: most already-processed neighbors (maximises EDGE_REF).
        Secondary key: centroid axis order (tie-break).
        If entry_face is given, the chain starts from a neighbor of entry_face
        inside the layer; otherwise falls back to centroid-sorted minimum.
        processed is updated in-place as each face is picked.
        """
        remaining = set(layer_faces)
        if not remaining:
            return []

        if processed is None:
            processed = set()

        def score(n):
            return (-sum(1 for nb in adj[n] if nb in processed), tuple(centroids[n]))

        if entry_face is not None:
            candidates = [f for f in adj[entry_face] if f in remaining]
            if candidates:
                candidates.sort(key=score)
                current = candidates[0]
            else:
                current = min(remaining, key=score)
        else:
            current = min(remaining, key=lambda n: tuple(centroids[n]))

        result = []
        while remaining:
            remaining.discard(current)
            result.append(current)
            processed.add(current)

            adj_remaining = [f for f in adj[current] if f in remaining]
            if adj_remaining:
                adj_remaining.sort(key=score)
                current = adj_remaining[0]
            elif remaining:
                current = min(remaining, key=score)

        return result

    def bfs_component(start_faces, visited):
        """BFS a connected component layer by layer; within each layer use greedy_chain.

        start_faces may be a single face index or a list of face indices (multi-source).
        """
        if not isinstance(start_faces, (list, np.ndarray)):
            start_faces = [start_faces]
        for f in start_faces:
            visited[f] = True
        layers = [list(start_faces)]

        while True:
            next_layer = []
            seen_next = set()
            for f in layers[-1]:
                for nb in adj[f]:
                    if not visited[nb] and nb not in seen_next:
                        seen_next.add(nb)
                        visited[nb] = True
                        next_layer.append(nb)
            if not next_layer:
                break
            layers.append(next_layer)

        comp_order = []
        processed = set()
        for layer in layers:
            entry = comp_order[-1] if comp_order else None
            comp_order.extend(greedy_chain(layer, entry_face=entry, processed=processed))

        return comp_order

    visited = [False] * num_faces
    if random_start_num > 0:
        k = min(random_start_num, num_faces)
        seeds = np.random.choice(num_faces, size=k, replace=False).tolist()
        order = bfs_component(seeds, visited)
        for i in sorted_indices:
            i = int(i)
            if not visited[i]:
                order.extend(bfs_component(i, visited))
    else:
        start_order = sorted_indices
        order = bfs_component(int(start_order[0]), visited)
        for i in start_order:
            i = int(i)
            if not visited[i]:
                order.extend(bfs_component(i, visited))

    order = np.array(order, dtype=np.int64)
    new_face_points = face_points[order]

    new_outer_edges = []
    new_face_outer_offsets = [0]
    for i in order:
        s, e = face_outer_offsets[i], face_outer_offsets[i + 1]
        new_outer_edges.extend(outer_edge_indices[s:e])
        new_face_outer_offsets.append(len(new_outer_edges))

    new_inner_edges = []
    new_inner_loop_offsets = [0]
    new_face_inner_offsets = [0]
    for i in order:
        for j in range(face_inner_offsets[i], face_inner_offsets[i + 1]):
            s, e = inner_loop_offsets[j], inner_loop_offsets[j + 1]
            new_inner_edges.extend(inner_edge_indices[s:e])
            new_inner_loop_offsets.append(len(new_inner_edges))
        new_face_inner_offsets.append(len(new_inner_loop_offsets) - 1)

    return (
        new_face_points,
        np.array(new_outer_edges, dtype=np.int32),
        np.array(new_face_outer_offsets, dtype=np.int32),
        np.array(new_inner_edges, dtype=np.int32),
        np.array(new_inner_loop_offsets, dtype=np.int32),
        np.array(new_face_inner_offsets, dtype=np.int32),
    )
