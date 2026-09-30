# Sequence format

FlowAR-BRep interleaves discrete topology tokens with continuous geometry. A surface or curve vector occupies one sequence position and is embedded directly; it is not a sequence of quantized coordinate tokens.

```text
BOS [optional COMPLEXITY_L1/L2/L3]
  FACE_START bbox(6) SURFACE_GEOM
    LOOP_START
      EDGE_NEW edge_id bbox(6) CURVE_GEOM
      ...
      EDGE_REF edge_id
    LOOP_END
    [additional inner loops]
  FACE_END
  [additional faces]
EOS
```

The first loop of each face is its outer loop. A new edge supplies geometry; a reference reuses that edge and releases its token ID for a later new edge. Faces use BFS ordering. Position IDs increase by one for every token or geometry slot.

## Vocabulary and bounding boxes

| IDs | Meaning |
| --- | --- |
| 0–4 | PAD, BOS, EOS, FACE_START, FACE_END |
| 7–8 | SURFACE_GEOM, CURVE_GEOM placeholders |
| 9–12 | LOOP_START, LOOP_END, EDGE_NEW, EDGE_REF |
| 17–19 | Complexity levels: 1–15, 16–30, 31+ faces |
| 32–1055 | 1024 coordinate bins over `[-1, 1]` |
| 1056–2035 | 980 available edge-ID tokens |

Other control IDs are reserved or unused. In particular, the historical BBOX_START/BBOX_END enum entries are not emitted.

Each bounding box is serialized as **z_min, z_max, y_min, y_max, x_min, x_max**. The geometry utilities accept and return boxes in **x_min, y_min, z_min, x_max, y_max, z_max** order; `bbox_to_tokens` and `tokens_to_bbox` perform the conversion.

## Array representation

`BRepSequence` stores text IDs and their positions separately from surface and curve arrays. Geometry placeholders describe the conceptual mixed sequence; they are not inserted into `text_ids` during serialization.

- Bézier surfaces: `[F, 4, 4, 4]`; curves: `[E, 4, 4]`. The final axis is `(x, y, z, w)`. Model inputs flatten these to 64 and 16 values per primitive.
- Experimental UV-grid surfaces: `[F, 32, 32, 3]`; curves: `[E, 32, 3]`. Frozen VAEs encode these to 48 and 12 latent values per primitive.
- Conditional inputs occupy 257 leading positions and shift text/geometry indexes accordingly. Conditional training omits the complexity token.

Use `arrays_to_sequence` for serialization and `sequence_to_arrays` for Bézier reconstruction. NPZ fields are listed in [data.md](data.md).

```python
from flowar.data.sequence import coord_to_token, token_to_coord, edge_id_to_token

assert coord_to_token(0.5) == 799
assert edge_id_to_token(42) == 1098
recovered = token_to_coord(799)  # approximately 0.4995
```
