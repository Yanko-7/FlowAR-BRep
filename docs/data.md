# Input data

Install `pythonocc-core` alongside `requirements.txt`, then convert a STEP/STP file or directory:

```bash
python preprocess.py --input /path/to/steps --output data/npz --workers 8
```

Each solid produces `<input_stem>__solid000.npz`, with coordinates normalized to `[-1, 1]`. Complex surfaces may use cubic approximation. Conversion results and rejection reasons are recorded in `data/npz/preprocessing.json`.

`data/splits.json` assigns accepted samples to train/val/test at 90/5/5. Source stems sharing the first 33 characters stay together. Small inputs may have empty validation or test splits.

Use `--split-file /path/to/source_splits.json` to preserve an existing partition, keeping it separate from the generated split file. Entries are filename prefixes:

```json
{"train": ["shape_a"], "val": ["shape_b"], "test": ["shape_c"]}
```

Run `python preprocess.py --help` for conversion limits, timeout, and overwrite options.

## NPZ format

| Field | Shape / meaning |
| --- | --- |
| `face_controls` | `[F, 4, 4, 4]`: cubic rational surface controls `(x, y, z, w)` |
| `edge_controls` | `[E, 4, 4]`: cubic rational curve controls |
| `outer_edge_indices` | Concatenated outer-loop edge indices |
| `face_outer_offsets` | Per-face outer-loop offsets, length `F + 1` |
| `inner_edge_indices` | Concatenated inner-loop edge indices |
| `inner_loop_offsets` | Offsets into inner-loop edges |
| `face_inner_offsets` | Per-face offsets into inner loops |

Experimental UV-grid support uses the same NPZ format with surface and edge VAEs.

## Conditional inputs

The matching ID is `Path(npz_path).stem[:33]`, including the solid suffix when it falls within those 33 characters:

- Point clouds: PLY files under `cond_pc_extra_dir`.
- Images: PNG files under `cond_image_dir`.
- Text: ID-to-caption entries in `cond_caption_json`.

See [sequence_format.md](sequence_format.md) for the token representation.
