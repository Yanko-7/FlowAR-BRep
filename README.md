# FlowAR-BRep

**Interleaved Autoregression and Flow Matching for Single-Stage B-Rep Generation**

Yongkang Qi<sup>1</sup>, Sunhong Qiu<sup>1</sup>, Hao Pan<sup>2</sup>, Haichuan Song<sup>1</sup>

<sup>1</sup>East China Normal University, <sup>2</sup>Tsinghua University

*SIGGRAPH Asia 2026 Conference Papers*

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-0055A4)](https://doi.org/10.1145/3829340.3842332)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[![FlowAR-BRep teaser: a gallery of CAD boundary representations](assets/teaser.png)](assets/teaser.pdf)

FlowAR-BRep generates CAD boundary representations in a single stage. A transformer predicts topology autoregressively as discrete tokens, and each surface or curve is a rational cubic Bézier, produced by flow matching as a continuous vector in the same sequence. This repository is the reference implementation, covering unconditional generation and generation conditioned on point clouds, images, or text.

## Installation

Use Python 3.12 and run all commands from the repository root.

```bash
pip install -r requirements.txt
```

Install these platform-specific packages separately:

- `pythonocc-core` for STEP preprocessing and CAD export
- PyTorch3D for point-cloud conditioning
- FlashAttention (optional)

## Data preparation

Convert a directory of STEP files into training samples:

```bash
python preprocess.py --input /path/to/steps --output data/npz --workers 8
```

This writes NPZ samples and `data/splits.json`, matching the paths in [`configs/train.yaml`](configs/train.yaml). See [docs/data.md](docs/data.md) for existing splits, conversion options, and the NPZ layout.

## Training

```bash
python train.py --config configs/train.yaml
```

Conditional configs are provided for [point clouds](configs/train_pointcloud.yaml), [images](configs/train_image.yaml), and [text](configs/train_text.yaml). An experimental [UV-grid config](configs/train_uvgrid.yaml) is also included and requires surface and edge VAE checkpoints.

<details>
<summary>Multi-GPU training</summary>

Set `--nproc_per_node` to the number of GPUs.

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/train.yaml
```

</details>

## Generation

Generate STEP files in `outputs/steps` from a trained checkpoint:

```bash
python generate.py --config configs/generate.yaml --checkpoint weights/model.pt
```

Sampling and export options live in [`configs/generate.yaml`](configs/generate.yaml). Add `--save-png` for previews or `--save-npz` for reconstructed geometry arrays.

<details>
<summary>Geometry checks and backtracking</summary>

For Bézier checkpoints, enable geometry verification with backtracking:

```bash
python generate.py --config configs/generate.yaml --checkpoint weights/model.pt \
  --use-geom-rejection
```

Add `--geom-check-intersect` to also check curve intersections.

</details>

<details>
<summary>Conditional generation</summary>

Use a checkpoint trained for the matching input modality, together with its encoder assets. Conditional previews support Bézier checkpoints.

```bash
# Point cloud
python generate_conditioned.py --checkpoint weights/conditional.pt --pc-input data/input.ply

# Image
python generate_conditioned.py --checkpoint weights/conditional.pt --image-input data/input.png

# Text
python generate_conditioned.py --checkpoint weights/conditional.pt --caption-json data/captions.json
```

</details>

<details>
<summary>Multi-GPU generation</summary>

Set `--nproc_per_node` to the number of GPUs. Each rank writes to its own output directory.

```bash
torchrun --standalone --nproc_per_node=4 generate.py --config configs/generate.yaml --checkpoint weights/model.pt
```

</details>

## Documentation

- [Input data](docs/data.md): STEP conversion, splits, NPZ fields, and conditional inputs
- [Sequence format](docs/sequence_format.md): how topology tokens and geometry slots are interleaved
- [GRPO extension](extensions/grpo/README.md): experimental policy optimization, outside the paper's reference path

## Citation

```bibtex
@inproceedings{qi2026flowarbrep,
  title     = {{FlowAR-BRep}: Interleaved Autoregression and Flow Matching for Single-Stage {B-Rep} Generation},
  author    = {Qi, Yongkang and Qiu, Sunhong and Pan, Hao and Song, Haichuan},
  booktitle = {Proceedings of the SIGGRAPH Asia 2026 Conference Papers},
  series    = {SA Conference Papers '26},
  year      = {2026},
  pages     = {1--12},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Kuala Lumpur, Malaysia},
  isbn      = {979-8-4007-2842-6},
  doi       = {10.1145/3829340.3842332},
  url       = {https://doi.org/10.1145/3829340.3842332}
}
```

## License

Released under the [MIT License](LICENSE).
