"""Generate and preview B-Reps from point clouds, images or text captions."""

import argparse
import json
from pathlib import Path

import torch

from flowar.checkpoint import load_model_from_checkpoint
from flowar.generation.conditioning import ConditionEncoder, ConditionInput
from flowar.generation.engine import Engine
from flowar.generation.results import collect_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional BRep generation from point cloud or image input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a conditional model checkpoint (.pt)",
    )

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "--pc-input",
        help="Path to a PLY/OBJ file or directory of PLY files (pointcloud mode)",
    )
    input_grp.add_argument(
        "--image-input",
        help="Path to a PNG/JPG file or directory of images (dino_image mode)",
    )
    input_grp.add_argument(
        "--caption-json",
        help="Path to caption JSON {sample_id: caption_text} for text-conditioned batch inference",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/cond_gen",
        help="Directory for PNG visualizations and STEP files",
    )
    parser.add_argument(
        "--num-samples", type=int, default=4, help="BRep samples to generate per input"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max autoregressive tokens per sample",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling (None = unrestricted)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pc-points",
        type=int,
        default=4096,
        help="Number of points sampled from each input shape (pointcloud mode)",
    )
    parser.add_argument(
        "--surf-res",
        type=int,
        default=20,
        help="Bezier surface evaluation resolution (per axis)",
    )
    parser.add_argument(
        "--curve-res", type=int, default=50, help="Bezier curve evaluation resolution"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Skip STEP file export (visualization only)",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Path to split JSON with train/val/test lists (used with --caption-json)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use from --split-file (default: test)",
    )
    parser.add_argument(
        "--max-captions",
        type=int,
        default=None,
        help="Limit number of captions to process (default: all)",
    )
    return parser.parse_args()


def input_items(args):
    if args.caption_json:
        with open(args.caption_json, encoding="utf-8") as stream:
            captions = json.load(stream)
        if not isinstance(captions, dict) or not all(isinstance(v, str) for v in captions.values()):
            raise ValueError("Caption JSON must map sample IDs to strings")
        keys = list(captions)
        if args.split_file:
            with open(args.split_file, encoding="utf-8") as stream:
                splits = json.load(stream)
            if args.split not in splits:
                raise ValueError(f"Unknown split: {args.split}")
            keys = [key for key in splits[args.split] if key in captions]
        if args.max_captions is not None:
            keys = keys[: args.max_captions]
        items = [ConditionInput(str(key), "clip_text", captions[key]) for key in keys]
    else:
        path = Path(args.pc_input or args.image_input)
        modality = "pointcloud" if args.pc_input else "dino_image"
        extensions = {".ply", ".obj"} if args.pc_input else {".png", ".jpg", ".jpeg", ".webp"}
        files = (
            sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)
            if path.is_dir()
            else [path]
        )
        if any(not p.is_file() for p in files):
            raise FileNotFoundError(path)
        items = [ConditionInput(p.stem, modality, p) for p in files]
    if not items:
        raise ValueError("No conditional inputs matched the requested source/split")
    return items


def save_preview(sample, item, preview, stem, args):
    # CAD and plotting imports are needed only when writing a generated sample.
    from OCC.Extend.DataExchange import write_step_file

    from flowar.geometry.builder import BRepShapeBuilder, convert_generated_to_brep_data
    from flowar.visualization.conditional import save_conditional_preview

    data = convert_generated_to_brep_data(**sample.geometry())
    saved = False
    if not args.no_step:
        builder = BRepShapeBuilder(data)
        shape = builder.build()
        if shape is None:
            print(f"[{stem}] CAD construction failed: {'; '.join(builder.get_error_report())}")
        else:
            write_step_file(shape, str(Path(args.output_dir) / f"{stem}.step"))
            saved = True
    save_conditional_preview(
        preview,
        item.modality,
        data,
        output_path=str(Path(args.output_dir) / f"{stem}.png"),
        stem=stem,
        surf_res=args.surf_res,
        curve_res=args.curve_res,
    )
    return saved


def main():
    args = parse_args()
    if args.num_samples < 1 or args.max_tokens < 1:
        raise ValueError("Sample count and token budget must be positive")
    items = input_items(args)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=args.device)
    if checkpoint.get("args", {}).get("geom_repr", "bezier") != "bezier":
        raise ValueError("Conditional preview currently supports Bézier checkpoints only")
    encoder = ConditionEncoder(model, args.pc_points)
    engine = Engine(model)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generated = valid = saved = 0
    for index, item in enumerate(items):
        seed = args.seed + index
        embedding, preview = encoder.encode(item, seed)
        samples = collect_samples(
            engine.generate(
                [1],
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
                top_p=0.9,
                condition_embed=embedding,
            ),
            args.num_samples,
        )
        generated += len(samples)
        for sample_index, sample in enumerate(samples):
            if not sample.valid:
                continue
            valid += 1
            # Use a basename and an index so caption IDs cannot escape the output directory.
            stem = f"{index:05d}_{Path(item.name).name}_s{sample_index:02d}"
            try:
                saved += save_preview(sample, item, preview, stem, args)
            except Exception as error:
                print(f"[{stem}] Export failed: {error}")
        print(f"[{item.name}] generated={generated} valid={valid} saved={saved}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
