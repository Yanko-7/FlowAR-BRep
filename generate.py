"""Generate B-Reps, complete prefixes, or apply geometric backtracking."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from flowar.checkpoint import load_model_from_checkpoint
from flowar.data.sequence import BRepTokenType, load_brep_sequence_from_npz
from flowar.generation.backtracking import GeomRejectionEngine
from flowar.generation.conditioning import ConditionEncoder, ConditionInput
from flowar.generation.engine import Engine
from flowar.generation.prefix import SequencePrefix
from flowar.generation.results import collect_samples
from flowar.geometry.validation import GeomValidationConfig


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    selected, _ = pre.parse_known_args(argv)
    defaults = {}
    if selected.config:
        with open(selected.config, encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream) or {}
        if not isinstance(defaults, dict):
            pre.error("Generation config must be a mapping")
        defaults = {key.replace("-", "_"): value for key, value in defaults.items()}

    parser = argparse.ArgumentParser(description="Distributed BRep generation and export")

    def option(name, default=None, **kwargs):
        key = name.replace("-", "_")
        value = defaults.pop(key, default)
        if value is None and default is not None:
            parser.error(f"Config {key} cannot be null")
        if value is not None:
            expected = (
                bool
                if kwargs.get("action") is argparse.BooleanOptionalAction
                else kwargs.get("type", str)
            )
            if not isinstance(value, expected) or (
                expected in (int, float) and isinstance(value, bool)
            ):
                if not (
                    expected is float and isinstance(value, int) and not isinstance(value, bool)
                ):
                    parser.error(f"Config {key} must be {expected.__name__}")
            if "choices" in kwargs and value not in kwargs["choices"]:
                parser.error(f"Invalid config {key}: {value!r}")
        parser.add_argument("--" + name, default=value, **kwargs)

    option("config", help="YAML defaults; command-line options take precedence")
    option("checkpoint", "outputs/checkpoint_step_100000.pt")
    option("total-samples", 4096, type=int, help="Global attempt limit across ranks")
    option("samples-per-batch", 512, type=int, help="Batch size per rank")
    option("max-tokens", 2048, type=int)
    option("seed", type=int, help="Base seed; omit for a random run")
    option("temperature", 1.0, type=float)
    option("top-p", 1.0, type=float)
    option("output-dir", "outputs/steps")
    option("output-format", "step", choices=("step", "brep", "none"))
    option("save-workers", 8, type=int, help="CAD worker processes per rank")
    option("save-timeout", 120, type=int, help="Timeout per CAD export, in seconds")
    option("trajectory-steps", 20, type=int, help="Geometry steps; 0 disables trajectory export")
    option("input-npz", help="One partial-BRep NPZ prefix")
    option("input-dir", help="Directory of partial-BRep NPZ prefixes")
    option("completions-per-prefix", 32, type=int)
    option(
        "complexity",
        2,
        type=int,
        choices=(0, 1, 2, 3),
        help="0: free; 1: 1–15; 2: 16–30; 3: 31+ faces",
    )
    option(
        "target-saved",
        type=int,
        help="Stop at this many successful builds, within total-samples attempts",
    )
    option("geom-endpoint-tol", 0.01, type=float)
    option("geom-budget-edge", 1, type=int)
    option("geom-budget-loop", 1, type=int)
    option("geom-budget-face", 20, type=int)
    option("geom-budget-total-face", 50, type=int)
    option("prompt", help="Text condition for a clip_text checkpoint")
    option(
        "gen-dtype",
        "fp16",
        choices=("fp16", "fp32"),
        help="Weight precision (CUDA decoding uses bf16 autocast)",
    )
    for name, default, help_text in (
        ("save-res", False, "Save raw tokens and geometry to .res.npz"),
        ("save-npz", False, "Save reconstructed geometry/topology to .npz"),
        ("save-png", True, "Save geometry previews"),
        ("use-fsm", False, "Constrain topology token sampling"),
        ("use-geom-rejection", False, "Enable geometry verification and backtracking"),
        ("geom-check-intersect", False, "Check curve intersections during backtracking"),
    ):
        option(name, default, action=argparse.BooleanOptionalAction, help=help_text)
    if defaults:
        parser.error("Unknown generation config fields: " + ", ".join(sorted(defaults)))
    parser.set_defaults(ordering="interleaved", reuse_edge_ids=True)
    return parser.parse_args(argv)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = 0
    local_rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def split_global_samples(total_samples: int, rank: int, world_size: int) -> int:
    base = total_samples // world_size
    rem = total_samples % world_size
    return base + (1 if rank < rem else 0)


def reduce_counters(local_counters: list[int], device: torch.device) -> list[int]:
    stats = torch.tensor(local_counters, device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats.tolist()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _predecode_uvgrid(res: dict, vae, device) -> dict:
    """Decode a result's per-primitive VAE latents into UV point grids in-place.

    Runs on the GPU in the main process so the CPU build workers receive numpy
    grids (picklable) rather than a CUDA VAE. No-op when there are no vectors.
    """
    if vae is None:
        return res
    if res.get("surf_vecs"):
        z = torch.as_tensor(np.stack(res["surf_vecs"]), dtype=torch.float32, device=device)
        res["surf_vecs"] = list(vae.decode_surf(z).cpu().numpy())
    if res.get("curv_vecs"):
        z = torch.as_tensor(np.stack(res["curv_vecs"]), dtype=torch.float32, device=device)
        res["curv_vecs"] = list(vae.decode_edge(z).cpu().numpy())
    return res


@dataclass
class GenerationStats:
    generated: int = 0
    valid: int = 0
    saved: int = 0
    failed: int = 0


def save_samples(samples, args, output, rank, batch_index, vae, representation, device, stats):
    from multiprocessing import get_context

    from pebble import ProcessPool

    from flowar.geometry.export import sample_stem, save_brep_sample, save_trajectories_batch

    errors = output / "errors.log"
    stats.generated += len(samples)
    valid = [(index, sample) for index, sample in enumerate(samples) if sample.valid]
    stats.valid += len(valid)
    if args.trajectory_steps:
        save_trajectories_batch(
            [s.export_data() for s in samples], rank, batch_index, output, errors
        )
    if not valid:
        return
    with ProcessPool(max_workers=args.save_workers, context=get_context("spawn")) as pool:
        pending = []
        for index, sample in valid:
            stem = sample_stem(rank, batch_index, index, sample.ids)
            data = _predecode_uvgrid(sample.geometry(), vae, device)
            future = pool.schedule(
                save_brep_sample,
                args=(
                    data,
                    stem,
                    output,
                    args.output_format,
                    args.save_res,
                    errors,
                    args.save_png,
                    args.save_npz,
                    "interleaved",
                    representation,
                ),
                kwargs={"raw_geometry": sample.geometry()},
                timeout=args.save_timeout,
            )
            pending.append((future, stem))
        for future, stem in pending:
            try:
                saved = future.result()
            except Exception as error:
                saved = False
                with errors.open("a", encoding="utf-8") as stream:
                    stream.write(f"{stem} | export_error | {error}\n")
            stats.saved += bool(saved)
            stats.failed += not saved


def generate_batch(model, args, count, seed, condition, prefix=None):
    common = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=seed,
        condition_embed=condition,
        num_sampling_steps=args.trajectory_steps or 20,
    )
    engine = Engine(model)
    if prefix is not None:
        steps = engine.generate_from_prefix(
            prefix,
            num_samples=count,
            max_new_tokens=args.max_tokens,
            use_fsm=args.use_fsm,
            return_with_trajectory=args.trajectory_steps > 0,
            **common,
        )
        return collect_samples(steps, count, SequencePrefix.from_sequence(prefix))
    tokens = [BRepTokenType.BOS]
    if args.complexity:
        tokens.append(int(BRepTokenType.COMPLEXITY_L1) + args.complexity - 1)
    if not args.use_geom_rejection:
        return collect_samples(
            engine.generate(
                tokens,
                num_samples=count,
                max_tokens=args.max_tokens,
                use_fsm=args.use_fsm,
                return_with_trajectory=args.trajectory_steps > 0,
                **common,
            ),
            count,
        )
    config = GeomValidationConfig(
        endpoint_tol=args.geom_endpoint_tol,
        check_intersect=args.geom_check_intersect,
        budget_edge=args.geom_budget_edge,
        budget_loop=args.geom_budget_loop,
        budget_face=args.geom_budget_face,
        budget_total_face=args.geom_budget_total_face,
    )
    rejection = GeomRejectionEngine(model)
    samples = []
    for index in range(count):
        steps = rejection.generate_single(
            tokens,
            max_tokens=args.max_tokens,
            use_fsm=True,
            geom_config=config,
            **{**common, "seed": seed + index},
        )
        samples.extend(collect_samples(steps, 1, SequencePrefix.from_tokens(tokens)))
    return samples


def report(stats, rank, device):
    totals = reduce_counters([stats.generated, stats.valid, stats.saved, stats.failed], device)
    if rank == 0:
        print(dict(zip(("generated", "valid", "saved", "failed"), totals)))


def run(args, rank, world_size, device):
    if args.samples_per_batch < 1 or args.max_tokens < 1 or args.total_samples < 0:
        raise ValueError("Batch/token limits must be positive and sample count nonnegative")
    if args.trajectory_steps < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid trajectory step count, temperature or top-p")
    if (
        min(
            args.geom_budget_edge,
            args.geom_budget_loop,
            args.geom_budget_face,
            args.geom_budget_total_face,
        )
        < 0
        or args.geom_endpoint_tol <= 0
    ):
        raise ValueError("Geometry retry budgets must be nonnegative and tolerance positive")
    if args.target_saved is not None and args.target_saved < 0:
        raise ValueError("Saved-sample target must be nonnegative")
    if args.save_workers < 1 or args.save_timeout <= 0:
        raise ValueError("Export worker count and timeout must be positive")
    if args.target_saved is not None and (args.input_npz or args.input_dir):
        raise ValueError("target-saved applies to fresh generation, not prefix completion")
    if args.input_npz and args.input_dir:
        raise ValueError("Choose one prefix file or one prefix directory")
    if args.use_geom_rejection and (args.input_npz or args.input_dir or args.trajectory_steps):
        raise ValueError(
            "Backtracking cannot be combined with prefix completion or trajectory export"
        )
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)
    representation = checkpoint.get("args", {}).get("geom_repr", "bezier")
    if representation != "bezier" and (args.input_npz or args.input_dir):
        raise ValueError("Prefix completion requires Bézier geometry")
    if args.use_geom_rejection and representation != "bezier":
        raise ValueError("Geometric backtracking requires Bézier geometry")
    dtype = torch.float16 if device.type == "cuda" and args.gen_dtype == "fp16" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    vae = None
    if representation == "uvgrid":
        from flowar.geometry.vae import GeomVAE

        values = checkpoint["args"]
        vae = GeomVAE(
            values["geom_vae_surf_ckpt"],
            values["geom_vae_edge_ckpt"],
            z_scale=values.get("geom_vae_z_scale", 1.0),
            device=device,
        )
    condition = None
    if args.prompt is not None:
        condition, _ = ConditionEncoder(model).encode(
            ConditionInput("prompt", "clip_text", args.prompt), args.seed or 0
        )
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    print(f"rank={rank} seed={seed}")
    output = Path(args.output_dir) / f"rank_{rank}"
    output.mkdir(parents=True, exist_ok=True)
    stats = GenerationStats()
    batch_index = 0
    if args.input_npz or args.input_dir:
        files = (
            [Path(args.input_npz)] if args.input_npz else sorted(Path(args.input_dir).glob("*.npz"))
        )
        if not files:
            raise ValueError("No prefix NPZ files found")
        if args.completions_per_prefix < 1:
            raise ValueError("Completion count must be positive")
        for file_index, path in enumerate(files):
            if file_index % world_size != rank:
                continue
            prefix = load_brep_sequence_from_npz(
                str(path), edge_id_range=checkpoint.get("args", {}).get("edge_id_range", 150)
            )
            for offset in range(0, args.completions_per_prefix, args.samples_per_batch):
                count = min(args.samples_per_batch, args.completions_per_prefix - offset)
                samples = generate_batch(
                    model,
                    args,
                    count,
                    seed + rank * 1000000 + file_index * args.completions_per_prefix + offset,
                    condition,
                    prefix,
                )
                save_samples(
                    samples, args, output, rank, batch_index, vae, representation, device, stats
                )
                batch_index += 1
    else:
        target = split_global_samples(args.total_samples, rank, world_size)
        wanted = (
            split_global_samples(args.target_saved, rank, world_size)
            if args.target_saved is not None
            else None
        )
        while stats.generated < target and (wanted is None or stats.saved < wanted):
            count = min(args.samples_per_batch, target - stats.generated)
            samples = generate_batch(
                model,
                args,
                count,
                seed + rank * 1000000 + batch_index * args.samples_per_batch,
                condition,
            )
            save_samples(
                samples, args, output, rank, batch_index, vae, representation, device, stats
            )
            print(
                f"rank={rank} generated={stats.generated} valid={stats.valid} saved={stats.saved}"
            )
            batch_index += 1
        if wanted is not None and stats.saved < wanted:
            print(f"rank={rank}: reached total-samples attempt limit before target-saved")
    report(stats, rank, device)


def main():
    args = parse_args()
    rank, _, world_size, device = setup_distributed()
    try:
        with torch.inference_mode():
            run(args, rank, world_size, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
