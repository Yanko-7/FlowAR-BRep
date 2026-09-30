import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from flowar.config import load_config, model_config
from flowar.data.batch import collate_packed
from flowar.data.dataset import PackedDataset
from flowar.data.sequence import SPATIAL_RESOLUTION
from flowar.models.model import FlowARBRep
from flowar.training.trainer import Trainer

torch.multiprocessing.set_sharing_strategy("file_system")


def load_split_paths(json_path: str, root_dir: str, ext: str = ".npz") -> dict[str, list[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p2s = {p: split for split, prefixes in data.items() for p in prefixes}
    lengths = sorted({len(p) for p in p2s}, reverse=True)

    result = {split: [] for split in data}

    for r, _, files in os.walk(root_dir):
        for f in filter(lambda x: x.endswith(ext), files):
            for length in lengths:
                if len(f) >= length and (prefix := f[:length]) in p2s:
                    result[p2s[prefix]].append(os.path.join(r, f))
                    break

    return result


def train(args: argparse.Namespace, device: torch.device, rank: int, world_size: int = 1):
    dataset_paths = (
        load_split_paths(getattr(args, "split_file", ""), args.data_path)
        if getattr(args, "split_file", None)
        else {"train": args.data_path}
    )

    cond_num_tokens = getattr(args, "cond_num_tokens", 0) if getattr(args, "cond_type", None) else 0
    cond_pc_points = getattr(args, "cond_pc_points", 2048)
    cond_pc_extra_dir = getattr(args, "cond_pc_extra_dir", "")
    cond_image_dir = getattr(args, "cond_image_dir", "")
    cond_caption_json = (
        getattr(args, "cond_caption_json", "")
        if getattr(args, "cond_type", None) == "clip_text"
        else ""
    )
    caption_dropout_prob = getattr(args, "caption_dropout_prob", 0.1)

    dataset_kwargs = dict(
        max_num_tokens=args.max_num_tokens,
        augment=args.augment,
        rank=rank,
        world_size=world_size,
        resolution=SPATIAL_RESOLUTION,
        edge_id_range=getattr(args, "edge_id_range", 980),
        cond_num_tokens=cond_num_tokens,
        cond_clip_model=getattr(args, "cond_clip_model", "openai/clip-vit-large-patch14"),
        cond_pc_points=cond_pc_points,
        cond_pc_extra_dir=cond_pc_extra_dir,
        cond_image_dir=cond_image_dir,
        cond_caption_json=cond_caption_json,
        caption_dropout_prob=caption_dropout_prob,
        random_rotation_prob=getattr(args, "random_rotation_prob", 0.0),
        face_reorder=getattr(args, "face_reorder", "bfs"),
        ordering=getattr(args, "ordering", "interleaved"),
        reuse_edge_ids=getattr(args, "reuse_edge_ids", True),
        geom_repr=getattr(args, "geom_repr", "bezier"),
        random_edgeid_start=getattr(args, "random_edgeid_start", True),
    )
    train_dataset = PackedDataset(dataset_paths.get("train", []), **dataset_kwargs)

    if not train_dataset.file_paths:
        raise ValueError("No training NPZ files matched the configured data path and split")

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=collate_packed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **({"prefetch_factor": 2, "persistent_workers": True} if args.num_workers else {}),
    )

    val_loader = None
    if val_paths := dataset_paths.get("val", dataset_paths.get("validation")):
        val_loader = DataLoader(
            PackedDataset(
                val_paths,
                **(
                    dataset_kwargs
                    | {
                        "augment": False,
                        "shuffle": False,
                        "caption_dropout_prob": 0.0,
                        "random_rotation_prob": 0.0,
                    }
                ),
            ),
            batch_size=1,
            collate_fn=collate_packed,
            pin_memory=True,
        )

    # UV-grid geometry representation uses fixed VAE-latent dimensions
    # (surf [3,4,4]=48, curve [3,4]=12), overriding the Bézier control-point dims.
    if getattr(args, "geom_repr", "bezier") == "uvgrid":
        args.surface_latent_dim = 48
        args.curve_latent_dim = 12

    config = model_config(vars(args))
    model = FlowARBRep(config=config)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n=== Model parameters ===\n  Total parameters: {total_params:,}\n  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        diff_params = sum(p.numel() for p in model.curve_diff_head.parameters()) + sum(
            p.numel() for p in model.surface_diff_head.parameters()
        )
        print(f"  Diff head parameters: {diff_params:,} ({diff_params / total_params:.2%})")

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        args=args,
        device=device,
        rank=rank,
    )

    resume_path = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs/train.yaml")))
    parser.add_argument("--data", dest="data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--resume-from", dest="resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--grad-accum", dest="gradient_accumulation_steps", type=int, default=None)
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    for k, v in vars(cli_args).items():
        if v is not None and k != "config":
            setattr(args, k, v)

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, local_rank, world_size = (
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, device = (
            0,
            1,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    train(args=args, device=device, rank=rank, world_size=world_size)


if __name__ == "__main__":
    main()
