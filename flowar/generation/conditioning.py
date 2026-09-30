"""Prepare modality-specific inputs; decoding and output handling stay modality-independent."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ConditionInput:
    name: str
    modality: str
    value: str | Path


class ConditionEncoder:
    def __init__(self, model, point_count=4096):
        if model.cond_adapter is None:
            raise ValueError("Checkpoint has no conditional encoder")
        if point_count < 1:
            raise ValueError("Point count must be positive")
        self.model = model
        self.point_count = point_count
        self.tokenizer = None

    @torch.inference_mode()
    def encode(self, item: ConditionInput, seed: int):
        if item.modality != self.model.config.cond_type:
            raise ValueError(
                f"Input modality {item.modality!r} does not match checkpoint {self.model.config.cond_type!r}"
            )
        device = next(self.model.parameters()).device
        if item.modality == "clip_text":
            from transformers import CLIPTokenizer

            if self.tokenizer is None:
                self.tokenizer = CLIPTokenizer.from_pretrained(self.model.config.cond_clip_model)
            inputs = self.tokenizer(
                str(item.value),
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            preview = str(item.value)
        elif item.modality == "dino_image":
            from PIL import Image
            from torchvision.transforms import v2

            with Image.open(item.value) as image:
                preview = image.convert("RGB")
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((256, 256), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            inputs = {"pixel_values": transform(preview).unsqueeze(0)}
        elif item.modality == "pointcloud":
            preview = load_pointcloud(item.value, self.point_count, seed)
            inputs = {"pts": torch.from_numpy(preview).unsqueeze(0)}
        else:
            raise ValueError(f"Unknown modality: {item.modality}")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            embedding = self.model.cond_adapter(
                **{key: value.to(device) for key, value in inputs.items()}
            )
        return embedding, preview


def load_pointcloud(path, count, seed):
    import trimesh

    rng = np.random.default_rng(seed)
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
        points, _ = trimesh.sample.sample_surface(geometry, count, seed=rng)
    else:
        points = np.asarray(geometry.vertices)
        if len(points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
        points = points[rng.choice(len(points), count, replace=len(points) < count)]
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    low, high = points.min(0), points.max(0)
    return ((points - (low + high) / 2) * (2 / max(float((high - low).max()), 1e-8))).astype(
        np.float32
    )
