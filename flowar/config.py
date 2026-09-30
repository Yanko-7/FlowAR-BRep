"""Small YAML configuration loader and model configuration conversion."""

import argparse
from pathlib import Path

import yaml


def load_config(path: str, _parents: tuple[Path, ...] = ()) -> argparse.Namespace:
    """Load sectioned YAML, optionally extending one file via `base`."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    with path.open(encoding="utf-8") as stream:
        sections = yaml.safe_load(stream)
    if not isinstance(sections, dict):
        raise ValueError("Configuration must be a mapping of sections")
    base = sections.pop("base", None)
    values = vars(load_config(path.parent / base, (*_parents, path))) if base else {}
    aliases = {
        "path": "data_path",
        "enabled": "use_wandb",
        "project": "wandb_project",
        "entity": "wandb_entity",
        "run_name": "wandb_run_name",
    }
    seen = set()
    for section, fields in sections.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Section {section!r} must contain key/value pairs")
        for key, value in fields.items():
            key = aliases.get(key, key)
            if key in seen:
                raise ValueError(f"Duplicate configuration field: {key}")
            seen.add(key)
            values[key] = value
    validate_method(values)
    condition = values.get("cond_type")
    required = {
        "pointcloud": ("cond_pc_ckpt", "cond_pc_extra_dir"),
        "dino_image": ("cond_dino_model", "cond_image_dir"),
        "clip_text": ("cond_caption_json",),
    }
    for key in required.get(condition, ()):
        if not values.get(key):
            raise ValueError(f"{key} is required for {condition} conditioning")
    return argparse.Namespace(**values)


def validate_method(values: dict) -> None:
    for key, expected in {
        "ordering": "interleaved",
        "face_reorder": "bfs",
        "reuse_edge_ids": True,
        "prediction_type": "x_prev",
        "use_diffusion_forcing": False,
    }.items():
        if key in values and values[key] != expected:
            raise ValueError(f"{key} must be {expected!r} in this reference implementation")
    if values.get("geom_repr", "bezier") not in ("bezier", "uvgrid"):
        raise ValueError("geom_repr must be 'bezier' or 'uvgrid'")
    if values.get("cond_type") not in (None, "pointcloud", "dino_image", "clip_text"):
        raise ValueError("cond_type must be null, pointcloud, dino_image, or clip_text")


def model_config(values: dict):
    """Translate training/checkpoint fields without duplicating model construction."""
    from flowar.models.model import FlowARBRepConfig

    validate_method(values)
    names = (
        "hidden_size",
        "dropout",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "surface_latent_dim",
        "curve_latent_dim",
        "rms_norm_eps",
        "max_position_embeddings",
        "cond_type",
        "cond_num_tokens",
        "cond_freeze_encoder",
        "cond_pc_ckpt",
        "cond_clip_model",
        "cond_dino_model",
    )
    result = {key: values[key] for key in names if key in values}
    mapping = {
        "depth": "diff_depth",
        "adaln_depth": "diff_adaln_depth",
        "model_dim": "diff_model_dim",
        "time_shift": "diff_time_shift",
        "P_mean": "diff_P_mean",
        "P_std": "diff_P_std",
        "batch_mul": "diff_batch_mul",
        "surf_batch_mul": "diff_surface_batch_mul",
        "curv_batch_mul": "diff_curve_batch_mul",
    }
    result.update({new: values[old] for old, new in mapping.items() if old in values})
    if values.get("geom_repr") == "uvgrid":
        result.update(surface_latent_dim=48, curve_latent_dim=12)
    return FlowARBRepConfig(**result)
