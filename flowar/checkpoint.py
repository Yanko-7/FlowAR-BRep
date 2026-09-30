from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        raw = model.module if hasattr(model, "module") else model
        self.model = copy.deepcopy(raw).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = (model.module if hasattr(model, "module") else model).state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.lerp_(src[k].to(v.device), 1 - self.decay)
            else:
                v.copy_(src[k].to(v.device))


# for inference
def load_model_from_checkpoint(
    path: str | Path,
    use_ema: bool = True,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    from flowar.config import model_config
    from flowar.models.model import FlowARBRep

    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    config = model_config(args)

    model = FlowARBRep(config=config)

    state_dict = None
    if use_ema and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
    if state_dict is None:
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), ckpt
