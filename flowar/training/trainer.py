import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from flowar.checkpoint import EMA

torch.multiprocessing.set_sharing_strategy("file_system")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def infinite_loader(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError(
                "The data loader produced no batches; check files and conditional inputs"
            )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: argparse.Namespace,
        device: torch.device,
        rank: int,
    ):
        self.args = args
        self.device = device
        self.rank = rank
        self.is_main = rank == 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.best_loss_metric = "val/loss" if val_loader is not None else "train/loss"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = train_loader.dataset

        self.model = model.to(device)
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        # DDP broadcasts the initial model parameters before they are copied into EMA.
        self.ema = EMA(self.model, decay=0.9999)

        if self.is_main:
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )

        if args.lr_scheduler == "cosine":
            self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.max_steps,
                min_lr=args.min_lr,
            )
        else:
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=args.warmup_steps
            )

        self.use_wandb = args.use_wandb and WANDB_AVAILABLE and self.is_main
        if self.use_wandb:
            wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                config=vars(args),
                name=args.wandb_run_name,
            )

        self.saved_checkpoints = []
        self.grad_accum_steps = args.gradient_accumulation_steps
        if self.grad_accum_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.use_amp = args.use_amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if getattr(args, "amp_dtype", "") == "bfloat16" else torch.float16
        )

        self.scaler = torch.amp.GradScaler(
            device.type, enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # UV-grid geometry representation: frozen VAE that compresses the
        # per-primitive UV grids emitted by the dataset into latents on the fly.
        self.geom_repr = getattr(args, "geom_repr", "bezier")
        self.geom_vae = None
        if self.geom_repr == "uvgrid":
            from flowar.geometry.vae import GeomVAE

            self.geom_vae = GeomVAE(
                surf_ckpt=getattr(args, "geom_vae_surf_ckpt", "abc_vae_surf.pt"),
                edge_ckpt=getattr(args, "geom_vae_edge_ckpt", "abc_vae_edge.pt"),
                z_scale=float(getattr(args, "geom_vae_z_scale", 1.0)),
                device=self.device,
            )
            if self.is_main:
                print(
                    f"✅ UV-grid geom_repr: VAE latents (surf 48 / curve 12), "
                    f"z_scale={self.geom_vae.z_scale}"
                )

        if self.is_main:
            if self.use_amp:
                print(f"✅ Mixed precision enabled: {self.amp_dtype}")
            if self.grad_accum_steps > 1:
                print(f"✅ Gradient accumulation steps: {self.grad_accum_steps}")

        self._val_call_count = 0

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        model_sd = checkpoint.get("model_state_dict")
        if model_sd is None:
            # Fallback: use ema_state_dict or treat checkpoint as bare state_dict
            model_sd = checkpoint.get("ema_state_dict")
        if model_sd is None:
            if any(k.startswith(("transformer.", "text_embed.", "text_head.")) for k in checkpoint):
                model_sd = checkpoint
            else:
                raise KeyError("Checkpoint missing key: model_state_dict")
        raw_model.load_state_dict(model_sd, strict=True)

        ema_sd = checkpoint.get("ema_state_dict")
        if ema_sd is not None:
            self.ema.model.load_state_dict(ema_sd, strict=True)
        else:
            self.ema = EMA(self.model, decay=0.9999)

        weights_only = bool(getattr(self.args, "resume_weights_only", False))

        if not weights_only:
            if bool(getattr(self.args, "resume_load_optimizer", True)):
                opt_sd = checkpoint.get("optimizer_state_dict")
                if opt_sd is not None:
                    self.optimizer.load_state_dict(opt_sd)
                    self._move_optimizer_state_to_device()
                if self.scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if bool(getattr(self.args, "resume_load_scheduler", True)):
                sch_sd = checkpoint.get("scheduler_state_dict")
                if sch_sd is not None:
                    self.scheduler.load_state_dict(sch_sd)

            if bool(getattr(self.args, "resume_load_dataset", False)):
                ds_sd = checkpoint.get("dataset_state_dict")
                if ds_sd is not None and hasattr(self.dataset, "load_state_dict"):
                    self.dataset.load_state_dict(ds_sd)

            self.global_step = int(checkpoint.get("step", 0))
            # Older checkpoints tracked training loss, even when validation was enabled.
            saved_metric = checkpoint.get("best_loss_metric", "train/loss")
            self.best_loss = (
                float(checkpoint.get("best_loss", float("inf")))
                if saved_metric == self.best_loss_metric
                else float("inf")
            )
            self._val_call_count = int(
                checkpoint.get(
                    "val_call_count",
                    self.global_step // max(1, int(getattr(self.args, "val_steps", 500))),
                )
            )

        if self.is_main:
            if weights_only:
                print(f"✅ Loaded pretrained weights from {ckpt_path} (weights only)")
            else:
                print(
                    f"✅ Resumed from {ckpt_path} | step={self.global_step} | "
                    f"best {self.best_loss_metric}={self.best_loss:.8f}"
                )

    def _build_checkpoint_payload(self) -> dict:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            "step": self.global_step,
            "best_loss": self.best_loss,
            "best_loss_metric": self.best_loss_metric,
            "val_call_count": self._val_call_count,
            "args": vars(self.args),
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": self.ema.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if hasattr(self.dataset, "state_dict"):
            checkpoint["dataset_state_dict"] = self.dataset.state_dict()
        return checkpoint

    def _prepare_inputs(self, batch) -> dict:
        return batch.to(self.device).model_inputs(self.geom_vae)

    def _compute_loss(self, ce, surface_mse, curve_mse):
        ce_mean = (
            ce.mean() if ce is not None and len(ce) > 0 else torch.tensor(0.0, device=self.device)
        )
        surf_mean = (
            surface_mse.mean()
            if surface_mse is not None and len(surface_mse) > 0
            else torch.tensor(0.0, device=self.device)
        )
        curv_mean = (
            curve_mse.mean()
            if curve_mse is not None and len(curve_mse) > 0
            else torch.tensor(0.0, device=self.device)
        )
        loss = ce_mean + surf_mean + curv_mean
        if loss == 0:
            raise ValueError("All losses are None!")
        return loss, ce_mean, surf_mean, curv_mean

    def train_step(self, batch, is_accumulating: bool):
        packed_data = self._prepare_inputs(batch)
        ctx = (
            self.model.no_sync()
            if (is_accumulating and isinstance(self.model, DDP))
            else nullcontext()
        )

        with ctx:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(**packed_data)
                loss, ce_mean, surf_mean, curv_mean = self._compute_loss(*outputs)
            self.scaler.scale(loss / self.grad_accum_steps).backward()

        if not is_accumulating:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), getattr(self.args, "gradient_clip", 1.0)
            )
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= previous_scale:
                # Overflow skips the optimizer update; scheduler and EMA must also wait.
                self.scheduler.step()
                self.ema.update(self.model)
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item(), ce_mean.item(), surf_mean.item(), curv_mean.item()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_iter = infinite_loader(self.train_loader)
        pbar = tqdm(
            range(self.global_step, self.args.max_steps),
            desc="Training",
            disable=not self.is_main,
        )

        run_metrics = {"loss": 0.0, "ce": 0.0, "surf_mse": 0.0, "curv_mse": 0.0}
        log_count, total_tokens, total_samples = 0, 0, 0

        last_loss = None
        for step in pbar:
            self.global_step = step + 1
            self.dataset.global_step = step
            step_metrics = {"loss": 0.0, "ce": 0.0, "surf_mse": 0.0, "curv_mse": 0.0}

            for micro_step in range(self.grad_accum_steps):
                batch = next(data_iter)
                total_samples += batch.sample_num()
                total_tokens += batch.get_token_num()

                is_accumulating = micro_step < self.grad_accum_steps - 1
                loss_step, ce, sm, cm = self.train_step(batch, is_accumulating)

                step_metrics["loss"] += loss_step / self.grad_accum_steps
                step_metrics["ce"] += ce / self.grad_accum_steps
                step_metrics["surf_mse"] += sm / self.grad_accum_steps
                step_metrics["curv_mse"] += cm / self.grad_accum_steps

            last_loss = step_metrics["loss"]
            for k in run_metrics:
                run_metrics[k] += step_metrics[k]
            log_count += 1

            if self.is_main:
                pbar.set_postfix(
                    {k: f"{v:.4f}" for k, v in step_metrics.items()}
                    | {"lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
                )

                if self.global_step % getattr(self.args, "logging_steps", 10) == 0:
                    if self.use_wandb:
                        wandb.log(
                            {f"train/{k}": v / log_count for k, v in run_metrics.items()}
                            | {
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/tokens": total_tokens,
                                "train/samples": total_samples,
                            },
                            step=self.global_step,
                        )
                    run_metrics = dict.fromkeys(run_metrics, 0.0)
                    log_count = 0

            selection_loss = last_loss if self.val_loader is None else None
            if self.val_loader is not None and (
                self.global_step % getattr(self.args, "val_steps", 1000) == 0
                or self.global_step == self.args.max_steps
            ):
                val_metrics = self.validate()
                selection_loss = val_metrics["val/loss"]
                torch.cuda.empty_cache()
                if self.is_main:
                    print(f"\n[Step {self.global_step}] Val loss: {val_metrics['val/loss']:.4f}")
                    if self.use_wandb:
                        wandb.log(val_metrics, step=self.global_step)

            if self.is_main:
                if self.global_step % getattr(self.args, "save_steps", 1000) == 0:
                    self.save_checkpoint(selection_loss)
                elif self.val_loader is not None and selection_loss is not None:
                    self.save_checkpoint(selection_loss, best_only=True)

        if self.is_main:
            self.save_checkpoint(last_loss if self.val_loader is None else None, final=True)
            if self.use_wandb:
                wandb.finish()
            print(f"\nTraining complete! Best {self.best_loss_metric}: {self.best_loss:.4f}")

    @torch.no_grad()
    def validate(self) -> dict:
        self._val_call_count += 1
        self.ema.model.eval()
        was_training = self.model.training
        self.model.eval()
        totals = torch.zeros(5, device=self.device, dtype=torch.float64)
        try:
            for index, batch in enumerate(self.val_loader):
                if index >= self.args.max_val_batches:
                    break
                with autocast(
                    device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    outputs = self.ema.model(**self._prepare_inputs(batch))
                    losses = self._compute_loss(*outputs)
                totals[:4] += torch.stack(losses).double()
                totals[4] += 1
            if dist.is_initialized():
                dist.all_reduce(totals)
            if totals[4] == 0:
                raise ValueError("Validation produced no batches; check its split and inputs")
            means = (totals[:4] / totals[4]).tolist()
            return dict(zip(("val/loss", "val/ce", "val/surf_mse", "val/curv_mse"), means))
        finally:
            self.model.train(was_training)

    def save_checkpoint(
        self, current_loss: float | None = None, final: bool = False, *, best_only: bool = False
    ):
        """Save state; only a fresh loss for the selected metric can update the best model."""
        is_best = (
            current_loss is not None
            and math.isfinite(current_loss)
            and current_loss < self.best_loss
        )
        if best_only and not is_best:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if is_best:
            self.best_loss = current_loss
        checkpoint = self._build_checkpoint_payload()

        if not best_only:
            save_path = self.output_dir / (
                "checkpoint_final.pt" if final else f"checkpoint_step_{self.global_step}.pt"
            )
            torch.save(checkpoint, save_path)
            print(f"\n💾 Saved checkpoint: {save_path}")

        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            print(f"🏆 New best model! {self.best_loss_metric}: {current_loss:.8f}")
            if self.use_wandb:
                wandb.run.summary["best_loss"] = self.best_loss

        if not final and not best_only:
            self.saved_checkpoints.append(save_path)
            if len(self.saved_checkpoints) > getattr(self.args, "save_total_limit", 3):
                old_ckpt = self.saved_checkpoints.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()
                    print(f"🗑️ Removed old checkpoint: {old_ckpt}")
