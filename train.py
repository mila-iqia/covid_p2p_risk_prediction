import os
from collections import defaultdict
from addict import Dict

import numpy as np
import torch
import torch.nn as nn

from speedrun import BaseExperiment, IOMixin, register_default_dispatch
from speedrun.logging.wandb import WandBMixin

from models import ContactTracingTransformer
from loader import get_dataloader
from losses import WeightedSum
from utils import to_device, momentum_accumulator
from scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from metrics import Metrics


class CTTTrainer(WandBMixin, IOMixin, BaseExperiment):
    WANDB_PROJECT = "ctt"

    def __init__(self):
        super(CTTTrainer, self).__init__()
        self.auto_setup()
        self._build()

    def _build(self):
        self._build_loaders()
        self._build_model()
        self._build_criteria_and_optim()
        self._build_scheduler()

    def _build_model(self):
        self.model: nn.Module = to_device(
            ContactTracingTransformer(**self.get("model/kwargs", {})), self.device
        )

    def _build_loaders(self):
        train_path = self.get("data/paths/train", ensure_exists=True)
        validate_path = self.get("data/paths/validate", ensure_exists=True)
        self.train_loader = get_dataloader(
            path=train_path, **self.get("data/loader_kwargs", ensure_exists=True)
        )
        self.validate_loader = get_dataloader(
            path=validate_path, **self.get("data/loader_kwargs", ensure_exists=True)
        )

    def _build_criteria_and_optim(self):
        # noinspection PyArgumentList
        self.loss = WeightedSum.from_config(self.get("losses", ensure_exists=True))
        self.optim = torch.optim.Adam(
            self.model.parameters(), **self.get("optim/kwargs")
        )
        self.metrics = Metrics()

    def _build_scheduler(self):
        if self.get("scheduler/use", False):
            self._base_scheduler = CosineAnnealingLR(
                self.optim,
                T_max=self.get("training/num_epochs"),
                **self.get("scheduler/kwargs", {}),
            )
        else:
            self._base_scheduler = None
        # Support for LR warmup
        if self.get("scheduler/warmup", False):
            assert self._base_scheduler is not None
            self.scheduler = GradualWarmupScheduler(
                self.optim,
                multiplier=1,
                total_epoch=5,
                after_scheduler=self._base_scheduler,
            )
        else:
            self.scheduler = self._base_scheduler

    @property
    def device(self):
        return self.get("device", "cpu")

    @register_default_dispatch
    def train(self):
        if self.get("wandb/use", True):
            self.initialize_wandb()
        for epoch in self.progress(
            range(self.get("training/num_epochs", ensure_exists=True)), tag="epochs"
        ):
            self.log_learning_rates()
            self.train_epoch()
            validation_stats = self.validate_epoch()
            self.checkpoint()
            self.log_progress("epochs", **validation_stats)
            self.step_scheduler(epoch)
            self.next_epoch()

    def train_epoch(self):
        self.clear_moving_averages()
        self.model.train()
        for model_input in self.progress(self.train_loader, tag="train"):
            # Evaluate model
            model_input = to_device(model_input, self.device)
            model_output = Dict(self.model(model_input))
            # Compute loss
            losses = self.loss(model_input, model_output)
            loss = losses.loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Log to wandb (if required)
            self.log_training_losses(losses)
            # Log to pbar
            self.accumulate_in_cache(
                "moving_loss", loss.item(), momentum_accumulator(0.9)
            )
            self.log_progress(
                "train", loss=self.read_from_cache("moving_loss"),
            )
            self.next_step()

    def validate_epoch(self):
        all_losses_and_metrics = defaultdict(list)
        self.metrics.reset()
        self.model.eval()
        for model_input in self.progress(self.validate_loader, tag="validation"):
            with torch.no_grad():
                model_input = to_device(model_input, self.device)
                model_output = Dict(self.model(model_input))
                losses = self.loss(model_input, model_output)
                self.metrics.update(model_input, model_output)
                all_losses_and_metrics["loss"].append(losses.loss.item())
                for key in losses.unweighted_losses:
                    all_losses_and_metrics[key].append(
                        losses.unweighted_losses[key].item()
                    )
        # Compute mean for all losses
        all_losses_and_metrics = Dict(
            {key: np.mean(val) for key, val in all_losses_and_metrics.items()}
        )
        all_losses_and_metrics.update(Dict(self.metrics.evaluate()))
        self.log_validation_losses_and_metrics(all_losses_and_metrics)
        # Store the validation loss in cache. This will be used for checkpointing.
        self.write_to_cache("current_validation_loss", all_losses_and_metrics.loss)
        return all_losses_and_metrics

    def log_training_losses(self, losses):
        if not self.get("wandb/use", True):
            return self
        if self.log_wandb_now:
            metrics = Dict({"training_loss": losses.loss})
            metrics.update(
                {f"training_{k}": v for k, v in losses.unweighted_losses.items()}
            )
            self.wandb_log(**metrics)
        return self

    def checkpoint(self, force=False):
        current_validation_loss = self.read_from_cache(
            "current_validation_loss", float("inf")
        )
        best_validation_loss = self.read_from_cache(
            "best_validation_loss", float("inf")
        )
        if current_validation_loss < best_validation_loss:
            self.write_to_cache("best_validation_loss", current_validation_loss)
            ckpt_path = os.path.join(self.checkpoint_directory, "best.ckpt")
        elif self.get_arg("force_checkpoint", force):
            ckpt_path = os.path.join(self.checkpoint_directory, "best.ckpt")
        else:
            ckpt_path = None
        if ckpt_path is not None:
            info_dict = {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
            }
            torch.save(info_dict, ckpt_path)
        return self

    def load(self, device=None):
        ckpt_path = os.path.join(self.checkpoint_directory, "best.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError
        info_dict = torch.load(
            ckpt_path,
            map_location=torch.device((self.device if device is None else device)),
        )
        self.model.load_state_dict(info_dict["model"])
        self.optim.load_state_dict(info_dict["optim"])
        return self

    def log_validation_losses_and_metrics(self, losses):
        if not self.get("wandb/use", True):
            return self
        metrics = {f"validation_{k}": v for k, v in losses.items()}
        self.wandb_log(**metrics)
        return self

    def clear_moving_averages(self):
        return self.clear_in_cache("moving_loss")

    def step_scheduler(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch)
        return self

    def log_learning_rates(self):
        if not self.get("wandb/use", True):
            return self
        lrs = {
            f"lr_{i}": param_group["lr"]
            for i, param_group in enumerate(self.optim.param_groups)
        }
        self.wandb_log(**lrs)
        return self


if __name__ == "__main__":
    CTTTrainer().run()
