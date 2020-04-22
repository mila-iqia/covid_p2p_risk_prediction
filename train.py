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

    @property
    def device(self):
        return self.get("device", "cpu")

    @register_default_dispatch
    def train(self):
        self.initialize_wandb()
        for epoch in self.progress(
            range(self.get("training/num_epochs", ensure_exists=True)), tag="epochs"
        ):
            self.train_epoch()
            validation_stats = self.validate_epoch()
            self.log_progress("epochs", **validation_stats)
            self.next_epoch()

    def train_epoch(self):
        self.clear_moving_averages()
        self.model.train()
        for model_input in self.progress(self.train_loader, tag="train"):
            # Evaluate model
            model_input = to_device(model_input, self.device)
            model_output = self.model(model_input)
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
        all_losses = defaultdict(list)
        self.model.eval()
        for model_input in self.progress(self.validate_loader, tag="validation"):
            with torch.no_grad():
                model_input = to_device(model_input, self.device)
                model_output = self.model(model_input)
                losses = self.loss(model_input, model_output)
                all_losses["loss"].append(losses.loss.item())
                for key in losses.unweighted_losses:
                    all_losses[key].append(losses.unweighted_losses[key].item())
        # Compute mean for all losses
        all_losses = Dict({key: np.mean(val) for key, val in all_losses.items()})
        self.log_validation_losses(all_losses)
        # Store the validation loss in cache. This will be used for checkpointing.
        self.write_to_cache("current_validation_loss", all_losses.loss)
        return all_losses

    def log_training_losses(self, losses):
        if self.log_wandb_now:
            metrics = Dict({"training_loss": losses.loss})
            metrics.update(
                {f"training_{k}": v for k, v in losses.unweighted_losses.items()}
            )
            self.wandb_log(**metrics)
        return self

    def checkpoint(self, force=True):
        current_validation_loss = self.read_from_cache(
            "current_validation_loss", float("inf")
        )
        best_validation_loss = self.read_from_cache(
            "best_validation_loss", float("inf")
        )
        if current_validation_loss < best_validation_loss:
            self.write_to_cache("best_validation_loss", current_validation_loss)
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

    def log_validation_losses(self, losses):
        metrics = {f"validation_{k}": v for k, v in losses.items()}
        self.wandb_log(**metrics)
        return self

    def clear_moving_averages(self):
        return self.clear_in_cache("moving_loss")


if __name__ == "__main__":
    CTTTrainer().run()
