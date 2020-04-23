from collections import defaultdict
from addict import Dict

import numpy as np
import torch
import torch.nn as nn

from speedrun import BaseExperiment, IOMixin, register_default_dispatch

from models import ContactTracingTransformer
from loader import get_dataloader
from losses import WeightedSum
from utils import to_device, momentum_accumulator


class CTTTrainer(IOMixin, BaseExperiment):
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
        for epoch in self.progress(
            range(self.get("training/num_epochs", ensure_exists=True)), tag="epochs"
        ):
            self.train_epoch()
            validation_stats = self.validate_epoch()
            self.log_progress("epochs", **validation_stats)

    def train_epoch(self):
        self.clear_moving_averages()
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
            # Log to pbar
            self.accumulate_in_cache(
                "moving_loss", loss.item(), momentum_accumulator(0.9)
            )
            self.log_progress(
                "train", loss=self.read_from_cache("moving_loss"),
            )

    def validate_epoch(self):
        all_losses = defaultdict(list)
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
        return all_losses

    def clear_moving_averages(self):
        return self.clear_in_cache("moving_loss")


if __name__ == "__main__":
    CTTTrainer().run()
