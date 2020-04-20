import torch
import torch.nn as nn

from speedrun import BaseExperiment, IOMixin, register_default_dispatch

from models import ContactTracingTransformer
from loader import get_dataloader
from losses import ContagionLoss, InfectiousnessLoss
from utils import to_device


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
        self.infectiousness_loss: nn.Module = InfectiousnessLoss(
            **self.get("losses/kwargs/infectiousness", {})
        )
        self.contagion_loss: nn.Module = ContagionLoss(
            **self.get("losses/kwargs/contagion", {})
        )
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
            # self.validate_epoch()

    def train_epoch(self):
        for model_input in self.progress(self.train_loader, tag="train"):
            # Evaluate model
            model_input = to_device(model_input, self.device)
            model_output = self.model(model_input)
            # Compute loss
            contagion_loss = self.contagion_loss(model_input, model_output)
            infectiousness_loss = self.infectiousness_loss(model_input, model_output)
            loss = (
                self.get("losses/weights/contagion", 1.0) * contagion_loss
                + self.get("losses/weights/infectiousness", 1.0) * infectiousness_loss
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Log to pbar
            self.log_progress(
                "train",
                loss=loss.item(),
                closs=contagion_loss.item(),
                iloss=infectiousness_loss.item(),
            )

    def validate_epoch(self):
        for model_input in self.progress(self.validate_loader, tag="validation"):
            with torch.no_grad():
                model_input = to_device(model_input, self.device)
                model_output = self.model(model_input)
                # TODO Continue


if __name__ == "__main__":
    CTTTrainer().run()
