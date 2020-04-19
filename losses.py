import torch
import torch.nn as nn
import torch.nn.functional as F


class InfectiousnessLoss(nn.Module):
    def forward(self, model_input, model_output):
        assert model_output.latent_variable.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        return F.mse_loss(model_output.latent_variable[:, :, 0:1], model_output)


class ContagionLoss(nn.Module):
    def forward(self, model_input, model_output):
        return F.binary_cross_entropy_with_logits(
            model_output.encounter_variables, model_input.encounter_is_contagion
        )
