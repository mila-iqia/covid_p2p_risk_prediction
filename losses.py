import torch
import torch.nn as nn
import torch.nn.functional as F


class InfectiousnessLoss(nn.Module):
    def forward(self, model_input, model_output):
        assert model_output.latent_variable.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        return F.mse_loss(
            model_output.latent_variable[:, :, 0:1], model_input.infectiousness_history
        )


class ContagionLoss(nn.Module):
    def __init__(self, allow_multiple_exposures=True):
        """
        Parameters
        ----------
        allow_multiple_exposures : bool
            If this is set to False, only one encounter can be the contagion,
            in which case, we use a softmax + cross-entropy loss. If set to True,
            multiple events can be contagions, in which case we use sigmoid +
            binary cross entropy loss.
        """
        super(ContagionLoss, self).__init__()
        self.allow_multiple_exposures = allow_multiple_exposures

    def forward(self, model_input, model_output):
        if self.allow_multiple_exposures:
            return F.binary_cross_entropy_with_logits(
                model_output.encounter_variables, model_input.encounter_is_contagion
            )
        else:
            # TODO
            raise NotImplementedError
