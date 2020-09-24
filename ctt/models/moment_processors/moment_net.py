from contextlib import contextmanager

import torch
import torch.nn as nn
import ctt.models.attn as attn
import ctt.models.modules as mod


class _MomentNet(nn.Module):
    def __init__(self, moments, network):
        super(_MomentNet, self).__init__()
        # Privates
        self._output_as_tuple = False
        # Public
        self.moments = moments
        self.network = network

    @contextmanager
    def output_as_tuple(self):
        old_output_as_tuple = self._output_as_tuple
        self._output_as_tuple = True
        yield
        self._output_as_tuple = old_output_as_tuple

    def flatten(self, inputs):
        # -------- Shape Wrangling --------
        batch_size = inputs["health_history"].shape[0]
        # -------- The Flattening --------
        health_history = (
            inputs["health_history"] * inputs["valid_history_mask"][:, :, None]
        )
        history_days = inputs["history_days"] * inputs["valid_history_mask"][:, :, None]
        health_variables = torch.cat(
            [
                health_history.reshape(batch_size, -1),
                inputs["health_profile"],
                history_days.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        encounter_variables = torch.cat(
            [
                inputs["encounter_health"],
                inputs["encounter_message"],
                inputs["encounter_day"],
                inputs["encounter_duration"],
            ],
            dim=-1,
        )
        # Compress encounter variables with their moments...
        encounter_moments = self.moments(
            encounter_variables, mask=inputs["mask"][:, :, None], dim=1
        )
        # ... and flatten them
        encounter_moments = encounter_moments.reshape(batch_size, -1)
        # Now concatenate everything together and call it a method
        return torch.cat([health_variables, encounter_moments], dim=-1)

    def forward(self, inputs: dict) -> dict:
        """
        inputs is a dict containing the below keys. The format of the tensors
        are indicated as e.g. `BTC`, `BMC` (etc), which can be interpreted as
        following.
            B: batch size,
            T: size of the rolling window over health history (i.e. number of
               time-stamps),
            C: number of generic channels,
            M: number of encounters,
        Elements with pre-determined shapes are indicated as such.
        For example:
            - B(14) indicates a tensor of shape (B, 14),
            - BM1 indicates a tensor of shape (B, M, 1)
            - B(T=14)C indicates a tensor of shape (B, 14, C) where 14
                is the currently set size of the rolling window.

        Parameters
        ----------
        inputs : dict
            A python dict with the following keys:
                -> `health_history`: a B(T=14)C tensor of the 14-day health
                    history (symptoms + test results + day) of the individual.
                -> `health_profile`: a BC tensor of the health profile
                    containing (age + health + preexisting_conditions) of the
                    individual.
                -> `history_days`: a B(T=14)1 tensor of the day corresponding to the
                    T dimension in `health_history`.
                -> `encounter_health`: a BMC tensor of health during an
                    encounter indexed by M.
                -> `encounter_message`: a BMC tensor of the received
                    message from the encounter partner.
                -> `encounter_day`: a BM1 tensor of the encounter day.
                -> `encounter_duration`: a BM1 tensor of the encounter duration.
                    This is not the actual duration, but a proxy (for the number
                    of encounters)
                -> `encounter_partner_id`: a binary  BMC tensor specifying
                    the ID of the encounter partner.
                -> `mask`: a BM mask tensor distinguishing the valid entries (1)
                    from padding (0) in the set valued inputs.
                -> `valid_history_mask`: a B(14) mask tensor distinguising valid
                    points in history (1) from padding (0).
        Returns
        -------
        dict
            A dict containing the keys "encounter_variables" and "latent_variable".
        """
        # -------- Shape Wrangling --------
        batch_size = inputs["health_history"].shape[0]
        num_history_days = inputs["health_history"].shape[1]
        # -------- Model Eval --------
        flattened_inputs = self.flatten(inputs)
        outputs = self.network(flattened_inputs).reshape(
            batch_size, num_history_days, 2
        )
        results = dict()
        results["encounter_variables"] = outputs[:, :, 0:1]
        results["latent_variable"] = outputs[:, :, 1:2]
        return results


class MomentNet(_MomentNet):
    RES_LINEAR_RELU_BLOCK_TYPE = "r"
    DOUBLE_RES_LINEAR_RELU_BLOCK_TYPE = "d"
    LINEAR_RELU_BLOCK_TYPE = "n"
    LINEAR_BLOCK_TYPE = "l"

    def __init__(
        self,
        *,
        # Feature construction
        num_health_history_features=28,
        num_health_profile_features=13,
        message_dim=1,
        num_days=14,
        num_moments=2,
        # Network
        capacity=128,
        block_types=f"{LINEAR_RELU_BLOCK_TYPE}{LINEAR_BLOCK_TYPE}",
        # Output
        encounter_output_features=1,
        latent_variable_output_features=1,
    ):
        assert encounter_output_features == latent_variable_output_features == 1
        # Get input and output dimensions
        health_in_dim = (
            num_health_history_features * num_days
            + num_health_profile_features
            + num_days
        )
        encounter_in_dim = (
            num_health_history_features + message_dim + 1 + 1
        ) * num_moments
        in_dim = health_in_dim + encounter_in_dim
        out_dim = num_days * 2
        # Build model
        blocks = []
        for block_idx, block_type in enumerate(block_types):
            if block_idx == 0:
                # First block
                _in = in_dim
                _out = capacity if len(block_types) > 1 else out_dim
            elif (block_idx == (len(block_types) - 1)) and len(block_types) > 1:
                # Last block
                _in = capacity
                _out = out_dim
            else:
                # Middle block
                _in = capacity
                _out = capacity
            if block_type == self.LINEAR_BLOCK_TYPE:
                blocks.append(nn.Linear(_in, _out))
            elif block_type == self.LINEAR_RELU_BLOCK_TYPE:
                blocks.append(attn.LinearReLU(_in, _out))
            elif block_type == self.RES_LINEAR_RELU_BLOCK_TYPE:
                assert _in == _out
                blocks.append(attn.ResLinearReLU(_in))
            elif block_type == self.DOUBLE_RES_LINEAR_RELU_BLOCK_TYPE:
                assert _in == _out
                blocks.append(attn.ResDoubleLinearReLU(_in))
            else:
                raise ValueError
        # Chain 'em
        network = nn.Sequential(*blocks)
        # Build moment processor
        moments = mod.Moments(num_moments=num_moments)
        # Done
        super(MomentNet, self).__init__(
            moments=moments, network=network,
        )
