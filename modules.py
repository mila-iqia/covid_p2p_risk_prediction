import torch
import torch.nn as nn

from utils import thermometer_encoding


class HealthHistoryEmbedding(nn.Sequential):
    def __init__(self, in_features, embedding_size, capacity=128, dropout=0.1):
        super(HealthHistoryEmbedding, self).__init__(
            nn.Linear(in_features, capacity),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(capacity, embedding_size),
        )

    def forward(self, input, mask=None):
        output = super(HealthHistoryEmbedding, self).forward(input)
        if mask is not None:
            output = output * mask[:, :, None]
        return output


class HealthProfileEmbedding(HealthHistoryEmbedding):
    pass


class MessageEmbedding(nn.Sequential):
    def __init__(self, message_dim, embedding_size, capacity=128, dropout=0.1):
        super(MessageEmbedding, self).__init__(
            nn.Linear(message_dim, capacity),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(capacity, embedding_size),
        )

    def forward(self, input, mask=None):
        output = super(MessageEmbedding, self).forward(input)
        if mask is not None:
            output = output * mask[:, :, None]
        return output


class PartnerIdEmbedding(nn.Linear):
    def __init__(self, num_id_bits, embedding_size):
        super(PartnerIdEmbedding, self).__init__(num_id_bits, embedding_size)

    def forward(self, input, mask=None):
        output = super(PartnerIdEmbedding, self).forward(input)
        if mask is not None:
            output = output * mask[:, :, None]
        return output


class DurationEmbedding(HealthHistoryEmbedding):
    EPS = 0.0001

    def __init__(
        self,
        embedding_size,
        num_thermo_bins=32,
        capacity=128,
        dropout=0.1,
        thermo_range=(0.0, 6.0),
    ):
        super(DurationEmbedding, self).__init__(
            in_features=num_thermo_bins,
            embedding_size=embedding_size,
            capacity=capacity,
            dropout=dropout,
        )
        self.num_thermo_bins = num_thermo_bins
        self.thermo_range = thermo_range

    def forward(self, input, mask=None):
        assert input.shape[-1] == 1
        encoded_input = thermometer_encoding(
            torch.log(input + self.EPS),
            value_range=self.thermo_range,
            size=self.num_thermo_bins,
        )
        return super(DurationEmbedding, self).forward(encoded_input, mask)


class EntityMasker(nn.Module):
    def forward(self, entities, mask):
        assert mask.shape[0:2] == entities.shape[0:2]
        return entities * mask[:, :, None]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        encoding_dim=16,
        position_dim=1,
        max_frequency=10000,
        normalize=True,
        trainable=False,
    ):
        super(PositionalEncoding, self).__init__()
        assert (
            encoding_dim % position_dim
        ) == 0, "Encoding dim must be divisible by the position dim."
        assert (
            (encoding_dim // position_dim) % 2
        ) == 0, "Encoding dim / postion dim must be even."
        self.encoding_dim = encoding_dim
        self.position_dim = position_dim
        self.max_frequency = max_frequency
        self.normalize = normalize
        self.trainable = trainable
        self._init_parameters()

    def _init_parameters(self):
        _exps = torch.arange(
            0, self.encoding_dim // self.position_dim, 2, dtype=torch.float
        )
        if self.trainable:
            # noinspection PyArgumentList
            _intervals = torch.nn.Parameter(
                torch.ones(self.encoding_dim // self.position_dim // 2)
            )
            # noinspection PyArgumentList
            _min_val = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
            # noinspection PyArgumentList
            _delta_val = torch.nn.Parameter(
                torch.tensor([_exps.max()], dtype=torch.float)
            )
            self.register_parameter("_intervals", _intervals)
            self.register_parameter("_min_val", _min_val)
            self.register_parameter("_delta_val", _delta_val)

    def get_exponents(self, device=None):
        if self.trainable:
            # Make sure that the min val and delta val are positive
            min_val = torch.relu(self._min_val)
            delta_val = torch.clamp_min(self._delta_val, 1e-4)
            intervals = torch.cumsum(torch.softmax(self._intervals, 0), 0)
            exps = min_val + delta_val * intervals
            return exps
        else:
            return torch.arange(
                0,
                self.encoding_dim // self.position_dim,
                2,
                dtype=torch.float,
                device=device,
            )

    def forward(self, positions, mask=None):
        squeeze = False
        if positions.ndim == 3:
            positions = positions[:, :, None, :]
            squeeze = True
        # positions.shape = NTAD, where D = self.position_dim
        N, T, A, D = positions.shape
        assert D == self.position_dim
        # The final encoding.shape = NTAC, where C = self.encoding_dim,
        # but per input dimension, we get C // D encoding dimensions. Let C' = C // D.
        encoding_dim_per_dim = self.encoding_dim // D
        # exps is like `i` in Attention is All You Need.
        exps = self.get_exponents(device=positions.device)
        # Divisor is 10000^(i/encoding_dim), but reshaped for proper broadcasting
        divisors = torch.pow(self.max_frequency, (exps / encoding_dim_per_dim))[
            None, None, None, None, :
        ]
        # pre_sinusoids is a NTAD(C'/2) tensor.
        pre_sinusoids = positions[:, :, :, :, None] / divisors
        # Apply sinusoids to obtain a NTADC' tensor.
        post_sinusoids = torch.cat(
            [torch.sin(pre_sinusoids), torch.cos(pre_sinusoids)], dim=-1
        )
        # Now flatten the last two dimensions to obtain a NTAC tensor (remember C = D * C')
        encodings = post_sinusoids.reshape(N, T, A, self.encoding_dim)
        # Normalize if required
        if self.normalize:
            encodings = encodings / torch.norm(encodings, dim=-1, keepdim=True)
        # Squeeze out the extra dimension if required
        if squeeze:
            encodings = encodings[:, :, 0, :]
        if mask is not None:
            encodings = encodings * (
                mask[:, :, None] if squeeze else mask[:, :, None, None]
            )
        return encodings
