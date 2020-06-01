import torch
import torch.nn as nn

from ctt.utils import thermometer_encoding, compute_moments


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
    EPS = 1e-7

    def __init__(self, mode="multiplicative"):
        super(EntityMasker, self).__init__()
        assert mode in ["multiplicative", "logsum"]
        self.mode = mode

    def forward(self, entities, mask):
        assert mask.shape[0:2] == entities.shape[0:2]
        if self.mode == "multiplicative":
            return entities * mask[:, :, None]
        elif self.mode == "logsum":
            with torch.no_grad():
                log_mask = torch.log(mask.clamp_min(0.0) + self.EPS)
            return entities + log_mask[:, :, None]
        else:
            raise NotImplementedError


class TimeEmbedding(nn.Embedding):
    def __init__(self, embedding_size, num_timestamps=14):
        super(TimeEmbedding, self).__init__(
            num_embeddings=num_timestamps, embedding_dim=embedding_size
        )

    def forward(self, timestamps, mask=None):
        timestamps = timestamps.abs().long()
        if timestamps.dim() == 3:
            timestamps = timestamps[..., 0]
        assert timestamps.dim() == 2
        output = super(TimeEmbedding, self).forward(timestamps)
        if mask is not None:
            output = output * mask[:, :, None]
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, encoding_dim=16, position_dim=1, max_frequency=10000, normalize=True,
    ):
        super(PositionalEncoding, self).__init__()
        assert (
            encoding_dim % position_dim
        ) == 0, "Encoding dim must be divisible by the position dim."
        assert (
            (encoding_dim // position_dim) % 2
        ) == 0, "Encoding dim / postion dim must be even."
        self.encoding_dim = encoding_dim
        self.position_dim = 1
        self.max_frequency = max_frequency
        self.normalize = normalize

    def get_exponents(self, device=None):
        return torch.arange(
            0,
            self.encoding_dim // self.position_dim,
            2,
            dtype=torch.float,
            device=device,
        )

    def forward(self, positions, mask=None):
        assert positions.ndim == 3
        # positions.shape = NTD, where D = self.position_dim
        N, T, D = positions.shape
        assert D == self.position_dim
        # The final encoding.shape = NTC, where C = self.encoding_dim,
        # but per input dimension, we get C // D encoding dimensions. Let C' = C // D.
        encoding_dim_per_dim = self.encoding_dim // D
        # exps is like `i` in Attention is All You Need.
        exps = self.get_exponents(device=positions.device)
        # Divisor is 10000^(i/encoding_dim), but reshaped for proper broadcasting
        divisors = torch.pow(self.max_frequency, (exps / encoding_dim_per_dim))[
            None, None, None, :
        ]
        # pre_sinusoids is a NTD(C'/2) tensor.
        pre_sinusoids = positions[:, :, :, None] / divisors
        # Apply sinusoids to obtain a NTDC' tensor.
        post_sinusoids = torch.cat(
            [torch.sin(pre_sinusoids), torch.cos(pre_sinusoids)], dim=-1
        )
        # Now flatten the last two dimensions to obtain a NTC tensor (remember C = D * C')
        encodings = post_sinusoids.reshape(N, T, self.encoding_dim)
        # Normalize if required
        if self.normalize:
            encodings = encodings / torch.norm(encodings, dim=-1, keepdim=True)
        if mask is not None:
            encodings = encodings * (mask[:, :, None])
        return encodings


class Moments(nn.Module):
    def __init__(self, num_moments=2, dim=None):
        super(Moments, self).__init__()
        self.num_moments = num_moments
        self.dim = dim

    def forward(self, x, mask=None, dim=None):
        dim = dim or self.dim
        assert dim is not None
        return compute_moments(x, mask=mask, dim=dim, num_moments=self.num_moments)
