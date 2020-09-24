# https://github.com/juho-lee/set_transformer/blob/master/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    EPS = 1e-7
    TF_COMPAT = False

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, weights=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = self._compute_attention_weights(Q_, K_, weights)
        # funky split spaghetti to avoid pytorch call overload issues
        split_size = Q.size(0)
        if self.TF_COMPAT:
            # ONNX doesn't like split() being called on non-constant inputs.
            split_size = int(split_size)
        elif not isinstance(split_size, torch.Tensor):
            split_size = torch.IntTensor([split_size])[0]
        # split will still throw a tracing warning, but we can't avoid the int conversion...
        O = torch.cat(super(torch.Tensor, Q_ + A.bmm(V_)).split(split_size, dim=0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O

    def _compute_attention_weights(self, Q_, K_, weights=None):
        if weights is None:
            # Simple codepath for unweighted attention
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        else:
            # If weights is a tensor, broadcast along all heads
            if torch.is_tensor(weights):
                weights = [weights] * self.num_heads
            assert isinstance(weights, list) and len(weights) == self.num_heads
            weights = torch.cat(weights, dim=0)
            assert weights.shape[0] == Q_.shape[0]
            # Log and clamp weights
            log_weights = torch.log(weights.clamp_min(0.0) + self.EPS)
            attention_scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
            A = torch.softmax(attention_scores + log_weights, 2)
        return A


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, weights=None):
        return self.mab(X, X, weights)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, weights=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, weights)
        return self.mab1(X, H, weights)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, weights=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, weights)


class SRB(nn.Module):
    """Set Residual Block"""

    def __init__(self, dim_in, dim_out, feature_size=None, aggregation="max"):
        super(SRB, self).__init__()
        feature_size = feature_size or dim_in
        self.fc_i = nn.Linear(dim_in, feature_size)
        self.fc_f = nn.Linear(feature_size + dim_in, dim_in)
        self.fc_o = nn.Linear(dim_in, dim_out)
        self.aggregation = aggregation

    def forward(self, X, weights=None):
        if weights is not None:
            mask = weights.gt(0.0)
        else:
            mask = None
        num_entities = X.shape[1]
        global_features = F.relu(self.fc_i(X))
        if self.aggregation == "max":
            if mask is not None:
                global_features = global_features.masked_fill(
                    (~mask)[..., None].expand_as(global_features), -float("inf")
                )
            global_features = global_features.max(1, keepdim=True).values.repeat(
                1, num_entities, 1
            )
        elif self.aggregation == "sum":
            if mask is not None:
                global_features = global_features.masked_fill(
                    (~mask)[..., None].expand_as(global_features), 0.0
                )
            global_features = global_features.sum(1, keepdim=True).repeat(
                1, num_entities, 1
            )
        elif self.aggregation == "mean":
            if mask is not None:
                global_features = global_features.masked_fill(
                    (~mask)[..., None].expand_as(global_features), 0.0
                )
            global_features = global_features.mean(1, keepdim=True).repeat(
                1, num_entities, 1
            )
            if mask is not None:
                # The mean divides by M, where M is the number of entities.
                # However, some of those M entities may have been padding and
                # accordingly set to 0. This affects the mean, so we undo
                # that here.
                normalizer = num_entities / mask.sum(1).float()
                global_features = global_features * normalizer[:, None, None]
        elif self.aggregation == "none":
            # Do not aggregate
            pass
        else:
            raise NotImplementedError
        Y = F.relu(self.fc_f(torch.cat([X, global_features], dim=-1))) + X
        Y = F.relu(self.fc_o(Y))
        return Y


class ResLinearReLU(nn.Module):
    """Fully Connected Residual Block"""
    def __init__(self, dim):
        super(ResLinearReLU, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, X):
        X = F.relu(self.fc(X)) + X
        return X


class ResDoubleLinearReLU(nn.Module):
    def __init__(self, dim):
        super(ResDoubleLinearReLU, self).__init__()
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, dim)

    def forward(self, X):
        hidden = F.relu(self.fc1(X))
        output = F.relu(self.fc2(hidden)) + X
        return output


class LinearReLU(nn.Linear):
    def __init__(self, dim_in, dim_out):
        super(LinearReLU, self).__init__(dim_in, dim_out)

    def forward(self, X):
        return F.relu(super(LinearReLU, self).forward(X))