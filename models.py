"""MLP classifier + flatten/unflatten of parameters to a single vector.

The flat-vector view is what gets homomorphically encrypted for secure aggregation.
The model uses only Linear/ReLU/Dropout (no BatchNorm buffers) so the parameter
vector fully describes the model state and aggregation is exact.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg, in_dim, n_classes):
    return MLP(in_dim, list(model_cfg["hidden"]), n_classes,
               float(model_cfg.get("dropout", 0.1)))


def get_param_vector(model) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    ).astype(np.float64)


def set_param_vector(model, vec: np.ndarray):
    i = 0
    for p in model.parameters():
        n = p.numel()
        chunk = np.asarray(vec[i:i + n], dtype=np.float32).reshape(p.shape)
        with torch.no_grad():
            p.copy_(torch.from_numpy(chunk))
        i += n


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
