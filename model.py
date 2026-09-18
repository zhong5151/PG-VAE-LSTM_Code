"""Neural-network architectures used in the PG-VAE-LSTM ablation study.

The implementation deliberately keeps the hidden size, number of recurrent layers,
and dropout configuration consistent across LSTM, ED-LSTM, VAE-LSTM, and
PG-VAE-LSTM. Non-negativity is not hard-coded in any model output; it is treated as
an explicit physical-consistency loss only for PG-VAE-LSTM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    pred: torch.Tensor
    q_mu: torch.Tensor | None = None
    q_logvar: torch.Tensor | None = None
    p_mu: torch.Tensor | None = None
    p_logvar: torch.Tensor | None = None


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> ModelOutput:
        h, _ = self.lstm(x.float())
        pred = self.head(self.dropout(h[:, -1, :]))
        return ModelOutput(pred=pred)


class EDLSTMRegressor(nn.Module):
    """Encoder-decoder LSTM used as a deterministic ablation baseline."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bridge = nn.Linear(hidden_size, 1)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, sample: bool = True) -> ModelOutput:
        enc, _ = self.encoder(x.float())
        compressed = self.bridge(enc)
        dec, _ = self.decoder(compressed)
        pred = self.head(self.dropout(dec[:, -1, :]))
        return ModelOutput(pred=pred)


class VAELSTM(nn.Module):
    """Conditional-prior VAE-LSTM.

    Posterior q(z_t | h_t) is parameterized from the current encoder state h_t.
    Prior p(z_t | h_{t-1}) is parameterized from the previous encoder state.
    The complete latent sequence is decoded by an LSTM and the final decoder state
    produces target-day ET.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.q_mu = nn.Linear(hidden_size, latent_size)
        self.q_logvar = nn.Linear(hidden_size, latent_size)
        self.p_mu = nn.Linear(hidden_size, latent_size)
        self.p_logvar = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, sample: bool = True) -> ModelOutput:
        h, _ = self.encoder(x.float())  # [B,T,H]
        q_mu = self.q_mu(h)
        q_logvar = self.q_logvar(h).clamp(min=-20.0, max=10.0)

        h_prev = torch.zeros_like(h)
        h_prev[:, 1:, :] = h[:, :-1, :]
        p_mu = self.p_mu(h_prev)
        p_logvar = self.p_logvar(h_prev).clamp(min=-20.0, max=10.0)

        z = self.reparameterize(q_mu, q_logvar) if sample else q_mu
        dec, _ = self.decoder(z)
        pred = self.head(self.dropout(dec[:, -1, :]))
        return ModelOutput(
            pred=pred,
            q_mu=q_mu,
            q_logvar=q_logvar,
            p_mu=p_mu,
            p_logvar=p_logvar,
        )


def build_model(cfg: Dict) -> nn.Module:
    name = str(cfg["model"]["name"]).lower()
    kwargs = dict(
        input_size=int(cfg["data"]["input_size"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    if name == "lstm":
        return LSTMRegressor(**kwargs)
    if name == "ed_lstm":
        return EDLSTMRegressor(**kwargs)
    if name in {"vae_lstm", "pg_vae_lstm"}:
        return VAELSTM(latent_size=int(cfg["model"]["latent_size"]), **kwargs)
    raise ValueError(f"Unknown model: {name}")
