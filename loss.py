"""Loss functions for LSTM, ED-LSTM, VAE-LSTM, and PG-VAE-LSTM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from model import ModelOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    physics: torch.Tensor
    energy: torch.Tensor
    water: torch.Tensor
    nonnegative: torch.Tensor

    def detached(self) -> Dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "reconstruction": float(self.reconstruction.detach().cpu()),
            "kl": float(self.kl.detach().cpu()),
            "physics": float(self.physics.detach().cpu()),
            "energy": float(self.energy.detach().cpu()),
            "water": float(self.water.detach().cpu()),
            "nonnegative": float(self.nonnegative.detach().cpu()),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not torch.any(mask):
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def conditional_gaussian_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL[q(z|h_t) || p(z|h_{t-1})], averaged over batch/time/latent dims."""
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1.0)
    if free_bits > 0:
        # Free bits are applied per latent dimension before averaging.
        kl = torch.clamp(kl, min=float(free_bits))
    return kl.mean()


def et_mm_day_to_latent_heat_wm2(et_mm_day: torch.Tensor, latent_heat_j_per_kg: float) -> torch.Tensor:
    # 1 mm water = 1 kg m-2. Divide daily energy by 86400 s.
    return et_mm_day * float(latent_heat_j_per_kg) / 86400.0


def energy_consistency_loss(
    pred_et_mm_day: torch.Tensor,
    energy_available_wm2: torch.Tensor,
    latent_heat_j_per_kg: float = 2.45e6,
    mode: str = "et_equivalent",
) -> torch.Tensor:

    pred = pred_et_mm_day.reshape(-1)
    cap = energy_available_wm2.reshape(-1)
    mask = torch.isfinite(pred) & torch.isfinite(cap)
    if not torch.any(mask):
        return pred.sum() * 0.0
    pred_flux = et_mm_day_to_latent_heat_wm2(pred[mask], latent_heat_j_per_kg)
    violation_flux = F.relu(pred_flux - cap[mask])
    if mode == "flux":
        return torch.mean(violation_flux.pow(2))
    if mode == "et_equivalent":
        conv = float(latent_heat_j_per_kg) / 86400.0
        return torch.mean((violation_flux / conv).pow(2))
    raise ValueError(f"Unknown energy loss mode: {mode}")


def water_consistency_loss(pred_et_mm_day: torch.Tensor, water_available_mm_day: torch.Tensor) -> torch.Tensor:
    pred = pred_et_mm_day.reshape(-1)
    cap = water_available_mm_day.reshape(-1)
    mask = torch.isfinite(pred) & torch.isfinite(cap)
    if not torch.any(mask):
        return pred.sum() * 0.0
    return torch.mean(F.relu(pred[mask] - cap[mask]).pow(2))


def nonnegative_loss(pred_et_mm_day: torch.Tensor) -> torch.Tensor:
    pred = pred_et_mm_day.reshape(-1)
    mask = torch.isfinite(pred)
    if not torch.any(mask):
        return pred.sum() * 0.0
    return torch.mean(F.relu(-pred[mask]).pow(2))


class CompositeLoss:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model_name = str(cfg["model"]["name"]).lower()
        self.beta = float(cfg["loss"]["beta_kl"])
        self.gamma = float(cfg["loss"]["gamma_phy"])
        self.free_bits = float(cfg["loss"].get("free_bits", 0.0))

    def __call__(
        self,
        out: ModelOutput,
        target_model_space: torch.Tensor,
        pred_physical: torch.Tensor,
        energy_available_wm2: torch.Tensor,
        water_available_mm_day: torch.Tensor,
    ) -> LossBreakdown:
        zero = out.pred.sum() * 0.0
        rec = masked_mse(out.pred, target_model_space)

        kl = zero
        if self.model_name in {"vae_lstm", "pg_vae_lstm"}:
            if any(v is None for v in (out.q_mu, out.q_logvar, out.p_mu, out.p_logvar)):
                raise RuntimeError("VAE model output is missing posterior/prior parameters")
            kl = conditional_gaussian_kl(
                out.q_mu, out.q_logvar, out.p_mu, out.p_logvar, free_bits=self.free_bits
            )

        energy = water = nonneg = physics = zero
        if self.model_name == "pg_vae_lstm":
            terms = []
            e_cfg = self.cfg["loss"]["energy_term"]
            if bool(e_cfg.get("enabled", True)):
                energy = energy_consistency_loss(
                    pred_physical,
                    energy_available_wm2,
                    latent_heat_j_per_kg=float(e_cfg["latent_heat_j_per_kg"]),
                    mode=str(e_cfg["mode"]),
                )
                terms.append(energy)
            if bool(self.cfg["loss"]["water_term"].get("enabled", True)):
                water = water_consistency_loss(pred_physical, water_available_mm_day)
                terms.append(water)
            if bool(self.cfg["loss"]["nonnegative_term"].get("enabled", True)):
                nonneg = nonnegative_loss(pred_physical)
                terms.append(nonneg)

            if terms:
                stack = torch.stack(terms)
                if bool(self.cfg["loss"].get("physical_equal_weight_average", True)):
                    physics = stack.mean()
                else:
                    physics = stack.sum()

        total = rec + self.beta * kl + self.gamma * physics
        return LossBreakdown(total, rec, kl, physics, energy, water, nonneg)
