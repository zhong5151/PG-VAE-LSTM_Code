from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import build_parser, namespace_to_config
from data import ETSplitDataset, MinMaxScaler
from metrics import all_metrics
from model import build_model
from utils import ensure_dir, resolve_device, save_json, set_seed


def _physics_violation_summary(pred, energy, water, latent_heat_j_per_kg=2.45e6):
    pred = np.asarray(pred, dtype=float)
    out = {}
    valid = np.isfinite(pred)
    out["negative_rate"] = float(np.mean(pred[valid] < 0)) if valid.any() else np.nan
    e = np.asarray(energy, dtype=float)
    me = valid & np.isfinite(e)
    pred_flux = pred * latent_heat_j_per_kg / 86400.0
    out["energy_violation_rate"] = float(np.mean(pred_flux[me] > e[me])) if me.any() else np.nan
    w = np.asarray(water, dtype=float)
    mw = valid & np.isfinite(w)
    out["water_violation_rate"] = float(np.mean(pred[mw] > w[mw])) if mw.any() else np.nan
    return out


def main(cfg: Dict, checkpoint: str | None = None):
    device = resolve_device(str(cfg["experiment"]["device"]))
    set_seed(int(cfg["experiment"]["seed"]), deterministic=True)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    ckpt_path = Path(checkpoint) if checkpoint else out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Use checkpoint config so architecture and loss-independent evaluation settings are reproducible.
    train_cfg = ckpt["config"]
    # Preserve current data paths/output path if the repository is relocated.
    train_cfg["data"]["root"] = cfg["data"]["root"]
    train_cfg["data"]["scaler_file"] = cfg["data"]["scaler_file"]
    train_cfg["experiment"]["output_dir"] = cfg["experiment"]["output_dir"]
    train_cfg["experiment"]["device"] = cfg["experiment"]["device"]

    scaler = MinMaxScaler.load(train_cfg["data"]["scaler_file"])
    ds = ETSplitDataset(train_cfg, str(train_cfg["data"]["test_split"]), scaler=scaler)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(train_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    obs, pred, sites, energy, water = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x, sample=False)
            pred_phys = scaler.inverse_y_torch(out.pred) if bool(train_cfg["data"]["normalize_target"]) else out.pred
            obs.append(batch["y_physical"].numpy().reshape(-1))
            pred.append(pred_phys.cpu().numpy().reshape(-1))
            sites.extend(batch["site_id"])
            energy.append(batch["energy_wm2"].numpy().reshape(-1))
            water.append(batch["water_mm_day"].numpy().reshape(-1))

    obs = np.concatenate(obs)
    pred = np.concatenate(pred)
    energy = np.concatenate(energy)
    water = np.concatenate(water)
    sites = np.asarray(sites, dtype=str)

    pooled = all_metrics(obs, pred)
    pooled.update(_physics_violation_summary(pred, energy, water, float(train_cfg["loss"]["energy_term"]["latent_heat_j_per_kg"])))
    save_json(pooled, out_dir / "metrics_pooled.json")

    rows = []
    min_n = int(cfg["evaluation"]["min_site_samples"])
    for site in np.unique(sites):
        m = sites == site
        if np.sum(np.isfinite(obs[m]) & np.isfinite(pred[m])) < min_n:
            continue
        row = {"site_id": site, **all_metrics(obs[m], pred[m])}
        rows.append(row)
    site_df = pd.DataFrame(rows)
    site_df.to_csv(out_dir / "metrics_site.csv", index=False)
    if len(site_df):
        means = {k: float(site_df[k].mean()) for k in ["R2", "KGE", "RMSE", "Bias"]}
        means["n_sites"] = int(len(site_df))
    else:
        means = {"R2": np.nan, "KGE": np.nan, "RMSE": np.nan, "Bias": np.nan, "n_sites": 0}
    save_json(means, out_dir / "metrics_site_mean.json")

    if bool(cfg["evaluation"]["save_predictions"]):
        np.savez_compressed(
            out_dir / "predictions_test.npz",
            observed=obs.astype(np.float32),
            predicted=pred.astype(np.float32),
            site_id=sites,
            energy_available_wm2=energy.astype(np.float32),
            water_available_mm_day=water.astype(np.float32),
        )

    print("Pooled:", pooled)
    print("Mean site-level:", means)


if __name__ == "__main__":
    parser = build_parser("Evaluate PG-VAE-LSTM")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    cfg = namespace_to_config(args)
    main(cfg, checkpoint=args.checkpoint)
