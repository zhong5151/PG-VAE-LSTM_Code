from __future__ import annotations
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange

from config import get_args
from data import ETSplitDataset, MinMaxScaler
from loss import CompositeLoss
from metrics import all_metrics
from model import build_model
from utils import ensure_dir, resolve_device, save_json, set_seed


def _to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def evaluate_loader(model, loader, loss_fn, scaler, cfg, device):
    model.eval()
    losses = []
    obs_all, sim_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(batch["x"], sample=False)
            if bool(cfg["data"]["normalize_target"]):
                pred_phys = scaler.inverse_y_torch(out.pred)
            else:
                pred_phys = out.pred
            bd = loss_fn(out, batch["y_model"], pred_phys, batch["energy_wm2"], batch["water_mm_day"])
            losses.append(bd.detached())
            obs_all.append(batch["y_physical"].detach().cpu().numpy().reshape(-1))
            sim_all.append(pred_phys.detach().cpu().numpy().reshape(-1))

    obs = np.concatenate(obs_all) if obs_all else np.array([])
    sim = np.concatenate(sim_all) if sim_all else np.array([])
    metrics = all_metrics(obs, sim)
    if losses:
        for key in losses[0]:
            metrics[f"loss_{key}"] = float(np.mean([x[key] for x in losses]))
    return metrics


def main(cfg: Dict):
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=bool(cfg["training"]["deterministic"]))
    device = resolve_device(str(cfg["experiment"]["device"]))
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])

    # Fit normalizer strictly on the training split.
    train_raw = ETSplitDataset(cfg, str(cfg["data"]["train_split"]), scaler=None)
    scaler_path = Path(cfg["data"]["scaler_file"])
    if scaler_path.exists():
        scaler = MinMaxScaler.load(scaler_path)
    else:
        scaler = MinMaxScaler.fit(train_raw.x, train_raw.y)
        scaler.save(scaler_path)

    train_ds = ETSplitDataset(cfg, str(cfg["data"]["train_split"]), scaler=scaler)
    val_ds = ETSplitDataset(cfg, str(cfg["data"]["val_split"]), scaler=scaler)

    batch_size = int(cfg["training"]["batch_size"])
    steps = int(cfg["training"]["steps_per_epoch"])
    sampler = RandomSampler(train_ds, replacement=True, num_samples=batch_size * steps)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["val_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    loss_fn = CompositeLoss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    maximize = bool(cfg["training"]["maximize_monitor"])
    best = -math.inf if maximize else math.inf
    patience = 0
    history = []
    ckpt_path = out_dir / "best.pt"

    for epoch in trange(1, int(cfg["training"]["epochs"]) + 1, desc="training"):
        model.train()
        running = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["x"], sample=True)
            pred_phys = scaler.inverse_y_torch(out.pred) if bool(cfg["data"]["normalize_target"]) else out.pred
            bd = loss_fn(out, batch["y_model"], pred_phys, batch["energy_wm2"], batch["water_mm_day"])
            bd.total.backward()
            clip = float(cfg["training"].get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running.append(bd.detached())

        val = evaluate_loader(model, val_loader, loss_fn, scaler, cfg, device)
        train_total = float(np.mean([x["total"] for x in running]))
        row = {"epoch": epoch, "train_total_loss": train_total, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)

        monitor = str(cfg["training"]["monitor"])
        if monitor == "val_total_loss":
            score = val["loss_total"]
        elif monitor == "val_rmse":
            score = val["RMSE"]
        elif monitor == "val_kge":
            score = val["KGE"]
        else:
            raise ValueError(f"Unsupported training.monitor={monitor}")

        improved = score > best + float(cfg["training"]["min_delta"]) if maximize else score < best - float(cfg["training"]["min_delta"])
        if improved:
            best = score
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "seed": seed,
                    "epoch": epoch,
                    "monitor": monitor,
                    "monitor_value": score,
                },
                ckpt_path,
            )
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch={epoch} train_loss={train_total:.6f} "
                f"val_R2={val['R2']:.4f} val_KGE={val['KGE']:.4f} "
                f"val_RMSE={val['RMSE']:.4f} val_Bias={val['Bias']:.4f}"
            )

        if patience >= int(cfg["training"]["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best {monitor}={best:.6f}")
            break

    save_json({"history": history, "best_monitor_value": best}, out_dir / "training_history.json")
    print(f"Saved best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main(get_args("Train PG-VAE-LSTM and ablation models"))
