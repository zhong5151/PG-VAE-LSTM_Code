"""Memory-efficient data loading and train-only normalization.

Expected split layout (train/val/test):
  X.npy                         float32 [N, T, F]
  y.npy                         float32 [N] or [N,1]
  site_id.npy                   strings/objects [N] (optional but recommended)
  energy_available_wm2.npy      float32 [N] (required for PG energy loss)
  water_available_mm_day.npy    float32 [N] (required for PG water loss)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MinMaxScaler:
    x_min: np.ndarray
    x_max: np.ndarray
    x_mean: np.ndarray
    y_min: float
    y_max: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, chunk_size: int = 2048) -> "MinMaxScaler":
        f = x.shape[-1]
        xmin = np.full(f, np.inf, dtype=np.float64)
        xmax = np.full(f, -np.inf, dtype=np.float64)
        xsum = np.zeros(f, dtype=np.float64)
        xcount = np.zeros(f, dtype=np.int64)
        for start in range(0, len(x), chunk_size):
            chunk = np.asarray(x[start:start + chunk_size], dtype=np.float64).reshape(-1, f)
            with np.errstate(all="ignore"):
                cmin = np.nanmin(chunk, axis=0)
                cmax = np.nanmax(chunk, axis=0)
            valid = np.isfinite(chunk)
            vals = np.where(valid, chunk, 0.0)
            xmin = np.minimum(xmin, np.where(np.isfinite(cmin), cmin, xmin))
            xmax = np.maximum(xmax, np.where(np.isfinite(cmax), cmax, xmax))
            xsum += vals.sum(axis=0)
            xcount += valid.sum(axis=0)
        xmin[~np.isfinite(xmin)] = 0.0
        xmax[~np.isfinite(xmax)] = 1.0
        xmean = np.divide(xsum, np.maximum(xcount, 1))

        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(yy)
        if not finite.any():
            raise ValueError("Training target contains no finite values")
        ymin = float(np.nanmin(yy))
        ymax = float(np.nanmax(yy))
        if np.isclose(ymax, ymin):
            ymax = ymin + 1.0
        return cls(xmin, xmax, xmean, ymin, ymax)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x_min=self.x_min.astype(np.float32),
            x_max=self.x_max.astype(np.float32),
            x_mean=self.x_mean.astype(np.float32),
            y_min=np.float32(self.y_min),
            y_max=np.float32(self.y_max),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScaler":
        z = np.load(path)
        return cls(z["x_min"], z["x_max"], z["x_mean"], float(z["y_min"]), float(z["y_max"]))

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = self.x_mean.astype(np.float32)
        x = np.where(np.isfinite(x), x, mean)
        denom = (self.x_max - self.x_min).astype(np.float32)
        denom[np.isclose(denom, 0.0)] = 1.0
        return (x - self.x_min.astype(np.float32)) / denom

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.y_min) / (self.y_max - self.y_min)

    def inverse_y_np(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) * (self.y_max - self.y_min) + self.y_min

    def inverse_y_torch(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.y_max - self.y_min) + self.y_min


class ETSplitDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, scaler: MinMaxScaler | None = None):
        dcfg = cfg["data"]
        root = Path(dcfg["root"]) / split
        self.x = np.load(root / dcfg["x_file"], mmap_mode="r")
        self.y = np.load(root / dcfg["y_file"], mmap_mode="r")
        if self.x.ndim != 3:
            raise ValueError(f"{root / dcfg['x_file']} must have shape [N,T,F], got {self.x.shape}")
        if self.x.shape[1] != int(dcfg["seq_len"]):
            raise ValueError(f"Expected seq_len={dcfg['seq_len']}, got {self.x.shape[1]}")
        if self.x.shape[2] != int(dcfg["input_size"]):
            raise ValueError(f"Expected input_size={dcfg['input_size']}, got {self.x.shape[2]}")
        if len(self.x) != len(self.y):
            raise ValueError("X and y have different sample counts")

        site_path = root / dcfg["site_file"]
        self.site = np.load(site_path, allow_pickle=True) if site_path.exists() else np.arange(len(self.x)).astype(str)

        epath = root / dcfg["energy_file"]
        wpath = root / dcfg["water_file"]
        self.energy = np.load(epath, mmap_mode="r") if epath.exists() else None
        self.water = np.load(wpath, mmap_mode="r") if wpath.exists() else None
        self.scaler = scaler
        self.normalize_inputs = bool(dcfg["normalize_inputs"])
        self.normalize_target = bool(dcfg["normalize_target"])

        if str(cfg["model"]["name"]).lower() == "pg_vae_lstm":
            if cfg["loss"]["energy_term"].get("enabled", True) and self.energy is None:
                raise FileNotFoundError(f"PG-VAE-LSTM requires {epath}")
            if cfg["loss"]["water_term"].get("enabled", True) and self.water is None:
                raise FileNotFoundError(f"PG-VAE-LSTM requires {wpath}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        y = np.asarray(self.y[idx], dtype=np.float32).reshape(-1)[0]
        if self.scaler is not None:
            if self.normalize_inputs:
                x = self.scaler.transform_x(x)
            if self.normalize_target and np.isfinite(y):
                y_model = self.scaler.transform_y(np.array([y], dtype=np.float32))[0]
            else:
                y_model = y
        else:
            y_model = y

        energy = np.nan if self.energy is None else float(np.asarray(self.energy[idx]).reshape(-1)[0])
        water = np.nan if self.water is None else float(np.asarray(self.water[idx]).reshape(-1)[0])
        return {
            "x": torch.from_numpy(x),
            "y_model": torch.tensor([y_model], dtype=torch.float32),
            "y_physical": torch.tensor([y], dtype=torch.float32),
            "energy_wm2": torch.tensor([energy], dtype=torch.float32),
            "water_mm_day": torch.tensor([water], dtype=torch.float32),
            "site_id": str(self.site[idx]),
            "index": int(idx),
        }
