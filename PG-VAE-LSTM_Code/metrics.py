"""Evaluation metrics used in the manuscript."""
from __future__ import annotations

from typing import Dict
import numpy as np


def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float).reshape(-1)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def coefficient_of_determination(obs, sim) -> float:
    o, s = _paired(obs, sim)
    if o.size < 2:
        return np.nan
    sst = np.sum((o - np.mean(o)) ** 2)
    if np.isclose(sst, 0.0):
        return np.nan
    return float(1.0 - np.sum((o - s) ** 2) / sst)


def rmse(obs, sim) -> float:
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if o.size else np.nan


def bias(obs, sim) -> float:
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if o.size else np.nan


def kge(obs, sim) -> float:
    o, s = _paired(obs, sim)
    if o.size < 2 or np.isclose(np.std(o), 0.0) or np.isclose(np.mean(o), 0.0):
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def pearson_r(obs, sim) -> float:
    o, s = _paired(obs, sim)
    if o.size < 2 or np.isclose(np.std(o), 0.0) or np.isclose(np.std(s), 0.0):
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def all_metrics(obs, sim) -> Dict[str, float]:
    o, s = _paired(obs, sim)
    return {
        "n": int(o.size),
        "R2": coefficient_of_determination(o, s),
        "KGE": kge(o, s),
        "RMSE": rmse(o, s),
        "Bias": bias(o, s),
        "Pearson_r": pearson_r(o, s),
    }
