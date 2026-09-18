"""Create a pooled density scatter plot from evaluate.py predictions."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from metrics import all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", help="predictions_test.npz from evaluate.py")
    p.add_argument("--out", default="density_scatter.png")
    args = p.parse_args()

    zf = np.load(args.predictions, allow_pickle=True)
    o = zf["observed"].reshape(-1)
    s = zf["predicted"].reshape(-1)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    met = all_metrics(o, s)
    xy = np.vstack([o, s])
    dens = gaussian_kde(xy)(xy)
    order = np.argsort(dens)
    o, s, dens = o[order], s[order], dens[order]
    # Relative density avoids an arbitrary multiplicative factor.
    dens = dens / np.nanmax(dens)

    slope, intercept = np.polyfit(o, s, 1)
    lim = max(float(np.nanmax(o)), float(np.nanmax(s)), 1.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(o, s, c=dens, s=2, cmap="viridis", rasterized=True)
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.2)
    xx = np.array([0.0, lim])
    ax.plot(xx, slope * xx + intercept, color="red", linewidth=1.2)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Observed ET (mm d$^{-1}$)", ylabel="Predicted ET (mm d$^{-1}$)")
    ax.text(0.04, 0.96, f"KGE={met['KGE']:.3f}\nRMSE={met['RMSE']:.3f}\nBias={met['Bias']:.3f}", transform=ax.transAxes, va="top")
    ax.text(0.50, 0.05, f"Y={slope:.3f}X+{intercept:.3f}", transform=ax.transAxes)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Relative density")
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
