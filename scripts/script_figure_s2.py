#!/usr/bin/env python
"""
Supplementary Figure S2 -- Climatology at three probe locations vs 2023.

Each panel shows the SIC time series at one probe location:
  - Grey lines: individual years 1989-2022
  - Blue line + shading: climatology mean +/- 1 standard error
  - Red line: 2023 observations
Small inset maps mark the probe positions on the Antarctic coastline.

Generates: figures/figure_s2_probes_climatology.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE, PROBES, YEAR_INDEX
from sic_dmd.plotting import plot_probe_inset


def main():
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    mask_land = r["mask_land"]
    probe_obs = r["probe_obs"]
    clim_probe_uq = r["clim_probe_uq"]

    n_clim_years = len(probe_obs["A"]) - 1   # years 0..N-2 used for climatology
    n_days = 365

    # Generic calendar axis (Jan 1 - Dec 31)
    dates = [datetime(2023, 1, 1) + timedelta(days=d) for d in range(n_days)]

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 2, width_ratios=[4, 1], wspace=0.12, hspace=0.35,
                           height_ratios=[1, 1, 1, 0.15])

    for row, (name, pt) in enumerate(PROBES.items()):
        ax_ts = fig.add_subplot(gs[row, 0])
        ax_map = fig.add_subplot(gs[row, 1])

        # Individual years in grey
        for yr in range(n_clim_years):
            ts = probe_obs[name][yr][:n_days]
            ax_ts.plot(dates[:len(ts)], ts, color="gray", alpha=0.15, lw=0.5)

        # Climatology mean +/- 1 SE
        clim_mean = clim_probe_uq[name].mean(axis=0)
        clim_se = clim_probe_uq[name].std(axis=0)
        ax_ts.fill_between(dates, clim_mean - clim_se, clim_mean + clim_se,
                           color="blue", alpha=0.4)
        ax_ts.plot(dates, clim_mean, color="blue", lw=1.5)

        # 2023 observations in red
        obs_2023 = probe_obs[name][YEAR_INDEX][:n_days]
        ax_ts.plot(dates[:len(obs_2023)], obs_2023, color="red", lw=0.8)

        ax_ts.set_ylabel("Sea Ice Concentration")
        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        if row == 2:
            ax_ts.set_xlabel("Date")

        plot_probe_inset(ax_map, mask_land, pt, name)

    # Legend at the bottom
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis("off")
    handles = [
        Line2D([0], [0], color="gray", lw=0.8, alpha=0.5,
               label="1989-2022"),
        Patch(facecolor="blue", alpha=0.4, edgecolor="blue",
              label="Climatology (mean of 1989-2022, +/-1 standard error shaded)"),
        Line2D([0], [0], color="red", lw=1.0, label="2023"),
    ]
    ax_leg.legend(handles=handles, loc="center", ncol=3, frameon=False,
                  fontsize=10)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_s2_probes_climatology.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
