#!/usr/bin/env python
"""
Figure 8 -- Combined panel: spatial SIC maps + probe time series.

Layout (3 rows x 3 columns):
    (a-c)  Observed SIC at three dates (Mar, Jun, Sep 2023)
    (d-f)  DMD-predicted SIC at the same dates
    (g-i)  Time series at three probe locations (A, B, C) with
           observations in red, DMD mean in black, and +/- 2 sigma
           uncertainty in grey.  Small inset maps show the probe positions.

Generates: figures/figure_8_prediction_combined.png
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE, PROBES, YEAR_INDEX
from sic_dmd.plotting import (
    day_index_to_date, day_of_year,
    plot_antarctic_map, plot_probe_inset,
)


def main():
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    X_true = r["X_test_true"]
    X_pred = r["X_test_mean_filt"]
    mask_land = r["mask_land"]
    probe_test = r["probe_test"]
    probe_obs = r["probe_obs"]

    # Dates for the spatial snapshots
    snapshot_days = [
        (day_of_year(3, 1), "01 Mar 2023"),
        (day_of_year(6, 1), "01 Jun 2023"),
        (day_of_year(9, 1), "01 Sep 2023"),
    ]

    n_days = 365
    dates = [day_index_to_date(t) for t in range(n_days)]

    # -- Build figure --
    fig = plt.figure(figsize=(14, 14))

    # Top 2 rows: 2x3 spatial maps.  Bottom row: 1x3 time-series with insets.
    outer = gridspec.GridSpec(2, 1, height_ratios=[2, 1.5], hspace=0.25)

    # --- Spatial maps (rows a-c, d-f) ---
    gs_maps = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0],
                                               wspace=0.05, hspace=0.15)
    labels_top = list("abc")
    labels_bot = list("def")
    for col, (day_idx, title) in enumerate(snapshot_days):
        # Observed
        ax_obs = fig.add_subplot(gs_maps[0, col])
        plot_antarctic_map(ax_obs, X_true[day_idx], mask_land, title=title)
        ax_obs.text(0.02, 0.95, f"({labels_top[col]})", transform=ax_obs.transAxes,
                    fontsize=11, va="top", fontweight="bold")
        if col == 0:
            ax_obs.set_ylabel("Observed", fontsize=13, labelpad=8)

        # Predicted
        ax_pred = fig.add_subplot(gs_maps[1, col])
        plot_antarctic_map(ax_pred, np.clip(X_pred[day_idx], 0, 1), mask_land)
        ax_pred.text(0.02, 0.95, f"({labels_bot[col]})", transform=ax_pred.transAxes,
                     fontsize=11, va="top", fontweight="bold")
        if col == 0:
            ax_pred.set_ylabel("Predicted (DMD)", fontsize=13, labelpad=8)

    # --- Probe time series (row g-i) ---
    gs_probes = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.35,
    )
    labels_probe = list("ghi")
    for col, (name, pt) in enumerate(PROBES.items()):
        # Each probe panel: time series on the left, small map inset on the right
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_probes[col], width_ratios=[3, 1], wspace=0.08,
        )
        ax_ts = fig.add_subplot(inner[0])
        ax_map = fig.add_subplot(inner[1])

        # Observations for 2023
        obs = probe_obs[name][YEAR_INDEX][:n_days]
        ax_ts.plot(dates[:len(obs)], obs, color="red", lw=0.8)

        # DMD ensemble mean +/- 2 sigma
        ens = probe_test[name][:, :n_days]
        m = ens.mean(axis=0)
        s = ens.std(axis=0)
        ax_ts.plot(dates, m, color="black", lw=1.2)
        ax_ts.fill_between(dates, m - 2 * s, m + 2 * s,
                           color="black", alpha=0.1)

        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_ts.tick_params(axis="x", rotation=30)
        ax_ts.set_ylabel("SIC")
        ax_ts.text(0.02, 0.95, f"({labels_probe[col]})",
                   transform=ax_ts.transAxes, fontsize=11, va="top",
                   fontweight="bold")

        # Inset map
        plot_probe_inset(ax_map, mask_land, pt, name)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_8_prediction_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
