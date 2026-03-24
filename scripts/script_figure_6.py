#!/usr/bin/env python
"""
Figure 6 -- Total ice coverage: DMD prediction vs climatology vs observations.

Generates: figures/figure_6_total_ice_prediction.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE
from sic_dmd.plotting import day_index_to_date


def main():
    # Load precomputed results
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    t_train = r["t_train"]
    t_test = r["t_test"]
    ice = r["ice_integral"]
    clim_int = r["climatology_integral"]
    clim_std = r["clim_stddev"]
    scale = r["area_scale"]

    dates_train = [day_index_to_date(t) for t in t_train]
    dates_test = [day_index_to_date(t) for t in t_test]
    n_true = len(ice["integral_true_test"])

    # Uncertainty bands (2 standard deviations)
    ice_std_test = ice["integral_test_filt"].std(axis=0)
    ice_std_train = ice["integral_train"][ice["keep_mask"]].std(axis=0)

    # Tile the 365-day climatology to cover the 730-day train/test windows
    clim_train = np.tile(scale * clim_int, 2)
    clim_test = np.tile(scale * clim_int, 2)
    clim_std_train = np.tile(scale * clim_std, 2)
    clim_std_test = np.tile(scale * clim_std, 2)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(15, 5))

    # Vertical line at the training / forecast boundary
    ax.axvline(day_index_to_date(0), ls="--", color="black", lw=1)

    # DMD uncertainty (grey shading, +/- 2 sigma)
    ax.fill_between(dates_test,
                     scale * (ice["integral_test_mean"] - 2 * ice_std_test),
                     scale * (ice["integral_test_mean"] + 2 * ice_std_test),
                     color="black", alpha=0.2, edgecolor=None)
    ax.fill_between(dates_train,
                     scale * (ice["integral_train_mean_filt"] - 2 * ice_std_train),
                     scale * (ice["integral_train_mean_filt"] + 2 * ice_std_train),
                     color="black", alpha=0.2, edgecolor=None)

    # Climatology uncertainty (blue shading)
    ax.fill_between(dates_test,
                     clim_test - 2 * clim_std_test,
                     clim_test + 2 * clim_std_test,
                     color="blue", alpha=0.2, edgecolor=None)
    ax.fill_between(dates_train,
                     clim_train - 2 * clim_std_train,
                     clim_train + 2 * clim_std_train,
                     color="blue", alpha=0.2, edgecolor=None)

    # Ground truth (red)
    ax.plot(dates_train, scale * ice["integral_true_train"],
            color="red", label="Ground Truth")
    ax.plot(dates_test[:n_true], scale * ice["integral_true_test"],
            color="red")

    # DMD mean (black)
    ax.plot(dates_train, scale * ice["integral_train_mean_filt"],
            color="black")
    ax.plot(dates_test, scale * ice["integral_test_mean"],
            color="black", label="DMD prediction")

    # Climatology mean (blue)
    ax.plot(dates_train, clim_train, color="blue")
    ax.plot(dates_test, clim_test, color="blue", label="Climatology prediction")

    ax.set_ylabel("Total Ice Coverage")
    ax.set_xlabel("Date of forecast or reconstruction")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M" if v else "0")
    )
    ax.grid(True, alpha=0.3)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_6_total_ice_prediction.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
