#!/usr/bin/env python
"""
Figure 7 -- Mean absolute error of the DMD sea-ice concentration forecast.

Generates: figures/figure_7_forecast_mae.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE
from sic_dmd.plotting import day_index_to_date, day_of_year


def main():
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    t_test = r["t_test"]
    X_pred = r["X_test_mean"]       # ensemble-mean prediction
    X_true = r["X_test_true"]       # ground-truth observations
    n_test = X_true.shape[0]

    dates = [day_index_to_date(t) for t in t_test[:n_test]]

    # MAE at each time step, averaged over all spatial pixels
    mae = np.array([
        np.mean(np.abs(np.clip(X_pred[t], 0, 1) - X_true[t]))
        for t in range(n_test)
    ])

    # -- Plot --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, mae, color="C0", lw=1)

    # Visual guide lines (mentioned in the caption)
    ax.axvline(day_index_to_date(day_of_year(2, 14)),
               ls=":", color="gray", lw=0.8)
    ax.axvline(day_index_to_date(365),
               ls=":", color="gray", lw=0.8)

    ax.set_ylabel("Mean Absolute Error in Sea Ice Concentration")
    ax.set_xlabel("Forecast date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    fig.autofmt_xdate(rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_7_forecast_mae.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
