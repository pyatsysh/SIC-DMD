#!/usr/bin/env python
"""
Supplementary Figure S1 -- DMD reconstruction of 2022 SIC configurations.

Shows observed vs. DMD-reconstructed spatial SIC patterns at three dates
during the training period (Mar, Jun, Sep 2022).

Generates: figures/figure_s1_reconstruction_2022.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sic_dmd.config import FIGURE_DIR, PRECOMPUTED_FILE
from sic_dmd.plotting import day_of_year, plot_antarctic_map


def main():
    with open(PRECOMPUTED_FILE, "rb") as fh:
        r = pickle.load(fh)

    X0 = r["X0"]                         # observed training data
    X_recon = r["X_train_mean_filt"]      # DMD reconstruction
    mask_land = r["mask_land"]

    # 2022 is the second year of the two-year training window.
    # Training indices: 0 = Jan 1 2021, so 2022 dates start at index 365.
    targets = [
        (365 + day_of_year(3, 1), "01 Mar 2022"),
        (365 + day_of_year(6, 1), "01 Jun 2022"),
        (365 + day_of_year(9, 1), "01 Sep 2022"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (idx, title) in enumerate(targets):
        plot_antarctic_map(axes[0, col], X0[idx], mask_land, title=title)
        plot_antarctic_map(axes[1, col], np.clip(X_recon[idx], 0, 1), mask_land)

    axes[0, 0].set_ylabel("Observed", fontsize=14, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Reconstructed (DMD)", fontsize=14, rotation=90,
                          labelpad=10)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out = os.path.join(FIGURE_DIR, "figure_s1_reconstruction_2022.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
