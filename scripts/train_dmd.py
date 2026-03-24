#!/usr/bin/env python
"""
Train the bootstrap DMD ensemble model from raw observations.

This script loads the raw Antarctic SIC data, preprocesses it (spatial
thinning, leap-day removal, window averaging), constructs a time-delay
embedding, and trains a 100-member bootstrap DMD ensemble.  The resulting
model file is required by precompute.py.

Usage
-----
    conda activate sic_dmd
    python scripts/train_dmd.py

Requirements
------------
    - At least 32 GB RAM (the raw observation file is ~21 GB).
    - Raw data file present in data/ (see README).

Output
------
    data/dum/year_2023/DMD_r_5_day_34_0_thin_2_win_7.pkl  (~616 MB)
"""

import os
import sys
import gc
import pickle
import dill
import numpy as np

# Make the sic_dmd package importable from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sic_dmd.config import (
    YEAR_INDEX, DAY_INDEX, WINDOW, THIN, RANK,
    N_BOOTSTRAP, T_TRAIN, TIME_DELAY, EIG_CONSTRAINTS,
    DATA_DIR, OBS_DATA_PATH, DMD_MODEL_PATH,
)
from sic_dmd.data_wrangle import thin_data, del_leap, get_days_before, window_mean
from sic_dmd.dmd_routines import (
    reshape_data2dmd, bootstrap_train_dmd, reshape_Psi2data,
)


# -----------------------------------------------------------------------
# 1. Load and preprocess observations
# -----------------------------------------------------------------------
def load_observations():
    """Load raw Antarctic SIC data, thin spatially, remove leap days."""
    print(f"Loading observations from {OBS_DATA_PATH} ...")
    with open(OBS_DATA_PATH, "rb") as fh:
        mask_land_raw, mask_ice_raw, data_raw, _, _, x_raw, y_raw = dill.load(fh)

    print("  Spatial thinning ...")
    data, x, y, mask_ice, mask_land = thin_data(
        THIN, data_raw, x_raw, y_raw, mask_ice_raw, mask_land_raw,
    )
    # Copy so the full-resolution arrays can be garbage-collected.
    data = [d.copy() for d in data]
    mask_ice = mask_ice.copy()
    x = x.copy()
    y = y.copy()
    del data_raw, mask_land_raw, mask_ice_raw, x_raw, y_raw
    gc.collect()

    print("  Removing leap days ...")
    data = del_leap(data)
    print(f"  {len(data)} years, each {data[0].shape}")
    return data, x, y, mask_ice


# -----------------------------------------------------------------------
# 2. Build training data and train DMD
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Training bootstrap DMD ensemble model")
    print("=" * 60)

    # -- Load and preprocess --
    data, x, y, mask_ice = load_observations()
    ny, nx = data[0].shape[1:]

    # -- Window-averaged training snapshots --
    print(f"\nBuilding training set: {T_TRAIN} snapshots before "
          f"year_index={YEAR_INDEX}, day_index={DAY_INDEX} ...")
    X0_raw = get_days_before(data, YEAR_INDEX, DAY_INDEX, T_TRAIN + WINDOW - 1)
    X0 = window_mean(X0_raw, window=WINDOW)
    del X0_raw
    gc.collect()
    print(f"  X0 shape: {X0.shape}")

    # -- Time-delay embedding --
    t_train = np.arange(-T_TRAIN, 0)
    X_delayed, t_delayed, data_shape = reshape_data2dmd(
        X0, t_train, time_delay=TIME_DELAY, mask=mask_ice, isKeepFirstTimes=True,
    )
    print(f"  X_delayed shape: {X_delayed.shape}")

    # -- Bootstrap DMD training --
    print(f"\nTraining {N_BOOTSTRAP} bootstrap DMD members (rank {RANK}) ...")
    L_s, Psi_s_, bn_s = bootstrap_train_dmd(
        N_BOOTSTRAP, X_delayed, t_delayed,
        svd_rank=RANK, eig_constraints=EIG_CONSTRAINTS,
    )

    # Reshape spatial modes back to (ny, nx) grid
    Psi_s = np.zeros((N_BOOTSTRAP, RANK, ny, nx), dtype=complex)
    for i, Psi_ in enumerate(Psi_s_):
        Psi_s[i] = reshape_Psi2data(Psi_, data_shape, mask=mask_ice)

    # -- Save model --
    out_dir = os.path.dirname(DMD_MODEL_PATH)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(DMD_MODEL_PATH):
        print(f"\nWARNING: {DMD_MODEL_PATH} already exists, overwriting.")

    print(f"\nSaving model to {DMD_MODEL_PATH} ...")
    with open(DMD_MODEL_PATH, "wb") as fh:
        pickle.dump((
            X0, mask_ice, t_train, T_TRAIN, x, y,
            N_BOOTSTRAP, RANK, EIG_CONSTRAINTS,
            L_s, Psi_s, bn_s,
        ), fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(DMD_MODEL_PATH) / 1e6
    print(f"  Done ({size_mb:.0f} MB)")

    print("\n" + "=" * 60)
    print("Model saved.  Next step: python scripts/precompute.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
