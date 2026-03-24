#!/usr/bin/env python
"""
Step 1: Load raw data, evaluate the DMD ensemble, and save results.

This script does all the heavy computation (loading 21 GB of observations,
evaluating 100 bootstrap DMD members) and writes the results to a single
file that the individual figure scripts can load in seconds.

Usage
-----
    conda activate sic_dmd
    python scripts/precompute.py

Requirements
------------
    - At least 32 GB RAM (the raw observation file is ~21 GB).
    - Data files present in data/ (see README).

Output
------
    data/precomputed_results.pkl  (~1 GB)
"""

import os
import sys
import gc
import pickle
import dill
import numpy as np
from tqdm import trange

# Make the sic_dmd package importable from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sic_dmd.config import (
    YEAR_INDEX, DAY_INDEX, WINDOW, THIN, RANK,
    THRESHOLD, T_PRED, N_BOOTSTRAP_CLIM,
    AREA_SCALE, PROBES, REF_YEAR,
    DATA_DIR, OBS_DATA_PATH, DMD_MODEL_PATH, PRECOMPUTED_FILE,
)
from sic_dmd.data_wrangle import thin_data, del_leap, get_test_set
from sic_dmd.dmd_routines import eval_dmd


# -----------------------------------------------------------------------
# 1. Load and preprocess observations
# -----------------------------------------------------------------------
def load_observations():
    print(f"Loading observations from {OBS_DATA_PATH} ...")
    with open(OBS_DATA_PATH, "rb") as fh:
        mask_land_raw, mask_ice_raw, data_raw, _, _, x_raw, y_raw = dill.load(fh)

    print("  Spatial thinning ...")
    data, x, y, mask_ice, mask_land = thin_data(
        THIN, data_raw, x_raw, y_raw, mask_ice_raw, mask_land_raw,
    )
    # Copy so that the full-resolution arrays can be garbage-collected.
    data = [d.copy() for d in data]
    mask_ice = mask_ice.copy()
    mask_land = mask_land.copy()
    x = x.copy()
    y = y.copy()
    del data_raw, mask_land_raw, mask_ice_raw, x_raw, y_raw
    gc.collect()

    print("  Removing leap days ...")
    data = del_leap(data)
    print(f"  {len(data)} years, each {data[0].shape}")
    return data, x, y, mask_ice, mask_land


# -----------------------------------------------------------------------
# 2. Load the pre-trained bootstrap DMD model
# -----------------------------------------------------------------------
def load_dmd_model():
    print(f"Loading DMD model from {DMD_MODEL_PATH} ...")
    with open(DMD_MODEL_PATH, "rb") as fh:
        (X0, mask_ice, t_train, T_train, x, y,
         n_boot, rank, eig_constraints,
         L_s, Psi_s, bn_s) = pickle.load(fh)
    print(f"  {n_boot} bootstrap members, rank {rank}, T_train {T_train}")
    return X0, mask_ice, t_train, x, y, L_s, Psi_s, bn_s


# -----------------------------------------------------------------------
# 3. Climatology (multi-year mean + bootstrap uncertainty)
# -----------------------------------------------------------------------
def compute_climatology(data, x, y):
    n_years = len(data) - 1        # exclude the last (forecast) year
    ny, nx = data[0].shape[1:]

    print("Computing climatology mean ...")
    climatology = np.zeros((365, ny, nx))
    for yr in range(n_years):
        climatology += data[yr]
    climatology /= n_years

    climatology_integral = np.trapz(np.trapz(climatology, y, axis=1), x, axis=1)

    print("Bootstrap climatology uncertainty ...")
    np.random.seed(42)
    boot_indices = np.random.randint(0, n_years, (N_BOOTSTRAP_CLIM, n_years))

    clim_integral_uq = np.zeros((N_BOOTSTRAP_CLIM, 365))
    clim_probe_uq = {name: np.zeros((N_BOOTSTRAP_CLIM, 365)) for name in PROBES}

    for b in trange(N_BOOTSTRAP_CLIM, desc="Climatology bootstrap"):
        sample_mean = np.zeros((365, ny, nx))
        for yr_idx in boot_indices[b]:
            sample_mean += data[yr_idx]
        sample_mean /= n_years

        clim_integral_uq[b] = np.trapz(
            np.trapz(sample_mean, y, axis=1), x, axis=1,
        )
        for name, pt in PROBES.items():
            clim_probe_uq[name][b] = sample_mean[:, pt[0], pt[1]]

    clim_stddev = clim_integral_uq.std(axis=0)
    return climatology, climatology_integral, clim_stddev, clim_probe_uq


# -----------------------------------------------------------------------
# 4. Evaluate the DMD bootstrap ensemble (memory-efficient)
# -----------------------------------------------------------------------
def evaluate_ensemble(L_s, Psi_s, bn_s, t_train, t_test, x, y, X0):
    n_ens = L_s.shape[0]
    ny, nx = Psi_s.shape[2], Psi_s.shape[3]
    n_train = len(t_train)
    n_test = len(t_test)

    # -- True training integral --
    integral_true_train = np.trapz(np.trapz(X0, x, axis=2), y, axis=1)

    # -- Pass 1: evaluate every member on the training window --
    print("Evaluating ensemble on training window ...")
    X0_flat = X0.reshape(n_train, -1)
    X0_norm = np.linalg.norm(X0_flat)

    norms = np.zeros(n_ens)
    integral_train = np.zeros((n_ens, n_train))
    X_train_mean = np.zeros((n_train, ny, nx))

    for i in trange(n_ens, desc="Train pass"):
        Xi = eval_dmd(L_s[i], Psi_s[i], bn_s[i], t_train, isPositive=True)
        X_train_mean += Xi
        norms[i] = np.linalg.norm(Xi.reshape(n_train, -1) - X0_flat) / X0_norm
        for j in range(n_train):
            integral_train[i, j] = np.trapz(np.trapz(Xi[j], x, axis=1), y)
        del Xi

    X_train_mean /= n_ens
    keep = norms < THRESHOLD
    n_kept = int(keep.sum())
    print(f"  {n_kept}/{n_ens} members pass threshold {THRESHOLD}")

    # If not all pass, recompute the filtered training mean.
    if keep.all():
        X_train_mean_filt = X_train_mean
    else:
        X_train_mean_filt = np.zeros((n_train, ny, nx))
        cnt = 0
        for i in range(n_ens):
            if not keep[i]:
                continue
            Xi = eval_dmd(L_s[i], Psi_s[i], bn_s[i], t_train, isPositive=True)
            X_train_mean_filt += Xi
            cnt += 1
            del Xi
        X_train_mean_filt /= max(cnt, 1)

    # -- Pass 2: evaluate every member on the prediction window --
    print("Evaluating ensemble on prediction window ...")
    X_test_mean = np.zeros((n_test, ny, nx))
    X_test_mean_filt = np.zeros((n_test, ny, nx))
    integral_test_filt = []
    probe_test = {name: np.zeros((n_ens, n_test)) for name in PROBES}
    cnt_filt = 0

    for i in trange(n_ens, desc="Test pass"):
        Xi = eval_dmd(L_s[i], Psi_s[i], bn_s[i], t_test, isPositive=True)
        X_test_mean += Xi
        for name, pt in PROBES.items():
            probe_test[name][i] = Xi[:, pt[0], pt[1]]
        if keep[i]:
            X_test_mean_filt += Xi
            row = np.zeros(n_test)
            for j in range(n_test):
                row[j] = np.trapz(np.trapz(Xi[j], x, axis=1), y)
            integral_test_filt.append(row)
            cnt_filt += 1
        del Xi

    X_test_mean /= n_ens
    X_test_mean_filt /= max(cnt_filt, 1)
    integral_test_filt = np.array(integral_test_filt)

    ice_integral = {
        "keep_mask": keep,
        "integral_train": integral_train,
        "integral_true_train": integral_true_train,
        "integral_train_mean_filt": integral_train[keep].mean(axis=0),
        "integral_test_filt": integral_test_filt,
        "integral_test_mean": integral_test_filt.mean(axis=0),
        # integral_true_test is filled in by main()
    }

    return (ice_integral,
            X_test_mean, X_test_mean_filt, X_train_mean_filt,
            probe_test)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Precomputing results for SIC-DMD paper figures")
    print("=" * 60)

    # Observations
    data, x_data, y_data, mask_ice_data, mask_land = load_observations()

    # Climatology
    climatology, clim_integral, clim_stddev, clim_probe_uq = \
        compute_climatology(data, x_data, y_data)

    # DMD model (its stored x, y override the data coordinates for integration)
    X0, mask_ice, t_train, x, y, L_s, Psi_s, bn_s = load_dmd_model()
    t_test = np.arange(0, T_PRED)

    # Test-period ground truth
    print("Extracting test-period truth ...")
    X_test_true = get_test_set(data, YEAR_INDEX, DAY_INDEX, WINDOW, T_PRED)
    print(f"  shape {X_test_true.shape}")

    integral_true_test = np.zeros(X_test_true.shape[0])
    for i in range(X_test_true.shape[0]):
        integral_true_test[i] = np.trapz(np.trapz(X_test_true[i], x, axis=1), y)

    # DMD ensemble evaluation
    (ice_integral,
     X_test_mean, X_test_mean_filt, X_train_mean_filt,
     probe_test) = evaluate_ensemble(L_s, Psi_s, bn_s, t_train, t_test, x, y, X0)

    ice_integral["integral_true_test"] = integral_true_test

    # Observation time series at probe locations (for climatology figure)
    probe_obs = {}
    for name, pt in PROBES.items():
        probe_obs[name] = [data[yr][:, pt[0], pt[1]].copy()
                           for yr in range(len(data))]

    # -- Save everything --
    results = {
        "x": x,
        "y": y,
        "mask_land": mask_land,
        "mask_ice": mask_ice,
        "t_train": t_train,
        "t_test": t_test,
        "X0": X0,
        "X_test_true": X_test_true,
        "X_test_mean": X_test_mean,
        "X_test_mean_filt": X_test_mean_filt,
        "X_train_mean_filt": X_train_mean_filt,
        "ice_integral": ice_integral,
        "climatology_integral": clim_integral,
        "clim_stddev": clim_stddev,
        "clim_probe_uq": clim_probe_uq,
        "probe_test": probe_test,
        "probe_obs": probe_obs,
        "area_scale": AREA_SCALE,
    }

    os.makedirs(os.path.dirname(PRECOMPUTED_FILE), exist_ok=True)
    print(f"\nSaving results to {PRECOMPUTED_FILE} ...")
    with open(PRECOMPUTED_FILE, "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(PRECOMPUTED_FILE) / 1e6
    print(f"  Done ({size_mb:.0f} MB)")

    print("\n" + "=" * 60)
    print("Precomputation complete.  Now run the script_figure_*.py scripts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
