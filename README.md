# SIC-DMD: Sea Ice Concentration Forecasting with Dynamic Mode Decomposition

Code to reproduce the figures in the paper.

## Quick start (just the figures)

```bash
conda env create -f environment.yml
conda activate sic_dmd
```

Download `precomputed_results.pkl` from the GitHub Release page and place
it in `data/`. Then run any figure script:

```bash
python scripts/script_figure_6.py
python scripts/script_figure_7.py
python scripts/script_figure_8.py
```

Output appears in `figures/`.

## Full reproduction from raw data

### 1. Obtain the data

Download the files listed below and place them under `data/`:

| File | Size | Source |
|------|------|--------|
| `precomputed_results.pkl` | 1.3 GB | GitHub Release (upload in progress) |
| `dum/year_2023/DMD_r_5_day_34_0_thin_2_win_7.pkl` | 616 MB | GitHub Release (upload in progress) |
| `Clean_Antarctic_data/Antarctic_years_1989_2024i.pkl` | 22 GB | [Dropbox](https://www.dropbox.com/scl/fi/brr8s8z9s5sty6h7gz89p/Antarctic_years_1989_2024i.pkl?rlkey=qu2olfboxy1yw7awaqavcklia&dl=0) |

> **Note:** The OSI SAF observation data is temporarily hosted on Dropbox via the
> link above. We will soon upload all data files to Zenodo and Hugging Face
> for permanent archival.

The precomputed file is enough to generate all figures. The DMD model file
is only needed if you want to rerun `precompute.py`. The raw observation
file is only needed if you want to retrain the DMD model from scratch.

### 2. Train DMD model (optional -- skip if you downloaded precomputed_results.pkl)

This loads the raw observations (~22 GB), builds a window-averaged training
set, and trains a 100-member bootstrap DMD ensemble via BOPDMD.
Requires at least 32 GB of RAM.

```bash
python scripts/train_dmd.py
```

Output: `data/dum/year_2023/DMD_r_5_day_34_0_thin_2_win_7.pkl` (~616 MB)

### 3. Precompute (optional -- skip if you downloaded precomputed_results.pkl)

This loads the raw observations and the DMD model from the previous step,
evaluates the ensemble on training and prediction windows, computes
climatology statistics, and writes `data/precomputed_results.pkl`.
Requires at least 32 GB of RAM.

```bash
python scripts/precompute.py
```

### 4. Generate figures

Each script loads the precomputed results and runs in a few seconds:

```bash
python scripts/script_figure_6.py     # Total ice coverage prediction
python scripts/script_figure_7.py     # Forecast MAE
python scripts/script_figure_8.py     # Combined spatial maps + probe time series
python scripts/script_figure_s1.py    # Supplementary: 2022 reconstruction
python scripts/script_figure_s2.py    # Supplementary: climatology at probes
```

## Figures

| Script | Paper figure | Description |
|--------|-------------|-------------|
| (co-author, to be added) | Figure 1 | Overview |
| (co-author, to be added) | Figure 2 | Interannual variability (mrCOSTS) |
| (co-author, to be added) | Figure 3 | Background mode spatial patterns |
| (co-author, to be added) | Figure 4 | mrCOSTS band amplitudes |
| (co-author, to be added) | Figure 5 | mrCOSTS point diagnosis |
| `script_figure_6.py` | Figure 6 | Total ice coverage: DMD vs climatology |
| `script_figure_7.py` | Figure 7 | Mean absolute error of DMD forecast |
| `script_figure_8.py` | Figure 8 | Observed vs predicted SIC maps + probe time series |
| `script_figure_s1.py` | Suppl. S1 | DMD reconstruction quality (2022) |
| `script_figure_s2.py` | Suppl. S2 | Climatology vs 2023 at probe locations |

## Repository structure

```
src/sic_dmd/           Python package with shared utilities
  config.py            All parameters, paths, and constants
  data_wrangle.py      Data loading, thinning, leap-year removal
  dmd_routines.py      DMD training, evaluation, bootstrap
  plotting.py          Shared plotting helpers (colormaps, maps)
scripts/               One script per figure + DMD training + precompute
data/                  Input data (not tracked in git)
figures/               Generated figures (not tracked in git)
```
