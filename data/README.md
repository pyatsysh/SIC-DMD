# Data

This directory should contain the following files.
See the release page on GitHub for download links.

```
data/
  precomputed_results.pkl                          (~1.3 GB, from GitHub Release)
  Clean_Antarctic_data/
    Antarctic_years_1989_2024i.pkl                 (~22 GB, from Zenodo)
  dum/
    year_2023/
      DMD_r_5_day_34_0_thin_2_win_7.pkl            (~616 MB, from GitHub Release or train_dmd.py)
```

**Quick start (figures only):** download `precomputed_results.pkl` and run
the `script_figure_*.py` scripts directly -- no need for the raw observations.

**Full reproduction:** download the raw observations, then run
`train_dmd.py` followed by `precompute.py`.
