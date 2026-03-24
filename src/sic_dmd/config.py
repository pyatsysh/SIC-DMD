"""
Shared configuration for the SIC-DMD figure generation pipeline.

All parameters here match the pre-trained DMD model and the analysis
described in the paper. Change these only if you retrain the model
or switch datasets.
"""

import os

# ---------------------------------------------------------------------------
# DMD model parameters
# ---------------------------------------------------------------------------
YEAR_INDEX = 34         # Year index in the dataset (0-based from 1989), so 34 = 2023
DAY_INDEX = 0           # Day-of-year index for forecast start (0 = January 1)
WINDOW = 7              # Temporal window for running-mean smoothing (days)
THIN = 2                # Spatial thinning factor applied to raw 432x432 grid
RANK = 5                # Number of DMD modes retained
THRESHOLD = 0.3         # Frobenius-norm threshold for filtering bootstrap members
T_PRED = 730            # Prediction horizon in days (2 years)
T_TRAIN = 365 * 2       # Training window length in days (2 years)
N_BOOTSTRAP = 100       # Number of bootstrap DMD ensemble members
N_BOOTSTRAP_CLIM = 100  # Number of bootstrap resamples for climatology uncertainty
TIME_DELAY = 2          # Time-delay embedding dimension for DMD
EIG_CONSTRAINTS = {"stable", "conjugate_pairs"}  # BOPDMD eigenvalue constraints

# Derived
REF_YEAR = 1989 + YEAR_INDEX   # Calendar year corresponding to t = 0

# Area-scaling factor: converts the normalised spatial integral to km^2.
# The raw grid is 512 pixels at 25 km resolution; after thinning each
# pixel represents (25 * THIN) km, and there are (512 // THIN) pixels
# per side.
AREA_SCALE = (512 // THIN) ** 2 * 25 ** 2 * THIN ** 2

# ---------------------------------------------------------------------------
# Probe locations  (row, col in the thinned 216x216 grid)
# ---------------------------------------------------------------------------
PROBES = {
    "A": (140, 62),    # Weddell Sea  -- high year-round SIC
    "B": (85, 162),    # East Antarctic -- seasonal (ice-free in summer)
    "C": (70, 90),     # Ross Sea -- strong seasonal cycle
}

# ---------------------------------------------------------------------------
# Paths  (relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURE_DIR = os.path.join(REPO_ROOT, "figures")
PRECOMPUTED_FILE = os.path.join(DATA_DIR, "precomputed_results.pkl")

# Raw observation data
OBS_DATA_PATH = os.path.join(
    DATA_DIR, "Clean_Antarctic_data", "Antarctic_years_1989_2024i.pkl"
)

# Pre-trained DMD model
DMD_MODEL_PATH = os.path.join(
    DATA_DIR, "dum", f"year_{REF_YEAR}",
    f"DMD_r_{RANK}_day_{YEAR_INDEX}_{DAY_INDEX}_thin_{THIN}_win_{WINDOW}.pkl",
)
