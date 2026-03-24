"""
Shared plotting utilities for the SIC-DMD paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

from .config import REF_YEAR

# Reference date: day index 0 corresponds to January 1 of this year.
REF_DATE = datetime(REF_YEAR, 1, 1)

# Colormap: dark navy (open ocean, SIC = 0) to white (full ice, SIC = 1).
ICE_CMAP = LinearSegmentedColormap.from_list(
    "ocean_ice", ["#1B2444", "#FFFFFF"], N=256
)


def day_index_to_date(t):
    """Convert a day index (t=0 is Jan 1 of the reference year) to a datetime."""
    return REF_DATE + timedelta(days=int(t))


def day_of_year(month, day):
    """Return the 0-based day-of-year for a given month and day (non-leap year)."""
    return (datetime(2023, month, day) - datetime(2023, 1, 1)).days


def plot_antarctic_map(ax, sic, mask_land, title=None):
    """
    Plot a single Antarctic SIC snapshot on the given axes.

    Ocean pixels are coloured from dark blue (SIC=0) to white (SIC=1).
    Land pixels are shown in grey.
    """
    # SIC field with land masked out
    ocean = np.where(~mask_land, sic, np.nan)
    ax.imshow(ocean, origin="lower", cmap=ICE_CMAP, vmin=0, vmax=1)

    # Land overlay in grey
    land = np.where(mask_land, 0.5, np.nan)
    ax.imshow(land, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=1)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)


def plot_probe_inset(ax, mask_land, row_col, label):
    """
    Draw a small Antarctic coastline map with a probe location marked.

    Parameters
    ----------
    ax : matplotlib Axes
    mask_land : 2-D bool array
    row_col : tuple (row, col) in the grid
    label : str, e.g. "A"
    """
    ax.contour(mask_land.astype(float), levels=[0.5],
               colors="purple", linewidths=0.8)
    ax.scatter(row_col[1], row_col[0],
               marker="x", color="purple", s=40, zorder=5)
    ax.text(row_col[1] + 5, row_col[0] + 5, label,
            color="purple", fontsize=12, fontweight="bold")
    ax.set_xlim(0, mask_land.shape[1])
    ax.set_ylim(0, mask_land.shape[0])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
