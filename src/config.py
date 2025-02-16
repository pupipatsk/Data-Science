"""
config.py

This module contains configuration settings for reproducibility and visualization customization.
"""

import matplotlib.pyplot as plt
from typing import Dict


class Config:
    """Configuration class"""

    SEED: int = 42

    PLOT_CONFIG: Dict[str, object] = {
        # Axes
        "axes.titlesize": 16,
        "axes.titlepad": 20,
        "axes.labelsize": 12,
        "axes.edgecolor": (0.1, 0.1, 0.1),
        "axes.labelcolor": (0.1, 0.1, 0.1),
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.grid": True,
        # Grid
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markeredgewidth": 0.0,
        # Scatter plot
        "scatter.marker": "x",
        # Ticks
        "xtick.labelsize": 12,
        "xtick.color": (0.1, 0.1, 0.1),
        "xtick.direction": "in",
        "ytick.labelsize": 12,
        "ytick.color": (0.1, 0.1, 0.1),
        "ytick.direction": "in",
        # Figure output
        "figure.figsize": (10, 6),
        "figure.dpi": 200,
        "savefig.dpi": 200,
        # Text
        "text.color": (0.2, 0.2, 0.2),
        # Font
        "font.family": ["serif", "Tahoma"],  # TH Font
    }

    @classmethod
    def apply_plot_config(cls) -> None:
        """Applies the matplotlib configuration settings."""
        plt.rcParams.update(cls.PLOT_CONFIG)


Config.apply_plot_config()
