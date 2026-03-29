"""Visualization helpers for pairs trading EDA."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_correlation_histogram(
    all_correlations: np.ndarray,
    *,
    cutoff: Optional[float] = None,
    bins: int = 80,
    title: str = "Pairwise Returns Correlation Distribution (Phase 1)",
) -> tuple[Figure, Axes]:
    """Histogram of all pairwise return correlations with optional cutoff line.

    Shows the full distribution of daily-returns correlations across all
    ticker pairs, with a vertical line marking the candidate selection
    threshold. The number of pairs above the cutoff is computed and
    displayed in the legend automatically.

    See compute_pairwise_return_correlations() for what r values mean
    and typical ranges used in pairs trading.

    Args:
        all_correlations: 1-D array of pairwise correlation values
            (output of compute_pairwise_return_correlations).
        cutoff: Correlation threshold for the dashed vertical line.
            Pairs above this are considered candidates. None = no line.
        bins: Number of histogram bins. Default 80.
        title: Plot title.

    Returns:
        (fig, ax) tuple for further customization if needed.

    Example:
        >>> all_corr = compute_pairwise_return_correlations(data_1d, end=p1_end)
        >>> fig, ax = plot_correlation_histogram(all_corr, cutoff=0.40)
    """
    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)

    all_correlations = all_correlations[~np.isnan(all_correlations)]

    ax.hist(
        all_correlations,
        bins=bins,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
    )

    if cutoff is not None:
        # Count how many pairs fall at or above the cutoff directly from the data
        num_above = int(np.sum(all_correlations >= cutoff))
        ax.axvline(
            cutoff,
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label=f"Candidate cutoff (r={cutoff:.3f}, {num_above} pairs)",
        )
        ax.legend(fontsize=10)

    ax.set_xlabel("Returns Correlation", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return fig, ax
