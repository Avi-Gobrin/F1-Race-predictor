"""Figure generation for the report.

Uses a non-interactive backend so plots render in headless runs. Each function
saves one PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402


def plot_walk_forward_metrics(metrics: pd.DataFrame, path: Path | None = None) -> Path:
    """Grouped bars: model winner accuracy vs the pole-sitter baseline."""
    path = path or FIGURES_DIR / "walk_forward_accuracy.png"
    seasons = metrics["test_season"].astype(str)
    x = range(len(seasons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], metrics["winner_acc"], width, label="Model")
    ax.bar([i + width / 2 for i in x], metrics["baseline_acc"], width,
           label="Pole baseline")
    ax.plot(x, metrics["podium_prec"], "o-", color="black", label="Model podium prec@3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Test season")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Walk-forward winner accuracy vs pole-sitter baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_lasso_coefficients(coef: pd.Series, path: Path | None = None) -> Path:
    """Horizontal bar chart of the (standardized) LASSO coefficients."""
    path = path or FIGURES_DIR / "lasso_coefficients.png"
    coef = coef.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in coef]
    ax.barh(coef.index, coef.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized features)")
    ax.set_title("LASSO logistic coefficients (most recent split)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_factor_loadings(loadings: pd.DataFrame, path: Path | None = None) -> Path:
    """Heatmap of factor-analysis loadings (feature x factor)."""
    path = path or FIGURES_DIR / "factor_loadings.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(loadings.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(loadings.shape[1]))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(loadings.shape[0]))
    ax.set_yticklabels(loadings.index)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            ax.text(j, i, f"{loadings.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    ax.set_title("Factor analysis loadings")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
