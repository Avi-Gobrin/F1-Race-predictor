"""Project configuration: paths, seasons, and the walk-forward evaluation plan.

The central design choice of this project is *walk-forward* (expanding-window)
evaluation. We never let the model see the future. Each test season is
predicted using only the seasons strictly before it:

    train 2023            -> predict 2024
    train 2023, 2024      -> predict 2025
    train 2023, 2024, 2025 -> predict 2026

This mirrors how a forecaster would actually have operated at the start of each
season, and it is the honest way to estimate out-of-sample skill for time-
ordered data (a plain shuffled cross-validation would leak future information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"            # FastF1 on-disk cache
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"    # fitted models
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Seasons and the walk-forward splits
# ---------------------------------------------------------------------------
SEASONS = [2023, 2024]


@dataclass(frozen=True)
class WalkForwardSplit:
    """One expanding-window train/test step."""

    train_seasons: tuple[int, ...]
    test_season: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        tr = ", ".join(str(s) for s in self.train_seasons)
        return f"train[{tr}] -> test[{self.test_season}]"


def build_walk_forward_splits(seasons: list[int] | None = None) -> list[WalkForwardSplit]:
    """Return expanding-window splits over the given seasons.

    The first season is only ever used for training (it has no predecessor to
    train on), so the first test season is the second entry in ``seasons``.
    """
    seasons = sorted(seasons or SEASONS)
    splits: list[WalkForwardSplit] = []
    for i in range(1, len(seasons)):
        splits.append(
            WalkForwardSplit(
                train_seasons=tuple(seasons[:i]),
                test_season=seasons[i],
            )
        )
    return splits


# ---------------------------------------------------------------------------
# Modelling knobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Controls feature engineering."""

    rolling_window: int = 5            # races used for "recent form"
    min_periods: int = 1               # allow early-season rows with partial history
    podium_cutoff: int = 3             # a "podium" is finishing position <= 3


@dataclass(frozen=True)
class ModelConfig:
    """Controls the estimators and cross-validation."""

    random_state: int = 42
    # LASSO logistic regression: C values searched by inner CV (C = 1/lambda).
    lasso_C_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    # Number of latent factors / principal components for the factor analysis.
    n_factors: int = 4
    # Cross-validation scheme for hyper-parameter selection within the training
    # window. "logo" = leave-one-race-out; "loo" = classic leave-one-out.
    cv_scheme: str = "logo"
    seeds: tuple[int, ...] = field(default_factory=lambda: (42,))


DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
