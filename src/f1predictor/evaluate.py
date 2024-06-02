"""Walk-forward training and scoring.

For each expanding-window split (train 2023 -> predict 2024, then add a season
and repeat) we fit the win classifier on the training seasons and score it on
the untouched test season. Within each test race the driver with the highest
predicted win probability is the predicted winner, and the top three are the
predicted podium.

Metrics reported per season:

* ``winner_acc``     fraction of races whose actual winner we ranked first.
* ``podium_prec``    average overlap between our predicted top-3 and the actual
                     top-3 (precision@3).
* ``baseline_acc``   the same winner accuracy for a pole-sitter baseline, the
                     reference any useful model must beat.
* ``log_loss`` / ``brier``  calibration of the per-driver win probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .config import DEFAULT_MODEL_CONFIG, ModelConfig, build_walk_forward_splits
from .features import FEATURE_COLUMNS, race_groups, split_Xy
from .models import fit_win_classifier

RACE_KEYS = ["season", "round"]


def _predicted_winner_accuracy(df: pd.DataFrame, proba_col: str) -> float:
    """Fraction of races where the highest-probability driver actually won."""
    correct = total = 0
    for _, g in df.groupby(RACE_KEYS, sort=False):
        if g["won"].sum() == 0:  # no classified winner in the data
            continue
        total += 1
        if int(g.loc[g[proba_col].idxmax(), "won"]) == 1:
            correct += 1
    return correct / total if total else float("nan")


def _podium_precision_at_3(df: pd.DataFrame, proba_col: str) -> float:
    """Average overlap of predicted top-3 and actual top-3 across races."""
    scores = []
    for _, g in df.groupby(RACE_KEYS, sort=False):
        actual = set(g.loc[g["finish_position"] <= 3, "driver"])
        if not actual:
            continue
        predicted = set(g.nlargest(3, proba_col)["driver"])
        scores.append(len(actual & predicted) / 3.0)
    return float(np.mean(scores)) if scores else float("nan")


def _baseline_winner_accuracy(df: pd.DataFrame) -> float:
    """Pole-sitter baseline: predict the front-row starter to win.

    Uses grid position, falling back to qualifying position when grid is
    missing (e.g. pit-lane starts recorded as 0).
    """
    correct = total = 0
    for _, g in df.groupby(RACE_KEYS, sort=False):
        if g["won"].sum() == 0:
            continue
        total += 1
        order = g.copy()
        order["_grid"] = order["grid_position"].replace(0, np.nan)
        order["_grid"] = order["_grid"].fillna(order["quali_position"])
        pole_idx = order["_grid"].idxmin()
        if int(g.loc[pole_idx, "won"]) == 1:
            correct += 1
    return correct / total if total else float("nan")


@dataclass
class WalkForwardReport:
    metrics: pd.DataFrame
    coefficients: dict[int, pd.Series] = field(default_factory=dict)
    predictions: dict[int, pd.DataFrame] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return self.metrics.to_string(index=False)


def run_walk_forward(
    features_df: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
    *,
    target: str = "won",
) -> WalkForwardReport:
    """Train and score the classifier across all expanding-window splits."""
    seasons = sorted(features_df["season"].unique().tolist())
    splits = build_walk_forward_splits(seasons)

    rows: list[dict] = []
    coefficients: dict[int, pd.Series] = {}
    predictions: dict[int, pd.DataFrame] = {}

    for split in splits:
        train_df = features_df[features_df["season"].isin(split.train_seasons)]
        test_df = features_df[features_df["season"] == split.test_season].copy()
        if train_df.empty or test_df.empty:
            continue

        X_train, y_train = split_Xy(train_df, target=target)
        groups = race_groups(train_df)
        clf = fit_win_classifier(X_train, y_train, groups, cfg)

        test_df["pred_proba"] = clf.predict_proba(test_df)
        coefficients[split.test_season] = clf.coefficients()
        predictions[split.test_season] = test_df

        y_test = test_df[target].astype(int).to_numpy()
        rows.append(
            {
                "split": str(split),
                "train_seasons": "+".join(str(s) for s in split.train_seasons),
                "test_season": split.test_season,
                "n_races": test_df.groupby(RACE_KEYS).ngroups,
                "winner_acc": _predicted_winner_accuracy(test_df, "pred_proba"),
                "podium_prec": _podium_precision_at_3(test_df, "pred_proba"),
                "baseline_acc": _baseline_winner_accuracy(test_df),
                "log_loss": log_loss(y_test, test_df["pred_proba"], labels=[0, 1]),
                "brier": brier_score_loss(y_test, test_df["pred_proba"]),
                "best_C": clf.best_C,
                "cv_log_loss": clf.cv_log_loss,
            }
        )

    metrics = pd.DataFrame(rows)
    return WalkForwardReport(
        metrics=metrics, coefficients=coefficients, predictions=predictions
    )
