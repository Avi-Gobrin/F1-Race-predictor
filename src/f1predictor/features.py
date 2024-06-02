"""Causal feature engineering.

Every engineered feature obeys one rule: **a row may only use information that
was available before that race started**. Rolling and historical statistics are
therefore shifted so the current race is excluded from its own features. Getting
this wrong is the classic way a sports model looks brilliant in backtest and
useless in reality, so the logic here is covered by unit tests.

The output is a feature matrix with metadata columns (season, round, driver,
team, event), engineered predictors, and the prediction targets ``won`` and
``podium``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE_CONFIG, FeatureConfig

# Engineered predictor columns (the design matrix X is exactly these).
FEATURE_COLUMNS = [
    "grid_position",
    "quali_position",
    "driver_form_points",
    "driver_form_finish",
    "team_form_points",
    "team_form_finish",
    "driver_track_finish",
    "driver_reliability",
    "air_temp",
    "track_temp",
    "rainfall",
    "humidity",
    "wind_speed",
]

META_COLUMNS = ["season", "round", "event", "circuit", "date", "driver", "team"]
TARGET_COLUMNS = ["won", "podium", "finish_position"]


def add_targets(df: pd.DataFrame, cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> pd.DataFrame:
    """Attach binary targets: ``won`` (P1) and ``podium`` (top ``podium_cutoff``)."""
    out = df.copy()
    out["won"] = (out["finish_position"] == 1).astype(int)
    out["podium"] = (out["finish_position"] <= cfg.podium_cutoff).astype(int)
    return out


def _causal_rolling_mean(
    df: pd.DataFrame, group_col: str, value_col: str, window: int, min_periods: int
) -> pd.Series:
    """Rolling mean of ``value_col`` within ``group_col``, excluding the current row.

    The ``shift(1)`` is what makes the statistic causal: at race k the value is
    the mean over races k-window .. k-1, never including race k itself.
    """
    grouped = df.groupby(group_col, sort=False)[value_col]
    shifted = grouped.shift(1)
    # Recombine the shifted values with their group key, then roll within group.
    tmp = pd.DataFrame({group_col: df[group_col], "v": shifted})
    return (
        tmp.groupby(group_col, sort=False)["v"]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _causal_expanding_mean(
    df: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.Series:
    """Expanding mean within ``group_cols`` using only earlier races."""
    grouped = df.groupby(group_cols, sort=False)[value_col]
    shifted = grouped.shift(1)
    tmp = pd.DataFrame({"_g": df[group_cols].astype(str).agg("|".join, axis=1), "v": shifted})
    return (
        tmp.groupby("_g", sort=False)["v"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )


def build_features(
    df: pd.DataFrame, cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG
) -> pd.DataFrame:
    """Engineer the full causal feature matrix from the canonical results table.

    The input must be the table produced by :mod:`f1predictor.data` covering all
    seasons of interest; rolling history is allowed to carry across seasons
    (a driver's late-2023 form informs early 2024), which is realistic.
    """
    if df.empty:
        return df.copy()

    work = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)
    work = add_targets(work, cfg)

    window, min_p = cfg.rolling_window, cfg.min_periods

    # Recent driver form: rolling mean of points scored and finishing position.
    work["driver_form_points"] = _causal_rolling_mean(
        work, "driver", "points", window, min_p
    )
    work["driver_form_finish"] = _causal_rolling_mean(
        work, "driver", "finish_position", window, min_p
    )

    # Recent constructor (team) form.
    work["team_form_points"] = _causal_rolling_mean(
        work, "team", "points", window, min_p
    )
    work["team_form_finish"] = _causal_rolling_mean(
        work, "team", "finish_position", window, min_p
    )

    # Driver-track history: average past finish for this driver at this circuit.
    work["driver_track_finish"] = _causal_expanding_mean(
        work, ["driver", "circuit"], "finish_position"
    )

    # Reliability proxy: rolling share of races the driver actually finished.
    work["_finished"] = (~work["dnf"].astype(bool)).astype(float)
    work["driver_reliability"] = _causal_rolling_mean(
        work, "driver", "_finished", window, min_p
    )
    work = work.drop(columns="_finished")

    keep = META_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
    return work[keep].reset_index(drop=True)


def split_Xy(
    df: pd.DataFrame, target: str = "won"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for modelling. X is the engineered design matrix."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be one of {TARGET_COLUMNS}, got {target!r}")
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[target].astype(int)
    return X, y


def race_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group id per race (season, round) for leave-one-race-out CV."""
    keys = df["season"].astype(str) + "_" + df["round"].astype(str)
    return keys.astype("category").cat.codes.to_numpy()
