"""Tests for the causal feature logic and the walk-forward splits.

These run on a tiny synthetic results table, so they need neither the network
nor FastF1. The point is to lock down the no-leakage property: an engineered
feature for a race must depend only on earlier races.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predictor.config import build_walk_forward_splits
from f1predictor.data import RESULT_COLUMNS
from f1predictor.features import build_features


def _synthetic() -> pd.DataFrame:
    """Two drivers across three rounds of one season, fully controlled."""
    rows = []
    # (round, driver, team, finish, points, dnf)
    schedule = [
        (1, "AAA", "Red", 1, 25, False),
        (1, "BBB", "Blue", 2, 18, False),
        (2, "AAA", "Red", 2, 18, False),
        (2, "BBB", "Blue", 1, 25, False),
        (3, "AAA", "Red", 1, 25, False),
        (3, "BBB", "Blue", 20, 0, True),
    ]
    for rnd, drv, team, finish, pts, dnf in schedule:
        rows.append(
            {
                "season": 2023,
                "round": rnd,
                "event": f"GP{rnd}",
                "circuit": "TestTrack",
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=7 * rnd),
                "driver": drv,
                "driver_number": 1.0,
                "team": team,
                "grid_position": float(finish),
                "quali_position": float(finish),
                "finish_position": float(finish),
                "status": "Finished" if not dnf else "Accident",
                "points": float(pts),
                "dnf": dnf,
                "air_temp": 25.0,
                "track_temp": 35.0,
                "rainfall": 0.0,
                "humidity": 50.0,
                "wind_speed": 2.0,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_first_race_has_no_form_history():
    feats = build_features(_synthetic())
    first = feats[feats["round"] == 1]
    # Nobody has prior races, so rolling form must be undefined (NaN).
    assert first["driver_form_points"].isna().all()
    assert first["driver_form_finish"].isna().all()


def test_rolling_form_excludes_current_race():
    feats = build_features(_synthetic())
    # AAA finished P1 then P2; entering round 3 the causal mean finish must be
    # (1 + 2) / 2 = 1.5, using only rounds 1 and 2, not round 3.
    aaa_r3 = feats[(feats["driver"] == "AAA") & (feats["round"] == 3)]
    assert aaa_r3["driver_form_finish"].iloc[0] == pytest.approx(1.5)


def test_reliability_is_causal():
    feats = build_features(_synthetic())
    # BBB finished every race before round 3, so reliability entering round 3
    # is 1.0; the round-3 DNF must not retroactively lower it.
    bbb_r3 = feats[(feats["driver"] == "BBB") & (feats["round"] == 3)]
    assert bbb_r3["driver_reliability"].iloc[0] == pytest.approx(1.0)


def test_targets():
    feats = build_features(_synthetic())
    winners = feats[feats["won"] == 1]["finish_position"].unique()
    assert list(winners) == [1.0]
    # Top-3 finishes: R1 (P1, P2), R2 (P1, P2), R3 (P1 only) = 5.
    assert feats["podium"].sum() == 5


def test_walk_forward_splits_are_expanding():
    splits = build_walk_forward_splits([2023, 2024, 2025, 2026])
    assert [s.test_season for s in splits] == [2024, 2025, 2026]
    assert splits[0].train_seasons == (2023,)
    assert splits[1].train_seasons == (2023, 2024)
    assert splits[2].train_seasons == (2023, 2024, 2025)
    # Training seasons are always strictly before the test season.
    for s in splits:
        assert all(ts < s.test_season for ts in s.train_seasons)
