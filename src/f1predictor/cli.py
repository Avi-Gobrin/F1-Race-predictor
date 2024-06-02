"""Command-line interface.

    f1predict build-dataset --seasons 2023 2024 2025 2026
    f1predict evaluate
    f1predict analyze
    f1predict predict --season 2026 --round 9

Also runnable as ``python -m f1predictor.cli ...``.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import PROCESSED_DIR, REPORTS_DIR, SEASONS
from .data import fetch_seasons
from .evaluate import run_walk_forward
from .features import build_features, race_groups, split_Xy
from .models import factor_analysis, fit_win_classifier, pca_explained_variance

FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def _load_features(build_if_missing: bool = True) -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    if not build_if_missing:
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run `f1predict build-dataset` first."
        )
    df = build_features(fetch_seasons(SEASONS))
    df.to_parquet(FEATURES_PATH, index=False)
    return df


def cmd_build_dataset(args: argparse.Namespace) -> None:
    seasons = args.seasons or SEASONS
    raw = fetch_seasons(seasons, force=args.force)
    if raw.empty:
        print("No data returned for the requested seasons.")
        return
    feats = build_features(raw)
    feats.to_parquet(FEATURES_PATH, index=False)
    print(f"Built features for seasons {seasons}: {len(feats)} driver-rows.")
    print(f"Saved to {FEATURES_PATH}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    feats = _load_features()
    report = run_walk_forward(feats, target=args.target)
    if report.metrics.empty:
        print("Not enough seasons with data to evaluate.")
        return

    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print("\nWalk-forward results")
    print("====================")
    print(report.summary())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "walk_forward_metrics.csv"
    report.metrics.to_csv(out, index=False)
    print(f"\nMetrics written to {out}")

    print("\nLASSO coefficients on the most recent split (standardized features)")
    last_season = max(report.coefficients)
    nonzero = report.coefficients[last_season]
    nonzero = nonzero[nonzero != 0]
    print(nonzero.to_string())


def cmd_analyze(args: argparse.Namespace) -> None:
    feats = _load_features()
    X, _ = split_Xy(feats, target="won")
    print("\nPCA explained variance ratio")
    print(pca_explained_variance(X).to_string())
    print("\nFactor analysis loadings (varimax)")
    print(factor_analysis(X).round(2).to_string())


def cmd_predict(args: argparse.Namespace) -> None:
    feats = _load_features()
    train = feats[feats["season"] < args.season]
    if train.empty:
        raise SystemExit(
            f"No seasons before {args.season} are available to train on."
        )
    race = feats[(feats["season"] == args.season) & (feats["round"] == args.round)]
    if race.empty:
        raise SystemExit(f"No data for season {args.season} round {args.round}.")

    X_train, y_train = split_Xy(train, target="won")
    clf = fit_win_classifier(X_train, y_train, race_groups(train))

    race = race.copy()
    race["win_probability"] = clf.predict_proba(race)
    ranked = race.sort_values("win_probability", ascending=False)
    event = ranked["event"].iloc[0]
    print(f"\nPredicted win probabilities: {args.season} {event}")
    print("=" * 48)
    cols = ["driver", "team", "grid_position", "win_probability"]
    print(ranked[cols].head(10).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1predict", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset", help="Fetch data and build the feature table.")
    b.add_argument("--seasons", type=int, nargs="*", default=None)
    b.add_argument("--force", action="store_true", help="Ignore caches and refetch.")
    b.set_defaults(func=cmd_build_dataset)

    e = sub.add_parser("evaluate", help="Run walk-forward evaluation.")
    e.add_argument("--target", choices=["won", "podium"], default="won")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("analyze", help="Factor analysis / PCA of the features.")
    a.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("predict", help="Predict win probabilities for one race.")
    pr.add_argument("--season", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
