"""Data ingestion from the official F1 timing API via FastF1.

This module turns raw sessions into a single tidy table with one row per
(season, round, driver). Downstream feature engineering and modelling never
touch FastF1 directly; they consume the DataFrame produced here.

Everything is cached twice over:

* FastF1 keeps its own on-disk cache of raw API responses (``CACHE_DIR``).
* We additionally persist the assembled per-season tables as parquet in
  ``RAW_DIR`` so repeated runs do not re-parse sessions.

Network access is only needed the first time a season is requested.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from .config import CACHE_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# Columns of the canonical results table.
RESULT_COLUMNS = [
    "season",
    "round",
    "event",
    "circuit",
    "date",
    "driver",
    "driver_number",
    "team",
    "grid_position",
    "quali_position",
    "finish_position",
    "status",
    "points",
    "dnf",
    "air_temp",
    "track_temp",
    "rainfall",
    "humidity",
    "wind_speed",
]


def _import_fastf1():
    """Import FastF1 lazily so the package imports without the dependency.

    Keeping the import inside the functions means unit tests that operate on
    synthetic frames do not require network access or the FastF1 install.
    """
    try:
        import fastf1  # noqa: WPS433 (intentional local import)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "FastF1 is required for data ingestion. Install it with "
            "`pip install fastf1`."
        ) from exc
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return fastf1


def _safe_int(value) -> float:
    """Coerce a possibly-missing position/grid value to a float (NaN if blank)."""
    try:
        if value is None or value == "" or pd.isna(value):
            return float("nan")
        return float(int(value))
    except (TypeError, ValueError):
        return float("nan")


def _summarize_weather(session) -> dict[str, float]:
    """Aggregate the per-lap weather stream into a few race-level numbers."""
    empty = {
        "air_temp": float("nan"),
        "track_temp": float("nan"),
        "rainfall": float("nan"),
        "humidity": float("nan"),
        "wind_speed": float("nan"),
    }
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0:
        return empty
    return {
        "air_temp": float(weather["AirTemp"].mean()),
        "track_temp": float(weather["TrackTemp"].mean()),
        # Rainfall is a boolean per sample; fraction of the session that was wet.
        "rainfall": float(weather["Rainfall"].mean()),
        "humidity": float(weather["Humidity"].mean()),
        "wind_speed": float(weather["WindSpeed"].mean()),
    }


def _load_quali_positions(fastf1, season: int, rnd: int) -> dict[str, float]:
    """Return {driver_abbreviation: qualifying_position} for a round."""
    try:
        quali = fastf1.get_session(season, rnd, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as exc:  # pragma: no cover - network/edge dependent
        logger.warning("No qualifying for %s round %s: %s", season, rnd, exc)
        return {}
    out: dict[str, float] = {}
    for _, row in quali.results.iterrows():
        out[row["Abbreviation"]] = _safe_int(row.get("Position"))
    return out


def fetch_season(season: int, *, force: bool = False) -> pd.DataFrame:
    """Build (or load from cache) the canonical results table for one season.

    Parameters
    ----------
    season:
        Championship year.
    force:
        If True, ignore the parquet cache and re-fetch from FastF1.
    """
    cache_path = RAW_DIR / f"results_{season}.parquet"
    if cache_path.exists() and not force:
        logger.info("Loading cached season %s from %s", season, cache_path)
        return pd.read_parquet(cache_path)

    fastf1 = _import_fastf1()
    logger.info("Fetching season %s from FastF1 (first run is slow)", season)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schedule = fastf1.get_event_schedule(season, include_testing=False)

    rows: list[dict] = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        if rnd < 1:
            continue
        try:
            race = fastf1.get_session(season, rnd, "R")
            race.load(telemetry=False, messages=False)
        except Exception as exc:  # pragma: no cover - season may be incomplete
            logger.warning("Skipping %s round %s: %s", season, rnd, exc)
            continue

        quali_pos = _load_quali_positions(fastf1, season, rnd)
        weather = _summarize_weather(race)

        for _, res in race.results.iterrows():
            abbr = res["Abbreviation"]
            status = str(res.get("Status", ""))
            finish = _safe_int(res.get("Position"))
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "event": event["EventName"],
                    "circuit": event.get("Location", event["EventName"]),
                    "date": pd.to_datetime(event.get("EventDate")),
                    "driver": abbr,
                    "driver_number": _safe_int(res.get("DriverNumber")),
                    "team": res.get("TeamName", ""),
                    "grid_position": _safe_int(res.get("GridPosition")),
                    "quali_position": quali_pos.get(abbr, float("nan")),
                    "finish_position": finish,
                    "status": status,
                    "points": float(res.get("Points", 0.0) or 0.0),
                    # A DNF is any classified result that is not "Finished" and
                    # not a lapped finish ("+N Lap(s)").
                    "dnf": not (status == "Finished" or status.endswith("Lap")
                                or status.endswith("Laps")),
                    **weather,
                }
            )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if df.empty:
        logger.warning("Season %s returned no races (not yet run?)", season)
        return df

    df = df.sort_values(["round", "finish_position"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached season %s (%d rows) to %s", season, len(df), cache_path)
    return df


def fetch_seasons(seasons: list[int], *, force: bool = False) -> pd.DataFrame:
    """Concatenate the canonical tables for several seasons."""
    frames = [fetch_season(s, force=force) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["season", "round", "finish_position"]).reset_index(
        drop=True
    )
