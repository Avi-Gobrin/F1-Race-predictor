"""Data ingestion from the official F1 timing feed via FastF1.

This module turns raw sessions into a single tidy table with one row per
(season, round, driver). Downstream feature engineering and modelling never
touch FastF1 directly; they consume the DataFrame produced here.

Data source note
----------------
The historical Ergast API (and its Jolpica mirror) is the usual source for
classified results, grid positions, points, and status. As of mid-2026 those
endpoints are unreliable (Ergast is decommissioned; the Jolpica mirror is
heavily rate-limited). This module therefore derives everything it needs from
the live timing feed, which FastF1 reads directly:

* finishing order        -> race session results (timing classification)
* qualifying / grid       -> qualifying session results (grid uses quali as a
                             proxy; grid penalties are not reflected)
* points                  -> computed from finishing position with the standard
                             post-2010 scoring table (fastest-lap bonus ignored)
* did-not-finish (DNF)    -> a driver who completed under 90% of the winner's
                             lap count is treated as not classified, matching
                             the FIA classification rule
* weather                 -> race session weather stream, summarized per race

Everything is cached: FastF1 keeps its own on-disk cache of raw API responses,
and assembled per-season tables are persisted as parquet so reruns avoid the
network.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from .config import CACHE_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# Standard F1 points for finishing positions 1..10 (post-2010 system).
POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# A driver completing fewer than this fraction of the winner's laps is treated
# as not classified (the FIA 90% rule), which we use as the DNF flag.
CLASSIFIED_LAP_FRACTION = 0.90

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
    """Import and configure FastF1 lazily (keeps unit tests dependency-free)."""
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "FastF1 is required for data ingestion. Install it with "
            "`pip install fastf1`."
        ) from exc
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return fastf1


def _points_for_position(position: float) -> float:
    """Standard championship points for a finishing position."""
    try:
        return float(POINTS_TABLE.get(int(position), 0))
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> float:
    """Coerce a possibly-missing position/number to a float (NaN if blank)."""
    try:
        if value is None or value == "" or pd.isna(value):
            return float("nan")
        return float(int(value))
    except (TypeError, ValueError):
        return float("nan")


def _completed_laps(session) -> pd.Series:
    """Laps completed per driver abbreviation (max lap number reached)."""
    laps = getattr(session, "laps", None)
    if laps is None or len(laps) == 0:
        return pd.Series(dtype=float)
    return laps.groupby("Driver")["LapNumber"].max()


def _summarize_weather(session) -> dict[str, float]:
    """Aggregate the per-sample weather stream into race-level numbers."""
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
        "rainfall": float(weather["Rainfall"].mean()),  # fraction of wet samples
        "humidity": float(weather["Humidity"].mean()),
        "wind_speed": float(weather["WindSpeed"].mean()),
    }


def _quali_positions(fastf1, season: int, rnd: int) -> dict[str, float]:
    """Return {driver_abbreviation: qualifying_position} for a round."""
    try:
        quali = fastf1.get_session(season, rnd, "Q")
        # Race-control messages are required for FastF1 to classify qualifying
        # (without them the result Position comes back as NaN).
        quali.load(telemetry=False, weather=False)
    except Exception as exc:  # pragma: no cover - network/edge dependent
        logger.warning("No qualifying for %s round %s: %s", season, rnd, exc)
        return {}
    out: dict[str, float] = {}
    for _, row in quali.results.iterrows():
        out[row["Abbreviation"]] = _safe_int(row.get("Position"))
    return out


def fetch_season(season: int, *, force: bool = False) -> pd.DataFrame:
    """Build (or load from cache) the canonical results table for one season."""
    cache_path = RAW_DIR / f"results_{season}.parquet"
    if cache_path.exists() and not force:
        logger.info("Loading cached season %s from %s", season, cache_path)
        return pd.read_parquet(cache_path)

    fastf1 = _import_fastf1()
    logger.info("Fetching season %s from the timing feed (first run is slow)", season)

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
        if race.results is None or len(race.results) == 0:
            continue

        quali_pos = _quali_positions(fastf1, season, rnd)
        weather = _summarize_weather(race)
        laps = _completed_laps(race)
        winner_laps = float(laps.max()) if len(laps) else float("nan")
        logger.info("Loaded %s round %s: %s", season, rnd, event["EventName"])

        for _, res in race.results.iterrows():
            abbr = res["Abbreviation"]
            finish = _safe_int(res.get("Position"))
            driver_laps = float(laps.get(abbr, float("nan")))
            if pd.notna(winner_laps) and pd.notna(driver_laps):
                dnf = driver_laps < CLASSIFIED_LAP_FRACTION * winner_laps
            else:
                dnf = False
            grid = quali_pos.get(abbr, float("nan"))
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
                    "grid_position": grid,
                    "quali_position": grid,
                    "finish_position": finish,
                    "status": "DNF" if dnf else "Classified",
                    "points": _points_for_position(finish),
                    "dnf": bool(dnf),
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
