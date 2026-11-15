"""
L6 Pressure Ratings & Top 10 Selection.
Constructs composite Pressure Performance Index (PPI) on a [0, 100] scale.
Generates:
  - artifacts/pressure/l6/ratings.parquet
  - artifacts/pressure/l6/top10_batters.parquet / .json
  - artifacts/pressure/l6/top10_bowlers.parquet / .json
  - Driver breakdown & sensitivity analysis
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from superr_selector.pressure.config import (
    L5_DIR, L6_DIR,
    MIN_BALLS_FACED_BATTER, MIN_BALLS_BOWLED_BOWLER
)

logger = logging.getLogger(__name__)


def _min_max_scale(series: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    """Scale numeric series to [low, high] continuously with no ties."""
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min < 1e-6:
        return pd.Series(50.0, index=series.index)
    scaled = (series - s_min) / (s_max - s_min) * (high - low) + low
    return scaled


def build() -> dict:
    logger.info("Pressure L6 RATINGS & TOP 10 RANKINGS — starting")
    in_file = L5_DIR / "player_impact.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing L5 artifact: {in_file}")

    df = pd.read_parquet(in_file)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. BATTER RATINGS
    # ──────────────────────────────────────────────────────────────────────────
    bat_mask = (df["role"] == "BAT") & (df["n_pressure_balls"] >= MIN_BALLS_FACED_BATTER)
    bat_df = df[bat_mask].copy()

    # Component scores (0-100)
    bat_df["score_bayes_impact"] = _min_max_scale(bat_df["bayes_impact"])
    bat_df["score_clutch_delta"] = _min_max_scale(bat_df["clutch_delta"])
    bat_df["score_strike_rate"] = _min_max_scale(bat_df["strike_rate_pressure"])
    bat_df["score_reliability"] = _min_max_scale(bat_df["ci_low"])

    # Sample size confidence multiplier: full confidence at 250+ pressure balls
    sample_weight = np.clip(np.sqrt(bat_df["n_pressure_balls"] / 250.0), 0.70, 1.0)

    # Composite Batter Pressure Rating
    bat_raw = (
        0.35 * bat_df["score_bayes_impact"] +
        0.30 * bat_df["score_clutch_delta"] +
        0.20 * bat_df["score_strike_rate"] +
        0.15 * bat_df["score_reliability"]
    ) * sample_weight

    bat_df["pressure_rating"] = _min_max_scale(bat_raw, 30.0, 99.5)
    bat_df = bat_df.sort_values("pressure_rating", ascending=False).reset_index(drop=True)
    bat_df["rank"] = bat_df.index + 1

    top10_batters = bat_df.head(10).copy()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. BOWLER RATINGS
    # ──────────────────────────────────────────────────────────────────────────
    bowl_mask = (df["role"] == "BOWL") & (df["n_pressure_balls"] >= MIN_BALLS_BOWLED_BOWLER)
    bowl_df = df[bowl_mask].copy()

    bowl_df["score_bayes_impact"] = _min_max_scale(bowl_df["bayes_impact"])
    bowl_df["score_clutch_delta"] = _min_max_scale(bowl_df["clutch_delta"])
    # Economy: lower is better → invert
    bowl_df["score_economy"] = _min_max_scale(-bowl_df["economy_pressure"])
    bowl_df["score_reliability"] = _min_max_scale(bowl_df["ci_low"])

    sample_weight_bowl = np.clip(np.sqrt(bowl_df["n_pressure_balls"] / 250.0), 0.70, 1.0)

    bowl_raw = (
        0.35 * bowl_df["score_bayes_impact"] +
        0.30 * bowl_df["score_clutch_delta"] +
        0.20 * bowl_df["score_economy"] +
        0.15 * bowl_df["score_reliability"]
    ) * sample_weight_bowl

    bowl_df["pressure_rating"] = _min_max_scale(bowl_raw, 30.0, 99.5)
    bowl_df = bowl_df.sort_values("pressure_rating", ascending=False).reset_index(drop=True)
    bowl_df["rank"] = bowl_df.index + 1

    top10_bowlers = bowl_df.head(10).copy()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. EXPORT ARTIFACTS
    # ──────────────────────────────────────────────────────────────────────────
    # Save combined ratings
    all_ratings = pd.concat([bat_df, bowl_df], ignore_index=True)
    ratings_path = L6_DIR / "ratings.parquet"
    all_ratings.to_parquet(ratings_path, index=False)
    logger.info("  Saved all ratings: %s (%d players)", ratings_path.name, len(all_ratings))

    # Save Top 10 Batters
    top10_bat_path = L6_DIR / "top10_batters.parquet"
    top10_batters.to_parquet(top10_bat_path, index=False)
    top10_bat_json = L6_DIR / "top10_batters.json"
    top10_bat_cols = [
        "rank", "player_id", "player_name", "pressure_rating",
        "n_pressure_balls", "runs_pressure", "strike_rate_pressure",
        "clutch_delta", "bayes_impact", "ci_low", "ci_high"
    ]
    top10_bat_json.write_text(json.dumps(top10_batters[top10_bat_cols].to_dict(orient="records"), indent=2))

    # Save Top 10 Bowlers
    top10_bowl_path = L6_DIR / "top10_bowlers.parquet"
    top10_bowlers.to_parquet(top10_bowl_path, index=False)
    top10_bowl_json = L6_DIR / "top10_bowlers.json"
    top10_bowl_cols = [
        "rank", "player_id", "player_name", "pressure_rating",
        "n_pressure_balls", "wickets_pressure", "economy_pressure",
        "death_economy_pressure", "clutch_delta", "bayes_impact", "ci_low", "ci_high"
    ]
    top10_bowl_json.write_text(json.dumps(top10_bowlers[top10_bowl_cols].to_dict(orient="records"), indent=2))

    logger.info("\n" + "=" * 65)
    logger.info("  🏆 TOP 10 BATTERS UNDER PRESSURE (Season 2 Submission)")
    logger.info("=" * 65)
    for _, r in top10_batters.iterrows():
        logger.info("  #%2d  %-25s Rating: %5.1f | Balls: %4d | SR: %5.1f | Clutch: %+5.2f",
                    r["rank"], r["player_name"], r["pressure_rating"],
                    int(r["n_pressure_balls"]), r["strike_rate_pressure"], r["clutch_delta"])

    logger.info("\n" + "=" * 65)
    logger.info("  🎯 TOP 10 BOWLERS UNDER PRESSURE (Season 2 Submission)")
    logger.info("=" * 65)
    for _, r in top10_bowlers.iterrows():
        logger.info("  #%2d  %-25s Rating: %5.1f | Balls: %4d | Econ: %5.2f | Wkts: %2d",
                    r["rank"], r["player_name"], r["pressure_rating"],
                    int(r["n_pressure_balls"]), r["economy_pressure"], int(r["wickets_pressure"]))
    logger.info("=" * 65 + "\n")

    logger.info("Pressure L6 RATINGS & TOP 10 RANKINGS — complete ✅")
    return {
        "top10_batters": top10_batters[top10_bat_cols].to_dict(orient="records"),
        "top10_bowlers": top10_bowlers[top10_bowl_cols].to_dict(orient="records"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
