"""
L5 Bayesian Player Impact & Baseline Comparison.
Computes:
  - Pressure vs Ordinary splits (E = baseline, R = pressure response)
  - Empirical Bayes shrinkage on pressure performance
  - Credible Intervals (95% CI)
  - Clutch Delta (Elevation over usual performance level)

Outputs: artifacts/pressure/l5/player_impact.parquet
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from superr_selector.pressure.config import (
    L4_DIR, L5_DIR,
    BAYESIAN_K_BATTER, BAYESIAN_K_BOWLER,
    MIN_BALLS_FACED_BATTER, MIN_BALLS_BOWLED_BOWLER
)

logger = logging.getLogger(__name__)


def compute_batter_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate delivery-level impacts to batter level with Empirical Bayes shrinkage."""
    logger.info("  Computing Bayesian Batter Impact...")

    # Group by striker
    records = []
    grouped = df.groupby(["striker_id", "striker_name"])

    for (p_id, p_name), p_df in grouped:
        n_total = len(p_df)
        runs_total = p_df["runs"].sum()

        # Pressure subset (DPI >= 60)
        p_press = p_df[p_df["is_high_pressure"]]
        n_press = len(p_press)
        runs_press = p_press["runs"].sum()
        outs_press = p_press["is_wicket"].sum()
        sr_press = (runs_press / max(1, n_press)) * 100.0
        boundary_press = (p_press["is_four"].sum() + p_press["is_six"].sum()) / max(1, n_press) * 100.0

        # Mean and std of pressure impact
        raw_press_impact = p_press["I_bat"].mean() if n_press > 0 else 0.0
        var_press_impact = p_press["I_bat"].var() if n_press > 1 else 0.5
        var_press_impact = 0.5 if np.isnan(var_press_impact) else var_press_impact

        # Ordinary subset (DPI < 40)
        p_ord = p_df[p_df["is_ordinary"]]
        n_ord = len(p_ord)
        runs_ord = p_ord["runs"].sum()
        sr_ord = (runs_ord / max(1, n_ord)) * 100.0
        raw_ord_impact = p_ord["I_bat"].mean() if n_ord > 0 else 0.0

        # Clutch delta: difference between high pressure and ordinary performance
        clutch_delta = raw_press_impact - raw_ord_impact

        # Matches and knockout games
        matches_count = p_df["match_id"].nunique()
        knockout_press_runs = p_press[p_press["knockout"] == True]["runs"].sum()

        records.append({
            "player_id": p_id,
            "player_name": p_name,
            "role": "BAT",
            "n_total_balls": n_total,
            "n_pressure_balls": n_press,
            "runs_pressure": runs_press,
            "strike_rate_pressure": sr_press,
            "boundary_pct_pressure": boundary_press,
            "dismissals_pressure": outs_press,
            "n_ordinary_balls": n_ord,
            "strike_rate_ordinary": sr_ord,
            "raw_pressure_impact": raw_press_impact,
            "raw_ordinary_impact": raw_ord_impact,
            "clutch_delta": clutch_delta,
            "var_pressure_impact": var_press_impact,
            "matches_count": matches_count,
            "knockout_press_runs": knockout_press_runs,
        })

    bat_df = pd.DataFrame(records)

    # Filter to pool with some minimum activity to establish empirical prior
    pool = bat_df[bat_df["n_pressure_balls"] >= 20]
    prior_mean = pool["raw_pressure_impact"].mean() if len(pool) > 0 else 0.0

    # Empirical Bayes Shrinkage
    # w = n / (n + K)
    w = bat_df["n_pressure_balls"] / (bat_df["n_pressure_balls"] + BAYESIAN_K_BATTER)
    bat_df["bayes_impact"] = w * bat_df["raw_pressure_impact"] + (1.0 - w) * prior_mean

    # Standard error & 95% Credible Interval
    se = np.sqrt(
        (bat_df["var_pressure_impact"] + 0.1) / (bat_df["n_pressure_balls"] + BAYESIAN_K_BATTER)
    )
    bat_df["ci_low"] = bat_df["bayes_impact"] - 1.96 * se
    bat_df["ci_high"] = bat_df["bayes_impact"] + 1.96 * se
    bat_df["is_eligible"] = bat_df["n_pressure_balls"] >= MIN_BALLS_FACED_BATTER

    return bat_df


def compute_bowler_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate delivery-level impacts to bowler level with Empirical Bayes shrinkage."""
    logger.info("  Computing Bayesian Bowler Impact...")

    records = []
    grouped = df.groupby(["bowler_id", "bowler_name"])

    for (p_id, p_name), p_df in grouped:
        n_total = len(p_df)
        runs_conceded_total = p_df["total_delivery_runs"].sum()
        wickets_total = p_df["is_wicket"].sum()

        # Pressure subset (DPI >= 60)
        p_press = p_df[p_df["is_high_pressure"]]
        n_press = len(p_press)
        runs_press = p_press["total_delivery_runs"].sum()
        wickets_press = p_press["is_wicket"].sum()
        econ_press = (runs_press / max(1, n_press)) * 6.0
        dots_press = (p_press["total_delivery_runs"] == 0).sum() / max(1, n_press) * 100.0

        # Mean and std of pressure impact
        raw_press_impact = p_press["I_bowl"].mean() if n_press > 0 else 0.0
        var_press_impact = p_press["I_bowl"].var() if n_press > 1 else 0.5
        var_press_impact = 0.5 if np.isnan(var_press_impact) else var_press_impact

        # Ordinary subset (DPI < 40)
        p_ord = p_df[p_df["is_ordinary"]]
        n_ord = len(p_ord)
        runs_ord = p_ord["total_delivery_runs"].sum()
        econ_ord = (runs_ord / max(1, n_ord)) * 6.0
        raw_ord_impact = p_ord["I_bowl"].mean() if n_ord > 0 else 0.0

        # Clutch delta: difference between high pressure and ordinary performance
        clutch_delta = raw_press_impact - raw_ord_impact

        # Death overs under pressure
        death_press = p_press[p_press["phase"] == "Death"]
        death_econ_press = (death_press["total_delivery_runs"].sum() / max(1, len(death_press))) * 6.0

        matches_count = p_df["match_id"].nunique()

        records.append({
            "player_id": p_id,
            "player_name": p_name,
            "role": "BOWL",
            "n_total_balls": n_total,
            "n_pressure_balls": n_press,
            "wickets_pressure": wickets_press,
            "economy_pressure": econ_press,
            "dot_pct_pressure": dots_press,
            "death_economy_pressure": death_econ_press,
            "n_ordinary_balls": n_ord,
            "economy_ordinary": econ_ord,
            "raw_pressure_impact": raw_press_impact,
            "raw_ordinary_impact": raw_ord_impact,
            "clutch_delta": clutch_delta,
            "var_pressure_impact": var_press_impact,
            "matches_count": matches_count,
        })

    bowl_df = pd.DataFrame(records)

    # Establish empirical prior from bowlers with >= 30 pressure balls
    pool = bowl_df[bowl_df["n_pressure_balls"] >= 30]
    prior_mean = pool["raw_pressure_impact"].mean() if len(pool) > 0 else 0.0

    # Empirical Bayes Shrinkage
    w = bowl_df["n_pressure_balls"] / (bowl_df["n_pressure_balls"] + BAYESIAN_K_BOWLER)
    bowl_df["bayes_impact"] = w * bowl_df["raw_pressure_impact"] + (1.0 - w) * prior_mean

    se = np.sqrt(
        (bowl_df["var_pressure_impact"] + 0.1) / (bowl_df["n_pressure_balls"] + BAYESIAN_K_BOWLER)
    )
    bowl_df["ci_low"] = bowl_df["bayes_impact"] - 1.96 * se
    bowl_df["ci_high"] = bowl_df["bayes_impact"] + 1.96 * se
    bowl_df["is_eligible"] = bowl_df["n_pressure_balls"] >= MIN_BALLS_BOWLED_BOWLER

    return bowl_df


def build() -> Path:
    logger.info("Pressure L5 BAYESIAN IMPACT — starting")
    in_file = L4_DIR / "predictions.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing L4 artifact: {in_file}")

    df = pd.read_parquet(in_file)
    logger.info("  Loaded %d delivery predictions", len(df))

    bat_df = compute_batter_impact(df)
    bowl_df = compute_bowler_impact(df)

    logger.info("  Batters processed: %d (Eligible for top 10: %d)",
                len(bat_df), bat_df["is_eligible"].sum())
    logger.info("  Bowlers processed: %d (Eligible for top 10: %d)",
                len(bowl_df), bowl_df["is_eligible"].sum())

    combined = pd.concat([bat_df, bowl_df], ignore_index=True)
    out_file = L5_DIR / "player_impact.parquet"
    combined.to_parquet(out_file, index=False)
    logger.info("  Saved player impact: %s (%d rows)", out_file.name, len(combined))
    logger.info("Pressure L5 BAYESIAN IMPACT — complete ✅")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
