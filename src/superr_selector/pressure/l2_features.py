"""
L2 Delivery Features — Add delivery state, match situation, target, and chase dynamics.
Outputs: artifacts/pressure/l2/state_deliveries.parquet
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from superr_selector.pressure.config import L1_DIR, L2_DIR

logger = logging.getLogger(__name__)


def build() -> Path:
    logger.info("Pressure L2 FEATURES — starting")
    in_file = L1_DIR / "clean_deliveries.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing L1 artifact: {in_file}")

    df = pd.read_parquet(in_file)
    logger.info("  Processing %d deliveries", len(df))

    # Sort sequentially
    df = df.sort_values(["match_id", "innings", "overid", "over_ball_num"]).reset_index(drop=True)

    # Legal ball calculation (wides & oversteps do not advance ball count)
    df["is_legal"] = ~df["extras_type"].isin(["wide", "Overstep"])
    df["innings_legal_balls"] = (
        df.groupby(["match_id", "innings"])["is_legal"].cumsum()
    )
    df["balls_remaining"] = np.maximum(0, 120 - df["innings_legal_balls"])
    df["overs_remaining"] = df["balls_remaining"] / 6.0

    # Cumulative runs in the innings
    df["innings_runs_after"] = (
        df.groupby(["match_id", "innings"])["total_delivery_runs"].cumsum()
    )
    df["innings_runs_before"] = df["innings_runs_after"] - df["total_delivery_runs"]

    # Cumulative wickets in the innings
    df["innings_wickets_after"] = (
        df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    )
    df["innings_wickets_before"] = df["innings_wickets_after"] - df["is_wicket"].astype(int)
    df["wickets_in_hand"] = np.maximum(0, 10 - df["innings_wickets_before"])

    # Phase assignment
    conditions = [
        df["overid"] <= 6,
        (df["overid"] >= 7) & (df["overid"] <= 15),
        df["overid"] >= 16,
    ]
    choices = ["Powerplay", "Middle", "Death"]
    df["phase"] = np.select(conditions, choices, default="Middle")
    df["phase_id"] = np.select([df["phase"] == "Powerplay", df["phase"] == "Middle", df["phase"] == "Death"], [0, 1, 2], default=1)

    # Compute target for Innings 2
    inn1_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_delivery_runs"]
        .sum()
        .rename("inn1_total")
    )
    df = df.merge(inn1_totals, on="match_id", how="left")
    df["target"] = df["inn1_total"] + 1

    # Innings 2 Chase features
    df["is_chase"] = df["innings"] == 2
    df["runs_needed"] = np.where(
        df["is_chase"],
        np.maximum(0, df["target"] - df["innings_runs_before"]),
        0.0,
    )
    df["required_run_rate"] = np.where(
        df["is_chase"] & (df["balls_remaining"] > 0),
        (df["runs_needed"] / df["balls_remaining"]) * 6.0,
        0.0,
    )

    # Innings 1 current run rate
    df["current_run_rate"] = np.where(
        df["innings_legal_balls"] > 0,
        (df["innings_runs_before"] / df["innings_legal_balls"]) * 6.0,
        6.0,
    )

    # Match result alignment
    df["batting_team_won"] = (df["batting_team_id"] == df["win_team_id"]).astype(int)

    out_file = L2_DIR / "state_deliveries.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("  Saved state deliveries: %s (%d rows)", out_file.name, len(df))
    logger.info("Pressure L2 FEATURES — complete ✅")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
