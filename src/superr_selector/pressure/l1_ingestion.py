"""
L1 Ingestion — Ingest and validate delivery-level data (ball_by_ball.csv).
Outputs: artifacts/pressure/l1/clean_deliveries.parquet
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

from superr_selector.pressure.config import CSV_PATHS, L1_DIR

logger = logging.getLogger(__name__)

DeliverySchema = DataFrameSchema(
    {
        "match_id": Column(int, Check.greater_than(0)),
        "innings": Column(int, Check.isin([1, 2, 3, 4])),
        "overid": Column(int, Check.in_range(1, 25)),
        "over_ball_num": Column(int, Check.greater_than(0)),
        "striker_id": Column(int, Check.greater_than(0)),
        "bowler_id": Column(int, Check.greater_than(0)),
        "runs": Column(int, Check.in_range(0, 7)),
        "is_wicket": Column(bool),
        "is_four": Column(bool),
        "is_six": Column(bool),
    },
    coerce=True,
)


def build() -> Path:
    logger.info("Pressure L1 INGESTION — starting")
    csv_file = CSV_PATHS["ball_by_ball"]
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing required CSV: {csv_file}")

    df = pd.read_csv(csv_file)
    logger.info("  Loaded %d raw delivery rows", len(df))

    # Pre-clean boolean types
    for bool_col in ["is_four", "is_six", "is_wicket"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].astype(str).str.lower().isin(["true", "1"])

    # Clean extras
    df["extras1_runs"] = df["extras1_runs"].fillna(0).astype(int)
    df["extras_2_runs"] = df["extras_2_runs"].fillna(0).astype(int)
    df["total_extras"] = df["extras1_runs"] + df["extras_2_runs"]
    df["total_delivery_runs"] = df["runs"] + df["total_extras"]

    # Validate with Pandera
    df = DeliverySchema.validate(df)
    logger.info("  Pandera validation passed ✅")

    # Add match date & competition info from matches.csv
    matches = pd.read_csv(CSV_PATHS["matches"])
    comp = pd.read_csv(CSV_PATHS["competitions"])

    match_meta = matches[[
        "match_id", "comp_id", "match_date", "stage", "venue_id",
        "venue_country", "home_team_id", "away_team_id", "toss_win_team_id",
        "toss_opted", "knock_out", "win_team_id", "result"
    ]].copy()
    match_meta["knockout"] = match_meta["knock_out"].astype(str).str.lower().isin(["true", "1"])
    match_meta = match_meta.drop(columns=["knock_out"])

    # Merge comp info
    match_meta = match_meta.merge(comp[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
    df = df.merge(match_meta, on="match_id", how="left")

    # Flag IPL vs International
    df["is_ipl"] = df["comp_name"].str.contains("IPL|Indian Premier League", case=False, na=False)

    out_file = L1_DIR / "clean_deliveries.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("  Saved clean deliveries: %s (%d rows)", out_file.name, len(df))
    logger.info("Pressure L1 INGESTION — complete ✅")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
