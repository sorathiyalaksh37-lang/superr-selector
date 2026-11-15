"""
L1 — INGESTION
Reads raw CSVs → validates with pandera → saves frozen parquet artifacts.
No downstream layer reads a CSV; they all read from artifacts/l1/.
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

from superr_selector.config import CSV, L1_DIR

logger = logging.getLogger(__name__)

# ── Pandera Schemas ──────────────────────────────────────────────────────────

COMPETITIONS_SCHEMA = DataFrameSchema({
    "comp_id":      Column(int, nullable=False),
    "comp_name":    Column(str, nullable=False),
    "match_format": Column(str, Check.isin(["T20", "T20I"]), nullable=False),
    "season":       Column(str, nullable=False),   # kept as str; "2022/23" is valid
}, coerce=True)

TEAMS_SCHEMA = DataFrameSchema({
    "comp_id":   Column(int, nullable=False),
    "team_id":   Column(int, nullable=False),
    "team_name": Column(str, nullable=False),
}, coerce=True)

PLAYER_MAP_SCHEMA = DataFrameSchema({
    "comp_id":    Column(int, nullable=False),
    "comp_name":  Column(str, nullable=False),
    "team_id":    Column(int, nullable=False),
    "team_name":  Column(str, nullable=False),
    "player_id":  Column(int, nullable=False),
    "player_name":Column(str, nullable=False),
}, coerce=True)

MATCHES_SCHEMA = DataFrameSchema({
    "match_id":        Column(int, nullable=False),
    "comp_id":         Column(int, nullable=False),
    "match_number":    Column(int, nullable=False),
    "match_date":      Column(str, nullable=False),
    "stage":           Column(str, nullable=False),
    "venue_id":        Column(int, nullable=True),
    "venue_country":   Column(str, nullable=True),
    "home_team_id":    Column(int, nullable=True),
    "away_team_id":    Column(int, nullable=True),
    "home_team":       Column(str, nullable=True),
    "away_team":       Column(str, nullable=True),
    "home_team_captain_id": Column(pa.Float, nullable=True),
    "away_team_captain_id": Column(pa.Float, nullable=True),
    "toss_win_team_id":     Column(pa.Float, nullable=True),
    "toss_opted":           Column(str, nullable=True),
    "man_of_the_match":     Column(pa.Float, nullable=True),
    "win_team_id":          Column(pa.Float, nullable=True),
    "result":               Column(str, nullable=True),
}, coerce=True)

PLAYING_ELEVEN_SCHEMA = DataFrameSchema({
    "match_id":      Column(int, nullable=False),
    "team_id":       Column(int, nullable=False),
    "player_id":     Column(int, nullable=False),
    "batting_order": Column(pa.Float, nullable=True),
    "is_played":     Column(str, nullable=False),
    "is_captain":    Column(str, nullable=False),
    "is_wk":         Column(str, nullable=False),
    "is_impact":     Column(int, nullable=False),
    "is_impact_off": Column(int, nullable=False),
}, coerce=True)


# ── Readers ──────────────────────────────────────────────────────────────────

def _read_competitions() -> pd.DataFrame:
    df = pd.read_csv(CSV["competitions"])
    df = COMPETITIONS_SCHEMA.validate(df)
    return df

def _read_teams() -> pd.DataFrame:
    df = pd.read_csv(CSV["teams"])
    df = TEAMS_SCHEMA.validate(df)
    return df

def _read_player_map() -> pd.DataFrame:
    df = pd.read_csv(CSV["player_map"])
    df = PLAYER_MAP_SCHEMA.validate(df)
    # deduplicate (same player can appear multiple rows in same comp/team)
    df = df.drop_duplicates(subset=["comp_id", "team_id", "player_id"])
    return df

def _read_matches() -> pd.DataFrame:
    df = pd.read_csv(CSV["matches"])
    df = MATCHES_SCHEMA.validate(df)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["knockout"] = df["stage"].str.lower().isin(
        ["qualifier", "eliminator", "final", "semi final", "semi-final"]
    )
    return df

def _read_playing_eleven() -> pd.DataFrame:
    df = pd.read_csv(CSV["playing_eleven"])
    # Fill NaN boolean columns BEFORE schema validation
    # (non-played squad rows have NaN for is_captain, is_wk, batting_order)
    for col in ["is_played", "is_captain", "is_wk"]:
        df[col] = df[col].fillna("false").astype(str)
    df["batting_order"] = pd.to_numeric(df["batting_order"], errors="coerce")
    df["is_impact"]     = pd.to_numeric(df["is_impact"],     errors="coerce").fillna(0).astype(int)
    df["is_impact_off"] = pd.to_numeric(df["is_impact_off"], errors="coerce").fillna(0).astype(int)
    df = PLAYING_ELEVEN_SCHEMA.validate(df)
    # normalise bool-like columns
    for col in ["is_played", "is_captain", "is_wk"]:
        df[col] = df[col].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)
    # remove duplicates (match 13613 super over issue)
    df = df.drop_duplicates(subset=["match_id", "team_id", "player_id", "batting_order"])
    return df


# ── Build ─────────────────────────────────────────────────────────────────────

def build() -> dict[str, pd.DataFrame]:
    """Run L1: ingest all CSVs → validate → save parquet artifacts."""
    logger.info("L1 INGESTION — starting")

    frames = {
        "competitions":  _read_competitions(),
        "teams":         _read_teams(),
        "player_map":    _read_player_map(),
        "matches":       _read_matches(),
        "playing_eleven": _read_playing_eleven(),
    }

    for name, df in frames.items():
        out = L1_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.info("  ✓ %s → %s  (%d rows)", name, out.name, len(df))

    logger.info("L1 INGESTION — complete ✅")
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
