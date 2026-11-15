"""Tests for L1 Ingestion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import pandas as pd
from superr_selector.config import L1_DIR


def test_l1_artifacts_exist():
    for name in ["competitions", "teams", "player_map", "matches", "playing_eleven"]:
        path = L1_DIR / f"{name}.parquet"
        assert path.exists(), f"Missing L1 artifact: {name}.parquet"


def test_competitions_schema():
    df = pd.read_parquet(L1_DIR / "competitions.parquet")
    assert "comp_id"      in df.columns
    assert "comp_name"    in df.columns
    assert "match_format" in df.columns
    assert "season"       in df.columns
    assert df["match_format"].isin(["T20", "T20I"]).all()


def test_matches_schema():
    df = pd.read_parquet(L1_DIR / "matches.parquet")
    assert "match_id"   in df.columns
    assert "match_date" in df.columns
    assert "knockout"   in df.columns
    assert df["match_id"].is_unique, "match_id should be unique"
    assert df["knockout"].dtype == bool


def test_playing_eleven_no_duplicates():
    df = pd.read_parquet(L1_DIR / "playing_eleven.parquet")
    dupes = df.duplicated(subset=["match_id", "team_id", "player_id", "batting_order"])
    assert not dupes.any(), f"Duplicate rows in playing_eleven: {dupes.sum()}"


def test_playing_eleven_bool_columns():
    df = pd.read_parquet(L1_DIR / "playing_eleven.parquet")
    for col in ["is_played", "is_captain", "is_wk"]:
        assert df[col].isin([True, False]).all(), f"{col} should be boolean"


def test_player_map_no_duplicates():
    df = pd.read_parquet(L1_DIR / "player_map.parquet")
    dupes = df.duplicated(subset=["comp_id", "team_id", "player_id"])
    assert not dupes.any(), f"Duplicate players in squad: {dupes.sum()}"
