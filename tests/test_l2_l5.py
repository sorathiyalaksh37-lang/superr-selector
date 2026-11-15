"""Tests for L2 Features and L5 Rating."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import pandas as pd
import numpy as np
from superr_selector.config import L2_DIR, L3_DIR, L5_DIR


def test_features_expected_columns():
    df = pd.read_parquet(L2_DIR / "features.parquet")
    required = [
        "player_id", "selection_rate", "captain_rate", "wk_rate",
        "batting_order_mean", "motm_rate", "recent_form_5", "recent_form_10",
        "games_played", "games_in_squad",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_selection_rate_bounded():
    df = pd.read_parquet(L2_DIR / "features.parquet")
    assert (df["selection_rate"] >= 0).all()
    assert (df["selection_rate"] <= 1).all()


def test_roles_valid():
    df = pd.read_parquet(L3_DIR / "player_roles.parquet")
    assert df["role"].isin(["BAT", "BOWL", "WK", "AR"]).all()
    dist = df["role"].value_counts()
    # sanity: we should have at least some of each type
    assert "BAT" in dist.index
    assert "BOWL" in dist.index


def test_ratings_bounded():
    df = pd.read_parquet(L5_DIR / "player_ratings.parquet")
    assert (df["rating"] >= 0).all()
    assert (df["rating_pct"] >= 0).all()
    assert (df["rating_pct"] <= 100).all()
    assert (df["ci_low"] <= df["selection_score"]).all()
    assert (df["ci_high"] >= df["selection_score"]).all()


def test_no_negative_games():
    df = pd.read_parquet(L2_DIR / "features.parquet")
    assert (df["games_played"] >= 0).all()
    assert (df["games_in_squad"] >= df["games_played"]).all()
