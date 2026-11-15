"""Tests for Phase 2 Pressure & Clutch Pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import pandas as pd
from superr_selector.pressure.config import L1_DIR, L3_DIR, L6_DIR, L7_DIR


def test_pressure_artifacts_exist():
    assert (L1_DIR / "clean_deliveries.parquet").exists()
    assert (L3_DIR / "dpi.parquet").exists()
    assert (L6_DIR / "top10_batters.json").exists()
    assert (L6_DIR / "top10_bowlers.json").exists()
    assert (L7_DIR / "backtest_report.json").exists()


def test_top10_batters_count_and_uniqueness():
    df = pd.read_parquet(L6_DIR / "top10_batters.parquet")
    assert len(df) == 10
    assert df["player_id"].nunique() == 10
    assert (df["n_pressure_balls"] >= 50).all()
    assert (df["strike_rate_pressure"] >= 150.0).all()


def test_top10_bowlers_count_and_uniqueness():
    df = pd.read_parquet(L6_DIR / "top10_bowlers.parquet")
    assert len(df) == 10
    assert df["player_id"].nunique() == 10
    assert (df["n_pressure_balls"] >= 60).all()
    assert (df["economy_pressure"] <= 9.2).all()


def test_dpi_range():
    df = pd.read_parquet(L3_DIR / "dpi.parquet")
    assert df["dpi"].min() >= 0.0
    assert df["dpi"].max() <= 100.0
    assert (df["dpi"] >= 60.0).sum() > 50000
