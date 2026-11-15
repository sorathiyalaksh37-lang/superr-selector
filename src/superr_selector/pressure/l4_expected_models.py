"""
L4 Expected Models — Train expected runs (xR) and expected wicket (xW) models.
Computes:
  I_bat        : Raw batter impact above expectation
  I_bowl       : Raw bowler impact above expectation
  PI_bat       : Pressure-weighted batter impact (I_bat * DPI / 50)
  PI_bowl      : Pressure-weighted bowler impact (I_bowl * DPI / 50)
  PI_ordinary  : Impact in ordinary conditions (DPI < 40)

Outputs: artifacts/pressure/l4/predictions.parquet + models/*.joblib
"""
from __future__ import annotations
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

from superr_selector.pressure.config import L3_DIR, L4_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

MODELS_DIR = L4_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build() -> Path:
    logger.info("Pressure L4 EXPECTED MODELS — starting")
    in_file = L3_DIR / "dpi.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing L3 artifact: {in_file}")

    df = pd.read_parquet(in_file)
    logger.info("  Training expected models on %d deliveries", len(df))

    # Feature engineering for expectation models
    df["pace_or_spin_code"] = (df["pace_or_spin"].str.lower() == "spin").astype(int)
    feature_cols = [
        "overid", "innings", "phase_id", "wickets_in_hand",
        "balls_remaining", "required_run_rate_clipped",
        "current_run_rate_clipped", "pace_or_spin_code", "batting_order"
    ]
    X = df[feature_cols].fillna(0)

    # 1. Expected Runs Model (xR)
    logger.info("  Fitting Expected Runs (xR) model...")
    y_runs = df["total_delivery_runs"]
    xr_model = LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    xr_model.fit(X, y_runs)
    df["xR"] = xr_model.predict(X)
    joblib.dump(xr_model, MODELS_DIR / "xr_model.joblib")

    # 2. Expected Wickets Model (xW)
    logger.info("  Fitting Expected Wickets (xW) model...")
    y_wickets = df["is_wicket"].astype(int)
    xw_model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    xw_model.fit(X, y_wickets)
    df["xW"] = xw_model.predict_proba(X)[:, 1]
    joblib.dump(xw_model, MODELS_DIR / "xw_model.joblib")

    logger.info("  Models saved. Computing delivery impacts...")

    # 3. Compute Value Added / Impacts
    # Wicket value in T20 runs equivalent
    WICKET_VALUE_RUNS = 15.0

    # Batter impact: runs scored minus expected, penalized if dismissed
    df["I_bat"] = (df["total_delivery_runs"] - df["xR"]) - np.where(
        df["is_wicket"], WICKET_VALUE_RUNS * (1.0 - df["xW"]), 0.0
    )

    # Bowler impact: expected runs minus conceded, plus wicket bonus
    df["I_bowl"] = (df["xR"] - df["total_delivery_runs"]) + np.where(
        df["is_wicket"], WICKET_VALUE_RUNS * (1.0 - df["xW"]), -WICKET_VALUE_RUNS * df["xW"]
    )

    # 4. Pressure-weighted impacts
    pressure_weight = df["dpi"] / 50.0
    df["PI_bat"] = df["I_bat"] * pressure_weight
    df["PI_bowl"] = df["I_bowl"] * pressure_weight

    # 5. Ordinary baseline impacts (DPI < 40)
    df["I_bat_ordinary"] = np.where(df["is_ordinary"], df["I_bat"], np.nan)
    df["I_bowl_ordinary"] = np.where(df["is_ordinary"], df["I_bowl"], np.nan)

    # 6. High pressure impacts (DPI >= 60)
    df["I_bat_high_pressure"] = np.where(df["is_high_pressure"], df["I_bat"], np.nan)
    df["I_bowl_high_pressure"] = np.where(df["is_high_pressure"], df["I_bowl"], np.nan)

    out_file = L4_DIR / "predictions.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("  Saved predictions: %s (%d rows)", out_file.name, len(df))
    logger.info("Pressure L4 EXPECTED MODELS — complete ✅")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
