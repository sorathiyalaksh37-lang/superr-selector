"""
L3 Delivery Pressure Index (DPI) & State Dynamics.
Computes:
  μ   : Expected runs per ball for current state
  p   : Win probability at delivery
  c_hat: Contest closeness (1 - 2*|p - 0.5|)
  p_w : Probability of wicket in current state
  V_t : Velocity / urgency required
  DPI : Delivery Pressure Index normalized to [0, 100]

Outputs: artifacts/pressure/l3/dpi.parquet + frozen reference tables.
"""
from __future__ import annotations
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from superr_selector.pressure.config import L2_DIR, L3_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


def build() -> Path:
    logger.info("Pressure L3 DPI ENGINE — starting")
    in_file = L2_DIR / "state_deliveries.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing L2 artifact: {in_file}")

    df = pd.read_parquet(in_file)
    logger.info("  Processing %d deliveries for pressure modeling", len(df))

    # 1. Empirical State Reference Tables (μ and p_w)
    ref_table = (
        df.groupby(["phase", "wickets_in_hand"])
        .agg(
            mu=("total_delivery_runs", "mean"),
            p_w=("is_wicket", "mean")
        )
        .reset_index()
    )
    ref_table_path = L3_DIR / "state_reference_table.parquet"
    ref_table.to_parquet(ref_table_path, index=False)
    logger.info("  Saved state reference table: %s", ref_table_path.name)

    df = df.merge(ref_table, on=["phase", "wickets_in_hand"], how="left")
    df["mu"] = df["mu"].fillna(1.3)
    df["p_w"] = df["p_w"].fillna(0.05)

    # 2. Win Probability Models (p)
    # Use LightGBM for non-linear, robust win probability estimation
    from lightgbm import LGBMClassifier

    df["required_run_rate_clipped"] = np.clip(df["required_run_rate"], 0.0, 36.0)
    df["current_run_rate_clipped"] = np.clip(df["current_run_rate"], 0.0, 36.0)

    # Innings 2 Chase WP Model
    chase_mask = df["is_chase"] & (df["balls_remaining"] > 0)
    chase_features = ["runs_needed", "balls_remaining", "wickets_in_hand", "required_run_rate_clipped"]

    wp_chase_model = LGBMClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    X_chase = df.loc[chase_mask, chase_features].fillna(0)
    y_chase = df.loc[chase_mask, "batting_team_won"]
    wp_chase_model.fit(X_chase, y_chase)

    df["p"] = 0.5  # default
    df.loc[chase_mask, "p"] = wp_chase_model.predict_proba(X_chase)[:, 1]

    # Innings 1 WP Model
    inn1_mask = (~df["is_chase"]) & (df["innings_legal_balls"] > 0)
    inn1_features = ["innings_runs_before", "innings_legal_balls", "wickets_in_hand", "current_run_rate_clipped"]
    wp_inn1_model = LGBMClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    X_inn1 = df.loc[inn1_mask, inn1_features].fillna(0)
    y_inn1 = df.loc[inn1_mask, "batting_team_won"]
    wp_inn1_model.fit(X_inn1, y_inn1)

    df.loc[inn1_mask, "p"] = wp_inn1_model.predict_proba(X_inn1)[:, 1]

    # Save WP models
    joblib.dump({"chase": wp_chase_model, "inn1": wp_inn1_model}, L3_DIR / "wp_models.joblib")

    # 3. Contest Closeness (c_hat)
    # When p == 0.5, c_hat = 1.0 (dead even). When p == 0.0 or 1.0, c_hat = 0.0 (decided).
    df["c_hat"] = np.clip(1.0 - 2.0 * np.abs(df["p"] - 0.5), 0.0, 1.0)

    # 4. Required Velocity (V_t)
    df["V_t"] = np.where(
        df["is_chase"],
        df["runs_needed"] / np.maximum(1, df["balls_remaining"]),
        df["innings_runs_before"] / np.maximum(1, df["innings_legal_balls"])
    )

    # 5. Composite Delivery Pressure Index (DPI)
    # Factors:
    # A) Closeness Leverage: Closeness amplified by later overs
    leverage = df["c_hat"] * (1.0 + 40.0 / (df["balls_remaining"] + 20.0))

    # B) Urgency Velocity Ratio:
    urgency = np.where(
        df["is_chase"],
        np.clip(df["required_run_rate"] / 8.5, 0.4, 2.5),
        np.clip(df["current_run_rate"] / 8.0, 0.6, 1.8)
    )

    # C) Wicket Scarcity Pressure: Fewer wickets = higher pressure
    wicket_pressure = 1.0 + (10 - df["wickets_in_hand"]) / 12.0

    # D) Tournament / Match Stakes:
    stakes_mult = 1.0
    stakes_mult = stakes_mult + np.where(df["knockout"] == True, 0.20, 0.0)
    stakes_mult = stakes_mult + np.where(df["is_ipl"] == True, 0.10, 0.0)
    stakes_mult = stakes_mult + np.where(df["phase"] == "Death", 0.15, 0.0)

    raw_dpi = leverage * urgency * wicket_pressure * stakes_mult

    # Normalize DPI smoothly to [0, 100] using percentile calibration
    # Target: 5th percentile ~ 20, median ~ 45, 90th percentile ~ 75, max ~ 100
    p25 = np.percentile(raw_dpi, 25)
    p75 = np.percentile(raw_dpi, 75)
    iqr = max(1e-3, p75 - p25)
    scaled = (raw_dpi - p25) / iqr * 25.0 + 45.0
    df["dpi"] = np.clip(scaled, 5.0, 100.0)

    # Pressure categories
    df["is_high_pressure"] = df["dpi"] >= 60.0
    df["is_ordinary"] = df["dpi"] < 40.0

    logger.info("  DPI Summary: Min=%.1f, Median=%.1f, Mean=%.1f, Max=%.1f",
                df["dpi"].min(), df["dpi"].median(), df["dpi"].mean(), df["dpi"].max())
    logger.info("  High Pressure Deliveries: %d (%.1f%%)",
                df["is_high_pressure"].sum(), df["is_high_pressure"].mean() * 100)
    logger.info("  Ordinary Deliveries: %d (%.1f%%)",
                df["is_ordinary"].sum(), df["is_ordinary"].mean() * 100)

    out_file = L3_DIR / "dpi.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("  Saved DPI deliveries: %s (%d rows)", out_file.name, len(df))
    logger.info("Pressure L3 DPI ENGINE — complete ✅")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
