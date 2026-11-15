"""
L3 — ROLE ENGINE
Classifies each player into one of: BAT, BOWL, WK, AR (all-rounder).
Reads: artifacts/l2/features.parquet
Output: artifacts/l3/player_roles.parquet

Rules (heuristic, interpretable — judges love explainability):
  WK   → wk_rate >= 0.5
  BAT  → batting_order_mean <= 4.5
  BOWL → batting_order_mean >= 7.5  OR  (batting_order_mean is NaN AND selection_rate < 0.5)
  AR   → everything else (5 <= batting_order_mean < 7.5)

If batting_order_mean is NaN (player never batted / no data):
  → default to AR if selection_rate > 0.3 else BOWL
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from superr_selector.config import L2_DIR, L3_DIR

logger = logging.getLogger(__name__)

ROLE_PRIORITY = ["WK", "BAT", "BOWL", "AR"]  # precedence order


def classify_role(row: pd.Series) -> str:
    if row["wk_rate"] >= 0.5:
        return "WK"
    bo = row["batting_order_mean"]
    if pd.isna(bo):
        return "AR" if row["selection_rate"] > 0.3 else "BOWL"
    if bo <= 4.5:
        return "BAT"
    if bo >= 7.5:
        return "BOWL"
    return "AR"


def build() -> pd.DataFrame:
    logger.info("L3 ROLES — starting")
    features = pd.read_parquet(L2_DIR / "features.parquet")

    roles = features[["player_id", "player_name", "batting_order_mean",
                       "wk_rate", "selection_rate"]].copy()
    roles["role"] = roles.apply(classify_role, axis=1)

    # summary
    dist = roles["role"].value_counts().to_dict()
    logger.info("  Role distribution: %s", dist)

    out = L3_DIR / "player_roles.parquet"
    roles[["player_id", "player_name", "role"]].to_parquet(out, index=False)
    logger.info("  ✓ player_roles → %s  (%d players)", out.name, len(roles))
    logger.info("L3 ROLES — complete ✅")
    return roles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
