"""
L5 — PLAYER RATING (Bayesian)
Computes a composite rating per player using Beta posterior + weighted scoring.

Rating Formula:
  β_alpha = BETA_PRIOR_ALPHA + games_played       (posterior successes)
  β_beta  = BETA_PRIOR_BETA  + (games_in_squad - games_played)  (posterior failures)
  
  selection_score = β_alpha / (β_alpha + β_beta)   # Beta posterior mean
  ci_low, ci_high = Beta(0.025, 0.975)             # 95% credible interval
  
  experience_weight = log1p(games_played) / log1p(max_games)   # 0–1
  form_weight       = 0.5*recent_form_5 + 0.5*recent_form_10   # 0–1
  
  raw_rating = (
      WEIGHTS["selection_rate"] × selection_score
    + WEIGHTS["captain_rate"]   × captain_rate
    + WEIGHTS["motm_rate"]      × motm_rate × 5   # scale up rare event
    + WEIGHTS["recent_form"]    × form_weight
  )
  
  rating = raw_rating × experience_weight  # penalise low-experience players
  
  Normalized rating = percentile rank (0–100) across all players.

Output: artifacts/l5/player_ratings.parquet
  Columns: player_id, player_name, role, rating, rating_pct,
           ci_low, ci_high, selection_score,
           experience_weight, form_weight, raw_rating
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from superr_selector.config import (
    L2_DIR, L3_DIR, L5_DIR,
    BETA_PRIOR_ALPHA, BETA_PRIOR_BETA, RATING_WEIGHTS,
)

logger = logging.getLogger(__name__)


def _beta_posterior(
    games_played: pd.Series,
    games_in_squad: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (posterior_mean, ci_low, ci_high)."""
    alpha = BETA_PRIOR_ALPHA + games_played
    beta  = BETA_PRIOR_BETA  + (games_in_squad - games_played)

    mean   = alpha / (alpha + beta)
    ci_low = pd.Series(
        [beta_dist.ppf(0.025, a, b) for a, b in zip(alpha, beta)],
        index=games_played.index
    )
    ci_high = pd.Series(
        [beta_dist.ppf(0.975, a, b) for a, b in zip(alpha, beta)],
        index=games_played.index
    )
    return mean, ci_low, ci_high


def build() -> pd.DataFrame:
    logger.info("L5 RATING — starting")

    features = pd.read_parquet(L2_DIR / "features.parquet")
    roles    = pd.read_parquet(L3_DIR / "player_roles.parquet")

    df = features.merge(roles[["player_id", "role"]], on="player_id", how="left")
    df["role"] = df["role"].fillna("AR")

    # ── Bayesian selection score ─────────────────────────────────────────────
    sel_mean, ci_low, ci_high = _beta_posterior(
        df["games_played"].fillna(0),
        df["games_in_squad"].fillna(1),
    )
    df["selection_score"] = sel_mean
    df["ci_low"]   = ci_low
    df["ci_high"]  = ci_high

    # ── Weights ───────────────────────────────────────────────────────────────
    max_games = df["games_played"].max()
    df["experience_weight"] = np.log1p(df["games_played"]) / np.log1p(max_games)

    rf5  = df["recent_form_5"].fillna(df["selection_rate"])
    rf10 = df["recent_form_10"].fillna(df["selection_rate"])
    df["form_weight"] = 0.5 * rf5 + 0.5 * rf10

    # ── Raw rating ────────────────────────────────────────────────────────────
    W = RATING_WEIGHTS
    df["raw_rating"] = (
        W["selection_rate"] * df["selection_score"]
      + W["captain_rate"]   * df["captain_rate"].fillna(0)
      + W["motm_rate"]      * df["motm_rate"].fillna(0) * 5.0
      + W["recent_form"]    * df["form_weight"]
    )

    # ── Experience-adjusted rating ───────────────────────────────────────────
    df["rating"] = df["raw_rating"] * df["experience_weight"]

    # ── Percentile rank (0–100) ───────────────────────────────────────────────
    df["rating_pct"] = df["rating"].rank(pct=True) * 100

    result = df[[
        "player_id", "player_name", "role", "rating", "rating_pct",
        "ci_low", "ci_high", "selection_score",
        "experience_weight", "form_weight", "raw_rating",
        "captain_rate", "motm_rate", "selection_rate",
        "games_played", "games_in_squad", "recent_form_5",
    ]].sort_values("rating", ascending=False).reset_index(drop=True)

    out = L5_DIR / "player_ratings.parquet"
    result.to_parquet(out, index=False)
    logger.info("  ✓ ratings → %s  (%d players)", out.name, len(result))
    logger.info("  Top 5 players by rating:")
    for _, row in result.head(5).iterrows():
        logger.info("    %s — %.4f (%.0f%%CI: %.3f–%.3f)",
                    row["player_name"], row["rating"], row["rating_pct"],
                    row["ci_low"], row["ci_high"])
    logger.info("L5 RATING — complete ✅")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
