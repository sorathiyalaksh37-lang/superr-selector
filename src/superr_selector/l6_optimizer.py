"""
L6 — TEAM OPTIMIZER (Integer Linear Program)
Selects optimal fantasy XI from a squad using player ratings.

Solver stack (tried in order):
  1. scipy.optimize.milp  — fast, built-in, no external binary needed
  2. Greedy fallback       — always works, slightly sub-optimal

Problem:
  Maximize: Σ rating_i × x_i
  Subject to:
    Σ x_i = 11                     (exactly 11 players)
    Σ x_i [role=WK]  ≥ 1
    Σ x_i [role=BAT] ≥ 3
    Σ x_i [role=BOWL]≥ 3
    x_i ∈ {0, 1}

Captain = highest-rated selected player
VC      = second-highest-rated selected player
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from superr_selector.config import L5_DIR, L6_DIR, SQUAD_CONSTRAINTS

logger = logging.getLogger(__name__)


# ── Primary solver: scipy milp ────────────────────────────────────────────────

def _ilp_scipy(
    squad_df: pd.DataFrame,
    n_total: int,
    min_wk: int,
    min_bat: int,
    min_bowl: int,
) -> pd.DataFrame:
    """
    Solve via scipy.optimize.milp (available since scipy 1.7, no external binary).
    Maximise Σ rating_i * x_i  subject to role and count constraints.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    n       = len(squad_df)
    ratings = squad_df["rating"].values
    roles   = squad_df["role"].values
    integrality = np.ones(n)   # all variables are integer (binary via bounds)
    bounds      = Bounds(lb=0, ub=1)

    # Constraint matrix rows:
    #   [0] Σ x_i  == n_total          (equality)
    #   [1] Σ x_i[WK]  >= min_wk
    #   [2] Σ x_i[BAT] >= min_bat
    #   [3] Σ x_i[BOWL]>= min_bowl

    rows = []
    lbs  = []
    ubs  = []

    # Equality: Σ x_i = n_total
    eq_row = np.ones(n)
    rows.append(eq_row); lbs.append(n_total); ubs.append(n_total)

    for role, min_count in [("WK", min_wk), ("BAT", min_bat), ("BOWL", min_bowl)]:
        row = (roles == role).astype(float)
        rows.append(row); lbs.append(min_count); ubs.append(n)

    A   = np.vstack(rows)
    lbs = np.array(lbs, dtype=float)
    ubs = np.array(ubs, dtype=float)

    constraint = LinearConstraint(A, lbs, ubs)

    # milp minimises → negate ratings
    result = milp(
        c            = -ratings,
        constraints  = constraint,
        integrality  = integrality,
        bounds       = bounds,
    )

    if not result.success:
        raise RuntimeError(f"scipy milp failed: {result.message}")

    selected_mask = result.x > 0.5
    return squad_df[selected_mask].copy()


# ── Fallback: greedy ──────────────────────────────────────────────────────────

def _greedy_select(
    squad_df: pd.DataFrame,
    n_total: int,
    min_wk: int,
    min_bat: int,
    min_bowl: int,
) -> pd.DataFrame:
    """
    Greedy: fill mandatory role slots first (highest rated), then fill remainder.
    Always feasible as long as squad has enough players of each role.
    """
    selected_ids = []
    remaining    = squad_df.copy()

    def _pick(role: str, n: int):
        nonlocal remaining
        sub = remaining[remaining["role"] == role].nlargest(n, "rating")
        selected_ids.extend(sub["player_id"].tolist())
        remaining = remaining[~remaining["player_id"].isin(sub["player_id"])]

    _pick("WK",   min_wk)
    _pick("BAT",  min_bat)
    _pick("BOWL", min_bowl)

    still_need = n_total - len(selected_ids)
    top_rest = remaining.nlargest(still_need, "rating")
    selected_ids.extend(top_rest["player_id"].tolist())

    return squad_df[squad_df["player_id"].isin(selected_ids)].copy()


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _optimize(
    squad_df: pd.DataFrame,
    n_total: int = 11,
    min_wk: int = 1,
    min_bat: int = 3,
    min_bowl: int = 3,
) -> pd.DataFrame:
    """Try scipy milp first, fall back to greedy."""
    try:
        result = _ilp_scipy(squad_df, n_total, min_wk, min_bat, min_bowl)
        logger.debug("  ILP (scipy) solved for %d players", len(result))
        return result
    except Exception as e:
        logger.debug("  ILP failed (%s), using greedy fallback", e)
        return _greedy_select(squad_df, n_total, min_wk, min_bat, min_bowl)


# ── Public API ────────────────────────────────────────────────────────────────

def select_xi(
    squad_player_ids: list[int],
    comp_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Select optimal fantasy XI from a registered squad.

    Args:
        squad_player_ids : player_ids in the squad for this match
        comp_id          : optional — future use for comp-specific ratings

    Returns:
        DataFrame (11 rows):
          player_id, player_name, role, rating, rating_pct,
          is_captain, is_vc, is_wk_role,
          captain_rate, motm_rate, selection_rate
    """
    ratings  = pd.read_parquet(L5_DIR / "player_ratings.parquet")
    squad_df = ratings[ratings["player_id"].isin(squad_player_ids)].copy()

    if len(squad_df) < 11:
        logger.warning(
            "Only %d rated players in squad (need 11). Padding with rating=0.",
            len(squad_df),
        )
        missing = set(squad_player_ids) - set(squad_df["player_id"])
        pad = pd.DataFrame({
            "player_id":   list(missing),
            "player_name": [f"Player_{p}" for p in missing],
            "role":        ["AR"] * len(missing),
            "rating":      [0.0]  * len(missing),
            "rating_pct":  [0.0]  * len(missing),
            "captain_rate":[0.0]  * len(missing),
            "motm_rate":   [0.0]  * len(missing),
            "selection_rate":[0.0]* len(missing),
        })
        squad_df = pd.concat([squad_df, pad], ignore_index=True)

    c  = SQUAD_CONSTRAINTS
    xi = _optimize(
        squad_df,
        n_total  = c["total"],
        min_wk   = c["min_wk"],
        min_bat  = c["min_bat"],
        min_bowl = c["min_bowl"],
    )

    # Assign captain & VC by rating rank
    xi = xi.sort_values("rating", ascending=False).reset_index(drop=True)
    xi["is_captain"] = False
    xi["is_vc"]      = False
    if len(xi) >= 1: xi.loc[0, "is_captain"] = True
    if len(xi) >= 2: xi.loc[1, "is_vc"]      = True

    xi["is_wk_role"] = xi["role"] == "WK"

    keep = ["player_id", "player_name", "role", "rating", "rating_pct",
            "is_captain", "is_vc", "is_wk_role",
            "captain_rate", "motm_rate", "selection_rate"]
    return xi[[c for c in keep if c in xi.columns]]


def build_for_match(match_id: int) -> pd.DataFrame:
    """Build optimal XI for a specific match from L4 selection probabilities."""
    probs      = pd.read_parquet(L6_DIR.parent / "l4" / "selection_probs.parquet")
    match_rows = probs[probs["match_id"] == match_id]
    if match_rows.empty:
        raise ValueError(f"No predictions found for match_id={match_id}")

    # top 15 by probability → simulate match-day squad shortlist
    squad_ids = match_rows.nlargest(15, "p_selected")["player_id"].tolist()
    xi        = select_xi(squad_ids)
    xi["match_id"] = match_id
    return xi


def build() -> None:
    """Demo build: generate optimal XI for the 5 most recent predicted matches."""
    logger.info("L6 OPTIMIZER — starting")
    probs    = pd.read_parquet(L6_DIR.parent / "l4" / "selection_probs.parquet")
    all_xis  = []

    sample_ids = probs["match_id"].unique()[-5:]
    for match_id in sample_ids:
        try:
            xi = build_for_match(match_id)
            all_xis.append(xi)
            logger.info("  match %d → %d players selected", match_id, len(xi))
        except Exception as e:
            logger.warning("  match %d skipped: %s", match_id, e)

    if all_xis:
        result = pd.concat(all_xis, ignore_index=True)
        out    = L6_DIR / "optimal_xi.parquet"
        result.to_parquet(out, index=False)
        logger.info("  ✓ optimal_xi → %s  (%d rows)", out.name, len(result))

    logger.info("L6 OPTIMIZER — complete ✅")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
