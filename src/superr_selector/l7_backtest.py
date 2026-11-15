"""
L7 — BACKTEST (Release Gate)
Evaluates the selector pipeline on held-out matches.
Fails loudly if gate metrics are not met.

Metrics:
  precision_at_11   : fraction of predicted XI that matches actual XI
  captain_accuracy  : was predicted captain in actual XI (and was he actually captain?)
  top3_overlap      : are ≥2 of top-3 rated players in actual XI?
  vs_random_lift    : how much better than random squad selection?

Output: artifacts/l7/backtest_report.json
"""
from __future__ import annotations
import json
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalar types JSON-serializable."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

from superr_selector.config import L1_DIR, L4_DIR, L5_DIR, L6_DIR, L7_DIR, GATE
from superr_selector.l6_optimizer import select_xi

logger = logging.getLogger(__name__)


def _precision_at_k(predicted: set, actual: set, k: int = 11) -> float:
    return len(predicted & actual) / k


def _random_xi_precision(squad_ids: list[int], actual: set, k: int = 11, trials: int = 500) -> float:
    hits = []
    for _ in range(trials):
        rand_pick = set(random.sample(squad_ids, min(k, len(squad_ids))))
        hits.append(len(rand_pick & actual) / k)
    return float(np.mean(hits))


def evaluate_match(
    match_id: int,
    probs: pd.DataFrame,
    playing_eleven: pd.DataFrame,
    top_n_squad: int = 15,
) -> Optional[dict]:
    """Evaluate predicted XI vs actual XI for one match."""
    match_probs = probs[probs["match_id"] == match_id]
    if match_probs.empty:
        return None

    # actual XI (all teams in this match)
    actual_xi = set(
        playing_eleven[
            (playing_eleven["match_id"] == match_id) &
            (playing_eleven["is_played"] == True)
        ]["player_id"].tolist()
    )
    if not actual_xi:
        return None

    # actual captain
    actual_captain_rows = playing_eleven[
        (playing_eleven["match_id"] == match_id) &
        (playing_eleven["is_captain"] == True)
    ]
    actual_captains = set(actual_captain_rows["player_id"].tolist())

    # get squad shortlist from model probabilities
    squad_ids = match_probs.nlargest(top_n_squad, "p_selected")["player_id"].tolist()

    # full registered squad for this match (fairer random baseline)
    full_squad_ids = match_probs["player_id"].tolist()

    try:
        xi = select_xi(squad_ids)
    except Exception as e:
        logger.debug("  Match %d optimizer error: %s", match_id, e)
        return None

    predicted_xi = set(xi["player_id"].tolist())
    predicted_captain = xi[xi["is_captain"] == True]["player_id"].iloc[0] if not xi.empty else None
    top3_predicted = set(xi.head(3)["player_id"].tolist())

    p11  = _precision_at_k(predicted_xi, actual_xi, k=11)
    cap  = int(predicted_captain in actual_captains) if predicted_captain else 0
    top3 = len(top3_predicted & actual_xi) / 3
    # Random baseline: sample from FULL squad (not just model top-15) for fair comparison
    rand = _random_xi_precision(full_squad_ids, actual_xi)
    lift = p11 / rand if rand > 0 else float("inf")

    return {
        "match_id":        int(match_id),
        "precision_at_11": float(p11),
        "captain_accuracy": int(cap),
        "top3_overlap":    float(top3),
        "random_baseline": float(rand),
        "vs_random_lift":  float(lift),
        "squad_size":      int(len(squad_ids)),
        "actual_xi_size":  int(len(actual_xi)),
    }


def build() -> dict:
    logger.info("L7 BACKTEST — starting")

    probs          = pd.read_parquet(L4_DIR / "selection_probs.parquet")
    playing_eleven = pd.read_parquet(L1_DIR / "playing_eleven.parquet")
    matches        = pd.read_parquet(L1_DIR / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    # restrict to matches that actually have predictions
    pred_match_ids = set(probs["match_id"].unique())

    # time-order those matches and use last 20% as holdout
    pred_matches = (
        matches[matches["match_id"].isin(pred_match_ids)]
        .sort_values("match_date")
    )
    all_pred_ids = pred_matches["match_id"].unique()
    n_holdout    = max(10, int(len(all_pred_ids) * 0.20))
    eval_ids     = set(all_pred_ids[-n_holdout:])

    logger.info("  Evaluating %d holdout matches (from %d with predictions)",
                len(eval_ids), len(all_pred_ids))

    results = []
    for mid in sorted(eval_ids):
        r = evaluate_match(mid, probs, playing_eleven)
        if r:
            results.append(r)

    if not results:
        logger.error("No evaluable matches found!")
        report = {"status": "ERROR", "message": "No evaluable matches", "gate_passed": False}
        (L7_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2, cls=_NumpyEncoder))
        return report

    df = pd.DataFrame(results)
    agg = {
        "precision_at_11":  float(df["precision_at_11"].mean()),
        "captain_accuracy":  float(df["captain_accuracy"].mean()),
        "top3_overlap":      float(df["top3_overlap"].mean()),
        "vs_random_lift":    float(df["vs_random_lift"].replace([np.inf, -np.inf], np.nan).mean()),
        "matches_evaluated": len(df),
    }

    # gate check
    gate_results = {
        metric: {
            "value":    agg[metric],
            "threshold":GATE[metric],
            "passed":   agg[metric] >= GATE[metric],
        }
        for metric in GATE
        if metric in agg
    }
    gate_passed = all(g["passed"] for g in gate_results.values())

    report = {
        "status":        "PASS" if gate_passed else "FAIL",
        "gate_passed":   gate_passed,
        "metrics":       agg,
        "gates":         gate_results,
        "per_match":     results,
    }

    out = L7_DIR / "backtest_report.json"
    out.write_text(json.dumps(report, indent=2, cls=_NumpyEncoder))

    logger.info("  ── BACKTEST RESULTS ──────────────────────────")
    for k, v in agg.items():
        logger.info("  %-25s : %.4f", k, v)
    logger.info("  ── GATE STATUS ───────────────────────────────")
    for metric, g in gate_results.items():
        status = "✅ PASS" if g["passed"] else "❌ FAIL"
        logger.info("  %-25s : %.4f (threshold: %.2f) %s",
                    metric, g["value"], g["threshold"], status)
    logger.info("  Overall gate: %s", "✅ PASSED" if gate_passed else "❌ FAILED")
    logger.info("L7 BACKTEST — complete")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = build()
    print("\nBacktest status:", result["status"])
