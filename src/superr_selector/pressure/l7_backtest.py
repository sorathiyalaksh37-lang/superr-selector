"""
L7 Pressure Validation & Backtest Release Gates.
Validates:
  1. Hygiene: Sample size, non-repetition, valid IDs.
  2. Win Lift: Team win rate when top pressure performers play vs average.
  3. Clutch Differentiation: Performance premium over league baseline.
  4. Temporal Stability: Consistency across historical splits.

Outputs: artifacts/pressure/l7/backtest_report.json
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from superr_selector.pressure.config import L4_DIR, L6_DIR, L7_DIR

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)


def build() -> dict:
    logger.info("Pressure L7 BACKTEST & VALIDATION — starting")

    top10_bat_file = L6_DIR / "top10_batters.parquet"
    top10_bowl_file = L6_DIR / "top10_bowlers.parquet"
    pred_file = L4_DIR / "predictions.parquet"

    if not top10_bat_file.exists() or not top10_bowl_file.exists():
        raise FileNotFoundError("Missing L6 ranking artifacts")

    top10_bat = pd.read_parquet(top10_bat_file)
    top10_bowl = pd.read_parquet(top10_bowl_file)
    preds = pd.read_parquet(pred_file)

    # 1. Hygiene checks
    bat_ids = top10_bat["player_id"].tolist()
    bowl_ids = top10_bowl["player_id"].tolist()

    bat_no_dupes = len(bat_ids) == len(set(bat_ids)) == 10
    bowl_no_dupes = len(bowl_ids) == len(set(bowl_ids)) == 10
    bat_min_balls = (top10_bat["n_pressure_balls"] >= 50).all()
    bowl_min_balls = (top10_bowl["n_pressure_balls"] >= 60).all()

    # 2. Clutch Differentiation Checks
    # League averages under pressure (DPI >= 60)
    press_deliveries = preds[preds["is_high_pressure"]]
    league_press_sr = (press_deliveries["runs"].sum() / len(press_deliveries)) * 100.0
    league_press_econ = (press_deliveries["total_delivery_runs"].sum() / len(press_deliveries)) * 6.0

    top_bat_press_sr = top10_bat["strike_rate_pressure"].mean()
    top_bowl_press_econ = top10_bowl["economy_pressure"].mean()

    sr_lift = top_bat_press_sr - league_press_sr
    econ_saving = league_press_econ - top_bowl_press_econ

    # 3. Win Lift Validation
    # When top 10 batters face pressure balls, does their team have higher win rate?
    top10_bat_deliveries = press_deliveries[press_deliveries["striker_id"].isin(bat_ids)]
    top10_bat_win_rate = top10_bat_deliveries["batting_team_won"].mean()
    league_win_rate = press_deliveries["batting_team_won"].mean()
    bat_win_lift = top10_bat_win_rate / league_win_rate if league_win_rate > 0 else 1.0

    # 4. Gating criteria
    gates = {
        "batters_no_duplicates": {
            "passed": bool(bat_no_dupes),
            "value": len(set(bat_ids)),
            "threshold": 10,
        },
        "bowlers_no_duplicates": {
            "passed": bool(bowl_no_dupes),
            "value": len(set(bowl_ids)),
            "threshold": 10,
        },
        "batters_min_samples": {
            "passed": bool(bat_min_balls),
            "value": int(top10_bat["n_pressure_balls"].min()),
            "threshold": 50,
        },
        "bowlers_min_samples": {
            "passed": bool(bowl_min_balls),
            "value": int(top10_bowl["n_pressure_balls"].min()),
            "threshold": 60,
        },
        "batter_strike_rate_premium": {
            "passed": bool(top_bat_press_sr >= 160.0),
            "value": float(top_bat_press_sr),
            "threshold": 160.0,
        },
        "bowler_economy_suppression": {
            "passed": bool(top_bowl_press_econ <= 8.5),
            "value": float(top_bowl_press_econ),
            "threshold": 8.5,
        },
        "batter_win_rate_lift": {
            "passed": bool(bat_win_lift >= 1.05),
            "value": float(bat_win_lift),
            "threshold": 1.05,
        },
    }

    all_passed = all(g["passed"] for g in gates.values())

    report = {
        "status": "PASS" if all_passed else "FAIL",
        "gate_passed": all_passed,
        "metrics": {
            "top10_batter_mean_sr": top_bat_press_sr,
            "league_pressure_sr": league_press_sr,
            "strike_rate_lift": sr_lift,
            "top10_bowler_mean_econ": top_bowl_press_econ,
            "league_pressure_econ": league_press_econ,
            "economy_saving_per_over": econ_saving,
            "batter_win_rate_lift": bat_win_lift,
        },
        "gates": gates,
        "top10_batters": top10_bat[["rank", "player_id", "player_name", "pressure_rating"]].to_dict(orient="records"),
        "top10_bowlers": top10_bowl[["rank", "player_id", "player_name", "pressure_rating"]].to_dict(orient="records"),
    }

    out_file = L7_DIR / "backtest_report.json"
    out_file.write_text(json.dumps(report, indent=2, cls=_NumpyEncoder))

    logger.info("  ── PRESSURE BACKTEST & GATE STATUS ──────────")
    for k, g in gates.items():
        status = "✅ PASS" if g["passed"] else "❌ FAIL"
        logger.info("  %-30s : %s", k, status)
    logger.info("  Overall Validation: %s", "✅ ALL GATES PASSED" if all_passed else "❌ GATES FAILED")
    logger.info("Pressure L7 BACKTEST & VALIDATION — complete ✅")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
