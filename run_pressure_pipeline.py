#!/usr/bin/env python3
"""
SupeRR Selector — Phase 2: Pressure & Clutch Analytics Pipeline
Rajasthan Royals Hackathon Season 2.

Usage:
  python run_pressure_pipeline.py              # Run complete L1->L7 pipeline
  python run_pressure_pipeline.py --top10      # Display Top 10 Batters & Bowlers
  python run_pressure_pipeline.py --api        # Start FastAPI backend
  python run_pressure_pipeline.py --dashboard  # Launch Streamlit web dashboard
"""
import sys
import logging
import argparse
from pathlib import Path

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PressurePipeline")


def run_pipeline():
    logger.info("=" * 65)
    logger.info("  🏏 SupeRR Selector — Pressure & Clutch Analytics Pipeline (Phase 2)")
    logger.info("=" * 65)

    from superr_selector.pressure import (
        l1_ingestion,
        l2_features,
        l3_pressure,
        l4_expected_models,
        l5_bayesian,
        l6_rating,
        l7_backtest,
    )

    layers = [
        ("L1 Ingestion", l1_ingestion.build),
        ("L2 Deliveries & State", l2_features.build),
        ("L3 DPI & Pressure Engine", l3_pressure.build),
        ("L4 Expected Runs & Wickets Models", l4_expected_models.build),
        ("L5 Bayesian Impact & Credible Intervals", l5_bayesian.build),
        ("L6 Pressure Ratings & Top 10 Selection", l6_rating.build),
        ("L7 Backtest & Release Gates", l7_backtest.build),
    ]

    import time
    t0 = time.time()
    for name, func in layers:
        t_start = time.time()
        logger.info(f"\n── Running {name} ──")
        func()
        logger.info(f"✓ {name} completed in {time.time() - t_start:.2f}s")

    logger.info("\n" + "=" * 65)
    logger.info(f"  🎉 Complete Phase 2 Pipeline finished in {time.time() - t0:.2f}s")
    logger.info("=" * 65)


def print_top10():
    import json
    from superr_selector.pressure.config import L6_DIR
    bat_file = L6_DIR / "top10_batters.json"
    bowl_file = L6_DIR / "top10_bowlers.json"

    if not bat_file.exists() or not bowl_file.exists():
        logger.error("Rankings not found. Running pipeline first...")
        run_pipeline()

    bat = json.loads(bat_file.read_text())
    bowl = json.loads(bowl_file.read_text())

    print("\n" + "=" * 75)
    print("  🏆 TOP 10 BATTERS UNDER PRESSURE (Season 2 Unstop Dropdowns)")
    print("=" * 75)
    for b in bat:
        print(f"  Rank {b['rank']:2d} : {b['player_name']:<24} (ID: {b['player_id']:<5}) | "
              f"Rating: {b['pressure_rating']:5.1f} | Balls: {b['n_pressure_balls']:4d} | "
              f"SR: {b['strike_rate_pressure']:5.1f} | Clutch Delta: {b['clutch_delta']:+5.2f}")

    print("\n" + "=" * 75)
    print("  🎯 TOP 10 BOWLERS UNDER PRESSURE (Season 2 Unstop Dropdowns)")
    print("=" * 75)
    for b in bowl:
        print(f"  Rank {b['rank']:2d} : {b['player_name']:<24} (ID: {b['player_id']:<5}) | "
              f"Rating: {b['pressure_rating']:5.1f} | Balls: {b['n_pressure_balls']:4d} | "
              f"Econ: {b['economy_pressure']:5.2f} | Wkts: {b['wickets_pressure']:2.0f} | "
              f"Clutch Delta: {b['clutch_delta']:+5.2f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SupeRR Selector Phase 2 Pipeline")
    parser.add_argument("--top10", action="store_true", help="Display Top 10 rankings")
    parser.add_argument("--api", action="store_true", help="Start FastAPI REST server")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.top10:
        print_top10()
    elif args.api:
        import uvicorn
        uvicorn.run("superr_selector.pressure.l8_api:app", host="0.0.0.0", port=8000, reload=True)
    elif args.dashboard:
        import subprocess
        subprocess.run(["streamlit", "run", "src/superr_selector/pressure/dashboard.py"])
    else:
        run_pipeline()
        print_top10()
