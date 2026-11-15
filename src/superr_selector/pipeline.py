"""
Master pipeline orchestrator — runs all layers in order.
Usage:
    python -m superr_selector.pipeline          # run all layers
    python -m superr_selector.pipeline --layer 3 # run from layer 3 onwards
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _layer_exists(layer_dir: Path, expected_file: str) -> bool:
    return (layer_dir / expected_file).exists()


def run(start_layer: int = 1, force: bool = False) -> None:
    from superr_selector.config import (
        L1_DIR, L2_DIR, L3_DIR, L4_DIR, L5_DIR, L6_DIR, L7_DIR
    )

    layers = [
        (1, "l1_ingestion",      L1_DIR, "matches.parquet",         "Ingestion"),
        (2, "l2_features",       L2_DIR, "features.parquet",        "Feature Engineering"),
        (3, "l3_roles",          L3_DIR, "player_roles.parquet",    "Role Classification"),
        (4, "l4_selection_model",L4_DIR, "selection_model.joblib",  "Selection Model"),
        (5, "l5_rating",         L5_DIR, "player_ratings.parquet",  "Player Rating"),
        (6, "l6_optimizer",      L6_DIR, "optimal_xi.parquet",      "Team Optimizer"),
        (7, "l7_backtest",       L7_DIR, "backtest_report.json",    "Backtest"),
    ]

    total_start = time.time()
    logger.info("=" * 60)
    logger.info("  SupeRR Selector — Phase 1 Pipeline")
    logger.info("=" * 60)

    for layer_num, module_name, artifact_dir, sentinel, label in layers:
        if layer_num < start_layer:
            logger.info("  L%d %-25s SKIPPED (before start_layer=%d)",
                        layer_num, label, start_layer)
            continue

        if not force and _layer_exists(artifact_dir, sentinel):
            logger.info("  L%d %-25s CACHED  (%s exists)",
                        layer_num, label, sentinel)
            continue

        logger.info("")
        logger.info("  ── L%d %s ──", layer_num, label.upper())
        t0 = time.time()
        try:
            mod = __import__(f"superr_selector.{module_name}", fromlist=["build"])
            mod.build()
            elapsed = time.time() - t0
            logger.info("  L%d %-25s DONE  (%.1fs)", layer_num, label, elapsed)
        except Exception as e:
            logger.error("  L%d %-25s FAILED: %s", layer_num, label, e)
            raise

    total = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1fs", total)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SupeRR Selector Pipeline")
    parser.add_argument("--layer", type=int, default=1,
                        help="Start from this layer (1-7)")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if artifacts exist")
    args = parser.parse_args()
    run(start_layer=args.layer, force=args.force)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
