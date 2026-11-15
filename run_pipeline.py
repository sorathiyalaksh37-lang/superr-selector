#!/usr/bin/env python3
"""
SupeRR Selector — Single entry point.

Usage:
  python run_pipeline.py                  # run full pipeline
  python run_pipeline.py --layer 3        # resume from layer 3
  python run_pipeline.py --force          # rerun all layers
  python run_pipeline.py --predict-xi 12345  # predict XI for a match
"""
import sys
import logging
from pathlib import Path

# ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SupeRR Selector Phase 1 Pipeline")
    parser.add_argument("--layer", type=int, default=1, help="Start from this layer (1-7)")
    parser.add_argument("--force", action="store_true", help="Rerun all layers")
    parser.add_argument("--predict-xi", type=int, metavar="MATCH_ID",
                        help="Print predicted XI for a match_id (requires pipeline to have run)")
    args = parser.parse_args()

    if args.predict_xi:
        from superr_selector.l6_optimizer import build_for_match
        xi = build_for_match(args.predict_xi)
        print(f"\n{'='*55}")
        print(f"  🏏 Predicted XI for Match {args.predict_xi}")
        print(f"{'='*55}")
        for _, row in xi.iterrows():
            cap = " 👑 CAPTAIN"  if row["is_captain"] else ""
            vc  = " ⭐ VC"       if row["is_vc"]      else ""
            wk  = " 🧤 WK"       if row["is_wk_role"] else ""
            print(f"  [{row['role']:4}] {row['player_name']:<30} "
                  f"Rating: {row['rating']:.4f}{cap}{vc}{wk}")
        print(f"{'='*55}\n")
    else:
        from superr_selector.pipeline import run
        run(start_layer=args.layer, force=args.force)
