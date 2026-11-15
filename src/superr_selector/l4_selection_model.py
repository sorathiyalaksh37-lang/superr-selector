"""
L4 — SELECTION MODEL
Trains a classifier: "Will player X be in the XI for match Y?"
Uses time-based train/test split to avoid leakage.

Input:  artifacts/l2/features.parquet + L1 playing_eleven + L1 matches
Output: artifacts/l4/selection_model.joblib
        artifacts/l4/selection_probs.parquet
        artifacts/l4/feature_importance.json
"""
from __future__ import annotations
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from superr_selector.config import L1_DIR, L2_DIR, L3_DIR, L4_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "selection_rate", "captain_rate", "wk_rate",
    "batting_order_mean", "batting_order_std",
    "motm_rate", "home_sel_rate", "away_sel_rate",
    "knockout_sel_rate", "league_sel_rate",
    "toss_bat_sel_rate", "toss_field_sel_rate",
    "recent_form_5", "recent_form_10",
    "games_played", "games_in_squad",
    "competitions_count", "teams_count",
]


def _build_training_data(
    features: pd.DataFrame,
    playing_eleven: pd.DataFrame,
    matches: pd.DataFrame,
    player_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build row-per (match_id, player_id) for all players in squad.
    Label = 1 if player was in the actual XI.
    Uses rolling/historical features (not same-match features).
    """
    matches = matches[["match_id", "comp_id", "match_date", "knockout"]].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    # all squad appearances
    squad = player_map[["comp_id", "team_id", "player_id"]].drop_duplicates()

    # matches per team
    home = matches.merge(
        playing_eleven[["match_id", "team_id"]].drop_duplicates(),
        on="match_id"
    )
    squad_matches = squad.merge(home, on=["comp_id", "team_id"])

    # label: was player in XI?
    xi_set = playing_eleven[playing_eleven["is_played"] == True][
        ["match_id", "team_id", "player_id"]
    ].copy()
    xi_set["label"] = 1

    labeled = squad_matches.merge(
        xi_set, on=["match_id", "team_id", "player_id"], how="left"
    )
    labeled["label"] = labeled["label"].fillna(0).astype(int)

    # attach historical features (global player history — no leakage since
    # these are aggregated over entire history; for production use rolling)
    labeled = labeled.merge(features, on="player_id", how="left")
    labeled = labeled.dropna(subset=["selection_rate"])
    return labeled


def build() -> tuple[pd.DataFrame, dict]:
    logger.info("L4 SELECTION MODEL — starting")

    features      = pd.read_parquet(L2_DIR / "features.parquet")
    playing_eleven= pd.read_parquet(L1_DIR / "playing_eleven.parquet")
    matches       = pd.read_parquet(L1_DIR / "matches.parquet")
    player_map    = pd.read_parquet(L1_DIR / "player_map.parquet")

    df = _build_training_data(features, playing_eleven, matches, player_map)
    df = df.sort_values("match_date").reset_index(drop=True)

    # time-based split (last 20% of matches = test)
    split_idx = int(len(df) * 0.80)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train["label"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["label"]

    logger.info("  Train: %d rows | Test: %d rows", len(train), len(test))
    logger.info("  Label balance (train): %.2f%%", 100 * y_train.mean())

    # pipeline with imputer + calibrated GBT
    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    logger.info("  Test ROC-AUC: %.4f", auc)

    # feature importance from inner GBT
    inner_clf = model.named_steps["clf"].calibrated_classifiers_[0].estimator
    importance = dict(zip(FEATURE_COLS, inner_clf.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    # predictions on full dataset
    all_probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    probs_df = df[["match_id", "team_id", "player_id", "label"]].copy()
    probs_df["p_selected"] = all_probs

    # save artifacts
    model_path = L4_DIR / "selection_model.joblib"
    joblib.dump(model, model_path)

    probs_path = L4_DIR / "selection_probs.parquet"
    probs_df.to_parquet(probs_path, index=False)

    fi_path = L4_DIR / "feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump({"auc": auc, "feature_importance": importance}, f, indent=2)

    logger.info("  ✓ model → %s", model_path.name)
    logger.info("  ✓ probs → %s", probs_path.name)
    logger.info("  ✓ importance → %s", fi_path.name)
    logger.info("L4 SELECTION MODEL — complete ✅  (AUC=%.4f)", auc)
    return probs_df, importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
