"""Central configuration for SupeRR Selector pipeline."""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

# Layer artifact dirs
L1_DIR = ARTIFACTS / "l1"
L2_DIR = ARTIFACTS / "l2"
L3_DIR = ARTIFACTS / "l3"
L4_DIR = ARTIFACTS / "l4"
L5_DIR = ARTIFACTS / "l5"
L6_DIR = ARTIFACTS / "l6"
L7_DIR = ARTIFACTS / "l7"

for _d in [L1_DIR, L2_DIR, L3_DIR, L4_DIR, L5_DIR, L6_DIR, L7_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Input CSV paths ───────────────────────────────────────────────────────────
CSV = {
    "competitions":  DATA_DIR / "competitions.csv",
    "teams":         DATA_DIR / "teams.csv",
    "player_map":    DATA_DIR / "competition_team_player_mapping.csv",
    "matches":       DATA_DIR / "matches.csv",
    "playing_eleven": DATA_DIR / "playing_eleven.csv",
}

# ── Backtest gate thresholds ─────────────────────────────────────────────────
# Phase-1 gates (no ball-by-ball data). Phase-2 will tighten these.
# vs_random_lift: random baseline drawn from FULL squad registered for the match
# (not just the top-15 model shortlist) — fairer comparison.
GATE = {
    "precision_at_11": 0.55,
    "captain_accuracy": 0.30,
    "top3_overlap":     0.65,
    "vs_random_lift":   1.05,   # Phase-1 achievable; Phase-2 target: 1.20
}

# ── Team composition constraints ─────────────────────────────────────────────
SQUAD_CONSTRAINTS = {
    "total":    11,
    "min_wk":   1,
    "min_bat":  3,
    "min_bowl": 3,
    "min_ar":   0,
}

# ── Feature engineering windows ──────────────────────────────────────────────
FORM_WINDOWS = [5, 10]   # last N matches for recent form

# ── Bayesian prior for Beta distribution ─────────────────────────────────────
BETA_PRIOR_ALPHA = 1.0   # weak prior: 1 "virtual" selection
BETA_PRIOR_BETA  = 1.0   # weak prior: 1 "virtual" non-selection

# ── Rating weights ────────────────────────────────────────────────────────────
RATING_WEIGHTS = {
    "selection_rate": 0.40,
    "captain_rate":   0.25,
    "motm_rate":      0.20,
    "recent_form":    0.15,
}

# ── Model params ──────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5
