"""
Phase 2 Configuration — SupeRR Selector: Pressure & Clutch Performance Engine.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts" / "pressure"

L1_DIR = ARTIFACTS_DIR / "l1"
L2_DIR = ARTIFACTS_DIR / "l2"
L3_DIR = ARTIFACTS_DIR / "l3"
L4_DIR = ARTIFACTS_DIR / "l4"
L5_DIR = ARTIFACTS_DIR / "l5"
L6_DIR = ARTIFACTS_DIR / "l6"
L7_DIR = ARTIFACTS_DIR / "l7"

for _d in [L1_DIR, L2_DIR, L3_DIR, L4_DIR, L5_DIR, L6_DIR, L7_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

CSV_PATHS = {
    "ball_by_ball": DATA_DIR / "ball_by_ball.csv",
    "matches": DATA_DIR / "matches.csv",
    "competitions": DATA_DIR / "competitions.csv",
    "teams": DATA_DIR / "teams.csv",
    "playing_eleven": DATA_DIR / "playing_eleven.csv",
    "player_mapping": DATA_DIR / "competition_team_player_mapping.csv",
}

# Pressure thresholds
HIGH_PRESSURE_DPI = 60.0    # DPI >= 60 is High Pressure
ORDINARY_DPI = 40.0         # DPI < 40 is Ordinary Context

# Eligibility criteria for top 10 ranking
MIN_BALLS_FACED_BATTER = 50     # Min deliveries faced in high pressure
MIN_BALLS_BOWLED_BOWLER = 60    # Min deliveries bowled in high pressure

# Bayesian prior constants
BAYESIAN_K_BATTER = 25.0        # Shrinkage strength (approx 25 balls pseudo-prior)
BAYESIAN_K_BOWLER = 30.0        # Shrinkage strength (approx 30 balls pseudo-prior)

RANDOM_STATE = 42
