"""
L2 — FEATURE ENGINEERING
Reads L1 parquet artifacts → computes per-player historical feature matrix.
Output: artifacts/l2/features.parquet

Features computed per player (across all their historical matches):
  - games_in_squad     : total matches where player was in registered squad
  - games_played       : matches where is_played=True
  - selection_rate     : games_played / games_in_squad
  - captain_count      : times named captain
  - captain_rate       : captain_count / games_played
  - wk_count           : times named wicket-keeper
  - wk_rate            : wk_count / games_played
  - batting_order_mean : average batting position when played
  - batting_order_std  : std of batting position
  - motm_count         : man of the match awards
  - motm_rate          : motm_count / games_played
  - home_sel_rate      : selection rate in home matches
  - away_sel_rate      : selection rate in away matches
  - knockout_sel_rate  : selection rate in knockout matches
  - league_sel_rate    : selection rate in league matches
  - toss_bat_sel_rate  : selection rate when team chose to bat
  - toss_field_sel_rate: selection rate when team chose to field
  - recent_form_5      : selection rate in last 5 squad appearances
  - recent_form_10     : selection rate in last 10 squad appearances
  - competitions_count : number of distinct competitions played
  - teams_count        : number of distinct teams represented
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from superr_selector.config import L1_DIR, L2_DIR, FORM_WINDOWS

logger = logging.getLogger(__name__)


def _load_l1() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(L1_DIR / f"{name}.parquet")
        for name in ["competitions", "teams", "player_map", "matches", "playing_eleven"]
    }


def _build_squad_appearances(player_map: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each (comp_id, team_id, player_id) → list of match_ids where they were in squad.
    Returns long table: (match_id, team_id, player_id, in_squad=True)
    """
    # matches have a home and away team
    home = matches[["match_id", "comp_id", "match_date", "home_team_id",
                    "knockout", "venue_country", "toss_win_team_id", "toss_opted"]].copy()
    home.rename(columns={"home_team_id": "team_id"}, inplace=True)
    home["is_home"] = True

    away = matches[["match_id", "comp_id", "match_date", "away_team_id",
                    "knockout", "venue_country", "toss_win_team_id", "toss_opted"]].copy()
    away.rename(columns={"away_team_id": "team_id"}, inplace=True)
    away["is_home"] = False

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.dropna(subset=["team_id"])
    team_matches["team_id"] = team_matches["team_id"].astype(int)

    # join players in squad per competition
    squad = player_map[["comp_id", "team_id", "player_id"]].copy()
    appearances = squad.merge(team_matches, on=["comp_id", "team_id"], how="inner")

    # toss context — did player's team win toss and choose to bat?
    appearances["team_toss_won"] = (
        appearances["toss_win_team_id"] == appearances["team_id"]
    )
    appearances["toss_bat"] = appearances["team_toss_won"] & (
        appearances["toss_opted"].str.lower() == "batting"
    )
    appearances["toss_field"] = appearances["team_toss_won"] & (
        appearances["toss_opted"].str.lower() == "fielding"
    )

    return appearances.sort_values(["player_id", "match_date"]).reset_index(drop=True)


def _merge_playing_status(appearances: pd.DataFrame, playing_eleven: pd.DataFrame) -> pd.DataFrame:
    """Add is_played, is_captain, is_wk, batting_order from playing_eleven."""
    pe = playing_eleven[["match_id", "team_id", "player_id",
                          "is_played", "is_captain", "is_wk", "batting_order"]].copy()
    merged = appearances.merge(pe, on=["match_id", "team_id", "player_id"], how="left")
    merged["is_played"]  = merged["is_played"].fillna(False)
    merged["is_captain"] = merged["is_captain"].fillna(False)
    merged["is_wk"]      = merged["is_wk"].fillna(False)
    merged["batting_order"] = merged["batting_order"].where(merged["is_played"])
    return merged


def _recent_form(group: pd.DataFrame, windows: list[int]) -> dict:
    """Compute rolling selection rate for each window (looking backwards)."""
    result = {}
    played = group["is_played"].astype(float).tolist()
    for w in windows:
        if len(played) < 2:
            result[f"recent_form_{w}"] = float(played[-1]) if played else np.nan
        else:
            # last w appearances BEFORE current
            last = played[-w:] if len(played) >= w else played
            result[f"recent_form_{w}"] = float(np.mean(last))
    return result


def _compute_player_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate full history per player."""
    rows = []
    for player_id, grp in merged.groupby("player_id"):
        grp = grp.sort_values("match_date")
        played = grp[grp["is_played"]]

        n_squad   = len(grp)
        n_played  = int(grp["is_played"].sum())
        sel_rate  = n_played / n_squad if n_squad > 0 else 0.0

        cap_count = int(grp["is_captain"].sum())
        wk_count  = int(grp["is_wk"].sum())
        cap_rate  = cap_count / n_played if n_played > 0 else 0.0
        wk_rate   = wk_count  / n_played if n_played > 0 else 0.0

        bo_mean = played["batting_order"].mean() if n_played > 0 else np.nan
        bo_std  = played["batting_order"].std()  if n_played > 1 else 0.0

        # MoM — computed from matches table (passed as extra col if available)
        motm_count = int(grp.get("is_motm", pd.Series([0]*len(grp))).sum())
        motm_rate  = motm_count / n_played if n_played > 0 else 0.0

        # home / away
        home_grp  = grp[grp["is_home"] == True]
        away_grp  = grp[grp["is_home"] == False]
        home_sel  = home_grp["is_played"].mean() if len(home_grp) > 0 else np.nan
        away_sel  = away_grp["is_played"].mean() if len(away_grp) > 0 else np.nan

        # knockout / league
        ko_grp     = grp[grp["knockout"] == True]
        lg_grp     = grp[grp["knockout"] == False]
        ko_sel     = ko_grp["is_played"].mean() if len(ko_grp) > 0 else np.nan
        lg_sel     = lg_grp["is_played"].mean() if len(lg_grp) > 0 else np.nan

        # toss context
        bat_grp    = grp[grp["toss_bat"] == True]
        fld_grp    = grp[grp["toss_field"] == True]
        bat_sel    = bat_grp["is_played"].mean() if len(bat_grp) > 0 else np.nan
        fld_sel    = fld_grp["is_played"].mean() if len(fld_grp) > 0 else np.nan

        # recent form
        form = _recent_form(grp, FORM_WINDOWS)

        # breadth
        comp_count = grp["comp_id"].nunique()
        team_count = grp["team_id"].nunique()

        # player name (last seen)
        player_name = grp.iloc[-1].get("player_name", "")

        rows.append({
            "player_id":          player_id,
            "player_name":        player_name,
            "games_in_squad":     n_squad,
            "games_played":       n_played,
            "selection_rate":     sel_rate,
            "captain_count":      cap_count,
            "captain_rate":       cap_rate,
            "wk_count":           wk_count,
            "wk_rate":            wk_rate,
            "batting_order_mean": bo_mean,
            "batting_order_std":  bo_std,
            "motm_count":         motm_count,
            "motm_rate":          motm_rate,
            "home_sel_rate":      home_sel,
            "away_sel_rate":      away_sel,
            "knockout_sel_rate":  ko_sel,
            "league_sel_rate":    lg_sel,
            "toss_bat_sel_rate":  bat_sel,
            "toss_field_sel_rate":fld_sel,
            "recent_form_5":      form[f"recent_form_5"],
            "recent_form_10":     form[f"recent_form_10"],
            "competitions_count": comp_count,
            "teams_count":        team_count,
        })

    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    """Run L2: compute feature matrix from L1 artifacts."""
    logger.info("L2 FEATURES — starting")
    data = _load_l1()

    # attach MoM flag to player_map context
    matches = data["matches"].copy()
    matches["man_of_the_match"] = pd.to_numeric(
        matches["man_of_the_match"], errors="coerce"
    )

    appearances = _build_squad_appearances(data["player_map"], matches)

    # inject MoM into appearances
    motm = matches[["match_id", "man_of_the_match"]].dropna()
    motm["player_id"] = motm["man_of_the_match"].astype(int)
    motm["is_motm"] = True
    appearances = appearances.merge(
        motm[["match_id", "player_id", "is_motm"]],
        on=["match_id", "player_id"], how="left"
    )
    appearances["is_motm"] = appearances["is_motm"].fillna(False)

    # inject player names
    pnames = data["player_map"][["player_id", "player_name"]].drop_duplicates("player_id")
    appearances = appearances.merge(pnames, on="player_id", how="left")

    merged = _merge_playing_status(appearances, data["playing_eleven"])

    features = _compute_player_features(merged)

    out = L2_DIR / "features.parquet"
    features.to_parquet(out, index=False)
    logger.info("  ✓ features → %s  (%d players)", out.name, len(features))
    logger.info("L2 FEATURES — complete ✅")
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
