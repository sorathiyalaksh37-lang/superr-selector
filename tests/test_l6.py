"""Tests for L6 Optimizer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import pandas as pd
from superr_selector.config import L5_DIR
from superr_selector.l6_optimizer import select_xi


def _get_squad(n=22):
    ratings = pd.read_parquet(L5_DIR / "player_ratings.parquet")
    return ratings.head(n)["player_id"].tolist()


def test_xi_has_11_players():
    xi = select_xi(_get_squad(22))
    assert len(xi) == 11, f"XI should have 11 players, got {len(xi)}"


def test_xi_exactly_one_captain():
    xi = select_xi(_get_squad(22))
    assert xi["is_captain"].sum() == 1


def test_xi_exactly_one_vc():
    xi = select_xi(_get_squad(22))
    assert xi["is_vc"].sum() == 1


def test_captain_and_vc_different():
    xi = select_xi(_get_squad(22))
    cap = xi[xi["is_captain"]]["player_id"].iloc[0]
    vc  = xi[xi["is_vc"]]["player_id"].iloc[0]
    assert cap != vc


def test_all_from_squad():
    squad = _get_squad(22)
    xi = select_xi(squad)
    assert xi["player_id"].isin(squad).all()


def test_ratings_sorted_descending():
    xi = select_xi(_get_squad(22))
    cap_rating = xi[xi["is_captain"]]["rating"].iloc[0]
    vc_rating  = xi[xi["is_vc"]]["rating"].iloc[0]
    assert cap_rating >= vc_rating, "Captain should have higher rating than VC"
