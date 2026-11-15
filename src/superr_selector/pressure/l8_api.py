"""
L8 FastAPI Server — Serves pressure ratings, top 10 rankings, match pressure timelines.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from superr_selector.pressure.config import L4_DIR, L6_DIR, L7_DIR

app = FastAPI(
    title="SupeRR Selector — Pressure & Clutch Analytics API",
    description="Rajasthan Royals Hackathon Season 2 — Delivery Pressure Index (DPI) & Clutch Rankings",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "project": "SupeRR Selector — Season 2",
        "hackathon": "Rajasthan Royals Hackathon Round 1",
        "description": "Identify the Most Reliable Players Under Pressure",
        "endpoints": [
            "/api/top10-batters",
            "/api/top10-bowlers",
            "/api/player/{player_id}",
            "/api/match/{match_id}/timeline",
            "/api/backtest-report",
            "/api/overview",
        ],
    }


@app.get("/api/overview")
def get_overview():
    ratings_path = L6_DIR / "ratings.parquet"
    if not ratings_path.exists():
        raise HTTPException(status_code=503, detail="Pipeline not run yet")
    df = pd.read_parquet(ratings_path)
    return {
        "total_rated_players": len(df),
        "total_batters": int((df["role"] == "BAT").sum()),
        "total_bowlers": int((df["role"] == "BOWL").sum()),
        "mean_pressure_rating": float(df["pressure_rating"].mean()),
    }


@app.get("/api/top10-batters")
def get_top10_batters():
    path = L6_DIR / "top10_batters.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Rankings not generated yet")
    return json.loads(path.read_text())


@app.get("/api/top10-bowlers")
def get_top10_bowlers():
    path = L6_DIR / "top10_bowlers.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Rankings not generated yet")
    return json.loads(path.read_text())


@app.get("/api/player/{player_id}")
def get_player(player_id: int):
    ratings_path = L6_DIR / "ratings.parquet"
    if not ratings_path.exists():
        raise HTTPException(status_code=503, detail="Pipeline not run yet")
    df = pd.read_parquet(ratings_path)
    player = df[df["player_id"] == player_id]
    if player.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found in rated pool")
    return player.iloc[0].to_dict()


@app.get("/api/match/{match_id}/timeline")
def get_match_timeline(match_id: int):
    pred_path = L4_DIR / "predictions.parquet"
    if not pred_path.exists():
        raise HTTPException(status_code=503, detail="Predictions not generated yet")
    df = pd.read_parquet(pred_path)
    match_df = df[df["match_id"] == match_id].sort_values(["innings", "overid", "over_ball_num"])
    if match_df.empty:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
    
    cols = [
        "innings", "overid", "over_ball_num", "striker_name", "bowler_name",
        "runs", "is_wicket", "total_delivery_runs", "dpi", "p", "c_hat",
        "innings_runs_before", "wickets_in_hand", "phase"
    ]
    return {
        "match_id": match_id,
        "total_deliveries": len(match_df),
        "deliveries": match_df[cols].to_dict(orient="records"),
    }


@app.get("/api/backtest-report")
def get_backtest_report():
    path = L7_DIR / "backtest_report.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Backtest not run yet")
    return json.loads(path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("superr_selector.pressure.l8_api:app", host="0.0.0.0", port=8000, reload=True)
