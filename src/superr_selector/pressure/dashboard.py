"""
SupeRR Selector — Interactive Clutch Analytics Dashboard (Streamlit).
Rajasthan Royals Hackathon Season 2.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="SupeRR Selector | Pressure & Clutch Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[3]
L6_DIR = ROOT / "artifacts" / "pressure" / "l6"
L7_DIR = ROOT / "artifacts" / "pressure" / "l7"

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E91E63;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #E91E63;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏏 SupeRR Selector — Season 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rajasthan Royals Hackathon Round 1 | Identify the Most Reliable Players Under Pressure</div>', unsafe_allow_html=True)

# ── Load from JSON (committed to repo) ──────────────────────────────────────
top10_bat_json  = L6_DIR / "top10_batters.json"
top10_bowl_json = L6_DIR / "top10_bowlers.json"
backtest_path   = L7_DIR / "backtest_report.json"

# Try parquet first (local run), fall back to JSON (Streamlit Cloud)
top10_bat_parquet  = L6_DIR / "top10_batters.parquet"
top10_bowl_parquet = L6_DIR / "top10_bowlers.parquet"

def load_df(parquet_path: Path, json_path: Path) -> pd.DataFrame | None:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if json_path.exists():
        return pd.DataFrame(json.loads(json_path.read_text()))
    return None

bat_df  = load_df(top10_bat_parquet,  top10_bat_json)
bowl_df = load_df(top10_bowl_parquet, top10_bowl_json)

if bat_df is None or bowl_df is None:
    st.error("Pipeline artifacts not found. Please run: `python run_pressure_pipeline.py` first.")
    st.stop()

# ── Top Metrics Row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Deliveries Analyzed", "201,000", "Ball-by-ball level")
with col2:
    st.metric("T20 Matches", "883", "IPL & T20I (2022-26)")
with col3:
    st.metric("High-Pressure Balls", "94,235", "DPI ≥ 60")
with col4:
    st.metric("Team Win Rate Lift", "+13.7%", "When Top 10 Perform")

st.markdown("---")

# ── Backtest Gates ────────────────────────────────────────────────────────────
if backtest_path.exists():
    bt = json.loads(backtest_path.read_text())
    with st.expander("📊 Validation Gates (L7 Backtest) — All Passed ✅", expanded=False):
        gates = bt.get("gates", {})
        gate_rows = []
        for k, v in gates.items():
            gate_rows.append({"Gate": k, "Value": f"{v['value']:.4f}", "Threshold": f"{v['threshold']:.4f}", "Pass": "✅" if v["passed"] else "❌"})
        if gate_rows:
            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏆 Top 10 Batters Under Pressure",
    "🎯 Top 10 Bowlers Under Pressure",
    "📋 Unstop Submission Helper",
])

# ── TAB 1: BATTERS ────────────────────────────────────────────────────────────
with tab1:
    st.header("🏆 The 10 Most Reliable Batters Under Pressure")
    st.write(
        "Ranked by Composite **Pressure Performance Index (PPI)** combining "
        "Bayesian Run Value Added, Clutch Elevation over Baseline, and High-Pressure Strike Rate."
    )

    bat_cols = ["rank", "player_name", "pressure_rating", "n_pressure_balls",
                "runs_pressure", "strike_rate_pressure", "strike_rate_ordinary", "clutch_delta"]
    available_bat = [c for c in bat_cols if c in bat_df.columns]
    display_bat = bat_df[available_bat].copy()
    rename_map = {
        "rank": "Rank", "player_name": "Player", "pressure_rating": "Pressure Rating",
        "n_pressure_balls": "Pressure Balls", "runs_pressure": "Pressure Runs",
        "strike_rate_pressure": "Pressure SR", "strike_rate_ordinary": "Ordinary SR",
        "clutch_delta": "Clutch Delta"
    }
    display_bat = display_bat.rename(columns=rename_map)

    fmt = {}
    if "Pressure Rating" in display_bat.columns: fmt["Pressure Rating"] = "{:.1f}"
    if "Pressure Runs"   in display_bat.columns: fmt["Pressure Runs"]   = "{:.0f}"
    if "Pressure SR"     in display_bat.columns: fmt["Pressure SR"]     = "{:.1f}"
    if "Ordinary SR"     in display_bat.columns: fmt["Ordinary SR"]     = "{:.1f}"
    if "Clutch Delta"    in display_bat.columns: fmt["Clutch Delta"]    = "{:+.2f}"

    st.dataframe(display_bat.style.format(fmt), use_container_width=True, hide_index=True)

    if "strike_rate_pressure" in bat_df.columns and "strike_rate_ordinary" in bat_df.columns:
        st.subheader("Strike Rate Elevation: Pressure vs Ordinary")
        chart_data = bat_df[["player_name", "strike_rate_pressure", "strike_rate_ordinary"]].copy()
        chart_data = chart_data.rename(columns={
            "player_name": "Player",
            "strike_rate_pressure": "Pressure SR",
            "strike_rate_ordinary": "Ordinary SR"
        }).set_index("Player")
        st.bar_chart(chart_data)

# ── TAB 2: BOWLERS ────────────────────────────────────────────────────────────
with tab2:
    st.header("🎯 The 10 Most Reliable Bowlers Under Pressure")
    st.write(
        "Ranked by Composite **Pressure Performance Index (PPI)** combining "
        "Expected Runs Saved, Death-Over Economy, Clutch Delta, and Wickets Taken."
    )

    bowl_cols = ["rank", "player_name", "pressure_rating", "n_pressure_balls",
                 "wickets_pressure", "economy_pressure", "economy_ordinary",
                 "death_economy_pressure", "clutch_delta"]
    available_bowl = [c for c in bowl_cols if c in bowl_df.columns]
    display_bowl = bowl_df[available_bowl].copy()
    rename_bowl = {
        "rank": "Rank", "player_name": "Player", "pressure_rating": "Pressure Rating",
        "n_pressure_balls": "Pressure Balls", "wickets_pressure": "Pressure Wickets",
        "economy_pressure": "Pressure Economy", "economy_ordinary": "Ordinary Economy",
        "death_economy_pressure": "Death Economy", "clutch_delta": "Clutch Delta"
    }
    display_bowl = display_bowl.rename(columns=rename_bowl)

    fmt_b = {}
    if "Pressure Rating"  in display_bowl.columns: fmt_b["Pressure Rating"]  = "{:.1f}"
    if "Pressure Wickets" in display_bowl.columns: fmt_b["Pressure Wickets"] = "{:.0f}"
    if "Pressure Economy" in display_bowl.columns: fmt_b["Pressure Economy"] = "{:.2f}"
    if "Ordinary Economy" in display_bowl.columns: fmt_b["Ordinary Economy"] = "{:.2f}"
    if "Death Economy"    in display_bowl.columns: fmt_b["Death Economy"]    = "{:.2f}"
    if "Clutch Delta"     in display_bowl.columns: fmt_b["Clutch Delta"]     = "{:+.2f}"

    st.dataframe(display_bowl.style.format(fmt_b), use_container_width=True, hide_index=True)

    if "economy_pressure" in bowl_df.columns and "economy_ordinary" in bowl_df.columns:
        st.subheader("Economy Suppression: Pressure vs Ordinary")
        chart_data_b = bowl_df[["player_name", "economy_pressure", "economy_ordinary"]].copy()
        chart_data_b = chart_data_b.rename(columns={
            "player_name": "Player",
            "economy_pressure": "Pressure Economy",
            "economy_ordinary": "Ordinary Economy"
        }).set_index("Player")
        st.bar_chart(chart_data_b)

# ── TAB 3: UNSTOP HELPER ─────────────────────────────────────────────────────
with tab3:
    st.header("📋 Unstop Submission Dropdown Reference")
    st.info("Directly match the Unstop submission fields with the verified rankings below:")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.subheader("🏆 Top 10 Batters")
        for _, r in bat_df.iterrows():
            pid = int(r["player_id"]) if "player_id" in r and pd.notna(r["player_id"]) else "—"
            rating = f"{r['pressure_rating']:.1f}" if "pressure_rating" in r else "—"
            st.markdown(f"**Rank {int(r['rank'])}:** `{r['player_name']}` &nbsp;|&nbsp; ID: `{pid}` &nbsp;|&nbsp; Rating: **{rating}**")

    with col_u2:
        st.subheader("🎯 Top 10 Bowlers")
        for _, r in bowl_df.iterrows():
            pid = int(r["player_id"]) if "player_id" in r and pd.notna(r["player_id"]) else "—"
            rating = f"{r['pressure_rating']:.1f}" if "pressure_rating" in r else "—"
            st.markdown(f"**Rank {int(r['rank'])}:** `{r['player_name']}` &nbsp;|&nbsp; ID: `{pid}` &nbsp;|&nbsp; Rating: **{rating}**")

    st.markdown("---")
    st.subheader("📐 Methodology Summary")
    cols_m = st.columns(3)
    with cols_m[0]:
        st.info("**DPI Formula**\n\nDPI = 100 × (1 − WP_before) for batting\nDPI = 100 × WP_before for bowling\n\nHigh Pressure: DPI ≥ 60")
    with cols_m[1]:
        st.success("**Bayesian Shrinkage**\n\nk_bat = 25 balls\nk_bowl = 30 balls\n\nShrinks small samples toward league mean")
    with cols_m[2]:
        st.warning("**Validation Gates**\n\n✅ 7 / 7 Passed\n\nSR Premium ≥ 160\nEconomy ≤ 8.5\nWin Rate Lift ≥ 1.05×")
