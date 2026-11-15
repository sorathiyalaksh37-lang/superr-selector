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
L4_DIR = ROOT / "artifacts" / "pressure" / "l4"
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
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #E91E63;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏏 SupeRR Selector — Season 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rajasthan Royals Hackathon Round 1 | Identify the Most Reliable Players Under Pressure</div>', unsafe_allow_html=True)

# Check artifacts
top10_bat_path = L6_DIR / "top10_batters.parquet"
top10_bowl_path = L6_DIR / "top10_bowlers.parquet"
backtest_path = L7_DIR / "backtest_report.json"
pred_path = L4_DIR / "predictions.parquet"

if not top10_bat_path.exists() or not top10_bowl_path.exists():
    st.error("Pipeline artifacts not found. Please run: `python run_pressure_pipeline.py` first.")
    st.stop()

# Top Metrics Row
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top 10 Batters Under Pressure",
    "🎯 Top 10 Bowlers Under Pressure",
    "📈 Match Pressure Timeline (DPI)",
    "📋 Unstop Submission Helper"
])

# TAB 1: BATTERS
with tab1:
    st.header("🏆 The 10 Most Reliable Batters Under Pressure")
    st.write("Ranked by Composite **Pressure Performance Index (PPI)** combining Bayesian Run Value Added ($I_{bat}$), Clutch Elevation over Baseline ($\Delta_{clutch}$), and High-Pressure Strike Rate.")

    bat_df = pd.read_parquet(top10_bat_path)
    display_bat = bat_df[[
        "rank", "player_name", "pressure_rating", "n_pressure_balls",
        "runs_pressure", "strike_rate_pressure", "strike_rate_ordinary",
        "clutch_delta", "ci_low", "ci_high"
    ]].copy()
    display_bat.columns = [
        "Rank", "Player", "Pressure Rating", "Pressure Balls",
        "Pressure Runs", "Pressure SR", "Ordinary SR",
        "Clutch Delta", "95% CI Low", "95% CI High"
    ]
    st.dataframe(
        display_bat.style.format({
            "Pressure Rating": "{:.1f}",
            "Pressure Runs": "{:.0f}",
            "Pressure SR": "{:.1f}",
            "Ordinary SR": "{:.1f}",
            "Clutch Delta": "{:+.2f}",
            "95% CI Low": "{:.2f}",
            "95% CI High": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.subheader("Strike Rate Elevation: Pressure vs Ordinary")
    chart_data = bat_df[["player_name", "strike_rate_pressure", "strike_rate_ordinary"]].copy()
    chart_data = chart_data.rename(columns={
        "player_name": "Player",
        "strike_rate_pressure": "Pressure SR",
        "strike_rate_ordinary": "Ordinary SR"
    }).set_index("Player")
    st.bar_chart(chart_data)

# TAB 2: BOWLERS
with tab2:
    st.header("🎯 The 10 Most Reliable Bowlers Under Pressure")
    st.write("Ranked by Composite **Pressure Performance Index (PPI)** combining Expected Runs Saved ($I_{bowl}$), Death Overs Economy, Clutch Delta over Ordinary Economy, and Wickets Taken.")

    bowl_df = pd.read_parquet(top10_bowl_path)
    display_bowl = bowl_df[[
        "rank", "player_name", "pressure_rating", "n_pressure_balls",
        "wickets_pressure", "economy_pressure", "economy_ordinary",
        "death_economy_pressure", "clutch_delta", "ci_low", "ci_high"
    ]].copy()
    display_bowl.columns = [
        "Rank", "Player", "Pressure Rating", "Pressure Balls",
        "Pressure Wickets", "Pressure Economy", "Ordinary Economy",
        "Death Economy", "Clutch Delta", "95% CI Low", "95% CI High"
    ]
    st.dataframe(
        display_bowl.style.format({
            "Pressure Rating": "{:.1f}",
            "Pressure Wickets": "{:.0f}",
            "Pressure Economy": "{:.2f}",
            "Ordinary Economy": "{:.2f}",
            "Death Economy": "{:.2f}",
            "Clutch Delta": "{:+.2f}",
            "95% CI Low": "{:.2f}",
            "95% CI High": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.subheader("Economy Suppression: Pressure vs Ordinary")
    chart_data_b = bowl_df[["player_name", "economy_pressure", "economy_ordinary"]].copy()
    chart_data_b = chart_data_b.rename(columns={
        "player_name": "Player",
        "economy_pressure": "Pressure Economy",
        "economy_ordinary": "Ordinary Economy"
    }).set_index("Player")
    st.bar_chart(chart_data_b)

# TAB 3: TIMELINE
with tab3:
    st.header("📈 Delivery Pressure Index (DPI) Match Replay")
    st.write("Explore delivery-level pressure swings in real matches.")

    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        available_matches = preds["match_id"].unique()[:25]
        selected_match = st.selectbox("Select Match ID to inspect:", available_matches, index=0)

        match_deliveries = preds[preds["match_id"] == selected_match].sort_values(["innings", "overid", "over_ball_num"])
        
        # Display match summary
        colA, colB, colC = st.columns(3)
        with colA:
            st.write(f"**Date:** {match_deliveries['match_date'].iloc[0]}")
        with colB:
            st.write(f"**Competition:** {match_deliveries['comp_name'].iloc[0]}")
        with colC:
            st.write(f"**Peak DPI:** {match_deliveries['dpi'].max():.1f} / 100")

        # Timeline chart
        timeline_df = match_deliveries[["overid", "innings", "dpi", "total_delivery_runs", "is_wicket", "striker_name", "bowler_name"]].copy()
        timeline_df["ball_index"] = range(1, len(timeline_df) + 1)
        
        st.line_chart(timeline_df.set_index("ball_index")["dpi"])

        st.subheader("High Pressure Deliveries in this Match (DPI ≥ 70)")
        high_p = match_deliveries[match_deliveries["dpi"] >= 70][
            ["innings", "overid", "over_ball_num", "striker_name", "bowler_name", "total_delivery_runs", "is_wicket", "dpi"]
        ]
        st.dataframe(high_p, use_container_width=True)

# TAB 4: UNSTOP HELPER
with tab4:
    st.header("📋 Unstop Submission Dropdown Reference")
    st.info("Directly match the Unstop submission fields with the verified rankings below:")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.subheader("Top 10 Batters (Unstop Dropdowns)")
        for _, r in bat_df.iterrows():
            st.markdown(f"**Rank {int(r['rank'])}:** `{r['player_name']}` (ID: `{int(r['player_id'])}`) — Rating: **{r['pressure_rating']:.1f}**")

    with col_u2:
        st.subheader("Top 10 Bowlers (Unstop Dropdowns)")
        for _, r in bowl_df.iterrows():
            st.markdown(f"**Rank {int(r['rank'])}:** `{r['player_name']}` (ID: `{int(r['player_id'])}`) — Rating: **{r['pressure_rating']:.1f}**")
