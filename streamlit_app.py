"""
SupeRR Selector — Season 2 | Rajasthan Royals Hackathon
Self-contained Streamlit dashboard (works on Streamlit Cloud without local artifacts).
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SupeRR Selector | Pressure & Clutch Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hardcoded results (from pipeline run — all 7 gates passed ✅) ─────────────
BATTERS = [
    {"rank":1,"player_id":50284,"player_name":"Vaibhav Sooryavanshi","pressure_rating":99.5,"n_pressure_balls":427,"runs_pressure":901,"strike_rate_pressure":211.0,"clutch_delta":0.55},
    {"rank":2,"player_id":4847,"player_name":"Finn Allen","pressure_rating":88.3,"n_pressure_balls":322,"runs_pressure":600,"strike_rate_pressure":186.3,"clutch_delta":-0.07},
    {"rank":3,"player_id":48504,"player_name":"Ayush Mhatre","pressure_rating":88.1,"n_pressure_balls":185,"runs_pressure":342,"strike_rate_pressure":184.9,"clutch_delta":5.47},
    {"rank":4,"player_id":6281,"player_name":"Abhishek Sharma","pressure_rating":87.9,"n_pressure_balls":995,"runs_pressure":1801,"strike_rate_pressure":181.0,"clutch_delta":-0.13},
    {"rank":5,"player_id":25669,"player_name":"Priyansh Arya","pressure_rating":87.8,"n_pressure_balls":345,"runs_pressure":641,"strike_rate_pressure":185.8,"clutch_delta":1.16},
    {"rank":6,"player_id":5039,"player_name":"Rajat Patidar","pressure_rating":87.8,"n_pressure_balls":665,"runs_pressure":1094,"strike_rate_pressure":164.5,"clutch_delta":0.71},
    {"rank":7,"player_id":1699,"player_name":"Suryakumar Yadav","pressure_rating":87.1,"n_pressure_balls":1430,"runs_pressure":2372,"strike_rate_pressure":165.9,"clutch_delta":0.05},
    {"rank":8,"player_id":4260,"player_name":"Heinrich Klaasen","pressure_rating":87.1,"n_pressure_balls":827,"runs_pressure":1349,"strike_rate_pressure":163.1,"clutch_delta":0.18},
    {"rank":9,"player_id":37052,"player_name":"Naman Dhir","pressure_rating":87.0,"n_pressure_balls":368,"runs_pressure":620,"strike_rate_pressure":168.5,"clutch_delta":0.60},
    {"rank":10,"player_id":772,"player_name":"Rilee Rossouw","pressure_rating":86.9,"n_pressure_balls":233,"runs_pressure":392,"strike_rate_pressure":168.2,"clutch_delta":0.46},
]

BOWLERS = [
    {"rank":1,"player_id":2382,"player_name":"Jamie Overton","pressure_rating":99.5,"n_pressure_balls":310,"wickets_pressure":29,"economy_pressure":7.94,"death_economy_pressure":9.03,"clutch_delta":1.23},
    {"rank":2,"player_id":7699,"player_name":"Abbas Afridi","pressure_rating":94.8,"n_pressure_balls":275,"wickets_pressure":28,"economy_pressure":8.51,"death_economy_pressure":8.81,"clutch_delta":-0.09},
    {"rank":3,"player_id":2681,"player_name":"Taskin Ahmed","pressure_rating":93.0,"n_pressure_balls":407,"wickets_pressure":33,"economy_pressure":7.83,"death_economy_pressure":8.56,"clutch_delta":0.84},
    {"rank":4,"player_id":6450,"player_name":"Mujeeb Ur Rahman","pressure_rating":92.1,"n_pressure_balls":214,"wickets_pressure":13,"economy_pressure":5.94,"death_economy_pressure":7.92,"clutch_delta":0.31},
    {"rank":5,"player_id":27504,"player_name":"Dushan Hemantha","pressure_rating":91.7,"n_pressure_balls":97,"wickets_pressure":15,"economy_pressure":6.25,"death_economy_pressure":3.39,"clutch_delta":2.19},
    {"rank":6,"player_id":25048,"player_name":"Umran Malik","pressure_rating":91.4,"n_pressure_balls":330,"wickets_pressure":30,"economy_pressure":8.91,"death_economy_pressure":9.43,"clutch_delta":0.42},
    {"rank":7,"player_id":5979,"player_name":"Abrar Ahmed","pressure_rating":91.1,"n_pressure_balls":338,"wickets_pressure":23,"economy_pressure":7.28,"death_economy_pressure":7.65,"clutch_delta":0.23},
    {"rank":8,"player_id":2490,"player_name":"Mohammad Nawaz","pressure_rating":89.7,"n_pressure_balls":436,"wickets_pressure":29,"economy_pressure":7.44,"death_economy_pressure":6.00,"clutch_delta":-0.01},
    {"rank":9,"player_id":830,"player_name":"Josh Hazlewood","pressure_rating":89.4,"n_pressure_balls":756,"wickets_pressure":57,"economy_pressure":8.64,"death_economy_pressure":9.71,"clutch_delta":0.39},
    {"rank":10,"player_id":3644,"player_name":"Kuldeep Yadav","pressure_rating":89.3,"n_pressure_balls":1129,"wickets_pressure":66,"economy_pressure":7.62,"death_economy_pressure":8.61,"clutch_delta":0.40},
]

# Also try loading from JSON files if available (local run with artifacts)
ROOT = Path(__file__).parent
_bat_json  = ROOT / "artifacts" / "pressure" / "l6" / "top10_batters.json"
_bowl_json = ROOT / "artifacts" / "pressure" / "l6" / "top10_bowlers.json"

if _bat_json.exists():
    BATTERS = json.loads(_bat_json.read_text())
if _bowl_json.exists():
    BOWLERS = json.loads(_bowl_json.read_text())

bat_df  = pd.DataFrame(BATTERS)
bowl_df = pd.DataFrame(BOWLERS)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size:2.2rem; font-weight:800; color:#E91E63; }
  .sub-title  { font-size:1.05rem; color:#888; margin-bottom:18px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏏 SupeRR Selector — Season 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rajasthan Royals Hackathon Round 1 &nbsp;|&nbsp; Identify the Most Reliable Players Under Pressure</div>', unsafe_allow_html=True)

# ── Top Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Deliveries Analyzed", "201,000", "Ball-by-ball")
c2.metric("T20 Matches",         "883",     "IPL & T20I 2022-26")
c3.metric("High-Pressure Balls", "94,235",  "DPI ≥ 60")
c4.metric("Win Rate Lift",       "+13.7%",  "Top 10 effect")

st.markdown("---")

# ── Validation Gates ──────────────────────────────────────────────────────────
with st.expander("📊 L7 Validation Gates — All 7 Passed ✅"):
    gates_df = pd.DataFrame([
        {"Gate": "batter_strike_rate_premium",  "Value": "177.92", "Threshold": "≥ 160.0", "Status": "✅"},
        {"Gate": "bowler_economy_suppression",  "Value": "7.64",   "Threshold": "≤ 8.5",   "Status": "✅"},
        {"Gate": "batter_win_rate_lift",        "Value": "1.137×", "Threshold": "≥ 1.05×", "Status": "✅"},
        {"Gate": "batters_min_samples",         "Value": "185",    "Threshold": "≥ 50",    "Status": "✅"},
        {"Gate": "bowlers_min_samples",         "Value": "97",     "Threshold": "≥ 60",    "Status": "✅"},
        {"Gate": "no_duplicate_batters",        "Value": "10",     "Threshold": "unique",  "Status": "✅"},
        {"Gate": "no_duplicate_bowlers",        "Value": "10",     "Threshold": "unique",  "Status": "✅"},
    ])
    st.dataframe(gates_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top 10 Batters",
    "🎯 Top 10 Bowlers",
    "📐 Methodology",
    "📋 Submission Reference",
])

# TAB 1 ── BATTERS
with tab1:
    st.header("🏆 Most Reliable Batters Under Pressure")
    st.caption("Ranked by Composite Pressure Performance Index: Bayesian Run Value + Clutch Delta + High-Pressure Strike Rate")

    display_b = bat_df[["rank","player_name","pressure_rating","n_pressure_balls","runs_pressure","strike_rate_pressure","clutch_delta"]].copy()
    display_b.columns = ["Rank","Player","Pressure Rating","Pressure Balls","Pressure Runs","Strike Rate","Clutch Δ"]
    st.dataframe(
        display_b.style.format({
            "Pressure Rating": "{:.1f}",
            "Pressure Runs":   "{:.0f}",
            "Strike Rate":     "{:.1f}",
            "Clutch Δ":        "{:+.2f}",
        }).background_gradient(subset=["Pressure Rating"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    st.subheader("Strike Rate Under Pressure")
    chart = bat_df.set_index("player_name")[["strike_rate_pressure"]].rename(columns={"strike_rate_pressure": "Strike Rate"})
    st.bar_chart(chart)

    # League average reference line note
    st.caption("📌 League average pressure SR: 141.9 | All top-10 exceed by ≥ 15%")

# TAB 2 ── BOWLERS
with tab2:
    st.header("🎯 Most Reliable Bowlers Under Pressure")
    st.caption("Ranked by Composite Pressure Performance Index: Expected Runs Saved + Death Economy + Clutch Delta + Wickets")

    display_bw = bowl_df[["rank","player_name","pressure_rating","n_pressure_balls","wickets_pressure","economy_pressure","death_economy_pressure","clutch_delta"]].copy()
    display_bw.columns = ["Rank","Player","Pressure Rating","Pressure Balls","Wickets","Economy","Death Economy","Clutch Δ"]
    st.dataframe(
        display_bw.style.format({
            "Pressure Rating": "{:.1f}",
            "Wickets":         "{:.0f}",
            "Economy":         "{:.2f}",
            "Death Economy":   "{:.2f}",
            "Clutch Δ":        "{:+.2f}",
        }).background_gradient(subset=["Pressure Rating"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    st.subheader("Economy Under Pressure (lower = better)")
    chart_bw = bowl_df.set_index("player_name")[["economy_pressure"]].rename(columns={"economy_pressure": "Economy"})
    st.bar_chart(chart_bw)

    st.caption("📌 League average pressure economy: 8.98 | Top-10 avg: 7.64 (saves 1.34 runs/over)")

# TAB 3 ── METHODOLOGY
with tab3:
    st.header("📐 Methodology Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
**🔥 Delivery Pressure Index (DPI)**

Built with LightGBM Win Probability model trained on 200K+ deliveries.

`DPI = 100 × (1 − WP_before)` for batting  
`DPI = 100 × WP_before` for bowling

- **High Pressure:** DPI ≥ 60 (94,235 balls)
- **Ordinary:** DPI < 40 (36,267 balls)
- **Median DPI:** 58.5
        """)
    with col2:
        st.success("""
**📊 Performance Measurement**

For each player under high-pressure deliveries:

- **Batters:** Runs scored, SR, xR (Expected Runs)
- **Bowlers:** Wickets, Economy, xW (Expected Wickets)
- **Clutch Δ:** Performance vs their own ordinary baseline

Bayesian shrinkage (k=25 bat, k=30 bowl) adjusts for small samples.
        """)
    with col3:
        st.warning("""
**⚖️ Fair Comparison**

Eligibility filters:
- Batters: ≥ 50 high-pressure balls
- Bowlers: ≥ 60 high-pressure balls

Empirical Bayes shrinks low-sample players toward the league mean — preventing noise from dominating.

95% Confidence Intervals computed via bootstrap.
        """)

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.code("""
L1  Ingestion          ball_by_ball.csv + 5 CSVs → validated
L2  Feature Eng.       clean_deliveries.parquet (extras, state, target)
L3  Pressure           dpi.parquet  (LightGBM WP → DPI per delivery)
L4  Expected Models    predictions.parquet  (xR, xW per ball)
L5  Bayesian Rating    player_impact.parquet  (Empirical Bayes + CI)
L6  Rankings           ratings.parquet  → top10_batters/bowlers.json
L7  Backtest           backtest_report.json  (7 gates — all passed ✅)
L8  API + Dashboard    FastAPI + Streamlit (this app)
    """, language="text")

# TAB 4 ── SUBMISSION REFERENCE
with tab4:
    st.header("📋 Unstop Submission Reference")
    st.info("Use these rankings exactly when filling Unstop dropdowns.")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.subheader("🏆 Top 10 Batters")
        for r in BATTERS:
            st.markdown(f"**#{r['rank']}** &nbsp; `{r['player_name']}` &nbsp;— Rating: **{r['pressure_rating']:.1f}** | Balls: {r['n_pressure_balls']} | SR: {r['strike_rate_pressure']:.1f}")
    with col_u2:
        st.subheader("🎯 Top 10 Bowlers")
        for r in BOWLERS:
            st.markdown(f"**#{r['rank']}** &nbsp; `{r['player_name']}` &nbsp;— Rating: **{r['pressure_rating']:.1f}** | Balls: {r['n_pressure_balls']} | Econ: {r['economy_pressure']:.2f}")

    st.markdown("---")
    st.markdown("""
**GitHub:** https://github.com/sorathiyalaksh37-lang/superr-selector  
**Hackathon:** Rajasthan Royals SupeRR Selector Season 2 — Round 1
    """)
