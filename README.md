# 🏏 SupeRR Selector (Season 2)
### Rajasthan Royals Data Science Hackathon — Round 1 Winning Submission

![Tests](https://img.shields.io/badge/Tests-21%2F21%20Passed-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Architecture](https://img.shields.io/badge/Architecture-8--Layer%20Parquet%20Pipeline-purple)
![Gates](https://img.shields.io/badge/Release%20Gates-All%20Passed-brightgreen)

An end-to-end AI sports analytics engine that **identifies the most reliable cricket players under pressure** across 201,000+ deliveries and predicts optimal fantasy lineups for T20 matches (IPL & T20Is, 2022–2026).

---

## 🏆 Official Round 1 Submission Rankings (Unstop Dropdowns)

### 🥇 Top 10 Batters Under Pressure

| Rank | Player Name | Player ID | Pressure Rating | Pressure Balls | Pressure Runs | Pressure SR | Clutch Delta ($\Delta_{clutch}$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Vaibhav Sooryavanshi** | `50284` | **99.5** | 427 | 901 | 211.0 | +0.55 |
| **2** | **Finn Allen** | `4847` | **88.3** | 322 | 600 | 186.3 | -0.07 |
| **3** | **Ayush Mhatre** | `48504` | **88.1** | 185 | 342 | 184.9 | +5.47 |
| **4** | **Abhishek Sharma** | `6281` | **87.9** | 995 | 1801 | 181.0 | -0.13 |
| **5** | **Priyansh Arya** | `25669` | **87.8** | 345 | 641 | 185.8 | +1.16 |
| **6** | **Rajat Patidar** | `5039` | **87.8** | 665 | 1094 | 164.5 | +0.71 |
| **7** | **Suryakumar Yadav** | `1699` | **87.1** | 1,430 | 2,372 | 165.9 | +0.05 |
| **8** | **Heinrich Klaasen** | `4260` | **87.1** | 827 | 1,349 | 163.1 | +0.18 |
| **9** | **Naman Dhir** | `37052` | **87.0** | 368 | 620 | 168.5 | +0.60 |
| **10** | **Rilee Rossouw** | `772` | **86.9** | 233 | 392 | 168.2 | +0.46 |

### 🎯 Top 10 Bowlers Under Pressure

| Rank | Player Name | Player ID | Pressure Rating | Pressure Balls | Wickets (Press) | Economy (Press) | Death Economy |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Jamie Overton** | `2382` | **99.5** | 310 | 29 | 7.94 | 9.03 |
| **2** | **Abbas Afridi** | `7699` | **94.8** | 275 | 28 | 8.51 | 8.81 |
| **3** | **Taskin Ahmed** | `2681` | **93.0** | 407 | 33 | 7.83 | 8.56 |
| **4** | **Mujeeb Ur Rahman** | `6450` | **92.1** | 214 | 13 | 5.94 | 7.92 |
| **5** | **Dushan Hemantha** | `27504` | **91.7** | 97 | 15 | 6.25 | 3.39 |
| **6** | **Umran Malik** | `25048` | **91.4** | 330 | 30 | 8.91 | 9.43 |
| **7** | **Abrar Ahmed** | `5979` | **91.1** | 338 | 23 | 7.28 | 7.65 |
| **8** | **Mohammad Nawaz** | `2490` | **89.7** | 436 | 29 | 7.44 | 9.15 |
| **9** | **Josh Hazlewood** | `830` | **89.4** | 756 | 57 | 8.64 | 9.71 |
| **10** | **Kuldeep Yadav** | `3644` | **89.3** | 1,129 | 66 | 7.62 | 8.11 |

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  L1  INGESTION          ball_by_ball.csv + matches.csv + competitions.csv   │
│      ↓ pandera validate                                                    │
│  L2  FEATURES           clean_deliveries.parquet → state_deliveries.parquet│
│      ↓ add state, extras, target, RRR, wickets in hand                     │
│  L3  PRESSURE           dpi.parquet (+ Win Probability & reference tables) │
│      ↓ μ, p, ĉ, p_w, V_t, DPI ∈ [0, 100]                                   │
│  L4  EXPECTED MODELS    models/*.joblib + predictions.parquet              │
│      ↓ I_bat, I_bowl, PI_bat, PI_bowl, PI_ordinary                         │
│  L5  BAYESIAN           player_impact.parquet (Empirical Bayes + 95% CI)   │
│      ↓ E, R, Clutch Delta (Δ_clutch)                                       │
│  L6  RATING             ratings.parquet (Top 10 Batters & Bowlers)         │
│      ↓ drivers, sensitivity, unstop dropdown tables                        │
│  L7  BACKTEST           backtest_report.json (automated release gates)     │
│      ↓                                                                     │
│  L8  API + DASHBOARD    FastAPI backend + Streamlit interactive web demo   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Execution

### 1. Run Complete Pressure & Clutch Pipeline (Phase 2)
```bash
python run_pressure_pipeline.py
```
*Runs all 7 layers (L1–L7) with validation in under 5 seconds!*

### 2. View Top 10 Dropdowns for Unstop
```bash
python run_pressure_pipeline.py --top10
```

### 3. Launch FastAPI REST Server
```bash
python run_pressure_pipeline.py --api
# Open http://localhost:8000/docs for Swagger API documentation
```

### 4. Launch Interactive Streamlit Dashboard
```bash
python run_pressure_pipeline.py --dashboard
# Opens interactive web app with match pressure replay, leaderboards, radar charts
```

### 5. Run Fantasy Selection Engine (Phase 1)
```bash
python run_pipeline.py
python run_pipeline.py --predict-xi 14094
```

### 6. Run Test Suite
```bash
python -m pytest tests/ -v
# 21/21 tests passing
```

---

## 📄 Submission Files
- **Methodology Report**: `Methodology_Report.md` (Detailed academic & statistical report)
- **Top 10 Batters JSON**: `artifacts/pressure/l6/top10_batters.json`
- **Top 10 Bowlers JSON**: `artifacts/pressure/l6/top10_bowlers.json`
- **Backtest Validation Report**: `artifacts/pressure/l7/backtest_report.json`
