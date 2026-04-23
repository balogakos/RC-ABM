# Retail Centre Agent-Based Model (RC-ABM)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-green)

> A spatially-explicit, agent-based model of retail consumer behaviour, calibrated to real-world household demographics and National Travel Survey (NTS) trip frequency data.

---

## Overview

The RC-ABM simulates how households in a study region make decisions about **where to shop**, **what for**, and **how they get there**. Each agent (household) is endowed with demographic attributes drawn from real data and makes daily shopping decisions driven by two independent systems:

1. **Grocery Replenishment (Stock-Based):** Agents maintain a household stock of grocery goods that depletes daily at an individually calibrated rate. When stock falls below a threshold, a shopping trip is triggered.
2. **Frequency-Based NTS Trips (Probability-Based):** Non-grocery trip types—Service, Comparison, Entertainment, and Food & Drink—are sampled daily against per-agent probabilities derived from the National Travel Survey.

A key feature is **Trip Chaining**: if an agent triggers multiple trips on the same day, there is a 50% chance they consolidate them into a single visit to a multi-purpose retail centre, provided one exists that satisfies all their requirements.

Destination choice is governed by pre-computed **utility scores** for each (household × retail centre) pair, incorporating exponential distance decay and centre-specific amenity presence. A **Softmax** sampling function is applied at the point of choice, ensuring probabilistic selection while weighting higher-utility alternatives.

---

## Key Features

- **Trip Chaining**: Multi-purpose trips are simulated via a stochastic chaining mechanism, where agents may consolidate two or more trip types into a single visit.
- **Joint Mode-Destination Choice**: Transport mode and destination are chosen simultaneously, emerging from a combined utility landscape rather than being pre-assigned.
- **Amenity Filtering**: Each trip type is constrained to retail centres offering the relevant facilities (e.g., a Comparison trip requires a centre with a `Retail` presence flag).
- **Adaptive Feedback Loop**: Agent utility scores are updated after each visit (+5% / -5% stochastic reinforcement), allowing visit patterns to evolve over the simulation period.
- **Retail Centre Evaluation**: An automated peer-ranking system identifies underperforming centres and applies targeted utility boosts, simulating real-world intervention policy.
- **Spatial Diffusion**: Word-of-mouth effects are modelled through postcode-sector-level diffusion, boosting the utility of popular centres for demographically similar agents in the same area.

---

## Project Structure

```
RC-ABM/
│
├── simulation/                  # Core model engine
│   ├── main.py                  # GUI entry point and daily simulation loop
│   ├── agent.py                 # All agent logic (vectorised NumPy/Pandas)
│   ├── config.py                # File paths and model constants
│   ├── assign_trip_frequencies.py   # Assigns NTS trip probabilities to agents (run once)
│   └── visualization.py         # Generates visitation maps and spatial analytics
│
├── website/                     # Streamlit web interface
│   ├── app.py                   # Streamlit dashboard (run/visualise from browser)
│   ├── templates/               # HTML templates (Flask fallback)
│   └── static/                  # Static assets
│
├── notebooks/                   # Exploratory analysis and result validation
│   ├── analysis_centre_makeup.ipynb
│   ├── analysis_results_notebook.ipynb
│   ├── latest_visit_mapping.ipynb
│   └── testing_results.ipynb
│
├── tests/                       # Unit tests and verification scripts
│   ├── tests.py
│   └── _verify.py
│
├── outputs/                     # Simulation outputs (gitignored)
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Prerequisites

Install dependencies:

```bash
pip install pandas numpy geopandas pyarrow streamlit plotly pydeck
```

### 2. One-Time Setup

Run this once whenever the underlying agent data changes. It assigns NTS trip frequency probabilities to each household:

```bash
python simulation/assign_trip_frequencies.py
```

### 3. Run the Desktop Simulation (Tkinter GUI)

```bash
python simulation/main.py
```

A configuration window will open. Set the number of agents and simulation days, then click **Run Simulation**. Results are saved to `outputs/` as a timestamped `.parquet` file alongside a visitation density map (`.png`).

### 4. Run the Streamlit Web Interface

```bash
streamlit run website/app.py
```

The dashboard provides interactive controls, live simulation progress, and post-run analytics including flow maps, market share dominance maps, and trip type breakdowns.

---

## Model Logic

### Trip Generation

Each simulation day, for every active agent:

| System | Trigger | Trip Types |
|---|---|---|
| **Stock-Based** | `Stock < REORDER_THRESHOLD` | `grocery` (bulk or convenience) |
| **Probability-Based** | `rand() < daily_prob` | `service`, `comparison`, `entertainment`, `food_drink` |

### Trip Chaining

If an agent is triggered for more than one trip on a given day, a **50/50 coin flip** determines whether they:
- **Chain**: Attempt a single visit to one retail centre that can satisfy all trip requirements simultaneously (amenity AND logic across all trip types).
- **Split**: Proceed to independent destinations for each trip (current default behaviour).

If no single centre can satisfy the combined amenity requirements, the agent falls back to split trips.

### Destination Choice

For each trip (or chain), destination selection follows this sequence:

1. **Amenity Filter**: Zero out utilities for all centres lacking the required facility.
2. **Modifier Application**: Apply neighbourhood conformity, distance sensitivity, and any active intervention boosts.
3. **Softmax Sampling**: Sample a (centre, mode) pair proportionally to `exp(β × utility)`, with `β = 5.0` by default.

### Feedback & Dynamics

- **Visit Feedback**: Each completed visit applies a stochastic +5% or -5% multiplier to the visiting agent's utility for that centre, reinforcing or weakening future preferences.
- **Periodic Evaluation**: Every `EVAL_FREQ` days, all retail centres are ranked against size-comparable and spatially proximate peers. Persistent underperformers (2+ consecutive failing periods) receive a utility boost (capped at +30%) across all agent utility matrices.
- **Spatial Diffusion**: Trending centres in a postcode sector see boosted utility for agents in the same area, weighted by demographic similarity.

---

## Configuration

All key parameters are set in `simulation/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `REORDER_THRESHOLD` | `20.0` | Grocery stock level that triggers a shopping trip |
| `MAX_STOCK_CAPACITY` | `100.0` | Maximum stock a household can hold |
| `DAILY_CONSUMPTION_MEAN` | `50.0` | Mean daily grocery consumption rate |
| `SOFTMAX_BETA` | `5.0` | Destination choice temperature (higher = more deterministic) |
| `DEMOGRAPHIC_DIFFUSION_WEIGHT` | `1.0` | Strength of demographic homophily in spatial diffusion |
| `DEMOGRAPHIC_BANDWIDTH` | `0.5` | Gaussian decay width for demographic similarity scoring |

---

## Outputs

| File | Description |
|---|---|
| `outputs/visits_log_<timestamp>.parquet` | Full trip log with AgentID, Trip_Type, Retail_Centre, Transport_Mode, Travel_Time_Min, Utility_Score |
| `outputs/visitation_map_<timestamp>.png` | Static choropleth map of retail centre visitation density |

---

## Data Sources

All data files are stored outside the repository and referenced via absolute paths in `simulation/config.py`:

| Data | Description |
|---|---|
| `utility_scores_<type>.parquet` | Pre-computed (household × retail centre) utility matrices for 6 trip types |
| `retail_centre_type_counts.gpkg` | Retail centre geometries and amenity binary flags |
| `final_transport_times.parquet` | Postcode-to-RetailCentre travel time lookup (Drive) |
| `Cleaned_NTS_Data.csv` | National Travel Survey trip frequency data |
