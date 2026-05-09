# Retail Centre Agent-Based Model (RC-ABM)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-green)

> A spatially-explicit, agent-based model of retail consumer behaviour, calibrated to real-world household demographics and National Travel Survey (NTS) trip frequency data.

---

## Overview

The RC-ABM simulates how households in a study region make decisions about **where to shop**, **what for**, and **how they get there**. Each agent is endowed with demographic attributes drawn from real data, parameterised with the results of an online study and makes daily shopping decisions driven by two independent systems:

Agents can trigger a range of trips, including  Grocery, Service, Comparison, Entertainment, and Food & Drink which are sampled daily against per-agent probabilities derived from the National Travel Survey.

Destination choice is governed by computed **utility scores** for each (household × retail centre) pair, incorporating exponential distance decay and centre-specific amenity presence

This work is part of my PhD project, an agent-based model of retail centres, at the Geographic Data Science Lab at the University of Liverpool and is partnered with the Liverpool City Region Combined Authority.

The project is supervised by Dr. Ron Mahabir, Dr. Les Dolega, and Dr Gabriele Filomena. 

---

## Key Features

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
├── simulation/                  # Core model engine (Modular Architecture)
│   ├── main.py                  # GUI entry point (Tkinter)
│   ├── agents/                  # Agent populations and behaviors
│   │   ├── consumer/            # Consumer population & individual behaviors
│   │   └── retail_centre/       # Retail centre tracking & interventions
│   ├── core/                    # Simulation engine, math, and path resolution
│   ├── config.py                # Model constants
│   └── visualization.py         # Spatial analytics and mapping
│
├── data_preprocessing/          # Data pipeline and preparation
│   └── assign_trip_frequencies.py   # Assigns NTS probabilities to agents
│
├── website/                     # Streamlit web interface
│   └── app.py                   # Dashboards and interactive visualization
│
├── outputs/                     # Simulation results (Parquet & PNG)
├── tests/                       # Unit and integration tests
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
python data_preprocessing/assign_trip_frequencies.py
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

### Destination Choice

For each trip, destination selection follows this sequence:

1. **Amenity Filter**: Zero out utilities for all centres lacking the required facility.
2. **Modifier Application**: Apply neighbourhood conformity, distance sensitivity, and any active intervention boosts.
3. **Softmax Sampling**: Sample a (centre, mode) pair proportionally to `exp(β × utility)`, with `β = 5.0` by default.

### Feedback & Dynamics

- **Visit Feedback**: Each completed visit applies a stochastic +5% or -5% multiplier to the visiting agent's utility for that centre, reinforcing or weakening future preferences.
- **Periodic Evaluation**: Every `EVAL_FREQ` days, all retail centres are ranked against size-comparable and spatially proximate peers. Persistent underperformers (2+ consecutive failing periods) receive a utility boost (capped at +30%) across all agent utility matrices.
- **Spatial Diffusion**: Trending centres in a postcode sector see boosted utility for agents in the same area, weighted by demographic similarity.

---

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
| `utility_scores_<type>.parquet` | Computed (household × retail centre) utility matrices for 6 trip types |
| `retail_centre_type_counts.gpkg` | Retail centre geometries and amenity binary flags |
| `final_transport_times.parquet` | Postcode-to-RetailCentre travel time lookup (Drive) |
| `Cleaned_NTS_Data.csv` | National Travel Survey trip frequency data |

---

## Technical Details

### Architecture & Performance
The RC-ABM is built on a **fully vectorised architecture** using Python's scientific stack (NumPy and Pandas). Unlike traditional agent-based frameworks that iterate over individual agent objects, this model performs operations on monolithic arrays representing the entire population. This approach allows the simulation to scale to 650,000+ agents while maintaining execution times of just a few seconds per day.

### Why Mesa was not used
While [Mesa](https://mesa.readthedocs.io/) is a popular framework for Python ABMs, it was deliberately avoided for this project for the following reasons:

1. **Overhead of Object-Oriented Agents**: Mesa's "agent-as-an-object" pattern incurs significant memory and CPU overhead when dealing with hundreds of thousands of agents. Each agent in Mesa is a Python object with its own state and methods, which slows down iteration.
2. **Vectorisation Potential**: The decision logic in retail destination choice is highly mathematical (Softmax sampling over utility matrices). Scientific libraries like NumPy and Pandas are designed to handle these operations across entire datasets simultaneously. By "flattening" the agents into rows of a DataFrame, we leverage C-optimised routines for choice logic.

### Performance Optimisations
Recent updates have introduced significant performance enhancements for large-scale runs:
- **Vectorised Travel Time Lookups**: Replaced Python-based nested dictionary lookups with NumPy-based indexing over pre-processed DataFrames, reducing trip recording time by over 90%.
- **Optimised Neighbourhood Conformity**: Implemented a `groupby().max().loc[]` pattern for social influence calculations, which is significantly faster than standard `transform` operations on wide utility matrices.
- **Vectorised Initialization**: Batch assignment of geodemographic subclusters during agent population creation.

### Troubleshooting: Memory Usage
When running simulations with 650,000+ agents, each individual run requires approximately 16GB of RAM. 
- **Sequential Execution**: The `multi_runs/runner.py` script is set to sequential mode by default to avoid memory exhaustion on machines with less than 64GB RAM.
- **Parallel Execution**: If your machine has 64GB+ RAM, you can manually enable parallelization in `runner.py` by using `multiprocessing.Pool`, but ensure `MAX_WORKERS` is carefully set.
3. **Control over Choice Logic**: Retail modelling often requires complex joint-choice algorithms that are more easily implemented and debugged using standard matrix operations than by attempting to shoehorn them into the Mesa `Step` scheduler.
