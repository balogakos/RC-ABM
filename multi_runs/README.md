# Multi-Run Ensemble Pipeline

This directory contains the scripts required to run the Retail ABM in ensemble mode (multiple iterations) to produce statistically robust results for research and analysis.

## Pipeline Structure
- `runner.py`: Executes $N$ iterations of the simulation and saves raw parquet results.
- `aggregate.py`: Processes the raw results to produce mean/std statistics for visits, transport modes, and trip types.
- `results/`: Directory for storing individual run outputs (ignored by git).

## Usage
1. Configure `runner.py` (iterations, agent count).
2. Run `python multi_runs/runner.py`.
3. Run `python multi_runs/aggregate.py` to get the final `ensemble_summary.csv`.
