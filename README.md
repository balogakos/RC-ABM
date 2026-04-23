# Retail ABM

![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)
![GitHub Repository](https://img.shields.io/badge/repository-published-success.svg)

An Agent-Based Model (ABM) simulating retail shopping behaviour. Agents make shopping decisions daily based on their stock levels, trip probabilities derived from National Travel Survey (NTS) data, and utility scores for each retail centre.

---

## How to Run

1. **One-time setup** (run when agent data changes):
   ```
   python assign_trip_frequencies.py
   ```

2. **Run the simulation**:
   ```
   python main.py
   ```
   A Tkinter window will open. Set your parameters and click **Run Simulation**.

3. **Outputs** are saved to `outputs/`:
   - `visits_log_<timestamp>.parquet` — full trip log
   - `visitation_map_<timestamp>.png` — map of retail centre visits

---

## File Structure

| File | Purpose |
|---|---|
| `main.py` | GUI entry point and simulation loop |
| `agent.py` | All agent logic (vectorized, NumPy/Pandas) |
| `config.py` | File paths and model constants |
| `assign_trip_frequencies.py` | Assigns NTS trip frequencies to agents (run once) |
| `visualization.py` | Generates the visitation density map |
| `app.py` | (Optional) Flask web interface alternative to `main.py` |

---

## Simulation Logic

### Initialisation
- Loads enriched consumer agent file (with NTS trip probability columns)
- Loads Bulk and Convenience utility score matrices
- Loads retail centre amenity binary data from GeoPackage
- Samples the requested number of agents and assigns each a random starting stock and consumption rate

### Each Day — Two Independent Systems

#### System 1: Grocery (Stock-Based)
1. Each agent's **stock** decreases by their daily consumption rate
2. Agents below the stock threshold **need to shop**
3. They choose a mode: **Online / Bulk / Convenience** (weighted by their probabilities)
4. Physical shoppers choose a retail centre weighted by **utility scores** (Bulk or Convenience matrix)
5. Stock is replenished to maximum after shopping

#### System 2: NTS Frequency Trips (Probability-Based)
Each day, for every agent, 4 independent trip types are rolled:

| Trip Type | NTS Column | Amenity Filter | Utility Matrix |
|---|---|---|---|
| **Service** | `PurposeCount_Service` | `Personal & Professional Services = 1` | Convenience |
| **Comparison** | `PurposeCount_Comparison` | `Retail = 1` | Bulk |
| **Entertainment** | `PurposeCount_Entertainment` | `Entertainment = 1` | Convenience |
| **Food & Drink** | `PurposeCount_Food/Drink` | `Cafe = 1 OR Restaurant = 1` | Convenience |

- Multiple trip types can fire on the same day for the same agent
- Centres that don't have the required amenity get a utility score of 0 (cannot be chosen)
- For Food & Drink: a centre is valid if it has **either** a Cafe or a Restaurant (or both)

---

## Data Sources

| Data | Path |
|---|---|
| Consumer agents | `Model/Utility/test_consumer_agents_bulk_prepared_with_trips.parquet` |
| Bulk utility scores | `Model/Utility/utility_scores_bulk.parquet` |
| Convenience utility scores | `Model/Utility/utility_scores_convenience.parquet` |
| Retail centre geometry + amenities | `Model/Retail Centre Data/retail_centre_type_counts.gpkg` |
| NTS trip frequencies | `Model/Distance/Data/NTS/Cleaned_NTS_Data.csv` |
