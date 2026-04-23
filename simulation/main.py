"""
Retail ABM - Main Simulation Script

Launches a GUI to configure and run the simulation. On each run:
  1. Loads/generates enriched agent data (utility_scores_bulk_with_trips.parquet).
  2. Initialises agent state (stock, consumption rate) — one agent = one household.
  3. Runs a daily loop for the specified number of days:
       a. Grocery trips  — stock-based: agents shop when stock falls below threshold.
                           Mode drawn from prob_online/bulk/convenience;
                           Transport mode drawn from prob_walk/prob_drive/prob_pt.
                           Matrices: bulk_walk, bulk_drive, bulk_pt,
                                      convenience_walk, convenience_drive, convenience_pt
       b. NTS trips      — probability-based: service, comparison, entertainment,
                           food/drink trips fire independently each day.
                           Matrices: average_walk, average_drive, average_pt
  4. Saves a visits log (parquet) and a visitation map (PNG) to outputs/.

Data Sources
------------
  utility_scores_bulk.parquet        : 1 row per household; columns = demographics
                                       + {RC_ID}_walk, {RC_ID}_drive, {RC_ID}_pt
  utility_scores_convenience.parquet : same households, same column structure
  utility_scores_average.parquet     : same households, same column structure
  retail_centre_type_counts.gpkg     : amenity binary flags per retail centre
  final_transport_times.parquet      : Postcode -> Walk/Drive/PT travel time (mins)

Output columns
--------------
    Day              Simulation day number (1-indexed)
    AgentID          Household identifier
    Trip_Type        'grocery' | 'service' | 'comparison' | 'entertainment' | 'food_drink'
    Retail_Centre    ID of the chosen retail centre
    Grocery_Mode     'bulk' | 'convenience' | 'online'
    Transport_Mode   'walk' | 'drive' | 'pt'  (None for online)
    Travel_Time_Min  Travel time in minutes from agent postcode (from transport data)
    Utility_Modifier Feedback multiplier applied
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import agent
from assign_trip_frequencies import assign_frequencies

# --- Configuration Override ---
TEST_MODE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRANSPORT_SUFFIXES = ['_walk', '_drive', '_pt']



def _clean_rc_id(x):
    """Standardises a Retail Centre ID to a plain string (strips trailing .0)."""
    try:
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
    except Exception:
        return str(x)


def _preprocess_transport_times(df):
    """
    Convert the raw transport-times DataFrame (Walk/Drive/PT columns containing
    {RC_ID: minutes} dicts) into a nested Python dict for O(1) lookup:
        {mode: {postcode: {rc_id_str: minutes}}}
    """
    if df is None or (hasattr(df, 'empty') and df.empty):
        return {}
    tt_lookup = {}
    for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
        if col not in df.columns:
            continue
        d = {}
        for postcode, val in df[col].items():
            if isinstance(val, dict):
                d[postcode] = {_clean_rc_id(k): float(v) for k, v in val.items()}
            elif val is not None:
                try:
                    d[postcode] = float(val)
                except Exception:
                    pass
        tt_lookup[mode] = d
    return tt_lookup


def _is_mode_col(name):
    """Returns True for columns like '6442.0_walk', '6442.0_drive', '6442.0_pt'."""
    for suf in TRANSPORT_SUFFIXES:
        if name.endswith(suf):
            prefix = name[:-len(suf)]
            try:
                float(prefix)
                return True
            except (ValueError, TypeError):
                pass
    return False


def _split_matrices(df, household_col='household'):
    """
    Splits a utility-scores DataFrame (with mode-suffixed RC columns) into:
      - meta_df    : demographic columns  (1 row per household)
      - mode_dfs   : dict {'walk': DataFrame, 'drive': DataFrame, 'pt': DataFrame}
                     each indexed by household ID, columns = RC IDs (no suffix, no .0)

    Column names are normalised: '1000.0_walk' → '1000' in the walk matrix.
    """
    mode_cols = {suf.lstrip('_'): [] for suf in TRANSPORT_SUFFIXES}
    meta_cols = []

    for col in df.columns:
        matched = False
        for suf in TRANSPORT_SUFFIXES:
            if col.endswith(suf):
                prefix = col[:-len(suf)]
                try:
                    float(prefix)
                    mode_cols[suf.lstrip('_')].append(col)
                    matched = True
                    break
                except (ValueError, TypeError):
                    pass
        
        if not matched:
            # Check if it's a raw RC ID (for the new single-mode format)
            try:
                float(col)
                # Default to 'drive' bucket for non-suffixed RC columns
                mode_cols['drive'].append(col)
                matched = True
            except (ValueError, TypeError):
                pass
                
        if not matched:
            meta_cols.append(col)

    meta_df = df[meta_cols].copy()

    mode_dfs = {}
    for mode, cols in mode_cols.items():
        if not cols:
            continue
        mat = df[cols].astype(np.float32)
        def _strip(c):
            # If it has a suffix, remove it
            for s in TRANSPORT_SUFFIXES:
                if c.endswith(s):
                    c = c[:-len(s)]
                    break
            return c[:-2] if c.endswith('.0') else c
        mat.columns = [_strip(c) for c in mat.columns]
        if household_col in df.columns:
            mat.index = df[household_col]
        mode_dfs[mode] = mat.fillna(0)

    return meta_df, mode_dfs


def _load_transport_times():
    """
    Loads the Postcode → travel-time table.
    Returns a DataFrame indexed by Postcode with columns Walk, Drive, PT.
    Returns an empty DataFrame if the file is missing.
    """
    path = getattr(config, 'TRANSPORT_TIMES_PATH', '')
    if not path or not os.path.exists(path):
        print(f"Transport times file not found ({path}), Distance will be 0.")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        if 'Postcode' in df.columns:
            df = df.set_index('Postcode')
        return df
    except Exception as e:
        print(f"Warning — could not load transport times: {e}")
        return pd.DataFrame()


def _lookup_travel_times(postcode_series, transport_mode_series, transport_times_df=None,
                         dest_series=None, tt_lookup=None):
    """
    Looks up per-agent travel time (minutes) given their postcode and chosen transport mode.
    Optimized to use the pre-built tt_lookup dictionary (O(1) lookup).
    """
    # ── Option 1: Fast Dictionary Lookup (Preferred) ──
    if tt_lookup:
        pcs = postcode_series.values
        tms = transport_mode_series.values
        rcs = dest_series.values if dest_series is not None else [None] * len(pcs)
        out = np.zeros(len(pcs), dtype=np.float32)

        for i in range(len(pcs)):
            mode_data = tt_lookup.get(tms[i], {})
            pc_data   = mode_data.get(pcs[i], None)
            
            if pc_data is None:
                # Postcode not in lookup at all
                out[i] = np.nan
            elif isinstance(pc_data, dict):
                key = str(rcs[i])
                if key in pc_data:
                    out[i] = pc_data[key]
                else:
                    # This specific RC has no travel time from this postcode
                    out[i] = np.nan
            else:
                out[i] = float(pc_data)
        return pd.Series(out, index=postcode_series.index)

    # ── Option 2: Legacy DataFrame Lookup (Fallback) ──
    if transport_times_df is None or transport_times_df.empty:
        return pd.Series(0.0, index=postcode_series.index)

    mode_to_col = {'walk': 'Walk', 'drive': 'Drive', 'pt': 'PT'}
    result = pd.Series(0.0, index=postcode_series.index)

    for tmode, col in mode_to_col.items():
        if col not in transport_times_df.columns:
            continue
        seg = postcode_series.index[transport_mode_series == tmode]
        if len(seg) == 0:
            continue
        pcs_sub = postcode_series.loc[seg]
        raw_vals = transport_times_df[col].reindex(pcs_sub.values)

        times_out = []
        for i, (idx, _pc) in enumerate(pcs_sub.items()):
            val  = raw_vals.iloc[i]
            dest = dest_series.loc[idx] if dest_series is not None else None
            if isinstance(val, dict):
                d_key = str(dest)
                if d_key in val:
                    times_out.append(float(val[d_key]))
                elif val:
                    times_out.append(float(np.mean(list(val.values()))))
                else:
                    times_out.append(0.0)
            else:
                try:
                    v = float(val)
                    times_out.append(0.0 if np.isnan(v) else v)
                except:
                    times_out.append(0.0)

        result.loc[seg] = times_out
    return result


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class RetailABMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail ABM Configuration")
        self.root.geometry("450x370")

        # Data is NOT loaded here — the GUI appears immediately.
        # Loading happens on the first "Run Simulation" click, filtered to
        # the user-specified agent count.
        self._base_data   = None   # raw (immutable) base data after first load
        self._loaded_n    = None   # the N that was used to build _base_data

        self.status_var = tk.StringVar(
            value="Configure settings and click Run Simulation.")
        ttk.Label(root, textvariable=self.status_var, wraplength=420).pack(pady=10)

        ttk.Label(root, text="Number of Agents to Simulate:").pack(pady=5)
        self.num_agents_var = tk.StringVar(value="1000")
        ttk.Entry(root, textvariable=self.num_agents_var).pack(pady=5)

        ttk.Label(root, text="Number of Days to Simulate:").pack(pady=5)
        self.days_var = tk.StringVar(value="30")
        ttk.Entry(root, textvariable=self.days_var).pack(pady=5)

        ttk.Label(root, text="Evaluation Frequency (Days):").pack(pady=5)
        self.eval_freq_var = tk.StringVar(value="10")
        ttk.Entry(root, textvariable=self.eval_freq_var).pack(pady=5)

        ttk.Button(root, text="Run Simulation",
                   command=self.run_simulation).pack(pady=20)

        self.status_var.set("Ready — enter parameters and click Run Simulation.")

    # ------------------------------------------------------------------

    def log(self, message):
        print(message)
        self.status_var.set(message)
        self.root.update()

    # ------------------------------------------------------------------

    def load_data(self, n_agents=None):
        """
        Loads all required data for the simulation.

        Parameters
        ----------
        n_agents : int or None
            If given, read only this many rows from the utility parquets,
            giving (n_agents, 259) matrices instead of (656K, 259) — much
            faster to copy, reindex, and mutate during the simulation.
            If None, loads all rows.
        """
        import pyarrow.parquet as pq
        
        utility_matrices = {}
        consumers = None
        
        trip_types = [
            'bulk', 'convenience', 'comparison', 
            'entertainment', 'food_drink', 'service'
        ]
        
        base_utility_dir = os.path.join(config.MODEL_DIR, "Utility")
        if TEST_MODE:
            base_utility_dir = os.path.join(base_utility_dir, "testing")
            
        self.log(f"Loading 6 trip-specific utility datasets from {base_utility_dir} ({n_agents or 'all'} agents)...")

        for trip_type in trip_types:
            file_path = os.path.join(base_utility_dir, f'utility_scores_{trip_type}.parquet')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
                
            if n_agents:
                _pf = pq.ParquetFile(file_path)
                df = next(_pf.iter_batches(batch_size=n_agents)).to_pandas()
            else:
                df = pd.read_parquet(file_path)
                
            meta_df, mode_dfs = _split_matrices(df)
            
            # The bulk dataset acts as our primary demographic source for initializing the population
            if trip_type == 'bulk':
                consumers = meta_df
                
            for tmode, mat in mode_dfs.items():
                utility_matrices[f'{trip_type}_{tmode}'] = mat
                
            # Aggressive Garbarge Collection to free the 4GB raw DF before the next iteration
            import gc
            del df
            del mode_dfs
            if trip_type != 'bulk':
                del meta_df
            gc.collect()

        # -- Amenity binary flags --
        self.log("Loading retail centre amenity data...")
        import geopandas as gpd
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        gdf['RC_ID'] = gdf['RC_ID'].apply(_clean_rc_id)
        gdf = gdf.set_index('RC_ID')

        amenity_cols = [
            'Foodstore', 'Personal Service', 'Professional Services',
            'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe',
        ]
        amenity_binary = {
            col: (gdf[col] > 0).astype(float)
            for col in amenity_cols
            if col in gdf.columns
        }

        # -- Transport times --
        self.log("Loading transport times lookup table...")
        transport_times = _load_transport_times()

        n_matrices = len(utility_matrices)
        shapes = {k: v.shape for k, v in utility_matrices.items()}
        self.log(
            f"Loaded {len(consumers):,} agents | {n_matrices} utility matrices | "
            f"Transport times: {'yes' if not transport_times.empty else 'not found'}"
        )
        print("Matrix shapes:", shapes)

        # Pre-process transport times into fast nested dict
        tt_lookup = _preprocess_transport_times(transport_times)
        return consumers, utility_matrices, amenity_binary, tt_lookup

    # ------------------------------------------------------------------

    def run_simulation(self):
        try:
            num_agents_req = int(self.num_agents_var.get())
            days           = int(self.days_var.get())
            eval_freq      = int(self.eval_freq_var.get())

            # ── Lazy load: only triggered on first Run, or when N changes ──
            if self._base_data is None or self._loaded_n != num_agents_req:
                self.log(f"Loading data for {num_agents_req:,} agents "
                         f"({'first load' if self._base_data is None else 'N changed'})…")
                raw = self.load_data(n_agents=num_agents_req)
                if raw is None:
                    raise RuntimeError("Failed to load base data. See terminal.")
                # Store base (immutable) — copy matrices per run to protect
                self._base_data = raw
                self._loaded_n  = num_agents_req

            consumers, base_matrices, amenity_binary, tt_lookup = self._base_data
            # Deep-copy matrices so in-place feedback doesn't corrupt cache
            utility_matrices = {k: v.copy() for k, v in base_matrices.items()}

            # Threshold Rule: only centres with > 0.8 utility can be selected.
            # We zero them out here, and agent.py sampling will treat 0 as impossible.
            THRESHOLD = 0.1
            for k, mat in utility_matrices.items():
                mat[mat < THRESHOLD] = 0.0

            # Load retail centres GeoDataFrame for evaluation
            import geopandas as gpd
            retail_gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
            retail_gdf['RC_ID'] = retail_gdf['RC_ID'].apply(_clean_rc_id)
            retail_gdf = retail_gdf.set_index('RC_ID')

            n = len(consumers)  # already filtered to num_agents_req
            self.log(f"Initialising {n} agents over {days} days…")

            state_df, consumers_sampled = agent.initialize_agent_state(n, consumers)

            # Persistent trackers for evaluation logic
            underperformer_tracker = {}
            cumulative_boosts = {}

            # ----------------------------------------------------------
            # Daily simulation loop
            # ----------------------------------------------------------
            all_visits = []

            for day in range(1, days + 1):
                self.log(f"Day {day}/{days}...")

                # ── A. Trip Generation & Chaining ──────────────────────────
                state_df = agent.consume(state_df)
                needs_grocery = agent.check_shopping_need(state_df)
                grocery_mode_series = agent.choose_mode(consumers_sampled, needs_grocery)
                nts_triggered = agent.trigger_trips(consumers_sampled)
                
                # Log Online Groceries (independent)
                online_mask = needs_grocery & (grocery_mode_series == 'online')
                if online_mask.any():
                    idx = state_df[online_mask].index
                    all_visits.append(pd.DataFrame({
                        'Day': day, 'AgentID': state_df.loc[idx, 'AgentID'].values,
                        'Postcode': state_df.loc[idx, 'Postcode'].values,
                        'Trip_Type': 'grocery', 'Retail_Centre': 'ONLINE',
                        'Grocery_Mode': 'online', 'Transport_Mode': None,
                        'Travel_Time_Min': 0.0, 'Utility_Modifier': 1.0, 'Utility_Score': 0.0
                    }))

                # Which physical trips still need destination selection?
                trips_to_place = {t: mask.copy() for t, mask in nts_triggered.items()}
                trips_to_place['grocery'] = needs_grocery & grocery_mode_series.isin(['bulk', 'convenience'])

                # Rolling the dice for chaining (50/50 for agents with > 1 trip)
                trip_counts = pd.Series(0, index=state_df.index)
                for t, mask in trips_to_place.items():
                    trip_counts += mask.astype(int)
                
                chain_candidates = state_df.index[trip_counts > 1]
                will_chain = pd.Series(False, index=state_df.index)
                if not chain_candidates.empty:
                    will_chain.loc[chain_candidates] = np.random.rand(len(chain_candidates)) < 0.5
                
                # Execute Chained Trips
                if will_chain.any():
                    cb = pd.Series([[] for _ in range(len(state_df))], index=state_df.index)
                    for t, mask in trips_to_place.items():
                        for i in state_df.index[mask & will_chain]:
                            cb[i].append(t)
                    
                    cb_tuples = cb[will_chain].apply(tuple)
                    for combo_tuple, idx_series in cb_tuples.groupby(cb_tuples):
                        combo_list = list(combo_tuple)
                        shoppers_idx = idx_series.index
                        dests, modes, scores = agent.choose_chained_destination(
                            combo_list, shoppers_idx, grocery_mode_series,
                            utility_matrices, amenity_binary, consumers_sampled)
                        
                        valid = dests.notna()
                        if valid.any():
                            v_idx = d_idx = dests[valid].index
                            print(f"  [Chaining] {len(v_idx)} agents chained {combo_list} at {len(dests[valid].unique())} centres")
                            pc_series = state_df.loc[v_idx, 'Postcode']
                            tm_series = modes.loc[v_idx].fillna('drive')
                            tt_series = _lookup_travel_times(pc_series, tm_series, None, dest_series=dests[valid], tt_lookup=tt_lookup)
                            
                            for t_type in combo_list:
                                util_p = agent.TRIP_TYPE_CONFIG[t_type]['util_prefix'] if t_type != 'grocery' else grocery_mode_series.loc[v_idx].iloc[0]
                                f_mults = pd.Series(1.0, index=v_idx)
                                for tmode in agent.TRANSPORT_MODES:
                                    tseg = (modes.loc[v_idx] == tmode)
                                    if tseg.any():
                                        s_idx = v_idx[tseg]
                                        key = f"{util_p}_{tmode}"
                                        if key in utility_matrices:
                                            f_mults.loc[s_idx] = agent.apply_feedback(utility_matrices[key], state_df.loc[s_idx, 'AgentID'].values, dests[s_idx].values)
                                
                                all_visits.append(pd.DataFrame({
                                    'Day': day, 'AgentID': state_df.loc[v_idx, 'AgentID'].values,
                                    'Postcode': pc_series.values, 'Trip_Type': t_type,
                                    'Retail_Centre': dests[valid].values, 'Grocery_Mode': grocery_mode_series.loc[v_idx].values,
                                    'Transport_Mode': modes.loc[v_idx].values, 'Travel_Time_Min': tt_series.values,
                                    'Utility_Modifier': f_mults.values, 'Utility_Score': scores.loc[v_idx].values
                                }))
                                trips_to_place[t_type].loc[v_idx] = False

                # ── B. Independent Trips ─────────────────────────────────────
                for t_type, mask in trips_to_place.items():
                    if not mask.any(): continue
                    
                    if t_type == 'grocery':
                        dests, modes, scores = agent.choose_destination(state_df, mask, grocery_mode_series, utility_matrices, amenity_binary)
                    else:
                        dests, modes, scores = agent.choose_destination_for_trip(t_type, mask, consumers_sampled, utility_matrices, amenity_binary)
                    
                    valid = dests.notna()
                    if valid.any():
                        v_idx = dests[valid].index
                        pc_series = (state_df.loc[v_idx, 'Postcode'] if t_type == 'grocery' else consumers_sampled.loc[v_idx, 'Postcode'])
                        tm_series = modes.loc[v_idx].fillna('drive')
                        tt_series = _lookup_travel_times(pc_series, tm_series, None, dest_series=dests[valid], tt_lookup=tt_lookup)
                        
                        util_p = agent.TRIP_TYPE_CONFIG[t_type]['util_prefix'] if t_type != 'grocery' else grocery_mode_series.loc[v_idx].iloc[0]
                        f_mults = pd.Series(1.0, index=v_idx)
                        for tmode in agent.TRANSPORT_MODES:
                            tseg = (modes.loc[v_idx] == tmode)
                            if tseg.any():
                                s_idx = v_idx[tseg]
                                key = f"{util_p}_{tmode}"
                                if key in utility_matrices:
                                    f_mults.loc[s_idx] = agent.apply_feedback(utility_matrices[key], state_df.loc[s_idx, 'AgentID'].values, dests[s_idx].values)
                                    
                        all_visits.append(pd.DataFrame({
                            'Day': day, 'AgentID': state_df.loc[v_idx, 'AgentID'].values,
                            'Postcode': pc_series.values, 'Trip_Type': t_type,
                            'Retail_Centre': dests[valid].values, 'Grocery_Mode': grocery_mode_series.loc[v_idx].values,
                            'Transport_Mode': modes.loc[v_idx].values, 'Travel_Time_Min': tt_series.values,
                            'Utility_Modifier': f_mults.values, 'Utility_Score': scores.loc[v_idx].values
                        }))

                state_df = agent.update_stock_after_shop(state_df, needs_grocery)

                # ── C. Retail Centre Agent Evaluation ───────────────────────
                if day % eval_freq == 0:
                    self.log(f"Evaluating retail centres (Day {day})...")
                    start_day = day - eval_freq + 1
                    period_visits = [df for df in all_visits
                                     if df['Day'].iloc[0] >= start_day
                                     and df['Day'].iloc[0] <= day]
                    if period_visits:
                        eval_df  = pd.concat(period_visits, ignore_index=True)
                        messages = agent.evaluate_retail_centres(
                            eval_df, retail_gdf, utility_matrices, amenity_binary,
                            tracker=underperformer_tracker,
                            cumulative_boosts=cumulative_boosts)
                        if messages:
                            print(f"\n--- Retail Centre Evaluation Day {day} ---")
                            for msg in messages:
                                print(msg)
                                self.log(msg)
                        else:
                            print(f"\n--- Retail Centre Evaluation Day {day}: "
                                  "No boosts applied ---")

                        diffusion_msgs = agent.apply_spatial_diffusion_bonus(
                            eval_df, consumers_sampled, utility_matrices)
                        if diffusion_msgs:
                            print(f"\n--- Spatial Diffusion Day {day} ---")
                            for msg in diffusion_msgs:
                                print(msg)
                                self.log(msg)

            # ----------------------------------------------------------
            # Save results
            # ----------------------------------------------------------
            self.log("Saving results...")
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

            if all_visits:
                total_visits_df = pd.concat(all_visits, ignore_index=True)
                output_path = os.path.join(
                    config.OUTPUT_DIR, f"visits_log_{int(time.time())}.parquet")
                total_visits_df.to_parquet(output_path)

                # Summary statistics
                print("\n=== Trip Type Breakdown ===")
                print(total_visits_df['Trip_Type'].value_counts().to_string())

                print("\n=== Transport Mode Breakdown ===")
                print(total_visits_df['Transport_Mode'].value_counts().to_string())

                print("\n=== Average Travel Time by Transport Mode (mins) ===")
                print(total_visits_df.groupby('Transport_Mode')['Travel_Time_Min']
                      .mean().round(1).to_string())

                print("\n=== Avg trips per agent per day ===")
                print(total_visits_df.groupby(['Day', 'AgentID']).size()
                      .describe().to_string())

                print("\n=== Most / Least trips per agent (total across all days) ===")
                for trip in ['grocery', 'comparison', 'service',
                             'entertainment', 'food_drink']:
                    sub = total_visits_df[total_visits_df['Trip_Type'] == trip]
                    if sub.empty:
                        print(f"  {trip:<15} : no trips recorded")
                        continue
                    counts   = sub.groupby('AgentID').size()
                    most_id  = counts.idxmax()
                    least_id = counts.idxmin()
                    print(f"  {trip:<15} : "
                          f"Most  -> Agent {most_id} ({counts[most_id]} trips) | "
                          f"Least -> Agent {least_id} ({counts[least_id]} trips) | "
                          f"Mean {counts.mean():.1f}")

                print("\n=== Average daily visits per agent by trip type ===")
                n_agents = total_visits_df['AgentID'].nunique()
                for trip in ['grocery', 'comparison', 'service',
                             'entertainment', 'food_drink']:
                    sub = total_visits_df[total_visits_df['Trip_Type'] == trip]
                    avg = len(sub) / (n_agents * days) if n_agents and days else 0
                    print(f"  {trip:<15} : {avg:.3f} trips/agent/day  "
                          f"({avg * 7:.2f} per agent per week)")

                self.log(f"Done. {len(total_visits_df):,} visits saved.")

                import visualization
                map_path = visualization.plot_visitation_map(output_path)

                msg = (f"Simulation complete!\n"
                       f"{len(total_visits_df):,} visits saved to:\n{output_path}")
                if map_path:
                    msg += f"\nMap: {map_path}"
                messagebox.showinfo("Success", msg)

            else:
                self.log("Simulation complete — no visits occurred.")
                messagebox.showinfo(
                    "Info",
                    "Simulation complete but no visits occurred.\n"
                    "Check that utility scores and amenity data exist.")

        except Exception as e:
            import traceback
            self.log(f"Error: {e}")
            print(traceback.format_exc())
            messagebox.showerror("Simulation Error",
                                 f"{e}\n\nSee terminal for full traceback.")
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    RetailABMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
