import pandas as pd
import numpy as np
import time
import config
from simulation.agents.consumer.consumer_population import ConsumerPopulation
from simulation.agents.retail_centre.retail_manager import RetailManager
from simulation.core.utility_engine import apply_feedback
from simulation.core.constants import TRANSPORT_MODES, TRIP_TYPE_CONFIG

class SimulationEngine:
    def __init__(self, consumers_df, utility_matrices, amenity_binary, tt_lookup):
        self.consumers_df = consumers_df
        self.utility_matrices = utility_matrices
        self.amenity_binary = amenity_binary
        
        # 1. Cache base utilities ONLY if decay is enabled (saves massive RAM for large runs)
        decay = getattr(config, 'SOCIAL_DECAY_FACTOR', 1.0)
        if decay < 1.0:
            self.base_utilities = {k: v.copy() for k, v in utility_matrices.items()}
        else:
            self.base_utilities = None

        # 2. Apply "Warm Start" using fast NumPy addition (instead of slow .update)
        start_sigma = getattr(config, 'SOCIAL_START_SIGMA', 0.0)
        if start_sigma > 0:
            for matrix in self.utility_matrices.values():
                # In-place addition to prevent memory spikes
                noise = np.random.normal(0, start_sigma, size=matrix.shape).astype(np.float16)
                np.add(matrix.values, noise, out=matrix.values)
        
        # Pre-process travel time lookup for vectorized access
        self.tt_lookup_dfs = self._prepare_tt_lookup(tt_lookup)
        
        self.population = None
        self.retail_manager = None

        # Convergence tracking state
        self.centre_ids = sorted(list(self.amenity_binary.index))
        self.utility_convergence_records = []
        self.prev_u = None
        self.rank_history_cumulative = []
        self.rank_history_daily = []
        self.prev_r_cumulative = None
        self.prev_r_daily = None


    def run(self, num_agents, days, eval_freq, log_callback=None):
        """
        Runs the simulation loop for a specified number of days.
        Includes performance monitoring (per-day timers) and periodic evaluation phases.
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        log(f"Initialising {num_agents} agents over {days} days...")
        self.population = ConsumerPopulation(num_agents, self.consumers_df)
        self.retail_manager = RetailManager(self.amenity_binary)

        # --- Real-Time Aggregation (Memory Efficient) ---
        # Instead of raw logs, we maintain running totals for Daily and Centre performance
        self.daily_summary = pd.DataFrame()
        self.centre_performance = pd.DataFrame()
        
        # Build subcluster map once upfront
        subcluster_map = None
        if 'Geo_Subcluster' in self.population.attributes.columns and 'household' in self.population.attributes.columns:
            _map_df = self.population.attributes[['household', 'Geo_Subcluster']].copy()
            _map_df['household'] = _map_df['household'].astype(str)
            subcluster_map = _map_df.drop_duplicates('household').set_index('household')['Geo_Subcluster']

        for day in range(1, days + 1):
            day_start = time.time()
            log(f"Day {day}/{days}...")
            current_period = []
            
            # Calculate "Daily Pulse" (Temporal Variability for NTS)
            variance = getattr(config, 'PROBABILITY_VARIANCE', 0.0)
            daily_pulse = np.random.normal(1.0, variance) if variance > 0 else 1.0
            daily_pulse = np.clip(daily_pulse, 0.5, 1.5) # Prevent extreme values

            # 1. Consumption & Need Detection (Stable Grocery)
            self.population.consume()
            needs_grocery = self.population.check_grocery_need()
            
            grocery_mode_series = self.population.choose_shopping_mode(needs_grocery)
            nts_triggered = self.population.trigger_nts_trips(multiplier=daily_pulse)

            # 2. Record Online Grocery
            online_mask = needs_grocery & (grocery_mode_series == 'online')
            if online_mask.any():
                idx = self.population.state_df[online_mask].index
                current_period.append(pd.DataFrame({
                    'Day': day, 'AgentID': self.population.state_df.loc[idx, 'AgentID'].values,
                    'Postcode': self.population.state_df.loc[idx, 'Postcode'].values,
                    'Trip_Type': 'grocery', 'Retail_Centre': 'ONLINE',
                    'Grocery_Mode': 'online', 'Transport_Mode': None,
                    'Utility_Modifier': 1.0, 'Utility_Score': 0.0
                }))

            # 3. Trip Chaining Logic
            trips_to_place = {t: mask.copy() for t, mask in nts_triggered.items()}
            trips_to_place['grocery'] = needs_grocery & grocery_mode_series.isin(['bulk', 'convenience'])

            trip_counts = pd.Series(0, index=self.population.state_df.index)
            for t, mask in trips_to_place.items():
                trip_counts += mask.astype(int)
            
            chain_candidates = self.population.state_df.index[trip_counts > 1]
            will_chain = pd.Series(False, index=self.population.state_df.index)
            if not chain_candidates.empty:
                will_chain.loc[chain_candidates] = np.random.rand(len(chain_candidates)) < 0.5
            
            if will_chain.any():
                # Vectorized Chaining: Instead of a list-of-lists, we use a boolean matrix or 
                # a more efficient grouping.
                chaining_idx = self.population.state_df.index[will_chain]
                
                # Identify which combinations of trips these agents are taking
                # We can use a bitmask or just string-join the trip types
                combo_strings = pd.Series("", index=chaining_idx)
                for t, mask in trips_to_place.items():
                    # Update combo strings for agents who are chaining AND have this trip triggered
                    affected_in_chain = mask[will_chain]
                    if affected_in_chain.any():
                        combo_strings.loc[affected_in_chain[affected_in_chain].index] += t + "|"
                
                # Group agents by their unique combination of trips
                for combo_str, idx_series in combo_strings.groupby(combo_strings):
                    if not combo_str: continue
                    combo_list = combo_str.strip("|").split("|")
                    shoppers_idx = idx_series.index
                    
                    dests, modes, scores = self.population.choose_chained_destinations(
                        combo_list, shoppers_idx, grocery_mode_series,
                        self.utility_matrices, self.amenity_binary)
                    
                    valid = dests.notna()
                    if valid.any():
                        v_idx = dests[valid].index
                        pc_series = self.population.state_df.loc[v_idx, 'Postcode']
                        tm_series = modes.loc[v_idx].fillna('drive')
                        
                        for t_type in combo_list:
                            util_p = TRIP_TYPE_CONFIG[t_type]['util_prefix'] if t_type != 'grocery' else grocery_mode_series.loc[v_idx].iloc[0]
                            f_mults = pd.Series(1.0, index=v_idx)
                            for tmode in TRANSPORT_MODES:
                                tseg = (modes.loc[v_idx] == tmode)
                                if tseg.any():
                                    s_idx = v_idx[tseg]
                                    key = f"{util_p}_{tmode}"
                                    if key in self.utility_matrices:
                                        f_mults.loc[s_idx] = apply_feedback(self.utility_matrices[key], self.population.state_df.loc[s_idx, 'AgentID'].values, dests[s_idx].values)
                            
                            current_period.append(pd.DataFrame({
                                'Day': day, 'AgentID': self.population.state_df.loc[v_idx, 'AgentID'].values,
                                'Postcode': pc_series.values, 'Trip_Type': t_type,
                                'Retail_Centre': dests[valid].values, 'Grocery_Mode': grocery_mode_series.loc[v_idx].values,
                                'Transport_Mode': modes.loc[v_idx].values,
                                'Utility_Modifier': f_mults.values, 'Utility_Score': scores.loc[v_idx].values
                            }))
                            trips_to_place[t_type].loc[v_idx] = False

            # 4. Independent Trips
            for t_type, mask in trips_to_place.items():
                if not mask.any(): continue
                
                if t_type == 'grocery':
                    dests, modes, scores = self.population.choose_destinations(mask, grocery_mode_series, self.utility_matrices, self.amenity_binary)
                else:
                    dests, modes, scores = self.population.choose_nts_destinations(t_type, mask, self.utility_matrices, self.amenity_binary)
                
                valid = dests.notna()
                if valid.any():
                    v_idx = dests[valid].index
                    pc_series = self.population.state_df.loc[v_idx, 'Postcode']
                    tm_series = modes.loc[v_idx].fillna('drive')
                    
                    util_p = TRIP_TYPE_CONFIG[t_type]['util_prefix'] if t_type != 'grocery' else grocery_mode_series.loc[v_idx].iloc[0]
                    f_mults = pd.Series(1.0, index=v_idx)
                    for tmode in TRANSPORT_MODES:
                        tseg = (modes.loc[v_idx] == tmode)
                        if tseg.any():
                            s_idx = v_idx[tseg]
                            key = f"{util_p}_{tmode}"
                            if key in self.utility_matrices:
                                f_mults.loc[s_idx] = apply_feedback(self.utility_matrices[key], self.population.state_df.loc[s_idx, 'AgentID'].values, dests[s_idx].values)
                                
                    current_period.append(pd.DataFrame({
                        'Day': day, 'AgentID': self.population.state_df.loc[v_idx, 'AgentID'].values,
                        'Postcode': pc_series.values, 'Trip_Type': t_type,
                        'Retail_Centre': dests[valid].values, 'Grocery_Mode': grocery_mode_series.loc[v_idx].values,
                        'Transport_Mode': modes.loc[v_idx].values,
                        'Utility_Modifier': f_mults.values, 'Utility_Score': scores.loc[v_idx].values
                    }))

            self.population.replenish_stock(needs_grocery, grocery_mode_series)

            # 5. Daily Aggregation & Evaluation
            perf_update = None
            if current_period:
                day_df = pd.concat(current_period, ignore_index=True)
                
                # A. Update Daily Summary (City-wide Pulse)
                # Count trips by type and mode for the daily overview
                trips_agg = day_df.groupby('Trip_Type').size()
                modes_agg = day_df.groupby('Transport_Mode').size()
                online_cnt = (day_df['Retail_Centre'] == 'ONLINE').sum()
                
                day_row = pd.Series(0, index=['Day', 'Total_Visits', 'Online'])
                day_row['Day'] = day
                day_row['Total_Visits'] = len(day_df)
                day_row['Online'] = online_cnt
                day_row = pd.concat([day_row, trips_agg, modes_agg])
                
                self.daily_summary = pd.concat([self.daily_summary, day_row.to_frame().T], ignore_index=True)

                # B. Update Retail Centre Performance (Granular Breakdown)
                # Create columns like 'grocery_drive', 'comparison_walk', etc.
                physical_trips = day_df[day_df['Retail_Centre'] != 'ONLINE'].copy()
                if not physical_trips.empty:
                    physical_trips['Metric'] = physical_trips['Trip_Type'] + "_" + physical_trips['Transport_Mode'].fillna('unknown')
                    perf_update = physical_trips.pivot_table(
                         index='Retail_Centre', columns='Metric', aggfunc='size', fill_value=0
                    )
                    
                    if self.centre_performance.empty:
                        self.centre_performance = perf_update
                    else:
                        self.centre_performance = self.centre_performance.add(perf_update, fill_value=0)

                # C. Periodic Retail Evaluation (for interventions)
                if day % eval_freq == 0:
                    log(f"Evaluating retail centres (Day {day})...")
                    messages = self.retail_manager.evaluate_centres(day_df, self.utility_matrices)
                    for msg in messages: log(msg)

                    diffusion_msgs = self.population.apply_social_influence(day_df, self.utility_matrices, self.base_utilities)
                    for msg in diffusion_msgs: log(msg)

                del day_df
                current_period.clear()
            
            # Calculate utility distribution convergence metrics and rank stability
            self._track_utility_convergence(day, perf_update=perf_update)
            
            day_elapsed = time.time() - day_start
            log(f"  -> Day {day} complete in {day_elapsed:.2f}s")

        # 6. Final Save (Summary Files)
        return self._save_summary_results()

    def _save_summary_results(self):
        """Saves the final aggregated summaries to CSV."""
        from simulation.core import paths
        import time
        from scipy.stats import spearmanr
        
        # Calculate Spearman correlation to final anchor R_T (for both cumulative and daily ranks)
        if self.rank_history_cumulative:
            r_T_cum = self.rank_history_cumulative[-1]
            for idx, r_t in enumerate(self.rank_history_cumulative):
                res = spearmanr(r_t, r_T_cum)
                val = float(res.statistic) if not np.isnan(res.statistic) else np.nan
                self.utility_convergence_records[idx]['Spearman_Cum_Final_Anchor'] = val
                
        if self.rank_history_daily:
            r_T_daily = self.rank_history_daily[-1]
            for idx, r_t in enumerate(self.rank_history_daily):
                res = spearmanr(r_t, r_T_daily)
                val = float(res.statistic) if not np.isnan(res.statistic) else np.nan
                self.utility_convergence_records[idx]['Spearman_Daily_Final_Anchor'] = val

        timestamp = int(time.time())
        daily_dir = paths.OUTPUT_DIR / 'daily_summaries'
        centre_dir = paths.OUTPUT_DIR / 'centre_performance'
        convergence_dir = paths.OUTPUT_DIR / 'utility_convergence'
        
        daily_dir.mkdir(parents=True, exist_ok=True)
        centre_dir.mkdir(parents=True, exist_ok=True)
        convergence_dir.mkdir(parents=True, exist_ok=True)
        
        daily_path = daily_dir / f'daily_summary_{timestamp}.csv'
        centre_path = centre_dir / f'retail_centre_performance_{timestamp}.csv'
        convergence_path = convergence_dir / f'utility_convergence_{timestamp}.csv'
        
        # Ensure all columns are present and clean up indices
        self.daily_summary.to_csv(daily_path, index=False)
        self.centre_performance.to_csv(centre_path)
        
        # Save convergence metrics, rank stability metrics, and raw distributions
        convergence_df = pd.DataFrame(self.utility_convergence_records)
        convergence_df.to_csv(convergence_path, index=False)
        
        print(f"\nSUCCESS: Daily totals saved to: {daily_path}")
        print(f"SUCCESS: Centre performance saved to: {centre_path}")
        print(f"SUCCESS: Utility convergence and distributions saved to: {convergence_path}")
        
        return [daily_path, centre_path, convergence_path]

    def _calculate_centre_attractiveness(self):
        """
        Calculates the attractiveness utility (U_t) for all retail centres.
        For each retail centre, it is the average utility score across all agents
        and all utility matrices where the centre is present.
        """
        u_sums = pd.Series(0.0, index=self.centre_ids)
        u_counts = pd.Series(0.0, index=self.centre_ids)
        
        for matrix in self.utility_matrices.values():
            # Use numpy mean with float32 dtype to avoid upcasting the entire large DataFrame to float64
            col_means_arr = np.mean(matrix.values, axis=0, dtype=np.float32)
            col_means = pd.Series(col_means_arr, index=matrix.columns)
            aligned_mean = col_means.reindex(self.centre_ids, fill_value=0.0)
            u_sums += aligned_mean
            u_counts += 1.0
            
        return (u_sums / u_counts).values

    def _track_utility_convergence(self, day, perf_update=None):
        """
        Extracts daily attractiveness utilities and calculates convergence
        metrics (JSD, KS distance) and rank stability (Spearman correlation).
        """
        from scipy.spatial.distance import jensenshannon
        import scipy.stats as stats
        
        # 1. Calculate current attractiveness vector U_t
        u_t = self._calculate_centre_attractiveness()
        u_t_clipped = np.clip(u_t, 0.0, None)
        
        # 2. Calculate utility distribution divergence if t > 1
        jsd_val = np.nan
        ks_val = np.nan
        
        if self.prev_u is not None:
            sum_u_t = np.sum(u_t_clipped)
            sum_prev_u = np.sum(self.prev_u)
            
            if sum_u_t > 0 and sum_prev_u > 0:
                p = u_t_clipped / sum_u_t
                q = self.prev_u / sum_prev_u
                js_dist = jensenshannon(p, q)
                jsd_val = float(js_dist ** 2)
            else:
                jsd_val = 0.0
                
            ks_res = stats.ks_2samp(u_t_clipped, self.prev_u)
            ks_val = float(ks_res.statistic)
            
        # 3. Calculate rank stability metrics (Spearman correlation)
        # Cumulative visits rank
        if not self.centre_performance.empty:
            v_t_cum_series = self.centre_performance.sum(axis=1)
            v_t_cum = v_t_cum_series.reindex(self.centre_ids, fill_value=0.0)
        else:
            v_t_cum = pd.Series(0.0, index=self.centre_ids)
            
        r_t_cum = v_t_cum.rank(ascending=False, method='min').values
        self.rank_history_cumulative.append(r_t_cum)
        
        spearman_cum_day_to_day = np.nan
        if self.prev_r_cumulative is not None:
            res_sp = stats.spearmanr(r_t_cum, self.prev_r_cumulative)
            spearman_cum_day_to_day = float(res_sp.statistic) if not np.isnan(res_sp.statistic) else np.nan
            
        self.prev_r_cumulative = r_t_cum.copy()
        
        # Daily visits rank
        if perf_update is not None and not perf_update.empty:
            v_t_daily_series = perf_update.sum(axis=1)
            v_t_daily = v_t_daily_series.reindex(self.centre_ids, fill_value=0.0)
        else:
            v_t_daily = pd.Series(0.0, index=self.centre_ids)
            
        r_t_daily = v_t_daily.rank(ascending=False, method='min').values
        self.rank_history_daily.append(r_t_daily)
        
        spearman_daily_day_to_day = np.nan
        if self.prev_r_daily is not None:
            res_sp = stats.spearmanr(r_t_daily, self.prev_r_daily)
            spearman_daily_day_to_day = float(res_sp.statistic) if not np.isnan(res_sp.statistic) else np.nan
            
        self.prev_r_daily = r_t_daily.copy()
        
        # Compile record containing both the convergence metrics, rank stability metrics, and the raw distributions
        record = {
            'Day': day,
            'JSD': jsd_val,
            'KS_Distance': ks_val,
            'Spearman_Cum_Day_to_Day': spearman_cum_day_to_day,
            'Spearman_Cum_Final_Anchor': np.nan,
            'Spearman_Daily_Day_to_Day': spearman_daily_day_to_day,
            'Spearman_Daily_Final_Anchor': np.nan
        }
        for i, centre_id in enumerate(self.centre_ids):
            record[centre_id] = u_t_clipped[i]
            
        self.utility_convergence_records.append(record)
        
        self.prev_u = u_t_clipped.copy()

    def _prepare_tt_lookup(self, tt_lookup: dict) -> dict:
        """
        Converts nested dicts to DataFrames for fast vectorized indexing.
        Supports both 2D lookups (postcode, rc_id) and 1D lookups (postcode).
        """
        if not tt_lookup:
            return {}
            
        processed = {}
        for mode, data in tt_lookup.items():
            if not data:
                continue
                
            # If data is a dict of dicts, convert to DataFrame
            # data = {postcode: {rc_id: time}}
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                df = pd.DataFrame.from_dict(data, orient='index')
                processed[mode] = df
            else:
                # Flat dict {postcode: time} (e.g. walk time to nearest)
                processed[mode] = pd.Series(data)
        return processed

    def _lookup_travel_times(self, postcode_series: pd.Series, transport_mode_series: pd.Series, dest_series: pd.Series) -> pd.Series:
        """
        Records the travel time for each agent trip using vectorized lookups.
        Handles both single-value times and location-specific destination times.
        """
        # Pre-convert indices to strings/categories once to avoid daily cost
        if not self.tt_lookup_dfs:
            return pd.Series(0.0, index=postcode_series.index)
            
        pcs = postcode_series.values
        tms = transport_mode_series.values
        rcs = dest_series.values
        out = np.full(len(pcs), np.nan, dtype=np.float32)

        for mode in np.unique(tms):
            if mode not in self.tt_lookup_dfs:
                continue
                
            mask = (tms == mode)
            lookup_obj = self.tt_lookup_dfs[mode]
            
            if isinstance(lookup_obj, pd.DataFrame):
                # Vectorized 2D lookup: (postcode, rc_id)
                # Filter to current mode agents
                m_pcs = pcs[mask]
                m_rcs = rcs[mask]
                
                # Get integer indices for postcodes and RCs
                # Use get_indexer to handle missing keys with -1
                pc_idxs = lookup_obj.index.get_indexer(m_pcs)
                rc_idxs = lookup_obj.columns.get_indexer(m_rcs)
                
                # Only attempt lookup where both keys exist
                valid = (pc_idxs != -1) & (rc_idxs != -1)
                if valid.any():
                    # Fancy indexing on the underlying numpy array
                    vals = lookup_obj.values[pc_idxs[valid], rc_idxs[valid]]
                    
                    # Map back to output array
                    # We need the indices of 'mask' where valid is True
                    mask_indices = np.where(mask)[0]
                    out[mask_indices[valid]] = vals.astype(np.float32)
            
            elif isinstance(lookup_obj, pd.Series):
                # Vectorized 1D lookup: (postcode)
                m_pcs = pcs[mask]
                pc_idxs = lookup_obj.index.get_indexer(m_pcs)
                valid = (pc_idxs != -1)
                if valid.any():
                    vals = lookup_obj.values[pc_idxs[valid]]
                    mask_indices = np.where(mask)[0]
                    out[mask_indices[valid]] = vals.astype(np.float32)

        return pd.Series(out, index=postcode_series.index)
