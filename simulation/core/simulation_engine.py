import pandas as pd
import numpy as np
import time
from simulation.agents.consumer.consumer_population import ConsumerPopulation
from simulation.agents.retail_centre.retail_manager import RetailManager
from simulation.core.utility_engine import apply_feedback
from simulation.core.constants import TRANSPORT_MODES, TRIP_TYPE_CONFIG

class SimulationEngine:
    def __init__(self, consumers_df, utility_matrices, amenity_binary, tt_lookup):
        self.consumers_df = consumers_df
        self.utility_matrices = utility_matrices
        self.amenity_binary = amenity_binary
        
        # Cache original utility scores to support social decay
        # This allows us to decay the "reputation" without shrinking the "geography"
        self.base_utilities = {k: v.copy() for k, v in utility_matrices.items()}

        # 1. Apply "Warm Start": Initial random social reputation
        start_sigma = getattr(config, 'SOCIAL_START_SIGMA', 0.0)
        if start_sigma > 0:
            for matrix in self.utility_matrices.values():
                # Add random noise N(0, sigma) to simulate pre-existing brand loyalty
                noise = np.random.normal(0, start_sigma, size=matrix.shape).astype(np.float16)
                matrix.update(matrix + noise)
        
        # Pre-process travel time lookup for vectorized access
        self.tt_lookup_dfs = self._prepare_tt_lookup(tt_lookup)
        
        self.population = None
        self.retail_manager = None

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

        from simulation.core import paths

        # --- Fix 1: period-streaming ---
        # current_period holds only the current eval window in RAM.
        # After each evaluation the period is written to a temp parquet and cleared,
        # keeping peak visit memory at eval_freq*rows/day instead of total_days*rows/day.
        tmp_dir = paths.OUTPUT_DIR / '_visits_tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        written_files = []     # temp parquet paths flushed to disk
        current_period = []    # DataFrames for the current eval window only

        # Build subcluster map once upfront so it is available during flush
        subcluster_map = None
        if 'Geo_Subcluster' in self.population.attributes.columns and 'household' in self.population.attributes.columns:
            # Ensure unique string mapping even if agents were sampled with replacement
            _map_df = self.population.attributes[['household', 'Geo_Subcluster']].copy()
            _map_df['household'] = _map_df['household'].astype(str)
            subcluster_map = _map_df.drop_duplicates('household').set_index('household')['Geo_Subcluster']

        for day in range(1, days + 1):
            day_start = time.time()
            log(f"Day {day}/{days}...")
            
            # 1. Consumption & Need Detection
            self.population.consume()
            needs_grocery = self.population.check_grocery_need()
            grocery_mode_series = self.population.choose_shopping_mode(needs_grocery)
            nts_triggered = self.population.trigger_nts_trips()

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

            # 5. Evaluation & Social Influence
            if day % eval_freq == 0:
                log(f"Evaluating retail centres (Day {day})...")
                if current_period:
                    eval_df = pd.concat(current_period, ignore_index=True)
                    messages = self.retail_manager.evaluate_centres(eval_df, self.utility_matrices)
                    for msg in messages: log(msg)

                    diffusion_msgs = self.population.apply_social_influence(eval_df, self.utility_matrices, self.base_utilities)
                    for msg in diffusion_msgs: log(msg)

                    # Attach Geo_Subcluster before flushing
                    if subcluster_map is not None:
                        eval_df['Geo_Subcluster'] = (
                            eval_df['AgentID'].astype(str)
                            .map(subcluster_map.astype(str))
                            .fillna('none')
                        )

                    # Write period to disk and free memory
                    fpath = tmp_dir / f'period_day{day}.parquet'
                    eval_df.to_parquet(fpath, index=False)
                    written_files.append(fpath)
                    del eval_df
                    current_period.clear()
            
            day_elapsed = time.time() - day_start
            log(f"  -> Day {day} complete in {day_elapsed:.2f}s")

        # Flush any remaining days (if days % eval_freq != 0)
        if current_period:
            remainder_df = pd.concat(current_period, ignore_index=True)
            if subcluster_map is not None:
                remainder_df['Geo_Subcluster'] = (
                    remainder_df['AgentID'].astype(str)
                    .map(subcluster_map.astype(str))
                    .fillna('none')
                )
            fpath = tmp_dir / 'period_remainder.parquet'
            remainder_df.to_parquet(fpath, index=False)
            written_files.append(fpath)
            del remainder_df
            current_period.clear()

        if not written_files:
            return []

        # Return the list of period parquet paths directly.
        # The caller is responsible for merging them on-disk via PyArrow
        # so that the full dataset is never materialised in RAM.
        return written_files

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
