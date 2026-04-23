import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import warnings

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import agent
from assign_trip_frequencies import assign_frequencies
import visualization

def log_msg(msg):
    st.session_state.log_messages.append(msg)

def _clean_rc_id(x):
    try:
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
    except:
        return str(x)

TRANSPORT_SUFFIXES = ['_walk', '_drive', '_pt']


# ---------------------------------------------------------------------------
# Module-level helpers (reused by cached loader)
# ---------------------------------------------------------------------------

def _split_matrices_fn(df, household_col='household'):
    """Split a utility-scores DataFrame into per-mode matrices."""
    mode_cols = {'walk': [], 'drive': [], 'pt': []}
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
            meta_cols.append(col)
    meta_df = df[meta_cols].copy()
    mode_dfs = {}
    for mode, cols in mode_cols.items():
        if not cols:
            continue
        mat = df[cols].astype(np.float32)
        suf = f'_{mode}'
        mat.columns = [c[:-len(suf)][:-2] if c[:-len(suf)].endswith('.0')
                       else c[:-len(suf)] for c in mat.columns]
        if household_col in df.columns:
            mat.index = df[household_col]
        mode_dfs[mode] = mat.fillna(0)
    return meta_df, mode_dfs


def _preprocess_transport_times(df):
    """
    Convert the raw transport-times DataFrame (Walk/Drive/PT columns containing
    {RC_ID: minutes} dicts) into a nested Python dict for O(1) lookup:
        {mode: {postcode: {rc_id_str: minutes}}}
    Built once at load time; never rebuilt during the simulation.
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


# ---------------------------------------------------------------------------
# Cached data loader  (runs ONCE per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading simulation data (first run only)…")
def _load_all_data(n_agents=None):
    """
    Reads parquets and builds static data structures.
    If n_agents is set, only loads those specific rows.
    """
    import geopandas as gpd

    import pyarrow.parquet as pq
    
    from main import TEST_MODE
    base_utility_dir = os.path.join(config.MODEL_DIR, "Utility")
    if TEST_MODE:
        base_utility_dir = os.path.join(base_utility_dir, "testing")

    print(f"Loading 6 trip-specific utility datasets from {base_utility_dir} ({n_agents or 'all'} agents)…")
    utility_matrices_base = {}
    consumers = None
    
    trip_types = [
        'bulk', 'convenience', 'comparison', 
        'entertainment', 'food_drink', 'service'
    ]
        
    for trip_type in trip_types:
        file_path = os.path.join(base_utility_dir, f'utility_scores_{trip_type}.parquet')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
            
        if n_agents:
            _pf = pq.ParquetFile(file_path)
            df = next(_pf.iter_batches(batch_size=n_agents)).to_pandas()
        else:
            df = pd.read_parquet(file_path)
            
        meta_df, mode_dfs = _split_matrices_fn(df)
        
        # The bulk dataset acts as our primary demographic source for initializing the population
        if trip_type == 'bulk':
            consumers = meta_df
            
        for tmode, mat in mode_dfs.items():
            utility_matrices_base[f'{trip_type}_{tmode}'] = mat
            
        import gc
        del df
        del mode_dfs
        if trip_type != 'bulk':
            del meta_df
        gc.collect()

    print("Loading retail centre amenity data…")
    gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
    gdf['RC_ID'] = gdf['RC_ID'].apply(_clean_rc_id)
    gdf = gdf.set_index('RC_ID')
    amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services',
                    'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
    amenity_binary = {col: (gdf[col] > 0).astype(float) for col in amenity_cols if col in gdf.columns}

    print("Loading & pre-processing transport times…")
    tt_df  = pd.DataFrame()
    tt_path = getattr(config, 'TRANSPORT_TIMES_PATH', '')
    if tt_path and os.path.exists(tt_path):
        try:
            tt_df = pd.read_parquet(tt_path)
            if 'Postcode' in tt_df.columns:
                tt_df = tt_df.set_index('Postcode')
            print(f"Transport times loaded ({tt_df.shape}). Building fast lookup dict…")
        except Exception as e:
            print(f"Warning: could not load transport times — {e}")
    tt_lookup = _preprocess_transport_times(tt_df)

    print(f"Cache ready: {len(consumers):,} agents | "
          f"{len(utility_matrices_base)} matrices | "
          f"tt modes: {list(tt_lookup.keys())}")
    return consumers, utility_matrices_base, amenity_binary, tt_lookup


def load_data(n_agents=None):
    """
    Returns simulation-ready data. Utility matrices are shallow-copied so that
    in-place feedback mutations during a run do NOT corrupt the shared cache.
    """
    consumers, utility_matrices_base, amenity_binary, tt_lookup = _load_all_data(n_agents=n_agents)
    log_msg(f"Using cached data ({len(consumers):,} agents)")
    utility_matrices = {k: v.copy() for k, v in utility_matrices_base.items()}
    return consumers, utility_matrices, amenity_binary, tt_lookup


# ---------------------------------------------------------------------------
# Cached visualization wrappers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating flow map…")
def _cached_flow_map(_visits_df, run_id, trip_type, tmode):
    """Cache HTML (string) by filter selection. Strings are safe to pickle."""
    return visualization.generate_hover_flow_map_html(_visits_df)


@st.cache_data(show_spinner="Calculating market share…")
def _cached_market_share(_visits_df, run_id, trip_type, tmode):
    """Cache Matplotlib Figure (picklable) by filter selection."""
    return visualization.plot_market_share_static_map(_visits_df)


@st.cache_resource(show_spinner="Building density map…")
def _cached_pydeck_map_v2(filtered_path, run_id, trip_type, tmode):
    """
    Cache PyDeck objects (non-picklable) by filter selection.
    Renamed to _v2 to force a fresh cache entry and avoid stale pickling errors.
    """
    return visualization.create_pydeck_map(filtered_path, as_2d=True)


def run_simulation(num_agents_req, days, status_text_placeholder, live_map_placeholder, progress_bar):
    consumers, utility_matrices, amenity_binary, tt_lookup = load_data(n_agents=num_agents_req)

    n = len(consumers)
    log_msg(f"Initialising {n} agents over {days} days…")

    state_df, consumers_sampled = agent.initialize_agent_state(n, consumers)

    # Persistent trackers for evaluation logic
    underperformer_tracker = {}
    cumulative_boosts = {}

    # Load retail centres GeoDataFrame for evaluation
    import geopandas as gpd
    retail_gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
    retail_gdf['RC_ID'] = retail_gdf['RC_ID'].apply(_clean_rc_id)
    retail_gdf = retail_gdf.set_index('RC_ID')

    def _lookup_tt(pc_series, tm_series, dest_series=None):
        """
        O(1) per-agent travel-time lookup using the pre-built nested dict.
        ~10-100× faster than querying the raw DataFrame per call.
        """
        if not tt_lookup:
            return pd.Series(0.0, index=pc_series.index)
        pcs = pc_series.values
        tms = tm_series.values
        rcs = dest_series.values if dest_series is not None else [None] * len(pcs)
        out = np.zeros(len(pcs), dtype=np.float32)
        for i in range(len(pcs)):
            mode_data = tt_lookup.get(tms[i], {})
            pc_data   = mode_data.get(pcs[i], None)
            
            if pc_data is None:
                # Postcode not in lookup at all
                out[i] = np.nan
            elif isinstance(pc_data, dict):
                rc_key = str(rcs[i]) if rcs[i] is not None else None
                if rc_key and rc_key in pc_data:
                    out[i] = pc_data[rc_key]
                else:
                    # Specific RC not found — return NaN rather than guessing with the mean
                    out[i] = np.nan
            else:
                try:
                    out[i] = float(pc_data)
                except Exception:
                    out[i] = np.nan
        return pd.Series(out, index=pc_series.index)


    all_visits = []
    track_ids = consumers_sampled.sample(n=min(10, len(consumers_sampled)))['household'].values
    live_visits = []
    pc_coords = visualization.load_postcode_coords()
    rc_coords = visualization.load_centre_coords_v2()
    home_postcodes = consumers_sampled.set_index('household')['Postcode'].to_dict()

    for day in range(1, days + 1):
        status_text_placeholder.text(f"Simulating Day {day}/{days}...")
        progress_bar.progress(day / days)
        day_trips = []

        # A. Grocery trips
        state_df      = agent.consume(state_df)
        needs_grocery = agent.check_shopping_need(state_df)
        if needs_grocery.any():
            grocery_mode_series = agent.choose_mode(consumers_sampled, needs_grocery)
            # Physical shops only (excludes online)
            physical_mask = needs_grocery & grocery_mode_series.isin(['bulk', 'convenience'])
            
            # Jointly select mode and destination
            destinations, transport_mode_series, utility_scores = agent.choose_destination(
                state_df, physical_mask,
                grocery_mode_series,
                utility_matrices,
                amenity_binary)

            valid = destinations.notna() & (grocery_mode_series != 'online')
            if valid.any():
                idx = destinations[valid].index
                pc_s  = state_df.loc[idx, 'Postcode']
                tm_s  = transport_mode_series.loc[idx]
                tt    = _lookup_tt(pc_s, tm_s, dest_series=destinations[valid])
                
                dt = pd.DataFrame({
                    'Day':            day,
                    'AgentID':        state_df.loc[idx, 'AgentID'].values,
                    'Postcode':       pc_s.values,
                    'Trip_Type':      'grocery',
                    'Retail_Centre':  destinations[valid].values,
                    'Grocery_Mode':   grocery_mode_series[valid].values,
                    'Transport_Mode': transport_mode_series.loc[idx].values,
                    'Travel_Time_Min': tt.values,
                    'Utility_Score':  utility_scores.loc[idx].values
                })
                # Filter out distance anomalies (246 min artifacts)
                dt = dt[dt['Travel_Time_Min'] < 240]
                if not dt.empty:
                    all_visits.append(dt)
                    day_trips.append(dt)

            state_df = agent.update_stock_after_shop(state_df, needs_grocery)

        # B. NTS frequency-based trips
        triggered_trips = agent.trigger_trips(consumers_sampled)

        for trip_type, triggered_mask in triggered_trips.items():
            if not triggered_mask.any():
                continue

            # Jointly select mode and destination for NTS trips
            dests, modes_used, utility_scores = agent.choose_destination_for_trip(
                trip_type, triggered_mask, consumers_sampled,
                utility_matrices, amenity_binary)

            valid = dests.notna()
            if valid.any():
                idx = dests[valid].index
                
                # Fetch pc_s from either state_df (if grocery) or consumers_sampled
                pc_s = consumers_sampled.loc[idx, 'Postcode']
                tm_s = modes_used.loc[idx]
                tt   = _lookup_tt(pc_s, tm_s, dest_series=dests[valid])

                dt = pd.DataFrame({
                    'Day':            day,
                    'AgentID':        consumers_sampled.loc[idx, 'household'].values,
                    'Postcode':       pc_s.values,
                    'Trip_Type':      trip_type,
                    'Retail_Centre':  dests[valid].values,
                    'Grocery_Mode':   'physical', # default for NTS
                    'Transport_Mode': modes_used.loc[idx].values,
                    'Travel_Time_Min': tt.values,
                    'Utility_Score':  utility_scores.loc[idx].values
                })
                # Filter out distance anomalies (246 min artifacts)
                dt = dt[dt['Travel_Time_Min'] < 240]
                if not dt.empty:
                    all_visits.append(dt)
                    day_trips.append(dt)


        # ── C. Dynamic Evaluation (every 10 days) ─────────────────────
        if day % 10 == 0:
            start_day     = day - 10 + 1
            period_visits = [df for df in all_visits
                             if df['Day'].iloc[0] >= start_day
                             and df['Day'].iloc[0] <= day]
            if period_visits:
                eval_df = pd.concat(period_visits, ignore_index=True)
                
                # 1. Hierarchy-Aware Peer Evaluation
                eval_msgs = agent.evaluate_retail_centres(
                    eval_df, retail_gdf, utility_matrices, amenity_binary,
                    tracker=underperformer_tracker, 
                    cumulative_boosts=cumulative_boosts)
                for m in eval_msgs: log_msg(m)
                
                # 2. Spatial Diffusion (Word-of-Mouth)
                diff_msgs = agent.apply_spatial_diffusion_bonus(
                    eval_df, consumers_sampled, utility_matrices)
                for m in diff_msgs: log_msg(m)

        # ── D. Update Live Map ────────────────────────────────────────
        if day_trips:
            current_day_all  = pd.concat(day_trips)
            tracked_this_day = current_day_all[current_day_all['AgentID'].isin(track_ids)]
            if not tracked_this_day.empty:
                for _, row in tracked_this_day.iterrows():
                    h_pc    = home_postcodes.get(row['AgentID'])
                    h_coord = pc_coords[pc_coords['Postcode'] == h_pc]
                    r_id    = str(row['Retail_Centre'])
                    # Safety check: if rc_coords failed to load, don't crash the simulation
                    if rc_coords.empty or 'clean_id' not in rc_coords.columns:
                        continue
                        
                    r_coord = rc_coords[rc_coords['clean_id'] == r_id]
                    if not h_coord.empty and not r_coord.empty:
                        live_visits.append({
                            'source': [h_coord.iloc[0]['longitude'], h_coord.iloc[0]['latitude']],
                            'target': [r_coord.iloc[0]['center_lon'], r_coord.iloc[0]['center_lat']],
                            'color':  [255, 165, 0, 200]
                        })
                import pydeck as pdk
                view  = pdk.ViewState(latitude=53.4, longitude=-2.9, zoom=9)
                layer = pdk.Layer("ArcLayer", live_visits[-50:],
                                  get_source_position="source",
                                  get_target_position="target",
                                  get_width=3,
                                  get_source_color="color",
                                  get_target_color="color")
                live_map_placeholder.pydeck_chart(
                    pdk.Deck(layers=[layer], initial_view_state=view, map_style="dark"))

    log_msg("Saving results...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if all_visits:
        total_visits_df = pd.concat(all_visits, ignore_index=True)
        output_path = os.path.join(
            config.OUTPUT_DIR, f"visits_log_{int(time.time())}.parquet")
        total_visits_df.to_parquet(output_path)
        log_msg(f"Done. {len(total_visits_df):,} visits saved.")
        return total_visits_df, output_path
    else:
        log_msg("Simulation complete — no visits occurred.")
        return None, None

def main():
    st.set_page_config(page_title="Retail ABM Simulation", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #F3F4F6;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Retail ABM Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure and run the Agent-Based Model to generate synthetic consumer mobility flows.</div>', unsafe_allow_html=True)
    
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
        
    with st.sidebar:
        st.header("Configuration")
        num_agents = st.number_input("Number of Agents", min_value=1, max_value=656817, value=1000, step=100, help="Total households to simulate (Max: 656,817).")
        days = st.number_input("Days to Simulate", min_value=1, max_value=365, value=30, step=1, help="Simulation duration in days.")
        
        st.markdown("---")
        st.subheader("Spatial & Intervention Controls")
        
        distance_decay = st.slider("Distance Sensitivity (Modifier)", 0.5, 3.0, 1.0, 0.1, help="Controls how sensitive agents are to distance/utility. >1 sharply penalises faraway/suboptimal destinations, forcing hyper-local shopping.")
        app_interv = st.slider("Retail Intervention (Boost Largest Centre)", 1.0, 10.0, 1.0, 0.5, help="Simulate a massive multi-million pound investment in the region's largest retail centre, multiplying its attractiveness score.")
        conformity = st.slider("Neighbourhood Conformity (Echo Chamber Effect)", 0.0, 1.0, 0.0, 0.1, help="Simulates an effect where households are heavily influenced by their immediate neighbours, causing them to blindly lock into whatever the most popular retail centre in their immediate postcode is.")
        
        st.markdown("---")
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.log_messages = []
        
        # Apply strict spatial behavioural overrides to the engine configuration
        config.DISTANCE_SENSITIVITY = distance_decay
        config.RETAIL_INTERVENTION = app_interv
        config.NEIGHBOURHOOD_CONFORMITY = conformity
        
        status_text_placeholder = st.empty()
        live_map_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status_text_placeholder.info("Starting simulation...")
        
        try:
            visits_df, output_path = run_simulation(num_agents, days, status_text_placeholder, live_map_placeholder, progress_bar)
            
            # Save to session state so they persist when dropdowns are changed
            if visits_df is not None:
                st.session_state.visits_df = visits_df
                st.session_state.output_path = output_path
                st.session_state.simulation_days = days
            else:
                if 'visits_df' in st.session_state:
                    del st.session_state.visits_df
            
            progress_bar.empty()
            status_text_placeholder.success("Simulation Complete")
            time.sleep(1) # Short delay to show success message
            status_text_placeholder.empty()
            live_map_placeholder.empty()
            
        except Exception as e:
            st.error(f"Error during simulation: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    # Outside the run_btn condition so it persists when dropdowns trigger re-runs
    if 'visits_df' in st.session_state and st.session_state.visits_df is not None:
        visits_df = st.session_state.visits_df
        output_path = st.session_state.output_path
        days = st.session_state.simulation_days
        
        st.markdown("---")
        st.header("Simulation Results Overview")
                
        # Format trip types nicely
        trip_type_mapping = {
            'grocery': 'Grocery',
            'comparison': 'Comparison',
            'service': 'Service',
            'entertainment': 'Entertainment',
            'food_drink': 'Food & Drink'
        }
        visits_df['Trip_Type_Display'] = visits_df['Trip_Type'].map(trip_type_mapping).fillna(visits_df['Trip_Type'].str.title())
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        total_visits = len(visits_df)
        total_agents_active = visits_df['AgentID'].nunique()
        avg_visits_day = total_visits / days
        visits_per_agent = total_visits / total_agents_active if total_agents_active > 0 else 0
        
        m1.metric("Total Visits Generated", f"{total_visits:,}")
        m2.metric("Active Agents", f"{total_agents_active:,}")
        m3.metric("Avg Visits / Day", f"{avg_visits_day:,.1f}")
        m4.metric("Avg Visits / Agent", f"{visits_per_agent:,.1f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Flow Analysis", "Market Share Analysis", "Visitation Map", "Trip Analytics", "Execution Logs"])
        
        with tab1:
            import streamlit.components.v1 as components
            st.subheader("Spatial Catchment & Flow Network Data")
            st.markdown("Hover your mouse over **any Retail Centre** to instantly reveal the geographic origins (postcodes) of its visitors. The background heatmap denotes regional market share dominance.")
            
            # Add dropdown filters
            col_t, col_m = st.columns([1, 1])
            with col_t:
                trip_types_unique = sorted([str(x) for x in visits_df['Trip_Type_Display'].unique() if pd.notna(x)])
                trip_types_available = ['All'] + trip_types_unique
                selected_trip_flow = st.selectbox("Filter flows by Trip Purpose:", trip_types_available, index=0, key="trip_type_filter_flow")
            with col_m:
                selected_transport_flow = st.selectbox("Filter flows by Transport Mode:", ['All Modes', 'Walk', 'Drive', 'PT'], index=0, key="transport_mode_filter_flow")
            
            # Filter data before passing to visualization
            visits_df_flow = visits_df.copy()
            if selected_trip_flow != 'All':
                rev_map = {v: k for k, v in trip_type_mapping.items()}
                internal_type = rev_map.get(selected_trip_flow, selected_trip_flow.lower())
                visits_df_flow = visits_df_flow[visits_df_flow['Trip_Type'] == internal_type]
            
            if selected_transport_flow != 'All Modes' and 'Transport_Mode' in visits_df_flow.columns:
                visits_df_flow = visits_df_flow[visits_df_flow['Transport_Mode'] == selected_transport_flow.lower()]
                
            html_str = _cached_flow_map(
                visits_df_flow, output_path, selected_trip_flow, selected_transport_flow)
            if html_str:
                components.html(html_str, height=620)
            else:
                st.info("No spatial flow data available for this selection.")
                    
        with tab2:
            st.subheader("Regional Market Share Dominance")
            st.markdown("This static map highlights the 'Dominant' Retail Centre for every postcode based on the simulated visitation frequency.")
            
            # Add dropdown filters
            col_t, col_m = st.columns([1, 1])
            with col_t:
                selected_trip_share = st.selectbox("Analyze Dominance by Trip Purpose:", trip_types_available, index=0, key="trip_type_filter_share")
            with col_m:
                selected_transport_share = st.selectbox("Analyze Dominance by Transport Mode:", ['All Modes', 'Walk', 'Drive', 'PT'], index=0, key="transport_mode_filter_share")
            
            # Filter data before passing to visualization
            visits_df_share = visits_df.copy()
            if selected_trip_share != 'All':
                rev_map = {v: k for k, v in trip_type_mapping.items()}
                internal_type = rev_map.get(selected_trip_share, selected_trip_share.lower())
                visits_df_share = visits_df_share[visits_df_share['Trip_Type'] == internal_type]

            if selected_transport_share != 'All Modes' and 'Transport_Mode' in visits_df_share.columns:
                visits_df_share = visits_df_share[visits_df_share['Transport_Mode'] == selected_transport_share.lower()]
                
            fig = _cached_market_share(
                visits_df_share, output_path, selected_trip_share, selected_transport_share)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.info("No spatial data available for this selection.")
                    
        with tab3:
            st.subheader("Visitation Density Map")

            col_left, col_right = st.columns([1, 1])
            with col_left:
                selected_trip_map = st.selectbox(
                    "Filter map by Trip Type:",
                    trip_types_available, index=0, key="trip_type_filter_map")
            with col_right:
                # ── NEW: Transport Mode dropdown ──────────────────────────────
                transport_mode_opts = ['All Modes', 'Walk', 'Drive', 'PT']
                selected_transport  = st.selectbox(
                    "Filter map by Transport Mode:",
                    transport_mode_opts, index=0, key="transport_mode_filter_map")

            # Filter data
            filtered_visits = visits_df.copy()
            if selected_trip_map != 'All':
                rev_map       = {v: k for k, v in trip_type_mapping.items()}
                internal_type = rev_map.get(selected_trip_map, selected_trip_map.lower())
                filtered_visits = filtered_visits[filtered_visits['Trip_Type'] == internal_type]

            tmode_arg = None
            if selected_transport != 'All Modes' and 'Transport_Mode' in filtered_visits.columns:
                tmode_arg = selected_transport.lower()
                filtered_visits = filtered_visits[
                    filtered_visits['Transport_Mode'] == tmode_arg]

            filtered_path = os.path.join(config.OUTPUT_DIR, f"tmp_{os.path.basename(output_path)}_{selected_trip_map}_{selected_transport}.parquet")
            filtered_visits.to_parquet(filtered_path)

            deck_map = _cached_pydeck_map_v2(
                filtered_path, output_path, selected_trip_map, selected_transport)
            if deck_map is not None:
                st.pydeck_chart(deck_map)
            else:
                st.warning("Interactive map could not be generated. Please check logs.")
                
        with tab4:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Simulated Visits by Trip Category")
                counts = visits_df['Trip_Type_Display'].value_counts().reset_index()
                counts.columns = ['Trip Category', 'Visit Count']

                import plotly.express as px
                fig = px.pie(counts, values='Visit Count', names='Trip Category',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Distribution Data Table")
                counts['% of Total'] = (
                    counts['Visit Count'] / total_visits * 100).round(1).astype(str) + '%'
                st.dataframe(counts, use_container_width=True, hide_index=True)

                # ── Transport Mode breakdown ──────────────────────────────────
                if 'Transport_Mode' in visits_df.columns:
                    st.subheader("Visits by Transport Mode")
                    mode_counts = visits_df['Transport_Mode'].value_counts().reset_index()
                    mode_counts.columns = ['Transport Mode', 'Visit Count']
                    mode_fig = px.bar(
                        mode_counts, x='Transport Mode', y='Visit Count',
                        color='Transport Mode',
                        color_discrete_map={'walk': '#22c55e', 'drive': '#3b82f6', 'pt': '#f59e0b'},
                        text='Visit Count')
                    mode_fig.update_layout(showlegend=False,
                                           margin=dict(t=10, b=0, l=0, r=0))
                    st.plotly_chart(mode_fig, use_container_width=True)

                    if 'Travel_Time_Min' in visits_df.columns:
                        st.subheader("Avg Travel Time by Mode (mins)")
                        tt_summary = (visits_df.groupby('Transport_Mode')['Travel_Time_Min']
                                      .mean().round(1).reset_index())
                        tt_summary.columns = ['Mode', 'Avg Travel Time (min)']
                        st.dataframe(tt_summary, use_container_width=True, hide_index=True)

                st.subheader("Temporal Activity Trend")
                daily_visits = visits_df.groupby('Day').size().reset_index(name='Visits')
                fig2 = px.line(daily_visits, x='Day', y='Visits', markers=True,
                               color_discrete_sequence=['#3b82f6'])
                fig2.update_layout(margin=dict(t=10, b=0, l=0, r=0),
                                   yaxis_title="Number of Visits",
                                   xaxis_title="Simulation Day")
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab5:
            st.code("\n".join(st.session_state.log_messages), language="text")

    else:
        st.warning("Simulation finished but no visits occurred.")
        if 'log_messages' in st.session_state:
            with st.expander("Execution Logs", expanded=True):
                st.code("\n".join(st.session_state.log_messages), language="text")

if __name__ == "__main__":
    main()
