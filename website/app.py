import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import warnings

# Add simulation/ to path so we can import the core model modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_SIM_DIR = os.path.join(_PROJECT_ROOT, "simulation")
sys.path.insert(0, _SIM_DIR)
import config
from simulation.core.simulation_engine import SimulationEngine
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

@st.cache_resource(show_spinner="Loading datasets...")
def _load_all_data(n_agents=None):
    """
    Reads 6 trip-specific parquets and builds static data structures.
    """
    import geopandas as gpd

    print(f"Loading 6 trip-specific utility datasets from {config.UTILITY_DIR}...")
    utility_matrices_base = {}
    consumers = None
    
    trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
    suffixes = ['_walk', '_drive', '_pt']

    for trip_type in trip_types:
        file_path = os.path.join(config.UTILITY_DIR, f'utility_scores_{trip_type}.parquet')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
            
        df = pd.read_parquet(file_path)
        if n_agents and len(df) > n_agents:
            df = df.sample(n=n_agents, random_state=42)
            
        # Extract matrices
        for suf in suffixes:
            mode = suf.lstrip('_')
            cols = [c for c in df.columns if c.endswith(suf)]
            if not cols: continue
            
            mat = df[cols].astype(np.float32)
            mat.columns = [_clean_rc_id(c[:-len(suf)]) for c in mat.columns]
            
            # CRITICAL FIX: Ensure index is unique to prevent reindex row multiplication
            mat.index = df['household']
            if not mat.index.is_unique:
                mat = mat[~mat.index.duplicated(keep='first')]
                
            utility_matrices_base[f'{trip_type}_{mode}'] = mat.fillna(0)

        # Bulk dataset provides the core population demographics
        if trip_type == 'bulk':
            meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in suffixes)]
            consumers = df[meta_cols].copy()
            if not consumers['household'].is_unique:
                consumers = consumers.drop_duplicates(subset='household', keep='first')

        import gc
        del df
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

    engine = SimulationEngine(consumers_df=consumers, utility_matrices=utility_matrices, 
                              amenity_binary=amenity_binary, tt_lookup=tt_lookup)
                              
    # Setup log callback to update Streamlit progress
    def log_callback(msg):
        status_text_placeholder.text(msg)
        log_msg(msg)
        if msg.startswith("Day "):
            try:
                # Parse "Day X/Y..."
                parts = msg.split()[1].split('/')
                current_day = int(parts[0])
                total_days = int(parts[1].replace('...', ''))
                progress_bar.progress(current_day / total_days)
            except:
                pass
                
    visits = engine.run(num_agents=num_agents_req, days=days, eval_freq=10, log_callback=log_callback)
    
    log_msg("Saving results...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if visits:
        total_visits_df = pd.concat(visits, ignore_index=True)
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
        st.header("Configuration Panel")
        
        # Organize settings into expanders for a cleaner UI
        with st.expander("Population & Time", expanded=True):
            num_agents = st.number_input("Number of Agents", min_value=1, max_value=656817, value=1000, step=100, help="Total households to simulate (Max: 656,817).")
            days = st.number_input("Days to Simulate", min_value=1, max_value=365, value=30, step=1, help="Simulation duration in days.")

            
        with st.expander("Social Dynamics", expanded=False):
            st.markdown("Control how agents are influenced by their peers.")
            randomize_social = st.toggle("Randomize Social Traits", value=getattr(config, 'RANDOMIZE_SOCIAL_ATTRIBUTES', True), help="If enabled, every agent gets unique, randomized social parameters. Disabling this allows manual global calibration below.")
            
            # Only show sliders if we aren't randomizing
            if not randomize_social:
                demo_weight = st.slider("Demographic Weight", 0.0, 1.0, getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.8), 0.1, help="0.0 = Copy physical neighbors. 1.0 = Copy demographic peers only.")
                demo_bw = st.slider("Demographic Bandwidth", 0.1, 1.0, getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5), 0.1, help="Tolerance for 'similarity' (Lower = Stricter).")
                conformity = st.slider("Neighbourhood Conformity", 0.0, 1.0, getattr(config, 'NEIGHBOURHOOD_CONFORMITY', 0.2), 0.1, help="0.0 = No peer pressure. 1.0 = Always follow the local trend.")
            else:
                demo_weight = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.8)
                demo_bw = getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5)
                conformity = getattr(config, 'NEIGHBOURHOOD_CONFORMITY', 0.2)
                
        with st.expander("Retail Economics & Policy", expanded=False):
            st.markdown("Set market interventions and rationality.")
            distance_decay = st.slider("Distance Sensitivity", 0.5, 3.0, getattr(config, 'DISTANCE_SENSITIVITY', 1.0), 0.1, help=">1 sharply penalizes faraway destinations.")
            app_interv = st.slider("Global Retail Intervention Boost", 1.0, 10.0, getattr(config, 'RETAIL_INTERVENTION', 1.0), 0.5, help="Multiply attractiveness score of the largest centre.")
            fail_thresh = st.slider("Failure Threshold (%)", 1, 30, int(getattr(config, 'RETAIL_FAILURE_THRESHOLD', 0.10) * 100), 1, help="Percentile below which a centre triggers automated welfare intervention.") / 100.0
            interv_boost = st.slider("Automated Intervention Boost", 1.0, 2.0, getattr(config, 'RETAIL_INTERVENTION_BOOST', 1.10), 0.05, help="Multiplier applied to centres falling below the failure threshold.")
            beta = st.slider("Decision Temperature (Beta)", 1.0, 15.0, getattr(config, 'SOFTMAX_BETA', 5.0), 1.0, help="Low = Exploratory choices. High = Strict optimization.")
        
        st.markdown("---")
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.log_messages = []
        
        # Apply strict behavioural overrides to the engine configuration
        config.DISTANCE_SENSITIVITY = distance_decay
        config.RETAIL_INTERVENTION = app_interv
        
        # Apply new social & economic parameters
        config.RANDOMIZE_SOCIAL_ATTRIBUTES = randomize_social
        if not randomize_social:
            config.DEMOGRAPHIC_DIFFUSION_WEIGHT = demo_weight
            config.DEMOGRAPHIC_BANDWIDTH = demo_bw
            config.NEIGHBOURHOOD_CONFORMITY = conformity
            
        config.RETAIL_FAILURE_THRESHOLD = fail_thresh
        config.RETAIL_INTERVENTION_BOOST = interv_boost
        config.SOFTMAX_BETA = beta
        
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
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.header("📊 Simulation Results Overview")
        with col_hdr2:
            st.markdown("<br>", unsafe_allow_html=True)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="💾 Download Raw Data (.parquet)",
                    data=file,
                    file_name=os.path.basename(output_path),
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
        # Format trip types nicely
        trip_type_mapping = {
            'grocery': 'Grocery',
            'comparison': 'Comparison',
            'service': 'Service',
            'entertainment': 'Entertainment',
            'food_drink': 'Food & Drink'
        }
        visits_df['Trip_Type_Display'] = visits_df['Trip_Type'].map(trip_type_mapping).fillna(visits_df['Trip_Type'].str.title())
        
        # Metrics Row (Using custom CSS classes)
        total_visits = len(visits_df)
        total_agents_active = visits_df['AgentID'].nunique()
        avg_visits_day = total_visits / days
        visits_per_agent = total_visits / total_agents_active if total_agents_active > 0 else 0
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
            <div class="metric-card" style="flex: 1;">
                <h3 style="margin:0; font-size:1rem; color:#6B7280;">Total Visits Generated</h3>
                <p style="margin:0; font-size:2rem; font-weight:700; color:#1E3A8A;">{total_visits:,}</p>
            </div>
            <div class="metric-card" style="flex: 1;">
                <h3 style="margin:0; font-size:1rem; color:#6B7280;">Active Agents</h3>
                <p style="margin:0; font-size:2rem; font-weight:700; color:#1E3A8A;">{total_agents_active:,}</p>
            </div>
            <div class="metric-card" style="flex: 1;">
                <h3 style="margin:0; font-size:1rem; color:#6B7280;">Avg Visits / Day</h3>
                <p style="margin:0; font-size:2rem; font-weight:700; color:#1E3A8A;">{avg_visits_day:,.1f}</p>
            </div>
            <div class="metric-card" style="flex: 1;">
                <h3 style="margin:0; font-size:1rem; color:#6B7280;">Avg Visits / Agent</h3>
                <p style="margin:0; font-size:2rem; font-weight:700; color:#1E3A8A;">{visits_per_agent:,.1f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
