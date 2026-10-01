from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mssso_core import (
    build_sensor_cube,
    compute_dtw_matrices,
    evaluate_gpr,
    load_sensor_data,
    measurement_columns,
    optimize_ga,
    spatial_coverage_score,
)


ROOT = Path(__file__).resolve().parent
INTEL_PATH = ROOT / "data" / "intel_lab_data.txt.gz"
LOCATIONS_PATH = ROOT / "data" / "mote_locs.txt"

st.set_page_config(page_title="MSSSO | Sensor Selection Lab", page_icon="◉", layout="wide")
appearance_mode = st.session_state.get("appearance_mode", "Light")
if appearance_mode == "Light":
    theme_tokens = ":root { --ink: #182722; --muted: #60736a; --paper: #f3f6f3; --panel: #ffffff; --line: #d8e1db; --teal: #14765f; --coral: #c65f45; --sidebar: #eaf0eb; --field: #f9fbf9; --header: rgba(243, 246, 243, .96); --chart: #ffffff; --chart-text: #334641; --axis: #6c7e75; --grid: #e3eae6; }"
else:
    theme_tokens = ":root { --ink: #e8f0ec; --muted: #a1b2ab; --paper: #111916; --panel: #1b2521; --line: #34443d; --teal: #67d4ad; --coral: #f09a78; --sidebar: #17211d; --field: #202c27; --header: rgba(17, 25, 22, .96); --chart: #19231f; --chart-text: #d5e2dc; --axis: #aebfb7; --grid: #2b3933; }"
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    __THEME_TOKENS__
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(ellipse at 84% 0%, rgba(20, 91, 76, .10), transparent 38%), var(--paper); }
    [data-testid="stAppViewContainer"] { color: var(--ink); }
    [data-testid="stHeader"] { background: var(--header); }
    [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--ink); }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, label, .stCaption, small { color: var(--ink); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    .block-container { max-width: 1680px; padding-top: 3.5rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { color: var(--ink); }
    h1 { font-family: 'Fraunces', Georgia, serif; font-size: 2.45rem !important; font-weight: 500 !important; }
    h2 { font-family: 'Fraunces', Georgia, serif; font-size: 1.55rem !important; font-weight: 500 !important; }
    .eyebrow, .mono, [data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.04em; }
    .eyebrow { font-size: .72rem; color: var(--teal); }
    .control-kicker { margin: .4rem 0 .15rem; color: var(--teal); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .04em; text-transform: uppercase; }
    .section-heading { display: flex; align-items: baseline; gap: .8rem; margin: 2rem 0 .2rem; border-bottom: 1px solid var(--line); padding-bottom: .6rem; }
    .section-number { color: var(--teal); font: 500 .72rem 'DM Mono', monospace; }
    [data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--muted); }
    [data-testid="stSlider"] [role="slider"] { background: var(--teal); }
    .intro { max-width: 760px; color: var(--muted); font-size: 1rem; line-height: 1.65; }
    .section-rule { border-top: 1px solid var(--line); margin: 1.3rem 0 1rem; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div,
    [data-testid="stFileUploader"] section { background: var(--field); border-color: var(--line); color: var(--ink); }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--muted); }
    [data-baseweb="tag"] { background: #165f53; color: #f1fffb; }
    [data-baseweb="tab-list"] { background: transparent; }
    [data-baseweb="tab"] { color: var(--muted); }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--teal); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    .stButton > button[kind="primary"] { background: var(--teal); color: var(--paper); border: 0; border-radius: 5px; min-height: 2.8rem; font-weight: 700; }
    .stButton > button[kind="primary"]:hover { filter: brightness(.92); }
    .stButton > button { border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { height: 2.8rem; }
    .caption { color: var(--muted); font-size: .88rem; }
    </style>
    """.replace("__THEME_TOKENS__", theme_tokens),
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_load(payload: bytes, filename: str, locations_payload: bytes | None) -> pd.DataFrame:
    return load_sensor_data(payload, filename, locations_payload)


def frame_payload(upload: object | None) -> bytes:
    if upload is None:
        return b""
    if isinstance(upload, bytes):
        return upload
    if isinstance(upload, bytearray):
        return bytes(upload)
    if isinstance(upload, (str, Path)):
        return Path(upload).read_bytes()
    if hasattr(upload, "getvalue"):
        return upload.getvalue()
    if hasattr(upload, "read"):
        position = upload.tell() if hasattr(upload, "tell") else None
        payload = upload.read()
        if position is not None and hasattr(upload, "seek"):
            upload.seek(position)
        return payload if isinstance(payload, bytes) else payload.encode("utf-8")
    raise ValueError("Unsupported file input.")


def chart_layout(figure: go.Figure, height: int = 340) -> go.Figure:
    chart_text = "#334641" if st.session_state.get("appearance_mode", "Light") == "Light" else "#d5e2dc"
    axis_text = "#6c7e75" if st.session_state.get("appearance_mode", "Light") == "Light" else "#aebfb7"
    grid = "#e3eae6" if st.session_state.get("appearance_mode", "Light") == "Light" else "#2b3933"
    chart_background = "#ffffff" if st.session_state.get("appearance_mode", "Light") == "Light" else "#19231f"
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chart_background,
        font={"family": "DM Sans, sans-serif", "color": chart_text, "size": 12},
        margin={"l": 8, "r": 8, "t": 24, "b": 8},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_xaxes(showgrid=False, linecolor=grid, tickfont={"color": axis_text}, title_font={"color": chart_text})
    figure.update_yaxes(gridcolor=grid, zeroline=False, tickfont={"color": axis_text}, title_font={"color": chart_text})
    return figure


def sensor_map_figure(cube: object, selected_indices: list[int]) -> go.Figure:
    selected_set = set(selected_indices)
    map_frame = pd.DataFrame(
        {
            "sensor_id": cube.sensor_ids,
            "x": cube.coordinates[:, 0],
            "y": cube.coordinates[:, 1],
            "selection": ["Selected" if index in selected_set else "Available" for index in range(len(cube.sensor_ids))],
        }
    )
    figure = px.scatter(
        map_frame,
        x="x",
        y="y",
        color="selection",
        hover_name="sensor_id",
        color_discrete_map={"Selected": "#14765f", "Available": "#aab8b0"},
        labels={"x": "X coordinate", "y": "Y coordinate", "selection": "Deployment"},
    )
    for trace in figure.data:
        is_selected = trace.name == "Selected"
        trace.marker.size = 16 if is_selected else 10
        trace.marker.opacity = 1 if is_selected else 0.58
        trace.marker.line = {"width": 1.5, "color": "white"}
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


st.sidebar.markdown("<div class='eyebrow'>MSSSO / FIELD LAB</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Experiment setup")
st.sidebar.markdown("<div class='control-kicker'>Data source</div>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Sensor readings", type=["csv", "txt", "gz"], accept_multiple_files=False)
locations_upload = st.sidebar.file_uploader("Sensor coordinates (optional)", type=["csv", "txt"], accept_multiple_files=False)

if uploaded is not None:
    payload = frame_payload(uploaded)
    locations_payload = frame_payload(locations_upload) if locations_upload is not None else None
    raw_data = cached_load(payload, uploaded.name, locations_payload)
    dataset_name = uploaded.name
    dataset_key = f"uploaded-{uploaded.name}-{uploaded.size}"
    st.sidebar.caption("Custom data source loaded")
else:
    try:
        payload = INTEL_PATH.read_bytes()
        locations_payload = LOCATIONS_PATH.read_bytes()
        raw_data = cached_load(payload, INTEL_PATH.name, locations_payload)
        dataset_name = "Intel Berkeley Research Lab (bundled)"
        dataset_key = f"intel-lab-{INTEL_PATH.stat().st_size}"
        st.sidebar.caption("Intel Berkeley Research Lab · bundled dataset")
    except Exception as error:
        st.error(f"Could not load this dataset: {error}")
        st.stop()

variables_available = measurement_columns(raw_data)
if not variables_available:
    st.error("No numeric environmental measurement columns were found.")
    st.stop()

preferred_variables = ["temperature", "humidity", "light", "voltage"]
default_variables = [variable for variable in preferred_variables if variable in variables_available]
if not default_variables:
    default_variables = variables_available[: min(2, len(variables_available))]

st.sidebar.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='control-kicker'>Environment variables</div>", unsafe_allow_html=True)
selected_variables = st.sidebar.multiselect(
    "Environmental variables",
    variables_available,
    default=default_variables,
)
sensor_count = raw_data["sensor_id"].nunique()
st.sidebar.markdown("<div class='control-kicker'>Deployment budget</div>", unsafe_allow_html=True)
subset_size = st.sidebar.slider("Sensors to deploy", min_value=2, max_value=max(2, sensor_count - 1), value=min(5, max(2, sensor_count - 1)))
st.sidebar.markdown("<div class='control-kicker'>Genetic algorithm</div>", unsafe_allow_html=True)
population_size = st.sidebar.slider("GA population", min_value=12, max_value=72, value=36, step=4)
generations = st.sidebar.slider("GA generations", min_value=5, max_value=60, value=24, step=1)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
run_requested = st.sidebar.button("Run TSEM + GA", type="primary", width="stretch")
st.sidebar.segmented_control("Appearance", ["Light", "Dark"], default=appearance_mode, key="appearance_mode")
st.sidebar.caption("Measurements: temperature, humidity, light, voltage.")
st.sidebar.caption("Record fields: date/time and sensor/mote ID (metadata, not optimization variables).")
st.sidebar.caption("A 70:30 chronological split is used for selection and held-out evaluation.")

current_signature = (dataset_key, tuple(selected_variables), subset_size, population_size, generations, int(seed))
st.markdown("<div class='eyebrow'>ADVANCED ALGORITHMS / SENSOR NETWORK OPTIMIZATION</div>", unsafe_allow_html=True)
st.title("Choose fewer sensors. Keep the signal.")
st.markdown(
    "<p class='intro'>A working MSSSO prototype: compare temporal behavior with Dynamic Time Warping, "
    "search fixed-size sensor subsets using a Genetic Algorithm, then test spatial reconstruction on "
    "unselected sensors.</p>",
    unsafe_allow_html=True,
)

if run_requested:
    if not selected_variables:
        st.sidebar.error("Choose at least one variable.")
    elif subset_size >= sensor_count:
        st.sidebar.error("Choose fewer sensors than the total network size.")
    else:
        try:
            with st.spinner("Computing pairwise DTW, evolving subsets, and evaluating reconstruction..."):
                started = time.perf_counter()
                cube = build_sensor_cube(raw_data, selected_variables)
                train_values = cube.values[:, : cube.train_end, :]
                distance_matrices = compute_dtw_matrices(train_values)
                selected_indices, fitness_history, best_fitness = optimize_ga(
                    distance_matrices,
                    subset_size,
                    population_size=population_size,
                    generations=generations,
                    seed=int(seed),
                )
                random_generator = np.random.default_rng(int(seed))
                random_indices = sorted(
                    random_generator.choice(sensor_count, size=subset_size, replace=False).tolist()
                )
                selected_evaluation = evaluate_gpr(cube, selected_indices)
                random_evaluation = evaluate_gpr(cube, random_indices)
                elapsed = time.perf_counter() - started
                st.session_state["experiment"] = {
                    "signature": current_signature,
                    "cube": cube,
                    "distances": distance_matrices,
                    "selected": selected_indices,
                    "random": random_indices,
                    "fitness_history": fitness_history,
                    "fitness": best_fitness,
                    "selected_evaluation": selected_evaluation,
                    "random_evaluation": random_evaluation,
                    "elapsed": elapsed,
                }
        except Exception as error:
            st.error(f"Experiment failed: {error}")

experiment = st.session_state.get("experiment")
is_current = experiment is not None and experiment["signature"] == current_signature
if experiment is not None and not is_current:
    st.warning("Settings or dataset changed. Run the optimizer again to refresh these results.")

times_count = raw_data["timestamp"].nunique()
metric_columns = st.columns(4)
metric_columns[0].metric("Candidate sensors", f"{sensor_count}")
metric_columns[1].metric("Time points", f"{times_count:,}")
metric_columns[2].metric("Signals", f"{len(variables_available)}")
metric_columns[3].metric("Data source", "Intel Lab" if uploaded is None else "Uploaded")

tab_selection, tab_evaluation, tab_dataset = st.tabs(["Selection", "Evaluation", "Dataset & method"])

with tab_selection:
    if is_current:
        selected = experiment["selected"]
        random_subset = experiment["random"]
        cube = experiment["cube"]
        score = experiment["fitness"]
        selected_ids = [cube.sensor_ids[index] for index in selected]
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        result_columns = st.columns(4)
        result_columns[0].metric("Selected sensors", f"{len(selected)} / {sensor_count}")
        result_columns[1].metric("TSEM entropy", f"{score:.3f}")
        result_columns[2].metric("GA runtime", f"{experiment['elapsed']:.2f} s")
        coverage = spatial_coverage_score(cube.coordinates, selected)
        result_columns[3].metric("Coverage proxy", f"{coverage:.1%}")

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.subheader("Where the network listens")
            map_frame = pd.DataFrame(
                {
                    "sensor_id": cube.sensor_ids,
                    "x": cube.coordinates[:, 0],
                    "y": cube.coordinates[:, 1],
                    "selection": ["Selected" if index in selected else "Not selected" for index in range(sensor_count)],
                }
            )
            figure = px.scatter(
                map_frame,
                x="x",
                y="y",
                color="selection",
                text="sensor_id",
                color_discrete_map={"Selected": "#087e70", "Not selected": "#b9c7c0"},
                labels={"x": "X coordinate", "y": "Y coordinate", "selection": "Deployment"},
            )
            figure.update_traces(marker={"size": 13, "line": {"width": 1, "color": "white"}}, textposition="top center")
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(chart_layout(figure, 360), width="stretch")
            if cube.synthetic_coordinates:
                st.caption("No complete x/y coordinates were supplied; this map uses a generated layout for visualization only.")
        with right:
            st.subheader("GA search progress")
            history_figure = go.Figure()
            history_figure.add_trace(
                go.Scatter(
                    x=list(range(len(experiment["fitness_history"]))),
                    y=experiment["fitness_history"],
                    mode="lines",
                    line={"color": "#087e70", "width": 3},
                    fill="tozeroy",
                    fillcolor="rgba(8,126,112,0.10)",
                    name="Best entropy",
                )
            )
            history_figure.update_layout(xaxis_title="Generation", yaxis_title="Normalized TSEM entropy")
            st.plotly_chart(chart_layout(history_figure, 360), width="stretch")
            st.markdown("**Chosen subset**")
            st.write("  ·  ".join(selected_ids))
            st.caption(f"Seeded random comparison: {', '.join(cube.sensor_ids[index] for index in random_subset)}")

        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        st.markdown(
            "**Selection rule:** each subset contains exactly the requested number of unique sensors. "
            "Fitness is the mean normalized Shannon entropy of the selected sensors’ pairwise DTW distances, computed separately per variable."
        )
    else:
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Sensor layout")
            try:
                preview_cube = build_sensor_cube(raw_data, [selected_variables[0]] if selected_variables else [variables_available[0]])
                preview = pd.DataFrame(
                    {"sensor_id": preview_cube.sensor_ids, "x": preview_cube.coordinates[:, 0], "y": preview_cube.coordinates[:, 1]}
                )
                figure = px.scatter(preview, x="x", y="y", text="sensor_id", color_discrete_sequence=["#087e70"])
                figure.update_traces(marker={"size": 12}, textposition="top center")
                figure.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(chart_layout(figure, 330), width="stretch")
            except ValueError as error:
                st.info(str(error))
        with right:
            st.subheader("Ready for a run")
            st.markdown("Adjust the subset size and GA budget in the left panel, then run the optimizer.")
            st.markdown("The bundled Intel Lab readings are loaded by default. Upload another CSV or sensor log to switch data sources.")

with tab_evaluation:
    if is_current:
        selected_result = experiment["selected_evaluation"]
        random_result = experiment["random_evaluation"]
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        selected_rmse = selected_result["overall_rmse"]
        random_rmse = random_result["overall_rmse"]
        comparison_columns = st.columns(3)
        comparison_columns[0].metric("TSEM + GA test RMSE", f"{selected_rmse:.3f}")
        comparison_columns[1].metric("Random test RMSE", f"{random_rmse:.3f}")
        delta = 100 * (random_rmse - selected_rmse) / random_rmse if random_rmse > 0 else 0
        comparison_columns[2].metric("Change vs random", f"{delta:+.1f}%")

        left, right = st.columns([0.85, 1.15], gap="large")
        with left:
            st.subheader("Variable-wise reconstruction error")
            variable_rows = []
            for variable in selected_variables:
                variable_rows.append({"Variable": variable, "TSEM + GA": selected_result["variable_rmse"][variable], "Random": random_result["variable_rmse"][variable]})
            error_figure = px.bar(
                pd.DataFrame(variable_rows).melt(id_vars="Variable", var_name="Subset", value_name="RMSE"),
                x="Variable",
                y="RMSE",
                color="Subset",
                barmode="group",
                color_discrete_map={"TSEM + GA": "#087e70", "Random": "#d96f52"},
            )
            st.plotly_chart(chart_layout(error_figure, 330), width="stretch")
        with right:
            st.subheader("Held-out sensor estimates")
            scatter = pd.DataFrame({"Observed": selected_result["actual"], "Predicted": selected_result["predicted"]})
            scatter_figure = px.scatter(scatter, x="Observed", y="Predicted", opacity=0.42, color_discrete_sequence=["#087e70"])
            low = min(scatter["Observed"].min(), scatter["Predicted"].min())
            high = max(scatter["Observed"].max(), scatter["Predicted"].max())
            scatter_figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"color": "#d96f52", "dash": "dash"}, name="Ideal"))
            st.plotly_chart(chart_layout(scatter_figure, 330), width="stretch")
        st.caption("RMSE is reported in each variable’s original units in the bar chart. The overall score is pooled RMSE after scaling each variable by its training-period standard deviation.")
        st.caption("GPR is fitted from the selected sensors’ readings at each test timestamp and predicts the unselected sensor locations. This tests spatial reconstruction, not future forecasting.")
    else:
        st.info("Run the optimizer to see held-out sensor reconstruction and the random-subset comparison.")

with tab_dataset:
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    dataset_left, dataset_right = st.columns([1.15, 0.85], gap="large")
    with dataset_left:
        st.subheader("Current readings")
        st.caption(f"Source: {dataset_name}. Record fields include sensor/mote ID, date/time, temperature, humidity, light, and voltage.")
        st.dataframe(raw_data.head(12), width="stretch", hide_index=True)
    with dataset_right:
        st.subheader("Dataset in this run")
        st.markdown(
            "**Bundled data:** official Intel Berkeley Research Lab readings from 54 Mica2Dot sensor motes collected in 2004. It contains temperature, humidity, light, voltage, date/time, and mote ID. Raw readings are aggregated into 30-minute records for the interactive demo."
            if uploaded is None
            else f"**Uploaded data:** `{dataset_name}`. The app aligns timestamps, averages duplicate sensor/time rows, interpolates missing readings, and samples long datasets to keep DTW interactive."
        )
        st.markdown("**Paper dataset:** Intel Berkeley Research Lab, 54 Mica2Dot sensors, recorded every 31 seconds in 2004. Variables include temperature, humidity, light, and voltage.")
        st.markdown("[Official dataset page](https://db.csail.mit.edu/labdata/labdata.html) · [Readings (.txt.gz)](https://db.csail.mit.edu/labdata/data.txt.gz) · [Coordinates](https://db.csail.mit.edu/labdata/mote_locs.txt)")
        st.download_button(
            "Download processed sensor data",
            data=raw_data.to_csv(index=False).encode(),
            file_name="intel_lab_30min_readings.csv" if uploaded is None else "processed_sensor_readings.csv",
            mime="text/csv",
        )
        st.markdown("**Dataset fields:** the environmental variables are temperature, humidity, light, and voltage. Date/time and sensor/mote ID identify each reading; they are used for alignment and grouping, not optimized as measurement variables.")
