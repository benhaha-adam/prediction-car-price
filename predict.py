"""
=============================================================
  APP.PY — AutoPredict (Streamlit web version)
  Ported from predict.py — no retraining needed.

  Run locally:
    streamlit run app.py

  Deploy:
    → Push to GitHub (include models/ folder)
    → Connect repo on share.streamlit.io
    → Done.
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st.* call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPredict — Vehicle Price Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# PALETTE  (same as predict.py)
# ─────────────────────────────────────────────
_C = dict(
    BG      = "#16181F",
    BG2     = "#1E2029",
    BG3     = "#272B38",
    BG4     = "#2D3242",
    ACCENT  = "#F0A500",
    ACCENT2 = "#FFD166",
    SUCCESS = "#50C896",
    DANGER  = "#E05555",
    TEXT    = "#E8EAF2",
    TEXT2   = "#7A8099",
    BORDER  = "#353A50",
)


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── App background ─────────────────────── */
.stApp {{
    background-color: {_C['BG']};
    color: {_C['TEXT']};
}}

/* ── Sidebar ────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {_C['BG2']};
    border-right: 1px solid {_C['BORDER']};
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {{
    color: {_C['TEXT2']} !important;
    font-size: 0.82rem !important;
}}

/* ── Number / text inputs ───────────────── */
.stNumberInput input,
.stTextInput input {{
    background-color: {_C['BG3']} !important;
    color: {_C['TEXT']} !important;
    border: 1px solid {_C['BORDER']} !important;
    border-radius: 5px !important;
}}

/* ── Select boxes ───────────────────────── */
.stSelectbox > div > div {{
    background-color: {_C['BG3']} !important;
    color: {_C['TEXT']} !important;
    border: 1px solid {_C['BORDER']} !important;
}}

/* ── Slider ─────────────────────────────── */
.stSlider > div > div > div > div {{
    background: {_C['ACCENT']} !important;
}}

/* ── Primary button ─────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {{
    background-color: {_C['ACCENT']} !important;
    color: #15161C !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 0.55rem 1.2rem !important;
    font-size: 0.95rem !important;
    transition: background-color 0.15s ease;
}}
.stButton > button:hover {{
    background-color: {_C['ACCENT2']} !important;
}}

/* ── Download button ────────────────────── */
.stDownloadButton > button {{
    background-color: {_C['BG4']} !important;
    color: {_C['TEXT']} !important;
    border: 1px solid {_C['BORDER']} !important;
    border-radius: 6px !important;
    width: 100% !important;
}}

/* ── Tabs ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {_C['BG2']};
    border-bottom: 1px solid {_C['BORDER']};
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {_C['TEXT2']} !important;
    background-color: transparent !important;
    border-radius: 6px 6px 0 0 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {_C['ACCENT']} !important;
    border-bottom: 2px solid {_C['ACCENT']} !important;
    background-color: {_C['BG3']} !important;
}}

/* ── Info / success banners ─────────────── */
.stAlert {{
    background-color: {_C['BG3']} !important;
    border-radius: 6px !important;
    border-left: 3px solid {_C['ACCENT']} !important;
    color: {_C['TEXT']} !important;
}}

/* ── Metric ─────────────────────────────── */
[data-testid="stMetricValue"] {{
    color: {_C['ACCENT']} !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}}

/* ── Expander ───────────────────────────── */
.streamlit-expanderHeader {{
    background-color: {_C['BG2']} !important;
    color: {_C['TEXT2']} !important;
    border: 1px solid {_C['BORDER']} !important;
    border-radius: 6px !important;
}}

/* ── Dataframe ──────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {_C['BORDER']};
    border-radius: 6px;
}}

/* ── Shared util classes ─────────────────── */
.section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    color: {_C['ACCENT']};
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid {_C['BORDER']};
    margin-bottom: 6px;
}}
.result-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-radius: 7px;
    margin-bottom: 5px;
    background-color: {_C['BG2']};
    border: 1px solid {_C['BORDER']};
}}
.result-row.best {{
    background-color: {_C['BG4']};
    border-color: {_C['ACCENT']};
}}
.result-name {{
    font-size: 0.9rem;
    color: {_C['TEXT']};
    min-width: 160px;
}}
.result-name.best {{
    color: {_C['ACCENT']};
    font-weight: 700;
}}
.result-price {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: {_C['SUCCESS']};
}}
.result-meta {{
    font-size: 0.78rem;
    color: {_C['TEXT2']};
    text-align: right;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1. LOAD METADATA  (cached — runs once)
# ─────────────────────────────────────────────
MODELS_DIR = Path("models/")


@st.cache_resource
def load_meta() -> dict:
    meta_path = MODELS_DIR / "training_metadata.json"
    if not meta_path.exists():
        st.error(
            f"**`{meta_path}` not found.**  "
            "Run `python train.py` first, then restart this app."
        )
        st.stop()
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


meta             = load_meta()
ALL_FEATURES     = meta["all_features"]
NUM_FEATURES     = meta["num_features"]
CAT_FEATURES     = meta["cat_features"]
ORD_FEATURES     = meta["ord_features"]
BEST_MODEL       = meta["best_model"]
AVAILABLE_MODELS = meta["models"]
CAT_VALUES       = meta["cat_values"]
CURRENT_YEAR     = meta["current_year"]
RESULTS          = meta["results"]


# ─────────────────────────────────────────────
# 2. BACKEND  (identical to predict.py)
# ─────────────────────────────────────────────

def safe_div(a: float, b: float, fallback: float = 1.0) -> float:
    """Division sûre — évite ZeroDivisionError quand b == 0."""
    return a / b if b != 0 else fallback


@st.cache_resource
def load_model(name: str):
    """Charge un pipeline joblib depuis models/ — mis en cache."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        st.error(
            f"**`{path}` not found.**  "
            "Run `python train.py` first, then restart this app."
        )
        st.stop()
    return joblib.load(path)


def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    if "year" not in data.columns or "car_age" in data.columns:
        return data
    data     = data.copy()
    data["car_age"] = (CURRENT_YEAR - data["year"]).clip(lower=0)
    age_safe = np.where(data["car_age"] == 0, 1, data["car_age"])
    if "mileage" in data.columns:
        data["mileage_per_year"] = data["mileage"] / age_safe
    if "engine_hp" in data.columns:
        data["hp_per_year"] = data["engine_hp"] / age_safe
    return data


def predict_price(pipe, data: pd.DataFrame) -> np.ndarray:
    data = apply_feature_engineering(data)
    for col in ALL_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
    data   = data[ALL_FEATURES]
    y_log  = pipe.predict(data)
    y_orig = np.clip(np.expm1(y_log), a_min=0, a_max=None)
    return y_orig


def build_input_df(row_dict: dict) -> pd.DataFrame:
    row = {k: v for k, v in row_dict.items() if k in ALL_FEATURES}
    for col in ALL_FEATURES:
        if col not in row:
            row[col] = np.nan
    return pd.DataFrame([row])[ALL_FEATURES]


def collect_row_from_values(
    year, mileage, engine_hp, owner_count, brand_popularity,
    make, fuel_type, drivetrain, body_type, transmission,
    exterior_color, interior_color, seller_type, accident_history,
    condition,
) -> dict:
    car_age          = max(CURRENT_YEAR - int(year), 0)
    mileage_per_year = safe_div(float(mileage), max(car_age, 1))
    hp_per_year      = safe_div(float(engine_hp), max(car_age, 1))
    return dict(
        year             = int(year),
        mileage          = float(mileage),
        engine_hp        = float(engine_hp),
        owner_count      = int(owner_count),
        brand_popularity = float(brand_popularity),
        car_age          = car_age,
        mileage_per_year = mileage_per_year,
        hp_per_year      = hp_per_year,
        make             = str(make),
        fuel_type        = str(fuel_type),
        drivetrain       = str(drivetrain),
        body_type        = str(body_type),
        transmission     = str(transmission),
        exterior_color   = str(exterior_color),
        interior_color   = str(interior_color),
        seller_type      = str(seller_type),
        accident_history = str(accident_history),
        condition        = str(condition),
    )


# ─────────────────────────────────────────────
# 3. GAUGE CHART (Plotly)
# ─────────────────────────────────────────────

def make_gauge(price: float, model_name: str) -> go.Figure:
    MAX_PRICE = 200_000
    ratio = min(max(price, 0) / MAX_PRICE, 1.0)

    # Colour ramp: green → amber → red (same logic as Tkinter canvas)
    if ratio < 0.5:
        r_c = int(80  + 160 * ratio * 2)
        g_c = int(200 -  40 * ratio * 2)
        b_c = 100
    else:
        r_c = 240
        g_c = int(160 - 120 * (ratio - 0.5) * 2)
        b_c = 50
    bar_color = f"#{min(r_c, 255):02x}{min(g_c, 255):02x}{min(b_c, 255):02x}"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        number={
            "prefix": "$",
            "font": {"size": 34, "color": _C["TEXT"], "family": "Segoe UI"},
            "valueformat": ",.0f",
        },
        gauge={
            "axis": {
                "range": [0, MAX_PRICE],
                "tickcolor":  _C["TEXT2"],
                "tickfont":   {"color": _C["TEXT2"], "size": 10},
                "nticks": 6,
            },
            "bar":         {"color": bar_color, "thickness": 0.38},
            "bgcolor":     _C["BG3"],
            "bordercolor": _C["BORDER"],
            "steps":       [{"range": [0, MAX_PRICE], "color": _C["BG3"]}],
            "threshold": {
                "line":      {"color": _C["ACCENT2"], "width": 2},
                "thickness": 0.75,
                "value":     price,
            },
        },
        title={
            "text": (
                f"<b style='color:{_C['TEXT']}'>Best estimate</b>"
                f"<br><span style='font-size:0.85em;color:{_C['TEXT2']}'>"
                f"{model_name}</span>"
            ),
            "font": {"color": _C["TEXT2"], "size": 13},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ─────────────────────────────────────────────
# 4. SIDEBAR FORM
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """Renders all inputs and returns their values + button state."""
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{_C['ACCENT']};margin-bottom:2px'>🚗 AutoPredict</h2>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Trained {meta['trained_at'][:10]}"
            f" · {meta['sample_size']:,} vehicles"
        )
        st.divider()

        # ── Section 1: General ────────────────────
        st.markdown("<div class='section-label'>General</div>",
                    unsafe_allow_html=True)
        year      = st.number_input("Year of manufacture",
                                    min_value=1990,
                                    max_value=CURRENT_YEAR,
                                    value=CURRENT_YEAR - 3,
                                    step=1)
        mileage   = st.number_input("Mileage (km)",
                                    min_value=0,
                                    value=50_000,
                                    step=1_000)
        engine_hp = (
            st.number_input("Engine horsepower",
                            min_value=50, max_value=1_500,
                            value=150, step=10)
            if "engine_hp" in NUM_FEATURES
            else 150
        )
        owner_cnt = st.number_input("Previous owners",
                                    min_value=0, max_value=20,
                                    value=1, step=1)
        brand_pop = st.slider("Brand popularity (0–100)", 0, 100, 60)
        st.divider()

        # ── Section 2: Vehicle identity ───────────
        st.markdown("<div class='section-label'>Vehicle</div>",
                    unsafe_allow_html=True)
        make       = st.selectbox("Make",
                                  CAT_VALUES.get("make", ["Toyota"]))
        body_type  = st.selectbox("Body type",
                                  CAT_VALUES.get("body_type", ["Sedan"]))
        fuel_type  = st.selectbox("Fuel type",
                                  CAT_VALUES.get("fuel_type", ["Gasoline"]))
        drivetrain = st.selectbox("Drivetrain",
                                  CAT_VALUES.get("drivetrain", ["FWD"]))
        trans      = st.selectbox("Transmission",
                                  CAT_VALUES.get("transmission", ["Automatic"]))
        st.divider()

        # ── Section 3: History & condition ────────
        st.markdown("<div class='section-label'>History & Condition</div>",
                    unsafe_allow_html=True)
        accidents = st.selectbox("Accident history",
                                 CAT_VALUES.get("accident_history", ["No", "Yes"]))
        seller    = st.selectbox("Seller type",
                                 CAT_VALUES.get("seller_type", ["Dealer"]))
        condition = (
            st.selectbox("Condition",
                         CAT_VALUES.get("condition",
                                        ["Poor", "Fair", "Good", "Excellent"]))
            if ORD_FEATURES
            else "Good"
        )
        st.divider()

        # ── Section 4: Colours ────────────────────
        st.markdown("<div class='section-label'>Colours</div>",
                    unsafe_allow_html=True)
        ext_col = st.selectbox("Exterior colour",
                               CAT_VALUES.get("exterior_color", ["White"]))
        int_col = st.selectbox("Interior colour",
                               CAT_VALUES.get("interior_color", ["Black"]))
        st.divider()

        # ── Model selector ────────────────────────
        st.markdown("<div class='section-label'>ML Model</div>",
                    unsafe_allow_html=True)
        model_opts   = ["— All models —"] + list(AVAILABLE_MODELS)
        model_choice = st.selectbox(
            "Model to use",
            model_opts,
            index=model_opts.index(BEST_MODEL)
                  if BEST_MODEL in model_opts else 1,
        )

        predict_btn = st.button("🔍 Estimate Price",
                                use_container_width=True)  # buttons keep this param

    return dict(
        year=year, mileage=mileage, engine_hp=engine_hp,
        owner_count=owner_cnt, brand_popularity=brand_pop,
        make=make, fuel_type=fuel_type, drivetrain=drivetrain,
        body_type=body_type, transmission=trans,
        exterior_color=ext_col, interior_color=int_col,
        seller_type=seller, accident_history=accidents,
        condition=condition,
        _model_choice=model_choice,
        _predict=predict_btn,
    )


# ─────────────────────────────────────────────
# 5. RESULT DISPLAY HELPERS
# ─────────────────────────────────────────────

def render_result_row(name: str, price: float) -> None:
    r       = RESULTS[name]
    is_best = (name == BEST_MODEL)
    cls     = "result-row best" if is_best else "result-row"
    name_cls = "result-name best" if is_best else "result-name"
    prefix  = "★ " if is_best else "· "
    st.markdown(
        f"""
        <div class='{cls}'>
          <div class='{name_cls}'>{prefix}{name}</div>
          <div class='result-price'>${price:,.0f}</div>
          <div class='result-meta'>R²&nbsp;{r['R2']:.3f}&nbsp;·&nbsp;MAE&nbsp;${r['MAE']:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 6. SESSION STATE  (keeps results across reruns)
# ─────────────────────────────────────────────
if "pred_results" not in st.session_state:
    st.session_state.pred_results = None
if "pred_input_df" not in st.session_state:
    st.session_state.pred_input_df = None


# ─────────────────────────────────────────────
# 7. MAIN UI
# ─────────────────────────────────────────────

def main() -> None:
    form = render_sidebar()

    # ── Page header ──────────────────────────
    left_hdr, right_hdr = st.columns([3, 2])
    with left_hdr:
        st.markdown(
            f"<h1 style='margin-bottom:0;line-height:1.1'>"
            f"<span style='color:{_C['ACCENT']}'>AUTO</span>"
            f"<span style='color:{_C['TEXT']}'>PREDICT</span>"
            f"</h1>",
            unsafe_allow_html=True,
        )
        st.caption("Vehicle price estimation — no retraining needed")
    with right_hdr:
        r = RESULTS[BEST_MODEL]
        st.markdown(
            f"<div style='text-align:right;color:{_C['TEXT2']};font-size:0.8rem;padding-top:10px'>"
            f"Best model: <b style='color:{_C['ACCENT']}'>{BEST_MODEL}</b> &nbsp;"
            f"R²&nbsp;{r['R2']:.3f} &nbsp;·&nbsp; MAE&nbsp;${r['MAE']:,.0f}<br>"
            f"Trained {meta['trained_at'][:10]}"
            f" on {meta['sample_size']:,} vehicles"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── Tabs ─────────────────────────────────
    tab_single, tab_batch, tab_models = st.tabs([
        "🚗 Single Prediction",
        "📂 Batch CSV",
        "📊 Model Performance",
    ])

    # ══════════════════════════════════════════
    # TAB 1 — SINGLE PREDICTION
    # ══════════════════════════════════════════
    with tab_single:

        # Run prediction when button clicked
        if form["_predict"]:
            row_dict = collect_row_from_values(
                form["year"],       form["mileage"],  form["engine_hp"],
                form["owner_count"], form["brand_popularity"],
                form["make"],       form["fuel_type"], form["drivetrain"],
                form["body_type"],  form["transmission"],
                form["exterior_color"], form["interior_color"],
                form["seller_type"], form["accident_history"],
                form["condition"],
            )
            input_df = build_input_df(row_dict)

            sel          = form["_model_choice"]
            model_names  = (list(AVAILABLE_MODELS)
                            if sel == "— All models —"
                            else [sel])

            results = {}
            with st.spinner("Running predictions…"):
                for name in model_names:
                    pipe           = load_model(name)
                    price          = float(predict_price(pipe, input_df.copy())[0])
                    results[name]  = price

            st.session_state.pred_results  = results
            st.session_state.pred_input_df = input_df

        # Display results (persists across reruns)
        if st.session_state.pred_results:
            results  = st.session_state.pred_results
            input_df = st.session_state.pred_input_df

            # Best price — look for BEST_MODEL first, fallback to first result
            best_price = results.get(BEST_MODEL, next(iter(results.values())))
            best_name  = BEST_MODEL if BEST_MODEL in results else next(iter(results))

            col_gauge, col_table = st.columns([2, 3])

            with col_gauge:
                st.plotly_chart(
                    make_gauge(best_price, best_name),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            with col_table:
                st.markdown(
                    f"<h4 style='color:{_C['TEXT']};margin-bottom:12px'>"
                    f"Results</h4>",
                    unsafe_allow_html=True,
                )
                # Show BEST_MODEL first, then the rest
                ordered = sorted(
                    results.items(),
                    key=lambda x: (x[0] != BEST_MODEL,),
                )
                for name, price in ordered:
                    render_result_row(name, price)

            with st.expander("📋 Input features used for this prediction"):
                st.dataframe(
                    input_df.T.rename(columns={0: "value"}),
                    width="stretch",
                )

        else:
            st.info(
                "👈 Fill in the vehicle details in the sidebar "
                "and click **Estimate Price** to get a prediction."
            )

    # ══════════════════════════════════════════
    # TAB 2 — BATCH CSV
    # ══════════════════════════════════════════
    with tab_batch:
        st.markdown(
            "Upload a CSV file with one vehicle per row. "
            "All available models will run on every row and the results "
            "will be available as a download."
        )

        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="batch_uploader",
        )

        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip().str.lower()
            st.write(f"**{len(df):,} rows loaded** — preview (first 5):")
            st.dataframe(df.head(), width="stretch")

            if st.button("▶ Run batch predictions", key="batch_run"):
                out = df.copy()
                progress = st.progress(0, text="Starting…")

                for i, name in enumerate(AVAILABLE_MODELS):
                    progress.progress(
                        (i + 1) / len(AVAILABLE_MODELS),
                        text=f"Running {name}…",
                    )
                    pipe  = load_model(name)
                    preds = predict_price(pipe, df.copy())
                    out[f"predicted_price_{name}"] = preds.round(2)

                progress.empty()
                st.success(f"✅ Done — {len(AVAILABLE_MODELS)} models × {len(df):,} rows")

                # Summary table
                summary_rows = []
                for name in AVAILABLE_MODELS:
                    col = out[f"predicted_price_{name}"]
                    summary_rows.append({
                        "Model": name,
                        "Min ($)":    f"{col.min():,.0f}",
                        "Median ($)": f"{col.median():,.0f}",
                        "Max ($)":    f"{col.max():,.0f}",
                        "R²":         f"{RESULTS[name]['R2']:.3f}",
                    })
                st.dataframe(
                    pd.DataFrame(summary_rows).set_index("Model"),
                    width="stretch",
                )

                # Download
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download predictions CSV",
                    data=csv_bytes,
                    file_name=f"predictions_output_{ts}.csv",
                    mime="text/csv",
                )

    # ══════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════
    with tab_models:
        st.markdown(f"### Model comparison")
        rows = []
        for name in AVAILABLE_MODELS:
            r = RESULTS[name]
            rows.append({
                "Model":    name,
                "R²":       round(r["R2"], 4),
                "MAE ($)":  int(r["MAE"]),
                "RMSE ($)": int(r.get("RMSE", 0)),
                "Best":     "★" if name == BEST_MODEL else "",
            })
        perf_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(perf_df, width="stretch")

        # R² bar chart
        fig_bar = go.Figure(go.Bar(
            x=[r["Model"] for r in rows],
            y=[r["R²"] for r in rows],
            marker_color=[
                _C["ACCENT"] if r["Best"] == "★" else _C["BG4"]
                for r in rows
            ],
            marker_line_color=_C["BORDER"],
            marker_line_width=1,
            text=[f"{r['R²']:.3f}" for r in rows],
            textposition="outside",
            textfont={"color": _C["TEXT"]},
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=_C["BG2"],
            font={"color": _C["TEXT"]},
            yaxis=dict(
                gridcolor=_C["BORDER"],
                range=[max(0, min(r["R²"] for r in rows) - 0.05), 1.02],
                tickfont={"color": _C["TEXT2"]},
            ),
            xaxis=dict(tickfont={"color": _C["TEXT2"]}),
            title=dict(
                text="R² by model (higher = better)",
                font={"color": _C["TEXT2"], "size": 13},
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            height=320,
        )
        st.plotly_chart(fig_bar, width="stretch")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()