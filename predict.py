import streamlit as st
import numpy as np
import json
import threading
from pathlib import Path
import joblib
import pandas as pd

# ── Reuse all your existing logic ──────────────────────────
from predict import (
    meta, ALL_FEATURES, NUM_FEATURES, CAT_FEATURES, ORD_FEATURES,
    BEST_MODEL, AVAILABLE_MODELS, CAT_VALUES, CURRENT_YEAR, RESULTS,
    load_model, predict_price, build_input_df, collect_row_from_values,
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPredict",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AutoPredict — Estimation de prix véhicule")
st.caption(
    f"Entraîné le {meta['trained_at'][:10]}  ·  "
    f"{meta['sample_size']:,} véhicules  ·  "
    f"Meilleur modèle : **{BEST_MODEL}** (R²={RESULTS[BEST_MODEL]['R2']:.3f})"
)

# ── Sidebar = your form ────────────────────────────────────
st.sidebar.header("Caractéristiques du véhicule")

year       = st.sidebar.number_input("Année de fabrication", 1990, CURRENT_YEAR, CURRENT_YEAR - 3)
mileage    = st.sidebar.number_input("Kilométrage (km)", 0, 500_000, 50_000, step=1000)
engine_hp  = st.sidebar.number_input("Puissance moteur (ch)", 50, 1500, 150) if "engine_hp" in NUM_FEATURES else 150
owner_count = st.sidebar.number_input("Nb de propriétaires", 0, 20, 1)
brand_pop  = st.sidebar.slider("Popularité marque (0–100)", 0, 100, 60)

make         = st.sidebar.selectbox("Marque",             CAT_VALUES.get("make", ["Toyota"]))
body_type    = st.sidebar.selectbox("Carrosserie",        CAT_VALUES.get("body_type", ["Sedan"]))
fuel_type    = st.sidebar.selectbox("Carburant",          CAT_VALUES.get("fuel_type", ["Gasoline"]))
drivetrain   = st.sidebar.selectbox("Traction",           CAT_VALUES.get("drivetrain", ["FWD"]))
transmission = st.sidebar.selectbox("Transmission",       CAT_VALUES.get("transmission", ["Automatic"]))
accident     = st.sidebar.selectbox("Accidents",          CAT_VALUES.get("accident_history", ["No", "Yes"]))
seller_type  = st.sidebar.selectbox("Type de vendeur",    CAT_VALUES.get("seller_type", ["Dealer"]))
ext_color    = st.sidebar.selectbox("Couleur extérieure", CAT_VALUES.get("exterior_color", ["White"]))
int_color    = st.sidebar.selectbox("Couleur intérieure", CAT_VALUES.get("interior_color", ["Black"]))
condition    = st.sidebar.selectbox("Condition",          CAT_VALUES.get("condition", ["Poor", "Fair", "Good", "Excellent"])) if ORD_FEATURES else "Good"

model_choice = st.sidebar.selectbox(
    "Modèle à utiliser",
    ["— Tous —"] + AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(BEST_MODEL) + 1
)

predict_btn = st.sidebar.button("✨ Estimer le prix", type="primary", use_container_width=True)

# ── Results area ───────────────────────────────────────────
if predict_btn:
    row_dict = collect_row_from_values(
        year, mileage, engine_hp, owner_count, brand_pop,
        make, fuel_type, drivetrain, body_type, transmission,
        ext_color, int_color, seller_type, accident, condition,
    )
    input_df = build_input_df(row_dict)
    names = AVAILABLE_MODELS if model_choice == "— Tous —" else [model_choice]

    rows = []
    with st.spinner("Calcul en cours…"):
        for name in names:
            pipe  = load_model(name)
            price = predict_price(pipe, input_df.copy())[0]
            r     = RESULTS[name]
            rows.append({
                "Modèle":       ("⭐ " if name == BEST_MODEL else "") + name,
                "Prix estimé":  f"${price:,.0f}",
                "R²":           f"{r['R2']:.3f}",
                "MAE":          f"${r['MAE']:,.0f}",
                "_price":       price,
                "_best":        name == BEST_MODEL,
            })

    best_row = next((r for r in rows if r["_best"]), rows[0])
    best_price = best_row["_price"]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Prix estimé (meilleur modèle)", f"${best_price:,.0f}")
    col2.metric("Modèle", BEST_MODEL)
    col3.metric("R²", f"{RESULTS[BEST_MODEL]['R2']:.3f}")

    st.divider()

    display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ── Batch CSV ──────────────────────────────────────────────
st.divider()
st.subheader("📂 Batch — Importer un CSV")
uploaded = st.file_uploader("Choisir un fichier CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip().str.lower()
    st.write(f"{len(df):,} lignes chargées")

    if st.button("Lancer le batch sur tous les modèles"):
        out = df.copy()
        with st.spinner("Prédiction batch…"):
            for name in AVAILABLE_MODELS:
                pipe  = load_model(name)
                preds = predict_price(pipe, df.copy())
                out[f"predicted_price_{name}"] = preds.round(2)
        st.success("Batch terminé !")
        st.dataframe(out, use_container_width=True)
        csv_bytes = out.to_csv(index=False).encode()
        st.download_button("⬇ Télécharger les résultats", csv_bytes,
                           "predictions_output.csv", "text/csv")