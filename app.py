# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────
# Page
# ─────────────────────────────────────
st.title("🚗 AutoPredict - Car Price Estimator")

# ─────────────────────────────────────
# Safe model path (IMPORTANT FIX)
# ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

model = joblib.load(MODEL_PATH)

# ─────────────────────────────────────
# Inputs
# ─────────────────────────────────────
year = st.number_input("Year", 1990, 2026, 2018)
mileage = st.number_input("Mileage (km)", 0, 300000, 50000)
hp = st.number_input("Engine HP", 50, 1500, 150)

# ─────────────────────────────────────
# Prediction
# ─────────────────────────────────────
if st.button("Predict"):
    data = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "engine_hp": hp
    }])

    try:
        pred_log = model.predict(data)[0]
        price = np.expm1(pred_log)

        price = max(price, 0)  # safety clamp

        st.success(f"Estimated price: ${price:,.0f}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")