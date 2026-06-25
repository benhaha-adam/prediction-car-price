# app.py
import streamlit as st  
import joblib
import pandas as pd
import numpy as np

st.title("🚗 AutoPredict - Car Price Estimator")

model = joblib.load("./models/random_forest.joblib")

year = st.number_input("Year", 1990, 2026, 2018)
mileage = st.number_input("Mileage", 0, 300000, 50000)
hp = st.number_input("Engine HP", 50, 1500, 150)

if st.button("Predict"):
    data = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "engine_hp": hp
    }])

    pred = model.predict(data)[0]
    price = np.expm1(pred)

    st.success(f"Estimated price: ${price:,.0f}")