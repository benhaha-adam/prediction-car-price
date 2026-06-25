# core.py
import warnings
warnings.filterwarnings("ignore")

import json
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path("models/")

meta_path = MODELS_DIR / "training_metadata.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

ALL_FEATURES     = meta["all_features"]
NUM_FEATURES     = meta["num_features"]
CAT_FEATURES     = meta["cat_features"]
ORD_FEATURES     = meta["ord_features"]
BEST_MODEL       = meta["best_model"]
AVAILABLE_MODELS = meta["models"]
CAT_VALUES       = meta["cat_values"]
CURRENT_YEAR     = meta["current_year"]
RESULTS          = meta["results"]


def safe_div(a, b, fallback=1.0):
    return a / b if b != 0 else fallback


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    if "year" not in data.columns or "car_age" in data.columns:
        return data
    data = data.copy()
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
    data = data[ALL_FEATURES]
    y_log  = pipe.predict(data)
    return np.clip(np.expm1(y_log), a_min=0, a_max=None)


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
    condition
) -> dict:
    car_age          = max(CURRENT_YEAR - int(year), 0)
    mileage_per_year = safe_div(float(mileage), max(car_age, 1))
    hp_per_year      = safe_div(float(engine_hp), max(car_age, 1))
    return dict(
        year=int(year), mileage=float(mileage), engine_hp=float(engine_hp),
        owner_count=int(owner_count), brand_popularity=float(brand_popularity),
        car_age=car_age, mileage_per_year=mileage_per_year, hp_per_year=hp_per_year,
        make=str(make), fuel_type=str(fuel_type), drivetrain=str(drivetrain),
        body_type=str(body_type), transmission=str(transmission),
        exterior_color=str(exterior_color), interior_color=str(interior_color),
        seller_type=str(seller_type), accident_history=str(accident_history),
        condition=str(condition),
    )