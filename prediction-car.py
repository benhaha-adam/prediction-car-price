"""
=============================================================
  PREDICT.PY — Prédiction sans réentraînement
  Branch : feature/model-persistence
==============================================p===============
  Prérequis : avoir exécuté train.py au moins une fois.

  Usages :
    python predict.py
      → saisie interactive, utilise le meilleur modèle sauvegardé

    python predict.py --model random_forest
      → utilise spécifiquement random_forest.joblib

    python predict.py --all
      → affiche les prédictions de TOUS les modèles sauvegardés

    python predict.py --csv mon_fichier.csv
      → mode batch : prédit les prix pour chaque ligne du CSV
        et exporte les résultats dans predictions_output.csv

    python predict.py --models-dir path/to/models/
      → dossier de modèles personnalisé
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys, json, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Charge les modèles et prédit sans réentraînement.")
parser.add_argument("--model",      type=str,  default=None,     help="Nom du modèle (ex: random_forest)")
parser.add_argument("--all",        action="store_true",          help="Utiliser tous les modèles sauvegardés")
parser.add_argument("--csv",        type=str,  default=None,     help="Fichier CSV à prédire (mode batch)")
parser.add_argument("--models-dir", type=str,  default="models/",help="Dossier contenant les .joblib")
args = parser.parse_args()

MODELS_DIR = Path(args.models_dir)


# ─────────────────────────────────────────────
# 1. CHARGEMENT DES MÉTADONNÉES
# ─────────────────────────────────────────────
meta_path = MODELS_DIR / "training_metadata.json"

if not meta_path.exists():
    print(f"\nERREUR : {meta_path} introuvable.")
    print("  Lancez d'abord : python train.py")
    sys.exit(1)

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

ALL_FEATURES    = meta["all_features"]
NUM_FEATURES    = meta["num_features"]
CAT_FEATURES    = meta["cat_features"]
ORD_FEATURES    = meta["ord_features"]
BEST_MODEL      = meta["best_model"]
AVAILABLE_MODELS = meta["models"]
CAT_VALUES      = meta["cat_values"]
CURRENT_YEAR    = meta["current_year"]
RESULTS         = meta["results"]

print("=" * 66)
print("  PREDICT.PY — Prédiction avec modèles pré-entraînés")
print("=" * 66)
print(f"\n  Entraîné le   : {meta['trained_at'][:19]}")
print(f"  Taille dataset : {meta['sample_size']:,} lignes")
print(f"  Meilleur modèle: {BEST_MODEL}  (R² = {RESULTS[BEST_MODEL]['R2']:.3f})")
print(f"\n  Modèles disponibles :")
for m in AVAILABLE_MODELS:
    r = RESULTS[m]
    star = " <-- meilleur" if m == BEST_MODEL else ""
    print(f"    {m:<22}  R²={r['R2']:.3f}  MAE=${r['MAE']:,.0f}{star}")


# ─────────────────────────────────────────────
# 2. CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────
def load_model(name: str):
    """Charge un pipeline joblib depuis le dossier models/."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"\nERREUR : {path} introuvable. Relancez train.py.")
        sys.exit(1)
    pipe = joblib.load(path)
    print(f"  Modèle chargé  : {path.name}")
    return pipe


def predict_price(pipe, data: pd.DataFrame) -> np.ndarray:
    """
    Prédit les prix à partir d'un DataFrame avec les colonnes ALL_FEATURES.
    Retourne un array de prix en $ (jamais négatifs).
    """
    # Feature engineering si les colonnes source sont présentes mais pas les engineered
    if "year" in data.columns and "car_age" not in data.columns:
        data = data.copy()
        data["car_age"]          = (CURRENT_YEAR - data["year"]).clip(lower=0)
        data["mileage_per_year"] = data["mileage"] / data["car_age"].replace(0, 1)
        if "engine_hp" in data.columns:
            data["hp_per_year"]  = data["engine_hp"] / data["car_age"].replace(0, 1)

    # Aligner les colonnes exactement comme lors de l'entraînement
    for col in ALL_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
    data = data[ALL_FEATURES]

    y_pred_log  = pipe.predict(data)
    y_pred_orig = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)
    return y_pred_orig


# ─────────────────────────────────────────────
# 3A. MODE BATCH (--csv)
# ─────────────────────────────────────────────
if args.csv:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\nERREUR : fichier introuvable : {csv_path}")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"Mode batch : {csv_path}")
    batch_df = pd.read_csv(csv_path)
    batch_df.columns = batch_df.columns.str.strip().str.lower()
    print(f"  {len(batch_df):,} lignes chargées")

    # Sélection des modèles à utiliser
    if args.all:
        model_names = AVAILABLE_MODELS
    elif args.model:
        model_names = [args.model]
    else:
        model_names = [BEST_MODEL]

    output_df = batch_df.copy()
    for name in model_names:
        print(f"\n  Prédiction avec {name}...")
        pipe = load_model(name)
        preds = predict_price(pipe, batch_df.copy())
        col_name = f"predicted_price_{name}"
        output_df[col_name] = preds.round(2)
        print(f"    Min : ${preds.min():,.0f}  |  Médiane : ${np.median(preds):,.0f}  |  Max : ${preds.max():,.0f}")

    out_path = csv_path.parent / f"predictions_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_df.to_csv(out_path, index=False)
    print(f"\n  Résultats exportés : {out_path}")
    sys.exit(0)


# ─────────────────────────────────────────────
# 3B. MODE INTERACTIF
# ─────────────────────────────────────────────
def ask_int(prompt, min_v=None, max_v=None):
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
        except ValueError:
            print("  Veuillez saisir un entier."); continue
        if min_v is not None and v < min_v:
            print(f"  Minimum : {min_v}"); continue
        if max_v is not None and v > max_v:
            print(f"  Maximum : {max_v}"); continue
        return v


def ask_float(prompt, min_v=None):
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            print("  Veuillez saisir un nombre."); continue
        if min_v is not None and v < min_v:
            print(f"  Minimum : {min_v}"); continue
        return v


def ask_choice(prompt, choices):
    cl = {c.lower(): c for c in choices}
    while True:
        print(f"  Options : {', '.join(choices)}")
        raw = input(prompt).strip()
        if raw.lower() in cl:
            return cl[raw.lower()]
        print("  Valeur invalide — réessayez.")


print(f"\n{'═'*66}")
print("  Saisie du véhicule à évaluer")
print(f"{'═'*66}")

year_in   = ask_int("Année de fabrication : ", 1990, CURRENT_YEAR)
mileage_in = ask_int("Kilométrage (km) : ", 0)

engine_hp_in = 150
if "engine_hp" in NUM_FEATURES:
    engine_hp_in = ask_int("Puissance moteur (ch) : ", 50, 1500)

owner_cnt  = ask_int("Nb de propriétaires précédents : ", 0, 20)
brand_pop  = ask_float("Popularité de la marque (0–100) : ", 0)

# Catégorielles : utilise les valeurs vues à l'entraînement
make_in    = ask_choice("Marque (make) : ",         CAT_VALUES.get("make", ["Toyota"]))
fuel_in    = ask_choice("Type de carburant : ",     CAT_VALUES.get("fuel_type", ["Gasoline"]))
drive_in   = ask_choice("Traction (drivetrain) : ", CAT_VALUES.get("drivetrain", ["FWD"]))
body_in    = ask_choice("Carrosserie (body_type) : ",CAT_VALUES.get("body_type", ["Sedan"]))
trans_in   = ask_choice("Transmission : ",          CAT_VALUES.get("transmission", ["Automatic"]))
acc_in     = ask_choice("Historique accidents : ",  CAT_VALUES.get("accident_history", ["No"]))
sell_in    = ask_choice("Type de vendeur : ",       CAT_VALUES.get("seller_type", ["Dealer"]))
ext_col_in = ask_choice("Couleur extérieure : ",    CAT_VALUES.get("exterior_color", ["White"]))
int_col_in = ask_choice("Couleur intérieure : ",    CAT_VALUES.get("interior_color", ["Black"]))
cond_in    = "Good"
if ORD_FEATURES:
    cond_in = ask_choice("Condition du véhicule : ", CAT_VALUES.get("condition", ["Poor","Fair","Good","Excellent"]))

# Feature engineering (identique à train.py)
car_age_in      = max(CURRENT_YEAR - year_in, 0)
mileage_yr_in   = mileage_in / max(car_age_in, 1)
hp_yr_in        = engine_hp_in / max(car_age_in, 1)

row = {
    "year":             year_in,
    "mileage":          mileage_in,
    "engine_hp":        engine_hp_in,
    "owner_count":      owner_cnt,
    "brand_popularity": brand_pop,
    "car_age":          car_age_in,
    "mileage_per_year": mileage_yr_in,
    "hp_per_year":      hp_yr_in,
    "make":             make_in,
    "fuel_type":        fuel_in,
    "drivetrain":       drive_in,
    "body_type":        body_in,
    "transmission":     trans_in,
    "exterior_color":   ext_col_in,
    "interior_color":   int_col_in,
    "seller_type":      sell_in,
    "accident_history": acc_in,
    "condition":        cond_in,
}
row   = {k: v for k, v in row.items() if k in ALL_FEATURES}
input_df = pd.DataFrame([row])[ALL_FEATURES]

print(f"\n{'─'*50}")
print("Caractéristiques saisies :")
for col, val in input_df.iloc[0].items():
    print(f"  {col:<25}: {val}")

# ─────────────────────────────────────────────
# 4. PRÉDICTIONS
# ─────────────────────────────────────────────
print(f"\n{'-'*70}")
print(f"  {'Model':<22} {'Estimated Price':>16}  {'R2 (test)':>10}  {'MAE':>11}")
print(f"{'-'*70}")

if args.model:
    if args.model not in AVAILABLE_MODELS:
        print(f"\nERREUR : modèle '{args.model}' inconnu. Disponibles : {AVAILABLE_MODELS}")
        sys.exit(1)
    model_names = [args.model]
elif args.all:
    model_names = AVAILABLE_MODELS
else:
    model_names = AVAILABLE_MODELS  # Show all models by default

for name in model_names:
    pipe        = load_model(name)
    pred        = predict_price(pipe, input_df.copy())[0]
    r           = RESULTS[name]
    star        = " [BEST]" if name ==BEST_MODEL else ""
    print(f"  {name:<22} ${pred:>11,.2f}   R²={r['R2']:.3f}   MAE=${r['MAE']:>9,.0f}{star}")

print(f"{'─'*60}")
print(f"\n  Entraînement effectué le : {meta['trained_at'][:19]}")
print(f"  Aucun réentraînement effectué — modèles chargés depuis {MODELS_DIR}/")
print(f"\n{'═'*66}\n")