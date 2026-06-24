"""
=============================================================
  TRAIN.PY — Entraînement & sauvegarde des modèles
  Dataset : Metawave Automotive Price Prediction
  Branch  : feature/model-persistence
=============================================================
  Usage :
    python train.py
    python train.py --sample 500000   # taille d'échantillon
    python train.py --full            # 100% des données
    python train.py --output models/  # dossier de sortie
=============================================================
  Sorties :
    models/preprocessor.joblib        ← pipeline de prétraitement
    models/<nom_modele>.joblib         ← un fichier par modèle
    models/training_metadata.json      ← métriques + feature names
    figures/fig1_exploration.png
    figures/fig2_nouvelles_features.png
    figures/fig3_comparaison_modeles.png
    figures/fig4_predictions.png
    figures/fig5_feature_importance.png
    figures/fig6_correlation.png
=============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys, json, argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost non installé (pip install xgboost)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Entraîne et sauvegarde les modèles.")
parser.add_argument("--sample",  type=int,   default=300_000, help="Taille d'échantillon (0 = tout)")
parser.add_argument("--full",    action="store_true",         help="Utiliser 100%% des données")
parser.add_argument("--data",    type=str,   default="csv/vehicle_price_prediction.csv")
parser.add_argument("--output",  type=str,   default="models/")
parser.add_argument("--figures", type=str,   default="figures/")
args = parser.parse_args()

DATA_PATH    = Path(args.data)
MODELS_DIR   = Path(args.output)
FIGURES_DIR  = Path(args.figures)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE  = 0 if args.full else args.sample
CURRENT_YEAR = datetime.now().year
PALETTE      = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0"]

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.15)
plt.rcParams.update({"figure.dpi": 140, "axes.titleweight": "bold"})

print("=" * 66)
print("  TRAIN.PY — Entraînement des modèles")
print("=" * 66)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
if not DATA_PATH.exists():
    print(f"\nERREUR : fichier introuvable : {DATA_PATH}")
    print("  Téléchargez le dataset :")
    print("  kaggle datasets download -d metawave/vehicle-price-prediction --unzip -p csv/")
    sys.exit(1)

print(f"\nChargement : {DATA_PATH}")
df_full = pd.read_csv(DATA_PATH)
total_rows = len(df_full)

if SAMPLE_SIZE and total_rows > SAMPLE_SIZE:
    df = df_full.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"  Échantillon : {SAMPLE_SIZE:,} / {total_rows:,} lignes ({SAMPLE_SIZE/total_rows*100:.1f}%)")
else:
    df = df_full.copy()
    print(f"  Chargement complet : {total_rows:,} lignes")
del df_full

df.columns = df.columns.str.strip().str.lower()
print(f"  Colonnes : {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. NETTOYAGE
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print("Nettoyage...")

n_before = len(df)
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]
print(f"  Prix invalides supprimés : {n_before - len(df)}")

if "mileage" in df.columns:
    n_bad = (df["mileage"] < 0).sum()
    df.loc[df["mileage"] < 0, "mileage"] = np.nan
    if n_bad:
        print(f"  Kilométrages négatifs corrigés : {n_bad}")

if "year" in df.columns:
    df = df[(df["year"] >= 1900) & (df["year"] <= CURRENT_YEAR)]

if "engine_hp" in df.columns:
    df = df[(df["engine_hp"] > 0) & (df["engine_hp"] < 2000)]

print(f"  Dataset final : {len(df):,} lignes")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print("Feature engineering...")

df["car_age"]         = (CURRENT_YEAR - df["year"]).clip(lower=0)
df["mileage_per_year"] = df["mileage"] / df["car_age"].replace(0, 1)
if "engine_hp" in df.columns:
    df["hp_per_year"] = df["engine_hp"] / df["car_age"].replace(0, 1)

df["price_segment"] = pd.qcut(
    df["price"], q=4,
    labels=["Budget", "Intermédiaire", "Premium", "Luxe"]
)
print("  car_age, mileage_per_year, hp_per_year créés")

# ─────────────────────────────────────────────
# 4. FEATURES & CIBLE
# ─────────────────────────────────────────────
TARGET       = "price"
NUM_FEATURES = [f for f in [
    "year", "mileage", "engine_hp",
    "owner_count", "brand_popularity",
    "car_age", "mileage_per_year", "hp_per_year",
] if f in df.columns]

CAT_FEATURES = [f for f in [
    "make", "fuel_type", "drivetrain",
    "body_type", "transmission",
    "exterior_color", "interior_color",
    "seller_type", "accident_history",
] if f in df.columns]

ORD_FEATURES     = ["condition"] if "condition" in df.columns else []
CONDITION_ORDER  = [["Poor", "Fair", "Good", "Excellent"]]
ALL_FEATURES     = NUM_FEATURES + CAT_FEATURES + ORD_FEATURES

print(f"  Numériques ({len(NUM_FEATURES)})  : {NUM_FEATURES}")
print(f"  Catégorielles ({len(CAT_FEATURES)}): {CAT_FEATURES}")
print(f"  Ordinal ({len(ORD_FEATURES)})      : {ORD_FEATURES}")

X      = df[ALL_FEATURES]
y_log  = np.log1p(df[TARGET])

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.20, random_state=42
)
y_test_orig = np.expm1(y_test_log)
print(f"\nSplit : {len(X_train):,} train · {len(X_test):,} test")

# ─────────────────────────────────────────────
# 5. PRÉPROCESSEUR
# ─────────────────────────────────────────────
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
])
transformers = [
    ("num", num_transformer, NUM_FEATURES),
    ("cat", cat_transformer, CAT_FEATURES),
]
if ORD_FEATURES:
    ord_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord",     OrdinalEncoder(
            categories=CONDITION_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    transformers.append(("ord", ord_transformer, ORD_FEATURES))

preprocessor = ColumnTransformer(transformers)

# ─────────────────────────────────────────────
# 6. MODÈLES
# ─────────────────────────────────────────────
models = {
    "ridge":             Ridge(alpha=10),
    "decision_tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
    "random_forest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                               random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.05, random_state=42),
}
if HAS_XGB:
    models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=6,
                                     learning_rate=0.05, random_state=42,
                                     n_jobs=-1, verbosity=0)

# ─────────────────────────────────────────────
# 7. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
results     = {}
predictions = {}
pipelines   = {}

col_w = max(len(n) for n in models) + 2
print(f"\n{'─'*70}")
print(f"  {'Modèle':<{col_w}} {'MAE':>10} {'RMSE':>10} {'R²':>7}  {'CV R²':>9}")
print(f"{'─'*70}")

for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train_log)

    y_pred_log  = pipe.predict(X_test)
    y_pred_orig = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

    mae   = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse  = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2    = r2_score(y_test_orig, y_pred_orig)
    cv_r2 = cross_val_score(pipe, X_train, y_train_log, cv=3, scoring="r2").mean()

    results[name]     = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv_r2}
    predictions[name] = y_pred_orig
    pipelines[name]   = pipe

    print(f"  {name:<{col_w}} {mae:>9,.0f}$ {rmse:>9,.0f}$ {r2:>6.3f}  {cv_r2:>8.3f}")

print(f"{'─'*70}")
best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\nMeilleur modèle : {best_model_name}  (R² = {results[best_model_name]['R2']:.3f})")

# ─────────────────────────────────────────────
# 8. SAUVEGARDE DES MODÈLES
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Sauvegarde dans : {MODELS_DIR}/")

for name, pipe in pipelines.items():
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipe, path, compress=3)
    size_mb = path.stat().st_size / 1_048_576
    print(f"  {path.name:<35} ({size_mb:.1f} MB)")

# Métadonnées : tout ce dont predict.py a besoin
metadata = {
    "trained_at":       datetime.now().isoformat(),
    "sample_size":      len(df),
    "current_year":     CURRENT_YEAR,
    "target":           TARGET,
    "num_features":     NUM_FEATURES,
    "cat_features":     CAT_FEATURES,
    "ord_features":     ORD_FEATURES,
    "all_features":     ALL_FEATURES,
    "condition_order":  CONDITION_ORDER,
    "best_model":       best_model_name,
    "models":           list(models.keys()),
    "results":          results,
    # Valeurs de référence pour la saisie interactive
    "cat_values": {
        col: sorted(df[col].dropna().unique().tolist()[:30])
        for col in CAT_FEATURES + ORD_FEATURES
        if col in df.columns
    },
}

meta_path = MODELS_DIR / "training_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  {meta_path.name:<35} (métadonnées)")

# ─────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Génération des figures dans : {FIGURES_DIR}/")

# FIG 1 : Distribution prix + condition
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Exploration — Dataset Metawave", fontsize=15)
ax = axes[0]
ax.hist(df[TARGET], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
ax.axvline(df[TARGET].median(), color=PALETTE[3], lw=2, linestyle="--",
           label=f"Médiane : ${df[TARGET].median():,.0f}")
ax.set_title("Distribution des prix")
ax.set_xlabel("Prix ($)"); ax.set_ylabel("Nb véhicules")
ax.legend(); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax = axes[1]
if "condition" in df.columns:
    conds = [c for c in ["Poor","Fair","Good","Excellent"] if c in df["condition"].unique()]
    sns.boxplot(data=df, x="condition", y=TARGET, order=conds, ax=ax,
                palette=sns.color_palette("Blues", len(conds)))
    ax.set_title("Prix par condition"); ax.set_xlabel("Condition"); ax.set_ylabel("Prix ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_exploration.png", bbox_inches="tight")
plt.close()

# FIG 2 : Nouvelles features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Impact des nouvelles features sur le prix", fontsize=15)
for (ax, col, title) in [
    (axes[0][0], "accident_history", "Prix médian / historique accidents"),
    (axes[0][1], "drivetrain",       "Prix médian / type de traction"),
    (axes[1][1], "seller_type",      "Prix médian / type de vendeur"),
]:
    if col in df.columns:
        gdata = df.groupby(col)[TARGET].median().sort_values(ascending=False)
        bars = ax.bar(gdata.index, gdata.values,
                      color=PALETTE[:len(gdata)], edgecolor="white", width=0.55)
        for b, v in zip(bars, gdata.values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    f"${v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title); ax.set_ylabel("Prix médian ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax = axes[1][0]
if "owner_count" in df.columns:
    own = df.groupby("owner_count")[TARGET].median().reset_index()
    ax.plot(own["owner_count"], own[TARGET],
            marker="o", color=PALETTE[0], lw=2, markersize=7)
    ax.set_title("Prix médian / nb propriétaires")
    ax.set_xlabel("Nb propriétaires"); ax.set_ylabel("Prix médian ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_nouvelles_features.png", bbox_inches="tight")
plt.close()

# FIG 3 : Comparaison modèles
n_m = len(results)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Comparaison des {n_m} modèles ML", fontsize=15)
for i, (label, key) in enumerate(zip(["MAE ($)","RMSE ($)","R²"], ["MAE","RMSE","R2"])):
    ax = axes[i]
    vals = [results[m][key] for m in results]
    bars = ax.bar(range(n_m), vals, color=(PALETTE*3)[:n_m],
                  edgecolor="white", width=0.6, alpha=0.9)
    ax.set_title(label); ax.set_ylabel(label)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m.replace("_","\n") for m in results], fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.015,
                f"{v:,.0f}" if key != "R2" else f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    bi = (vals.index(max(vals)) if key=="R2" else vals.index(min(vals)))
    bars[bi].set_edgecolor(PALETTE[3]); bars[bi].set_linewidth(2.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_comparaison.png", bbox_inches="tight")
plt.close()

# FIG 4 : Réel vs prédit + résidus
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Prédictions — {best_model_name}", fontsize=14)
y_pred_best = predictions[best_model_name]
lim = (0, max(y_test_orig.max(), y_pred_best.max()) * 1.05)
ax = axes[0]
ax.scatter(y_test_orig, y_pred_best, alpha=0.3, s=15, color=PALETTE[0], edgecolors="none")
ax.plot(lim, lim, "--", color=PALETTE[3], lw=2, label="Parfait")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title("Prix réel vs prédit"); ax.set_xlabel("Réel ($)"); ax.set_ylabel("Prédit ($)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.legend()
ax = axes[1]
res = y_test_orig - y_pred_best
ax.scatter(y_pred_best, res, alpha=0.3, s=15, color=PALETTE[2], edgecolors="none")
ax.axhline(0, color=PALETTE[3], lw=2, linestyle="--")
ax.set_title("Résidus"); ax.set_xlabel("Prédit ($)"); ax.set_ylabel("Réel − Prédit ($)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_predictions.png", bbox_inches="tight")
plt.close()

# FIG 5 : Feature importance
best_pipe_obj = pipelines[best_model_name]
best_est      = best_pipe_obj.named_steps["model"]
try:
    ohe_feats     = (best_pipe_obj.named_steps["prep"]
                                  .named_transformers_["cat"]
                                  .named_steps["ohe"]
                                  .get_feature_names_out(CAT_FEATURES))
    feat_names_all = NUM_FEATURES + list(ohe_feats) + ORD_FEATURES
except Exception:
    feat_names_all = [f"f{i}" for i in range(500)]

if hasattr(best_est, "feature_importances_"):
    imps = best_est.feature_importances_
    fi_df = (pd.DataFrame({"feature": feat_names_all[:len(imps)], "importance": imps})
               .sort_values("importance", ascending=True).tail(20))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi_df["feature"], fi_df["importance"],
                   color=sns.color_palette("Blues_d", len(fi_df)), edgecolor="white")
    ax.set_title(f"Importance — {best_model_name} (Top 20)", fontsize=14, pad=12)
    ax.set_xlabel("Importance (Gini/gain)")
    for b, v in zip(bars, fi_df["importance"]):
        ax.text(v+0.0005, b.get_y()+b.get_height()/2, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_importance.png", bbox_inches="tight")
    plt.close()

# FIG 6 : Corrélation
fig, ax = plt.subplots(figsize=(11, 9))
num_corr = [c for c in NUM_FEATURES + [TARGET] if c in df.columns]
corr = df[num_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Corrélations numériques + features engineered", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_correlation.png", bbox_inches="tight")
plt.close()

print("  6 figures générées")

# ─────────────────────────────────────────────
# 10. RÉSUMÉ FINAL
# ─────────────────────────────────────────────
print(f"\n{'='*66}")
print("  ENTRAÎNEMENT TERMINÉ")
print(f"{'='*66}")
print(f"\n  Modèles sauvegardés dans : {MODELS_DIR}/")
for name in pipelines:
    r = results[name]
    star = " <-- meilleur" if name == best_model_name else ""
    print(f"    {name}.joblib   R²={r['R2']:.3f}  MAE=${r['MAE']:,.0f}{star}")
print(f"\n  Pour prédire un nouveau véhicule :")
print(f"    python predict.py")
print(f"    python predict.py --model {best_model_name}")
print(f"{'='*66}\n")