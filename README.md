# Vehicle Price Prediction

## Overview

This repository contains a vehicle price prediction pipeline built on the Metawave automotive dataset. It includes training, model persistence, and prediction scripts for both interactive and batch prediction workflows.

## Files

- `training-car-dataset.py`: trains regression models, evaluates performance, saves preprocessing and model artifacts to `models/`, and exports diagnostic plots to `figures/`.
- `prediction-car.py`: loads pretrained models from `models/` and predicts prices either interactively or from a CSV file.
- `csv/vehicle_price_prediction.csv`: expected dataset file used for training and prediction.
- `models/`: saved model pipelines and metadata.
- `figures/`: generated charts and visualizations.
- `unicode_report.txt`: diagnostic report for Unicode encoding issues in the repository.

## Requirements

Recommended Python packages:

```bash
pip install scikit-learn pandas matplotlib seaborn joblib numpy
```

Optional packages for extended functionality:

```bash
pip install xgboost shap
```

## Dataset

Download the dataset from Kaggle:

```bash
kaggle datasets download -d metawave/vehicle-price-prediction --unzip -p csv/
```

Then ensure the file is available at:

```
csv/vehicle_price_prediction.csv
```

## Training

Train models and save outputs with:

```bash
python training-car-dataset.py
```

Optional arguments:

- `--sample 500000` : use a sample size for training
- `--full` : train on the full dataset
- `--data csv/vehicle_price_prediction.csv` : custom dataset path
- `--output models/` : output folder for saved models
- `--figures figures/` : output folder for plots

## Prediction

Run interactive prediction:

```bash
python prediction-car.py
```

Predict with a specific saved model:

```bash
python prediction-car.py --model random_forest
```

Predict using all saved models:

```bash
python prediction-car.py --all
```

Predict from a CSV file in batch mode:

```bash
python prediction-car.py --csv input.csv
```

Use a custom models directory:

```bash
python prediction-car.py --models-dir custom_models/
```

## Outputs

- `models/`: saved `joblib` model files and `training_metadata.json`
- `figures/`: visualizations such as feature importance, prediction comparisons, and correlation heatmaps
- `predictions_output_<timestamp>.csv`: batch prediction results

## Notes

- `training-car-dataset.py` performs preprocessing, feature engineering, and model training.
- `prediction-car.py` expects prepared metadata in `models/training_metadata.json`.
- The dataset contains vehicle features such as year, mileage, engine horsepower, condition, seller type, and more.
