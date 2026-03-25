"""
config.py — Shared constants and utility functions
===================================================
Imported by train.py, predict.py, and app.py.
Change DATA_DIR and MODEL_DIR to match your folder structure.
"""

import pickle
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")        # folder containing train_FD00x.txt / test_FD00x.txt
MODEL_DIR = Path("models")   # folder where .pkl model artifacts are saved

# ── Raw data schema ───────────────────────────────────────────────────────────
COLUMN_NAMES = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

# ── Feature engineering knobs ─────────────────────────────────────────────────
RUL_CAP = 125   # piecewise-linear RUL cap (cycles)
WINDOW  = 30    # rolling window size (cycles)

# ── Informative sensors per subset (zero-variance ones excluded) ──────────────
SENSORS = {
    1: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s17","s20","s21"],
    2: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
    3: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
    4: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
}

# ── XGBoost hyper-parameters (tuned) ─────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 1000,
    max_depth        = 4,
    learning_rate    = 0.02,
    subsample        = 0.9,
    colsample_bytree = 0.8,
    reg_alpha        = 0.0,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_feature_columns(sensors: list) -> list:
    """Return ordered feature column list for a given sensor set."""
    return (
        sensors
        + [f"{s}_rm" for s in sensors]   # rolling mean
        + [f"{s}_rs" for s in sensors]   # rolling std
        + ["cycle"]
    )


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int = WINDOW) -> pd.DataFrame:
    """
    Add per-engine rolling mean and std for each informative sensor.
    Must be called AFTER sorting by [unit, cycle].
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    for s in sensors:
        grp = df.groupby("unit")[s]
        df[f"{s}_rm"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{s}_rs"] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))  # std of 1 sample = NaN → 0 (no variation)
    return df


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM'08 asymmetric scoring function.

    d = predicted_RUL - true_RUL
      d < 0  (early prediction) → penalty = exp(-d / 13) - 1
      d >= 0 (late  prediction) → penalty = exp( d / 10) - 1  ← heavier

    Lower is better. Perfect prediction = 0.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(penalties))


def model_path(subset: int) -> Path:
    """Return the .pkl metadata path for a given subset number."""
    return MODEL_DIR / f"rul_model_FD00{subset}.pkl"


# ── Model artifact versioning ─────────────────────────────────────────────────
ARTIFACT_VERSION = 2   # v1: XGBRegressor pickled inline  |  v2: JSON + meta pkl


def load_model_artifact(subset: int) -> dict:
    """
    Load a model artifact for the given subset.

    v1 artifacts (legacy): XGBRegressor is pickled directly inside the .pkl.
    v2 artifacts (current): XGBRegressor saved as native JSON alongside the .pkl.

    Always returns a dict with a live 'model' key ready for inference.
    """
    path = model_path(subset)
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    if artifact.get("artifact_version") == 2:
        model_file = MODEL_DIR / artifact["model_file"]
        model = xgb.XGBRegressor()
        model.load_model(str(model_file))
        artifact["model"] = model

    return artifact
