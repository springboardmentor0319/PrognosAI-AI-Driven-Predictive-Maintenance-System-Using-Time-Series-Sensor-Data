"""
train.py — Train XGBoost RUL models for all 4 C-MAPSS subsets
==============================================================
Usage:
    python train.py               # trains all 4 subsets
    python train.py --subset 1    # trains FD001 only

Output:
    models/rul_model_FD001.pkl
    models/rul_model_FD002.pkl
    models/rul_model_FD003.pkl
    models/rul_model_FD004.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    DATA_DIR, MODEL_DIR, COLUMN_NAMES,
    RUL_CAP, WINDOW, SENSORS, XGB_PARAMS,
    get_feature_columns, add_rolling_features, nasa_score, model_path,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(subset: int):
    """Read train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt from DATA_DIR."""
    def read(fname):
        return pd.read_csv(DATA_DIR / fname, sep=r"\s+", header=None, names=COLUMN_NAMES)

    train = read(f"train_FD00{subset}.txt")
    test  = read(f"test_FD00{subset}.txt")
    rul   = pd.read_csv(DATA_DIR / f"RUL_FD00{subset}.txt",
                        sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul


# ── RUL label generation ──────────────────────────────────────────────────────

def compute_rul_labels(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Piecewise-linear RUL cap:
      - Healthy region (true RUL > cap): label = cap  (no useful signal here)
      - Degradation region (true RUL <= cap): label = actual remaining cycles
    Prevents the model from fitting noise in the early stable-operation phase.
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=cap)
    return df.drop(columns="max_cycle")


# ── Training ──────────────────────────────────────────────────────────────────

def train_subset(subset: int) -> dict:
    """
    Full pipeline for one subset:
      load → label → feature engineering → scale → train → evaluate → save
    Returns a dict with all metrics.
    """
    sensors      = SENSORS[subset]
    feature_cols = get_feature_columns(sensors)

    print(f"\n{'─' * 58}")
    print(f"  Training FD00{subset}  |  RUL cap={RUL_CAP}  |  Window={WINDOW}")
    print(f"{'─' * 58}")

    # 1. Load raw data
    train_raw, test_raw, rul_true = load_dataset(subset)
    print(f"  Loaded  →  train: {len(train_raw):,} rows  |  "
          f"test engines: {test_raw['unit'].nunique()}")

    # 2. Generate RUL labels on training data
    train = compute_rul_labels(train_raw)

    # 3. Rolling features on both train and test
    train = add_rolling_features(train, sensors)
    test  = add_rolling_features(test_raw, sensors)

    # 4. Test set: take the LAST observed cycle per engine
    #    RUL_FD00x.txt ground truth corresponds exactly to this snapshot
    test_last = (test.sort_values(["unit", "cycle"])
                     .groupby("unit").last()
                     .reset_index())

    # 5. Scale features (fit on train only — no leakage)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)
    X_test  = scaler.transform(test_last[feature_cols].values)
    y_train = train["RUL"].values
    y_true  = rul_true["RUL"].values

    # 6. Train XGBoost
    print(f"  Training XGBoost ({XGB_PARAMS['n_estimators']} trees, "
          f"depth={XGB_PARAMS['max_depth']}, lr={XGB_PARAMS['learning_rate']}) ...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # 7. Predict and clip to valid range
    y_pred = model.predict(X_test).clip(0, RUL_CAP)

    # 8. Evaluate
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(1 - np.sum((y_true - y_pred) ** 2)
                      / np.sum((y_true - np.mean(y_true)) ** 2))
    score = nasa_score(y_true, y_pred)

    print(f"\n  ── Evaluation Results ──────────────────────────────")
    print(f"  RMSE         : {rmse:.2f} cycles")
    print(f"  MAE          : {mae:.2f} cycles")
    print(f"  R²           : {r2:.4f}")
    print(f"  NASA Score   : {score:.1f}   ← primary metric (lower = better)")

    # 9. Top 5 feature importances
    fi   = pd.Series(model.feature_importances_, index=feature_cols)
    top5 = fi.nlargest(5)
    print(f"\n  Top 5 predictive features:")
    for feat, imp in top5.items():
        print(f"    {feat:<22} importance = {imp:.4f}")

    # 10. Save model artifact (v2: XGBoost model as native JSON + metadata pkl)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json_file = f"rul_model_FD00{subset}.json"
    json_path = MODEL_DIR / json_file
    model.save_model(str(json_path))

    artifact = {
        "artifact_version": 2,
        "algorithm" : "XGBoost",
        "model_file": json_file,       # XGBoost native JSON — cross-version compatible
        "scaler"    : scaler,
        "features"  : feature_cols,
        "sensors"   : sensors,
        "rul_cap"   : RUL_CAP,
        "window"    : WINDOW,
        "subset"    : subset,
        "metrics"   : {"RMSE": rmse, "MAE": mae, "R2": r2, "NASA_Score": score},
    }
    out_path = model_path(subset)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  Model saved  → {json_path}")
    print(f"  Meta saved   → {out_path}")

    return {"subset": f"FD00{subset}", "RMSE": round(rmse, 2),
            "MAE": round(mae, 2), "R2": round(r2, 4), "NASA_Score": round(score, 1)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RUL models on C-MAPSS subsets.")
    parser.add_argument("--subset", type=int, choices=[1, 2, 3, 4],
                        help="Which subset to train (1–4). Omit to train all.")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else [1, 2, 3, 4]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   NASA C-MAPSS — XGBoost RUL Training                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Metric guide")
    print("  NASA Score : primary. Asymmetric — late predictions penalised more")
    print("               than early ones. Lower is better. Perfect model = 0.")
    print("  RMSE       : secondary. Standard ML regression benchmark.")
    print("  R²         : proportion of RUL variance explained (higher = better).")

    all_results = [train_subset(s) for s in subsets]

    # Summary table
    print(f"\n{'═' * 62}")
    print(f"  {'Dataset':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'NASA Score':>12}")
    print(f"{'─' * 62}")
    for r in all_results:
        print(f"  {r['subset']:<10} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} "
              f"{r['R2']:>8.4f} {r['NASA_Score']:>12.1f}")
    print(f"{'═' * 62}")
    print("\nDone. Run predict.py or start app.py to use the trained models.")


if __name__ == "__main__":
    main()
