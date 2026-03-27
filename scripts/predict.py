"""
predict.py — Load a saved model and run RUL predictions
=========================================================
Two modes:

  1. Batch mode (default): predict RUL for every engine in the test file
     and print results alongside ground truth.

     python predict.py --subset 1

  2. Single engine mode: feed a CSV of one engine's sensor history,
     get back its predicted RUL.

     python predict.py --subset 1 --engine_csv my_engine.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_DIR, COLUMN_NAMES, RUL_CAP,
    add_rolling_features, nasa_score, model_path, load_model_artifact,
)


# ── Load model artifact ───────────────────────────────────────────────────────

def load_model(subset: int) -> dict:
    path = model_path(subset)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. "
            f"Run:  python train.py --subset {subset}"
        )
    return load_model_artifact(subset)


# ── Core prediction function ──────────────────────────────────────────────────

def predict_from_sequence(sensor_df: pd.DataFrame, artifact: dict) -> float:
    """
    Predict RUL from an engine's full sensor history (all cycles up to now).

    The function:
      1. Adds rolling features over the engine's history.
      2. Takes the LAST cycle row (most recent snapshot).
      3. Scales and runs the model.

    Parameters
    ----------
    sensor_df : DataFrame with COLUMN_NAMES columns (no RUL column needed).
    artifact  : loaded .pkl dict from train.py.

    Returns
    -------
    float : predicted RUL in cycles, clipped to [0, RUL_CAP].
    """
    sensors  = artifact["sensors"]
    features = artifact["features"]
    scaler   = artifact["scaler"]
    model    = artifact["model"]
    rul_cap  = artifact["rul_cap"]
    window   = artifact["window"]

    df       = add_rolling_features(sensor_df.copy(), sensors, window=window)
    last_row = df.sort_values("cycle").tail(1)

    X    = scaler.transform(last_row[features].values)
    pred = float(model.predict(X)[0])
    return round(float(np.clip(pred, 0, rul_cap)), 1)


# ── Batch prediction on the full test file ────────────────────────────────────

def predict_batch(subset: int):
    """
    Load test_FD00x.txt, predict RUL for every engine,
    compare with ground truth in RUL_FD00x.txt, and print a report.
    """
    artifact = load_model(subset)
    sensors  = artifact["sensors"]
    features = artifact["features"]
    scaler   = artifact["scaler"]
    model    = artifact["model"]
    rul_cap  = artifact["rul_cap"]
    window   = artifact["window"]

    # Load test data and ground-truth RUL
    test = pd.read_csv(DATA_DIR / f"test_FD00{subset}.txt",
                       sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_true_df = pd.read_csv(DATA_DIR / f"RUL_FD00{subset}.txt",
                               sep=r"\s+", header=None, names=["RUL"])

    # Feature engineering
    test = add_rolling_features(test, sensors, window=window)
    test_last = (test.sort_values(["unit", "cycle"])
                     .groupby("unit").last()
                     .reset_index())

    X_test = scaler.transform(test_last[features].values)
    y_pred = model.predict(X_test).clip(0, rul_cap)
    y_true = rul_true_df["RUL"].values

    # Metrics
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    r2    = float(1 - np.sum((y_true - y_pred) ** 2)
                      / np.sum((y_true - np.mean(y_true)) ** 2))
    score = nasa_score(y_true, y_pred)

    print(f"\n{'═' * 62}")
    print(f"  Batch Prediction Report — FD00{subset}")
    print(f"{'─' * 62}")
    print(f"  Engines evaluated : {len(y_true)}")
    print(f"  RMSE              : {rmse:.2f} cycles")
    print(f"  MAE               : {mae:.2f} cycles")
    print(f"  R²                : {r2:.4f}")
    print(f"  NASA Score        : {score:.1f}  (lower = better)")
    print(f"{'═' * 62}")

    # Per-engine table (first 20 engines)
    results = pd.DataFrame({
        "engine_id"  : test_last["unit"].values,
        "true_RUL"   : y_true,
        "pred_RUL"   : y_pred.round(1),
        "error"      : (y_pred - y_true).round(1),
    })
    results["status"] = results["error"].apply(
        lambda e: "LATE ⚠" if e > 10 else ("EARLY" if e < -10 else "OK ✓")
    )

    print(f"\n  Sample predictions (first 20 engines):")
    print(f"  {'EngineID':>10} {'True RUL':>10} {'Pred RUL':>10} {'Error':>8} {'Status':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")
    for _, row in results.head(20).iterrows():
        print(f"  {int(row.engine_id):>10} {int(row.true_RUL):>10} "
              f"{row.pred_RUL:>10.1f} {row.error:>+8.1f} {row.status:>10}")

    late  = (results["error"] > 10).sum()
    early = (results["error"] < -10).sum()
    ok    = len(results) - late - early
    print(f"\n  Summary: {ok} OK ✓  |  {early} early  |  {late} LATE ⚠")
    print(f"  (Late = predicted more RUL than actual — safety risk)\n")

    return results


# ── Single engine prediction from CSV ─────────────────────────────────────────

def predict_single(subset: int, engine_csv: str):
    """
    Predict RUL for a single engine given its sensor history as a CSV.

    CSV format: same columns as the raw data files (no header needed,
    or with header matching COLUMN_NAMES). Must include unit/cycle columns.

    Example CSV (no header):
        1,150,-0.001,0.0002,100.0,518.67,642.0,...
        1,151,-0.001,0.0002,100.0,518.67,641.9,...
    """
    artifact = load_model(subset)

    path = Path(engine_csv)
    if not path.exists():
        raise FileNotFoundError(f"Engine CSV not found: {engine_csv}")

    # Try reading with header first, fall back to no header
    try:
        df = pd.read_csv(path)
        if list(df.columns) != COLUMN_NAMES[:len(df.columns)]:
            raise ValueError
    except (ValueError, KeyError):
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    # Ensure unit column is consistent (treat whole file as one engine)
    df["unit"] = df["unit"].iloc[0]

    required_cols = {"unit", "cycle"} | set(artifact["sensors"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns for FD00{subset}: {sorted(missing)}"
        )

    pred = predict_from_sequence(df, artifact)

    print(f"\n  Engine CSV     : {engine_csv}")
    print(f"  Subset         : FD00{subset}")
    print(f"  Cycles in file : {len(df)}")
    print(f"  Last cycle     : {int(df['cycle'].max())}")
    print(f"\n  ► Predicted RUL : {pred:.1f} cycles\n")
    return pred


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict RUL using a trained model.")
    parser.add_argument("--subset", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Which C-MAPSS subset model to use (1–4).")
    parser.add_argument("--engine_csv", type=str, default=None,
                        help="Path to a single engine's sensor history CSV. "
                             "If omitted, runs batch prediction on the full test file.")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   NASA C-MAPSS — RUL Prediction                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.engine_csv:
        predict_single(args.subset, args.engine_csv)
    else:
        predict_batch(args.subset)


if __name__ == "__main__":
    main()
