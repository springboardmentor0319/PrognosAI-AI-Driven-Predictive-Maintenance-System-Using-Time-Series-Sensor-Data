"""
bulk_predict.py — Load test CSVs and send predictions to the API
=================================================================
Reads test_FD00x.txt, takes the last cycle per engine, calls
POST /predict/batch in chunks, then optionally submits true RUL
via POST /ground-truth so Dashboard 3 accuracy panels populate.

Usage:
    python bulk_predict.py --subset 1
    python bulk_predict.py --subset 1 --with-ground-truth
    python bulk_predict.py --subset all --with-ground-truth
    python bulk_predict.py --subset all --with-ground-truth --batch-size 20

Requirements:
    pip install requests pandas
"""

import argparse
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(".")               # folder with test/RUL txt files
BASE_URL   = "http://127.0.0.1:8000"
BATCH_SIZE = 10

COLUMN_NAMES = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

SENSORS = {
    1: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s17","s20","s21"],
    2: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
    3: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
    4: ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_test_last(subset: int) -> pd.DataFrame:
    """Load test file — keep only the LAST cycle per engine."""
    df = pd.read_csv(
        DATA_DIR / f"test_FD00{subset}.txt",
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    return (
        df.sort_values(["unit", "cycle"])
          .groupby("unit").last()
          .reset_index()
    )


def load_rul(subset: int) -> list:
    """Load ground-truth RUL values from RUL_FD00x.txt."""
    df = pd.read_csv(
        DATA_DIR / f"RUL_FD00{subset}.txt",
        sep=r"\s+", header=None, names=["RUL"]
    )
    return df["RUL"].tolist()


def row_to_payload(row: pd.Series, subset: int) -> dict:
    """Convert a DataFrame row into a SensorSnapshot API payload."""
    payload = {
        "subset":    subset,
        "engine_id": int(row["unit"]),
        "cycle":     int(row["cycle"]),
        "setting_1": float(row["setting_1"]),
        "setting_2": float(row["setting_2"]),
        "setting_3": float(row["setting_3"]),
    }
    for s in SENSORS[subset]:
        payload[s] = float(row[s])
    return payload


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── NASA score ────────────────────────────────────────────────────────────────

def nasa_score(y_true, y_pred):
    d = np.array(y_pred, dtype=float) - np.array(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))


# ── Core ──────────────────────────────────────────────────────────────────────

def run_subset(subset: int, with_ground_truth: bool, batch_size: int):

    print(f"\n{'═'*60}")
    print(f"  Processing FD00{subset}")
    print(f"{'═'*60}")

    # ── 1. Health check ───────────────────────────────────────────────────────
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
    except Exception as e:
        print(f"  ✗  Cannot reach API at {BASE_URL}")
        print(f"     Is uvicorn running?  Error: {e}")
        return

    if f"FD00{subset}" not in health.get("models_loaded", []):
        print(f"  ✗  Model FD00{subset} is not loaded in the API.")
        print(f"     Run:  python train.py --subset {subset}")
        return

    db_status = health.get("database", "unknown")
    print(f"  ✓  API reachable | DB: {db_status} | "
          f"Models loaded: {health['models_loaded']}")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    test_file = DATA_DIR / f"test_FD00{subset}.txt"
    rul_file  = DATA_DIR / f"RUL_FD00{subset}.txt"

    if not test_file.exists():
        print(f"  ✗  {test_file} not found. Check DATA_DIR at top of script.")
        return

    test_df  = load_test_last(subset)
    rul_true = load_rul(subset) if with_ground_truth else None
    n        = len(test_df)

    print(f"  Engines in test file : {n}")
    if rul_true:
        print(f"  Ground-truth loaded  : {len(rul_true)} values")

    # ── 3. Build payloads ─────────────────────────────────────────────────────
    payloads = [row_to_payload(test_df.iloc[i], subset) for i in range(n)]

    # ── 4. Send batch predictions ─────────────────────────────────────────────
    prediction_ids = []
    predicted_ruls = []
    batch_errors   = 0
    t_start        = time.time()

    print(f"\n  Sending {n} engines in batches of {batch_size}...")
    print(f"  {'Batch':<8} {'Sent':>6} {'Total':>7} {'Status'}")
    print(f"  {'─'*5:<8} {'─'*5:>6} {'─'*6:>7} {'─'*10}")

    for batch_num, batch in enumerate(chunks(payloads, batch_size), 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/predict/batch",
                json={"instances": batch},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            for pred in result["predictions"]:
                prediction_ids.append(pred.get("prediction_id"))
                predicted_ruls.append(pred["predicted_rul"])

            print(f"  {batch_num:<8} {len(batch):>6} {len(predicted_ruls):>7}   ✓ OK")

        except requests.HTTPError as e:
            batch_errors += 1
            print(f"  {batch_num:<8} {len(batch):>6} {len(predicted_ruls):>7}   ✗ HTTP {e.response.status_code}: {e.response.text[:80]}")
        except Exception as e:
            batch_errors += 1
            print(f"  {batch_num:<8} {len(batch):>6} {len(predicted_ruls):>7}   ✗ {e}")

        time.sleep(0.05)  # small pause — avoid hammering the API

    elapsed = time.time() - t_start
    print(f"\n  Predictions sent  : {len(predicted_ruls)} / {n}")
    print(f"  Batch errors      : {batch_errors}")
    print(f"  Time taken        : {elapsed:.1f}s")

    if not predicted_ruls:
        print("  No predictions received — stopping.")
        return

    # ── 5. Local accuracy summary ─────────────────────────────────────────────
    if rul_true and len(predicted_ruls) == len(rul_true):
        y_true = np.array(rul_true[:len(predicted_ruls)])
        y_pred = np.array(predicted_ruls)
        rmse   = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        mae    = float(np.mean(np.abs(y_true - y_pred)))
        r2     = float(1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2))
        score  = nasa_score(y_true, y_pred)
        late   = int(np.sum((y_pred - y_true) > 10))
        early  = int(np.sum((y_pred - y_true) < -10))
        on_time= len(y_true) - late - early

        print(f"\n  ── Accuracy Summary (local) ────────────────────────")
        print(f"  RMSE         : {rmse:.2f} cycles")
        print(f"  MAE          : {mae:.2f} cycles")
        print(f"  R²           : {r2:.4f}")
        print(f"  NASA Score   : {score:.1f}  (lower = better)")
        print(f"  On time (±10): {on_time}  |  Early: {early}  |  Late ⚠: {late}")

    # ── 6. Submit ground truth to API → populates prediction_errors table ─────
    if with_ground_truth and rul_true:
        valid_pairs = [
            (pid, true)
            for pid, true in zip(prediction_ids, rul_true)
            if pid is not None
        ]

        if not valid_pairs:
            print("\n  ⚠  No prediction_ids returned — ground truth skipped.")
            print("     This happens when the DB is unavailable.")
        else:
            print(f"\n  Submitting {len(valid_pairs)} ground-truth values to API...")
            gt_ok     = 0
            gt_errors = 0

            for pred_id, true_rul in valid_pairs:
                try:
                    r = requests.post(
                        f"{BASE_URL}/ground-truth",
                        json={"prediction_id": pred_id, "true_rul": float(true_rul)},
                        timeout=10,
                    )
                    r.raise_for_status()
                    gt_ok += 1
                except Exception as e:
                    gt_errors += 1
                    if gt_errors <= 3:   # only print first 3 to avoid spam
                        print(f"    ✗ prediction_id={pred_id}: {e}")

                # small progress indicator every 20
                if gt_ok % 20 == 0:
                    print(f"    submitted {gt_ok}/{len(valid_pairs)}...", end="\r")

            print(f"  Ground truth submitted : {gt_ok} ✓  |  errors: {gt_errors}")
            print(f"  → Dashboard 3 accuracy panels will now populate in Grafana.")

    # ── 7. Per-engine sample table ────────────────────────────────────────────
    print(f"\n  Sample predictions (first 15 engines):")
    print(f"  {'Engine':>8} {'True RUL':>10} {'Pred RUL':>10} "
          f"{'Error':>8} {'Status':>10}")
    print(f"  {'─'*7:>8} {'─'*8:>10} {'─'*8:>10} {'─'*6:>8} {'─'*8:>10}")

    for i in range(min(15, len(predicted_ruls))):
        pred = predicted_ruls[i]
        true = rul_true[i] if rul_true else "—"
        if rul_true:
            err    = pred - true
            status = "LATE ⚠" if err > 10 else ("EARLY" if err < -10 else "OK ✓")
            print(f"  {i+1:>8} {true:>10} {pred:>10.1f} {err:>+8.1f} {status:>10}")
        else:
            print(f"  {i+1:>8} {'—':>10} {pred:>10.1f} {'—':>8} {'—':>10}")

    print(f"\n  ✅  FD00{subset} complete — data is now in PostgreSQL.")
    print(f"     Open Grafana to see the dashboards update.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description="Send C-MAPSS test set predictions to the RUL API."
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["1","2","3","4","all"],
        help="Which subset to run: 1 / 2 / 3 / 4 / all"
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        default=False,
        help="Also submit true RUL values via POST /ground-truth "
             "(populates Model Accuracy dashboard)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Engines per batch request (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help=f"API base URL (default: {BASE_URL})"
    )
    args = parser.parse_args()

    # Allow overriding base URL from CLI
    BASE_URL = args.base_url.rstrip("/")

    subsets = [1, 2, 3, 4] if args.subset == "all" else [int(args.subset)]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PrognosAI — Bulk Prediction Runner                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  API         : {BASE_URL}")
    print(f"  Subsets     : {[f'FD00{s}' for s in subsets]}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Ground truth: {'YES — Dashboard 3 will populate' if args.with_ground_truth else 'NO  — run with --with-ground-truth to enable'}")

    overall_start = time.time()
    for s in subsets:
        run_subset(s, args.with_ground_truth, args.batch_size)

    total = time.time() - overall_start
    print(f"\n{'═'*60}")
    print(f"  All done in {total:.1f}s")
    print(f"  Open Grafana → Fleet Overview to see your data.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
