"""
generate_demo.py — Extract 5 engines per subset from training data.
Creates demo_FD001.txt through demo_FD004.txt in scripts/data/.

Run from the scripts/ directory:
    python generate_demo.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
N_ENGINES = 5

COLUMN_NAMES = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

for i in range(1, 5):
    src = DATA_DIR / f"train_FD00{i}.txt"
    dst = DATA_DIR / f"demo_FD00{i}.txt"

    df = pd.read_csv(src, sep=r"\s+", header=None, names=COLUMN_NAMES)
    engines = sorted(df["unit"].unique())[:N_ENGINES]
    demo = df[df["unit"].isin(engines)].copy()
    demo.to_csv(dst, sep=" ", header=False, index=False, float_format="%.4f")

    cycles = demo.groupby("unit")["cycle"].max().to_dict()
    size_kb = dst.stat().st_size // 1024
    print(f"FD00{i}: engines {engines}, max cycles per engine: {cycles}")
    print(f"  -> {dst}  ({size_kb} KB, {len(demo)} rows)")
