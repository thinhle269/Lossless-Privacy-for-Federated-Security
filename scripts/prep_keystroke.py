"""CMU keystroke dynamics -> processed arrays.

51-class subject identification from 31 timing features (hold / down-down /
up-down latencies). Partition key = sessionIndex, so the `natural` strategy puts
different typing sessions on different clients (temporal/feature skew, all
subjects present everywhere) — a genuine cross-session federation.
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from src.config_io import load_config
from src.data_io import save_processed
from src.utils import set_seed

NON_FEATURES = ["subject", "sessionIndex", "rep"]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    src = cfg["datasets_raw"]["keystroke"]
    print(f"[keystroke] reading {src}")
    df = pd.read_csv(src)
    print(f"[keystroke] raw shape {df.shape}")

    subjects = sorted(df["subject"].unique())
    cls_map = {s: i for i, s in enumerate(subjects)}
    y = df["subject"].map(cls_map).to_numpy()
    key = df["sessionIndex"].astype(int).to_numpy()

    feat = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

    meta = {
        "n_classes": len(subjects), "task": "multiclass", "target": "subject",
        "feature_names": feat, "class_names": [str(s) for s in subjects],
        "counts": {str(subjects[i]): int((y == i).sum()) for i in range(len(subjects))},
    }
    save_processed("keystroke", X, y, key, meta, cfg)
    print(f"[keystroke] saved X{X.shape}  subjects={len(subjects)}  sessions={len(np.unique(key))}")


if __name__ == "__main__":
    main()
