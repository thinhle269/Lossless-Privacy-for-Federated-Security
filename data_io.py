"""Read/write the processed datasets produced by scripts/prep_*.py.

Each dataset is stored as `<name>.npz` (X, y, partition_key) plus `<name>.meta.json`
(n_classes, task, target, feature_names, class_names, counts).
"""
import os
import json
import numpy as np
from src.config_io import load_config
from src.utils import ensure_dir


def processed_paths(name, cfg=None):
    cfg = cfg or load_config()
    base = cfg["paths"]["processed"]
    return os.path.join(base, f"{name}.npz"), os.path.join(base, f"{name}.meta.json")


def save_processed(name, X, y, partition_key, meta, cfg=None):
    npz, mp = processed_paths(name, cfg)
    ensure_dir(os.path.dirname(npz))
    np.savez_compressed(
        npz,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        partition_key=np.asarray(partition_key, dtype=np.int64),
    )
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)


def load_processed(name, cfg=None):
    npz, mp = processed_paths(name, cfg)
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"processed '{name}' missing ({npz}). Run scripts/prep_{name}.py first.")
    d = np.load(npz)
    with open(mp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {"X": d["X"], "y": d["y"], "partition_key": d["partition_key"]}
    out.update(meta)
    return out


def is_processed(name, cfg=None):
    npz, mp = processed_paths(name, cfg)
    return os.path.exists(npz) and os.path.exists(mp)
