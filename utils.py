"""Small shared helpers: seeding, IO, timing."""
import os
import json
import time
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    """`with Timer() as t: ...` then read t.dt (seconds)."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self._t0
        return False
