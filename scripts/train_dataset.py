"""Run every method on one processed dataset and dump results/raw/<name>.json.

Usage:  python scripts/train_dataset.py --dataset edgeiiot
"""
import os
import sys
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_io import load_config
from src.data_io import load_processed
from src.partition import partition_clients
from src.fed import run_method
from src.utils import set_seed, save_json


def build_data(name, cfg):
    d = load_processed(name, cfg)
    X, y, key = d["X"], d["y"], d["partition_key"]

    Xtr, Xte, ytr, yte, ktr, _ = train_test_split(
        X, y, key, test_size=cfg["data"]["test_size"],
        random_state=cfg["seed"], stratify=y)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    strat = cfg["fl"]["partition"][name]
    clients = partition_clients(ytr, ktr, strat, cfg["fl"]["n_clients"],
                                cfg["fl"]["dirichlet_alpha"], cfg["seed"])
    return {
        "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
        "clients": clients, "in_dim": X.shape[1],
        "n_classes": int(d["n_classes"]), "task": d["task"],
    }, d, strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    name = args.dataset

    data, meta, strat = build_data(name, cfg)
    cfg["_model"] = cfg["model"][name]
    sizes = [int(len(c)) for c in data["clients"]]
    print(f"\n=== {name} === in_dim={data['in_dim']} classes={data['n_classes']} "
          f"task={data['task']} partition={strat} client_sizes={sizes}")

    results = {"dataset": name, "task": data["task"], "n_classes": data["n_classes"],
               "in_dim": data["in_dim"], "partition": strat,
               "client_sizes": sizes, "class_counts": meta.get("counts", {}),
               "methods": {}}

    for method in cfg["methods"]:
        res = run_method(method, data, cfg)
        results["methods"][method] = res
        line = (f"  {method:14s} acc={res['accuracy']:.4f} "
                f"macroF1={res['macro_f1']:.4f}")
        if "roc_auc" in res:
            line += f" roc_auc={res['roc_auc']:.4f} pr_auc={res.get('pr_auc', float('nan')):.4f}"
        line += f"  time={res['wall_time_s']:.1f}s"
        if "he" in res:
            h = res["he"]
            line += (f"  [{h['he_backend']}] enc={h['enc_ms_per_client_round']:.1f}ms "
                     f"agg={h['agg_ms_per_round']:.1f}ms exp={h['expansion_x']:.1f}x")
        print(line)

    out = os.path.join(cfg["paths"]["results_raw"], f"{name}.json")
    save_json(results, out)
    print(f"  -> saved {out}")


if __name__ == "__main__":
    main()
