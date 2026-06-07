"""Edge-IIoTset -> processed arrays.

15-class IoT intrusion detection from the ML-ready CSV. We drop string/identifier
columns (IPs, URIs, payloads, timestamps) and the binary label, keep numeric
features, and use `Attack_type` as the 15-class target. Partition key = label
(so a Dirichlet split yields realistic label-skew across edge clients).
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

DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4", "http.request.method", "http.request.full_uri",
    "http.request.version", "tcp.options", "tcp.payload", "tcp.srcport",
    "http.file_data", "http.request.uri.query", "http.referer",
    "dns.qry.name", "mqtt.msg", "mqtt.topic", "Attack_label",
]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    src = cfg["datasets_raw"]["edgeiiot_ml"]
    cap = int(cfg["data"]["edgeiiot"].get("sample_cap", 0))
    print(f"[edgeiiot] reading {src}")
    df = pd.read_csv(src, low_memory=False)
    print(f"[edgeiiot] raw shape {df.shape}")

    label = "Attack_type"
    y_raw = df[label].astype(str).str.strip()
    classes = sorted(y_raw.unique())
    cls_map = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(cls_map).to_numpy()

    X = df.drop(columns=[c for c in DROP_COLS + [label] if c in df.columns],
                errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any()]                       # drop all-NaN cols
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.loc[:, X.std(numeric_only=True) > 0]          # drop constant cols
    feat = X.columns.tolist()
    X = X.to_numpy(dtype=np.float32)

    if cap and len(y) > cap:                            # stratified subsample
        rng = np.random.default_rng(cfg["seed"])
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            k = max(1, int(round(cap * len(idx) / len(y))))
            keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        X, y = X[keep], y[keep]

    meta = {
        "n_classes": len(classes), "task": "multiclass", "target": label,
        "feature_names": feat, "class_names": classes,
        "counts": {classes[i]: int((y == i).sum()) for i in range(len(classes))},
    }
    save_processed("edgeiiot", X, y, y.copy(), meta, cfg)  # key = label for skew
    print(f"[edgeiiot] saved X{X.shape}  classes={len(classes)}")


if __name__ == "__main__":
    main()
