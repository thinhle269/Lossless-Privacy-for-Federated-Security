"""RBA login dataset -> processed arrays (chunked; the raw file is ~9 GB).

Binary detection of malicious logins. Default target `is_attack_ip` (~9% positive);
`is_account_takeover` is also supported (extremely rare -> all positives kept).
We scan up to `scan_rows`, keep all positives (capped) and a downsampled set of
negatives, then engineer compact numeric features. Partition key = country, so the
`natural` strategy gives a geo-distributed federation.
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

TARGETS = {"is_attack_ip": "Is Attack IP", "is_account_takeover": "Is Account Takeover"}
USECOLS = ["Login Timestamp", "Round-Trip Time [ms]", "Country", "ASN",
           "Browser Name and Version", "OS Name and Version", "Device Type",
           "Login Successful", "Is Attack IP", "Is Account Takeover"]


def _as_bool(s):
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def _freq_encode(col):
    vc = col.value_counts(normalize=True)
    return col.map(vc).fillna(0.0).to_numpy(np.float32)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    rc = cfg["data"]["rba"]
    src = cfg["datasets_raw"]["rba"]
    target_key = rc.get("target", "is_attack_ip")
    target_col = TARGETS[target_key]
    scan_rows = int(rc.get("scan_rows", 1_200_000))
    chunksize = int(rc.get("chunksize", 200_000))
    pos_cap = int(rc.get("pos_cap", 40_000))
    neg_cap = int(rc.get("neg_cap", 120_000))
    rng = np.random.default_rng(cfg["seed"])
    keep_prob = min(1.0, neg_cap / max(1, scan_rows))

    print(f"[rba] target={target_col}  scan<= {scan_rows:,}  src={src}")
    pos_parts, neg_parts, scanned, npos, nneg = [], [], 0, 0, 0
    for chunk in pd.read_csv(src, usecols=USECOLS, chunksize=chunksize, low_memory=False):
        scanned += len(chunk)
        tgt = _as_bool(chunk[target_col])
        pos = chunk[tgt]
        neg = chunk[~tgt]
        if npos < pos_cap and len(pos):
            take = pos.iloc[:pos_cap - npos]
            pos_parts.append(take); npos += len(take)
        if nneg < neg_cap and len(neg):
            mask = rng.random(len(neg)) < keep_prob
            take = neg[mask].iloc[:neg_cap - nneg]
            neg_parts.append(take); nneg += len(take)
        if scanned >= scan_rows or (npos >= pos_cap and nneg >= neg_cap):
            break
        if scanned % (chunksize * 5) == 0:
            print(f"[rba]   scanned {scanned:,}  pos={npos:,} neg={nneg:,}")

    df = pd.concat(pos_parts + neg_parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    y = _as_bool(df[target_col]).astype(np.int64)
    print(f"[rba] assembled {len(df):,} rows  positives={int(y.sum()):,} "
          f"({100*y.mean():.2f}%)")

    # --- feature engineering -------------------------------------------------
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    feats = {
        "rtt_log": np.log1p(pd.to_numeric(df["Round-Trip Time [ms]"],
                                          errors="coerce").fillna(0.0)).to_numpy(np.float32),
        "login_ok": _as_bool(df["Login Successful"]).astype(np.float32),
        "hour": (ts.dt.hour.fillna(0) / 23.0).to_numpy(np.float32),
        "dow": (ts.dt.dayofweek.fillna(0) / 6.0).to_numpy(np.float32),
        "country_freq": _freq_encode(df["Country"].astype(str)),
        "asn_freq": _freq_encode(df["ASN"].astype(str)),
        "browser_freq": _freq_encode(df["Browser Name and Version"].astype(str)),
        "os_freq": _freq_encode(df["OS Name and Version"].astype(str)),
    }
    if target_key == "is_account_takeover":          # attack-IP is a valid signal here
        feats["is_attack_ip"] = _as_bool(df["Is Attack IP"]).astype(np.float32)

    feat_df = pd.DataFrame(feats)
    dev = pd.get_dummies(df["Device Type"].astype(str).fillna("na"),
                         prefix="dev").astype(np.float32)
    feat_df = pd.concat([feat_df, dev.reset_index(drop=True)], axis=1)

    feat = feat_df.columns.tolist()
    X = feat_df.to_numpy(np.float32)
    key = pd.factorize(df["Country"].astype(str))[0].astype(np.int64)  # geo partition

    meta = {
        "n_classes": 2, "task": "binary", "target": target_col,
        "feature_names": feat, "class_names": ["benign", "malicious"],
        "counts": {"benign": int((y == 0).sum()), "malicious": int((y == 1).sum())},
        "n_countries": int(len(np.unique(key))),
    }
    save_processed("rba", X, y, key, meta, cfg)
    print(f"[rba] saved X{X.shape}  pos_rate={y.mean():.3f}  countries={meta['n_countries']}")


if __name__ == "__main__":
    main()
