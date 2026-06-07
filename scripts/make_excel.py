"""Collect results/raw/*.json into one Excel workbook (+ CSV copies)."""
import os
import sys
import glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from src.config_io import load_config
from src.utils import load_json, ensure_dir

METRIC_COLS = ["accuracy", "macro_f1", "weighted_f1", "roc_auc", "pr_auc"]


def main():
    cfg = load_config()
    raw = sorted(glob.glob(os.path.join(cfg["paths"]["results_raw"], "*.json")))
    if not raw:
        print("[excel] no results found; run training first.")
        return
    runs = [load_json(p) for p in raw]

    summary, overhead, convergence = [], [], []
    for r in runs:
        for method, m in r["methods"].items():
            row = {"dataset": r["dataset"], "task": r["task"], "method": method}
            for c in METRIC_COLS:
                row[c] = m.get(c)
            row["wall_time_s"] = m.get("wall_time_s")
            row["comm_upload_MB"] = m.get("comm_upload_MB")
            row["n_params"] = m.get("n_params")
            summary.append(row)
            if "he" in m:
                h = dict(m["he"]); h["dataset"] = r["dataset"]
                h["accuracy"] = m.get("accuracy")
                overhead.append(h)
            for i, hh in enumerate(m.get("history", []), 1):
                convergence.append({"dataset": r["dataset"], "method": method,
                                    "round": i, "accuracy": hh["accuracy"],
                                    "macro_f1": hh["macro_f1"]})

    sm = pd.DataFrame(summary)
    ov = pd.DataFrame(overhead)
    cv = pd.DataFrame(convergence)

    ensure_dir(cfg["paths"]["excel"])
    xlsx = os.path.join(cfg["paths"]["excel"], "HE-FedSec-results.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        sm.to_excel(xw, sheet_name="summary", index=False)
        if not ov.empty:
            ov.to_excel(xw, sheet_name="he_overhead", index=False)
        if not cv.empty:
            cv.to_excel(xw, sheet_name="convergence", index=False)
        # accuracy pivot for quick reading / paper table
        piv = sm.pivot_table(index="method", columns="dataset", values="accuracy")
        piv.to_excel(xw, sheet_name="accuracy_pivot")

    sm.to_csv(os.path.join(cfg["paths"]["excel"], "summary.csv"), index=False)
    if not ov.empty:
        ov.to_csv(os.path.join(cfg["paths"]["excel"], "he_overhead.csv"), index=False)
    print(f"[excel] wrote {xlsx}  ({len(sm)} rows, {len(runs)} datasets)")


if __name__ == "__main__":
    main()
