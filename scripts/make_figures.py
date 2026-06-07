"""Collect results/raw/*.json into publication figures (PNG, 300 dpi)."""
import os
import sys
import glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config_io import load_config
from src.utils import load_json, ensure_dir

sns.set_theme(style="whitegrid", context="paper")
METHOD_ORDER = ["centralized", "local", "fedavg", "he_fedavg", "fedavg_dp"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def main():
    cfg = load_config()
    figdir = cfg["paths"]["figures"]
    ensure_dir(figdir)
    runs = [load_json(p) for p in sorted(glob.glob(os.path.join(cfg["paths"]["results_raw"], "*.json")))]
    if not runs:
        print("[fig] no results; run training first.")
        return
    datasets = [r["dataset"] for r in runs]

    # 1) accuracy comparison (methods grouped per dataset) -------------------
    methods = [m for m in METHOD_ORDER if any(m in r["methods"] for r in runs)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(datasets)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [r["methods"].get(m, {}).get("accuracy", np.nan) for r in runs]
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by method and dataset")
    ax.legend(ncol=len(methods), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _save(fig, os.path.join(figdir, "accuracy_comparison.png"))

    # 2) convergence: FedAvg vs HE-FedAvg per dataset ------------------------
    for r in runs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ["fedavg", "he_fedavg", "fedavg_dp"]:
            hist = r["methods"].get(m, {}).get("history")
            if hist:
                ax.plot(range(1, len(hist) + 1), [h["accuracy"] for h in hist],
                        marker="o", ms=3, label=m)
        for m in ["centralized"]:
            if m in r["methods"]:
                ax.axhline(r["methods"][m]["accuracy"], ls="--", c="gray",
                           label="centralized")
        ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
        ax.set_title(f"Convergence — {r['dataset']}"); ax.legend(fontsize=8)
        _save(fig, os.path.join(figdir, f"convergence_{r['dataset']}.png"))

    # 3) HE overhead: time breakdown + ciphertext expansion ------------------
    he = [(r["dataset"], r["methods"]["he_fedavg"]["he"]) for r in runs
          if "he_fedavg" in r["methods"]]
    if he:
        labels = [d for d, _ in he]
        enc = [h["enc_ms_per_client_round"] for _, h in he]
        agg = [h["agg_ms_per_round"] for _, h in he]
        dec = [h["dec_ms_per_round"] for _, h in he]
        fig, ax = plt.subplots(figsize=(7, 4))
        xi = np.arange(len(labels))
        ax.bar(xi, enc, label="encrypt (client)")
        ax.bar(xi, agg, bottom=enc, label="aggregate (server)")
        ax.bar(xi, dec, bottom=np.array(enc) + np.array(agg), label="decrypt (client)")
        ax.set_xticks(xi); ax.set_xticklabels(labels)
        ax.set_ylabel("ms per round"); ax.set_title("CKKS overhead breakdown")
        ax.legend(fontsize=8)
        _save(fig, os.path.join(figdir, "he_overhead_time.png"))

        fig, ax = plt.subplots(figsize=(7, 4))
        exp = [h["expansion_x"] for _, h in he]
        ax.bar(xi, exp, color="indianred")
        for i, v in enumerate(exp):
            ax.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xi); ax.set_xticklabels(labels)
        ax.set_ylabel("ciphertext / plaintext bytes")
        ax.set_title("Communication expansion under CKKS")
        _save(fig, os.path.join(figdir, "he_comm_expansion.png"))

    # 4) privacy-utility-cost trade-off --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"fedavg": "o", "he_fedavg": "s", "fedavg_dp": "^"}
    for r in runs:
        for m, mk in markers.items():
            mm = r["methods"].get(m)
            if mm and mm.get("comm_upload_MB"):
                ax.scatter(mm["comm_upload_MB"], mm["accuracy"], marker=mk, s=60)
                ax.annotate(f"{r['dataset']}/{m.replace('fedavg','FA')}",
                            (mm["comm_upload_MB"], mm["accuracy"]),
                            fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Upload communication (MB, log)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Privacy–utility–cost: HE matches FedAvg accuracy at higher comm")
    _save(fig, os.path.join(figdir, "privacy_utility_tradeoff.png"))


if __name__ == "__main__":
    main()
