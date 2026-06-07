"""Run ONCE to configure the whole project.

    

run `python run_all.py`.
"""
import os
import sys
import argparse
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(ROOT)          # datasets live one level up
RAW = {
    "edgeiiot_ml": "ML-EdgeIIoT-dataset.csv",
    "keystroke": "DSL-StrongPasswordData.csv",
    "rba": "rba-dataset.csv",
}

PROFILES = {
    "quick": dict(rounds=15, n_clients=4, edge_cap=80_000,
                  rba=dict(target="is_attack_ip", scan_rows=1_200_000, chunksize=200_000,
                           pos_cap=40_000, neg_cap=120_000)),
    "full": dict(rounds=40, n_clients=6, edge_cap=0,
                 rba=dict(target="is_attack_ip", scan_rows=4_000_000, chunksize=200_000,
                          pos_cap=80_000, neg_cap=240_000)),
}


def detect_he():
    try:
        import tenseal  # noqa: F401
        return "tenseal"
    except Exception:
        return "emulated"


def selftest_ckks(backend):
    if backend != "tenseal":
        print("[config] TenSEAL not available -> HE overhead will be ESTIMATED (emulated).")
        return
    import numpy as np, tenseal as ts
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    a, b = np.random.randn(4096), np.random.randn(4096)
    ea, eb = ts.ckks_vector(ctx, a.tolist()), ts.ckks_vector(ctx, b.tolist())
    dec = np.array((ea * 0.5 + eb * 0.5).decrypt())
    err = float(np.max(np.abs(dec - (a * 0.5 + b * 0.5))))
    print(f"[config] CKKS self-test OK  max_abs_err={err:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg_path = os.path.join(ROOT, "config.yaml")
    if os.path.exists(cfg_path) and not args.force:
        print(f"[config] {cfg_path} already exists (use --force to overwrite).")
    P = PROFILES[args.profile]

    # resolve raw dataset paths
    datasets_raw, missing = {}, []
    for key, fname in RAW.items():
        path = os.path.join(DATA_DIR, fname)
        datasets_raw[key] = path.replace("\\", "/")
        if not os.path.exists(path):
            missing.append(fname)

    paths = {k: os.path.join(ROOT, v).replace("\\", "/") for k, v in {
        "processed": "data_processed", "results_raw": "results/raw",
        "excel": "results/excel", "figures": "results/figures"}.items()}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    backend = detect_he()
    cfg = {
        "project_root": ROOT.replace("\\", "/"),
        "profile": args.profile,
        "seed": 42,
        "datasets_raw": datasets_raw,
        "paths": paths,
        "he_backend": backend,
        "he": {"poly_modulus_degree": 8192,
               "coeff_mod_bit_sizes": [60, 40, 40, 60], "scale_bits": 40},
        "fl": {"rounds": P["rounds"], "local_epochs": 5, "lr": 0.01,
               "batch_size": 128, "n_clients": P["n_clients"],
               "partition": {"edgeiiot": "dirichlet", "keystroke": "natural",
                             "rba": "dirichlet"},
               "dirichlet_alpha": 0.5},
        "methods": ["centralized", "local", "fedavg", "he_fedavg", "fedavg_dp"],
        "dp": {"clip": 4.0, "sigma": 0.02},
        "model": {"edgeiiot": {"hidden": [128, 64], "dropout": 0.1},
                  "keystroke": {"hidden": [128, 64], "dropout": 0.1},
                  "rba": {"hidden": [64, 32], "dropout": 0.1}},
        "data": {"test_size": 0.2,
                 "edgeiiot": {"sample_cap": P["edge_cap"]},
                 "keystroke": {},
                 "rba": P["rba"]},
        "datasets_order": ["edgeiiot", "keystroke", "rba"],
    }

    if not (os.path.exists(cfg_path) and not args.force):
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[config] wrote {cfg_path}  (profile={args.profile})")

    selftest_ckks(backend)

    print("\n[config] datasets:")
    for key, path in datasets_raw.items():
        tag = "OK " if os.path.exists(path) else "MISSING"
        print(f"   {tag} {key:12s} {path}")
    if missing:
        print(f"[config] WARNING missing raw files: {missing}")
        print("         Place them next to this project or edit datasets_raw in config.yaml.")
    print(f"[config] HE backend: {backend}")
    print(f"[config] profile={args.profile}  rounds={P['rounds']}  clients={P['n_clients']}")
    print("\nNext:  python run_all.py")


if __name__ == "__main__":
    main()
