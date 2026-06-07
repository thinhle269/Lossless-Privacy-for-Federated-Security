"""Run the full pipeline in order: preprocess -> train -> excel -> figures.

    python run_all.py                  # everything
    python run_all.py --only edgeiiot  # one dataset end-to-end
    python run_all.py --skip-prep      # reuse processed data
    python run_all.py --force-prep     # re-preprocess even if cached
"""
import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.config_io import load_config
from src.data_io import is_processed

PY = sys.executable


def step(title, cmd):
    print(f"\n{'='*70}\n>> {title}\n{'='*70}")
    t0 = time.perf_counter()
    r = subprocess.run([PY] + cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"!! step failed ({title}) rc={r.returncode} after {dt:.1f}s")
        sys.exit(r.returncode)
    print(f"-- done ({title}) in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single dataset name")
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--force-prep", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    datasets = [args.only] if args.only else cfg["datasets_order"]
    t_start = time.perf_counter()

    # 1) preprocess
    if not args.skip_prep:
        for name in datasets:
            if is_processed(name, cfg) and not args.force_prep:
                print(f"[run_all] processed '{name}' exists -> skip prep "
                      f"(use --force-prep to redo)")
                continue
            step(f"preprocess {name}", [os.path.join("scripts", f"prep_{name}.py")])

    # 2) train every method per dataset
    for name in datasets:
        step(f"train {name}", [os.path.join("scripts", "train_dataset.py"),
                               "--dataset", name])

    # 3) aggregate -> excel + figures
    step("export excel", [os.path.join("scripts", "make_excel.py")])
    step("export figures", [os.path.join("scripts", "make_figures.py")])

    dt = time.perf_counter() - t_start
    print(f"\n{'#'*70}\n# PIPELINE COMPLETE in {dt/60:.1f} min")
    print(f"# excel   : {cfg['paths']['excel']}")
    print(f"# figures : {cfg['paths']['figures']}")
    print(f"# raw json: {cfg['paths']['results_raw']}\n{'#'*70}")


if __name__ == "__main__":
    main()
