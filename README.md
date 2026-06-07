[README.md](https://github.com/user-attachments/files/28678413/README.md)
# Lossless-Privacy-for-Federated-Security# HE-FedSec

**Lossless Privacy for Federated Security** — Federated Learning with CKKS Homomorphic
Encryption, evaluated across three complementary security domains:

| Key | Dataset | Task | Natural clients |
|-----|---------|------|-----------------|
| `edgeiiot` | Edge-IIoTset (`ML-EdgeIIoT-dataset.csv`) | 15-class IoT intrusion detection | label-skew (Dirichlet) |
| `keystroke` | CMU `DSL-StrongPasswordData.csv` | 51-class keystroke-dynamics identification | by typing session |
| `rba` | RBA `rba-dataset.csv` | binary malicious-login / account-takeover detection | by country |

Methods compared per dataset: **Centralized** (upper bound) · **Local-only** (lower bound) ·
**FedAvg** (plaintext baseline) · **HE-FedAvg** (proposed, CKKS secure aggregation) ·
**FedAvg+DP** (privacy contrast).

## How to run

```bash
python config.py            # 1) run ONCE — sets up dirs, detects HE backend, writes config.yaml
python run_all.py           # 2) runs the whole pipeline: preprocess -> train -> excel -> figures
```

Options:

```bash
python config.py --profile full     # heavier settings for the paper (more rounds/clients/data)
python config.py --profile quick    # fast smoke run (default)
python run_all.py --only edgeiiot   # run a single dataset end-to-end
python run_all.py --skip-prep       # reuse existing processed data
```

## Outputs

* `data_processed/*.npz` — cleaned, ready-to-train arrays.
* `results/raw/*.json` — every metric + per-round history + HE overhead.
* `results/excel/HE-FedSec-results.xlsx` — summary / HE-overhead / convergence sheets (+ CSVs).
* `results/figures/*.png` — accuracy comparison, convergence, HE overhead, privacy–utility trade-off.

## Layout

```
config.py          run-once configurator        run_all.py        pipeline orchestrator
src/               core library                  scripts/          pipeline steps
  he_ckks.py         CKKS secure aggregation       prep_*.py          raw -> processed
  fed.py             FL loops + baselines          train_dataset.py   all methods for one dataset
  models.py          MLP + param<->vector          make_excel.py      collect -> xlsx
  partition.py       non-IID client splits         make_figures.py    collect -> png
  metrics.py         accuracy/F1/AUC
```

After results exist, write the paper from the numbers in `results/`.
