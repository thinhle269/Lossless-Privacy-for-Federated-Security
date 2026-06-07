"""Training loops: Centralized / Local-only / FedAvg / HE-FedAvg / FedAvg+DP.

All federated methods share one loop (`_run_fl`); the only difference is how the
per-client parameter vectors are aggregated each round:
  * fedavg     -> plaintext weighted mean
  * he_fedavg  -> CKKS encrypted weighted mean (server sees only ciphertexts)
  * fedavg_dp  -> clip per-client delta + Gaussian noise (privacy contrast)
Every method is evaluated on the SAME held-out global test set for fairness.
"""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model, get_param_vector, set_param_vector, num_params
from src.metrics import compute_metrics
from src.he_ckks import CKKSAggregator
from src.utils import Timer

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# low-level training / evaluation
# --------------------------------------------------------------------------- #
def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(np.asarray(X, np.float32)),
                       torch.from_numpy(np.asarray(y, np.int64)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _train_epochs(model, X, y, epochs, lr, batch, class_weight=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(
        weight=None if class_weight is None
        else torch.tensor(class_weight, dtype=torch.float32))
    for _ in range(epochs):
        for xb, yb in _loader(X, y, batch, True):
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def _evaluate(model, X, y, task):
    model.eval()
    logits = model(torch.from_numpy(np.asarray(X, np.float32)))
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    pred = prob.argmax(1)
    y_prob = prob[:, 1] if (task == "binary" and prob.shape[1] == 2) else None
    return compute_metrics(y, pred, y_prob, task)


def _class_weight(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).tolist()


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def run_centralized(data, cfg):
    fl, m = cfg["fl"], cfg["_model"]
    cw = _class_weight(data["y_train"], data["n_classes"]) if data["task"] == "binary" else None
    model = build_model(m, data["in_dim"], data["n_classes"])
    with Timer() as t:
        _train_epochs(model, data["X_train"], data["y_train"],
                      fl["rounds"] * fl["local_epochs"], fl["lr"], fl["batch_size"], cw)
    res = _evaluate(model, data["X_test"], data["y_test"], data["task"])
    res.update(method="centralized", wall_time_s=t.dt, n_params=num_params(model))
    return res


def run_local(data, cfg):
    fl, m = cfg["fl"], cfg["_model"]
    per = []
    with Timer() as t:
        for idx in data["clients"]:
            cw = _class_weight(data["y_train"][idx], data["n_classes"]) if data["task"] == "binary" else None
            model = build_model(m, data["in_dim"], data["n_classes"])
            _train_epochs(model, data["X_train"][idx], data["y_train"][idx],
                          fl["rounds"] * fl["local_epochs"], fl["lr"], fl["batch_size"], cw)
            per.append(_evaluate(model, data["X_test"], data["y_test"], data["task"]))
    res = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    res.update(method="local", wall_time_s=t.dt,
               n_params=num_params(build_model(m, data["in_dim"], data["n_classes"])))
    return res


# --------------------------------------------------------------------------- #
# federated loop (fedavg / he_fedavg / fedavg_dp)
# --------------------------------------------------------------------------- #
def _run_fl(data, cfg, method):
    fl, m = cfg["fl"], cfg["_model"]
    rounds = fl["rounds"]
    he = CKKSAggregator(**cfg["he"]) if method == "he_fedavg" else None
    sizes = np.array([len(idx) for idx in data["clients"]], dtype=np.float64)
    weights = sizes / sizes.sum()

    glob = build_model(m, data["in_dim"], data["n_classes"])
    n_params = num_params(glob)
    plain_bytes = n_params * 4  # float32 update, per client per round

    ov = {"enc_s": 0.0, "agg_s": 0.0, "dec_s": 0.0, "cipher_bytes": 0}
    history = []

    with Timer() as t:
        for _ in range(rounds):
            gvec = get_param_vector(glob)
            local_vecs = []
            for idx in data["clients"]:
                lm = build_model(m, data["in_dim"], data["n_classes"])
                set_param_vector(lm, gvec)
                cw = _class_weight(data["y_train"][idx], data["n_classes"]) if data["task"] == "binary" else None
                _train_epochs(lm, data["X_train"][idx], data["y_train"][idx],
                              fl["local_epochs"], fl["lr"], fl["batch_size"], cw)
                local_vecs.append(get_param_vector(lm))

            if method == "fedavg":
                new = sum(w * v for w, v in zip(weights, local_vecs))

            elif method == "fedavg_dp":
                clip, sigma = cfg["dp"]["clip"], cfg["dp"]["sigma"]
                agg = np.zeros_like(gvec)
                for w, v in zip(weights, local_vecs):
                    d = v - gvec
                    norm = np.linalg.norm(d)
                    d = d * min(1.0, clip / (norm + 1e-12))
                    agg += w * d
                agg += np.random.normal(0.0, sigma * clip / len(local_vecs), size=agg.shape)
                new = gvec + agg

            elif method == "he_fedavg":
                encs = []
                for v in local_vecs:
                    e, dt, nb = he.encrypt(v)
                    encs.append(e)
                    ov["enc_s"] += dt
                    ov["cipher_bytes"] = nb  # identical every client/round
                agg_enc, dt = he.aggregate(encs, weights)
                ov["agg_s"] += dt
                new, dt = he.decrypt(agg_enc, n_params)
                ov["dec_s"] += dt
            else:
                raise ValueError(method)

            set_param_vector(glob, new)
            met = _evaluate(glob, data["X_test"], data["y_test"], data["task"])
            history.append(met)

    res = dict(history[-1])
    res.update(method=method, wall_time_s=t.dt, n_params=n_params,
               history=history, rounds=rounds, n_clients=len(data["clients"]))

    if method == "he_fedavg":
        nc = len(data["clients"])
        res["he"] = {
            **he.info(),
            "enc_ms_per_client_round": 1e3 * ov["enc_s"] / (rounds * nc),
            "agg_ms_per_round": 1e3 * ov["agg_s"] / rounds,
            "dec_ms_per_round": 1e3 * ov["dec_s"] / rounds,
            "cipher_bytes_per_client": int(ov["cipher_bytes"]),
            "plain_bytes_per_client": int(plain_bytes),
            "expansion_x": ov["cipher_bytes"] / plain_bytes,
        }
        res["comm_upload_MB"] = ov["cipher_bytes"] * nc * rounds / 1e6
    else:
        res["comm_upload_MB"] = plain_bytes * len(data["clients"]) * rounds / 1e6
    return res


def run_method(method, data, cfg):
    if method == "centralized":
        return run_centralized(data, cfg)
    if method == "local":
        return run_local(data, cfg)
    return _run_fl(data, cfg, method)
