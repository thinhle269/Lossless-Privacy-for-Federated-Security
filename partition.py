"""Split a training pool into federated clients.

Strategies
----------
natural    : group rows by a real-world key (typing session, country, ...) so the
             federation mirrors deployment. `key` carries the grouping id per row.
dirichlet  : label-skew non-IID — each class is spread over clients with a
             Dirichlet(alpha) proportion (small alpha = more skew).
iid        : random equal-sized shards.

Returns a list of np.int64 index arrays (into the training pool), one per client.
"""
import numpy as np


def partition_clients(y, key, strategy, n_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)

    if strategy == "iid":
        idx = rng.permutation(n)
        return [np.sort(s) for s in np.array_split(idx, n_clients)]

    if strategy == "natural":
        key = np.asarray(key)
        uniq, counts = np.unique(key, return_counts=True)
        # assign whole groups to clients, greedily balancing client sizes
        order = uniq[np.argsort(-counts)]
        load = np.zeros(n_clients)
        group_to_client = {}
        for g in order:
            c = int(np.argmin(load))
            group_to_client[g] = c
            load[c] += counts[list(uniq).index(g)]
        buckets = [[] for _ in range(n_clients)]
        for i in range(n):
            buckets[group_to_client[key[i]]].append(i)
        out = [np.array(sorted(b), dtype=np.int64) for b in buckets]
        # guard against an empty client (tiny / few-group datasets)
        if any(len(b) == 0 for b in out):
            return partition_clients(y, key, "dirichlet", n_clients, alpha, seed)
        return out

    if strategy == "dirichlet":
        classes = np.unique(y)
        buckets = [[] for _ in range(n_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_c, cuts)):
                buckets[cid].extend(part.tolist())
        out = [np.array(sorted(b), dtype=np.int64) for b in buckets]
        # ensure no empty client
        for cid in range(n_clients):
            if len(out[cid]) == 0:
                donor = int(np.argmax([len(b) for b in out]))
                out[cid] = out[donor][:1]
                out[donor] = out[donor][1:]
        return out

    raise ValueError(f"unknown partition strategy: {strategy}")
