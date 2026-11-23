"""Stage 3: Pre-fit feature clash pruning, K-fold CV, and threshold-guided speed selection."""
import statistics
import time
import warnings
import numpy as np
from sklearn.model_selection import KFold

from .adapters.dense import dense_clash_signature
from .adapters.noc import noc_features
from .adapters.sparse import sparse_clash_signature
from .graph import candidate_registry, make_candidate, task_backend


def compute_errors(y_true, y_pred):
    rel = np.abs(np.asarray(y_pred, dtype=float) / np.asarray(y_true, dtype=float) - 1.0)
    return {
        "mape": float(rel.mean()),
        "p90": float(np.quantile(rel, 0.9)),
        "maximum": float(rel.max()),
    }


def check_feature_clashes(task, observed, threshold=0.05):
    """Compute Err_min = |y2 - y1| / (y2 + y1) on equal-feature sample pairs."""
    b = task_backend(task)
    coarse_edge = "scalar" if b in ("accelerator", "noc") else "total"
    groups = {}
    if b == "accelerator":
        for r in observed:
            groups.setdefault(dense_clash_signature(r, coarse_edge), []).append(r)
    elif b == "noc":
        for r, z in zip(observed, noc_features(observed, coarse_edge)):
            groups.setdefault(tuple(z), []).append(r)
    else:
        for r in observed:
            groups.setdefault(sparse_clash_signature(r, coarse_edge), []).append(r)

    clashes = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        lo = min(rows, key=lambda r: r["result"]["value"])
        hi = max(rows, key=lambda r: r["result"]["value"])
        y1, y2 = lo["result"]["value"], hi["result"]["value"]
        err_min = (y2 - y1) / (y2 + y1)
        if err_min > threshold:
            clashes.append({
                "interface": coarse_edge,
                "ids": [lo["id"], hi["id"]],
                "values": [y1, y2],
                "lower_bound": float(err_min),
            })
    clashes.sort(key=lambda c: -c["lower_bound"])
    return {
        "status": "confirmed" if clashes else "unconfirmed",
        "forbidden_interfaces": [coarse_edge] if clashes else [],
        "certificates": clashes,
    }


def _profile_query_seconds(model, sample_rows):
    probe = sample_rows[: min(8, len(sample_rows))]
    model.predict([probe[0]])
    blocks = []
    for _ in range(5):
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 0.01:
            model.predict([probe[n % len(probe)]])
            n += 1
        blocks.append((time.perf_counter() - t0) / max(n, 1))
    return statistics.median(blocks)


def evaluate_candidates(task, observed, threshold_avg=0.05):
    folds = list(KFold(n_splits=3, shuffle=True, random_state=80017).split(observed))
    y = [r["result"]["value"] for r in observed]
    clash_info = check_feature_clashes(task, observed, threshold=threshold_avg)
    ranked = []
    for spec in candidate_registry(task):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                preds = np.zeros(len(observed))
                for tr, va in folds:
                    m = make_candidate(spec).fit([observed[i] for i in tr])
                    preds[va] = m.predict([observed[i] for i in va])
                metrics = compute_errors(y, preds)
                full_model = make_candidate(spec).fit(observed)
                q_sec = _profile_query_seconds(full_model, observed)
                ranked.append({"spec": spec, "status": "ok", "query_seconds": q_sec, **metrics})
            except Exception as exc:
                ranked.append({"spec": spec, "status": "fit_failure", "error": str(exc)})
    return {"labels": len(observed), "candidates": ranked, "diagnosis": clash_info}


def select_path(evidence_table, family=None, prune_clashes=True, eps_avg=0.05, eps_tail=0.10, speed_slack=0.10):
    forbidden = set(evidence_table["diagnosis"]["forbidden_interfaces"]) if prune_clashes else set()
    valid = [
        c for c in evidence_table["candidates"]
        if c["status"] == "ok"
        and (family is None or c["spec"]["family"] == family)
        and c["spec"]["interface"] not in forbidden
    ]
    if not valid:
        raise ValueError("No valid candidate path remains in the feature-model graph.")
    reached = [c for c in valid if c["mape"] <= eps_avg and c["p90"] <= eps_tail]
    if reached:
        fastest = min(c["query_seconds"] for c in reached)
        fast_band = [c for c in reached if c["query_seconds"] <= (1.0 + speed_slack) * fastest]
        chosen = min(fast_band, key=lambda c: (c["mape"], c["spec"]["id"]))
        return chosen, True
    chosen = min(valid, key=lambda c: (c["mape"], c["spec"]["id"]))
    return chosen, False
