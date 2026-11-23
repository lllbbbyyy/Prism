"""Deterministic public-parameter sampling operators (Coverage, Paired, SingleAxis, Joint)."""
import math
import numpy as np

from .adapters.dense import dense_all_features
from .adapters.noc import noc_features
from .adapters.sparse import sparse_features
from .graph import task_backend


def acquire_batch(task, pool, observed, target_count, mode="coverage", seed=80001):
    """Select unqueried configurations using only public parameters (never unqueried labels)."""
    b = task_backend(task)
    public_pool = [{k: r[k] for k in ("id", "x", "white")} for r in pool]
    lookup = {r["id"]: r for r in pool}
    selected = [{k: r[k] for k in ("id", "x", "white")} for r in observed]
    rng = np.random.default_rng(seed)

    if not selected:
        order = rng.permutation(len(public_pool))[:target_count]
        return [lookup[public_pool[int(i)]["id"]] for i in order]

    while len(selected) < target_count:
        seen_ids = {s["id"] for s in selected}
        remaining = [r for r in public_pool if r["id"] not in seen_ids]
        if not remaining:
            break
        if mode == "random":
            idx = int(rng.integers(len(remaining)))
            selected.append(remaining[idx])
            continue

        if b == "accelerator" and mode == "joint":
            cells = {}
            for r in remaining:
                if r["white"]["output_capacity"] >= r["white"]["output_words"]:
                    continue
                cells.setdefault((r["x"]["bw_o"], r["x"]["sram_o"]), []).append(r)
            if cells:
                existing = np.asarray([[math.log2(r["x"]["bw_o"]), math.log2(r["x"]["sram_o"])] for r in selected])
                scored = []
                for key, rows in cells.items():
                    low = min(rows, key=lambda r: (r["white"]["read_max"], r["id"]))
                    high = max(rows, key=lambda r: r["white"]["read_max"])
                    dist = float(np.min(np.sum((existing - np.log2(np.asarray(key))) ** 2, axis=1)))
                    x, w = low["x"], low["white"]
                    exposure = w["output_words"] / (x["bw_o"] * w["read_max"]) + max(x["r"] * x["c"] / w["output_capacity"] - 1, 0) / x["bw_o"]
                    suppression = math.log1p(high["white"]["read_max"] / low["white"]["read_max"])
                    scored.append((dist, exposure * suppression, low, high))
                _, _, low, high = max(scored, key=lambda item: ((1 + item[0]) * (1 + item[1]), item[2]["id"]))
                for row in (low, high):
                    if len(selected) < target_count and row["id"] not in {s["id"] for s in selected}:
                        selected.append(row)
                continue
            mode = "coverage"

        feat_fn = dense_all_features if b == "accelerator" else (lambda rs: noc_features(rs, "all")) if b == "noc" else (lambda rs: sparse_features(rs, "distribution"))
        z_rem = feat_fn(remaining)
        z_obs = feat_fn(selected)
        scale = np.maximum(np.std(np.concatenate([z_rem, z_obs], axis=0), axis=0), 1e-8)
        dist = np.min(np.mean(((z_rem[:, None, :] - z_obs[None, :, :]) / scale) ** 2, axis=2), axis=1)

        if mode == "single_axis" and b == "accelerator":
            knobs = ("bw_i", "bw_w", "bw_o", "sram_o")
            bonus = np.asarray([any(sum(r["x"][k] != s["x"][k] for k in knobs) == 1 for s in selected) for r in remaining], dtype=float)
            dist = dist + bonus * 1e5
        elif mode == "pairs" and b == "noc":
            bonus = np.asarray([any(r["x"]["load"] == s["x"]["load"] and r["x"]["traffic"] != s["x"]["traffic"] for s in selected) for r in remaining], dtype=float)
            dist = dist + bonus * 1e5
        elif mode == "pairs" and b == "sparse":
            knobs = ("m", "n", "k", "nnz", "units", "bw", "tn", "tk")
            bonus = np.asarray([any(all(r["x"][k] == s["x"][k] for k in knobs) and r["x"]["pattern"] != s["x"]["pattern"] for s in selected) for r in remaining], dtype=float)
            dist = dist + bonus * 1e5

        selected.append(remaining[int(np.argmax(dist))])

    return [lookup[r["id"]] for r in selected]
