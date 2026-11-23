"""Analytical node and feature edges (total / distribution) for sparse accelerators (STONNE)."""
import math
import numpy as np


def _row_counts(x):
    m, k = x["m"], x["k"]
    counts = [1] * m
    left = x["nnz"] - m
    pattern = x["pattern"]
    while left:
        for i in range(m):
            weight = (
                1 if pattern == 0 else
                k if pattern == 1 else
                (4 if i < m // 2 else 1) if pattern == 2 else
                1 + i % 5
            )
            take = min(left, weight, k - counts[i])
            counts[i] += take
            left -= take
            if not left:
                break
    return counts


def sparse_quantities(x):
    arr = np.asarray(_row_counts(x))
    tiles = np.ceil(arr / x["tk"])
    return {
        "macs": x["nnz"] * x["n"],
        "row_max": int(arr.max()),
        "row_std": float(arr.std()),
        "row_tiles": float(tiles.sum()),
        "max_tiles": float(tiles.max()),
        "row_square": float(np.sum(arr * arr)),
    }


def sparse_features(rows, edge="distribution"):
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        v = [math.log(x[k]) for k in ["m", "n", "k", "nnz", "units", "bw", "tn", "tk"]]
        v += [x["units"] / x["bw"], x["tn"] * x["tk"] / x["units"], x["nnz"] / (x["m"] * x["k"])]
        if edge == "distribution":
            v += [math.log1p(w[k]) for k in ["row_max", "row_std", "row_tiles", "max_tiles", "row_square"]]
            v += [w["row_max"] / (x["nnz"] / x["m"]), w["row_tiles"] * x["tk"] / x["nnz"]]
        out.append(v)
    return np.asarray(out, dtype=float)


def sparse_clash_signature(row, edge="total"):
    return tuple(sparse_features([row], edge)[0]) + (row["white"]["macs"] / row["x"]["units"],)
