"""Analytical node and feature edges (scalar / ports / all) for systolic arrays (SCALE-Sim v3)."""
import math
import numpy as np


def dense_all_features(rows):
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        e, c = w["read_max"], w["span"]
        cap, n = w["output_capacity"], w["output_words"]
        burst, bo = x["r"] * x["c"], x["bw_o"]
        raw = [math.log2(x[k]) for k in ["bw_i", "bw_w", "bw_o", "sram_o", "m", "n", "k", "r", "c", "sram_i", "sram_w"]]
        raw += [math.log1p(w[k]) for k in ["span", "input_wait", "weight_wait", "read_max", "read_sum", "output_words", "output_capacity"]]
        raw += [
            c / e,
            n / (bo * e),
            burst / cap,
            n / cap,
            burst / (bo * (x["k"] + x["r"] + x["c"] - 2)),
            max(n / bo - e, 0) / e,
            max(burst / bo - x["r"] - x["c"], 0) / e,
        ]
        out.append(raw)
    return np.asarray(out, dtype=float)


def dense_boundary_features(rows, edge="ports"):
    if edge == "all":
        return dense_all_features(rows)
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        e, n, cap = w["read_max"], w["output_words"], w["output_capacity"]
        bo, burst = x["bw_o"], x["r"] * x["c"]
        feats = [
            math.log1p(e),
            math.log2(bo),
            math.log2(cap),
            n / (bo * e),
            burst / cap,
            max(burst / bo - x["r"] - x["c"], 0) / e,
        ]
        if edge == "ports":
            feats += [
                math.log1p(w["input_wait"]),
                math.log1p(w["weight_wait"]),
                w["input_wait"] / e,
                w["weight_wait"] / e,
            ]
        out.append(feats)
    return np.asarray(out, dtype=float)


def dense_clash_signature(row, edge="scalar"):
    return (
        tuple(dense_boundary_features([row], edge)[0])
        + (row["white"]["read_max"], row["white"]["output_capacity"] >= row["white"]["output_words"])
    )
