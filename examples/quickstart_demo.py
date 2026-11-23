#!/usr/bin/env python3
"""Quickstart walkthrough of Prism feature clash pruning, path matching, and standalone DSE querying."""
from __future__ import annotations

import json
import time
from pathlib import Path

from prism import check_feature_clashes, compile_model, evaluate_candidates, predict, select_path
from prism.samplers import acquire_batch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # 1. Pre-fit Feature Clash Check on Sparse-S (rules out coarse 'total' edge before fitting)
    sparse_pool = [json.loads(l) for l in (ROOT / "data/train_pools/Sparse-S.jsonl").read_text(encoding="utf-8").splitlines()]
    sample_16 = acquire_batch("Sparse-S", sparse_pool, [], 16, mode="random", seed=87001)
    clash = check_feature_clashes("Sparse-S", sample_16)
    print("[1] Pre-fit Feature Clash Check on Sparse-S (16 samples):")
    if clash["certificates"]:
        c0 = clash["certificates"][0]
        print(
            f"    Eliminated coarse edge '{c0['interface']}': equal-NNZ tiles took "
            f"{c0['values'][0]:.0f} vs. {c0['values'][1]:.0f} cycles -> Err_min = {c0['lower_bound']*100:.2f}% > 5%"
        )

    # 2. Adaptive Feature-Model Graph Matching across Distinct Hardware Bottlenecks
    print("\n[2] Adaptive Feature-Model Graph Path Selection across Hardware Regimes:")
    for task in ("Dense-S", "Dense-L", "NoC-L", "Sparse-S"):
        t_pool = [json.loads(l) for l in (ROOT / f"data/train_pools/{task}.jsonl").read_text(encoding="utf-8").splitlines()]
        sample_24 = acquire_batch(task, t_pool, [], 24, mode="random", seed=87002)
        ev = evaluate_candidates(task, sample_24)
        winner, reached = select_path(ev)
        spec = winner["spec"]
        print(
            f"    {task:<9} -> selected '{spec['id']:<20}' "
            f"(edge='{spec['interface']}', family='{spec['family']}', CV MAPE={winner['mape']*100:4.2f}%, reached_eps={reached})"
        )

    # 3. Standalone NumPy Predictor Query Speed (0 LLM calls, 0 simulator invocations)
    artifact = json.loads((ROOT / "data/exported_models/Dense-L.json").read_text(encoding="utf-8"))
    compiled = compile_model(artifact)
    test_pts = [
        json.loads(l)["x"]
        for l in (ROOT / "data/test_144.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(l)["task"] == "Dense-L"
    ]
    # Warmup + timed queries
    for _ in range(20):
        predict(compiled, test_pts)
    t0 = time.perf_counter()
    for _ in range(100):
        preds = predict(compiled, test_pts)
    dt_us = (time.perf_counter() - t0) * 1e6 / (100 * len(test_pts))
    print(
        f"\n[3] Standalone NumPy Predictor (Dense-L): {dt_us:.1f} us/config "
        f"(first 3 unseen test predictions: {preds[:3].round(1).tolist()} cycles)"
    )


if __name__ == "__main__":
    main()
