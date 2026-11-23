#!/usr/bin/env python3
"""Replay standalone NumPy predictors on all 144 unseen test configurations and print paper tables."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prism_standalone", ROOT / "prism/standalone.py")
standalone = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(standalone)
TASKS = ["Dense-S", "Dense-L", "NoC-S", "NoC-L", "Sparse-S", "Sparse-L"]
DOMAINS = {
    "Dense": ("Dense-S", "Dense-L"),
    "NoC": ("NoC-S", "NoC-L"),
    "Sparse": ("Sparse-S", "Sparse-L"),
}


def main() -> None:
    test_rows = [json.loads(line) for line in (ROOT / "data/test_144.jsonl").read_text(encoding="utf-8").splitlines()]
    acc_summary = json.loads((ROOT / "data/summaries/main_accuracy.json").read_text(encoding="utf-8"))
    spd_summary = json.loads((ROOT / "data/summaries/query_speedup.json").read_text(encoding="utf-8"))
    collab = json.loads((ROOT / "data/summaries/collaboration_12runs.json").read_text(encoding="utf-8"))

    spd_lookup = {(r["domain"], r["method"]): r for r in spd_summary}
    task_metrics: dict[str, tuple[float, float]] = {}

    print("=== [1] Standalone NumPy Predictor Replay (144 Unseen Test Configurations) ===")
    for task in TASKS:
        artifact = json.loads((ROOT / f"data/exported_models/{task}.json").read_text(encoding="utf-8"))
        compiled = standalone.compile_model(artifact)
        pts = [r for r in test_rows if r["task"] == task]
        preds = standalone.predict(compiled, [r["x"] for r in pts])
        truth = np.asarray([r["result"]["value"] for r in pts], dtype=float)
        rel = np.abs(preds / truth - 1.0)
        mape, p90 = float(rel.mean() * 100.0), float(np.quantile(rel, 0.9) * 100.0)
        task_metrics[task] = (mape, p90)
        print(f"  {task:<9} | n={len(pts):2d} | MAPE = {mape:5.2f}% | P90 = {p90:5.2f}%")

    # Verify zero ML/HTTP framework modules were imported by standalone replay
    forbidden = {"sklearn", "scipy", "torch", "requests"}
    loaded_forbidden = forbidden.intersection({m.split(".")[0] for m in sys.modules})
    assert not loaded_forbidden, f"Unexpected non-standalone imports loaded: {loaded_forbidden}"

    # Verify exact match between live standalone replay and main_accuracy.json
    for dom, (t_s, t_l) in DOMAINS.items():
        ref = next(r for r in acc_summary if r["domain"] == dom and r["method"] == "prism")
        np.testing.assert_allclose(task_metrics[t_s][0] / 100.0, ref["small_mape"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(task_metrics[t_l][0] / 100.0, ref["large_mape"], rtol=1e-6, atol=1e-7)

    print("\n=== [2] Accuracy & Paired Query Speedup Across Baselines (RQ1 & RQ2) ===")
    header = f"  {'Hardware':<8} | {'Method':<18} | {'Small MAPE/P90':<15} | {'Large MAPE/P90':<15} | {'Mean MAPE':<9} | {'Pts':<4} | {'Lat(us)':<7} | {'Speedup':<7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in acc_summary:
        sp = spd_lookup[(row["domain"], row["method"])]
        sm = f"{row['small_mape']*100:5.2f}%/{row['small_p90']*100:5.2f}%"
        lg = f"{row['large_mape']*100:5.2f}%/{row['large_p90']*100:5.2f}%"
        print(
            f"  {row['domain']:<8} | {row['paper_name']:<18} | {sm:<15} | {lg:<15} | "
            f"{row['mape']*100:7.2f}%  | {row['samples']:4.1f} | {sp['latency_us']:7.1f} | {sp['native_speedup']:6.0f}x"
        )

    print("\n=== [3] On-Demand Agent Routing & LLM Cost Savings (RQ3) ===")
    c_hdr = f"  {'Controller':<16} | {'Builds':<6} | {'LLM Calls':<9} | {'Tokens':<8} | {'Reach eps':<9} | {'Dense':<7} | {'NoC':<7} | {'Sparse':<7}"
    print(c_hdr)
    print("  " + "-" * (len(c_hdr) - 2))
    for c in collab["controllers"]:
        reach = f"{c['reach_threshold']}/{c['builds']}"
        tok = f"{c['tokens']/1000.0:.1f}K" if c["tokens"] else "0"
        print(
            f"  {c['paper_name']:<16} | {c['builds']:<6d} | {c['llm_calls']:<9d} | {tok:<8} | "
            f"{reach:<9} | {c['dense_mape']*100:5.2f}% | {c['noc_mape']*100:5.2f}% | {c['sparse_mape']*100:5.2f}%"
        )


if __name__ == "__main__":
    main()
