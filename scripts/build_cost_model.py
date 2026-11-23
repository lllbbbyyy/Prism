#!/usr/bin/env python3
"""Run Prism offline Feature-Model Graph construction and optional standalone export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from prism import CONTROLLERS, TASK_NAMES, construct_model, predict, write_standalone_bundle

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPACES = {
    "full_graph": None,
    "direct_black": "black",
    "predefined_hybrid": "mixed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and export a hybrid hardware cost model with Prism.")
    parser.add_argument("--task", default="Dense-L", choices=list(TASK_NAMES.keys()))
    parser.add_argument("--budget", type=int, default=32, choices=[16, 32])
    parser.add_argument("--seed", type=int, default=87001, help="Initial random sample seed (e.g. 87001 Trial 1, 87002 Trial 2).")
    parser.add_argument("--controller", default="adaptive_rules", choices=list(CONTROLLERS))
    parser.add_argument("--model-space", default="full_graph", choices=list(MODEL_SPACES.keys()))
    parser.add_argument("--no-clash-pruning", action="store_true", help="Disable pre-fit Eq. (1) clash pruning (ablation).")
    parser.add_argument("--llm-config", default=None, help="Path to OpenAI-compatible JSON config for live LLM agent calls.")
    parser.add_argument("--export-dir", default=None, help="Optional directory to write standalone `predictor.py` + `artifact.json`.")
    args = parser.parse_args()

    pool_path = ROOT / f"data/train_pools/{args.task}.jsonl"
    pool = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines()]
    test_rows = [
        r for r in (json.loads(line) for line in (ROOT / "data/test_144.jsonl").read_text(encoding="utf-8").splitlines())
        if r["task"] == args.task
    ]

    batches = (8, 16) if args.budget == 16 else (8, 16, 32)
    res = construct_model(
        task=args.task,
        pool=pool,
        batches=batches,
        seed=args.seed,
        controller=args.controller,
        llm_config=args.llm_config,
        family=MODEL_SPACES[args.model_space],
        prune_clashes=not args.no_clash_pruning,
    )

    preds = predict(res["compiled_artifact"], [r["x"] for r in test_rows])
    truth = np.asarray([r["result"]["value"] for r in test_rows], dtype=float)
    rel = np.abs(preds / truth - 1.0)

    print(f"Task: {args.task} | Budget: {args.budget} | Seed: {args.seed} | Controller: {args.controller}")
    print(f"Selected Path: {res['selected']['spec']['id']} (edge='{res['selected']['spec']['interface']}', family='{res['selected']['spec']['family']}')")
    print(f"Reached CV Threshold: {res['reached_threshold']} | LLM Calls: {res['total_llm_calls']} | Tokens: {res['total_tokens']}")
    for step in res["history"]:
        print(f"  Step -> {step['budget']:2d} pts | Route: {step['route']:<13} | Calls: {step['llm_calls']} | Sampler: {step['sampler']}")
    print(f"Unseen Test Set ({len(test_rows)} pts): MAPE = {rel.mean()*100:.2f}%, P90 = {np.quantile(rel, 0.9)*100:.2f}%")

    if args.export_dir:
        art_path, py_path = write_standalone_bundle(res["artifact"], Path(args.export_dir))
        print(f"Exported standalone bundle: {py_path} + {art_path}")


if __name__ == "__main__":
    main()
