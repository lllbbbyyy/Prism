"""Unit tests for Prism Feature-Model Graph identification, construction, and standalone export."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from prism import (
    check_feature_clashes,
    compile_model,
    construct_model,
    evaluate_candidates,
    export_artifact,
    make_candidate,
    predict,
    select_path,
    write_standalone_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


class TestPrismFramework(unittest.TestCase):
    def test_feature_clash_bound_on_sparse(self) -> None:
        pool = [json.loads(l) for l in (ROOT / "data/train_pools/Sparse-S.jsonl").read_text(encoding="utf-8").splitlines()]
        clash = check_feature_clashes("Sparse-S", pool, threshold=0.05)
        self.assertEqual(clash["status"], "confirmed")
        self.assertIn("total", clash["forbidden_interfaces"])
        top = clash["certificates"][0]
        y1, y2 = top["values"]
        self.assertAlmostEqual(top["lower_bound"], (y2 - y1) / (y2 + y1), places=9)
        self.assertGreater(top["lower_bound"], 0.05)

    def test_speed_aware_path_selection_prefers_white_box_on_dense_s(self) -> None:
        pool = [json.loads(l) for l in (ROOT / "data/train_pools/Dense-S.jsonl").read_text(encoding="utf-8").splitlines()][:24]
        ev = evaluate_candidates("Dense-S", pool)
        winner, reached = select_path(ev)
        self.assertTrue(reached)
        self.assertEqual(winner["spec"]["id"], "white_read")

    def test_standalone_export_parity_and_bundle_execution(self) -> None:
        for task, spec_id in [
            ("Dense-S", "white_read"),
            ("Dense-L", "mixed_ports_trees"),
            ("NoC-L", "mixed_route_ridge_0.1"),
            ("Sparse-S", "black_ridge_1"),
        ]:
            pool = [json.loads(l) for l in (ROOT / f"data/train_pools/{task}.jsonl").read_text(encoding="utf-8").splitlines()][:24]
            test_rows = [
                json.loads(l)
                for l in (ROOT / "data/test_144.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(l)["task"] == task
            ]
            from prism import candidate_registry
            spec = next(s for s in candidate_registry(task) if s["id"] == spec_id)
            fitted = make_candidate(spec).fit(pool)
            artifact = export_artifact(task, spec, fitted)
            compiled = compile_model(artifact)
            expected = fitted.predict(test_rows)
            actual = predict(compiled, [r["x"] for r in test_rows])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

        # Verify emitted standalone predictor.py runs in an isolated subprocess
        with tempfile.TemporaryDirectory() as tmp:
            _, py_path = write_standalone_bundle(artifact, Path(tmp))
            cfg_file = Path(tmp) / "configs.json"
            cfg_file.write_text(json.dumps([r["x"] for r in test_rows[:3]]), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", str(py_path), str(cfg_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            preds = np.asarray(json.loads(proc.stdout), dtype=float)
            np.testing.assert_allclose(preds, expected[:3], rtol=1e-6, atol=1e-7)

    def test_offline_construction_loop_and_144_unseen_replay(self) -> None:
        pool = [json.loads(l) for l in (ROOT / "data/train_pools/NoC-S.jsonl").read_text(encoding="utf-8").splitlines()]
        res = construct_model("NoC-S", pool, batches=(8, 16, 32), seed=87001, controller="adaptive_rules")
        self.assertEqual(res["total_llm_calls"], 0)
        self.assertTrue(res["reached_threshold"])

        test_rows = [json.loads(l) for l in (ROOT / "data/test_144.jsonl").read_text(encoding="utf-8").splitlines()]
        expected_domain_mape = {"Dense": 2.08, "NoC": 0.86, "Sparse": 7.14}
        for dom, pair in [("Dense", ("Dense-S", "Dense-L")), ("NoC", ("NoC-S", "NoC-L")), ("Sparse", ("Sparse-S", "Sparse-L"))]:
            mapes = []
            for task in pair:
                art = compile_model(json.loads((ROOT / f"data/exported_models/{task}.json").read_text(encoding="utf-8")))
                pts = [r for r in test_rows if r["task"] == task]
                p = predict(art, [r["x"] for r in pts])
                y = np.asarray([r["result"]["value"] for r in pts], dtype=float)
                mapes.append(float(np.mean(np.abs(p / y - 1.0)) * 100.0))
            self.assertAlmostEqual(float(np.mean(mapes)), expected_domain_mape[dom], places=2)


if __name__ == "__main__":
    unittest.main()
