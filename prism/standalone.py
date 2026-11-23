"""Standalone cost predictor runtime and serializer (Python stdlib + NumPy only at query time).

During online DSE queries, `compile_model` and `predict` depend strictly on `math`, `copy`,
and `numpy`—requiring 0 LLM calls, 0 simulator invocations, and 0 ML framework imports.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np

_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data/exported_models"


# ==============================================================================
# Analytical Quantities & Multi-Granularity Feature Extraction (NumPy + stdlib)
# ==============================================================================
def all_features(rows):
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


def boundary_features(rows, boundary):
    if boundary == "all":
        return all_features(rows)
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
        if boundary == "ports":
            feats += [
                math.log1p(w["input_wait"]),
                math.log1p(w["weight_wait"]),
                w["input_wait"] / e,
                w["weight_wait"] / e,
            ]
        out.append(feats)
    return np.asarray(out, dtype=float)


def route_features(x):
    size = x["size"]
    n = size * size
    edges = {}
    hops = 0.0
    self_prob = 0.0
    for source in range(n):
        sx, sy = source % size, source // size
        if x["traffic"] == "uniform":
            destinations = [(d, 1.0 / n) for d in range(n)]
        elif x["traffic"] == "transpose":
            destinations = [(sy + sx * size, 1.0)]
        else:
            raise ValueError("Unsupported traffic contract")
        for destination, p in destinations:
            dx, dy = destination % size, destination // size
            a, b = sx, sy
            if source == destination:
                self_prob += p / n
            while a != dx:
                step = 1 if dx > a else -1
                key = (a, b, a + step, b)
                edges[key] = edges.get(key, 0.0) + p
                a += step
                hops += p / n
            while b != dy:
                step = 1 if dy > b else -1
                key = (a, b, a, b + step)
                edges[key] = edges.get(key, 0.0) + p
                b += step
                hops += p / n
    values = [
        edges.get((a, b, a + da, b + db), 0.0)
        for a in range(size)
        for b in range(size)
        for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if 0 <= a + da < size and 0 <= b + db < size
    ]
    return {
        "mean_hops": float(hops),
        "max_link_load": float(max(values)),
        "link_std": float(np.std(values)),
        "self_probability": float(self_prob),
        "rho": float(x["load"] * max(values)),
    }


def noc_features(rows, boundary):
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        rho = x["load"] * x["size"] / 2.0 if boundary == "scalar" else w["rho"]
        q = rho / max(1.0 - rho, 0.02)
        z = [x["load"], w["mean_hops"], rho, q, q * q, q ** 3, q ** 5]
        if boundary in ("route", "all"):
            z += [w["max_link_load"], w["link_std"], w["self_probability"]]
        if boundary == "all":
            z += [float(x["traffic"] == "transpose")]
        out.append(z)
    return np.asarray(out, dtype=float)


def describe(x, backend, white_model=None):
    if backend == "noc":
        return {"x": x, "white": route_features(x)}
    m = white_model
    if any(x[k] != v for k, v in m["context"].items()):
        raise ValueError("Unknown workload/array context")
    if x.get("dataflow", "os") != m["dataflow"]:
        raise ValueError("Unknown dataflow")
    waits = []
    for port, name in zip(m["ports"], ("i", "w")):
        if x["sram_" + name] * 512 != port["capacity"]:
            raise ValueError("Unknown read capacity")
        if x["bw_" + name] <= 0:
            raise ValueError("Bandwidth must be positive")
        q = math.ceil(port["capacity"] / x["bw_" + name]) + 1
        waits.append(max([0] + [rank * q - t - 2 for rank, t in port["lines"]]))
    w = {
        "span": m["span"],
        "input_wait": waits[0],
        "weight_wait": waits[1],
        "read_max": m["span"] + max(waits),
        "read_sum": m["span"] + sum(waits),
        "output_words": m["output_words"],
        "output_capacity": x["sram_o"] * 512,
    }
    return {"x": x, "white": w}


def row_counts(x):
    m, k = x["m"], x["k"]
    counts = [1] * m
    left = x["nnz"] - m
    pattern = x["pattern"]
    if not 0 <= left <= m * (k - 1):
        raise ValueError("Invalid nnz")
    while left:
        for i in range(m):
            weight = (
                1 if pattern == 0 else
                k if pattern == 1 else
                (4 if i < m // 2 else 1) if pattern == 2 else
                1 + i % 5 if pattern == 3 else None
            )
            if weight is None:
                raise ValueError("Invalid pattern")
            take = min(left, weight, k - counts[i])
            counts[i] += take
            left -= take
            if not left:
                break
    return counts


def sparse_describe(x):
    a = np.asarray(row_counts(x))
    tiles = np.ceil(a / x["tk"])
    return {
        "macs": x["nnz"] * x["n"],
        "row_max": int(max(a)),
        "row_std": float(a.std()),
        "row_tiles": float(tiles.sum()),
        "max_tiles": float(tiles.max()),
        "row_square": float(np.sum(a * a)),
    }


def sparse_matrix(rows, boundary):
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        v = [math.log(x[k]) for k in ["m", "n", "k", "nnz", "units", "bw", "tn", "tk"]]
        v += [x["units"] / x["bw"], x["tn"] * x["tk"] / x["units"], x["nnz"] / (x["m"] * x["k"])]
        if boundary == "distribution":
            v += [math.log1p(w[k]) for k in ["row_max", "row_std", "row_tiles", "max_tiles", "row_square"]]
            v += [w["row_max"] / (x["nnz"] / x["m"]), w["row_tiles"] * x["tk"] / x["nnz"]]
        out.append(v)
    return np.asarray(out, dtype=float)


# ==============================================================================
# Pure-NumPy Inference Engine
# ==============================================================================
def packed_forest_predict(state, z, average=True):
    packed = state["_packed_forest"]
    count = packed["left"].shape[0]
    nodes = np.zeros((count, len(z)), dtype=int)
    ti = np.arange(count)[:, None]
    while True:
        active = packed["left"][ti, nodes] >= 0
        if not np.any(active):
            break
        trees, rows = np.nonzero(active)
        n = nodes[trees, rows]
        go_left = z[rows, packed["feature"][trees, n]] <= packed["threshold"][trees, n]
        nodes[trees, rows] = np.where(go_left, packed["left"][trees, n], packed["right"][trees, n])
    values = np.sum(packed["value"][ti, nodes], axis=0)
    return values / count if average else values


def _predict_forest(state, z_f32, average=True):
    if "_packed_forest" in state:
        return packed_forest_predict(state, z_f32, average=average)
    compiled = compile_model({"_tmp": state})["_tmp"]
    return packed_forest_predict(compiled, z_f32, average=average)


def predict_rows(state, rows):
    backend = state["backend"]
    mode = state["mode"]
    kind = state["kind"]
    if backend == "noc" or mode == "black":
        base = np.ones(len(rows))
    else:
        base = np.asarray([r["white"]["span" if mode == "compute" else "read_max"] for r in rows], dtype=float)
    if kind == "none":
        return base
    if backend == "noc":
        x = noc_features(rows, state["boundary"])
    elif mode == "local":
        x = boundary_features(rows, state["boundary"])
    else:
        x = all_features(rows)
    if kind == "analytic":
        design = np.concatenate([np.ones((len(x), 1)), x[:, [0, 1, 3, 5, 6]]], axis=1)
        return np.maximum(design @ np.asarray(state["coef"]), 1.0)
    z = (x - np.asarray(state["mean"])) / np.asarray(state["std"])
    if state.get("constant") is not None:
        q = np.full(len(rows), state["constant"])
    elif kind == "trees":
        q = _predict_forest(state, z.astype(np.float32), average=True)
    elif kind == "kernel":
        train = np.asarray(state["train"])
        dist = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(train * train, axis=1)[None, :] - 2.0 * z @ train.T, 0.0)
        q = np.exp(-state["gamma"] * dist) @ np.asarray(state["dual"])
    else:
        q = z @ np.asarray(state["coef"]) + state["intercept"]
    if backend == "accelerator" and mode == "local":
        q = np.where([r["white"]["output_capacity"] >= r["white"]["output_words"] for r in rows], 0.0, np.maximum(q, 0.0))
    return base * np.exp(np.clip(q, -15.0, 15.0))


def dense_or_noc_predict(artifact, configs):
    backend = artifact["state"]["backend"]
    support = artifact["support"]
    for x in configs:
        for key, values in support.items():
            if x[key] not in values:
                raise ValueError(f"Unsupported setting for {key}")
    rows = [describe(x, backend, artifact.get("white_model")) for x in configs]
    return predict_rows(artifact["state"], rows)


def sparse_predict(artifact, configs):
    for x in configs:
        if any(x[k] != v for k, v in artifact["context"].items()):
            raise ValueError("Unknown sparse workload context")
        if (
            x["units"] not in (16, 32, 64)
            or x["bw"] not in (4, 8, 16)
            or (x["tn"], x["tk"]) not in ((4, 4), (8, 2), (4, 8))
            or x["tn"] * x["tk"] > x["units"]
        ):
            raise ValueError("Unsupported sparse resource/tiling setting")
        if x["pattern"] not in range(4) or not x["m"] <= x["nnz"] <= x["m"] * x["k"]:
            raise ValueError("Unsupported sparse nonzero distribution")
    rows = [{"x": x, "white": sparse_describe(x)} for x in configs]
    state = artifact["state"]
    z = (sparse_matrix(rows, state["boundary"]) - np.asarray(state["mean"])) / np.asarray(state["std"])
    base = np.ones(len(rows)) if state["mode"] == "black" else np.asarray([r["white"]["macs"] / r["x"]["units"] for r in rows], dtype=float)
    if state.get("constant") is not None:
        q = np.full(len(rows), state["constant"])
    elif state["kind"].startswith("ridge"):
        q = z @ np.asarray(state["coef"]) + state["intercept"]
    elif state["kind"] == "kernel":
        tr = np.asarray(state["train"])
        dist = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(tr * tr, axis=1)[None, :] - 2.0 * z @ tr.T, 0.0)
        q = np.exp(-state["gamma"] * dist) @ np.asarray(state["dual"])
    else:
        q = _predict_forest(state, z.astype(np.float32), average=True)
    return base * np.exp(np.clip(q, -15.0, 15.0))


def _black_matrix(backend, configs, white_model):
    if backend == "sparse":
        return sparse_matrix([{"x": x, "white": sparse_describe(x)} for x in configs], "distribution")
    rows = [describe(x, backend, white_model) for x in configs]
    return noc_features(rows, "all") if backend == "noc" else all_features(rows)


def black_predict(artifact, configs):
    state = artifact["black"]
    x = _black_matrix(artifact["backend"], configs, artifact.get("white_model"))
    z = (x - np.asarray(state["mean"])) / np.asarray(state["std"])
    kind = state["type"]
    if kind == "Ridge":
        q = z @ np.asarray(state["coef"]) + state["intercept"]
    elif kind in ("ExtraTreesRegressor", "GradientBoostingRegressor"):
        q = _predict_forest(state, z.astype(np.float32), average=False)
        q = q / len(state["trees"]) if kind == "ExtraTreesRegressor" else state["initial"] + state["learning_rate"] * q
    elif kind in ("KernelRidge", "GaussianProcessRegressor"):
        tr = np.asarray(state["train"])
        dist = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(tr * tr, axis=1)[None, :] - 2.0 * z @ tr.T, 0.0)
        if kind == "KernelRidge":
            q = np.exp(-state["gamma"] * dist) @ np.asarray(state["dual"])
        else:
            scaled = np.sum((z[:, None, :] - tr[None, :, :]) ** 2, axis=2) / (state["length"] ** 2)
            if state["kernel"] == "RBF":
                k_mat = np.exp(-0.5 * scaled)
            else:
                r = np.sqrt(5.0 * scaled)
                k_mat = (1.0 + r + 5.0 * scaled / 3.0) * np.exp(-r)
            q = (state["amplitude"] * (k_mat @ np.asarray(state["alpha"]))) * state["target_std"] + state["target_mean"]
    elif kind == "MLPRegressor":
        q = z
        for i, (w, b) in enumerate(zip(state["weights"], state["bias"])):
            q = q @ np.asarray(w) + np.asarray(b)
            if i < len(state["weights"]) - 1:
                q = np.maximum(q, 0.0)
        q = q.ravel() * state["target_std"] + state["target_mean"]
    else:
        raise ValueError(f"Unsupported black-box model type: {kind}")
    return np.exp(np.clip(q, -15.0, 15.0))


def compile_model(artifact):
    """Pre-pack tree arrays and NumPy weight vectors for fast single-query execution."""
    compiled = deepcopy(artifact)

    def _visit(node):
        if isinstance(node, list):
            for item in node:
                _visit(item)
        elif isinstance(node, dict):
            for key, value in list(node.items()):
                if key in ("mean", "std", "coef", "train", "dual", "alpha") and isinstance(value, list):
                    node[key] = np.asarray(value)
                elif key in ("weights", "bias") and isinstance(value, list):
                    node[key] = [np.asarray(v) for v in value]
                else:
                    _visit(value)
            if "trees" in node and isinstance(node["trees"], list) and node["trees"]:
                trees = node["trees"]
                count = len(trees)
                width = max(len(t["left"]) for t in trees)
                packed = {
                    k: np.zeros((count, width), dtype=int if k in ("left", "right", "feature") else float)
                    for k in ("left", "right", "feature", "threshold", "value")
                }
                packed["left"].fill(-1)
                packed["right"].fill(-1)
                for i, t in enumerate(trees):
                    for k in packed:
                        packed[k][i, : len(t[k])] = t[k]
                node["_packed_forest"] = packed

    _visit(compiled)
    return compiled


def predict(artifact, configs, method="adaptive"):
    """Evaluate hardware latency predictions using only stdlib + NumPy."""
    selected = artifact["selected"] if method == "adaptive" else method
    if selected == "black":
        if artifact["backend"] == "sparse":
            context = artifact["legacy"]["context"]
            if any(any(x[k] != v for k, v in context.items()) for x in configs):
                raise ValueError("Unknown sparse context")
        else:
            for x in configs:
                if any(x[k] not in vals for k, vals in artifact["legacy"]["support"].items()):
                    raise ValueError("Unsupported configuration setting")
        return black_predict(artifact, configs)
    if artifact["backend"] == "sparse":
        return sparse_predict(artifact["legacy"], configs)
    return dense_or_noc_predict(artifact["legacy"], configs)


# ==============================================================================
# Step 6 Offline Serializer: Fitted Graph Candidate -> Standalone NumPy Artifact
# ==============================================================================
def _serialize_tree(estimator):
    t = estimator.tree_
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value[:, 0, 0].tolist(),
    }


def _serialize_black_pipeline(pipeline):
    scaler, est = pipeline.steps[0][1], pipeline.steps[-1][1]
    state = {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(), "type": type(est).__name__}
    if state["type"] == "TransformedTargetRegressor":
        tf = est.transformer_
        state.update(target_mean=float(tf.mean_[0]), target_std=float(tf.scale_[0]))
        est = est.regressor_
        state["type"] = "MLPRegressor"
    if state["type"] == "Ridge":
        state.update(coef=est.coef_.tolist(), intercept=float(est.intercept_))
    elif state["type"] == "ExtraTreesRegressor":
        state["trees"] = [_serialize_tree(t) for t in est.estimators_]
    elif state["type"] == "GradientBoostingRegressor":
        state.update(
            trees=[_serialize_tree(t[0]) for t in est.estimators_],
            learning_rate=float(est.learning_rate),
            initial=float(est.init_.constant_[0, 0]),
        )
    elif state["type"] == "KernelRidge":
        state.update(
            train=est.X_fit_.tolist(),
            dual=est.dual_coef_.tolist(),
            gamma=float(est.gamma or 1.0 / est.X_fit_.shape[1]),
        )
    elif state["type"] == "GaussianProcessRegressor":
        k = est.kernel_
        state.update(
            train=est.X_train_.tolist(),
            alpha=est.alpha_.tolist(),
            amplitude=float(k.k1.k1.constant_value),
            length=float(k.k1.k2.length_scale),
            kernel=type(k.k1.k2).__name__,
            target_mean=float(np.asarray(est._y_train_mean).item()),
            target_std=float(np.asarray(est._y_train_std).item()),
        )
    elif state["type"] == "MLPRegressor":
        state.update(weights=[w.tolist() for w in est.coefs_], bias=[b.tolist() for b in est.intercepts_])
    else:
        raise ValueError(f"Unsupported estimator for standalone export: {state['type']}")
    return state


def _serialize_hybrid_or_white(node):
    state = {
        "backend": node.backend,
        "boundary": node.edge,
        "mode": "local",
        "kind": node.kind,
    }
    if node.kind == "none":
        return state
    state.update(mean=node.mean.tolist(), std=node.std.tolist())
    if node.kind == "analytic":
        state["coef"] = node.coef.tolist()
        return state
    state["constant"] = node.constant
    if node.constant is not None:
        return state
    est = node.estimator
    if node.kind == "trees":
        state["trees"] = [_serialize_tree(t) for t in est.estimators_]
    elif node.kind == "kernel":
        state.update(train=est.X_fit_.tolist(), dual=est.dual_coef_.tolist(), gamma=float(est.gamma))
    else:
        state.update(coef=est.coef_.tolist(), intercept=float(est.intercept_))
    return state


def export_artifact(task: str, spec: dict, fitted_model, template_dir: Path | None = None) -> dict:
    """Serialize a fitted Feature-Model Graph candidate into a standalone JSON artifact."""
    tdir = Path(template_dir) if template_dir is not None else _DEFAULT_TEMPLATE_DIR
    base_tpl = json.loads((tdir / f"{task}.json").read_text(encoding="utf-8"))
    backend = spec["backend"]
    artifact = {
        "task": base_tpl["task"],
        "task_name": task,
        "backend": backend,
        "selected": "black" if spec["family"] == "black" else "legacy",
        "legacy": deepcopy(base_tpl["legacy"]),
    }
    if "white_model" in base_tpl:
        artifact["white_model"] = deepcopy(base_tpl["white_model"])
    if spec["family"] == "black":
        artifact["black"] = _serialize_black_pipeline(fitted_model.model)
    else:
        state = _serialize_hybrid_or_white(fitted_model)
        if backend == "sparse" and state["kind"].startswith("ridge"):
            state["kind"] = "ridge"
        artifact["legacy"]["state"] = state
    return artifact


def write_standalone_bundle(artifact: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write `artifact.json` and a self-contained `predictor.py` bundle for DSE deployment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "artifact.json"
    predictor_path = out_dir / "predictor.py"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    runtime_src = Path(__file__).read_text(encoding="utf-8").split("# ==============================================================================\n# Step 6 Offline Serializer")[0].rstrip()
    cli_footer = (
        "\n\n"
        "def load_compiled(artifact_path=None):\n"
        "    p = Path(artifact_path) if artifact_path else Path(__file__).with_name('artifact.json')\n"
        "    return compile_model(json.loads(p.read_text(encoding='utf-8')))\n\n\n"
        "if __name__ == '__main__':\n"
        "    import sys\n"
        "    bundle = load_compiled()\n"
        "    if len(sys.argv) > 1:\n"
        "        cfgs = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "        print(json.dumps(predict(bundle, cfgs).tolist(), indent=2))\n"
        "    else:\n"
        "        print(f\"Loaded standalone predictor for {bundle.get('task_name', bundle['task'])} (selected={bundle['selected']})\")\n"
    )
    predictor_path.write_text(runtime_src + cli_footer, encoding="utf-8")
    return artifact_path, predictor_path
