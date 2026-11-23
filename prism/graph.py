"""Feature-Model Graph candidate definitions and predictor node implementations."""
import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from .adapters import dense_all_features, dense_boundary_features, noc_features, sparse_features
from .learners import build_estimators

TASK_NAMES = {
    "Dense-S": "accelerator",
    "Dense-L": "accelerator",
    "NoC-S": "noc",
    "NoC-L": "noc",
    "Sparse-S": "sparse",
    "Sparse-L": "sparse",
}


def task_backend(task):
    if task in TASK_NAMES:
        return TASK_NAMES[task]
    if task.startswith("graph"):
        return "accelerator"
    if task.startswith("noc"):
        return "noc"
    return "sparse"


class DirectBlackBoxNode:
    def __init__(self, backend, learner_name):
        self.backend = backend
        self.learner_name = learner_name

    def _matrix(self, rows):
        if self.backend == "accelerator":
            return dense_all_features(rows)
        if self.backend == "noc":
            return noc_features(rows, "all")
        return sparse_features(rows, "distribution")

    def fit(self, rows):
        y = np.log([r["result"]["value"] for r in rows])
        self.model = clone(build_estimators()[self.learner_name]).fit(self._matrix(rows), y)
        return self

    def predict(self, rows):
        return np.exp(np.clip(self.model.predict(self._matrix(rows)), -15.0, 15.0))


class HybridOrWhiteNode:
    def __init__(self, backend="accelerator", edge="ports", kind="trees"):
        self.backend = backend
        self.edge = edge
        self.kind = kind

    def _matrix(self, rows):
        if self.backend == "noc":
            return noc_features(rows, self.edge)
        if self.backend == "sparse":
            return sparse_features(rows, self.edge)
        return dense_boundary_features(rows, self.edge)

    def _prior(self, rows):
        if self.backend == "noc":
            return np.ones(len(rows))
        if self.backend == "sparse":
            return np.asarray([r["white"]["macs"] / r["x"]["units"] for r in rows], dtype=float)
        return np.asarray([r["white"]["read_max"] for r in rows], dtype=float)

    def fit(self, rows):
        if self.kind == "none":
            return self
        x = self._matrix(rows)
        y = np.asarray([r["result"]["value"] for r in rows], dtype=float)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-8)
        if self.kind == "analytic":
            a = np.concatenate([np.ones((len(x), 1)), x[:, [0, 1, 3, 5, 6]]], axis=1)
            scaled = a / y[:, None]
            norm = np.maximum(np.linalg.norm(scaled, axis=0), 1e-12)
            self.coef = nnls(scaled / norm, np.ones(len(y)), maxiter=1000)[0] / norm
            return self
        target = np.log(y / self._prior(rows))
        self.constant = float(target.mean()) if np.max(np.abs(target - target.mean())) < 1e-12 else None
        if self.constant is not None:
            return self
        z = (x - self.mean) / self.std
        if self.kind == "trees":
            seed = 60017 if self.backend == "sparse" else 56001
            self.estimator = ExtraTreesRegressor(n_estimators=48, random_state=seed, max_features=1.0, min_samples_leaf=1, n_jobs=1)
        elif self.kind == "kernel":
            self.estimator = KernelRidge(alpha=0.03, kernel="rbf", gamma=1.0 / z.shape[1])
        elif self.kind.startswith("ridge"):
            alpha = float(self.kind.split("_")[1]) if "_" in self.kind else 1.0
            self.estimator = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unknown learner kind: {self.kind}")
        self.estimator.fit(z, target)
        return self

    def predict(self, rows):
        prior = self._prior(rows)
        if self.kind == "none":
            return prior
        x = self._matrix(rows)
        if self.kind == "analytic":
            return np.maximum(np.concatenate([np.ones((len(x), 1)), x[:, [0, 1, 3, 5, 6]]], axis=1) @ self.coef, 1.0)
        z = (x - self.mean) / self.std
        q = np.full(len(rows), self.constant) if self.constant is not None else self.estimator.predict(z)
        if self.backend == "accelerator":
            q = np.maximum(q, 0.0)
            q = np.where([r["white"]["output_capacity"] >= r["white"]["output_words"] for r in rows], 0.0, q)
        return prior * np.exp(np.clip(q, -15.0, 15.0))


def candidate_registry(task):
    b = task_backend(task)
    learners = list(build_estimators().keys())
    if b == "noc":
        out = [{"id": f"black_{k}", "family": "black", "interface": "all", "kind": k, "backend": "noc"} for k in learners]
        for edge in ("scalar", "route"):
            for k in ("analytic", "ridge_0.1", "ridge_1.0", "kernel", "trees"):
                out.append({"id": f"mixed_{edge}_{k}", "family": "mixed", "interface": edge, "kind": k, "backend": "noc"})
        return out
    default_edge = "all" if b == "accelerator" else "distribution"
    out = [{"id": f"black_{k}", "family": "black", "interface": default_edge, "kind": k, "backend": b} for k in learners]
    if b == "accelerator":
        out.append({"id": "white_read", "family": "white", "interface": "scalar", "kind": "none", "backend": b})
        for edge in ("scalar", "ports"):
            for k in ("ridge_0.1", "ridge_1.0", "kernel", "trees"):
                out.append({"id": f"mixed_{edge}_{k}", "family": "mixed", "interface": edge, "kind": k, "backend": b})
    else:
        for edge in ("total", "distribution"):
            for k in ("ridge", "kernel", "trees"):
                out.append({"id": f"mixed_{edge}_{k}", "family": "mixed", "interface": edge, "kind": k, "backend": b})
    return out


def make_candidate(spec):
    if spec["family"] == "black":
        return DirectBlackBoxNode(spec["backend"], spec["kind"])
    return HybridOrWhiteNode(backend=spec["backend"], edge=spec["interface"], kind=spec["kind"])
