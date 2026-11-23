"""Analytical node and feature edges (scalar / route / all) for XY-mesh interconnects (BookSim 2.0)."""
import numpy as np


def noc_route_quantities(x):
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
            raise ValueError(f"Unsupported traffic pattern: {x['traffic']}")
        for destination, p in destinations:
            dx, dy = destination % size, destination // size
            a, b = sx, sy
            if source == destination:
                self_prob += p / n
            while a != dx:
                step = 1 if dx > a else -1
                edges[(a, b, a + step, b)] = edges.get((a, b, a + step, b), 0.0) + p
                a += step
                hops += p / n
            while b != dy:
                step = 1 if dy > b else -1
                edges[(a, b, a, b + step)] = edges.get((a, b, a, b + step), 0.0) + p
                b += step
                hops += p / n
    loads = [
        edges.get((a, b, a + da, b + db), 0.0)
        for a in range(size)
        for b in range(size)
        for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if 0 <= a + da < size and 0 <= b + db < size
    ]
    return {
        "mean_hops": float(hops),
        "max_link_load": float(max(loads)),
        "link_std": float(np.std(loads)),
        "self_probability": float(self_prob),
        "rho": float(x["load"] * max(loads)),
    }


def noc_features(rows, edge="route"):
    out = []
    for r in rows:
        x, w = r["x"], r["white"]
        rho = x["load"] * x["size"] / 2.0 if edge == "scalar" else w["rho"]
        q = rho / max(1.0 - rho, 0.02)
        z = [x["load"], w["mean_hops"], rho, q, q * q, q ** 3, q ** 5]
        if edge in ("route", "all"):
            z += [w["max_link_load"], w["link_std"], w["self_probability"]]
        if edge == "all":
            z += [float(x["traffic"] == "transpose")]
        out.append(z)
    return np.asarray(out, dtype=float)
