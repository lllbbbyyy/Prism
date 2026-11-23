"""On-demand 0 / 1 / 2 LLM-call construction loop over the Feature-Model Graph."""
from __future__ import annotations

import numpy as np

from .agents import (
    DIAGNOSIS_PROMPT,
    EXPERIMENT_PROMPT,
    SINGLE_AGENT_PROMPT,
    SYSTEM_COMMON,
    LLMClient,
    build_verified_state,
    validate_hypothesis,
)
from .graph import make_candidate, task_backend
from .identify import evaluate_candidates, select_path
from .samplers import acquire_batch
from .standalone import compile_model, export_artifact, predict as standalone_predict

CONTROLLERS = ("prism", "fixed_dual_role", "single_agent", "concise_state", "adaptive_rules")


def allowed_samplers_for(task: str) -> list[str]:
    b = task_backend(task)
    if b == "accelerator":
        return ["coverage", "joint", "single_axis"]
    return ["coverage", "pairs"]


def adapter_tool_summary(task: str) -> str:
    b = task_backend(task)
    if b == "accelerator":
        return (
            "scalar merges read timing; ports preserves per-port input/weight timings; all contains full "
            "public descriptors. white_read omits output SRAM stalls; mixed retains read_max. "
            "joint pairs wide read bandwidth with small output SRAM; single_axis sweeps one knob; coverage spreads inputs."
        )
    if b == "sparse":
        return (
            "total omits row distribution; distribution retains per-row statistics and tile counts. "
            "mixed scales by MACs/units. pairs fixes shape/nnz/resources/tiling and varies row skew; coverage spreads inputs."
        )
    return (
        "scalar omits routed link-load distribution; route retains link loads; all includes traffic indicator. "
        "pairs fixes injection load and varies traffic pattern; coverage spreads inputs."
    )


def default_fallback_sampler(task: str, evidence_table: dict, pool: list[dict]) -> str:
    b = task_backend(task)
    if b == "accelerator":
        has_small_buf = any(r["white"]["output_capacity"] < r["white"]["output_words"] for r in pool)
        return "joint" if has_small_buf else "coverage"
    return "coverage" if evidence_table["diagnosis"]["certificates"] else "pairs"


def route_step(controller: str, reached_threshold: bool, evidence_table: dict) -> str:
    if controller == "fixed_dual_role":
        return "2-Call"
    if controller == "single_agent":
        return "1-Call-Merged"
    if reached_threshold:
        return "0-Call"
    if evidence_table["diagnosis"]["certificates"]:
        return "1-Call"
    return "2-Call"


def construct_model(
    task: str,
    pool: list[dict],
    batches: tuple[int, ...] = (8, 16, 32),
    seed: int = 87001,
    controller: str = "prism",
    llm_config: str | None = None,
    family: str | None = None,
    prune_clashes: bool = True,
) -> dict:
    """Run batched offline Feature-Model Graph construction and export a standalone predictor."""
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller '{controller}'. Expected one of {CONTROLLERS}.")

    llm = LLMClient(llm_config) if (controller != "adaptive_rules" and llm_config) else None
    observed = acquire_batch(task, pool, [], batches[0], mode="random", seed=seed)
    samplers = allowed_samplers_for(task)
    tools_desc = adapter_tool_summary(task)
    history = []

    for next_budget in batches[1:]:
        ev = evaluate_candidates(task, observed)
        chosen, reached = select_path(ev, family=family, prune_clashes=prune_clashes)
        path_kind = route_step(controller, reached, ev)
        fallback = "coverage" if reached else default_fallback_sampler(task, ev, pool)
        step_calls = 0

        if controller == "adaptive_rules" or path_kind == "0-Call" or llm is None:
            chosen_sampler = "coverage" if path_kind == "0-Call" else fallback
        else:
            compact_cv = [
                {"id": c["spec"]["id"], "edge": c["spec"]["interface"], "mape": round(c["mape"], 4), "p90": round(c["p90"], 4)}
                for c in ev["candidates"]
                if c["status"] == "ok"
            ]
            payload = {
                "task": task,
                "tools": tools_desc,
                "allowed_samplers": samplers,
                "spent": len(observed),
                "acquire_to": next_budget,
                "active_path": chosen["spec"],
                "clash_diagnosis": ev["diagnosis"],
                "candidate_cv": compact_cv,
                "observations": [{"id": r["id"], "x": r["x"], "latency": r["result"]["value"]} for r in observed],
            }
            default_hyp = {
                "component": "response",
                "issue": "fit" if ev["diagnosis"]["certificates"] else "coverage",
                "evidence_ids": [chosen["spec"]["id"]],
                "hypothesis": "Improve validation coverage on non-clashing edges.",
                "interface": "auto",
            }
            if controller == "single_agent":
                raw = llm.invoke_json(SYSTEM_COMMON + " " + SINGLE_AGENT_PROMPT, payload)
                step_calls += 1
                chosen_sampler = raw.get("sampler") if raw.get("sampler") in samplers else fallback
            else:
                proposal = default_hyp
                if path_kind == "2-Call":
                    raw_hyp = llm.invoke_json(SYSTEM_COMMON + " " + DIAGNOSIS_PROMPT, payload)
                    step_calls += 1
                    if validate_hypothesis(raw_hyp, observed, ev):
                        proposal = raw_hyp
                payload["contract"] = proposal
                if controller != "concise_state":
                    payload["verified_state"] = build_verified_state(ev, proposal)
                raw_exp = llm.invoke_json(SYSTEM_COMMON + " " + EXPERIMENT_PROMPT, payload)
                step_calls += 1
                chosen_sampler = raw_exp.get("sampler") if raw_exp.get("sampler") in samplers else fallback

        observed = acquire_batch(task, pool, observed, next_budget, mode=chosen_sampler, seed=seed)
        history.append({
            "budget": next_budget,
            "route": path_kind,
            "llm_calls": step_calls,
            "sampler": chosen_sampler,
            "selected_before": chosen["spec"]["id"],
            "reached_threshold": reached,
        })

    final_ev = evaluate_candidates(task, observed)
    winner, final_reached = select_path(final_ev, family=family, prune_clashes=prune_clashes)
    fitted = make_candidate(winner["spec"]).fit(observed)
    artifact = export_artifact(task, winner["spec"], fitted)
    compiled = compile_model(artifact)
    np.testing.assert_allclose(
        standalone_predict(compiled, [r["x"] for r in observed]),
        fitted.predict(observed),
        rtol=1e-6,
        atol=1e-7,
    )
    return {
        "task": task,
        "selected": winner,
        "reached_threshold": final_reached,
        "model": fitted,
        "artifact": artifact,
        "compiled_artifact": compiled,
        "observed": observed,
        "history": history,
        "total_llm_calls": llm.calls if llm else 0,
        "total_tokens": llm.total_tokens if llm else 0,
    }
