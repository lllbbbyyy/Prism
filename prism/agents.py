"""Diagnosis and Experiment LLM agents with resilient JSON transport and validity checks."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
import requests

SYSTEM_COMMON = (
    "Construct a component latency cost model for AI accelerator design from public primitives. "
    "Reference threshold: training CV MAPE <= 5% and P90 <= 10%, then minimize single-query runtime. "
    "Equal MAC counts alone do not prove feature loss; only equal executable feature inputs with "
    "unequal positive latencies confirm a feature clash. Return exactly one JSON object."
)

DIAGNOSIS_PROMPT = (
    "You are the Diagnosis Agent. Inspect candidate CV errors and high-error observed pairs. "
    'Return {"component": "response", "issue": "interface|fit|coverage|satisfied", '
    '"evidence_ids": [1 or 2 valid observed/candidate IDs], "hypothesis": "...", '
    '"interface": "scalar|ports|all|route|total|distribution|auto", "rationale": "..."}.'
)

EXPERIMENT_PROMPT = (
    "You are the Experiment Agent receiving verified numerical state and any pending hypothesis. "
    'Select one sampling strategy from allowed_samplers. Return '
    '{"sampler": "...", "contract_component": "response", "rationale": "..."}.'
)

SINGLE_AGENT_PROMPT = (
    "You are a single generalist agent handling both bottleneck diagnosis and sampling selection. "
    'Inspect candidate CV errors and observed samples, then return '
    '{"component": "response", "issue": "interface|fit|coverage|satisfied", '
    '"evidence_ids": [1 or 2 valid IDs], "hypothesis": "...", '
    '"interface": "scalar|ports|all|route|total|distribution|auto", '
    '"sampler": "...", "contract_component": "response", "rationale": "..."}.'
)


class LLMClient:
    """OpenAI-compatible JSON client with exponential backoff and malformed-JSON retries."""

    def __init__(self, config_path: str | Path, max_retries: int = 3):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.api_url = cfg["api_url"]
        self.api_key = cfg["api_key"]
        self.model = cfg.get("model", "deepseek-chat")
        self.max_retries = max_retries
        self.calls = 0
        self.total_tokens = 0

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.calls += 1
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=(10, 75),
                )
                resp.raise_for_status()
                data = resp.json()
                self.total_tokens += int(data.get("usage", {}).get("total_tokens", 0))
                text = data["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (2 ** attempt))
        return {"transport_failure": str(last_err) if last_err else "invalid_json_response"}


def validate_hypothesis(hyp: dict, observed: list[dict], evidence_table: dict) -> bool:
    valid_ids = {r["id"] for r in observed} | {c["spec"]["id"] for c in evidence_table["candidates"]}
    forbidden = set(evidence_table["diagnosis"]["forbidden_interfaces"])
    return (
        isinstance(hyp, dict)
        and hyp.get("component") == "response"
        and hyp.get("issue") in ("interface", "fit", "coverage", "satisfied")
        and isinstance(hyp.get("evidence_ids"), list)
        and 0 < len(hyp["evidence_ids"]) <= 2
        and all(isinstance(i, str) and i in valid_ids for i in hyp["evidence_ids"])
        and isinstance(hyp.get("hypothesis"), str)
        and hyp.get("interface") in ("scalar", "ports", "all", "route", "total", "distribution", "auto")
        and hyp.get("interface") not in forbidden
    )


def build_verified_state(evidence_table: dict, proposal: dict) -> dict:
    """Explicitly separate numerically confirmed clashes from unverified LLM hypotheses."""
    diag = evidence_table["diagnosis"]
    has_clash = bool(diag["certificates"])
    status = "confirmed" if has_clash else ("hypothesis" if proposal.get("issue") == "interface" else "observed")
    return {
        "component": "response",
        "status": status,
        "proposal": proposal,
        "forbidden_interfaces": diag["forbidden_interfaces"],
        "certificates": diag["certificates"],
        "action": "refine_interface_then_validate_fit" if has_clash else "test_hypothesis_or_improve_fit",
    }
