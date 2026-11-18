# Prism: Agent-Collaborative Hybrid Cost Model Design for AI Accelerators

**Prism** is an automated hybrid cost model construction framework for AI accelerator design space exploration (DSE). Prism connects first-order analytical priors, multi-granularity feature edges (`coarse`, `refined`, `full`), and white-box, hybrid correction, and black-box predictor nodes in a **Feature-Model Graph**, prunes lossy feature edges via pre-fit feature clash bounds, and wakes diagnosis and experiment LLM agents on demand to guide batched sample acquisition.

Once offline construction completes within simulation budget $B$, the selected path exports as a standalone NumPy predictor (`prism/standalone.py`) requiring **0 LLM calls and 0 simulator invocations** during online DSE queries.

---

## Repository Structure

```text
Prism/
├── prism/
│   ├── __init__.py
│   ├── adapters/           # Analytical nodes & feature edges (Dense, NoC, Sparse)
│   ├── learners.py         # Portfolio of 9 regression configurations
│   ├── graph.py            # Feature-Model Graph candidate definitions & fitting
│   ├── identify.py         # Pre-fit feature clash pruning & speed-aware selection
│   ├── samplers.py         # Deterministic public-parameter samplers (Coverage, Paired, SingleAxis, Joint)
│   ├── agents.py           # Diagnosis & Experiment LLM agents + validity contract checks
│   ├── orchestrator.py     # Batched construction loop with 0 / 1 / 2 LLM-call routing
│   └── standalone.py       # Standalone NumPy predictor runtime & serializer
├── configs/
│   └── llm.example.json    # Example OpenAI-compatible LLM endpoint configuration
├── data/
│   ├── train_pools/        # Evaluated configuration pools for offline construction demos
│   ├── test_144.jsonl      # 144 disjoint unseen test configurations across 6 tasks
│   ├── exported_models/    # Exported standalone NumPy model artifacts for all 6 tasks
│   └── summaries/          # Sanitized evaluation & timing summaries
├── examples/
│   └── quickstart_demo.py  # End-to-end walkthrough of clash pruning, selection, and fast query
└── scripts/
    ├── build_cost_model.py # Run offline cost model construction on a target hardware task
    └── verify_results.py   # Replay exported standalone predictors on all 144 unseen test points
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Exported Standalone Predictors (Zero API / Simulator Required)

Replay the exported standalone NumPy predictors across all six tasks (`Dense-S`, `Dense-L`, `NoC-S`, `NoC-L`, `Sparse-S`, `Sparse-L`) on the 144 unseen test configurations:

```bash
python scripts/verify_results.py
```

### 3. Run Feature-Model Graph Construction

Run offline model construction on any task (`Dense-S`, `Dense-L`, `NoC-S`, `NoC-L`, `Sparse-S`, `Sparse-L`). Without an LLM key, pass `--controller adaptive_rules` (deterministic fallback); to enable live Diagnosis and Experiment LLM agents, copy `configs/llm.example.json` to `configs/llm.json` and pass `--llm-config configs/llm.json`:

```bash
# Deterministic offline build demo (no API key needed)
python scripts/build_cost_model.py --task Dense-L --budget 32 --controller adaptive_rules

# Live agent-collaborative build with OpenAI-compatible endpoint
python scripts/build_cost_model.py --task Dense-L --budget 32 --controller prism --llm-config configs/llm.json
```

### 4. Standalone Demo

```bash
python examples/quickstart_demo.py
```

---

## Evaluated Hardware Tasks

| Task | Simulator Backend | Hardware Block & Workload Regime |
|---|---|---|
| `Dense-S` | SCALE-Sim v3 | $8\times 8$ output-stationary systolic array GEMM $(48, 32, 64)$, $3.25\times$ read BW extrapolation |
| `Dense-L` | SCALE-Sim v3 | $16\times 16$ output-stationary systolic array GEMM $(64, 64, 128)$, small output SRAM stalls + $3.25\times$ read BW extrapolation |
| `NoC-S` | BookSim 2.0 | $2\times 2$ XY mesh interconnect ($2$ VCs, $4$-flit buffers), uniform & transpose traffic |
| `NoC-L` | BookSim 2.0 | $4\times 4$ XY mesh interconnect ($2$ VCs, $4$-flit buffers), multihop queuing contention |
| `Sparse-S` | STONNE | Sparse matrix kernel $(24, 24, 48)$ across $4$ row-skew patterns, multipliers, bandwidths, and tilings |
| `Sparse-L` | STONNE | Sparse matrix kernel $(32, 32, 64)$ across uneven row distributions and narrow tile shapes |
