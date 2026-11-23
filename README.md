# Prism: Agent-Collaborative Hybrid Cost Model Design for AI Accelerators

**Prism** automates hybrid cost model design across diverse AI accelerator modules for fast design space exploration (DSE). Rather than manually fixing one analytical formula, feature set, and regression model per architecture, Prism organizes first-order analytical priors, multi-granularity feature edges (`coarse`, `refined`, `full`), and white-box, hybrid, and black-box predictor nodes into a unified **Feature-Model Graph**:

1. **Pre-Fit Feature Clash Pruning**: Before fitting regressors, Prism scans observed sample pairs sharing identical simplified features ($\phi(x_1) = \phi(x_2)$ with $0 < y_1 < y_2$) and rules out overly coarse edges whenever $\mathrm{Err}_{\min} = (y_2 - y_1) / (y_2 + y_1) > \epsilon_{\mathrm{avg}}$.
2. **Accuracy-First, Speed-Aware Path Selection**: Using reference cross-validation threshold $\boldsymbol{\epsilon} = (5\%, 10\%)$, Prism picks the lowest-error path within $(1 + \delta)c_{\min}$ of the fastest qualifying query time once $\boldsymbol{\epsilon}$ is met, or retains the lowest-error path otherwise.
3. **On-Demand Batched Agent Routing (`0 / 1 / 2` LLM Calls)**: Prism skips LLM calls once $\boldsymbol{\epsilon}$ is reached (`0-Call`), makes `1` call to the Experiment Agent when a feature clash already explains the bottleneck (`1-Call`), and wakes both Diagnosis and Experiment Agents only on unresolved errors (`2-Call`), keeping unverified LLM `hypothesis` proposals distinct from numerically `confirmed` clashes.
4. **Standalone Pure-NumPy Export (`predictor.py`)**: At budget completion ($B = 32$), the selected path serializes into a standalone NumPy predictor requiring **0 LLM calls, 0 simulator invocations, and 0 ML framework dependencies** during online DSE queries.

---

## Repository Structure

```text
Prism/
├── prism/
│   ├── __init__.py
│   ├── adapters/           # Analytical nodes & feature edges (Dense, NoC, Sparse)
│   ├── learners.py         # Portfolio of 9 regression configurations
│   ├── graph.py            # Feature-Model Graph candidate definitions & fitting
│   ├── identify.py         # Pre-fit feature clash pruning & speed-aware path selection
│   ├── samplers.py         # Deterministic public-parameter samplers (Coverage, Paired, SingleAxis, Joint)
│   ├── agents.py           # Diagnosis & Experiment LLM agents + resilient JSON client & state checks
│   ├── orchestrator.py     # Batched construction loop (Prism + 4 baseline controllers & model spaces)
│   └── standalone.py       # Pure-NumPy standalone runtime & Step-6 bundle serializer (`predictor.py`)
├── configs/
│   └── llm.example.json    # Example OpenAI-compatible LLM endpoint configuration
├── data/
│   ├── train_pools/        # Pre-evaluated configuration pools for offline construction runs
│   ├── test_144.jsonl      # 144 disjoint unseen test configurations across all 6 tasks (24/task)
│   ├── exported_models/    # Exported standalone NumPy model artifacts for all 6 tasks
│   └── summaries/          # Evaluation accuracy, paired query speedup & 12-build agent cost summaries
├── examples/
│   └── quickstart_demo.py  # Walkthrough of clash pruning, path matching, and fast standalone querying
├── scripts/
│   ├── build_cost_model.py # Build & export a standalone cost model across tasks, controllers & ablations
│   └── verify_results.py   # Replay standalone predictors on all 144 unseen test points & print tables
└── tests/
    └── test_prism.py       # Unit tests for clash bounds, selection, standalone parity & unseen replay
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Replay Standalone Predictors & Reproduce Main Tables (Zero API / Simulator Required)

Replay the exported pure-NumPy predictors across all six tasks (`Dense-S`, `Dense-L`, `NoC-S`, `NoC-L`, `Sparse-S`, `Sparse-L`) on the 144 unseen test configurations and display the baseline accuracy, speedup, and agent-routing tables:

```bash
python scripts/verify_results.py
```

### 3. Run End-to-End Walkthrough & Unit Tests

```bash
python examples/quickstart_demo.py
python -m unittest discover -s tests -v
```

### 4. Build & Export a Standalone Cost Model (`predictor.py`)

Run offline Feature-Model Graph construction on any task (`Dense-S`, `Dense-L`, `NoC-S`, `NoC-L`, `Sparse-S`, `Sparse-L`). Without an API key, use `--controller adaptive_rules`; to run live LLM agents (`prism`, `fixed_dual_role`, `single_agent`, `concise_state`), copy `configs/llm.example.json` to `configs/llm.json` and pass `--llm-config configs/llm.json`:

```bash
# Deterministic offline build + standalone `predictor.py` bundle export (no API key needed)
python scripts/build_cost_model.py --task Dense-L --budget 32 --seed 87002 \
    --controller adaptive_rules --export-dir out/dense_l_bundle

# Live on-demand agent-collaborative build
python scripts/build_cost_model.py --task Dense-L --budget 32 --seed 87002 \
    --controller prism --llm-config configs/llm.json --export-dir out/dense_l_prism

# Model-structure baseline builds (Direct Black-Box or Predefined Hybrid)
python scripts/build_cost_model.py --task Sparse-S --budget 32 --model-space predefined_hybrid
```

---

## Evaluated Hardware Tasks & Summary Results ($B = 32$, $144$ Unseen Test Points)

| Category | Tasks | Reference Simulator | Prism Mean MAPE / P90 | Median Query Speedup | Selected Feature-Model Path |
|---|---|---|---|---|---|
| **Dense Systolic Arrays** | `Dense-S`, `Dense-L` | SCALE-Sim v3 | **2.08%** / 4.13% | **6,683x** | `S`: Prior $\to$ White-Box / `L`: `ports` $\to$ Hybrid Trees |
| **On-Chip Interconnects** | `NoC-S`, `NoC-L` | BookSim 2.0 | **0.86%** / 1.83% | **661x** | `route` $\to$ Hybrid Ridge |
| **Sparse Accelerators** | `Sparse-S`, `Sparse-L` | STONNE | **7.14%** / 17.31% | **220x** | `dist` $\to$ Black-Box Ridge |

Across 12 independent builds from two random initial sample sets (Trial 1 `seed=87001` and Trial 2 `seed=87002`), Prism reduces total LLM calls by **66.7%** (`16` vs. `48`) and token usage by **67.7%** (`85.7K` vs. `265.0K`) compared with fixed dual-role polling.

---

## License

Released under the [MIT License](LICENSE).
