---
title: CloudSREEnv
emoji: 🛠️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
- openenv
- sre
- multi-agent
- llm-training
- root-cause-analysis
---

# CloudSREEnv 🛠️

**OpenEnv × Meta Developers Hackathon Submission**

CloudSREEnv is a multi-agent SRE incident-response environment for training and evaluating LLM agents on realistic production failures. The agent must diagnose partial-observability incidents, avoid symptom-fixing traps, coordinate across roles, and execute the correct remediation through structured tools.

---

## Submission Artifacts

| Artifact | Link |
|---|---|
| Hugging Face Space | **TBD: add HF Space URL** |
| Colab Training Notebook | **TBD: add Colab URL** |
| Mini-blog / Demo Video | **TBD: add HF blog or YouTube URL** |
| Trained SFT Adapter | **TBD: add model/adapters URL if uploaded** |

---

## Why This Matters

Modern LLM agents can often produce plausible incident-response text, but they struggle with the hard part of SRE work: maintaining state across tools, separating symptoms from root causes, coordinating handoffs, and knowing when an incident is actually safe to close.

`CloudSREEnv` turns that into an OpenEnv training problem. The agent sees partial observations, uses structured tools, receives deterministic rewards, and must progress through an incident workflow:

```text
Incident Commander -> L1 Triage -> evidence gathering -> root cause report -> L2 remediation -> closure
```

This directly targets two OpenEnv Hackathon themes:

- **Theme #1: Multi-Agent Interactions**: IC, L1, and L2 agents coordinate through explicit messages and role boundaries.
- **Theme #3: World Modeling / Professional Tasks**: the environment simulates dynamic infrastructure state, tool feedback, causal failures, and delayed success criteria.

---

## What The Agent Learns

The environment is designed to teach an LLM to:

- diagnose before remediating,
- avoid fixing the service that only shows the symptom,
- gather enough evidence before reporting,
- respect role-based access control,
- use the correct tool for the root cause,
- close incidents only after RCA or remediation is complete.

The current training approach is **SFT-first**:

```text
Base model -> SFT on expert trajectories -> strict evaluation -> optional GRPO refinement
```

GRPO is intentionally not the main claim right now. SFT teaches the workflow reliably; GRPO can later refine robustness, efficiency, and reward optimization.

---

## Project Structure

```text
CloudSREEnv/
├── server/
│   ├── app.py             # OpenEnv environment, simulator, rewards, terminal graders
│   └── __init__.py
├── prompts.py             # Shared IC / L1 / L2 prompts
├── train.py               # SFT-first entry point
├── train_sft.py           # Transformers/PEFT SFT trainer
├── train_unsloth.py       # Colab-friendly Unsloth 4-bit LoRA SFT trainer
├── inference.py           # BASE / SFT / TRAINED evaluator
├── scripts/               # Benchmark and plot helpers
├── openenv.yaml           # OpenEnv task metadata
├── Dockerfile             # Hugging Face Space container
└── README.md
```

---

## Environment Design

`CloudSREEnv` is an OpenEnv-style simulator with:

- `reset(task_id)` to start an incident,
- `step(action)` to execute structured actions,
- `state()` / `GET /state` for dashboard/debug visibility,
- observations from logs and service tables,
- dense rewards from composable rubrics,
- deterministic terminal grading for task success.

### Roles

| Role | Responsibility |
|---|---|
| `IC` | Delegates investigation/remediation and closes the incident |
| `L1_Triage` | Read-only diagnosis using `LIST_SERVICES` and `GET_LOGS` |
| `L2_DB_SME` | Applies infrastructure fixes like restart, scale, config update, or replica repair |

### Action Space

| Action | Purpose |
|---|---|
| `LIST_SERVICES` | Inspect service status, latency, memory, CPU, and cache epoch |
| `GET_LOGS(service_id)` | Fetch logs for a specific service |
| `MESSAGE_CHANNEL(target, message)` | Communicate between IC, L1, and L2 |
| `RESTART(service_id)` | Recover a crashed service |
| `SCALE(service_id, cpu_value)` | Increase CPU allocation for saturation |
| `UPDATE_CONFIG(service_id, memory_limit_mb)` | Cap memory for noisy-neighbor remediation |
| `REPAIR_REPLICA(service_id)` | Repair a stale cache replica after split-brain |
| `CLOSE_INCIDENT` | Close only when success criteria are satisfied |

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /reset` | Start a task/scenario |
| `POST /step` | Execute one structured action and return observation, reward, done, info |
| `GET /state` | Return full simulator state for dashboards/debugging |

`/state` is intended for visualization and debugging, not as an agent tool. The agent should solve incidents through observations returned by `step`.

---

## Scenarios And Tasks

| Task | Difficulty | Scenario | Success Condition |
|------|-----------|----------|-------------------|
| `task1_tls_certificate_rca` | Easy | Login failures caused by expired upstream TLS certificate | L1 checks `auth-api` logs, reports RCA, IC closes without remediation |
| `task2_self_healing` | Medium | `payment-db` CrashLoopBackOff / OOMKilled | L1 identifies crash, IC delegates, L2 restarts `payment-db`, IC closes |
| `task3_latency_resolution` | Hard | `auth-api` CPU saturation causing latency | L1 diagnoses CPU saturation, L2 scales `auth-api` to `>=2048`, IC closes |
| `task4_noisy_neighbor` | Hard | `payment-db` is slow because `notification-worker` consumes ~8000MB | Agent must avoid fixing victim `payment-db`; L2 caps `notification-worker` memory to `<=2048MB` |
| `task5_cache_split_brain` | Hard | `checkout-api` sees cart/session mismatch due to stale cache replica | L1 checks checkout and both cache nodes, compares epochs, L2 repairs `session-cache-replica` |

### Why Task 4 And 5 Matter

These tasks are designed as **victim-vs-root-cause traps**:

- Task 4: `payment-db` looks slow, but `notification-worker` is the root cause.
- Task 5: `checkout-api` looks broken, but stale cache replica state is the root cause.

This makes the environment harder to game than simple “restart the failing service” benchmarks.

---

## Reward Design

The reward logic is implemented in `server/app.py` using composable signals:

- **Tool-use reward** for useful diagnostic actions.
- **Collaboration reward** for correct role handoff.
- **RBAC enforcement** so L1 cannot mutate infrastructure.
- **Principle reward** for new evidence, confidence gain, correct fixability classification, and correct remediation timing.
- **Duplicate-action penalty** to discourage repeated useless actions.
- **Terminal graders** that verify the final environment state, not just message text.

This makes the environment suitable for both supervised training and future RL refinement.

---

## Training Pipeline

### Recommended Flow

```text
Qwen/Qwen2.5-3B-Instruct
    -> LoRA SFT on expert SRE trajectories
    -> strict evaluation without controller guardrails
    -> optional GRPO reward refinement later
```

SFT is used first because the model must learn the workflow structure before RL rewards can reliably refine it.

### Standard SFT Training

```bash
python train.py
```

This launches the plain Transformers/PEFT SFT path (`train_sft.py`).

### Colab Low-Memory Training With Unsloth

For Colab GPUs, use the Unsloth 4-bit LoRA path:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
USE_UNSLOTH=1 python train.py
```

`train_unsloth.py` trains the same expert SRE action dataset as `train_sft.py`, but loads Qwen through Unsloth's 4-bit LoRA path to reduce VRAM use. It saves the adapter to the same location:

```text
./sft_sre_model/final
```

### Strict Evaluation

```bash
python inference.py
```

`inference.py` supports:

```bash
EVAL_MODE = "BASE"     # raw base model
EVAL_MODE = "SFT"      # SFT adapter from ./sft_sre_model/final
EVAL_MODE = "TRAINED"  # optional GRPO adapter from ./grpo_sre_model/final
```

Strict mode is enabled by default:

```python
STRICT_EVAL = True
```

That means inference does not force IC/L1/L2 transitions. The model must produce the actions itself.

To enable optional non-strict helper guardrails for debugging, set `STRICT_EVAL = False` in `inference.py`.

### Benchmark And Plot Artifacts

Run the full BASE/SFT benchmark:

```bash
python scripts/evaluate_benchmarks.py
```

This writes:

```text
episode_traces/benchmark_results.json
```

Generate README plot assets:

```bash
python scripts/generate_readme_assets.py
```

This reads:

```text
episode_traces/benchmark_results.json
episode_traces/unsloth_training_metrics.json
```

and writes plot images under:

```text
assets/
```

---

## Results

### Strict Evaluation Summary

Fill this table after rerunning `BASE` and retrained `SFT` on the current 5-task version.

| Task | BASE Strict | SFT Strict | Notes |
|---|---:|---:|---|
| TLS Certificate RCA | TBD | TBD | RCA-only, no local remediation |
| Self-Healing | TBD | TBD | Restart `payment-db` |
| Latency Resolution | TBD | TBD | Scale `auth-api` |
| Noisy Neighbor | TBD | TBD | Cap `notification-worker`, not `payment-db` |
| Cache Split-Brain | TBD | TBD | Repair stale `session-cache-replica` |
| **Pass Rate** | **TBD** | **TBD** | Use strict mode only |

Previously, the SFT model passed the original 3-task strict evaluation. The current 5-task benchmark should be rerun after retraining with the updated SFT data.

### Expected Plot Assets

The plot generator produces:

- `assets/sft_training_loss.png`
- `assets/reward_progress.png`

`sft_training_loss.png` shows supervised loss over training steps.  
`reward_progress.png` shows a training-time SFT reward proxy from Unsloth logs, computed as `exp(-loss)`.

The reward proxy is separate from the OpenEnv terminal reward used during strict evaluation.

---

## Outputs And Dashboard Support

Normal inference does **not** write per-episode trace files by default. This prevents `episode_traces/` from filling up with timestamped files after every run.

The main benchmark artifact is:

```bash
episode_traces/benchmark_results.json
```

Generate it explicitly with:

```bash
python scripts/evaluate_benchmarks.py
```

If you want optional per-task trace files for debugging, enable them manually:

```bash
WRITE_EPISODE_TRACES=1 python inference.py
```

When enabled, only stable latest files are written and overwritten:

```bash
episode_traces/latest_<task_id>.json
episode_traces/latest_<task_id>.md
```

Optional traces show:

- agent role per step,
- raw model reply,
- parsed action,
- environment observation,
- reward breakdown,
- success/failure state.

For live dashboards, use:

```bash
GET /state
```

This returns the current task, service table, action history, incident state, and scenario profile.

---

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python inference.py
```

For Unsloth training in Colab:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
USE_UNSLOTH=1 python train.py
```

For Docker / Hugging Face Space:

```bash
docker build -t cloudsreenv .
docker run -p 8000:8000 cloudsreenv
```

---

## Hackathon Judging Alignment

| Criterion | How CloudSREEnv Addresses It | Status |
|---|---|---|
| Environment Innovation (40%) | Multi-agent SRE simulator with victim-vs-root-cause traps, dynamic state, and deterministic graders | Strong |
| Storytelling (30%) | README explains problem, environment, agent workflow, and demo path | Needs final artifact links |
| Showing Improvement (20%) | BASE vs SFT strict eval, benchmark JSON, and generated README plots | Needs current 5-task rerun |
| Reward + Pipeline (10%) | Coherent composable reward design plus SFT-first training pipeline | Strong |

---

## Submission Checklist

- [ ] Add Hugging Face Space URL.
- [ ] Add Colab training notebook URL.
- [ ] Add mini-blog or YouTube video URL.
- [ ] Rerun SFT training after Task 4/5 additions.
- [ ] Run strict `BASE` benchmark across all 5 tasks.
- [ ] Run strict `SFT` benchmark across all 5 tasks.
- [ ] Run held-out strict benchmark.
- [ ] Generate final plot images under `assets/`.
- [x] Confirm `openenv.yaml` lists all 5 tasks.
- [ ] Confirm Hugging Face Space starts successfully.

---

## What You Still Need To Fill

- `TBD: add HF Space URL`
- `TBD: add Colab URL`
- `TBD: add HF blog or YouTube URL`
- `TBD: trained adapter URL if uploaded`
- BASE strict 5-task results
- SFT strict 5-task results after retraining
- Held-out strict results
- Final plot images in `assets/` after rerunning the 5-task benchmark

---

Built for the OpenEnv × Meta Developers Hackathon 2026.
