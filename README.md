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
Base model -> LoRA SFT on expert trajectories -> strict evaluation (no scripted controller)
```

Supervised fine-tuning teaches the multi-step incident workflow directly from demonstrations; evaluation uses the same environment reward and terminal graders as production runs.

---

## Project Structure

```text
CloudSREEnv/
├── server/
│   └── app.py             # OpenEnv environment, simulator, rewards, terminal graders
├── scripts/
│   ├── evaluate_benchmarks.py      # BASE/SFT in-template and held-out benchmark suite -> episode_traces/benchmark_results.json
│   ├── generate_readme_assets.py   # Loss + reward proxy plots for README -> assets/
│   ├── generate_benchmark_plot.py  # Benchmark pass-rate plots from benchmark_results.json -> assets/, training_logs/
│   └── unsloth_train_loss_epoch_plot.py  # Standalone plots from unsloth_training_metrics.json -> assets/
├── prompts.py             # Shared IC / L1 / L2 prompts
├── train.py               # SFT-first entry point
├── train_sft.py           # Transformers/PEFT SFT trainer
├── train_unsloth.py       # Colab-friendly Unsloth 4-bit LoRA SFT trainer
├── inference.py           # BASE vs SFT adapter evaluator
├── openenv.yaml           # OpenEnv task metadata
├── requirements.txt       # pip dependencies for local / Space runtime
├── pyproject.toml         # uv / project metadata
├── uv.lock                # Locked dependency versions (uv)
├── Dockerfile             # Hugging Face Space container
├── .dockerignore          # Docker build context exclusions
├── BLOG.md                # Longer-form writeup (companion to README)
├── assets/                # Optional; created by plot scripts (loss, reward, benchmark figures)
├── episode_traces/        # benchmark_results.json, optional traces, Unsloth metrics (created at runtime)
└── README.md
```

---

## Environment Design

`CloudSREEnv` is an OpenEnv-style simulator with:

- `reset(task_id)` to start an incident,
- `step(action)` to execute structured actions,
- `state()` / `GET /state` for dashboard/debug visibility (full simulator snapshot),
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

All step rewards are computed in `CloudSREEnv.step()` in `server/app.py`. Each `POST /step` returns a scalar in **[-1.0, 1.0]** plus a `breakdown` dictionary for logging and traces.

### How a step is scored

1. **Episode limits** — If the episode is already done or `max_steps` is exceeded, the step returns reward `0.0` and terminates.

2. **Hard duplicate block** — Before the action is recorded or executed, `_is_duplicate()` compares the action to prior steps (same type, agent, service, parameters where relevant). Duplicates return **-0.4** immediately (`duplicate_block`); `INVALID_FORMAT` and `CLOSE_INCIDENT` are exempt so parsing and closure logic always run.

3. **Rubric base (`_calculate_rubric`)** — Seeds `total_reward` and `breakdown`: small positive **tool_discovery** (+0.05) for `LIST_SERVICES` / `GET_LOGS`; **collaboration** (+0.1) for `MESSAGE_CHANNEL`; **authorized_action** (+0.2) when L2 runs a mutating action; **-0.5** for invalid JSON or RBAC violations (non-L2 attempting restart/scale/config/replica repair). A small **time_penalty** (`-0.02 * steps_taken`) encourages efficiency.

4. **Action execution** — Mutating actions may apply extra penalties on failure (e.g. unknown service, healthy restart, missing `cpu_value` / `memory_limit_mb`); successful state transitions update the simulated cloud.

5. **Principle shaping (`_calculate_principle_reward`)** — After execution, task-agnostic signals update `IncidentState`: rewards for new observations (e.g. first service list, first log fetch per service), **confidence** and **fixability** when logs match the scenario’s evidence service, penalties for irrelevant logs or messaging out of order (report without evidence, L2 delegation too early, wrong remediation vs `scenario_profile`). Remediation actions compare against the expected action/service/thresholds (e.g. scale CPU, memory cap, replica repair).

6. **`CLOSE_INCIDENT`** — Principle rewards for closing only when ready; `_grade_terminal_state()` checks task-specific success (evidence, correct fix, no trap actions). Success adds **task_completion** to the total; premature close adds **false_closure_penalty** (-0.5).

7. **Clamp** — `max(-1.0, min(1.0, total_reward))` before returning.

This is the **environment reward** used when the agent interacts with the simulator. It is separate from the **training-only** metric `exp(-loss)` logged during Unsloth SFT (see Results below), which does not drive `step()` and is only a proxy for how well the model fits the expert demonstrations.

---

## Training Pipeline

### Recommended Flow

```text
Qwen/Qwen2.5-3B-Instruct
    -> LoRA SFT on expert SRE trajectories
    -> strict evaluation without controller guardrails
```

SFT aligns the model with expert JSON actions and channel messages; strict evaluation then measures whether it follows the real environment rules without scripted transitions.

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
EVAL_MODE = "BASE"     # raw base model (no adapter)
EVAL_MODE = "SFT"      # LoRA adapter from ./sft_sre_model/final
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

Built for the OpenEnv × Meta Developers Hackathon 2026.
