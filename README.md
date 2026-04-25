---
title: CloudSREEnv
emoji: 🛠️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
- openenv
---

# CloudSREEnv 🛠️

> **OpenEnv × Meta Developers Hackathon Submission**
> An advanced, OpenEnv-compliant SRE simulator featuring cascading failure logic and Root Cause Analysis (RCA) challenges.

---

## 🔗 Submission Links
- **Hugging Face Space:** TODO: add deployed Space URL
- **Colab Training Notebook:** TODO: add shared Colab URL
- **Mini Blog / Video / Slides:** TODO: add presentation URL

---

## 💡 Motivation
Most LLM agents are good at taking the obvious action: restart the failing service, scale the slow API, or close once the logs look clean. Real SRE work is harder. The symptom is often not the root cause, the right specialist may need to be delegated, and premature remediation can make the incident worse.

`CloudSREEnv` is a multi-agent SRE training environment for teaching LLMs **root-cause analysis over a partially observable cloud system**. It simulates realistic incidents where agents must inspect services, read logs, coordinate between roles, avoid trap actions, and apply the correct fix only after collecting evidence.

The environment targets three hackathon themes:
- **Multi-Agent Interactions:** Incident Commander, L1 Triage, and L2 Database SME coordinate through an incident channel.
- **Long-Horizon Planning:** Later tasks require multi-step evidence gathering before remediation.
- **World Modeling / Professional Tasks:** Agents interact with a dynamic microservice simulator with cascading failures, resource contention, and consistency bugs.

---

## 🏗️ Project Structure
This project follows the canonical OpenEnv multi-mode deployment structure:
```text
CloudSREEnv/
├── server/
│   ├── app.py        # Core logic (MockCloud + CloudSREEnv)
├── pyproject.toml    # Project metadata & entry points
├── uv.lock           # Deterministic dependency lockfile
├── openenv.yaml      # Environment metadata
├── inference.py      # RCA-optimized Baseline Agent
├── Dockerfile        # uv-based container definition
└── README.md         # Documentation
```

---

## 🔗 Service Dependency Graph
Unlike basic simulators, `CloudSREEnv` implements cascading failures. An issue at the base of the stack propagates upward:
- **auth-api** ⮕ depends on ⮕ **payment-db**
- **inventory-svc** ⮕ depends on ⮕ **payment-db**
- **notification-worker** ⮕ depends on ⮕ **inventory-svc**
- **checkout-api** ⮕ depends on ⮕ **session-cache-primary / session-cache-replica**

*Example: If `payment-db` is slow, `auth-api` will report "Upstream Slowness" in its logs, requiring the agent to trace the bottleneck to the database.*

---

## 🎯 Scenarios & Tasks

| Task | Difficulty | Scenario | Success Condition |
|------|-----------|----------|-------------------|
| **TLS Certificate RCA** | Easy | Certificate Expiry | Identify expired cert in auth-api logs (no fix needed) |
| **Self-Healing** | Medium | DB Recovery | Restore DB health to fix entire cluster |
| **RCA & Scaling** | Hard | Auth API CPU Bottleneck | Scale auth-api after detecting CPU saturation |
| **Noisy Neighbor** | Hard | Resource Contention | Limit notification-worker memory to restore payment-db latency |
| **Split-Brain Cache** | Expert | Cache Consistency | Repair stale session-cache-replica after detecting divergent cache epochs |

---

## 🧠 Agent Roles
- **Incident Commander (IC):** Owns coordination, delegates investigation and fixes, and closes incidents.
- **L1_Triage:** Read-only investigator that lists services, gathers logs, and reports root cause evidence.
- **L2_DB_SME:** Authorized remediator that can restart, scale, update config, or repair replicas.

This role split makes the environment more than a single-agent tool-calling benchmark: success requires routing work to the right specialist and respecting RBAC constraints.

---

## 🕹️ Observation & Action Spaces

### Action Space (Inputs)
- `LIST_SERVICES`: Returns a status table of all pods.
- `GET_LOGS(service_id)`: Fetches logs (includes cascading error propagation).
- `RESTART(service_id)`: Restores a crashed pod.
- `SCALE(service_id, cpu_value)`: Modifies CPU (Success: `cpu >= 2048`).
- `UPDATE_CONFIG(service_id, memory_limit_mb)`: Applies a memory limit to a service (Task 4 success: `notification-worker <= 2048MB`).
- `REPAIR_REPLICA(service_id)`: Resyncs a stale cache replica after split-brain (Task 5 success: repair `session-cache-replica`).
- `MESSAGE_CHANNEL(target, message)`: Passes findings or fix requests between agents.
- `CLOSE_INCIDENT`: Attempts terminal grading.

### Observation Space (Outputs)
- `text_output`: Human-readable terminal logs and status tables.
- `structured_data`: Pydantic-validated `ServiceMetrics` (Status, CPU, Memory, Latency).

---

## 🎯 Reward Design
CloudSREEnv uses dense, composable rewards so training receives signal before the final close step:

- Rewards useful diagnostics such as `LIST_SERVICES` and targeted `GET_LOGS`.
- Rewards collaboration through `MESSAGE_CHANNEL`.
- Enforces RBAC by penalizing unauthorized remediation.
- Penalizes duplicate actions, hallucinated services, premature closures, and trap fixes.
- Adds task-specific bonuses for correct RCA evidence, such as cache epoch comparison in task 5.
- Gives final completion bonuses only when deterministic task criteria are satisfied.

The reward logic is deterministic and implemented in `server/app.py`, while `train.py` adds workflow-aware GRPO reward shaping for multi-agent trajectories.

---

## 📈 Training Evidence
TODO: replace this section with final Colab results before submission.

| Evaluation | task1 | task2 | task3 | task4 | task5 |
|------------|-------|-------|-------|-------|-------|
| Base model | TODO | TODO | TODO | TODO | TODO |
| Trained model | TODO | TODO | TODO | TODO | TODO |

Planned artifacts:
- `assets/reward_curve.png`: reward over training steps.
- `assets/loss_curve.png`: GRPO loss over training steps.
- `assets/before_after_success.png`: base vs trained task success comparison.

Qualitative improvement to highlight: after training, the agent should stop chasing symptom services like `checkout-api` or `payment-db` and instead identify root causes such as split-brain cache replicas or noisy-neighbor memory pressure.

---

## 🚀 Quick Start (Local)

### 1. Setup with `uv`
```bash
pip install uv
uv sync
```

### 2. Run Environment Server
```bash
uv run server
# Serving on http://0.0.0.0:8000
```

### 3. Run RCA Agent
```bash
uv run python inference.py
```

### 4. Train with GRPO
```bash
uv run python train.py
```

---

## 🐳 Docker Deployment
This environment is containerized using `uv` for high-performance builds.

```bash
# Build
docker build -t cloudsreenv .

# Run
docker run -p 8000:8000 cloudsreenv
```

---

## ⚖️ Evaluation Compliance
- **Real-World Utility:** Models genuine microservice cascading failures.
- **Deterministic Graders:** 100% logic-based scoring (no LLM variance).
- **Spec Compliance:** Fully compatible with `openenv validate` and `openenv serve`.
- **Infrastructure:** Optimized for 8GB RAM / 2 vCPU limits.

---
*Built for the OpenEnv × Meta Developers Hackathon 2026*
