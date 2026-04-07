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

## 💡 Motivation
Standard LLM benchmarks often focus on isolated text tasks. `CloudSREEnv` addresses **Real-World Utility** (30% scoring weight) by simulating complex microservice dependencies. It forces agents to move beyond simple "restarts" and perform true **Root Cause Analysis** by tracing performance bottlenecks across a distributed system.

---

## 🏗️ Project Structure
This project follows the canonical OpenEnv multi-mode deployment structure:
```text
CloudSREEnv/
├── server/
│   ├── app.py        # Core logic (MockCloud + CloudSREEnv)
│   └── __init__.py   # Python package marker
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

*Example: If `payment-db` is slow, `auth-api` will report "Upstream Slowness" in its logs, requiring the agent to trace the bottleneck to the database.*

---

## 🎯 Scenarios & Tasks

| Task | Difficulty | Scenario | Success Condition |
|------|-----------|----------|-------------------|
| **Status Audit** | Easy | Cascading Crash | Trace 503 error from API to DB logs |
| **Self-Healing** | Medium | DB Recovery | Restore DB health to fix entire cluster |
| **RCA & Scaling** | Hard | Latency Bottleneck | Scale the DB (root cause) to fix worker latency |

---

## 🕹️ Observation & Action Spaces

### Action Space (Inputs)
- `LIST_SERVICES`: Returns a status table of all pods.
- `GET_LOGS(service_id)`: Fetches logs (includes cascading error propagation).
- `RESTART(service_id)`: Restores a crashed pod.
- `SCALE(service_id, cpu_value)`: Modifies CPU (Success: `cpu >= 2048`).

### Observation Space (Outputs)
- `text_output`: Human-readable terminal logs and status tables.
- `structured_data`: Pydantic-validated `ServiceMetrics` (Status, CPU, Memory, Latency).

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
export API_BASE_URL="https://aipipe.org/v1"
export MODEL_NAME="meta-llama/Llama-3-70b-Instruct"
export HF_TOKEN="your_scaler_jwt"

python inference.py
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
