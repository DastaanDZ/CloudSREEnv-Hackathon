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

> **OpenEnv × Meta Developers Hackathon**  
> An OpenEnv-compliant Kubernetes-style SRE simulator powered by an LLM agent.

---

## 💡 Motivation

Standard LLM benchmarks often focus on static text or simple games. `CloudSREEnv` addresses **Real-World Utility** (30% scoring weight) by simulating a task humans actually do: Site Reliability Engineering. It forces agents to bridge the gap between "reasoning" and "system action" by requiring them to interpret raw system logs and execute stateful commands to resolve infrastructure failures.

---

## Overview

`CloudSREEnv` simulates a small cloud cluster of four microservices. On each episode reset, the environment randomly triggers one of three real-world failure scenarios. An AI agent acting as a Site Reliability Engineer must diagnose and repair the cluster using typed actions.

```
auth-api  ·  payment-db  ·  inventory-svc  ·  notification-worker
```

---

## Project Structure

```
CloudSREEnv/
├── openenv.yaml      # Environment metadata & task definitions
├── env.py            # Core OpenEnv implementation (MockCloud + CloudSREEnv)
├── inference.py      # Baseline LLM agent (OpenAI-compatible)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md
```

---

## Scenarios & Tasks

The environment simulates a cluster of four microservices: `auth-api`, `payment-db`, `inventory-svc`, and `notification-worker`. On reset, it triggers one of three escalating scenarios:

| Task | Difficulty | Scenario | Success Condition |
|------|-----------|----------|-------------------|
| Status Audit | Easy | CrashLoopBackOff | Last action is `GET_LOGS` on failing service |
| Self-Healing | Medium | CrashLoopBackOff | Failing service status → `Running` |
| Latency Resolution | Hard | Performance Bottleneck | `cpu_allocated ≥ 2048` AND `latency_ms < 100` |

### Reward Shaping

| Event | Δ Reward |
|-------|---------|
| `LIST_SERVICES` (useful diagnostic) | +0.1 |
| Hallucinated service ID | −0.2 |
| Restart already-Running service | −0.5 |
| Task completion | +1.0 |

---

## 🕹️ Observation & Action Spaces

### Action Space (Inputs)
The agent interacts with the cluster using typed JSON actions:
- `LIST_SERVICES`: Returns a status table of all pods.
- `GET_LOGS(service_id)`: Fetches recent logs (e.g., OOMKilled errors).
- `RESTART(service_id)`: Restores a crashed service to `Running`.
- `SCALE(service_id, cpu_value)`: Modifies CPU allocation to resolve bottlenecks.

### Observation Space (Outputs)
Each step returns a structured observation:
- `text_output`: Human-readable terminal responses (logs, tables, or error messages).
- `structured_data`: A Pydantic-validated list of `ServiceMetrics` including Status, CPU, Memory, and Latency.

---

## 📈 Baseline Performance
The provided `inference.py` script serves as the reproducible baseline. 

**Expected Output Format:**
```text
[START] task=T1-status-audit env=CloudSRE model=gpt-5-nano
[STEP]  step=1 action=LIST_SERVICES reward=0.10 done=False error=none
[STEP]  step=2 action=GET_LOGS(payment-db) reward=1.00 done=True error=none
[END]   success=True steps=2 rewards=0.10,1.00
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the environment server

```bash
python env.py
# Serving on http://0.0.0.0:8000
```

### 3. Run the baseline agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."

python inference.py
```

Expected STDOUT:

```
[START] task=T1-status-audit env=CloudSRE model=gpt-4o-mini
[STEP]  step=1 action=LIST_SERVICES reward=0.10 done=False error=none
[STEP]  step=2 action=GET_LOGS(payment-db) reward=1.10 done=True error=none
[END]   success=True steps=2 rewards=0.10,1.10
```

---

## Docker

### Build

```bash
docker build -t cloudsreenv:latest .
```

### Run server

```bash
docker run -p 8000:8000 cloudsreenv:latest
```

### Run inference

```bash
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  cloudsreenv:latest \
  python inference.py
```

---

## HTTP API (OpenEnv)

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/reset` | `?task_id=...` | Start new episode |
| POST | `/step` | `Action` JSON | Execute one action |
| GET | `/state` | — | Current environment state |
| GET | `/health` | — | Liveness check |

### Example curl

```bash
# Reset
curl -X POST "http://localhost:8000/reset?task_id=task1_status_audit"

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "LIST_SERVICES"}'

# State
curl http://localhost:8000/state
```

---

## Action Schema

```json
// List all services
{"action_type": "LIST_SERVICES"}

// Get pod logs
{"action_type": "GET_LOGS", "service_id": "payment-db"}

// Restart a crashed pod
{"action_type": "RESTART", "service_id": "payment-db"}

// Scale CPU for a service
{"action_type": "SCALE", "service_id": "auth-api", "cpu_value": 2048}
```

---

## Constraints

- Memory: Python process stays well under 8 GB  
- Runtime: All 3 tasks complete in under 20 minutes  
- Graders: 100% deterministic — no LLMs used for scoring  

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CloudSREEnv                        │
│                                                     │
│  ┌─────────────┐    step(Action)     ┌───────────┐  │
│  │  LLM Agent  │ ──────────────────► │   env.py  │  │
│  │ inference.py│ ◄────────────────── │ CloudSREEnv│  │
│  └─────────────┘  (Obs, Reward, done)└─────┬─────┘  │
│                                            │        │
│                                    ┌───────▼──────┐ │
│                                    │  MockCloud   │ │
│                                    │  (Services)  │ │
│                                    └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

*Built for OpenEnv × Meta Developers Hackathon*
