"""
inference.py — Baseline SRE agent for CloudSREEnv.
Updated for Phase 2 (0, 1) score compliance and mandatory stdout formatting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Ensure this import matches your directory structure (e.g., 'from app import ...')
from server.app import Action, ActionType, CloudSREEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

MAX_STEPS_PER_TASK = 18
TASKS = [
    "task1_status_audit",
    "task2_self_healing",
    "task3_latency_resolution",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) operating inside a Kubernetes-style cloud environment called CloudSREEnv. 

You have access to these actions (respond with ONLY valid JSON):
  {"action_type": "LIST_SERVICES"}
  {"action_type": "GET_LOGS",  "service_id": "<id>"}
  {"action_type": "RESTART",   "service_id": "<id>"}
  {"action_type": "SCALE",     "service_id": "<id>", "cpu_value": <int>}

Rules:
- Always start with LIST_SERVICES to assess cluster health.
- PERFORM ROOT CAUSE ANALYSIS (RCA): If a service has high latency (>200ms) or returns errors, run GET_LOGS on it immediately.
- TRACE UPSTREAM: If logs mention an 'Upstream' error or provider slowness, pivot immediately to investigate that provider service.
- TO FIX BOTTLENECKS: Use SCALE with cpu_value >= 2048 only on the service identified as the root cause (the one reporting CPU throttling or OOM).
- IF A SERVICE STATUS IS 'Error':
    1. Run GET_LOGS to confirm the reason.
    2. Run RESTART to bring the service back online.
- SEQUENTIAL FIXING: If a service is both crashed AND needs scaling, SCALE first, then RESTART.
- Never restart a service that is already Running.
- Never repeat LIST_SERVICES more than twice in a row; take a corrective action instead.
- Respond ONLY with a single JSON object — no prose, no markdown fences.

Valid service IDs: auth-api, payment-db, inventory-svc, notification-worker
"""

def build_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key",
    )

def parse_action(raw: str) -> Optional[Action]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        data: Dict[str, Any] = json.loads(raw)
        return Action(**data)
    except Exception:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return Action(**data)
            except Exception: pass
    return None

def run_task(client: OpenAI, env: CloudSREEnv, task_id: str) -> tuple[bool, int, List[float]]:
    obs = env.reset(task_id=task_id)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.text_output},
    ]

    # [START] Mandatory line
    print(f"[START] task={task_id} env=CloudSRE model={MODEL_NAME}", flush=True)

    reward_history: List[float] = []
    success = False
    final_task_score = 0.01  # Baseline non-zero score

    for step_n in range(1, MAX_STEPS_PER_TASK + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0 # Switched to 0.0 for reproducibility
            )
            raw_reply = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[STEP] step={step_n} action=ERROR reward=0.00 done=false error=API_CALL_FAILED", flush=True)
            break

        action = parse_action(raw_reply)
        if action is None:
            action_str, reward_val, done, error_msg = "PARSE_ERROR", -0.2, False, "parse_error"
            print(f"[STEP] step={step_n} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_msg}", flush=True)
            reward_history.append(reward_val)
            messages.append({"role": "assistant", "content": raw_reply})
            messages.append({"role": "user", "content": "[ERROR] Invalid JSON action."})
            continue

        step_obs, reward, done, info = env.step(action)
        
        # Update scores from environment info
        final_task_score = info.get("score", 0.01)
        error_msg = info.get("error") or "null"
        if error_msg == "none": error_msg = "null"

        action_str = action.action_type
        if action.service_id: action_str += f"({action.service_id})"

        # [STEP] Mandatory line (lowercase booleans)
        print(f"[STEP] step={step_n} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error={error_msg}", flush=True)
        reward_history.append(reward.value)

        messages.append({"role": "assistant", "content": raw_reply})
        messages.append({"role": "user", "content": step_obs.text_output})

        if done:
            success = True
            break

    # [END] Mandatory line: Score MUST be strictly (0, 1)
    success_str = str(success).lower()
    reward_str = ",".join(f"{r:.2f}" for r in reward_history)
    
    # Ensure score is never exactly 0.0 or 1.0 for the validator
    clamped_score = max(0.01, min(0.99, final_task_score))
    
    print(f"[END] success={success_str} steps={len(reward_history)} score={clamped_score:.2f} rewards={reward_str}", flush=True)
    print("", flush=True)

    return success, len(reward_history), reward_history

def main():
    if not HF_TOKEN and API_BASE_URL == "https://api.openai.com/v1":
        print("[ERROR] HF_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    client = build_client()
    env = CloudSREEnv()

    for task_id in TASKS:
        run_task(client, env, task_id)

if __name__ == "__main__":
    main()

