"""
inference.py — Baseline SRE agent for CloudSREEnv.

Reads environment variables:
  API_BASE_URL  : OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    : Model identifier
  HF_TOKEN      : Bearer token (Hugging Face or OpenAI API key)

STDOUT format (required):
  [START] task=<task> env=CloudSRE model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg>
  [END]   success=<bool> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

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
- GET_LOGS before restarting any service.
- To fix high CPU/latency: use SCALE with cpu_value >= 2048.
- If a service status is 'Error', you MUST:
    1. Run GET_LOGS to confirm the reason.
    2. Run RESTART to bring the service back online.
- Use SCALE only for 'Running' services that have high latency (>200ms).
- Never restart a service that is already Running.
- Never invent service IDs not listed in the observation.
- Respond ONLY with a single JSON object — no prose, no markdown fences.

Valid service IDs: auth-api, payment-db, inventory-svc, notification-worker
"""


# ---------------------------------------------------------------------------
# OpenAI client builder
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key",
    )


# ---------------------------------------------------------------------------
# Parse model response into Action
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> Optional[Action]:
    """Best-effort parse of a JSON action from the model's reply."""
    raw = raw.strip()
    # Strip markdown code fences if any
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()
    try:
        data: Dict[str, Any] = json.loads(raw)
        return Action(**data)
    except Exception:
        # Attempt to extract JSON substring
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return Action(**data)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    env: CloudSREEnv,
    task_id: str,
) -> tuple[bool, int, List[float]]:
    """
    Run a full episode for task_id.
    Returns (success, steps_taken, reward_list).
    """
    obs = env.reset(task_id=task_id)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.text_output},
    ]

    task_short = task_id.replace("task", "T").replace("_", "-")
    print(f"[START] task={task_short} env=CloudSRE model={MODEL_NAME}", flush=True)

    reward_history: List[float] = []
    success = False

    for step_n in range(1, MAX_STEPS_PER_TASK + 1):
        # ---- Ask the model --------------------------------------------
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=1.0
            )
            raw_reply = response.choices[0].message.content or ""
        except Exception as exc:
            print(
                f"[STEP] step={step_n} action=ERROR reward=0.0 done=False "
                f"error=API_CALL_FAILED:{exc}",
                flush=True,
            )
            break

        # ---- Parse action ---------------------------------------------
        action = parse_action(raw_reply)
        if action is None:
            # Treat as a hallucination / bad response
            action_str = "PARSE_ERROR"
            obs_text = "[ERROR] Could not parse your response as valid JSON action."
            reward_val = -0.2
            done = False
            error_msg = "parse_error"

            print(
                f"[STEP] step={step_n} action={action_str} reward={reward_val:.2f} "
                f"done={done} error={error_msg}",
                flush=True,
            )
            reward_history.append(reward_val)
            messages.append({"role": "assistant", "content": raw_reply})
            messages.append({"role": "user", "content": obs_text})
            continue

        # ---- Execute action in env ------------------------------------
        step_obs, reward, done, info = env.step(action)
        error_msg = info.get("error") or "none"
        action_str = action.action_type
        if action.service_id:
            action_str += f"({action.service_id})"
        if action.cpu_value:
            action_str += f",cpu={action.cpu_value}"

        print(
            f"[STEP] step={step_n} action={action_str} reward={reward.value:.2f} "
            f"done={done} error={error_msg}",
            flush=True,
        )
        reward_history.append(reward.value)

        # ---- Append to conversation history --------------------------
        messages.append({"role": "assistant", "content": raw_reply})
        messages.append({"role": "user", "content": step_obs.text_output})

        if done:
            success = True
            break

    total_r = round(sum(reward_history), 4)
    reward_str = ",".join(f"{r:.2f}" for r in reward_history)
    print(
        f"[END] success={success} steps={len(reward_history)} rewards={reward_str}",
        flush=True,
    )
    print(f"      cumulative_reward={total_r}", flush=True)
    print("", flush=True)

    return success, len(reward_history), reward_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN and API_BASE_URL == "https://api.openai.com/v1":
        print("[ERROR] HF_TOKEN not set. Export your API key as HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    client = build_client()
    env = CloudSREEnv()

    overall_results = []
    wall_start = time.time()

    for task_id in TASKS:
        t0 = time.time()
        success, steps, rewards = run_task(client, env, task_id)
        elapsed = round(time.time() - t0, 1)
        overall_results.append(
            {
                "task": task_id,
                "success": success,
                "steps": steps,
                "total_reward": round(sum(rewards), 4),
                "elapsed_s": elapsed,
            }
        )

    total_elapsed = round(time.time() - wall_start, 1)

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in overall_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"  {status}  {r['task']:<30}  "
            f"steps={r['steps']:>3}  "
            f"reward={r['total_reward']:>6.2f}  "
            f"time={r['elapsed_s']}s"
        )
    total_successes = sum(1 for r in overall_results if r["success"])
    print(f"\n  Passed {total_successes}/{len(TASKS)} tasks in {total_elapsed}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
