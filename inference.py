"""
inference.py — Multi-Agent Incident Orchestrator with Detailed Logging
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Dict, List

from openai import OpenAI
from server.app import Action, ActionType, CloudSREEnv

# --- Setup Logging ---
# Change level to logging.INFO if you want to hide the deep debugging noise
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Orchestrator")
# Silence external noisy loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Multi-Agent Prompts
# ---------------------------------------------------------------------------

IC_PROMPT = """You are the Incident Commander (IC). You orchestrate the response.
Read the alerts and messages from the channel.
- If the Current Task is an "audit" or "triage" (e.g., task1), your goal is ONLY to find the root cause. Once L1_Triage reports the logs, CLOSE the incident.
- If the Current Task involves "recovery" or "resolution" (e.g., task2, task3), you MUST explicitly tell L2_DB_SME the exact service_id (e.g., "payment-db") to fix.
- Once L2_DB_SME reports the fix is successfully applied, DO NOT message them again. IMMEDIATELY use CLOSE_INCIDENT.

Tools:
{"action_type": "MESSAGE_CHANNEL", "target": "<L1_Triage or L2_DB_SME>", "message": "<explicit instructions>"}
{"action_type": "CLOSE_INCIDENT"}

CRITICAL: Output EXACTLY ONE valid JSON object. No markdown, no explanations, no conversational text.
"""

L1_PROMPT = """You are the L1 Triage Agent. You monitor cluster health.
You CANNOT restart or scale services. You can only investigate.
1. Run LIST_SERVICES to find slow or broken pods.
2. Run GET_LOGS on suspicious pods to find the root cause.
3. If the root cause is upstream (e.g., payment-db), use MESSAGE_CHANNEL to escalate to the IC.

Tools:
{"action_type": "LIST_SERVICES"}
{"action_type": "GET_LOGS", "service_id": "<id>"}
{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<your findings>"}

CRITICAL: Output EXACTLY ONE valid JSON object. No markdown, no explanations, no conversational text.
"""

L2_PROMPT = """You are the L2 Database SME.
You only wake up when paged by the IC. You are authorized to use RESTART and SCALE.
- If the IC tells you a database is crashing, use RESTART on that specific service_id.
- If CPU is throttling (e.g., 99.8% CPU usage), use SCALE and set cpu_value to 2048.
- IMPORTANT: Once you execute RESTART or SCALE and receive an "[OK]" response, you are DONE. Immediately use MESSAGE_CHANNEL to tell the IC the fix is applied. Do NOT run any more commands.

Tools:
{"action_type": "GET_LOGS", "service_id": "<actual_id>"}
{"action_type": "RESTART", "service_id": "<actual_id>"}
{"action_type": "SCALE", "service_id": "<actual_id>", "cpu_value": <int>}
{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<fix applied>"}

CRITICAL: Output EXACTLY ONE valid JSON object. No markdown, no explanations. NEVER use the literal string "<id>", replace it with the actual service name provided by the IC.
"""

PROMPTS = {
    "IC": IC_PROMPT,
    "L1_Triage": L1_PROMPT,
    "L2_DB_SME": L2_PROMPT
}

# ---------------------------------------------------------------------------
# Agent Orchestration Loop
# ---------------------------------------------------------------------------

def run_multi_agent_task(client: OpenAI, env: CloudSREEnv, task_id: str):
    logger.info(f"========== STARTING SCENARIO: {task_id} ==========")
    obs = env.reset(task_id=task_id)
    
    agent_memory: Dict[str, List[Dict[str, str]]] = {
        agent: [{"role": "system", "content": prompt}] for agent, prompt in PROMPTS.items()
    }

    current_agent = "IC"
    task_context = f"Current Task: {task_id}\n\nINITIAL ALERT:\n{obs.text_output}"
    agent_memory["IC"].append({"role": "user", "content": task_context})

    reward_history = []
    success = False
    max_steps = 20

    for step_n in range(1, max_steps + 1):
        logger.info(f"--- Turn {step_n}: {current_agent} is Thinking ---")
        
        # --- Deep Log: What the agent is seeing right now ---
        last_context = agent_memory[current_agent][-1]['content']
        logger.debug(f"[{current_agent} INPUT CONTEXT] \n{last_context}")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=agent_memory[current_agent],
            # temperature=0.0
        )
        
        raw_reply = response.choices[0].message.content.strip()
        
        # --- Deep Log: What the LLM literally generated ---
        logger.debug(f"[{current_agent} RAW LLM OUTPUT] \n{raw_reply}")

        json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = raw_reply

        try:
            action_dict = json.loads(clean_json_str)
            action_dict["agent_id"] = current_agent 
            action = Action(**action_dict)
        except Exception as e:
            logger.error(f"[{current_agent} PARSE ERROR] Invalid JSON format. Exception: {e}")
            agent_memory[current_agent].append({"role": "user", "content": "Format Error: Output must be ONLY valid JSON."})
            continue

        action_str = f"{action.action_type}"
        if action.service_id: action_str += f"({action.service_id})"
        if action.target: action_str += f" -> {action.target}"
        logger.info(f"[{current_agent} ACTION EXECUTION] {action_str}")

        agent_memory[current_agent].append({"role": "assistant", "content": raw_reply})

        # Execute Action in Environment
        step_obs, reward, done, info = env.step(action)
        reward_history.append(reward.value)
        
        logger.debug(f"[ENVIRONMENT RESPONSE] \n{step_obs.text_output}")

        # --- ROUTING LOGIC ---
        if action.action_type == ActionType.MESSAGE_CHANNEL:
            target_agent = action.target
            handoff_msg = f"New message in channel from {current_agent}: {action.message}"
            agent_memory[target_agent].append({"role": "user", "content": handoff_msg})
            
            logger.info(f"[ROUTER] Passing context from {current_agent} to {target_agent}")
            current_agent = target_agent
        
        elif action.action_type == ActionType.CLOSE_INCIDENT:
            if done:
                logger.info(f"[SYSTEM] Incident Closed Successfully by IC.")
                success = True
                break
            else:
                agent_memory[current_agent].append({"role": "user", "content": step_obs.text_output})
        
        else:
            agent_memory[current_agent].append({"role": "user", "content": step_obs.text_output})

    score = sum(reward_history) if success else 0.01
    score = max(0.01, min(0.99, score))
    logger.info(f"========== SCENARIO END: success={success} | steps={len(reward_history)} | score={score:.2f} ==========\n")
    return success

def main():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("HF_TOKEN"):
        logger.warning("No API Key provided.")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"), base_url=API_BASE_URL)
    env = CloudSREEnv()

    TASKS = ["task1_status_audit", "task2_self_healing", "task3_latency_resolution"]
    
    for task in TASKS:
        run_multi_agent_task(client, env, task)

if __name__ == "__main__":
    main()