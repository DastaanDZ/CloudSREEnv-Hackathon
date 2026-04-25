"""
inference.py — Evaluator for Local GRPO-Trained Models
"""

from __future__ import annotations

import json
import os
import re
import logging
import torch
from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from server.app import Action, ActionType, CloudSREEnv

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Evaluator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Point this to your trained adapter folder
TRAINED_MODEL_PATH = "./grpo_sre_model/final"
# The original base model we started with
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Prompts (Same as Training)
# ---------------------------------------------------------------------------

IC_PROMPT = """You are the Incident Commander (IC). You orchestrate the response.
- If the Current Task is task1, once L1_Triage reports logs, use CLOSE_INCIDENT.
- If the Task is task2/task3, explicitly tell L2_DB_SME the service_id (e.g. payment-db) to fix.
- Once L2_DB_SME reports success, use CLOSE_INCIDENT."""

L1_PROMPT = """You are the L1 Triage Agent. You monitor cluster health.
1. Run LIST_SERVICES.
2. Run GET_LOGS on suspicious pods.
3. Use MESSAGE_CHANNEL to escalate findings to the IC."""

L2_PROMPT = """You are the L2 Database SME.
- If crashing, use RESTART.
- If high CPU, use SCALE to 2048.
- Once [OK], use MESSAGE_CHANNEL to tell IC 'fix applied'."""

PROMPTS = {"IC": IC_PROMPT, "L1_Triage": L1_PROMPT, "L2_DB_SME": L2_PROMPT}

# ---------------------------------------------------------------------------
# Local Model Loader
# ---------------------------------------------------------------------------

logger.info(f"Loading Local Model from {TRAINED_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base + LoRA Adapters
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH).to(DEVICE)
model.eval()

def generate_action(agent_role: str, history: str) -> str:
    """Generates a response from the locally trained model."""
    system_prompt = PROMPTS[agent_role]
    full_input = f"{system_prompt}\n\n{history}\n\nAssistant:"
    
    inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False # Greedy decoding for consistent eval
        )
    
    # Decode only the newly generated part
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Orchestration Loop
# ---------------------------------------------------------------------------

def run_multi_agent_task(env: CloudSREEnv, task_id: str):
    logger.info(f"========== EVALUATING SCENARIO: {task_id} ==========")
    obs = env.reset(task_id=task_id)
    
    # Track history per agent for the session
    agent_histories = {agent: f"INITIAL ALERT:\n{obs.text_output}" for agent in PROMPTS}
    current_agent = "IC"
    
    success = False
    max_steps = 15

    for step_n in range(1, max_steps + 1):
        logger.info(f"Turn {step_n}: {current_agent} is Thinking...")
        
        raw_reply = generate_action(current_agent, agent_histories[current_agent])
        logger.debug(f"RAW OUTPUT: {raw_reply}")

        # Extract JSON logic
        json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
        clean_json_str = json_match.group(0) if json_match else raw_reply

        try:
            action_dict = json.loads(clean_json_str)
            action_dict["agent_id"] = current_agent 
            action = Action(**action_dict)
        except Exception as e:
            logger.error(f"PARSE ERROR: {e} | Raw: {raw_reply}")
            agent_histories[current_agent] += f"\nError: Your last output was not valid JSON."
            continue

        logger.info(f"ACTION: {action.action_type} -> {action.target if action.target else action.service_id}")

        # Environment Step
        step_obs, reward, done, _ = env.step(action)
        
        # Update Histories
        if action.action_type == ActionType.MESSAGE_CHANNEL:
            target = action.target
            msg = f"\nNew channel message from {current_agent}: {action.message}"
            agent_histories[target] += msg
            current_agent = target # Hand off
        elif action.action_type == ActionType.CLOSE_INCIDENT:
            if done:
                logger.info("SUCCESS: Incident Closed.")
                success = True
                break
            else:
                agent_histories[current_agent] += f"\nObservation: {step_obs.text_output}"
        else:
            agent_histories[current_agent] += f"\nObservation: {step_obs.text_output}"

    return success

def main():
    env = CloudSREEnv()
    tasks = ["task1_status_audit", "task2_self_healing", "task3_latency_resolution"]
    for task in tasks:
        run_multi_agent_task(env, task)

if __name__ == "__main__":
    main()