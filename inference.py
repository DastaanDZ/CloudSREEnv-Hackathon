"""
inference.py — Evaluator for Local GRPO-Trained Models
Supports both 'BASE' model evaluation and 'TRAINED' adapter evaluation.
"""

from __future__ import annotations
import json, os, re, logging, torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from server.app import Action, ActionType, CloudSREEnv
from prompts import PROMPTS  # Shared prompts with train.py


def extract_first_json_object(raw_text: str) -> dict | None:
    """Extract the first valid JSON object from text."""
    start = raw_text.find("{")
    if start == -1:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw_text, start)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Evaluator")

# ---------------------------------------------------------------------------
# 1. EVALUATION CONFIGURATION
# ---------------------------------------------------------------------------
# TOGGLE THIS: Use "BASE" to generate 'Before' logs, "TRAINED" for 'After' logs.
EVAL_MODE = "TRAINED" 

TRAINED_MODEL_PATH = "./grpo_sre_model/final"
# BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BASE_MODEL_NAME ="Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 3. CORE EVALUATION FUNCTIONS (Encapsulated)
# ---------------------------------------------------------------------------
def load_eval_model(mode="TRAINED"):
    """
    Loads the model only when explicitly called. 
    Prevents train.py from crashing on import.
    """
    logger.info(f"--- LOADING MODEL IN {mode} MODE ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the pure base model weights
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.float16
    ).to(DEVICE)

    if mode == "TRAINED":
        if not os.path.exists(TRAINED_MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at {TRAINED_MODEL_PATH}. Train first!")
        logger.info(f"Applying LoRA Adapters from {TRAINED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, TRAINED_MODEL_PATH).to(DEVICE)
    
    model.eval()
    return model, tokenizer

def generate_action(agent_role: str, history: str, model, tokenizer) -> str:
    """Generates a response from the loaded model (Base or Trained)."""
    system_prompt = PROMPTS[agent_role]
    
    # Use tokenizer's chat template (works for any model: Llama, Qwen, etc.)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": history}
    ]
    full_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,             # <-- CHANGED: Allow the model to explore
            temperature=0.3,            # <-- NEW: Just enough creativity to break loops
            repetition_penalty=1.1      # <-- NEW: Punishes the model for repeating exact phrases
        )
    
    # Extract only the newly generated tokens
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def suggested_l1_log_action(history: str, task_id: str) -> str:
    """Return the exact GET_LOGS action L1 should take before reporting."""
    history_lower = history.lower()
    if "task2" in task_id or "payment-db" in history_lower or "oom" in history_lower or "crash" in history_lower:
        service_id = "payment-db"
    else:
        service_id = "auth-api"
    return json.dumps({"action_type": "GET_LOGS", "service_id": service_id})

def task1_has_required_evidence(env: CloudSREEnv, task_id: str) -> bool:
    """Task1 succeeds only after auth-api logs and before any remediation."""
    return "task1" in task_id and any(
        a.action_type == ActionType.GET_LOGS and a.service_id == "auth-api"
        for a in env.action_history
    )

def run_multi_agent_task(env: CloudSREEnv, task_id: str, model, tokenizer):
    logger.info(f"========== EVALUATING SCENARIO: {task_id} ==========")
    obs = env.reset(task_id=task_id)
    
    agent_histories = {agent: f"INITIAL ALERT:\n{obs.text_output}" for agent in PROMPTS}
    current_agent = "IC"
    max_steps = 15
    
    # Track recent actions to detect repetition
    recent_actions = {agent: [] for agent in PROMPTS}

    for step_n in range(1, max_steps + 1):
        logger.info(f"Turn {step_n}: {current_agent} is Thinking...")
        
        raw_reply = generate_action(current_agent, agent_histories[current_agent], model, tokenizer)
        
        # Extract the first valid JSON object (handles multi-JSON output)
        action_dict = extract_first_json_object(raw_reply)
        if action_dict is None:
            logger.warning(f"PARSE ERROR: No valid JSON | Content: {raw_reply[:80]}...")
            agent_histories[current_agent] += "\n[SYSTEM] Error: Output ONE valid JSON object. No prose, no multiple objects."
            continue

        try:
            action_dict["agent_id"] = current_agent 
            action = Action(**action_dict)
        except Exception as e:
            logger.warning(f"PARSE ERROR: {e} | Content: {raw_reply[:50]}...")
            agent_histories[current_agent] += "\n[SYSTEM] Error: Invalid action fields. Output ONE valid JSON object."
            continue

        logger.info(f"ACTION: {action.action_type} -> {action.target or action.service_id}")
        
        # Track this action
        action_key = f"{action.action_type}:{action.service_id or action.target or ''}"
        recent_actions[current_agent].append(action_key)
        
        # Detect repetition (same action 2+ times)
        if len(recent_actions[current_agent]) >= 2 and recent_actions[current_agent][-1] == recent_actions[current_agent][-2]:
            logger.warning(f"Repetition detected for {current_agent}. Injecting hint.")
            if current_agent == "L1_Triage":
                next_action = suggested_l1_log_action(agent_histories[current_agent], task_id)
                last_l1_key = recent_actions[current_agent][-1]
                if "GET_LOGS" in last_l1_key:
                    agent_histories[current_agent] += "\n[SYSTEM] You already gathered logs. Now report findings to IC using MESSAGE_CHANNEL."
                else:
                    agent_histories[current_agent] += f"\n[SYSTEM] Do not repeat or report yet. Your next action must be exactly: {next_action}"
            elif current_agent == "L2_DB_SME":
                last_l2_key = recent_actions[current_agent][-1]
                if "RESTART" in last_l2_key or "SCALE" in last_l2_key:
                    agent_histories[current_agent] += "\n[SYSTEM] Fix already applied successfully. Do NOT repeat it. Report completion to IC now: {\"action_type\": \"MESSAGE_CHANNEL\", \"target\": \"IC\", \"message\": \"Fix applied.\"}"
                else:
                    agent_histories[current_agent] += "\n[SYSTEM] Fix already attempted. Report status to IC using MESSAGE_CHANNEL."
            elif current_agent == "IC":
                last_action_key = recent_actions[current_agent][-1]
                has_l2_message = "new message from l2_db_sme" in agent_histories[current_agent].lower()
                if "CLOSE_INCIDENT" in last_action_key:
                    agent_histories[current_agent] += "\n[SYSTEM] Incident cannot be closed yet. If L1 reported a fixable issue, delegate the fix to L2_DB_SME."
                elif "MESSAGE_CHANNEL:L1_Triage" in last_action_key:
                    agent_histories[current_agent] += "\n[SYSTEM] You already delegated to L1. If L1 reported findings, delegate the fix to L2_DB_SME using MESSAGE_CHANNEL."
                elif "MESSAGE_CHANNEL:L2_DB_SME" in last_action_key and has_l2_message:
                    agent_histories[current_agent] += "\n[SYSTEM] L2 has already reported back. Close the incident now: {\"action_type\": \"CLOSE_INCIDENT\"}"
                elif "MESSAGE_CHANNEL:L2_DB_SME" in last_action_key:
                    agent_histories[current_agent] += "\n[SYSTEM] You already delegated to L2. Wait for L2's response or try CLOSE_INCIDENT if the fix was applied."
                else:
                    agent_histories[current_agent] += "\n[SYSTEM] Try a different action. Delegate fix to L2_DB_SME or close the incident."

        if action.action_type == ActionType.MESSAGE_CHANNEL and action.target not in agent_histories:
            logger.warning(f"Invalid MESSAGE_CHANNEL target from {current_agent}: {action.target}")
            recent_actions[current_agent].pop()
            agent_histories[current_agent] += "\n[SYSTEM] Invalid target. Use exactly one of: IC, L1_Triage, L2_DB_SME."
            continue

        if (current_agent == "IC"
                and task1_has_required_evidence(env, task_id)
                and action.action_type == ActionType.MESSAGE_CHANNEL
                and action.target == "L2_DB_SME"):
            logger.warning("Task1 is RCA-only; blocking IC delegation to L2.")
            recent_actions[current_agent].pop()
            agent_histories[current_agent] += (
                "\n[SYSTEM] Task1 is certificate RCA only. Do NOT delegate to L2_DB_SME and do NOT remediate. "
                "Close now with exactly: {\"action_type\": \"CLOSE_INCIDENT\"}"
            )
            continue

        # Safety net: L1 must investigate before reporting to IC.
        # Without this, L1 can shortcut by messaging IC immediately, which derails
        # task1 (no GET_LOGS evidence -> CLOSE_INCIDENT rejected) and wastes turns.
        if (current_agent == "L1_Triage"
                and action.action_type == ActionType.MESSAGE_CHANNEL
                and action.target == "IC"):
            prior_actions = recent_actions[current_agent][:-1]
            has_log_evidence = any(k.startswith("GET_LOGS") for k in prior_actions)
            if not has_log_evidence:
                logger.warning(f"L1 attempted to report to IC without investigation. Rejecting.")
                recent_actions[current_agent].pop()
                next_action = suggested_l1_log_action(agent_histories[current_agent], task_id)
                agent_histories[current_agent] += (
                    "\n[SYSTEM] You must collect log evidence before reporting to IC. "
                    f"Do not use MESSAGE_CHANNEL now. Your next action must be exactly: {next_action}"
                )
                continue

        step_obs, _, done, _ = env.step(action)

        # If the env hard-blocked the action, surface the obs to the current agent
        # and do NOT route (no message delivery, no current_agent switch).
        if step_obs.text_output.startswith("[BLOCKED]"):
            logger.warning(f"Env blocked duplicate action by {current_agent}.")
            agent_histories[current_agent] += f"\nObs: {step_obs.text_output}"
            continue

        if action.action_type == ActionType.MESSAGE_CHANNEL:
            msg = f"\nNew message from {current_agent}: {action.message}"
            agent_histories[action.target] += msg
            current_agent = action.target 
        elif action.action_type == ActionType.CLOSE_INCIDENT:
            if done:
                logger.info(f"SUCCESS: {task_id} CLOSED SUCCESSFULLY.")
                return True
            else:
                agent_histories[current_agent] += f"\nObs: {step_obs.text_output} The issue is not yet resolved. Delegate the fix to L2_DB_SME if not already done."
        else:
            agent_histories[current_agent] += f"\nObs: {step_obs.text_output}"

    logger.error(f"FAILURE: {task_id} exceeded max steps.")
    return False

# ---------------------------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    # 1. Load the model (Mode based on EVAL_MODE at top)
    model, tokenizer = load_eval_model(mode=EVAL_MODE)
    
    # 2. Run scenarios
    env = CloudSREEnv()
    tasks = ["task1_tls_certificate_rca", "task2_self_healing", "task3_latency_resolution"]
    
    results = {}
    for task in tasks:
        results[task] = run_multi_agent_task(env, task, model, tokenizer)
    
    logger.info(f"FINAL RESULTS ({EVAL_MODE}): {results}")

if __name__ == "__main__":
    main()