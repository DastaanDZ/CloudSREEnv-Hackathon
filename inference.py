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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Evaluator")

# ---------------------------------------------------------------------------
# 1. EVALUATION CONFIGURATION
# ---------------------------------------------------------------------------
# TOGGLE THIS: Use "BASE" to generate 'Before' logs, "TRAINED" for 'After' logs.
EVAL_MODE = "TRAINED" 

TRAINED_MODEL_PATH = "./grpo_sre_model/final"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 2. PROMPTS (Shared with train.py)
# ---------------------------------------------------------------------------
IC_PROMPT = """You are the Incident Commander (IC). Orchestrate the response.
- Task1: Once L1_Triage reports logs, use CLOSE_INCIDENT.
- Task2/3: Tell L2_DB_SME the service_id (e.g. payment-db) to fix.
- Use CLOSE_INCIDENT once fixed."""

L1_PROMPT = """You are the L1 Triage Agent. Monitor cluster health.
1. Run LIST_SERVICES. 2. Run GET_LOGS on suspicious pods. 3. MESSAGE_CHANNEL to IC."""

L2_PROMPT = """You are the L2 Database SME.
- If crashing: RESTART. - If high CPU: SCALE to 2048. - Use MESSAGE_CHANNEL to tell IC 'fix applied'."""

PROMPTS = {"IC": IC_PROMPT, "L1_Triage": L1_PROMPT, "L2_DB_SME": L2_PROMPT}

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
    # Llama 3 format
    full_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    full_input += f"<|start_header_id|>user<|end_header_id|>\n\n{history}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=128, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    # Extract only the newly generated tokens
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def run_multi_agent_task(env: CloudSREEnv, task_id: str, model, tokenizer):
    logger.info(f"========== EVALUATING SCENARIO: {task_id} ==========")
    obs = env.reset(task_id=task_id)
    
    agent_histories = {agent: f"INITIAL ALERT:\n{obs.text_output}" for agent in PROMPTS}
    current_agent = "IC"
    max_steps = 15

    for step_n in range(1, max_steps + 1):
        logger.info(f"Turn {step_n}: {current_agent} is Thinking...")
        
        raw_reply = generate_action(current_agent, agent_histories[current_agent], model, tokenizer)
        
        # Regex to find JSON blocks
        json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
        clean_json_str = json_match.group(0) if json_match else "{}"

        try:
            action_dict = json.loads(clean_json_str)
            action_dict["agent_id"] = current_agent 
            action = Action(**action_dict)
        except Exception as e:
            logger.warning(f"PARSE ERROR: {e} | Content: {raw_reply[:50]}...")
            agent_histories[current_agent] += f"\nError: Your output was not valid JSON."
            continue

        logger.info(f"ACTION: {action.action_type} -> {action.target or action.service_id}")
        step_obs, _, done, _ = env.step(action)
        
        if action.action_type == ActionType.MESSAGE_CHANNEL:
            msg = f"\nNew message from {current_agent}: {action.message}"
            agent_histories[action.target] += msg
            current_agent = action.target 
        elif action.action_type == ActionType.CLOSE_INCIDENT:
            if done:
                logger.info(f"SUCCESS: {task_id} CLOSED SUCCESSFULLY.")
                return True
            else:
                agent_histories[current_agent] += f"\nObs: Cannot close yet. {step_obs.text_output}"
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
    tasks = ["task1_status_audit", "task2_self_healing", "task3_latency_resolution"]
    
    results = {}
    for task in tasks:
        results[task] = run_multi_agent_task(env, task, model, tokenizer)
    
    logger.info(f"FINAL RESULTS ({EVAL_MODE}): {results}")

if __name__ == "__main__":
    main()