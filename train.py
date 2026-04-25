"""
train.py — Hugging Face TRL GRPO Training Loop for CloudSREEnv (Colab T4 Optimized)
"""

import json
import logging
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# Import our environment and data models
from server.app import CloudSREEnv, Action, ActionType
from prompts import PROMPTS, SCENARIO_MESSAGES

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GRPOTrainer")

# Valid services in the environment (for semantic scoring)
VALID_SERVICES = {"auth-api", "payment-db", "inventory-svc", "notification-worker"}
VALID_ACTIONS = {"LIST_SERVICES", "GET_LOGS", "RESTART", "SCALE", "MESSAGE_CHANNEL", "CLOSE_INCIDENT"}
VALID_TARGETS = {"IC", "L1_Triage", "L2_DB_SME"}

# ---------------------------------------------------------------------------
# 1. Workflow-Aware Reward Function for Multi-Agent SRE
# ---------------------------------------------------------------------------
def sre_rubric_reward(prompts, completions, **kwargs):
    """
    Workflow-aware reward function that teaches correct SRE behavior patterns.
    
    Key insight: Single-turn training must encode multi-turn workflow knowledge
    by heavily rewarding/penalizing actions based on scenario context.
    
    Workflow rules encoded:
    - L1_Triage MUST use LIST_SERVICES or GET_LOGS before MESSAGE_CHANNEL
    - IC should delegate to L2_DB_SME when error/fix is mentioned
    - Only IC should use CLOSE_INCIDENT
    - L2_DB_SME should use RESTART/SCALE, not just MESSAGE_CHANNEL
    """
    rewards = []
    
    for prompt_str, completion in zip(prompts, completions):
        raw_text = completion[0]['content'] if isinstance(completion, list) else completion
        reward = 0.0
        
        # =====================================================================
        # STAGE 1: Hard penalties for refusals (-1.0 immediate return)
        # =====================================================================
        refusal_patterns = ["I cannot", "I'm sorry", "I apologize", "As an AI", 
                          "I'm not able", "I don't", "I can't"]
        if any(p in raw_text for p in refusal_patterns):
            rewards.append(-1.0)
            continue
        
        # =====================================================================
        # STAGE 2: Prose/verbosity penalties
        # =====================================================================
        prose_indicators = ["**", "Here's", "Let me", "Sure,", "Certainly", 
                          "```", "I will", "I'll", "First,", "To "]
        prose_count = sum(1 for p in prose_indicators if p in raw_text)
        reward -= prose_count * 0.1
        
        # =====================================================================
        # STAGE 3: JSON structure detection
        # =====================================================================
        if "{" not in raw_text or "}" not in raw_text:
            rewards.append(max(-1.0, reward - 0.5))
            continue
        
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            rewards.append(max(-1.0, reward - 0.4))
            continue
        
        json_str = json_match.group(0)
        json_start_pos = json_match.start()
        
        # Strong bonus for JSON appearing immediately (position 0-5)
        if json_start_pos <= 5:
            reward += 0.15
        elif json_start_pos <= 20:
            reward += 0.08
        else:
            reward -= 0.05 * (json_start_pos / 50)
        
        # =====================================================================
        # STAGE 4: JSON parsing and compactness
        # =====================================================================
        try:
            action_dict = json.loads(json_str)
            reward += 0.15
        except json.JSONDecodeError:
            rewards.append(max(-1.0, reward - 0.3))
            continue
        
        # Strongly reward compact JSON
        json_length = len(json_str)
        if json_length < 60:
            reward += 0.12
        elif json_length < 100:
            reward += 0.06
        elif json_length > 150:
            reward -= 0.10
        
        if '\n' not in json_str:
            reward += 0.05
        
        # =====================================================================
        # STAGE 5: Action type validation
        # =====================================================================
        action_type = action_dict.get("action_type", "")
        
        if not action_type:
            rewards.append(max(-1.0, reward - 0.3))
            continue
        
        if action_type in VALID_ACTIONS:
            reward += 0.10
        else:
            reward -= 0.20
        
        # =====================================================================
        # STAGE 6: WORKFLOW-AWARE SCORING (Critical for learning correct behavior)
        # =====================================================================
        role = _detect_role_from_prompt(prompt_str)
        prompt_lower = prompt_str.lower()
        
        # --- IC Workflow ---
        if role == "IC":
            if action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                # IC should delegate to L1 for investigation
                if "initial alert" in prompt_lower or "system alert" in prompt_lower:
                    if target == "L1_Triage":
                        reward += 0.25  # Correct: delegate investigation
                    elif target == "L2_DB_SME":
                        reward += 0.10  # Acceptable but not ideal first step
                # IC should delegate to L2 when root cause is known
                elif "root cause" in prompt_lower or "found" in prompt_lower or "reports:" in prompt_lower:
                    if target == "L2_DB_SME":
                        reward += 0.30  # Correct: delegate fix to L2
                    elif target == "L1_Triage":
                        reward -= 0.10  # Wrong: already have diagnosis
                        
            elif action_type == "CLOSE_INCIDENT":
                # Only close when fix is confirmed
                if "fix applied" in prompt_lower or "restarted" in prompt_lower or "scaled" in prompt_lower or "healthy" in prompt_lower:
                    reward += 0.35  # Correct: close after fix confirmed
                else:
                    reward -= 0.25  # Premature closure
                    
            elif action_type in ["LIST_SERVICES", "GET_LOGS", "RESTART", "SCALE"]:
                reward -= 0.20  # IC shouldn't do these directly
        
        # --- L1_Triage Workflow ---
        elif role == "L1_Triage":
            if action_type == "LIST_SERVICES":
                # L1 should list services when asked to investigate
                if "investigate" in prompt_lower or "check" in prompt_lower or "audit" in prompt_lower or "list" in prompt_lower:
                    reward += 0.35  # Correct diagnostic action
                else:
                    reward += 0.15  # Still valid
                    
            elif action_type == "GET_LOGS":
                service_id = action_dict.get("service_id", "")
                if service_id in VALID_SERVICES:
                    reward += 0.30  # Correct diagnostic action
                    # Extra if service matches context
                    if service_id.replace("-", "") in prompt_lower.replace("-", ""):
                        reward += 0.15
                else:
                    reward -= 0.10  # Hallucinated service
                    
            elif action_type == "MESSAGE_CHANNEL":
                # L1 should only message AFTER doing diagnostics
                # Penalize if prompt suggests they should be investigating
                if "investigate" in prompt_lower or "check" in prompt_lower or "list" in prompt_lower or "get logs" in prompt_lower:
                    reward -= 0.20  # Should do diagnostics first!
                else:
                    reward += 0.10  # Reporting findings is okay
                    
            elif action_type == "CLOSE_INCIDENT":
                reward -= 0.40  # L1 should NEVER close incidents
                
            elif action_type in ["RESTART", "SCALE"]:
                reward -= 0.35  # L1 has no write permissions
        
        # --- L2_DB_SME Workflow ---
        elif role == "L2_DB_SME":
            if action_type == "RESTART":
                service_id = action_dict.get("service_id", "")
                if service_id in VALID_SERVICES:
                    reward += 0.35  # Correct fix action
                    if "restart" in prompt_lower or "crash" in prompt_lower or "error" in prompt_lower:
                        reward += 0.15  # Matches requested action
                else:
                    reward -= 0.10
                    
            elif action_type == "SCALE":
                service_id = action_dict.get("service_id", "")
                cpu_value = action_dict.get("cpu_value")
                if service_id in VALID_SERVICES and isinstance(cpu_value, int):
                    if cpu_value >= 2048:
                        reward += 0.35  # Correct scale value
                    elif cpu_value >= 1024:
                        reward += 0.15
                    else:
                        reward -= 0.10  # Too low
                    if "scale" in prompt_lower or "cpu" in prompt_lower or "latency" in prompt_lower:
                        reward += 0.15
                else:
                    reward -= 0.15
                    
            elif action_type == "MESSAGE_CHANNEL":
                # L2 should message only AFTER applying fix
                if "apply" in prompt_lower or "fix" in prompt_lower or "restart" in prompt_lower or "scale" in prompt_lower:
                    reward -= 0.15  # Should fix first, not just message!
                else:
                    reward += 0.10  # Reporting completion is okay
                    
            elif action_type == "CLOSE_INCIDENT":
                reward -= 0.40  # L2 should NEVER close incidents
                
            elif action_type in ["LIST_SERVICES", "GET_LOGS"]:
                reward -= 0.15  # L2 should act, not investigate
        
        # =====================================================================
        # STAGE 7: Field completeness validation
        # =====================================================================
        if action_type in ["GET_LOGS", "RESTART", "SCALE"]:
            if not action_dict.get("service_id"):
                reward -= 0.15
                
        if action_type == "SCALE":
            if not isinstance(action_dict.get("cpu_value"), int):
                reward -= 0.15
                
        if action_type == "MESSAGE_CHANNEL":
            if not action_dict.get("target"):
                reward -= 0.15
            if not action_dict.get("message"):
                reward -= 0.10
        
        # =====================================================================
        # Final: Clamp to [-1.0, 1.0]
        # =====================================================================
        rewards.append(max(-1.0, min(1.0, reward)))
    
    return rewards


def _detect_role_from_prompt(prompt_str: str) -> str:
    """Extract the agent role from the prompt string."""
    prompt_lower = prompt_str.lower()
    if "incident commander" in prompt_lower or "role: ic" in prompt_lower:
        return "IC"
    elif "l1 triage" in prompt_lower or "l1_triage" in prompt_lower:
        return "L1_Triage"
    elif "l2" in prompt_lower or "database sme" in prompt_lower or "l2_db_sme" in prompt_lower:
        return "L2_DB_SME"
    return "IC"
# ---------------------------------------------------------------------------
# 2. Multi-Role Dataset (uses SCENARIO_MESSAGES from prompts.py)
# ---------------------------------------------------------------------------
def build_dataset(num_samples: int = 500):
    """
    Builds a diverse dataset that shuffles roles and scenarios.
    
    Args:
        num_samples: Number of training examples to generate (default 500)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts_list = []
    
    roles = list(PROMPTS.keys())
    
    for _ in range(num_samples):
        # Randomly select a role
        role_key = roles[torch.randint(0, len(roles), (1,)).item()]
        
        # Randomly select a scenario message for this role
        messages_for_role = SCENARIO_MESSAGES[role_key]
        msg = messages_for_role[torch.randint(0, len(messages_for_role), (1,)).item()]
        
        messages = [
            {"role": "system", "content": PROMPTS[role_key]},
            {"role": "user", "content": msg}
        ]
        
        # Convert the chat list into a single formatted string for Llama 3
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        prompts_list.append(prompt_str)
    
    logger.info(f"Built dataset with {len(prompts_list)} training examples across {len(roles)} roles")
    return Dataset.from_dict({"prompt": prompts_list})

# ---------------------------------------------------------------------------
# 3. The Main Training Loop
# ---------------------------------------------------------------------------
def main():
    logger.info(f"Loading {MODEL_NAME} onto {DEVICE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model in 16-bit float to save VRAM on T4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    # Build diverse training dataset (500 samples by default)
    dataset = build_dataset(num_samples=500)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # --- GRPO Training Configuration ---
    training_args = GRPOConfig(
        output_dir="./grpo_sre_model",
        learning_rate=5e-5,           # Increased LR for faster convergence
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # More accumulation for stability
        num_generations=4, 
        generation_batch_size=4, 
        logging_steps=5,
        max_steps=150,                # More steps to learn workflow patterns
        report_to="none",
        fp16=True, 
        gradient_checkpointing=True,
        max_completion_length=64,     # Limit verbosity - force concise JSON
        # Temperature for generation diversity
        temperature=0.7,
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sre_rubric_reward], 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    logger.info("\n========== STARTING GRPO RL TRAINING ==========")
    trainer.train()
    
    logger.info("\nSaving trained model to ./grpo_sre_model/final ...")
    trainer.save_model("./grpo_sre_model/final")
    logger.info("Training complete. You can now use this model in inference.py!")

if __name__ == "__main__":
    main()