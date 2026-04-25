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
        
        # Detect context phases
        is_initial_alert = any(kw in prompt_lower for kw in ["initial alert", "system alert", "alert:", "escalation:", "incident:"])
        is_after_investigation = any(kw in prompt_lower for kw in ["l1_triage reports", "l1_triage found", "l1_triage identified", "l1_triage diagnostic", "root cause"])
        is_after_fix = any(kw in prompt_lower for kw in ["l2_db_sme confirms", "l2_db_sme reports", "all services now healthy", "fix successfully", "restarted and is now", "scaled to"])
        
        # --- IC Workflow ---
        if role == "IC":
            if action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                message = action_dict.get("message", "")
                
                if is_initial_alert:
                    # On initial alert, IC MUST delegate to L1
                    if target == "L1_Triage":
                        reward += 0.50  # Strong reward for correct delegation
                    elif target == "L2_DB_SME":
                        reward += 0.15  # Acceptable but skipping investigation
                    else:
                        reward -= 0.10
                        
                elif is_after_investigation:
                    # After L1 reports, IC should delegate to L2
                    if target == "L2_DB_SME":
                        reward += 0.50  # Correct: delegate fix
                    elif target == "L1_Triage":
                        reward -= 0.15  # Wrong: already have diagnosis
                else:
                    reward += 0.20  # Generic delegation is okay
                        
            elif action_type == "CLOSE_INCIDENT":
                # CLOSE_INCIDENT only appropriate after fix confirmed
                if is_after_fix:
                    reward += 0.45  # Correct timing
                elif is_initial_alert:
                    reward -= 0.60  # VERY WRONG: closing on first alert!
                elif is_after_investigation:
                    reward -= 0.40  # Wrong: need to fix first
                else:
                    reward -= 0.30  # Probably premature
                    
            elif action_type in ["LIST_SERVICES", "GET_LOGS", "RESTART", "SCALE"]:
                reward -= 0.30  # IC shouldn't do these directly
        
        # --- L1_Triage Workflow ---
        elif role == "L1_Triage":
            if action_type == "LIST_SERVICES":
                # L1 should always be willing to list services
                reward += 0.45  # Strong reward for diagnostic action
                if any(kw in prompt_lower for kw in ["status", "cluster", "what's", "investigate", "check"]):
                    reward += 0.15  # Extra for matching context
                    
            elif action_type == "GET_LOGS":
                service_id = action_dict.get("service_id", "")
                if service_id in VALID_SERVICES:
                    reward += 0.45  # Correct diagnostic action
                    # Extra if service matches context
                    service_mentioned = service_id.replace("-", "").lower()
                    if service_mentioned in prompt_lower.replace("-", ""):
                        reward += 0.20  # Investigating the right service
                elif service_id:
                    reward -= 0.15  # Hallucinated service name
                else:
                    reward -= 0.20  # Missing service_id
                    
            elif action_type == "MESSAGE_CHANNEL":
                # L1 messaging is okay but not as good as investigating
                reward += 0.10
                    
            elif action_type == "CLOSE_INCIDENT":
                reward -= 0.60  # L1 should NEVER close incidents
                
            elif action_type in ["RESTART", "SCALE"]:
                reward -= 0.50  # L1 has no write permissions - RBAC violation
        
        # --- L2_DB_SME Workflow ---
        elif role == "L2_DB_SME":
            if action_type == "RESTART":
                service_id = action_dict.get("service_id", "")
                if service_id in VALID_SERVICES:
                    reward += 0.45  # Correct fix action
                    # Extra if context mentions crash/error/down
                    if any(kw in prompt_lower for kw in ["crash", "error", "down", "oom", "recover"]):
                        reward += 0.20
                    # Extra if service matches
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        reward += 0.15
                elif service_id:
                    reward -= 0.10  # Wrong service
                else:
                    reward -= 0.25  # Missing service_id
                    
            elif action_type == "SCALE":
                service_id = action_dict.get("service_id", "")
                cpu_value = action_dict.get("cpu_value")
                
                if service_id in VALID_SERVICES and isinstance(cpu_value, int):
                    if cpu_value >= 2048:
                        reward += 0.45  # Correct scale value
                    elif cpu_value >= 1024:
                        reward += 0.20
                    else:
                        reward -= 0.10  # Too low to help
                        
                    # Context match
                    if any(kw in prompt_lower for kw in ["cpu", "resource", "overload", "performance", "latency", "scale"]):
                        reward += 0.20
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        reward += 0.15
                else:
                    reward -= 0.20  # Missing required fields
                    
            elif action_type == "MESSAGE_CHANNEL":
                # L2 should fix, not just message
                reward -= 0.10
                    
            elif action_type == "CLOSE_INCIDENT":
                reward -= 0.60  # L2 should NEVER close incidents
                
            elif action_type in ["LIST_SERVICES", "GET_LOGS"]:
                reward -= 0.25  # L2 should act, not investigate
        
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
    
    # Build diverse training dataset
    dataset = build_dataset(num_samples=800)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # --- GRPO Training Configuration ---
    training_args = GRPOConfig(
        output_dir="./grpo_sre_model",
        learning_rate=3e-5,           # Moderate LR for stable convergence
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=6,            # More generations for better variance
        generation_batch_size=6, 
        logging_steps=10,
        max_steps=200,                # More steps for workflow learning
        report_to="none",
        fp16=True, 
        gradient_checkpointing=True,
        max_completion_length=80,     # Slightly longer for MESSAGE_CHANNEL with content
        temperature=0.9,              # Higher temp for more diverse exploration
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