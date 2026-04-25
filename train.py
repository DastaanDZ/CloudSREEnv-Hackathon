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
# 1. Continuous Reward Function (High Variance for GRPO)
# ---------------------------------------------------------------------------
def sre_rubric_reward(prompts, completions, **kwargs):
    """
    Continuous reward function designed for GRPO variance.
    
    Unlike discrete rewards, this function produces fine-grained floating-point
    scores that differentiate even valid JSON completions based on:
    - Format quality (conciseness, cleanliness)
    - Semantic correctness (valid action types, service names, targets)
    - Contextual relevance (appropriate action for the given scenario)
    - Environment feedback (dense rewards from CloudSREEnv rubric)
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
        # STAGE 2: Prose/verbosity penalties (continuous, based on severity)
        # =====================================================================
        prose_indicators = ["**", "Here's", "Let me", "Sure,", "Certainly", 
                          "```", "The ", "This ", "I will", "I'll"]
        prose_count = sum(1 for p in prose_indicators if p in raw_text)
        reward -= prose_count * 0.08  # -0.08 per prose indicator (continuous)
        
        # =====================================================================
        # STAGE 3: JSON structure detection
        # =====================================================================
        if "{" not in raw_text or "}" not in raw_text:
            rewards.append(max(-1.0, reward - 0.4))
            continue
        
        # Extract JSON - reward based on position (earlier = better)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            rewards.append(max(-1.0, reward - 0.3))
            continue
        
        json_str = json_match.group(0)
        json_start_pos = json_match.start()
        
        # Reward for JSON appearing early in response (continuous: 0.0 to 0.15)
        early_bonus = max(0, 0.15 - (json_start_pos * 0.01))
        reward += early_bonus
        
        # =====================================================================
        # STAGE 4: JSON parsing and format quality
        # =====================================================================
        try:
            action_dict = json.loads(json_str)
            reward += 0.20  # Base reward for valid JSON
        except json.JSONDecodeError:
            rewards.append(max(-1.0, reward - 0.25))
            continue
        
        # Format quality bonuses (continuous variance)
        # Compact JSON (fewer characters) is better
        json_length = len(json_str)
        if json_length < 80:
            reward += 0.10
        elif json_length < 120:
            reward += 0.05
        elif json_length > 200:
            reward -= 0.05 * ((json_length - 200) / 100)  # Progressive penalty
        
        # Single-line JSON bonus
        if '\n' not in json_str:
            reward += 0.05
        
        # No extra whitespace bonus
        if '  ' not in json_str and '\t' not in json_str:
            reward += 0.03
        
        # =====================================================================
        # STAGE 5: Semantic validation (action schema quality)
        # =====================================================================
        action_type = action_dict.get("action_type", "")
        
        if not action_type:
            reward -= 0.20
            rewards.append(max(-1.0, min(1.0, reward)))
            continue
        
        # Valid action type bonus
        if action_type in VALID_ACTIONS:
            reward += 0.15
        else:
            reward -= 0.15
            # Partial credit for close matches
            for valid in VALID_ACTIONS:
                if valid.lower() in action_type.lower() or action_type.lower() in valid.lower():
                    reward += 0.05
                    break
        
        # Service ID validation (continuous based on correctness)
        service_id = action_dict.get("service_id", "")
        if action_type in ["GET_LOGS", "RESTART", "SCALE"]:
            if service_id in VALID_SERVICES:
                reward += 0.10
                # Extra bonus if service matches scenario context
                if service_id in prompt_str.lower():
                    reward += 0.08
            elif service_id:
                reward -= 0.08  # Hallucinated service name
            else:
                reward -= 0.12  # Missing required field
        
        # Target validation for MESSAGE_CHANNEL
        target = action_dict.get("target", "")
        if action_type == "MESSAGE_CHANNEL":
            if target in VALID_TARGETS:
                reward += 0.08
            elif target:
                reward -= 0.06
            else:
                reward -= 0.10
            
            # Has message content
            if action_dict.get("message"):
                msg_len = len(action_dict["message"])
                if 5 < msg_len < 200:
                    reward += 0.05
        
        # CPU value validation for SCALE
        if action_type == "SCALE":
            cpu_value = action_dict.get("cpu_value")
            if isinstance(cpu_value, int):
                if cpu_value >= 2048:
                    reward += 0.10  # Correct scaling value
                elif cpu_value >= 1024:
                    reward += 0.03
                else:
                    reward -= 0.05
            else:
                reward -= 0.08  # Missing or invalid cpu_value
        
        # =====================================================================
        # STAGE 6: Role-appropriate action bonus
        # =====================================================================
        role = _detect_role_from_prompt(prompt_str)
        
        # Role-action appropriateness (continuous rewards)
        role_action_scores = {
            "IC": {"MESSAGE_CHANNEL": 0.12, "CLOSE_INCIDENT": 0.10},
            "L1_Triage": {"LIST_SERVICES": 0.12, "GET_LOGS": 0.12, "MESSAGE_CHANNEL": 0.08},
            "L2_DB_SME": {"RESTART": 0.12, "SCALE": 0.12, "MESSAGE_CHANNEL": 0.08},
        }
        
        if action_type in role_action_scores.get(role, {}):
            reward += role_action_scores[role][action_type]
        elif action_type in ["RESTART", "SCALE"] and role != "L2_DB_SME":
            reward -= 0.15  # RBAC violation
        
        # =====================================================================
        # STAGE 7: Environment execution (dense rewards)
        # =====================================================================
        try:
            action_dict["agent_id"] = role
            env = CloudSREEnv()
            
            # Select scenario based on prompt context
            prompt_lower = prompt_str.lower()
            if "error" in prompt_lower or "crash" in prompt_lower or "oom" in prompt_lower:
                env.reset(task_id="task2_self_healing")
            elif "latency" in prompt_lower or "slow" in prompt_lower or "cpu" in prompt_lower:
                env.reset(task_id="task3_latency_resolution")
            else:
                env.reset(task_id="task1_status_audit")
            
            action = Action(**action_dict)
            _, reward_obj, done, _ = env.step(action)
            
            # Use environment's breakdown for fine-grained rewards
            for component, value in reward_obj.breakdown.items():
                reward += value * 0.3  # Scale env rewards
            
            if done and reward_obj.value > 0:
                reward += 0.20  # Task completion bonus
                
        except Exception:
            reward -= 0.05  # Small penalty for env execution failure
        
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

    # --- GRPO Magic Settings ---
    training_args = GRPOConfig(
        output_dir="./grpo_sre_model",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4, 
        generation_batch_size=4, 
        logging_steps=1,
        max_steps=50, 
        report_to="none",
        # CHANGED: Enable mixed-precision training for Nvidia T4
        fp16=True, 
        # CHANGED: Enable gradient checkpointing so we don't OOM (Out of Memory)
        gradient_checkpointing=True 
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