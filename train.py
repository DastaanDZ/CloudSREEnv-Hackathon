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
from inference import PROMPTS

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
# CHANGED: Detect Nvidia CUDA instead of Apple MPS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GRPOTrainer")

# ---------------------------------------------------------------------------
# 1. Improved Multi-Role Reward Function
# ---------------------------------------------------------------------------
def sre_rubric_reward(prompts, completions, **kwargs):
    rewards = []
    env = CloudSREEnv()
    
    for prompt_str, completion in zip(prompts, completions):
        # Identify role based on keywords in the formatted prompt string
        if "Incident Commander" in prompt_str:
            current_role = "IC"
        elif "L2 Database SME" in prompt_str:
            current_role = "L2_DB_SME"
        else:
            current_role = "L1_Triage"
        
        raw_text = completion[0]['content'] if isinstance(completion, list) else completion
        
        # --- FORMATTING RUBRIC ---
        format_reward = 0.2 if raw_text.strip().startswith("{") and raw_text.strip().endswith("}") else -0.3
            
        # --- ENVIRONMENT RUBRIC ---
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        clean_json = match.group(0) if match else "{}"
        
        # Set scenario based on role [cite: 5, 37]
        env.reset(task_id="task1_status_audit" if current_role != "L2_DB_SME" else "task2_self_healing")
        
        try:
            action_dict = json.loads(clean_json)
            action_dict["agent_id"] = current_role
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type=ActionType.INVALID_FORMAT, agent_id=current_role)
            
        _, reward_obj, _, _ = env.step(action)
        
        # Composable reward: Format Adherence + Environment Success [cite: 92]
        rewards.append(format_reward + float(reward_obj.value))

    return rewards
# ---------------------------------------------------------------------------
# 2. Multi-Role Dataset
# ---------------------------------------------------------------------------
def build_dataset():
    """Builds a dataset that shuffles between IC, L1, and L2 roles with chat templates."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data = []
    roles = [
        ("IC", "INITIAL ALERT: payment-db in Error"), 
        ("L1_Triage", "IC Message: Audit the database"), 
        ("L2_DB_SME", "IC Message: Fix payment-db")
    ]
             
    for _ in range(100):
        role, msg = roles[torch.randint(0, len(roles), (1,)).item()]
        
        messages = [
            {"role": "system", "content": PROMPTS[role]},
            {"role": "user", "content": msg}
        ]
        
        # Apply the chat template to create a single string prompt
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        data.append({"prompt": prompt_str})
        
    return Dataset.from_dict({"prompt": data})

# ---------------------------------------------------------------------------
# 3. The Main Training Loop
# ---------------------------------------------------------------------------
def main():
    logger.info(f"Loading {MODEL_NAME} onto {DEVICE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CHANGED: Load the model in 16-bit float to save massive amounts of VRAM on the T4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    dataset = build_dataset()

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