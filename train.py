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
# 1. The GRPO Reward Function
# ---------------------------------------------------------------------------
def sre_rubric_reward(prompts, completions, **kwargs):
    """
    GRPO calls this function. It passes in a batch of generated completions.
    We run each completion through our CloudSREEnv to get the score.
    """
    rewards = []
    env = CloudSREEnv()
    
    for completion in completions:
        raw_text = completion[0]['content'] if isinstance(completion, list) else completion
        
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        clean_json = match.group(0) if match else raw_text
        
        env.reset(task_id="task1_status_audit")
        
        try:
            action_dict = json.loads(clean_json)
            action_dict["agent_id"] = "L1_Triage"
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type=ActionType.INVALID_FORMAT, agent_id="L1_Triage")
            
        _, reward_obj, _, _ = env.step(action)
        rewards.append(float(reward_obj.value))
        
        action_name = action.action_type if action.action_type else "INVALID"
        logger.info(f"Action: {action_name:<15} | Reward: {reward_obj.value:+.2f} | Reason: {reward_obj.reason}")

    return rewards

# ---------------------------------------------------------------------------
# 2. The Training Dataset
# ---------------------------------------------------------------------------
def build_dataset():
    system_prompt = PROMPTS["L1_Triage"]
    user_prompt = "New message in channel from IC: Audit/Triage: Payment-db is in Error. Please collect and analyze logs to determine the root cause."
    
    dataset_dict = {
        "prompt": [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        ] * 50
    }
    return Dataset.from_dict(dataset_dict)

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
        processing_class=tokenizer
    )

    logger.info("\n========== STARTING GRPO RL TRAINING ==========")
    trainer.train()
    
    logger.info("\nSaving trained model to ./grpo_sre_model/final ...")
    trainer.save_model("./grpo_sre_model/final")
    logger.info("Training complete. You can now use this model in inference.py!")

if __name__ == "__main__":
    main()