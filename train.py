"""
train.py — Hugging Face TRL GRPO Training Loop for CloudSREEnv
"""

import json
import logging
import re
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# Import our environment and data models
from server.app import CloudSREEnv, Action, ActionType
from inference import PROMPTS

# --- Configuration ---
# Llama 3.2 1B is perfect for local Mac training (MPS). 
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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
        # TRL passes completions as list of dicts or strings depending on version
        raw_text = completion[0]['content'] if isinstance(completion, list) else completion
        
        # Extract JSON
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        clean_json = match.group(0) if match else raw_text
        
        # Reset env for a fair test (Task 1: Status Audit)
        env.reset(task_id="task1_status_audit")
        
        try:
            action_dict = json.loads(clean_json)
            action_dict["agent_id"] = "L1_Triage"
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type=ActionType.INVALID_FORMAT, agent_id="L1_Triage")
            
        # Step the environment and get the dense reward from our rubric
        _, reward_obj, _, _ = env.step(action)
        
        # GRPO needs a list of floats
        rewards.append(float(reward_obj.value))
        
        # Log for visibility
        action_name = action.action_type.value if action.action_type else "INVALID"
        logger.info(f"Action: {action_name:<15} | Reward: {reward_obj.value:+.2f} | Reason: {reward_obj.reason}")

    return rewards

# ---------------------------------------------------------------------------
# 2. The Training Dataset
# ---------------------------------------------------------------------------
def build_dataset():
    """
    GRPO needs a dataset of prompts to practice on. 
    We will simulate the IC paging L1 about a broken database.
    """
    system_prompt = PROMPTS["L1_Triage"]
    user_prompt = "New message in channel from IC: Audit/Triage: Payment-db is in Error. Please collect and analyze logs to determine the root cause."
    
    # We create 50 identical rows. GRPO will practice this scenario 50 times.
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

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    dataset = build_dataset()

    # --- GRPO Magic Settings ---
    training_args = GRPOConfig(
        output_dir="./grpo_sre_model",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4, # <--- The GRPO magic: Generates 4 answers, compares them, keeps the best!
        max_prompt_length=512,
        max_completion_length=150,
        logging_steps=1,
        max_steps=50, # Keep small for local testing
        report_to="none" # Disable wandb for local runs
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sre_rubric_reward], # Hooking up our CloudSREEnv!
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    logger.info("\n========== STARTING GRPO RL TRAINING ==========")
    trainer.train()
    
    # Save the smart model!
    logger.info("\nSaving trained model to ./grpo_sre_model/final ...")
    trainer.save_model("./grpo_sre_model/final")
    logger.info("Training complete. You can now use this model in inference.py!")

if __name__ == "__main__":
    main()