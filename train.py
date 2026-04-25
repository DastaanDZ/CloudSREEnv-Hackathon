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
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
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
# Reward weighting: Adjusts based on prompt stage
# Early-stage (fresh context): Trust env more (70% env, 30% manual)
# Later-stage (accumulated context): Trust manual more (30% env, 70% manual)
#   because fresh env doesn't reflect actions described in prompt
ENV_WEIGHT_EARLY = 0.70
MANUAL_WEIGHT_EARLY = 0.30
ENV_WEIGHT_LATER = 0.30
MANUAL_WEIGHT_LATER = 0.70

def sre_rubric_reward(prompts, completions, **kwargs):
    """
    Hybrid reward function: 70% environment ground-truth + 30% workflow heuristics.
    
    Environment provides:
    - RBAC enforcement, tool discovery bonuses, collaboration rewards, task completion
    
    Manual heuristics provide:
    - Context-aware workflow guidance (initial alert vs after investigation)
    - Format/syntax checking
    """
    rewards = []
    
    for prompt_str, completion in zip(prompts, completions):
        raw_text = completion[0]['content'] if isinstance(completion, list) else completion
        manual_reward = 0.0  # Accumulates hand-written heuristic rewards
        env_reward = 0.0     # Accumulates environment rewards
        
        # =====================================================================
        # STAGE 1: Hard penalties for refusals (-1.0 immediate return)
        # =====================================================================
        refusal_patterns = ["I cannot", "I'm sorry", "I apologize", "As an AI", 
                          "I'm not able", "I don't", "I can't"]
        if any(p in raw_text for p in refusal_patterns):
            rewards.append(-1.0)
            continue
        
        # =====================================================================
        # STAGE 2: Prose/verbosity penalties (manual)
        # =====================================================================
        prose_indicators = ["**", "Here's", "Let me", "Sure,", "Certainly", 
                          "```", "I will", "I'll", "First,", "To "]
        prose_count = sum(1 for p in prose_indicators if p in raw_text)
        manual_reward -= prose_count * 0.1
        
        # =====================================================================
        # STAGE 3: JSON structure detection (manual)
        # =====================================================================
        if "{" not in raw_text or "}" not in raw_text:
            rewards.append(max(-1.0, manual_reward * MANUAL_WEIGHT_EARLY - 0.5))
            continue
        
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            rewards.append(max(-1.0, manual_reward * MANUAL_WEIGHT_EARLY - 0.4))
            continue
        
        json_str = json_match.group(0)
        json_start_pos = json_match.start()
        
        # Bonus for JSON appearing early (manual)
        if json_start_pos <= 5:
            manual_reward += 0.15
        elif json_start_pos <= 20:
            manual_reward += 0.08
        else:
            manual_reward -= 0.05 * (json_start_pos / 50)
        
        # =====================================================================
        # STAGE 4: JSON parsing and compactness (manual)
        # =====================================================================
        try:
            action_dict = json.loads(json_str)
            manual_reward += 0.15
        except json.JSONDecodeError:
            rewards.append(max(-1.0, manual_reward * MANUAL_WEIGHT_EARLY - 0.3))
            continue
        
        # Reward compact JSON (manual)
        json_length = len(json_str)
        if json_length < 60:
            manual_reward += 0.12
        elif json_length < 100:
            manual_reward += 0.06
        elif json_length > 150:
            manual_reward -= 0.10
        
        if '\n' not in json_str:
            manual_reward += 0.05
        
        # =====================================================================
        # STAGE 5: Action type validation (manual)
        # =====================================================================
        action_type = action_dict.get("action_type", "")
        
        if not action_type:
            rewards.append(max(-1.0, manual_reward * MANUAL_WEIGHT_EARLY - 0.3))
            continue
        
        if action_type in VALID_ACTIONS:
            manual_reward += 0.10
        else:
            manual_reward -= 0.20
        
        # =====================================================================
        # STAGE 6: WORKFLOW-AWARE SCORING (manual heuristics)
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
                
                if is_initial_alert:
                    if target == "L1_Triage":
                        manual_reward += 0.50
                    elif target == "L2_DB_SME":
                        manual_reward += 0.15
                    else:
                        manual_reward -= 0.10
                elif is_after_investigation:
                    if target == "L2_DB_SME":
                        manual_reward += 0.50
                    elif target == "L1_Triage":
                        manual_reward -= 0.15
                else:
                    manual_reward += 0.20
                        
            elif action_type == "CLOSE_INCIDENT":
                if is_after_fix:
                    manual_reward += 0.45
                elif is_initial_alert:
                    manual_reward -= 0.60
                elif is_after_investigation:
                    manual_reward -= 0.40
                else:
                    manual_reward -= 0.30
                    
            elif action_type in ["LIST_SERVICES", "GET_LOGS", "RESTART", "SCALE"]:
                manual_reward -= 0.30
        
        # --- L1_Triage Workflow ---
        elif role == "L1_Triage":
            # More precise detection of investigation stages:
            # - has_service_list: After LIST_SERVICES (saw service table)
            # - has_log_content: After GET_LOGS (saw actual log entries)
            has_service_list = any(kw in prompt_lower for kw in ["running", "error      0ms", "obs:"])
            has_log_content = any(kw in prompt_lower for kw in 
                ["=== logs:", "[error]", "[warn]", "oomkilled", "crashloopbackoff",
                 "cpu usage", "rps=", "503 service unavailable"])
            has_reported = any(kw in prompt_lower for kw in 
                ["found:", "identified", "reports:", "root cause", "report findings"])
            
            if action_type == "LIST_SERVICES":
                if has_log_content or has_reported:
                    # Already have logs/findings - don't list again, report to IC!
                    manual_reward -= 0.40
                elif has_service_list:
                    # Already listed services - should GET_LOGS now, not list again
                    manual_reward -= 0.25
                else:
                    # Fresh investigation - LIST_SERVICES is correct first step
                    manual_reward += 0.45
                    if any(kw in prompt_lower for kw in ["status", "cluster", "what's", "investigate", "check"]):
                        manual_reward += 0.15
                    
            elif action_type == "GET_LOGS":
                service_id = action_dict.get("service_id", "")
                if has_log_content or has_reported:
                    # Already have log content - don't keep getting logs, report to IC!
                    manual_reward -= 0.40
                elif service_id in VALID_SERVICES:
                    # GET_LOGS is correct after LIST_SERVICES or at start
                    manual_reward += 0.45
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        manual_reward += 0.20
                elif service_id:
                    manual_reward -= 0.15
                else:
                    manual_reward -= 0.20
                    
            elif action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                if has_log_content and target == "IC":
                    # Have log content - correct to report findings to IC
                    manual_reward += 0.70
                elif has_reported:
                    # Already reported - don't message again
                    manual_reward -= 0.20
                elif not has_service_list and not has_log_content:
                    # Haven't investigated at all - should investigate first
                    manual_reward -= 0.15
                else:
                    manual_reward += 0.10
                    
            elif action_type == "CLOSE_INCIDENT":
                manual_reward -= 0.60
                
            elif action_type in ["RESTART", "SCALE"]:
                manual_reward -= 0.50
        
        # --- L2_DB_SME Workflow ---
        elif role == "L2_DB_SME":
            fix_already_applied = any(kw in prompt_lower for kw in 
                ["restarted", "scaled", "fix applied", "recovered", "back online"])
            
            if action_type == "RESTART":
                service_id = action_dict.get("service_id", "")
                if fix_already_applied:
                    manual_reward -= 0.25
                elif service_id in VALID_SERVICES:
                    manual_reward += 0.45
                    if any(kw in prompt_lower for kw in ["crash", "error", "down", "oom", "recover"]):
                        manual_reward += 0.20
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        manual_reward += 0.15
                elif service_id:
                    manual_reward -= 0.10
                else:
                    manual_reward -= 0.25
                    
            elif action_type == "SCALE":
                service_id = action_dict.get("service_id", "")
                cpu_value = action_dict.get("cpu_value")
                
                if fix_already_applied:
                    manual_reward -= 0.25
                elif service_id in VALID_SERVICES and isinstance(cpu_value, int):
                    if cpu_value >= 2048:
                        manual_reward += 0.45
                    elif cpu_value >= 1024:
                        manual_reward += 0.20
                    else:
                        manual_reward -= 0.10
                    if any(kw in prompt_lower for kw in ["cpu", "resource", "overload", "performance", "latency", "scale"]):
                        manual_reward += 0.20
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        manual_reward += 0.15
                else:
                    manual_reward -= 0.20
                    
            elif action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                if fix_already_applied and target == "IC":
                    manual_reward += 0.40
                elif not fix_already_applied:
                    manual_reward -= 0.20
                else:
                    manual_reward += 0.05
                    
            elif action_type == "CLOSE_INCIDENT":
                manual_reward -= 0.60
                
            elif action_type in ["LIST_SERVICES", "GET_LOGS"]:
                manual_reward -= 0.25
        
        # =====================================================================
        # STAGE 7: Field completeness validation (manual)
        # =====================================================================
        if action_type in ["GET_LOGS", "RESTART", "SCALE"]:
            if not action_dict.get("service_id"):
                manual_reward -= 0.15
                
        if action_type == "SCALE":
            if not isinstance(action_dict.get("cpu_value"), int):
                manual_reward -= 0.15
                
        if action_type == "MESSAGE_CHANNEL":
            if not action_dict.get("target"):
                manual_reward -= 0.15
            if not action_dict.get("message"):
                manual_reward -= 0.10
        
        # =====================================================================
        # STAGE 8: Environment execution (70% of total reward)
        # =====================================================================
        try:
            action_dict["agent_id"] = role
            env = CloudSREEnv()
            
            # Select scenario based on prompt context
            if "error" in prompt_lower or "crash" in prompt_lower or "oom" in prompt_lower:
                env.reset(task_id="task2_self_healing")
            elif "latency" in prompt_lower or "slow" in prompt_lower or "cpu" in prompt_lower:
                env.reset(task_id="task3_latency_resolution")
            else:
                env.reset(task_id="task1_status_audit")
            
            # Execute action in environment
            action = Action(**action_dict)
            _, reward_obj, done, _ = env.step(action)
            
            # Accumulate ALL environment rewards (will be weighted at 70%)
            for component, value in reward_obj.breakdown.items():
                env_reward += value
            
            # Task completion is a strong env signal
            if done and reward_obj.value > 0:
                env_reward += 0.5
                
        except Exception:
            # If action can't be executed, env gives negative signal
            env_reward -= 0.3
        
        # =====================================================================
        # FINAL: Combine rewards with stage-aware weighting
        # =====================================================================
        # Detect if prompt is early-stage (env state matches) or later-stage (env state is stale)
        is_later_stage = any(kw in prompt_lower for kw in [
            "obs:", "=== logs:", "new message from l1", "new message from l2",
            "restarted", "scaled", "fix applied", "l1_triage reports", "l2_db_sme"
        ])
        
        if is_later_stage:
            # Later-stage: Trust manual heuristics more (env state doesn't match prompt context)
            total_reward = (env_reward * ENV_WEIGHT_LATER) + (manual_reward * MANUAL_WEIGHT_LATER)
        else:
            # Early-stage: Trust env more (fresh env matches INITIAL ALERT context)
            total_reward = (env_reward * ENV_WEIGHT_EARLY) + (manual_reward * MANUAL_WEIGHT_EARLY)
        
        rewards.append(max(-1.0, min(1.0, total_reward)))
    
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
# 2. Multi-Role Dataset with Synthetic Trajectories
# ---------------------------------------------------------------------------
def generate_synthetic_trajectories(num_episodes: int = 50):
    """
    Generate synthetic multi-turn trajectories using expert policy.
    Returns list of (role, prompt) tuples representing each turn.
    """
    trajectories = []
    
    for _ in range(num_episodes):
        env = CloudSREEnv()
        # Randomly pick a task
        task = ["task1_status_audit", "task2_self_healing", "task3_latency_resolution"][torch.randint(0, 3, (1,)).item()]
        obs = env.reset(task_id=task)
        
        # Simulate expert trajectory
        agent_histories = {
            "IC": f"INITIAL ALERT:\n{obs.text_output}",
            "L1_Triage": "",
            "L2_DB_SME": ""
        }
        
        # Turn 1: IC delegates to L1
        trajectories.append(("IC", agent_histories["IC"]))
        agent_histories["L1_Triage"] = "New message from IC: Investigate the incident. Check cluster status."
        
        # Turn 2: L1 lists services
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        list_obs, _, _, _ = env.step(Action(action_type=ActionType.LIST_SERVICES, agent_id="L1_Triage"))
        agent_histories["L1_Triage"] += f"\nObs: {list_obs.text_output}"
        
        # Turn 3: L1 gets logs from problematic service
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        target_svc = "payment-db" if "task1" in task or "task2" in task else "auth-api"
        log_obs, _, _, _ = env.step(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id=target_svc))
        agent_histories["L1_Triage"] += f"\nObs: {log_obs.text_output}"
        
        # Turn 4: L1 reports to IC (should MESSAGE_CHANNEL)
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        finding = f"Root cause: {target_svc} issue found in logs."
        agent_histories["IC"] += f"\nNew message from L1_Triage: {finding}"
        
        # Turn 5: IC delegates to L2
        trajectories.append(("IC", agent_histories["IC"]))
        agent_histories["L2_DB_SME"] = f"New message from IC: Fix {target_svc}. Apply RESTART or SCALE as needed."
        
        # Turn 6: L2 applies fix
        trajectories.append(("L2_DB_SME", agent_histories["L2_DB_SME"]))
        if "task3" in task:
            env.step(Action(action_type=ActionType.SCALE, agent_id="L2_DB_SME", service_id=target_svc, cpu_value=2048))
            fix_msg = f"{target_svc} scaled to 2048 CPU."
        else:
            env.step(Action(action_type=ActionType.RESTART, agent_id="L2_DB_SME", service_id=target_svc))
            fix_msg = f"{target_svc} restarted."
        agent_histories["IC"] += f"\nNew message from L2_DB_SME: Fix applied. {fix_msg}"
        
        # Turn 7: IC closes incident
        trajectories.append(("IC", agent_histories["IC"]))
    
    return trajectories


def build_dataset(num_samples: int = 800):
    """
    Builds a diverse dataset combining:
    1. Static scenario messages from prompts.py
    2. Synthetic multi-turn trajectories from expert policy
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts_list = []
    
    roles = list(PROMPTS.keys())
    
    # Part 1: Static scenario messages (60% of dataset)
    static_samples = int(num_samples * 0.6)
    for _ in range(static_samples):
        role_key = roles[torch.randint(0, len(roles), (1,)).item()]
        messages_for_role = SCENARIO_MESSAGES[role_key]
        msg = messages_for_role[torch.randint(0, len(messages_for_role), (1,)).item()]
        
        messages = [
            {"role": "system", "content": PROMPTS[role_key]},
            {"role": "user", "content": msg}
        ]
        
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts_list.append(prompt_str)
    
    # Part 2: Synthetic trajectories (40% of dataset)
    trajectory_samples = num_samples - static_samples
    num_episodes = trajectory_samples // 7  # ~7 turns per episode
    trajectories = generate_synthetic_trajectories(num_episodes)
    
    for role_key, user_msg in trajectories[:trajectory_samples]:
        messages = [
            {"role": "system", "content": PROMPTS[role_key]},
            {"role": "user", "content": user_msg}
        ]
        
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts_list.append(prompt_str)
    
    # Shuffle to mix static and trajectory samples
    import random
    random.shuffle(prompts_list)
    
    logger.info(f"Built dataset with {len(prompts_list)} examples ({static_samples} static + {len(trajectories)} trajectory)")
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