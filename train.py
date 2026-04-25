"""
train.py — Hugging Face TRL GRPO Training Loop for CloudSREEnv (Colab T4 Optimized)
"""

import json
import logging
import random
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# Import our environment and data models
from server.app import CloudSREEnv, Action, ActionType
from prompts import PROMPTS, SCENARIO_MESSAGES


def extract_first_json_object(raw_text: str) -> tuple[dict | None, int]:
    """Extract the first valid JSON object from text. Returns (dict, start_pos) or (None, -1)."""
    start = raw_text.find("{")
    if start == -1:
        return None, -1
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw_text, start)
        if isinstance(obj, dict):
            return obj, start
    except (json.JSONDecodeError, ValueError):
        pass
    return None, -1

# --- Configuration ---
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GRPOTrainer")

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Valid services in the environment (for semantic scoring)
VALID_SERVICES = {"auth-api", "payment-db", "inventory-svc", "notification-worker"}
VALID_ACTIONS = {"LIST_SERVICES", "GET_LOGS", "RESTART", "SCALE", "MESSAGE_CHANNEL", "CLOSE_INCIDENT"}
VALID_TARGETS = {"IC", "L1_Triage", "L2_DB_SME"}

# ---------------------------------------------------------------------------
# 1. Workflow-Aware Reward Function for Multi-Agent SRE
# ---------------------------------------------------------------------------
# Reward weighting: the environment now includes principle-based state rewards
# (new evidence, confidence gain, fixability, remediation readiness), while
# manual heuristics remain as formatting/workflow backup signals.
ENV_WEIGHT_EARLY = 0.70
MANUAL_WEIGHT_EARLY = 0.30
ENV_WEIGHT_LATER = 0.55
MANUAL_WEIGHT_LATER = 0.45

def sre_rubric_reward(prompts, completions, **kwargs):
    """
    Hybrid reward function: principle-based environment signal + workflow heuristics.
    
    Environment provides:
    - RBAC, tool use, task completion, and state-transition rewards
    - Rewards for new evidence, confidence gain, fixability classification, and correct remediation timing
    
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
        action_dict, json_start_pos = extract_first_json_object(raw_text)
        if action_dict is None:
            rewards.append(max(-1.0, manual_reward * MANUAL_WEIGHT_EARLY - 0.5))
            continue
        
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
        json_str = json.dumps(action_dict, separators=(',', ':'))
        manual_reward += 0.15
        
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
        
        # Detect context phases. IC histories always keep the original
        # INITIAL ALERT, so explicit later-state evidence must drive scoring.
        has_l1_message = "new message from l1_triage" in prompt_lower
        task1_cert_evidence = (
            has_l1_message
            and "auth-api" in prompt_lower
            and any(kw in prompt_lower for kw in ["certificate", "expired", "tls", "no local fix"])
        )
        task2_crash_evidence = (
            has_l1_message
            and "payment-db" in prompt_lower
            and any(kw in prompt_lower for kw in ["oomkilled", "crashloopbackoff", "crash", "error"])
        )
        task3_perf_evidence = (
            has_l1_message
            and "auth-api" in prompt_lower
            and any(kw in prompt_lower for kw in ["cpu", "rps", "latency", "scale", "saturated", "overloaded"])
        )
        has_fixable_l1_evidence = task2_crash_evidence or task3_perf_evidence
        is_initial_alert = any(kw in prompt_lower for kw in ["initial alert", "system alert", "alert:", "escalation:", "incident:"])
        is_after_investigation = (
            any(kw in prompt_lower for kw in ["l1_triage reports", "l1_triage found", "l1_triage identified", "l1_triage diagnostic", "root cause"])
            or task1_cert_evidence
            or has_fixable_l1_evidence
        )
        is_after_fix = any(kw in prompt_lower for kw in [
            "l2_db_sme confirms", "l2_db_sme reports", "new message from l2_db_sme",
            "all services now healthy", "fix successfully", "fix applied",
            "restarted and is now", "restarted.", "scaled to"
        ])
        has_cert_log_evidence = "=== logs: auth-api ===" in prompt_lower and any(kw in prompt_lower for kw in ["tls handshake failed", "certificate has expired", "certificate expired"])
        is_investigation_only = has_cert_log_evidence or task1_cert_evidence or any(kw in prompt_lower for kw in ["no local fix", "no fix available", "expired upstream certificate"])
        
        # --- IC Workflow ---
        if role == "IC":
            if action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                
                # Phase priority matters because every IC history keeps the
                # original INITIAL ALERT. Later-state evidence must win.
                if is_after_fix:
                    manual_reward -= 0.55
                    if target in ["L1_Triage", "L2_DB_SME"]:
                        manual_reward -= 0.10
                elif is_investigation_only:
                    if target == "L2_DB_SME":
                        manual_reward -= 0.50
                    elif target == "L1_Triage":
                        manual_reward -= 0.35
                    else:
                        manual_reward -= 0.20
                elif has_fixable_l1_evidence:
                    if target == "L2_DB_SME":
                        manual_reward += 0.70
                    elif target == "L1_Triage":
                        manual_reward -= 0.60
                    else:
                        manual_reward -= 0.20
                elif is_after_investigation:
                    if target == "L2_DB_SME":
                        manual_reward += 0.45
                    elif target == "L1_Triage":
                        manual_reward -= 0.35
                    else:
                        manual_reward -= 0.10
                elif is_initial_alert:
                    if target == "L1_Triage":
                        manual_reward += 0.50
                    elif target == "L2_DB_SME":
                        manual_reward += 0.15
                    else:
                        manual_reward -= 0.10
                else:
                    manual_reward += 0.20
                        
            elif action_type == "CLOSE_INCIDENT":
                if is_after_fix:
                    manual_reward += 0.75
                elif is_after_investigation and is_investigation_only:
                    manual_reward += 0.70
                elif has_fixable_l1_evidence:
                    manual_reward -= 0.45
                elif is_initial_alert:
                    manual_reward -= 0.50
                elif is_after_investigation:
                    manual_reward -= 0.25
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
            is_login_context = any(kw in prompt_lower for kw in ["login failure", "authentication failure", "authentication flow"])
            
            if action_type == "LIST_SERVICES":
                if has_log_content or has_reported:
                    manual_reward -= 0.40
                elif has_service_list:
                    manual_reward -= 0.25
                    if is_login_context:
                        manual_reward -= 0.10
                else:
                    manual_reward += 0.45
                    if any(kw in prompt_lower for kw in ["status", "cluster", "what's", "investigate", "check"]):
                        manual_reward += 0.15
                    
            elif action_type == "GET_LOGS":
                service_id = action_dict.get("service_id", "")
                if has_log_content or has_reported:
                    manual_reward -= 0.60
                elif service_id in VALID_SERVICES:
                    manual_reward += 0.45
                    if service_id.replace("-", "").lower() in prompt_lower.replace("-", ""):
                        manual_reward += 0.20
                    if is_login_context and service_id == "auth-api":
                        manual_reward += 0.25
                elif service_id:
                    manual_reward -= 0.15
                else:
                    manual_reward -= 0.20
                    
            elif action_type == "MESSAGE_CHANNEL":
                target = action_dict.get("target", "")
                if has_log_content and target == "IC":
                    # Have log content - correct to report findings to IC
                    manual_reward += 0.85
                elif has_reported:
                    # Already reported - don't message again
                    manual_reward -= 0.35
                elif not has_log_content:
                    # Haven't investigated at all - L1 must NEVER report without evidence
                    manual_reward -= 0.75
                else:
                    manual_reward -= 0.20
                    
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
                    manual_reward -= 0.45
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
                    manual_reward -= 0.45
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
                    manual_reward += 0.50
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
        # STAGE 8: Environment execution with principle-based state reward
        # =====================================================================
        try:
            action_dict["agent_id"] = role
            env = CloudSREEnv()
            
            # Select scenario based on prompt context
            if "login failure" in prompt_lower or "authentication error" in prompt_lower or "authentication flow" in prompt_lower or "certificate has expired" in prompt_lower:
                env.reset(task_id="task1_tls_certificate_rca")
            elif "error" in prompt_lower or "crash" in prompt_lower or "oom" in prompt_lower:
                env.reset(task_id="task2_self_healing")
            elif "latency" in prompt_lower or "slow" in prompt_lower or "cpu" in prompt_lower:
                env.reset(task_id="task3_latency_resolution")
            else:
                env.reset(task_id="task1_tls_certificate_rca")
            
            # Replay prior actions to sync env state with prompt context
            _prepare_env_for_prompt(env, prompt_lower)
            
            # Execute action in environment (now env state matches prompt)
            action = Action(**action_dict)
            _, reward_obj, done, _ = env.step(action)
            
            # Use the canonical env reward, including principle-based state transitions.
            env_reward = float(reward_obj.value)
            
            # Task completion is a strong env signal
            if done and env_reward > 0:
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
            # Later-stage: env replay now reconstructs evidence/fix state, so trust
            # the principle-based environment signal slightly more than manual rules.
            total_reward = (env_reward * ENV_WEIGHT_LATER) + (manual_reward * MANUAL_WEIGHT_LATER)
        else:
            # Early-stage: Trust env more (fresh env matches INITIAL ALERT context)
            total_reward = (env_reward * ENV_WEIGHT_EARLY) + (manual_reward * MANUAL_WEIGHT_EARLY)
        
        rewards.append(max(-1.0, min(1.0, total_reward)))
    
    return rewards


def _prepare_env_for_prompt(env: CloudSREEnv, prompt_lower: str) -> None:
    """
    Replay prior actions to sync env state with prompt context.
    
    This makes the env's internal state (cloud status, action_history, steps_taken)
    match what the prompt describes, so scoring is accurate.
    """
    def already_replayed(action_type: ActionType, service_id: str | None = None) -> bool:
        return any(
            a.action_type == action_type and (service_id is None or a.service_id == service_id)
            for a in env.action_history
        )

    def replay_once(action: Action) -> None:
        if already_replayed(action.action_type, action.service_id):
            return
        try:
            env.step(action)
        except Exception:
            pass

    # Replay LIST_SERVICES if prompt contains service listing output
    if any(svc in prompt_lower for svc in ["auth-api", "payment-db", "inventory-svc"]) and "running" in prompt_lower:
        replay_once(Action(action_type=ActionType.LIST_SERVICES, agent_id="L1_Triage"))
    
    # Replay GET_LOGS if prompt contains actual log output (=== Logs: <svc> ===)
    if "=== logs:" in prompt_lower or "oomkilled" in prompt_lower:
        if "payment-db" in prompt_lower:
            replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="payment-db"))
        if "=== logs: auth-api ===" in prompt_lower:
            replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="auth-api"))
    
    # IC prompts often contain L1 summaries rather than raw logs. Replay the
    # implied read-only evidence so IC routing actions get accurate env scores.
    if "new message from l1_triage" in prompt_lower:
        if "payment-db" in prompt_lower and any(kw in prompt_lower for kw in ["oomkilled", "crashloopbackoff", "crash", "error"]):
            replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="payment-db"))
        if "auth-api" in prompt_lower and any(kw in prompt_lower for kw in ["cpu", "rps", "latency", "saturated", "overloaded"]):
            replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="auth-api"))
    
    # IC only sees L1's RCA report, not the raw L1 action history. Recreate the
    # required task1 evidence so CLOSE_INCIDENT is scored like inference.
    if ("new message from l1_triage" in prompt_lower
            and "auth-api" in prompt_lower
            and any(kw in prompt_lower for kw in ["expired tls certificate", "expired upstream certificate", "no local fix"])):
        replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="auth-api"))
    
    # Later prompts may mention only the L2 fix. Recreate the diagnosis first so
    # principle rewards know the remediation was evidence-backed, not premature.
    later_fix_context = any(kw in prompt_lower for kw in ["new message from ic", "new message from l2_db_sme", "obs:", "root cause", "fix applied"])
    if later_fix_context and "payment-db" in prompt_lower and any(kw in prompt_lower for kw in ["restarted", "restart", "fix applied", "recovered"]):
        replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="payment-db"))
    if later_fix_context and "auth-api" in prompt_lower and any(kw in prompt_lower for kw in ["scaled", "scale", "fix applied", "cpu saturation", "root cause is auth-api"]):
        replay_once(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id="auth-api"))

    # Replay RESTART if prompt says service was restarted (L2_DB_SME is authorized)
    if "restarted" in prompt_lower or ("fix applied" in prompt_lower and "payment-db" in prompt_lower):
        if "payment-db" in prompt_lower:
            replay_once(Action(action_type=ActionType.RESTART, agent_id="L2_DB_SME", service_id="payment-db"))
    
    # Replay SCALE if prompt says service was scaled (L2_DB_SME is authorized)
    if "scaled" in prompt_lower or ("fix applied" in prompt_lower and "auth-api" in prompt_lower):
        if "auth-api" in prompt_lower:
            replay_once(Action(action_type=ActionType.SCALE, agent_id="L2_DB_SME", service_id="auth-api", cpu_value=2048))
        if "payment-db" in prompt_lower:
            replay_once(Action(action_type=ActionType.SCALE, agent_id="L2_DB_SME", service_id="payment-db", cpu_value=4096))


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
    
    tasks = ["task1_tls_certificate_rca", "task2_self_healing", "task3_latency_resolution"]
    for episode_idx in range(num_episodes):
        env = CloudSREEnv()
        # Round-robin tasks so every training build gets balanced coverage.
        task = tasks[episode_idx % len(tasks)]
        obs = env.reset(task_id=task)
        
        # Simulate expert trajectory
        agent_histories = {
            "IC": f"INITIAL ALERT:\n{obs.text_output}",
            "L1_Triage": "",
            "L2_DB_SME": ""
        }
        
        # Turn 1: IC delegates to L1
        trajectories.append(("IC", agent_histories["IC"]))
        if "task1" in task:
            agent_histories["L1_Triage"] = "New message from IC: Investigate customer login failures in the authentication flow. Check auth-api if needed."
        else:
            agent_histories["L1_Triage"] = "New message from IC: Investigate the incident. Check cluster status."
        
        # Turn 2: L1 lists services
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        list_obs, _, _, _ = env.step(Action(action_type=ActionType.LIST_SERVICES, agent_id="L1_Triage"))
        agent_histories["L1_Triage"] += f"\nObs: {list_obs.text_output}"
        
        # Turn 3: L1 gets logs from problematic service
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        target_svc = "payment-db" if "task2" in task else "auth-api"
        log_obs, _, _, _ = env.step(Action(action_type=ActionType.GET_LOGS, agent_id="L1_Triage", service_id=target_svc))
        agent_histories["L1_Triage"] += f"\nObs: {log_obs.text_output}"
        
        # Turn 4: L1 reports to IC (should MESSAGE_CHANNEL)
        trajectories.append(("L1_Triage", agent_histories["L1_Triage"]))
        
        if "task1" in task:
            # Task1 TLS RCA: investigation-only, no L2 needed
            finding = f"Root cause: {target_svc} has expired TLS certificate. No local fix available."
            agent_histories["IC"] += f"\nNew message from L1_Triage: {finding}"
            # Turn 5: IC closes incident (no remediation possible)
            trajectories.append(("IC", agent_histories["IC"]))
        else:
            # Task2/Task3: full remediation workflow via L2
            if "task2" in task:
                finding = "Root cause: payment-db is in CrashLoopBackOff with OOMKilled errors. It needs RESTART."
            else:
                finding = "Root cause: auth-api CPU is saturated under high RPS, causing latency. It needs SCALE to 2048 CPU."
            agent_histories["IC"] += f"\nNew message from L1_Triage: {finding}"
            
            # Turn 5: IC delegates to L2
            trajectories.append(("IC", agent_histories["IC"]))
            if "task2" in task:
                agent_histories["L2_DB_SME"] = "New message from IC: Restart payment-db to recover from CrashLoopBackOff."
            else:
                agent_histories["L2_DB_SME"] = "New message from IC: Scale auth-api to 2048 CPU to resolve latency."
            
            # Turn 6: L2 applies fix
            trajectories.append(("L2_DB_SME", agent_histories["L2_DB_SME"]))
            if "task3" in task:
                fix_obs, _, _, _ = env.step(Action(action_type=ActionType.SCALE, agent_id="L2_DB_SME", service_id=target_svc, cpu_value=2048))
                fix_msg = f"{target_svc} scaled to 2048 CPU."
            else:
                fix_obs, _, _, _ = env.step(Action(action_type=ActionType.RESTART, agent_id="L2_DB_SME", service_id=target_svc))
                fix_msg = f"{target_svc} restarted."
            agent_histories["L2_DB_SME"] += f"\nObs: {fix_obs.text_output}"
            
            # Turn 7: L2 reports completion to IC
            trajectories.append(("L2_DB_SME", agent_histories["L2_DB_SME"]))
            agent_histories["IC"] += f"\nNew message from L2_DB_SME: Fix applied. {fix_msg}"
            
            # Turn 8: IC closes incident
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
    
    # Part 1: Static scenario messages (50% of dataset)
    static_samples = int(num_samples * 0.5)
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
    
    # Part 2: Synthetic trajectories (50% of dataset)
    trajectory_samples = num_samples - static_samples
    num_episodes = max(1, (trajectory_samples + 5) // 6)  # Generate enough turns, then slice to target.
    trajectories = generate_synthetic_trajectories(num_episodes)
    
    selected_trajectories = trajectories[:trajectory_samples]
    for role_key, user_msg in selected_trajectories:
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
    random.shuffle(prompts_list)
    
    logger.info(f"Built dataset with {len(prompts_list)} examples ({static_samples} static + {len(selected_trajectories)} trajectory)")
    return Dataset.from_dict({"prompt": prompts_list})

# ---------------------------------------------------------------------------
# 3. The Main Training Loop
# ---------------------------------------------------------------------------
def main():
    seed_everything()
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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=4,            # Lower generation count speeds up GRPO on larger GPUs
        generation_batch_size=32,
        logging_steps=10,
        max_steps=200,                # More steps for workflow learning
        report_to="none",
        fp16=True, 
        gradient_checkpointing=False,
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