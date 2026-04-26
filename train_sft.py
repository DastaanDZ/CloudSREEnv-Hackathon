"""
train_sft.py - Supervised fine-tuning warm start for CloudSREEnv.

This trains the model to emit the next expert JSON action directly. It is
intended to be evaluated with STRICT_EVAL=True before running any GRPO.
"""

from __future__ import annotations

import json
import logging
import random
from importlib.metadata import PackageNotFoundError, version

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from prompts import PROMPTS
from server.app import Action, ActionType, CloudSREEnv


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./sft_sre_model/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEED = 42
MAX_LENGTH = 1024

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SFTTrainer")


def log_dependency_versions() -> None:
    for package in ("torch", "transformers", "accelerate", "datasets", "peft"):
        try:
            logger.info(f"{package}: {version(package)}")
        except PackageNotFoundError:
            logger.warning(f"{package}: not installed")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_json(action: dict) -> str:
    """Compact JSON action target for supervised labels."""
    return json.dumps(action, separators=(",", ":"))


def add_example(examples: list[dict], role: str, user: str, action: dict) -> None:
    examples.append({
        "role": role,
        "user": user,
        "assistant": action_json(action),
    })


def build_expert_examples(num_episodes: int = 80) -> list[dict]:
    """Build expert next-action examples from the same environment states used in eval."""
    examples: list[dict] = []
    tasks = [
        "task1_tls_certificate_rca",
        "task2_self_healing",
        "task3_latency_resolution",
        "task4_noisy_neighbor",
        "task5_cache_split_brain",
    ]

    for episode_idx in range(num_episodes):
        task = tasks[episode_idx % len(tasks)]
        env = CloudSREEnv()
        obs = env.reset(task_id=task)
        histories = {
            "IC": f"INITIAL ALERT:\n{obs.text_output}",
            "L1_Triage": "",
            "L2_DB_SME": "",
        }

        # IC initial triage delegation.
        if "task1" in task:
            l1_message = "Investigate customer login failures in the authentication flow. Check auth-api logs."
        elif "task4" in task:
            l1_message = "Investigate checkout/payment latency. Start with service status and identify whether payment-db is the root cause or only the victim."
        elif "task5" in task:
            l1_message = "Investigate intermittent cart/session mismatches. Check checkout-api and compare session cache primary/replica epochs."
        else:
            l1_message = "Investigate the incident. Check affected service logs and report root cause."
        add_example(examples, "IC", histories["IC"], {
            "action_type": "MESSAGE_CHANNEL",
            "target": "L1_Triage",
            "message": l1_message,
        })
        histories["L1_Triage"] = f"New message from IC: {l1_message}"

        # Optional service discovery state.
        add_example(examples, "L1_Triage", histories["L1_Triage"], {"action_type": "LIST_SERVICES"})
        list_obs, _, _, _ = env.step(Action(action_type=ActionType.LIST_SERVICES, agent_id="L1_Triage"))
        histories["L1_Triage"] += f"\nObs: {list_obs.text_output}"

        if "task5" in task:
            add_example(examples, "L1_Triage", histories["L1_Triage"], {
                "action_type": "GET_LOGS",
                "service_id": "checkout-api",
            })
            checkout_obs, _, _, _ = env.step(Action(
                action_type=ActionType.GET_LOGS,
                agent_id="L1_Triage",
                service_id="checkout-api",
            ))
            histories["L1_Triage"] += f"\nObs: {checkout_obs.text_output}"

            add_example(examples, "L1_Triage", histories["L1_Triage"], {
                "action_type": "GET_LOGS",
                "service_id": "session-cache-primary",
            })
            primary_obs, _, _, _ = env.step(Action(
                action_type=ActionType.GET_LOGS,
                agent_id="L1_Triage",
                service_id="session-cache-primary",
            ))
            histories["L1_Triage"] += f"\nObs: {primary_obs.text_output}"

            add_example(examples, "L1_Triage", histories["L1_Triage"], {
                "action_type": "GET_LOGS",
                "service_id": "session-cache-replica",
            })
            replica_obs, _, _, _ = env.step(Action(
                action_type=ActionType.GET_LOGS,
                agent_id="L1_Triage",
                service_id="session-cache-replica",
            ))
            histories["L1_Triage"] += f"\nObs: {replica_obs.text_output}"

            l1_report = "Root cause is cache split-brain: session-cache-primary epoch 1842 but session-cache-replica epoch 1837. Repair session-cache-replica."
            add_example(examples, "L1_Triage", histories["L1_Triage"], {
                "action_type": "MESSAGE_CHANNEL",
                "target": "IC",
                "message": l1_report,
            })
            blocked_l1 = (
                histories["L1_Triage"]
                + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
            )
            add_example(examples, "L1_Triage", blocked_l1, {
                "action_type": "MESSAGE_CHANNEL",
                "target": "IC",
                "message": l1_report,
            })

            histories["IC"] += f"\nNew message from L1_Triage: {l1_report}"
            l2_message = "Repair stale split-brain cache replica session-cache-replica."
            fix_action = {"action_type": "REPAIR_REPLICA", "service_id": "session-cache-replica"}
            fix_done_message = "Fix applied. session-cache-replica repaired and synced to primary."

            add_example(examples, "IC", histories["IC"], {
                "action_type": "MESSAGE_CHANNEL",
                "target": "L2_DB_SME",
                "message": l2_message,
            })
            blocked_ic = (
                histories["IC"]
                + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
            )
            add_example(examples, "IC", blocked_ic, {
                "action_type": "MESSAGE_CHANNEL",
                "target": "L2_DB_SME",
                "message": l2_message,
            })
            histories["L2_DB_SME"] = f"New message from IC: {l2_message}"
            add_example(examples, "L2_DB_SME", histories["L2_DB_SME"], fix_action)
            fix_obs, _, _, _ = env.step(Action(agent_id="L2_DB_SME", **fix_action))
            histories["L2_DB_SME"] += f"\nObs: {fix_obs.text_output}"
            add_example(examples, "L2_DB_SME", histories["L2_DB_SME"], {
                "action_type": "MESSAGE_CHANNEL",
                "target": "IC",
                "message": fix_done_message,
            })
            blocked_l2 = (
                histories["L2_DB_SME"]
                + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
            )
            add_example(examples, "L2_DB_SME", blocked_l2, {
                "action_type": "MESSAGE_CHANNEL",
                "target": "IC",
                "message": fix_done_message,
            })
            histories["IC"] += f"\nNew message from L2_DB_SME: {fix_done_message}"
            add_example(examples, "IC", histories["IC"], {"action_type": "CLOSE_INCIDENT"})
            continue

        if "task2" in task:
            target_svc = "payment-db"
        elif "task4" in task:
            payment_db_symptom_history = (
                histories["L1_Triage"]
                + "\nObs: === Logs: payment-db ===\n"
                "[WARN] queries waiting on node memory reclaim\n"
                "[INFO] no CrashLoopBackOff or database corruption detected"
            )
            add_example(examples, "L1_Triage", payment_db_symptom_history, {
                "action_type": "GET_LOGS",
                "service_id": "notification-worker",
            })
            target_svc = "notification-worker"
        else:
            target_svc = "auth-api"
        add_example(examples, "L1_Triage", histories["L1_Triage"], {
            "action_type": "GET_LOGS",
            "service_id": target_svc,
        })
        log_obs, _, _, _ = env.step(Action(
            action_type=ActionType.GET_LOGS,
            agent_id="L1_Triage",
            service_id=target_svc,
        ))
        histories["L1_Triage"] += f"\nObs: {log_obs.text_output}"

        if "task1" in task:
            l1_report = "Root cause is expired upstream TLS certificate on auth-api. No local fix available."
        elif "task2" in task:
            l1_report = "Root cause is payment-db CrashLoopBackOff/OOMKilled. payment-db needs restart."
        elif "task4" in task:
            l1_report = "Root cause is notification-worker noisy-neighbor memory pressure at 8000MB. It is starving payment-db; cap notification-worker to 2048MB."
        else:
            l1_report = "Root cause is auth-api CPU saturation under high RPS. auth-api needs scaling to 2048 CPU."

        add_example(examples, "L1_Triage", histories["L1_Triage"], {
            "action_type": "MESSAGE_CHANNEL",
            "target": "IC",
            "message": l1_report,
        })

        # Strict-mode recovery after the env reports a repeated read action.
        blocked_l1 = (
            histories["L1_Triage"]
            + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
        )
        add_example(examples, "L1_Triage", blocked_l1, {
            "action_type": "MESSAGE_CHANNEL",
            "target": "IC",
            "message": l1_report,
        })

        histories["IC"] += f"\nNew message from L1_Triage: {l1_report}"
        if "task1" in task:
            add_example(examples, "IC", histories["IC"], {"action_type": "CLOSE_INCIDENT"})
            blocked_ic = (
                histories["IC"]
                + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
            )
            add_example(examples, "IC", blocked_ic, {"action_type": "CLOSE_INCIDENT"})
            continue

        if "task2" in task:
            l2_message = "Root cause is payment-db CrashLoopBackOff/OOMKilled. Restart payment-db."
            fix_action = {"action_type": "RESTART", "service_id": "payment-db"}
            fix_done_message = "Fix applied. payment-db restarted and is now Running."
        elif "task4" in task:
            l2_message = "notification-worker is starving payment-db. Set notification-worker memory_limit_mb to 2048."
            fix_action = {"action_type": "UPDATE_CONFIG", "service_id": "notification-worker", "memory_limit_mb": 2048}
            fix_done_message = "Fix applied. notification-worker memory capped at 2048MB and payment-db latency recovered."
        else:
            l2_message = "Root cause is auth-api CPU saturation. Scale auth-api to 2048 CPU."
            fix_action = {"action_type": "SCALE", "service_id": "auth-api", "cpu_value": 2048}
            fix_done_message = "Fix applied. auth-api scaled to 2048 CPU. Latency resolved."

        add_example(examples, "IC", histories["IC"], {
            "action_type": "MESSAGE_CHANNEL",
            "target": "L2_DB_SME",
            "message": l2_message,
        })
        blocked_ic = (
            histories["IC"]
            + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
        )
        add_example(examples, "IC", blocked_ic, {
            "action_type": "MESSAGE_CHANNEL",
            "target": "L2_DB_SME",
            "message": l2_message,
        })

        histories["L2_DB_SME"] = f"New message from IC: {l2_message}"
        add_example(examples, "L2_DB_SME", histories["L2_DB_SME"], fix_action)
        fix_obs, _, _, _ = env.step(Action(agent_id="L2_DB_SME", **fix_action))
        histories["L2_DB_SME"] += f"\nObs: {fix_obs.text_output}"

        add_example(examples, "L2_DB_SME", histories["L2_DB_SME"], {
            "action_type": "MESSAGE_CHANNEL",
            "target": "IC",
            "message": fix_done_message,
        })
        blocked_l2 = (
            histories["L2_DB_SME"]
            + "\nObs: [BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
        )
        add_example(examples, "L2_DB_SME", blocked_l2, {
            "action_type": "MESSAGE_CHANNEL",
            "target": "IC",
            "message": fix_done_message,
        })

        histories["IC"] += f"\nNew message from L2_DB_SME: {fix_done_message}"
        add_example(examples, "IC", histories["IC"], {"action_type": "CLOSE_INCIDENT"})

    return examples


def tokenize_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    prompt_messages = [
        {"role": "system", "content": PROMPTS[example["role"]]},
        {"role": "user", "content": example["user"]},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": example["assistant"]}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )["input_ids"]

    labels = list(tokenized_full["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    tokenized_full["labels"] = labels
    return tokenized_full


def main() -> None:
    seed_everything()
    log_dependency_versions()
    logger.info(f"Loading {MODEL_NAME} onto {DEVICE} for SFT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
    ).to(DEVICE)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    examples = build_expert_examples(num_episodes=120)
    random.shuffle(examples)
    dataset = Dataset.from_list(examples).map(
        lambda row: tokenize_example(row, tokenizer),
        remove_columns=["role", "user", "assistant"],
    )
    logger.info(f"Built SFT dataset with {len(dataset)} expert action examples.")

    training_args = TrainingArguments(
        output_dir="./sft_sre_model",
        # Fits smaller Colab/T4-style GPUs while preserving effective batch size.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    logger.info("\n========== STARTING SFT TRAINING ==========")
    trainer.train()

    logger.info(f"\nSaving SFT adapter to {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("SFT training complete. Set EVAL_MODE = 'SFT' in inference.py to evaluate.")


if __name__ == "__main__":
    main()
