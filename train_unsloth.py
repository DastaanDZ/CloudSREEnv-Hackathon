"""
train_unsloth.py - Colab-friendly Unsloth LoRA SFT training for CloudSREEnv.

Use this script when Colab GPU memory is tight. It trains the same expert
action dataset as train_sft.py, but loads the base model through Unsloth's
4-bit optimized path.

Colab install:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
"""

from __future__ import annotations

import logging
import random
from importlib.metadata import PackageNotFoundError, version

from datasets import Dataset
from transformers import TrainingArguments

from prompts import PROMPTS
from train_sft import MAX_LENGTH, MODEL_NAME, OUTPUT_DIR, SEED, build_expert_examples


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("UnslothSFT")


def log_dependency_versions() -> None:
    for package in ("unsloth", "torch", "transformers", "trl", "accelerate", "datasets", "peft", "bitsandbytes"):
        try:
            logger.info(f"{package}: {version(package)}")
        except PackageNotFoundError:
            logger.warning(f"{package}: not installed")


def build_text_dataset(tokenizer, num_episodes: int = 120) -> Dataset:
    examples = build_expert_examples(num_episodes=num_episodes)
    random.shuffle(examples)

    rows = []
    for example in examples:
        messages = [
            {"role": "system", "content": PROMPTS[example["role"]]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
        rows.append({
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        })
    return Dataset.from_list(rows)


def main() -> None:
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Unsloth/TRL dependencies are not installed. In Colab, run:\n"
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
            "pip install --no-deps trl peft accelerate bitsandbytes"
        ) from exc

    random.seed(SEED)
    log_dependency_versions()
    logger.info(f"Loading {MODEL_NAME} with Unsloth 4-bit LoRA path...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    dataset = build_text_dataset(tokenizer, num_episodes=120)
    logger.info(f"Built Unsloth SFT dataset with {len(dataset)} expert action examples.")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            output_dir="./sft_sre_model",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            optim="adamw_8bit",
            seed=SEED,
        ),
    )

    logger.info("\n========== STARTING UNSLOTH SFT TRAINING ==========")
    trainer.train()

    logger.info(f"\nSaving Unsloth SFT adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Unsloth SFT training complete. Set EVAL_MODE = 'SFT' in inference.py to evaluate.")


if __name__ == "__main__":
    main()

