import json
import re
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
json_file = ROOT_DIR / "episode_traces" / "unsloth_training_metrics.json"
legacy_log_file = ROOT_DIR / "episode_traces" / "sft_train.log"
output_file = ROOT_DIR / "assets" / "unsloth_train_loss_epoch.png"
reward_output_file = ROOT_DIR / "assets" / "unsloth_sft_reward_proxy_epoch.png"


def load_json_history(path: Path):
    """Load clean JSON history: [{"step": 10, "epoch": ..., "loss": ..., ...}, ...]."""
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of metric dictionaries.")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "epoch" not in item or "loss" not in item:
            continue
        rows.append({
            "step": int(item.get("step", len(rows) + 1)),
            "epoch": float(item["epoch"]),
            "loss": float(item["loss"]),
            "sft_reward_proxy": (
                float(item["sft_reward_proxy"])
                if item.get("sft_reward_proxy") is not None
                else None
            ),
        })
    return rows


def load_legacy_log_history(path: Path):
    """Fallback parser for old stdout logs containing {'loss': ..., 'epoch': ...}."""
    if not path.exists():
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            if "{'loss':" in line and "'epoch':" in line:
                try:
                    match = re.search(r"(\{.*?'loss':.*?'epoch':.*?\})", line)
                    if match:
                        data = ast.literal_eval(match.group(1))
                        rows.append({
                            "step": int(data.get("step", len(rows) + 1)),
                            "epoch": float(data["epoch"]),
                            "loss": float(data["loss"]),
                            "sft_reward_proxy": None,
                        })
                except Exception:
                    pass

    return rows


def split_runs(rows):
    """Split into runs if epoch resets."""
    runs = []
    current = []

    for row in rows:
        if current and row["epoch"] < current[-1]["epoch"]:
            runs.append(current)
            current = []
        current.append(row)

    if current:
        runs.append(current)
    return runs


def plot_loss(runs):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for i, run in enumerate(runs):
        epochs = [row["epoch"] for row in run]
        losses = [row["loss"] for row in run]
        plt.plot(epochs, losses, marker="o", linestyle="-", markersize=4, label=f"Training Run {i + 1}")

    plt.title("Unsloth SFT Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"Saved loss plot to {output_file}")


def plot_reward_proxy(runs):
    has_reward = any(row["sft_reward_proxy"] is not None for run in runs for row in run)
    if not has_reward:
        return

    reward_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for i, run in enumerate(runs):
        filtered = [row for row in run if row["sft_reward_proxy"] is not None]
        if not filtered:
            continue
        epochs = [row["epoch"] for row in filtered]
        rewards = [row["sft_reward_proxy"] for row in filtered]
        plt.plot(epochs, rewards, marker="o", linestyle="-", markersize=4, label=f"Training Run {i + 1}")

    plt.title("Unsloth SFT Reward Proxy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SFT Reward Proxy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_output_file, dpi=180, bbox_inches="tight")
    print(f"Saved reward proxy plot to {reward_output_file}")


def main():
    rows = load_json_history(json_file)
    source = json_file

    if not rows:
        rows = load_legacy_log_history(legacy_log_file)
        source = legacy_log_file

    if not rows:
        print(
            "No valid training data found. Expected clean JSON at "
            f"{json_file} with fields step, epoch, loss, sft_reward_proxy."
        )
        return

    runs = split_runs(rows)
    print(f"Loaded {len(rows)} points from {source}")
    plot_loss(runs)
    plot_reward_proxy(runs)


if __name__ == "__main__":
    main()