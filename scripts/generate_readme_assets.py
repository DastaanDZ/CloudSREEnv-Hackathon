"""
Generate README plot assets for the hackathon submission.

This script only writes images under assets/. It does not import or modify the
environment, training loop, or inference logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
BENCHMARK_PATH = ROOT / "episode_traces" / "benchmark_results.json"


SFT_LOSS_POINTS = [
    0.7984,
    0.2221,
    0.01329,
    0.01692,
    0.0005245,
    0.0002241,
    0.0002414,
    0.00009852,
    0.0002369,
    0.00006665,
    0.0000628,
    0.00005948,
    0.00005692,
    0.0000507,
    0.00004422,
    0.00004727,
    0.00003871,
    0.00004042,
    0.00003748,
    0.000035,
    0.00003694,
    0.0000315,
    0.00003852,
    0.00002922,
    0.00002985,
    0.00002747,
    0.0000277,
    0.0000267,
    0.00002903,
    0.00002741,
    0.00002534,
    0.0000241,
    0.00002711,
    0.00002693,
    0.00002709,
    0.0000253,
    0.00002544,
    0.00002604,
    0.00002515,
    0.00002718,
    0.0000262,
    0.0000225,
    0.00002698,
]


def load_benchmark_results() -> dict:
    if not BENCHMARK_PATH.exists():
        return {}
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_loss_plot() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = list(range(1, len(SFT_LOSS_POINTS) + 1))
    ax.plot(steps, SFT_LOSS_POINTS, marker="o", linewidth=2)
    ax.set_title("SFT Training Loss")
    ax.set_xlabel("Logged training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "sft_training_loss.png", dpi=160)
    plt.close(fig)


def save_pass_rate_plot(benchmark: dict) -> None:
    labels = ["BASE", "SFT"]
    default_rates = [0.0, 1.0]
    rates = [
        benchmark.get("base_in_template", {}).get("pass_rate", default_rates[0]),
        benchmark.get("sft_in_template", {}).get("pass_rate", default_rates[1]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, [r * 100 for r in rates])
    ax.set_title("Strict Evaluation Pass Rate")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    subtitle = "Uses benchmark_results.json when present; defaults show observed 3-task baseline/SFT comparison."
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ASSETS_DIR / "base_vs_sft_pass_rate.png", dpi=160)
    plt.close(fig)


def save_heldout_plot(benchmark: dict) -> None:
    labels = ["SFT in-template", "SFT held-out"]
    rates = [
        benchmark.get("sft_in_template", {}).get("pass_rate", 1.0),
        benchmark.get("sft_heldout", {}).get("pass_rate", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, [r * 100 for r in rates])
    ax.set_title("Generalization Check")
    ax.set_xlabel("Evaluation split")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    if "sft_heldout" not in benchmark:
        fig.text(0.5, 0.01, "Held-out result TBD: run strict held-out benchmark before final submission.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ASSETS_DIR / "heldout_pass_rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_results()
    save_loss_plot()
    save_pass_rate_plot(benchmark)
    save_heldout_plot(benchmark)
    print(f"Generated README assets in {ASSETS_DIR}")


if __name__ == "__main__":
    main()

