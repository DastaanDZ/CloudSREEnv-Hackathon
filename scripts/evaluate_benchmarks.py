"""
Run BASE/SFT strict benchmarks and save a README-friendly summary.

This script does not modify core environment or inference logic. For held-out
evaluation, it wraps only the initial alert text while preserving the same
environment state and deterministic terminal graders.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import inference
from server.app import CloudSREEnv, Observation


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "episode_traces" / "benchmark_results.json"
TASKS = [
    "task1_tls_certificate_rca",
    "task2_self_healing",
    "task3_latency_resolution",
    "task4_noisy_neighbor",
    "task5_cache_split_brain",
]


def heldout_alert(task_id: str, original: str) -> str:
    """Use unseen wording/numbers to reduce memorization risk in evaluation."""
    if "task1" in task_id:
        return random.choice([
            "[SYSTEM ALERT] Customer login auth handshakes are failing intermittently.",
            "[SYSTEM ALERT] Production authentication requests show a sharp failure spike.",
        ])
    if "task2" in task_id:
        return random.choice([
            "[SYSTEM ALERT] payment-db backend is unhealthy and repeatedly entering Error.",
            "[SYSTEM ALERT] Payment database pod instability detected.",
        ])
    if "task3" in task_id:
        latency = random.choice([720, 880, 940])
        return f"[SYSTEM ALERT] auth-api latency elevated to approximately {latency}ms."
    if "task4" in task_id:
        latency = random.choice([610, 760, 930])
        return f"[SYSTEM ALERT] Checkout payment path degraded; payment-db query latency near {latency}ms."
    if "task5" in task_id:
        expected = random.choice([100, 140, 180])
        observed = expected - random.choice([12, 18, 25])
        return (
            "[SYSTEM ALERT] Intermittent checkout cart/session mismatch: "
            f"expected_total={expected}, observed_total={observed}."
        )
    return original


class HeldoutAlertEnv(CloudSREEnv):
    def reset(self, task_id: str | None = None, scenario: str | None = None) -> Observation:
        obs = super().reset(task_id=task_id, scenario=scenario)
        return Observation(
            text_output=heldout_alert(task_id or self.current_task, obs.text_output),
            structured_data=obs.structured_data,
        )


def pass_rate(results: dict[str, bool]) -> float:
    return sum(1 for value in results.values() if value) / max(1, len(results))


def run_suite(mode: str, heldout: bool) -> dict:
    inference.EVAL_MODE = mode
    inference.STRICT_EVAL = True
    model, tokenizer = inference.load_eval_model(mode=mode)
    env_cls = HeldoutAlertEnv if heldout else CloudSREEnv

    results = {}
    for task_id in TASKS:
        results[task_id] = inference.run_multi_agent_task(env_cls(), task_id, model, tokenizer)

    return {
        "eval_mode": mode,
        "strict_eval": True,
        "heldout_eval": heldout,
        "results": results,
        "pass_rate": pass_rate(results),
    }


def main() -> None:
    random.seed(42)
    runs = {
        "base_in_template": ("BASE", False),
        "base_heldout": ("BASE", True),
        "sft_in_template": ("SFT", False),
        "sft_heldout": ("SFT", True),
    }

    summary = {}
    for name, (mode, heldout) in runs.items():
        summary[name] = run_suite(mode, heldout)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary to {OUT_PATH}")


if __name__ == "__main__":
    main()

