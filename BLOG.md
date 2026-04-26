# CloudSREEnv: Training LLM Agents To Handle Real SRE Incidents

## The Problem

Modern LLM agents are getting good at calling tools, but real operational work is not just tool calling. In production incident response, the first visible symptom is often misleading. A database may be slow because a background worker is starving the node. A checkout service may return bad carts because a cache replica is serving stale state. Restarting the obvious service can waste time or make the incident worse.

CloudSREEnv was built for the OpenEnv x Meta Developers Hackathon to test whether an LLM can learn a more realistic SRE workflow:

- investigate before acting;
- identify root cause instead of chasing symptoms;
- coordinate between agents with different permissions;
- apply the right remediation;
- confirm recovery and close the incident.

The environment is not a game clone. It is a partially observable cloud simulator for a professional workflow that LLM agents are already expected to perform in the real world.

## Hackathon Theme Fit

CloudSREEnv is primarily aligned with three hackathon themes.

### Theme 1: Multi-Agent Interactions

The incident is handled by three roles:

- `IC`: Incident Commander, responsible for coordination and closure.
- `L1_Triage`: read-only investigator, responsible for gathering evidence.
- `L2_DB_SME`: remediation specialist, responsible for infrastructure changes.

The model must learn role boundaries. L1 cannot restart or scale services. IC should not directly change infrastructure. L2 should not close incidents. This creates a real multi-agent coordination task where success depends on handoffs, not just choosing the right tool.

### Theme 2: Long-Horizon Planning And Instruction Following

Each incident requires a sequence of decisions. A successful trajectory may include listing services, choosing the right logs, reporting findings, delegating remediation, applying the fix, reporting completion, and closing the incident.

This pushes the agent beyond shallow one-step actions. If it skips evidence gathering, repeats a tool call, delegates to the wrong role, or closes too early, the environment rejects the solution.

### Theme 3: World Modeling For Professional Tasks

The environment contains a dynamic microservice world with service health, logs, latency, memory, CPU, dependencies, and consistency state. The agent must update its belief after each observation.

For example, `payment-db` latency can be a database crash in one task, a CPU-independent symptom in another task, or a victim of node-level memory pressure in the noisy-neighbor task. The same service name does not always imply the same fix.

## What The Agent Sees And Does

The agent interacts through a small JSON action space:

- `LIST_SERVICES`
- `GET_LOGS`
- `RESTART`
- `SCALE`
- `UPDATE_CONFIG`
- `REPAIR_REPLICA`
- `MESSAGE_CHANNEL`
- `CLOSE_INCIDENT`

Observations include readable incident output and structured service metrics. The agent never sees the hidden answer. It must infer the root cause from service state and logs.

## The Five Tasks

### Task 1: TLS Certificate RCA

Users cannot log in because `auth-api` is failing TLS handshakes due to an expired certificate.

The correct behavior is not to restart anything. The agent should inspect `auth-api`, identify the certificate issue, report that no local remediation is available, and close the incident.

This task tests whether the agent can avoid unnecessary remediation.

### Task 2: Self-Healing Payment DB

`payment-db` is in `CrashLoopBackOff` with `OOMKilled` logs. The right response is to restart the database through the L2 remediation role.

This task teaches the standard incident path: gather evidence, delegate to the right specialist, apply a fix, report completion, and close.

### Task 3: Auth API CPU Saturation

`auth-api` is running but slow under high request volume. Logs show high RPS and CPU saturation.

The correct fix is to scale `auth-api` CPU to `2048`, not restart it and not scale the database.

This task tests whether the agent can distinguish a running-but-overloaded service from a crashed one.

### Task 4: Noisy Neighbor Resource Contention

Checkout latency appears on `payment-db`, but `payment-db` is only the symptom. The real root cause is `notification-worker`, a low-priority background service using `8000MB` of memory and starving the node.

The trained agent must inspect the service table, notice abnormal worker memory usage, gather `notification-worker` logs, and apply:

```json
{"action_type":"UPDATE_CONFIG","service_id":"notification-worker","memory_limit_mb":2048}
```

This is intentionally hard because an untrained model often scales or restarts `payment-db`. That action is plausible, but wrong.

### Task 5: Split-Brain Cache Consistency

Checkout users see intermittent cart and session mismatches. The visible symptom is `checkout-api`, but the root cause is a split-brain cache cluster.

The agent must compare `session-cache-primary` and `session-cache-replica`, discover divergent `cache_epoch` values, and repair the stale replica:

```json
{"action_type":"REPAIR_REPLICA","service_id":"session-cache-replica"}
```

This task tests distributed-systems reasoning. Restarting `checkout-api` does not solve stale cache state.

## Reward Design

CloudSREEnv uses dense, composable rewards so the model receives learning signal throughout the episode, not only at the final close step.

The reward function encourages:

- targeted diagnostics;
- evidence-based root-cause reporting;
- correct role handoffs;
- RBAC-compliant remediation;
- task-specific causal evidence such as memory pressure or divergent cache epochs;
- successful closure after the simulator state has recovered.

It penalizes:

- invalid JSON;
- hallucinated services;
- unauthorized remediation by the wrong role;
- duplicate actions;
- premature closure;
- symptom-chasing fixes such as `SCALE payment-db` in Task 4 or `RESTART checkout-api` in Task 5.

This makes the environment hard to game. The agent cannot get a high score by repeatedly calling tools or closing early.

## Training Pipeline

The training pipeline uses Hugging Face tooling and is designed to run in Colab.

The current flow is:

1. Load `Qwen/Qwen2.5-3B-Instruct`.
2. Attach LoRA adapters with PEFT.
3. Generate expert trajectories from the OpenEnv environment.
4. Run completion-only supervised fine-tuning so only the assistant JSON action contributes to loss.
5. Optionally run HF TRL GRPO against the environment reward.
6. Evaluate raw BASE and TRAINED behavior on the same five tasks.
7. Save logs and plots for judging.

The important point is that the plots are generated by the code after real training and evaluation runs. They are not manually drawn from pasted logs.

Run:

```bash
uv run python train.py
uv run python evaluate_compare.py
```

Generated artifacts:

- `training_logs/sft_trainer_log_history.json`
- `training_logs/sft_loss_history.csv`
- `training_logs/base_evaluation_summary.json`
- `training_logs/trained_evaluation_summary.json`
- `training_logs/base_vs_trained_comparison.csv`
- `assets/sft_loss_curve.png`
- `assets/before_after_success.png`
- `assets/reward_comparison.png`

## What Improvement Looks Like

The base model tends to produce reasonable-sounding but operationally wrong behavior. It often:

- repeats diagnostics after evidence is already available;
- asks the wrong role to act;
- restarts a symptom service;
- scales `payment-db` during noisy-neighbor memory pressure;
- repairs or restarts the wrong service in split-brain cache incidents;
- fails to close after a fix has already been applied.

After training, the desired behavior is visible in the trajectory:

- L1 gathers the right evidence.
- IC delegates to L2 only after root cause is reported.
- L2 applies the correct task-specific remediation.
- L2 reports the fix instead of repeating it.
- IC closes only after recovery.

For judges, the key comparison is the raw BASE vs TRAINED success and reward plots generated by `evaluate_compare.py`. These show whether training changed how the model acts in the environment.

## Why This Matters

SRE work is a strong testbed for LLM agents because it requires causal reasoning under partial observability. A tool-calling model can look impressive while still making dangerous operational choices. CloudSREEnv makes those choices measurable.

The environment teaches an agent to ask:

- What evidence do I have?
- Is this service the root cause or only the symptom?
- Which role is allowed to act?
- Did the system actually recover?
- Is it safe to close?

That is the difference between a chatbot that can call APIs and an agent that can participate in real incident response.

## Submission Checklist

- OpenEnv-compliant environment with `reset`, `step`, and deterministic terminal grading.
- Hugging Face Space deployment using Docker.
- Colab-compatible training script using HF TRL and PEFT.
- Generated training logs and loss plots.
- Generated BASE vs TRAINED comparison plots.
- README with Space, Colab, and blog/video links.

## Links

- Hugging Face Space: TODO
- Colab notebook: TODO
- GitHub repository: TODO
- Demo video or slides: TODO
