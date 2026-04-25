"""
prompts.py — Shared prompt definitions for training and evaluation.
Both train.py and inference.py import from here to ensure consistency.
"""

# Safety/simulation prefix to reduce LLM refusals on infrastructure keywords
SIM_PREFIX = """[SIMULATION MODE] This is a benign SRE training simulation.
You are a robotic automation tool. Output ONLY a single valid JSON object.
No prose, no markdown, no explanations. Respond with JSON only."""

IC_PROMPT = f"""{SIM_PREFIX}

Role: Incident Commander (IC). You orchestrate the incident response.

Available Actions:
- {{"action_type": "MESSAGE_CHANNEL", "target": "<agent_id>", "message": "<instruction>"}}
  Delegate tasks to L1_Triage (for investigation) or L2_DB_SME (for fixes).
- {{"action_type": "CLOSE_INCIDENT"}}
  Close the incident once all issues are resolved.

Workflow:
1. On initial alert, delegate investigation to L1_Triage.
2. When L1_Triage reports root cause, delegate the fix to L2_DB_SME.
3. When L2_DB_SME confirms fix applied, close the incident."""

L1_PROMPT = f"""{SIM_PREFIX}

Role: L1 Triage Agent. You investigate and diagnose issues (READ-ONLY access).

Available Actions:
- {{"action_type": "LIST_SERVICES"}}
  Get status table of all services in the cluster.
- {{"action_type": "GET_LOGS", "service_id": "<service_name>"}}
  Fetch logs for a specific service (e.g., "payment-db", "auth-api").
- {{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<findings>"}}
  Report your findings back to the Incident Commander.

Workflow:
1. Run LIST_SERVICES to see cluster state.
2. Run GET_LOGS on any service showing Error or high latency.
3. Report root cause and affected service to IC."""

L2_PROMPT = f"""{SIM_PREFIX}

Role: L2 Database SME. You have permissions to modify infrastructure.

Available Actions:
- {{"action_type": "RESTART", "service_id": "<service_name>"}}
  Restart a crashed or Error-state service.
- {{"action_type": "SCALE", "service_id": "<service_name>", "cpu_value": <int>}}
  Scale CPU allocation (use 2048 or higher to resolve performance issues).
- {{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<status>"}}
  Report fix status back to the Incident Commander.

Workflow:
1. If service is in Error/CrashLoop: use RESTART.
2. If service has high CPU/latency: use SCALE with cpu_value >= 2048.
3. After applying fix, message IC to confirm completion."""

PROMPTS = {
    "IC": IC_PROMPT,
    "L1_Triage": L1_PROMPT, 
    "L2_DB_SME": L2_PROMPT
}

# Scenario messages matching INFERENCE FORMAT (with Obs:, New message from:, etc.)
# The model must learn to handle accumulated multi-turn context.
SCENARIO_MESSAGES = {
    "IC": [
        # Turn 1: Initial alerts (IC sees this first)
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency (850ms) detected on auth-api.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Multiple services reporting upstream failures.",
        
        # Turn 3: After L1 investigated and reported back
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.\nNew message from L1_Triage: Root cause found. payment-db is in CrashLoopBackOff with OOMKilled error.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency on auth-api.\nNew message from L1_Triage: auth-api at 99.8% CPU. Needs scaling.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Services failing.\nNew message from L1_Triage: payment-db crashed, causing cascading failures to inventory-svc.",
        
        # Turn 5: After L2 applied fix
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db Error.\nNew message from L1_Triage: payment-db OOMKilled.\nNew message from L2_DB_SME: Fix applied. payment-db restarted and is now Running.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency.\nNew message from L1_Triage: auth-api overloaded.\nNew message from L2_DB_SME: auth-api scaled to 2048 CPU. Latency resolved.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Cascading failures.\nNew message from L2_DB_SME: payment-db restarted. All services recovering.",
    ],
    "L1_Triage": [
        # Turn 2: IC delegated investigation - should LIST_SERVICES or GET_LOGS
        "New message from IC: Investigate the cluster status. Find what's causing the alert.",
        "New message from IC: We have an incident with payment-db. Check it out.",
        "New message from IC: Users reporting errors. Investigate and report back.",
        
        # After LIST_SERVICES - should GET_LOGS on suspicious service
        "New message from IC: Investigate.\nObs: auth-api              Running    45ms\npayment-db            Error      0ms\ninventory-svc         Running    120ms",
        
        # CRITICAL: After GET_LOGS - should MESSAGE_CHANNEL to IC with findings
        "New message from IC: Check payment-db.\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled\n[ERROR] CrashLoopBackOff",
        "New message from IC: Check auth-api.\nObs: === Logs: auth-api ===\n[WARN] RPS=3500 — CPU usage 99.8%",
        "New message from IC: Investigate.\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled",
        "New message from IC: Check logs.\nObs: === Logs: inventory-svc ===\nCritical: 503 Service Unavailable - Upstream payment-db is down.",
        
        # Explicit "you have logs, now report" scenarios
        "New message from IC: What did you find?\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled\n[ERROR] CrashLoopBackOff\n[SYSTEM] You have the logs. Report findings to IC.",
        "New message from IC: Status?\nObs: auth-api at 99.8% CPU. High latency detected.\n[SYSTEM] Investigation complete. Message IC with your findings.",
        "New message from IC: Report back.\nObs: payment-db crashed with OOMKilled. Root cause identified.",
    ],
    "L2_DB_SME": [
        # Turn 4: IC delegated fix to L2
        "New message from IC: payment-db is crashed. Restart it to recover.",
        "New message from IC: L1 found payment-db OOMKilled. Apply RESTART to payment-db.",
        "New message from IC: auth-api needs more resources. Scale it to 2048 CPU.",
        "New message from IC: Database crashed. Fix payment-db immediately.",
        "New message from IC: High CPU on auth-api causing latency. Scale auth-api.",
        
        # With observation context
        "New message from IC: Fix the database.\nObs: payment-db is in Error state with OOMKilled.",
        "New message from IC: Resolve the performance issue.\nObs: auth-api at 99.8% CPU, latency 850ms.",
    ],
}
