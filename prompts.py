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
- {{"action_type": "MESSAGE_CHANNEL", "target": "L1_Triage", "message": "<instruction>"}}
  Delegate investigation to L1_Triage.
- {{"action_type": "MESSAGE_CHANNEL", "target": "L2_DB_SME", "message": "<instruction>"}}
  Delegate fixes to L2_DB_SME only for remediable crash, scaling, or config issues.
- {{"action_type": "CLOSE_INCIDENT"}}
  Close the incident once all issues are resolved.

Workflow:
1. On initial alert, delegate investigation to L1_Triage.
2. When L1_Triage reports root cause:
   - If the issue requires remediation (crash, scaling, config), delegate the fix to L2_DB_SME.
   - If the issue is external or non-remediable (e.g., expired certificate), close the incident after documenting the RCA.
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
2. Run GET_LOGS on any service showing Error, high latency, or warnings.
   If users report login/authentication failures, inspect auth-api logs even if status shows Running.
   If payment-db is slow but another service has extreme memory usage, report that noisy neighbor as root cause.
3. Report root cause and affected service to IC."""

L2_PROMPT = f"""{SIM_PREFIX}

Role: L2 Database SME. You have permissions to modify infrastructure.

Available Actions:
- {{"action_type": "RESTART", "service_id": "<service_name>"}}
  Restart a crashed or Error-state service.
- {{"action_type": "SCALE", "service_id": "<service_name>", "cpu_value": <int>}}
  Scale CPU allocation (use 2048 or higher to resolve performance issues).
- {{"action_type": "UPDATE_CONFIG", "service_id": "<service_name>", "memory_limit_mb": <int>}}
  Apply a strict memory limit to a noisy neighbor service.
- {{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<status>"}}
  Report fix status back to the Incident Commander.

Workflow:
1. If service is in Error/CrashLoop: use RESTART.
2. If service has high CPU/latency: use SCALE with cpu_value >= 2048.
3. If a low-priority worker is hogging node memory: use UPDATE_CONFIG on that worker with memory_limit_mb <= 2048.
4. After applying fix, message IC to confirm completion."""

PROMPTS = {
    "IC": IC_PROMPT,
    "L1_Triage": L1_PROMPT, 
    "L2_DB_SME": L2_PROMPT
}

# Scenario messages matching INFERENCE FORMAT (with Obs:, New message from:, etc.)
# The model must learn to handle accumulated multi-turn context.
SCENARIO_MESSAGES = {
    "IC": [
        # Task1: Login failures (symptom-only alert, root cause hidden in logs)
        "INITIAL ALERT:\n[SYSTEM ALERT] Login failures reported across customer-facing authentication flow.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Increased authentication error rate detected for user login requests.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Login failures reported across customer-facing authentication flow.\nNew message from L1_Triage: Root cause found. auth-api has expired upstream certificate. No local fix available.",
        
        # Task2: payment-db crash (needs RESTART)
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.\nNew message from L1_Triage: Root cause found. payment-db is in CrashLoopBackOff with OOMKilled error.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.\nNew message from L1_Triage: payment-db logs show OOMKilled and CrashLoopBackOff. This requires L2 remediation.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db status transition to Error detected.\nNew message from L1_Triage: I already checked logs. Root cause is payment-db CrashLoopBackOff/OOMKilled. Delegate restart to L2_DB_SME.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db Error.\nNew message from L1_Triage: payment-db OOMKilled.\nNew message from L2_DB_SME: Fix applied. payment-db restarted and is now Running.",
        
        # Task3: High latency (needs SCALE) - IC must delegate to L2 after L1 report
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency (850ms) detected on auth-api.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency on auth-api.\nNew message from L1_Triage: auth-api at 99.8% CPU. Needs scaling.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency on auth-api.\nNew message from L1_Triage: auth-api CPU overloaded, latency 850ms. Needs scaling to 2048 CPU.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency on auth-api.\nNew message from L1_Triage: I already checked logs. Root cause is auth-api CPU saturation under high RPS. Delegate scaling to L2_DB_SME.",
        "INITIAL ALERT:\n[SYSTEM ALERT] High latency.\nNew message from L1_Triage: auth-api overloaded.\nNew message from L2_DB_SME: auth-api scaled to 2048 CPU. Latency resolved.",

        # Task4: Noisy neighbor worker memory leak (needs UPDATE_CONFIG)
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout latency detected on payment-db.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout latency detected on payment-db.\nNew message from L1_Triage: Root cause found. notification-worker is using 8000MB RAM and starving payment-db.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout latency detected on payment-db.\nNew message from L1_Triage: payment-db is throttled by node memory pressure from notification-worker. Apply a strict memory limit to notification-worker.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout latency.\nNew message from L1_Triage: notification-worker is the noisy neighbor at 8000MB RAM.\nNew message from L2_DB_SME: Fix applied. notification-worker memory limit set to 2048MB and payment-db latency recovered.",
    ],
    "L1_Triage": [
        # Initial investigation prompts
        "New message from IC: Investigate the cluster status. Find what's causing the alert.",
        "New message from IC: Users reporting login failures. Investigate the authentication flow.",
        "New message from IC: Users reporting errors. Investigate and report back.",
        "New message from IC: Checkout latency on payment-db. First run LIST_SERVICES and look for noisy neighbors before checking DB logs.",
        
        # Task1: After LIST_SERVICES with login context - auth-api shows high latency, should GET_LOGS(auth-api)
        "New message from IC: Investigate login failures in the authentication flow.\nObs: auth-api              Running    350ms\npayment-db            Running    12ms\ninventory-svc         Running    45ms",
        "New message from IC: Users reporting authentication failures. Check auth-api.\nObs: auth-api              Running    350ms\npayment-db            Running    12ms\ninventory-svc         Running    45ms",
        
        # Task2: After LIST_SERVICES - payment-db shows Error
        "New message from IC: Investigate.\nObs: auth-api              Running    45ms\npayment-db            Error      0ms\ninventory-svc         Running    120ms",
        
        # Task1: TLS Certificate logs - should MESSAGE_CHANNEL to IC (no fix needed)
        "New message from IC: Check auth-api.\nObs: === Logs: auth-api ===\n[ERROR] TLS handshake failed: certificate has expired\n[ERROR] x509: certificate signed by unknown authority",
        "New message from IC: Check auth-api logs.\nObs: === Logs: auth-api ===\n[ERROR] TLS handshake failed: certificate has expired\n[WARN] Upstream certificate expired 2 days ago",
        
        # Task2: payment-db crash logs
        "New message from IC: Check payment-db.\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled\n[ERROR] CrashLoopBackOff",
        
        # Task3: High CPU logs
        "New message from IC: Check auth-api.\nObs: === Logs: auth-api ===\n[WARN] RPS=3500 — CPU usage 99.8%",

        # Task4: Noisy neighbor memory contention
        "New message from IC: Checkout latency on payment-db. Investigate all services.\nObs: auth-api              Running    CPU=1024m MEM=2048MB LAT=370ms\npayment-db            Running    CPU=2048m MEM=4096MB LAT=650ms\ninventory-svc         Running    CPU=1024m MEM=2048MB LAT=370ms\nnotification-worker   Running    CPU=1024m MEM=8000MB LAT=180ms",
        "New message from IC: payment-db is slow but CPU looks normal.\nObs: === Logs: notification-worker ===\n[WARN] Heap growth detected: notification batch cache at 8000MB\n[WARN] Node memory pressure: worker has no memory limit",
        
        # Explicit "you have logs, now report" scenarios
        "New message from IC: What did you find?\nObs: === Logs: auth-api ===\n[ERROR] TLS handshake failed: certificate has expired\n[SYSTEM] Certificate issue identified. Report to IC.",
        "New message from IC: What did you find?\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled\n[ERROR] CrashLoopBackOff\n[SYSTEM] You have the logs. Report findings to IC.",
        "New message from IC: Status?\nObs: auth-api at 99.8% CPU. High latency detected.\n[SYSTEM] Investigation complete. Message IC with your findings.",
        "New message from IC: What did you find?\nObs: notification-worker is using 8000MB RAM while payment-db is throttled by node memory pressure.\n[SYSTEM] Investigation complete. Report noisy neighbor root cause to IC.",
    ],
    "L2_DB_SME": [
        # Task2: payment-db crash fix
        "New message from IC: payment-db is crashed. Restart it to recover.",
        "New message from IC: L1 found payment-db OOMKilled. Apply RESTART to payment-db.",
        "New message from IC: Database crashed. Fix payment-db immediately.",
        "New message from IC: Fix the database.\nObs: payment-db is in Error state with OOMKilled.",
        "New message from IC: Restart payment-db to recover from CrashLoopBackOff.\nObs: [OK] payment-db restarted by L2_DB_SME.",
        
        # Task3: auth-api scaling fix
        "New message from IC: auth-api needs more resources. Scale it to 2048 CPU.",
        "New message from IC: High CPU on auth-api causing latency. Scale auth-api.",
        "New message from IC: Resolve the performance issue.\nObs: auth-api at 99.8% CPU, latency 850ms.",
        "New message from IC: Scale auth-api to 2048 CPU to resolve latency.\nObs: [OK] auth-api scaled to 2048m CPU.",

        # Task4: notification-worker config fix
        "New message from IC: notification-worker is using 8000MB RAM and starving payment-db. Set a strict memory limit.",
        "New message from IC: Apply UPDATE_CONFIG to notification-worker with memory_limit_mb 2048.",
        "New message from IC: Noisy neighbor worker is causing DB latency.\nObs: notification-worker MEM=8000MB, payment-db LAT=650ms.",
        "New message from IC: Set notification-worker memory limit to 2048MB.\nObs: [OK] notification-worker memory limit set to 2048MB.",
    ],
}
