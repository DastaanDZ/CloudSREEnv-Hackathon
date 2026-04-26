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
  Delegate fixes to L2_DB_SME for remediable crash, scaling, config, or replica repair issues.
- {{"action_type": "CLOSE_INCIDENT"}}
  Close the incident once all issues are resolved.

Workflow:
1. On initial alert, delegate investigation to L1_Triage.
2. When L1_Triage reports root cause:
   - If the issue requires remediation (crash, scaling, config, replica repair), delegate the fix to L2_DB_SME.
   - Do not message L1_Triage again after L1 has reported a fixable root cause.
   - If the issue is external or non-remediable (e.g., expired certificate), close the incident after documenting the RCA.
3. When L2_DB_SME confirms fix applied, close the incident."""

L1_PROMPT = f"""{SIM_PREFIX}

Role: L1 Triage Agent. You investigate and diagnose issues (READ-ONLY access).

Available Actions:
- {{"action_type": "LIST_SERVICES"}}
  Get status table of all services in the cluster.
- {{"action_type": "GET_LOGS", "service_id": "<service_name>"}}
  Fetch logs for a specific service (e.g., "payment-db", "auth-api", "notification-worker", "session-cache-replica").
- {{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<findings>"}}
  Report your findings back to the Incident Commander.

Workflow:
1. Run LIST_SERVICES to see cluster state.
2. Run GET_LOGS on any service showing Error, high latency, or warnings.
   If users report login/authentication failures, inspect auth-api logs even if status shows Running.
   If symptoms point at a victim service, inspect related services before choosing a root cause.
   For cache/session mismatches, inspect checkout-api, session-cache-primary, and session-cache-replica before reporting.
3. Do not repeat GET_LOGS for the same service. You may inspect a different related service if diagnosis requires comparison.
4. Report root cause and affected service to IC with MESSAGE_CHANNEL."""

L2_PROMPT = f"""{SIM_PREFIX}

Role: L2 Database SME. You have permissions to modify infrastructure.

Available Actions:
- {{"action_type": "RESTART", "service_id": "<service_name>"}}
  Restart a crashed or Error-state service.
- {{"action_type": "SCALE", "service_id": "<service_name>", "cpu_value": <int>}}
  Scale CPU allocation (use 2048 or higher to resolve performance issues).
- {{"action_type": "UPDATE_CONFIG", "service_id": "<service_name>", "memory_limit_mb": <int>}}
  Apply a memory limit to a noisy-neighbor service.
- {{"action_type": "REPAIR_REPLICA", "service_id": "<service_name>"}}
  Repair a stale cache replica after split-brain.
- {{"action_type": "MESSAGE_CHANNEL", "target": "IC", "message": "<status>"}}
  Report fix status back to the Incident Commander.

Workflow:
1. If service is in Error/CrashLoop: use RESTART.
2. If service has high CPU/latency: use SCALE with cpu_value >= 2048.
3. If notification-worker is starving payment-db, use UPDATE_CONFIG on notification-worker with memory_limit_mb <= 2048.
4. If session-cache-replica is stale, use REPAIR_REPLICA on session-cache-replica.
5. After applying fix, message IC to confirm completion."""

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

        # Task4: noisy neighbor memory pressure
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout/payment latency: payment-db queries are slow.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout latency on payment flow.\nNew message from L1_Triage: notification-worker is using 8000MB and causing node memory pressure that throttles payment-db. Cap notification-worker memory.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db latency.\nNew message from L1_Triage: payment-db is only the victim. Root cause is notification-worker noisy neighbor memory pressure. Delegate UPDATE_CONFIG notification-worker memory_limit_mb 2048.",
        "INITIAL ALERT:\n[SYSTEM ALERT] payment-db latency.\nNew message from L1_Triage: notification-worker is starving payment-db.\nNew message from L2_DB_SME: Fix applied. notification-worker memory limited to 2048MB and payment-db latency recovered.",

        # Task5: cache split-brain
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout users see intermittent cart/session mismatches.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Cart/session mismatches.\nNew message from L1_Triage: checkout-api is only seeing stale cache data. session-cache-primary epoch 1842, session-cache-replica epoch 1837. Repair replica.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Checkout mismatches.\nNew message from L1_Triage: session-cache-replica is stale after split-brain and must be repaired, not checkout-api restarted.",
        "INITIAL ALERT:\n[SYSTEM ALERT] Cart/session mismatches.\nNew message from L1_Triage: split-brain cache found.\nNew message from L2_DB_SME: Fix applied. session-cache-replica repaired and synced to primary.",
    ],
    "L1_Triage": [
        # Initial investigation prompts
        "New message from IC: Investigate the cluster status. Find what's causing the alert.",
        "New message from IC: Users reporting login failures. Investigate the authentication flow.",
        "New message from IC: Users reporting errors. Investigate and report back.",
        
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

        # Task4: noisy neighbor investigation
        "New message from IC: Investigate checkout/payment latency.\nObs: service_id             status     latency  memory_mb  cpu_m  cache_epoch\npayment-db             Running    780     4096      2048   -\nnotification-worker    Running    120     8000      1024   -",
        "New message from IC: payment-db is slow. Find root cause.\nObs: === Logs: payment-db ===\n[WARN] queries waiting on node memory reclaim\n[INFO] no CrashLoopBackOff or database corruption detected\n[SYSTEM] payment-db may be a victim. Inspect notification-worker.",
        "New message from IC: payment latency issue.\nObs: === Logs: notification-worker ===\n[WARN] heap usage 8000MB; background batch queue consuming node memory\n[WARN] cgroup memory pressure detected on shared node",

        # Task5: split-brain investigation
        "New message from IC: Investigate intermittent cart/session mismatches.\nObs: service_id             status     latency  memory_mb  cpu_m  cache_epoch\ncheckout-api           Running    220     2048      2048   -\nsession-cache-primary  Running    8       2048      1024   1842\nsession-cache-replica  Running    8       2048      1024   1837",
        "New message from IC: Check checkout-api first.\nObs: === Logs: checkout-api ===\n[ERROR] cart total mismatch: expected=100 observed=80\n[WARN] session lookup returned stale data from cache replica\n[SYSTEM] Do not stop at checkout-api. Inspect both cache nodes.",
        "New message from IC: Compare cache epochs.\nObs: === Logs: session-cache-primary ===\n[INFO] cache role=primary\n[INFO] cache_epoch=1842",
        "New message from IC: Compare cache epochs.\nObs: === Logs: session-cache-primary ===\n[INFO] cache_epoch=1842\nObs: === Logs: session-cache-replica ===\n[WARN] cache_epoch=1837\n[ERROR] replica epoch behind primary",
        
        # Explicit "you have logs, now report" scenarios
        "New message from IC: What did you find?\nObs: === Logs: auth-api ===\n[ERROR] TLS handshake failed: certificate has expired\n[SYSTEM] Certificate issue identified. Report to IC.",
        "New message from IC: What did you find?\nObs: === Logs: payment-db ===\n[ERROR] OOMKilled\n[ERROR] CrashLoopBackOff\n[SYSTEM] You have the logs. Report findings to IC.",
        "New message from IC: Status?\nObs: auth-api at 99.8% CPU. High latency detected.\n[SYSTEM] Investigation complete. Message IC with your findings.",
        "New message from IC: Status?\nObs: === Logs: notification-worker ===\n[WARN] heap usage 8000MB; background batch queue consuming node memory\n[SYSTEM] Investigation complete. Report that notification-worker is starving payment-db.",
        "New message from IC: Status?\nObs: === Logs: checkout-api ===\n[ERROR] cart total mismatch: expected=100 observed=80\nObs: === Logs: session-cache-primary ===\n[INFO] cache_epoch=1842\nObs: === Logs: session-cache-replica ===\n[WARN] cache_epoch=1837\n[SYSTEM] Investigation complete. Report split-brain stale replica to IC.",
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

        # Task4: notification-worker memory cap
        "New message from IC: notification-worker is starving payment-db. Set notification-worker memory limit to 2048MB.",
        "New message from IC: payment-db is the victim. Apply UPDATE_CONFIG to notification-worker with memory_limit_mb 2048.",
        "New message from IC: Cap notification-worker memory to resolve node memory pressure.\nObs: [OK] notification-worker memory limit set to 2048MB.",

        # Task5: stale cache replica repair
        "New message from IC: session-cache-replica is stale after split-brain. Repair it.",
        "New message from IC: primary epoch is 1842 and replica epoch is 1837. Run REPAIR_REPLICA on session-cache-replica.",
        "New message from IC: Repair session-cache-replica to resolve checkout cart/session mismatches.\nObs: [OK] session-cache-replica repaired and resynced with primary.",
    ],
}
