"""
CloudSREEnv v4.0 — RL Training Edition (Dense Rewards & Composable Rubrics)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

# --- Setup Logging ---
logger = logging.getLogger("CloudEnv")

# ---------------------------------------------------------------------------
# Typed Models (OpenEnv Spec)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    LIST_SERVICES = "LIST_SERVICES"
    GET_LOGS = "GET_LOGS"
    RESTART = "RESTART"
    SCALE = "SCALE"
    UPDATE_CONFIG = "UPDATE_CONFIG"
    REPAIR_REPLICA = "REPAIR_REPLICA"
    MESSAGE_CHANNEL = "MESSAGE_CHANNEL"
    CLOSE_INCIDENT = "CLOSE_INCIDENT"
    INVALID_FORMAT = "INVALID_FORMAT" # NEW: For RL Syntax Penalties

class Action(BaseModel):
    action_type: ActionType
    agent_id: str = "SYSTEM"
    service_id: Optional[str] = None
    cpu_value: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    target: Optional[str] = None
    message: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)

class ServiceMetrics(BaseModel):
    id: str
    status: str
    cpu_allocated: int
    memory_allocated: int
    latency_ms: int
    memory_limit_mb: Optional[int] = None
    cache_epoch: Optional[int] = None

class Observation(BaseModel):
    text_output: str
    structured_data: List[ServiceMetrics] = Field(default_factory=list)

class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str = ""
    breakdown: Dict[str, float] = Field(default_factory=dict) # NEW: For OpenEnv Rubrics

# ---------------------------------------------------------------------------
# Service & MockCloud
# ---------------------------------------------------------------------------

class Service:
    def __init__(self, id: str, status: str = "Running", cpu: int = 1024, mem: int = 2048, lat: int = 45):
        self.id = id
        self.status = status
        self.cpu_allocated = cpu
        self.memory_allocated = mem
        self.memory_limit_mb: Optional[int] = None
        self.latency_ms = lat
        self.logs: List[str] = []
        self.error_message: str = ""
        self.cache_epoch: Optional[int] = None

    @property
    def metrics(self) -> ServiceMetrics:
        return ServiceMetrics(
            id=self.id, status=self.status, cpu_allocated=self.cpu_allocated,
            memory_allocated=self.memory_allocated, latency_ms=self.latency_ms,
            memory_limit_mb=self.memory_limit_mb, cache_epoch=self.cache_epoch
        )

class MockCloud:
    RPS_NORMAL = 500
    RPS_HIGH = 3500
    WORKER_MEMORY_PRESSURE_MB = 7000
    STRICT_WORKER_MEMORY_LIMIT_MB = 2048

    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.rps = self.RPS_NORMAL
        self.incident_channel: List[str] = []

    def reset(self, scenario: str = "healthy"):
        self.services = {
            "auth-api": Service("auth-api"),
            "payment-db": Service("payment-db", cpu=2048, mem=4096, lat=12),
            "inventory-svc": Service("inventory-svc"),
            "notification-worker": Service("notification-worker", lat=60),
        }
        self.rps = self.RPS_NORMAL
        self.incident_channel = []

        if scenario == "crash_loop":
            self._apply_crash_loop()
        elif scenario == "performance_bottleneck":
            self._apply_performance_bottleneck()
        elif scenario == "tls_certificate_expiry":
            self._apply_tls_certificate_expiry()
        elif scenario == "resource_contention":
            self._apply_resource_contention()
        elif scenario == "split_brain_cache":
            self._apply_split_brain_cache()

        self._propagate_failures()

    def _apply_crash_loop(self):
        svc = self.services["payment-db"]
        svc.status = "Error"
        svc.latency_ms = 0
        svc.logs = ["[ERROR] OOMKilled", "[ERROR] CrashLoopBackOff"]
        self.incident_channel.append("[SYSTEM ALERT] payment-db status transition to Error detected.")

    def _apply_performance_bottleneck(self):
        self.rps = self.RPS_HIGH
        svc = self.services["auth-api"]
        svc.cpu_allocated = 128
        svc.latency_ms = 850
        svc.logs = [f"[WARN] RPS={self.rps} — CPU usage 99.8%"]
        self.incident_channel.append("[SYSTEM ALERT] High latency (850ms) detected on auth-api.")

    def _apply_tls_certificate_expiry(self):
        svc = self.services["auth-api"]
        svc.status = "Running"
        svc.latency_ms = 350
        svc.error_message = "Warning: authentication failures detected; inspect auth-api logs."
        svc.logs = [
            "[ERROR] TLS handshake failed: certificate has expired",
            "[ERROR] x509: certificate signed by unknown authority",
            "[WARN] Upstream certificate expired 2 days ago"
        ]
        self.incident_channel.append("[SYSTEM ALERT] Login failures reported across customer-facing authentication flow.")

    def _apply_resource_contention(self):
        worker = self.services["notification-worker"]
        worker.memory_allocated = 8000
        worker.latency_ms = 180
        worker.logs = [
            "[WARN] Heap growth detected: notification batch cache at 8000MB",
            "[WARN] Node memory pressure: worker has no memory limit"
        ]

        db = self.services["payment-db"]
        db.latency_ms = 650
        db.logs = [
            "[WARN] Checkout queries throttled by node memory pressure",
            "[INFO] Database CPU is normal; memory reclaim stalls detected"
        ]
        self.incident_channel.append("[SYSTEM ALERT] Checkout latency detected on payment-db.")

    def _apply_split_brain_cache(self):
        self.services["checkout-api"] = Service("checkout-api", lat=420)
        self.services["session-cache-primary"] = Service("session-cache-primary", mem=1024, lat=20)
        self.services["session-cache-replica"] = Service("session-cache-replica", mem=1024, lat=25)

        checkout = self.services["checkout-api"]
        checkout.logs = [
            "[ERROR] cart_total_mismatch user=user_123 expected=100 observed=80",
            "[WARN] intermittent session lookup mismatch from cache pool"
        ]
        checkout.error_message = "Warning: inconsistent cart/session data returned by cache layer."

        auth = self.services["auth-api"]
        auth.latency_ms = 120
        auth.logs = [
            "[WARN] session token validation intermittently failed",
            "[INFO] auth-api CPU and upstream DB checks normal"
        ]

        primary = self.services["session-cache-primary"]
        primary.cache_epoch = 1842
        primary.logs = [
            "[INFO] role=primary cache_epoch=1842 writes_enabled=true",
            "[INFO] cart:user_123 total=100 version=91"
        ]

        replica = self.services["session-cache-replica"]
        replica.cache_epoch = 1837
        replica.logs = [
            "[WARN] role=replica cache_epoch=1837 serving_traffic=true",
            "[WARN] replication_lag=5 epochs; cart:user_123 total=80 version=86",
            "[ERROR] split-brain suspected after network partition"
        ]

        self.incident_channel.append("[SYSTEM ALERT] Intermittent checkout cart mismatches and session failures detected.")

    def repair_cache_replica(self, service_id: str) -> bool:
        if service_id != "session-cache-replica":
            return False

        primary = self.services.get("session-cache-primary")
        replica = self.services.get("session-cache-replica")
        checkout = self.services.get("checkout-api")
        auth = self.services.get("auth-api")
        if not primary or not replica:
            return False

        replica.cache_epoch = primary.cache_epoch
        replica.latency_ms = 20
        replica.logs = [
            f"[INFO] Replica resynced to cache_epoch={primary.cache_epoch}",
            "[INFO] Split-brain repaired; serving consistent cache data"
        ]
        if checkout:
            checkout.latency_ms = 45
            checkout.error_message = ""
            checkout.logs = ["[INFO] Cart totals consistent after cache replica repair."]
        if auth:
            auth.latency_ms = 45
            auth.logs = ["[INFO] Session validation stable after cache repair."]
        return True

    def _has_worker_memory_pressure(self) -> bool:
        worker = self.services.get("notification-worker")
        if not worker:
            return False
        has_strict_limit = (
            worker.memory_limit_mb is not None
            and worker.memory_limit_mb <= self.STRICT_WORKER_MEMORY_LIMIT_MB
        )
        return worker.memory_allocated >= self.WORKER_MEMORY_PRESSURE_MB and not has_strict_limit

    def apply_worker_memory_limit(self, limit_mb: int) -> None:
        worker = self.services["notification-worker"]
        worker.memory_limit_mb = limit_mb
        if limit_mb <= self.STRICT_WORKER_MEMORY_LIMIT_MB:
            worker.memory_allocated = min(worker.memory_allocated, limit_mb)
            worker.latency_ms = 60
            worker.logs = [f"[INFO] Memory limit set to {limit_mb}MB; cache trimmed."]
        else:
            worker.logs = [f"[WARN] Memory limit set to {limit_mb}MB; still too high for node pressure."]
        self._propagate_failures()

    def _propagate_failures(self):
        deps = {
            "auth-api": "payment-db",
            "inventory-svc": "payment-db",
            "notification-worker": "inventory-svc"
        }
        # Preserve scenario-set error_messages, only clear cascade-generated ones
        preserved = {svc.id: svc.error_message for svc in self.services.values() if svc.error_message}
        for svc in self.services.values():
            svc.error_message = ""
            if svc.status == "Running" and svc.id != "auth-api":
                svc.latency_ms = 45 if svc.id != "payment-db" else 12

        if self._has_worker_memory_pressure():
            db = self.services["payment-db"]
            db.latency_ms = 650
            db.error_message = "Warning: throttled by node memory pressure from notification-worker."

        for dependent, provider_id in deps.items():
            provider = self.services.get(provider_id)
            dep_svc = self.services.get(dependent)
            if not provider or not dep_svc: continue

            if provider.status == "Error":
                dep_svc.error_message = f"Critical: 503 Service Unavailable - Upstream {provider_id} is down."
            elif provider.latency_ms > 200:
                dep_svc.latency_ms += int(provider.latency_ms * 0.5)
                dep_svc.error_message = f"Warning: Upstream {provider_id} is slow."

        # Restore scenario-set error_messages that weren't overwritten by cascade
        for svc_id, msg in preserved.items():
            svc = self.services.get(svc_id)
            if svc and not svc.error_message:
                svc.error_message = msg

class CloudSREEnv:
    def __init__(self):
        self.cloud = MockCloud()
        self.current_task = ""
        self.action_history: List[Action] = []
        self.done = False
        self.steps_taken = 0
        self.max_steps = 15 # RL needs a strict cutoff

    def reset(self, task_id: Optional[str] = None, scenario: Optional[str] = None) -> Observation:
        self.current_task = task_id or "task1_tls_certificate_rca"
        self.action_history.clear()
        self.done = False
        self.steps_taken = 0

        if "task1" in self.current_task:
            self.cloud.reset(scenario="tls_certificate_expiry")
        elif "task2" in self.current_task:
            self.cloud.reset(scenario="crash_loop")
        elif "task3" in self.current_task:
            self.cloud.reset(scenario="performance_bottleneck")
        elif "task4" in self.current_task:
            self.cloud.reset(scenario="resource_contention")
        elif "task5" in self.current_task:
            self.cloud.reset(scenario="split_brain_cache")
        else:
            self.cloud.reset(scenario=scenario or "healthy")

        return Observation(text_output="\n".join(self.cloud.incident_channel))

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.steps_taken += 1
        
        if self.done or self.steps_taken > self.max_steps:
            return Observation(text_output="Episode Terminated."), Reward(value=0.0, reason="Max steps reached or already done."), True, {}

        self.action_history.append(action)
        obs_text = ""
        structured_data: List[ServiceMetrics] = []

        # --- HARD DUPLICATE BLOCK ---
        # Reject any repeat of an action already taken this episode (per-type key).
        # Side effects, rubric, and RBAC are all skipped; the agent gets a clear
        # signal to pick a different action. INVALID_FORMAT and CLOSE_INCIDENT
        # have their own dedicated logic and are intentionally exempt.
        if (action.action_type not in (ActionType.INVALID_FORMAT, ActionType.CLOSE_INCIDENT)
                and self._is_duplicate(action)):
            block_reward = -0.4
            block_breakdown = {"duplicate_block": block_reward}
            obs = "[BLOCKED] Duplicate action. You already did this. Choose a different action or report findings."
            return (
                Observation(text_output=obs),
                Reward(value=block_reward, reason="Duplicate action blocked.", breakdown=block_breakdown),
                False,
                {},
            )

        # --- NEW: OPENENV COMPOSABLE RUBRIC EVALUATION ---
        total_reward, breakdown, reason = self._calculate_rubric(action)

        # If it was an invalid format, immediately return the penalty (Don't execute)
        if action.action_type == ActionType.INVALID_FORMAT:
            return Observation(text_output="[SYSTEM] INVALID JSON FORMAT."), Reward(value=total_reward, reason=reason, breakdown=breakdown), False, {}

        # If it was an RBAC violation, immediately return the penalty (Don't execute)
        if "rbac_penalty" in breakdown:
            obs_text = f"[403 Forbidden] Agent {action.agent_id} lacks permissions to modify cluster resources."
            return Observation(text_output=obs_text), Reward(value=total_reward, reason=reason, breakdown=breakdown), False, {}

        # --- EXECUTE ACTIONS ---
        if action.action_type == ActionType.LIST_SERVICES:
            structured_data = [svc.metrics for svc in self.cloud.services.values()]
            lines = [
                f"{svc.id:<20} {svc.status:<10} CPU={svc.cpu_allocated}m "
                f"MEM={svc.memory_allocated}MB LAT={svc.latency_ms}ms"
                + (f" EPOCH={svc.cache_epoch}" if svc.cache_epoch is not None else "")
                for svc in self.cloud.services.values()
            ]
            obs_text = "\n".join(lines)

        elif action.action_type == ActionType.GET_LOGS:
            svc = self.cloud.services.get(action.service_id)
            if not svc:
                obs_text = f"[ERROR] Service {action.service_id} not found."
                total_reward -= 0.1 # Small penalty for hallucinating a service name
            else:
                logs = list(svc.logs)
                if svc.error_message: logs.append(svc.error_message)
                obs_text = f"=== Logs: {action.service_id} ===\n" + "\n".join(logs) if logs else "No logs."

        elif action.action_type == ActionType.RESTART:
            svc = self.cloud.services.get(action.service_id)
            if svc and svc.status == "Running":
                obs_text = f"[WARN] {action.service_id} is already Running."
                total_reward -= 0.2 # Penalty for restarting a healthy service
            elif svc:
                svc.status = "Running"
                self.cloud._propagate_failures()
                obs_text = f"[OK] {action.service_id} restarted by {action.agent_id}."

        elif action.action_type == ActionType.SCALE:
            if action.cpu_value is None:
                obs_text = "[ERROR] SCALE requires cpu_value parameter."
                total_reward -= 0.2
                breakdown["missing_param_penalty"] = -0.2
            else:
                svc = self.cloud.services.get(action.service_id)
                if svc:
                    svc.cpu_allocated = action.cpu_value
                    if action.cpu_value >= 2048:
                        svc.latency_ms = 45 if svc.id != "payment-db" else 12
                        svc.logs = ["[INFO] Scaled successfully."]
                        self.cloud._propagate_failures()
                    obs_text = f"[OK] {action.service_id} scaled to {action.cpu_value}m CPU."

                    if "task4" in self.current_task and action.service_id == "payment-db":
                        total_reward -= 0.4
                        breakdown["wrong_root_cause_penalty"] = -0.4
                        obs_text += " [WARN] Latency persists; DB was not the root cause."

                    if "task5" in self.current_task and action.service_id == "checkout-api":
                        total_reward -= 0.4
                        breakdown["wrong_root_cause_penalty"] = -0.4
                        obs_text += " [WARN] Mismatches persist; checkout-api was only the symptom."

        elif action.action_type == ActionType.UPDATE_CONFIG:
            if action.memory_limit_mb is None:
                obs_text = "[ERROR] UPDATE_CONFIG requires memory_limit_mb parameter."
                total_reward -= 0.2
                breakdown["missing_param_penalty"] = -0.2
            else:
                svc = self.cloud.services.get(action.service_id)
                if not svc:
                    obs_text = f"[ERROR] Service {action.service_id} not found."
                    total_reward -= 0.1
                else:
                    svc.memory_limit_mb = action.memory_limit_mb
                    if action.service_id == "notification-worker":
                        self.cloud.apply_worker_memory_limit(action.memory_limit_mb)
                    else:
                        self.cloud._propagate_failures()
                    obs_text = f"[OK] {action.service_id} memory limit set to {action.memory_limit_mb}MB."

        elif action.action_type == ActionType.REPAIR_REPLICA:
            repaired = self.cloud.repair_cache_replica(action.service_id or "")
            if repaired:
                obs_text = f"[OK] {action.service_id} resynced to primary cache epoch."
            else:
                obs_text = f"[ERROR] {action.service_id} is not a repairable cache replica."
                total_reward -= 0.2
                breakdown["wrong_repair_target_penalty"] = -0.2

        elif action.action_type == ActionType.MESSAGE_CHANNEL:
            msg = f"[{action.agent_id} -> {action.target}]: {action.message}"
            self.cloud.incident_channel.append(msg)
            obs_text = "[OK] Message posted to Incident Channel."

        elif action.action_type == ActionType.CLOSE_INCIDENT:
            task_done, final_score = self._grade_terminal_state()
            if task_done:
                self.done = True
                total_reward += final_score
                breakdown["task_completion"] = final_score
                return Observation(text_output="[INCIDENT CLOSED] Task completed successfully."), Reward(value=total_reward, reason="Task Success", breakdown=breakdown), True, {}
            else:
                total_reward -= 0.5
                breakdown["false_closure_penalty"] = -0.5
                return Observation(text_output="[ERROR] Cannot close incident. Criteria not met."), Reward(value=total_reward, reason="Premature Closure", breakdown=breakdown), False, {}

        # Cap the reward between -1.0 and 1.0 to comply with OpenEnv Spec
        capped_reward = max(-1.0, min(1.0, total_reward))
        
        return Observation(text_output=obs_text, structured_data=structured_data), Reward(value=capped_reward, reason=reason, breakdown=breakdown), self.done, {}

    def _action_match_key(self, action: Action) -> tuple:
        """Per-type identity used for hard-duplicate detection.

        - LIST_SERVICES: (type, agent)
        - GET_LOGS / RESTART: (type, agent, service)
        - SCALE: (type, agent, service, cpu_value)  -- different cpu_value is allowed
        - UPDATE_CONFIG: (type, agent, service, memory_limit_mb)
        - REPAIR_REPLICA: (type, agent, service)
        - MESSAGE_CHANNEL: (type, agent, target)    -- message text intentionally ignored
        """
        if action.action_type == ActionType.SCALE:
            return (action.action_type, action.agent_id, action.service_id, action.cpu_value)
        if action.action_type == ActionType.UPDATE_CONFIG:
            return (action.action_type, action.agent_id, action.service_id, action.memory_limit_mb)
        if action.action_type == ActionType.MESSAGE_CHANNEL:
            return (action.action_type, action.agent_id, action.target)
        if action.action_type in (ActionType.GET_LOGS, ActionType.RESTART, ActionType.REPAIR_REPLICA):
            return (action.action_type, action.agent_id, action.service_id)
        return (action.action_type, action.agent_id)

    def _is_duplicate(self, action: Action) -> bool:
        """True if any prior action this episode matches the current action's key."""
        key = self._action_match_key(action)
        for prev in self.action_history[:-1]:
            if prev.action_type in (ActionType.INVALID_FORMAT, ActionType.CLOSE_INCIDENT):
                continue
            if self._action_match_key(prev) == key:
                return True
        return False

    # --- DENSE REWARD RUBRIC ---
    def _calculate_rubric(self, action: Action) -> tuple[float, dict, str]:
        """Calculates dense rewards for RL training."""
        reward = 0.0
        breakdown = {}
        reason = "Step execution."

        # 1. Syntax Rubric (Punish hallucinations heavily)
        if action.action_type == ActionType.INVALID_FORMAT:
            return -0.5, {"syntax_penalty": -0.5}, "Invalid JSON format."

        # 2. Tool-Use Rubric (Reward discovery and communication)
        # Note: duplicate detection is now a hard block in step(), not a soft penalty here.
        if action.action_type in [ActionType.LIST_SERVICES, ActionType.GET_LOGS]:
            reward += 0.05
            breakdown["tool_discovery"] = 0.05
            reason = "Gathered information."
            
        elif action.action_type == ActionType.MESSAGE_CHANNEL:
            reward += 0.1
            breakdown["collaboration"] = 0.1
            reason = "Communicated in channel."

        # 3. RBAC Rubric (Strictly enforce roles)
        if action.action_type in [ActionType.RESTART, ActionType.SCALE, ActionType.UPDATE_CONFIG, ActionType.REPAIR_REPLICA]:
            if action.agent_id != "L2_DB_SME":
                return -0.5, {"rbac_penalty": -0.5}, "Unauthorized cluster modification."
            else:
                reward += 0.2
                breakdown["authorized_action"] = 0.2
                reason = "Executed authorized modification."

        if "task4" in self.current_task:
            if action.action_type == ActionType.LIST_SERVICES:
                reward += 0.15
                breakdown["memory_discovery"] = 0.15
                reason = "Inspected resource usage."
            elif (
                action.action_type == ActionType.UPDATE_CONFIG
                and action.service_id == "notification-worker"
                and action.memory_limit_mb is not None
                and action.memory_limit_mb <= self.cloud.STRICT_WORKER_MEMORY_LIMIT_MB
            ):
                reward += 0.45
                breakdown["correct_config_fix"] = 0.45
                reason = "Constrained noisy neighbor memory."
            elif action.action_type == ActionType.SCALE and action.service_id == "payment-db":
                reward -= 0.35
                breakdown["db_scaling_trap"] = -0.35
                reason = "Scaled symptom service instead of noisy neighbor."

        if "task5" in self.current_task:
            checked_primary = any(a.action_type == ActionType.GET_LOGS and a.service_id == "session-cache-primary" for a in self.action_history)
            checked_replica = any(a.action_type == ActionType.GET_LOGS and a.service_id == "session-cache-replica" for a in self.action_history)
            if action.action_type == ActionType.GET_LOGS and action.service_id == "checkout-api":
                reward += 0.1
                breakdown["symptom_trace"] = 0.1
                reason = "Inspected checkout symptom logs."
            elif action.action_type == ActionType.GET_LOGS and action.service_id in ("session-cache-primary", "session-cache-replica"):
                reward += 0.15
                breakdown["cache_epoch_discovery"] = 0.15
                if checked_primary and checked_replica:
                    reward += 0.2
                    breakdown["split_brain_evidence"] = 0.2
                reason = "Inspected cache consistency evidence."
            elif action.action_type == ActionType.REPAIR_REPLICA and action.service_id == "session-cache-replica":
                reward += 0.5
                breakdown["correct_replica_repair"] = 0.5
                reason = "Repaired split-brain cache replica."
            elif action.action_type in (ActionType.RESTART, ActionType.SCALE) and action.service_id == "checkout-api":
                reward -= 0.4
                breakdown["checkout_symptom_trap"] = -0.4
                reason = "Remediated symptom service instead of cache split-brain."

        # 4. Time Penalty (Encourages efficiency)
        time_penalty = -0.02 * self.steps_taken
        reward += time_penalty
        breakdown["time_penalty"] = time_penalty

        return reward, breakdown, reason

    # --- TERMINAL GRADER (Task Success) ---
    def _grade_terminal_state(self) -> tuple[bool, float]:
        """Evaluates if the final state matches the scenario objective."""
        if "task1" in self.current_task:
            # TLS Certificate RCA: GET_LOGS(auth-api) happened AND no RESTART/SCALE
            found_logs = any(a.action_type == ActionType.GET_LOGS and a.service_id == "auth-api" for a in self.action_history)
            no_remediation = not any(
                a.action_type in [ActionType.RESTART, ActionType.SCALE, ActionType.UPDATE_CONFIG, ActionType.REPAIR_REPLICA]
                for a in self.action_history
            )
            return (True, 0.8) if (found_logs and no_remediation) else (False, 0.0)
            
        elif "task2" in self.current_task:
            return (True, 0.9) if self.cloud.services["payment-db"].status == "Running" else (False, 0.0)
            
        elif "task3" in self.current_task:
            svc = self.cloud.services["auth-api"]
            return (True, 1.0) if svc.cpu_allocated >= 2048 and svc.latency_ms < 100 else (False, 0.0)

        elif "task4" in self.current_task:
            worker = self.cloud.services["notification-worker"]
            db = self.cloud.services["payment-db"]
            config_applied = any(
                a.action_type == ActionType.UPDATE_CONFIG
                and a.service_id == "notification-worker"
                and a.memory_limit_mb is not None
                and a.memory_limit_mb <= self.cloud.STRICT_WORKER_MEMORY_LIMIT_MB
                for a in self.action_history
            )
            return (True, 1.0) if config_applied and db.latency_ms < 100 and worker.memory_allocated <= 2048 else (False, 0.0)

        elif "task5" in self.current_task:
            primary = self.cloud.services.get("session-cache-primary")
            replica = self.cloud.services.get("session-cache-replica")
            checkout = self.cloud.services.get("checkout-api")
            checked_both_caches = all(
                any(a.action_type == ActionType.GET_LOGS and a.service_id == svc for a in self.action_history)
                for svc in ("session-cache-primary", "session-cache-replica")
            )
            repaired_replica = any(
                a.action_type == ActionType.REPAIR_REPLICA and a.service_id == "session-cache-replica"
                for a in self.action_history
            )
            epochs_match = bool(primary and replica and primary.cache_epoch == replica.cache_epoch)
            checkout_recovered = bool(checkout and checkout.latency_ms < 100)
            return (True, 1.0) if checked_both_caches and repaired_replica and epochs_match and checkout_recovered else (False, 0.0)
            
        return False, 0.0

# ---------------------------------------------------------------------------
# FastAPI Server Setup
# ---------------------------------------------------------------------------
def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI(title="CloudSREEnv", version="4.0.0-RL")
        env = CloudSREEnv()

        @app.post("/reset")
        def api_reset(task_id: Optional[str] = None, scenario: Optional[str] = None):
            obs = env.reset(task_id=task_id, scenario=scenario)
            return JSONResponse(obs.model_dump())

        @app.post("/step")
        def api_step(action: Action):
            obs, reward, done, info = env.step(action)
            return JSONResponse({
                "observation": obs.model_dump(), 
                "reward": reward.model_dump(), 
                "done": done, 
                "info": info
            })
        
        return app, uvicorn
    except ImportError as e:
        logger.error(f"Failed to start API: {e}")
        return None, None

def main():
    app_instance, uvicorn_module = create_app()
    if app_instance and uvicorn_module:
        logger.info("Starting CloudSREEnv RL Training API Server on http://0.0.0.0:8000")
        uvicorn_module.run(app_instance, host="0.0.0.0", port=8000)
    else:
        print("Could not start the server.")

if __name__ == "__main__":
    main()