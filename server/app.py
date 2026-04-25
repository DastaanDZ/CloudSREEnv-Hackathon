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
    MESSAGE_CHANNEL = "MESSAGE_CHANNEL"
    CLOSE_INCIDENT = "CLOSE_INCIDENT"
    INVALID_FORMAT = "INVALID_FORMAT" # NEW: For RL Syntax Penalties

class Action(BaseModel):
    action_type: ActionType
    agent_id: str = "SYSTEM"
    service_id: Optional[str] = None
    cpu_value: Optional[int] = None
    target: Optional[str] = None
    message: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)

class ServiceMetrics(BaseModel):
    id: str
    status: str
    cpu_allocated: int
    memory_allocated: int
    latency_ms: int

class Observation(BaseModel):
    text_output: str
    structured_data: List[ServiceMetrics] = Field(default_factory=list)

class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str = ""
    breakdown: Dict[str, float] = Field(default_factory=dict) # NEW: For OpenEnv Rubrics

class IncidentState(BaseModel):
    """Principle-level SRE state used for task-agnostic reward shaping."""
    observations_collected: List[str] = Field(default_factory=list)
    confidence_level: float = 0.0
    root_cause: Optional[str] = None
    fixability: str = "unknown"  # unknown | fixable | external_or_rca_only
    remediation_attempted: bool = False
    remediation_successful: bool = False
    l1_reported: bool = False
    l2_delegated: bool = False
    l2_reported_fix: bool = False

# ---------------------------------------------------------------------------
# Service & MockCloud
# ---------------------------------------------------------------------------

class Service:
    def __init__(self, id: str, status: str = "Running", cpu: int = 1024, mem: int = 2048, lat: int = 45):
        self.id = id
        self.status = status
        self.cpu_allocated = cpu
        self.memory_allocated = mem
        self.latency_ms = lat
        self.logs: List[str] = []
        self.error_message: str = ""

    @property
    def metrics(self) -> ServiceMetrics:
        return ServiceMetrics(
            id=self.id, status=self.status, cpu_allocated=self.cpu_allocated,
            memory_allocated=self.memory_allocated, latency_ms=self.latency_ms
        )

class MockCloud:
    RPS_NORMAL = 500
    RPS_HIGH = 3500

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
        self.incident_state = IncidentState()
        self.scenario_profile: Dict[str, object] = {}
        self.done = False
        self.steps_taken = 0
        self.max_steps = 15 # RL needs a strict cutoff

    def reset(self, task_id: Optional[str] = None, scenario: Optional[str] = None) -> Observation:
        self.current_task = task_id or "task1_tls_certificate_rca"
        self.action_history.clear()
        self.incident_state = IncidentState()
        self.done = False
        self.steps_taken = 0

        if "task1" in self.current_task:
            self.cloud.reset(scenario="tls_certificate_expiry")
        elif "task2" in self.current_task:
            self.cloud.reset(scenario="crash_loop")
        elif "task3" in self.current_task:
            self.cloud.reset(scenario="performance_bottleneck")
        else:
            self.cloud.reset(scenario=scenario or "healthy")

        self.scenario_profile = self._build_scenario_profile()
        return Observation(text_output="\n".join(self.cloud.incident_channel))

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.steps_taken += 1
        
        if self.done or self.steps_taken > self.max_steps:
            return Observation(text_output="Episode Terminated."), Reward(value=0.0, reason="Max steps reached or already done."), True, {}

        obs_text = ""

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

        self.action_history.append(action)

        # --- NEW: OPENENV COMPOSABLE RUBRIC EVALUATION ---
        total_reward, breakdown, reason = self._calculate_rubric(action)
        state_before = self.incident_state.model_copy(deep=True)

        # If it was an invalid format, immediately return the penalty (Don't execute)
        if action.action_type == ActionType.INVALID_FORMAT:
            return Observation(text_output="[SYSTEM] INVALID JSON FORMAT."), Reward(value=total_reward, reason=reason, breakdown=breakdown), False, {}

        # If it was an RBAC violation, immediately return the penalty (Don't execute)
        if "rbac_penalty" in breakdown:
            obs_text = f"[403 Forbidden] Agent {action.agent_id} lacks permissions to modify cluster resources."
            return Observation(text_output=obs_text), Reward(value=total_reward, reason=reason, breakdown=breakdown), False, {}

        # --- EXECUTE ACTIONS ---
        if action.action_type == ActionType.LIST_SERVICES:
            lines = [f"{svc.id:<20} {svc.status:<10} {svc.latency_ms}ms" for svc in self.cloud.services.values()]
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

        elif action.action_type == ActionType.MESSAGE_CHANNEL:
            msg = f"[{action.agent_id} -> {action.target}]: {action.message}"
            self.cloud.incident_channel.append(msg)
            obs_text = "[OK] Message posted to Incident Channel."

        elif action.action_type == ActionType.CLOSE_INCIDENT:
            principle_reward, principle_breakdown = self._calculate_principle_reward(action, state_before, "")
            total_reward += principle_reward
            breakdown.update(principle_breakdown)
            task_done, final_score = self._grade_terminal_state()
            if task_done:
                self.done = True
                total_reward += final_score
                breakdown["task_completion"] = final_score
                capped_reward = max(-1.0, min(1.0, total_reward))
                return Observation(text_output="[INCIDENT CLOSED] Task completed successfully."), Reward(value=capped_reward, reason="Task Success", breakdown=breakdown), True, {}
            else:
                total_reward -= 0.5
                breakdown["false_closure_penalty"] = -0.5
                capped_reward = max(-1.0, min(1.0, total_reward))
                return Observation(text_output="[ERROR] Cannot close incident. Criteria not met."), Reward(value=capped_reward, reason="Premature Closure", breakdown=breakdown), False, {}

        principle_reward, principle_breakdown = self._calculate_principle_reward(action, state_before, obs_text)
        total_reward += principle_reward
        breakdown.update(principle_breakdown)

        # Cap the reward between -1.0 and 1.0 to comply with OpenEnv Spec
        capped_reward = max(-1.0, min(1.0, total_reward))
        
        return Observation(text_output=obs_text), Reward(value=capped_reward, reason=reason, breakdown=breakdown), self.done, {}

    def _action_match_key(self, action: Action) -> tuple:
        """Per-type identity used for hard-duplicate detection.

        - LIST_SERVICES: (type, agent)
        - GET_LOGS / RESTART: (type, agent, service)
        - SCALE: (type, agent, service, cpu_value)  -- different cpu_value is allowed
        - MESSAGE_CHANNEL: (type, agent, target)    -- message text intentionally ignored
        """
        if action.action_type == ActionType.SCALE:
            return (action.action_type, action.agent_id, action.service_id, action.cpu_value)
        if action.action_type == ActionType.MESSAGE_CHANNEL:
            return (action.action_type, action.agent_id, action.target)
        if action.action_type in (ActionType.GET_LOGS, ActionType.RESTART):
            return (action.action_type, action.agent_id, action.service_id)
        return (action.action_type, action.agent_id)

    def _is_duplicate(self, action: Action) -> bool:
        """True if any prior action this episode matches the current action's key."""
        key = self._action_match_key(action)
        for prev in self.action_history:
            if prev.action_type in (ActionType.INVALID_FORMAT, ActionType.CLOSE_INCIDENT):
                continue
            if self._action_match_key(prev) == key:
                return True
        return False

    def _build_scenario_profile(self) -> Dict[str, object]:
        """Scenario facts consumed by task-agnostic principle rewards."""
        if "task1" in self.current_task:
            return {
                "evidence_service": "auth-api",
                "root_cause": "expired upstream TLS certificate",
                "fixability": "external_or_rca_only",
                "remediation_action": None,
                "remediation_service": None,
            }
        if "task2" in self.current_task:
            return {
                "evidence_service": "payment-db",
                "root_cause": "payment-db CrashLoopBackOff/OOMKilled",
                "fixability": "fixable",
                "remediation_action": ActionType.RESTART,
                "remediation_service": "payment-db",
            }
        if "task3" in self.current_task:
            return {
                "evidence_service": "auth-api",
                "root_cause": "auth-api CPU saturation",
                "fixability": "fixable",
                "remediation_action": ActionType.SCALE,
                "remediation_service": "auth-api",
                "min_cpu": 2048,
            }
        return {
            "evidence_service": None,
            "root_cause": None,
            "fixability": "unknown",
            "remediation_action": None,
            "remediation_service": None,
        }

    def _remember_observation(self, key: str) -> bool:
        """Record a newly collected observation. Returns True if it is new."""
        if key in self.incident_state.observations_collected:
            return False
        self.incident_state.observations_collected.append(key)
        return True

    def _action_matches_remediation(self, action: Action) -> bool:
        expected_action = self.scenario_profile.get("remediation_action")
        expected_service = self.scenario_profile.get("remediation_service")
        if not expected_action or action.action_type != expected_action or action.service_id != expected_service:
            return False
        if action.action_type == ActionType.SCALE:
            min_cpu = int(self.scenario_profile.get("min_cpu", 0) or 0)
            return bool(action.cpu_value is not None and action.cpu_value >= min_cpu)
        return True

    def _calculate_principle_reward(
        self,
        action: Action,
        state_before: IncidentState,
        obs_text: str,
    ) -> tuple[float, dict]:
        """Reward SRE principles instead of only task-specific action names."""
        reward = 0.0
        breakdown: Dict[str, float] = {}
        evidence_service = self.scenario_profile.get("evidence_service")
        fixability = str(self.scenario_profile.get("fixability", "unknown"))

        if action.action_type == ActionType.LIST_SERVICES:
            if self._remember_observation("service_list"):
                breakdown["principle_new_observation"] = 0.12
                reward += 0.12

        elif action.action_type == ActionType.GET_LOGS:
            obs_key = f"logs:{action.service_id}"
            if action.service_id and self._remember_observation(obs_key):
                breakdown["principle_new_observation"] = 0.18
                reward += 0.18
            if action.service_id == evidence_service:
                old_confidence = self.incident_state.confidence_level
                self.incident_state.confidence_level = max(self.incident_state.confidence_level, 0.75)
                self.incident_state.root_cause = str(self.scenario_profile.get("root_cause") or "")
                self.incident_state.fixability = fixability
                if self.incident_state.confidence_level > old_confidence:
                    breakdown["principle_confidence_gain"] = 0.22
                    reward += 0.22
                    breakdown["principle_fixability_classified"] = 0.15
                    reward += 0.15
            elif action.service_id:
                breakdown["principle_irrelevant_evidence"] = -0.08
                reward -= 0.08

        elif action.action_type == ActionType.MESSAGE_CHANNEL:
            target = action.target
            if action.agent_id == "IC" and target == "L1_Triage" and state_before.confidence_level < 0.5:
                breakdown["principle_delegate_investigation"] = 0.18
                reward += 0.18
            elif action.agent_id == "L1_Triage" and target == "IC":
                self.incident_state.l1_reported = True
                if state_before.confidence_level >= 0.5:
                    breakdown["principle_report_after_evidence"] = 0.25
                    reward += 0.25
                else:
                    breakdown["principle_report_without_evidence"] = -0.30
                    reward -= 0.30
            elif action.agent_id == "IC" and target == "L2_DB_SME":
                self.incident_state.l2_delegated = True
                if state_before.fixability == "fixable" and state_before.confidence_level >= 0.5:
                    breakdown["principle_delegate_fixable_issue"] = 0.25
                    reward += 0.25
                elif state_before.fixability == "external_or_rca_only":
                    breakdown["principle_unneeded_l2_escalation"] = -0.30
                    reward -= 0.30
                else:
                    breakdown["principle_delegate_without_diagnosis"] = -0.20
                    reward -= 0.20
            elif action.agent_id == "L2_DB_SME" and target == "IC":
                self.incident_state.l2_reported_fix = True
                if state_before.remediation_successful:
                    breakdown["principle_report_fix_complete"] = 0.20
                    reward += 0.20
                else:
                    breakdown["principle_l2_report_without_fix"] = -0.12
                    reward -= 0.12

        elif action.action_type in (ActionType.RESTART, ActionType.SCALE):
            self.incident_state.remediation_attempted = True
            if state_before.fixability == "unknown" or state_before.confidence_level < 0.5:
                breakdown["principle_remediate_before_diagnosis"] = -0.30
                reward -= 0.30
            elif state_before.fixability == "external_or_rca_only":
                breakdown["principle_remediate_nonfixable_issue"] = -0.40
                reward -= 0.40
            elif self._action_matches_remediation(action):
                self.incident_state.remediation_successful = True
                breakdown["principle_correct_remediation"] = 0.35
                reward += 0.35
            else:
                breakdown["principle_wrong_remediation"] = -0.25
                reward -= 0.25

        elif action.action_type == ActionType.CLOSE_INCIDENT:
            ready_for_close = (
                state_before.fixability == "external_or_rca_only" and state_before.confidence_level >= 0.5
            ) or state_before.remediation_successful
            if ready_for_close:
                breakdown["principle_close_when_ready"] = 0.25
                reward += 0.25
            else:
                breakdown["principle_premature_close"] = -0.30
                reward -= 0.30

        return reward, breakdown

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
        if action.action_type in [ActionType.RESTART, ActionType.SCALE]:
            if action.agent_id != "L2_DB_SME":
                return -0.5, {"rbac_penalty": -0.5}, "Unauthorized cluster modification."
            else:
                reward += 0.2
                breakdown["authorized_action"] = 0.2
                reason = "Executed authorized modification."

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
            no_remediation = not any(a.action_type in [ActionType.RESTART, ActionType.SCALE] for a in self.action_history)
            return (True, 0.8) if (found_logs and no_remediation) else (False, 0.0)
            
        elif "task2" in self.current_task:
            return (True, 0.9) if self.cloud.services["payment-db"].status == "Running" else (False, 0.0)
            
        elif "task3" in self.current_task:
            svc = self.cloud.services["auth-api"]
            return (True, 1.0) if svc.cpu_allocated >= 2048 and svc.latency_ms < 100 else (False, 0.0)
            
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