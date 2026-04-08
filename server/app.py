"""
CloudSREEnv v2.2 — Enhanced with Cascading Failures and (0, 1) Non-Binary Graders.
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Typed Models (OpenEnv Spec)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    LIST_SERVICES = "LIST_SERVICES"
    GET_LOGS = "GET_LOGS"
    RESTART = "RESTART"
    SCALE = "SCALE"


class Action(BaseModel):
    """Typed action model consumed by env.step()."""
    action_type: ActionType
    service_id: Optional[str] = None
    cpu_value: Optional[int] = None

    class Config:
        use_enum_values = True


class ServiceMetrics(BaseModel):
    id: str
    status: str
    cpu_allocated: int
    memory_allocated: int
    latency_ms: int


class Observation(BaseModel):
    """Returned after every env.step()."""
    text_output: str
    structured_data: List[ServiceMetrics] = Field(default_factory=list)


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str = ""


# ---------------------------------------------------------------------------
# Service & MockCloud
# ---------------------------------------------------------------------------

class Service:
    def __init__(
        self,
        id: str,
        status: str = "Running",
        cpu_allocated: int = 512,
        memory_allocated: int = 1024,
        latency_ms: int = 30,
        logs: Optional[List[str]] = None,
        error_message: str = "",
    ):
        self.id = id
        self.status = status
        self.cpu_allocated = cpu_allocated
        self.memory_allocated = memory_allocated
        self.latency_ms = latency_ms
        self.logs: List[str] = logs or []
        self.error_message = error_message

    def to_metrics(self) -> ServiceMetrics:
        return ServiceMetrics(
            id=self.id,
            status=self.status,
            cpu_allocated=self.cpu_allocated,
            memory_allocated=self.memory_allocated,
            latency_ms=self.latency_ms,
        )


class MockCloud:
    """Internal state machine simulating a Kubernetes cluster."""

    RPS_NORMAL = 200
    RPS_HIGH = 3500

    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.rps: int = self.RPS_NORMAL
        self.scenario: str = "healthy"
        self._build_default_services()

    def _propagate_failures(self):
        """Cascading Failure Logic: Propagates health and latency issues."""
        baselines = {"auth-api": 28, "payment-db": 12, "inventory-svc": 45, "notification-worker": 60}
        for svc_id, svc in self.services.items():
            svc.error_message = ""
            if svc.status == "Running":
                svc.latency_ms = baselines.get(svc_id, 30)

        deps = {"auth-api": "payment-db", "inventory-svc": "payment-db", "notification-worker": "inventory-svc"}

        for dependent, provider_id in deps.items():
            provider = self.services[provider_id]
            dep_svc = self.services[dependent]

            if provider.status == "Error":
                dep_svc.error_message = f"Critical: 503 Service Unavailable - Upstream {provider_id} is down."
            elif provider.latency_ms > 200:
                dep_svc.latency_ms += int(provider.latency_ms * 0.5)
                dep_svc.error_message = f"Warning: Upstream {provider_id} is slow."

    def _build_default_services(self):
        self.services = {
            "auth-api": Service("auth-api", status="Running", cpu_allocated=512, memory_allocated=2048, latency_ms=28),
            "payment-db": Service("payment-db", status="Running", cpu_allocated=1024, memory_allocated=4096, latency_ms=12),
            "inventory-svc": Service("inventory-svc", status="Running", cpu_allocated=256, memory_allocated=512, latency_ms=45),
            "notification-worker": Service("notification-worker", status="Running", cpu_allocated=128, memory_allocated=256, latency_ms=60),
        }
        self.rps = self.RPS_NORMAL

    def _apply_crash_loop(self):
        svc = self.services["payment-db"]
        svc.status = "Error"
        svc.latency_ms = 0
        svc.logs = ["[ERROR] OOMKilled", "[ERROR] CrashLoopBackOff"]

    def _apply_performance_bottleneck(self):
        self.rps = self.RPS_HIGH
        svc = self.services["auth-api"]
        svc.cpu_allocated = 128
        svc.latency_ms = 850
        svc.logs = [f"[WARN] RPS={self.rps} — CPU usage 99.8%"]
        self._propagate_failures()

    def reset(self, scenario: Optional[str] = None) -> str:
        self._build_default_services()
        self.scenario = scenario or random.choice(SCENARIOS)
        if self.scenario == "crash_loop": self._apply_crash_loop()
        elif self.scenario == "performance_bottleneck": self._apply_performance_bottleneck()
        self._propagate_failures()
        return self.scenario

    def get_all_metrics(self) -> List[ServiceMetrics]:
        return [s.to_metrics() for s in self.services.values()]

    def service_ids(self) -> List[str]:
        return list(self.services.keys())


# ---------------------------------------------------------------------------
# CloudSREEnv (OpenEnv Interface)
# ---------------------------------------------------------------------------

class CloudSREEnv:
    def __init__(self):
        self.cloud = MockCloud()
        self.current_task: Optional[str] = None
        self.step_count: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.action_history: List[Action] = []
        self._error_detected: bool = False
        self._scenario: str = "healthy"

    def reset(self, task_id: Optional[str] = None, scenario: Optional[str] = None) -> Observation:
        self.current_task = task_id or "task1_status_audit"
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.action_history = []
        self._error_detected = False

        if scenario is None:
            task_scenario_map = {
                "task1_status_audit": "crash_loop",
                "task2_self_healing": "crash_loop",
                "task3_latency_resolution": "performance_bottleneck",
            }
            scenario = task_scenario_map.get(self.current_task, "healthy")

        self._scenario = self.cloud.reset(scenario=scenario)
        obs_text = f"=== CloudSREEnv Initialized ===\nTask: {self.current_task}\nScenario: {self._scenario}\n"
        return Observation(text_output=obs_text, structured_data=self.cloud.get_all_metrics())

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            return Observation(text_output="Episode finished."), Reward(value=0.0), True, {}

        self.step_count += 1
        self.action_history.append(action)
        reward_val, obs_text, error_msg = 0.0, "", None

        # Validation
        service_id = action.service_id
        service_exists = service_id in self.cloud.services if service_id else True

        if action.action_type in (ActionType.GET_LOGS, ActionType.RESTART, ActionType.SCALE):
            if not service_id:
                obs_text, reward_val, error_msg = "[ERROR] Missing service_id.", -0.2, "Missing service_id"
            elif not service_exists:
                obs_text, reward_val, error_msg = f"[ERROR] Service '{service_id}' not found.", -0.2, "hallucinated_service"

        # Execution
        if error_msg is None:
            if action.action_type == ActionType.LIST_SERVICES:
                if len(self.action_history) > 1 and self.action_history[-2].action_type == ActionType.LIST_SERVICES:
                    reward_val, obs_text = -0.1, "Cluster state hasn't changed."
                else:
                    lines = [f"{svc.id:<20} {svc.status:<10} {svc.latency_ms}" for svc in self.cloud.services.values()]
                    obs_text = "\n".join(lines)
                    reward_val = 0.1
                    for svc in self.cloud.services.values():
                        if svc.status == "Error": self._error_detected = True

            elif action.action_type == ActionType.GET_LOGS:
                svc = self.cloud.services[service_id]
                obs_text = f"=== Logs: {service_id} ===\n" + (f"[ERROR] {svc.error_message}" if svc.error_message else "\n".join(svc.logs))
                reward_val = 0.1

            elif action.action_type == ActionType.RESTART:
                svc = self.cloud.services[service_id]
                if svc.status == "Running":
                    obs_text, reward_val = f"[WARN] {service_id} is already Running.", -0.5
                else:
                    svc.status = "Running"
                    self.cloud._propagate_failures()
                    obs_text, reward_val = f"[OK] {service_id} restarted.", 0.0

            elif action.action_type == ActionType.SCALE:
                if not action.cpu_value or action.cpu_value <= 0:
                    obs_text, reward_val = "[ERROR] Invalid cpu_value.", -0.2
                else:
                    svc = self.cloud.services[service_id]
                    svc.cpu_allocated = action.cpu_value
                    if action.cpu_value >= 2048: svc.latency_ms = 45
                    self.cloud._propagate_failures()
                    obs_text, reward_val = f"[OK] {service_id} scaled to {action.cpu_value}m.", 0.0

        # ---- Grader Fix: Strictly (0, 1) ----
        task_done, task_score = self._grade()
        if task_done:
            reward_val += task_score
            self.done = True

        reward = Reward(value=max(-1.0, min(1.0, reward_val)), reason="step_evaluation")
        self.cumulative_reward += reward.value

        info = {
            "step": self.step_count,
            "scenario": self._scenario,
            "task": self.current_task,
            "error": error_msg,
            "score": task_score # Ensure score is passed for inference.py
        }
        return Observation(text_output=obs_text, structured_data=self.cloud.get_all_metrics()), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {"task": self.current_task, "done": self.done, "cumulative_reward": round(self.cumulative_reward, 4), "services": [s.to_metrics().dict() for s in self.cloud.services.values()]}

    # ------------------------------------------------------------------
    # Graders (Deterministic — No LLMs)
    # ------------------------------------------------------------------

    def _grade(self) -> tuple[bool, float]:
        """Returns (done, score). Success scores are strictly (0, 1)."""
        if self.current_task == "task1_status_audit":
            return self._grade_task1()
        elif self.current_task == "task2_self_healing":
            return self._grade_task2()
        elif self.current_task == "task3_latency_resolution":
            return self._grade_task3()
        return False, 0.01

    def _grade_task1(self) -> tuple[bool, float]:
        """Success: 0.90"""
        if self.action_history and self.action_history[-1].action_type == ActionType.GET_LOGS:
            if self.action_history[-1].service_id == "payment-db":
                return True, 0.90
        return False, 0.01

    def _grade_task2(self) -> tuple[bool, float]:
        """Success: 0.95"""
        if self._error_detected and self.cloud.services["payment-db"].status == "Running":
            return True, 0.95
        return False, 0.01

    def _grade_task3(self) -> tuple[bool, float]:
        """Success: 0.99"""
        svc = self.cloud.services.get("auth-api")
        if svc and svc.cpu_allocated >= 2048 and svc.latency_ms < 100:
            return True, 0.99
        return False, 0.01


# ---------------------------------------------------------------------------
# HTTP Shim
# ---------------------------------------------------------------------------

def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
        app = FastAPI(title="CloudSREEnv")
        env = CloudSREEnv()

        @app.post("/reset")
        def api_reset(task_id: Optional[str] = None, scenario: Optional[str] = None):
            return JSONResponse(env.reset(task_id=task_id, scenario=scenario).dict())

        @app.post("/step")
        def api_step(action: Action):
            obs, reward, done, info = env.step(action)
            return JSONResponse({"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info})

        @app.get("/state")
        def api_state(): return JSONResponse(env.state())

        @app.get("/health")
        def health(): return {"status": "ok"}

        return app, uvicorn
    except ImportError: return None, None

if __name__ == "__main__":
    app, uvicorn = create_app()
    if app and uvicorn: uvicorn.run(app, host="0.0.0.0", port=8000)
