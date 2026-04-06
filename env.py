"""
CloudSREEnv — OpenEnv-compliant Kubernetes-style SRE simulator.
Implements step(), reset(), and state() per the OpenEnv spec.
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
    ):
        self.id = id
        self.status = status
        self.cpu_allocated = cpu_allocated
        self.memory_allocated = memory_allocated
        self.latency_ms = latency_ms
        self.logs: List[str] = logs or []

    def to_metrics(self) -> ServiceMetrics:
        return ServiceMetrics(
            id=self.id,
            status=self.status,
            cpu_allocated=self.cpu_allocated,
            memory_allocated=self.memory_allocated,
            latency_ms=self.latency_ms,
        )

    def __repr__(self):
        return f"<Service id={self.id} status={self.status}>"


SCENARIOS = ["healthy", "crash_loop", "performance_bottleneck"]


class MockCloud:
    """Internal state machine simulating a Kubernetes cluster."""

    RPS_NORMAL = 200
    RPS_HIGH = 3500

    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.rps: int = self.RPS_NORMAL
        self.scenario: str = "healthy"
        self._build_default_services()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_default_services(self):
        self.services = {
            "auth-api": Service(
                id="auth-api",
                status="Running",
                cpu_allocated=512,
                memory_allocated=2048,
                latency_ms=28,
                logs=["[INFO] auth-api started", "[INFO] Listening on :8080"],
            ),
            "payment-db": Service(
                id="payment-db",
                status="Running",
                cpu_allocated=1024,
                memory_allocated=4096,
                latency_ms=12,
                logs=["[INFO] payment-db started", "[INFO] Connected to PG primary"],
            ),
            "inventory-svc": Service(
                id="inventory-svc",
                status="Running",
                cpu_allocated=256,
                memory_allocated=512,
                latency_ms=45,
                logs=["[INFO] inventory-svc started"],
            ),
            "notification-worker": Service(
                id="notification-worker",
                status="Running",
                cpu_allocated=128,
                memory_allocated=256,
                latency_ms=60,
                logs=["[INFO] notification-worker ready"],
            ),
        }
        self.rps = self.RPS_NORMAL

    def _apply_crash_loop(self):
        svc = self.services["payment-db"]
        svc.status = "Error"
        svc.latency_ms = 0
        svc.logs = [
            "[INFO] payment-db started",
            "[WARN] Memory pressure detected: 3800/4096 MB used",
            "[ERROR] OOMKilled: container exceeded memory limit",
            "[ERROR] CrashLoopBackOff: back-off 5m0s restarting failed container",
            "[ERROR] Failed to allocate 512MB for query buffer — OOM",
        ]

    def _apply_performance_bottleneck(self):
        self.rps = self.RPS_HIGH
        svc = self.services["auth-api"]
        svc.cpu_allocated = 128          # Under-provisioned
        svc.latency_ms = 850             # Spiked
        svc.logs = [
            "[INFO] auth-api started",
            f"[WARN] RPS={self.rps} — CPU throttling detected",
            "[WARN] Request queue depth: 2048",
            "[ERROR] p99 latency=850ms exceeds SLO threshold (200ms)",
            "[ERROR] CPU usage 99.8% — scaling required",
        ]

    def reset(self, scenario: Optional[str] = None) -> str:
        """Randomly (or explicitly) set a failure scenario."""
        self._build_default_services()
        if scenario is None:
            scenario = random.choice(SCENARIOS)
        self.scenario = scenario

        if scenario == "crash_loop":
            self._apply_crash_loop()
        elif scenario == "performance_bottleneck":
            self._apply_performance_bottleneck()

        return scenario

    def get_all_metrics(self) -> List[ServiceMetrics]:
        return [s.to_metrics() for s in self.services.values()]

    def service_ids(self) -> List[str]:
        return list(self.services.keys())


# ---------------------------------------------------------------------------
# CloudSREEnv  (OpenEnv interface)
# ---------------------------------------------------------------------------

class CloudSREEnv:
    """
    OpenEnv-compliant environment.
    Public API: reset(), step(), state()
    """

    TASK_IDS = ["task1_status_audit", "task2_self_healing", "task3_latency_resolution"]

    def __init__(self):
        self.cloud = MockCloud()
        self.current_task: Optional[str] = None
        self.step_count: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.reward_history: List[float] = []
        self.action_history: List[Action] = []
        self._error_detected: bool = False   # For task2 grader
        self._scenario: str = "healthy"

    # ------------------------------------------------------------------
    # OpenEnv required methods
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, scenario: Optional[str] = None) -> Observation:
        """Reset environment to a new episode."""
        self.current_task = task_id or "task1_status_audit"
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.reward_history = []
        self.action_history = []
        self._error_detected = False

        # Map task → scenario
        if scenario is None:
            task_scenario_map = {
                "task1_status_audit": "crash_loop",
                "task2_self_healing": "crash_loop",
                "task3_latency_resolution": "performance_bottleneck",
            }
            scenario = task_scenario_map.get(self.current_task, "healthy")

        self._scenario = self.cloud.reset(scenario=scenario)

        obs_text = (
            f"=== CloudSREEnv Initialized ===\n"
            f"Task      : {self.current_task}\n"
            f"Scenario  : {self._scenario}\n"
            f"Services  : {', '.join(self.cloud.service_ids())}\n"
            f"RPS       : {self.cloud.rps}\n"
            f"Hint      : Run LIST_SERVICES to assess cluster health.\n"
        )
        return Observation(text_output=obs_text, structured_data=self.cloud.get_all_metrics())

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action.
        Returns (observation, reward, done, info)
        """
        if self.done:
            obs = Observation(text_output="[WARN] Episode already finished. Call reset().")
            return obs, Reward(value=0.0, reason="episode_done"), True, {}

        self.step_count += 1
        self.action_history.append(action)

        reward_val = 0.0
        reward_reason = ""
        obs_text = ""
        error_msg = None

        # ---- Validate service_id when required -------------------------
        service_id = action.service_id
        service_exists = service_id in self.cloud.services if service_id else True

        if action.action_type in (ActionType.GET_LOGS, ActionType.RESTART, ActionType.SCALE):
            if not service_id:
                obs_text = "[ERROR] Missing service_id for this action."
                reward_val = -0.2
                reward_reason = "missing_service_id"
                error_msg = "Missing service_id"
            elif not service_exists:
                obs_text = (
                    f"[ERROR] Service '{service_id}' not found. "
                    f"Valid IDs: {self.cloud.service_ids()}"
                )
                reward_val = -0.2
                reward_reason = "hallucinated_service"
                error_msg = f"Unknown service: {service_id}"

        # ---- Execute action if no prior error ---------------------------
        if error_msg is None:
            if action.action_type == ActionType.LIST_SERVICES:
                if len(self.action_history) > 1 and self.action_history[-2].action_type == ActionType.LIST_SERVICES:
                    reward_val = -0.1 # Penalty for repeating the same diagnostic
                    obs_text = "Cluster state hasn't changed. Take a corrective action."
                else:
                    lines = ["SERVICE            STATUS     CPU(m)  MEM(Mi)  LATENCY(ms)"]
                    lines.append("-" * 60)
                    for svc in self.cloud.services.values():
                        lines.append(
                            f"{svc.id:<20} {svc.status:<10} {svc.cpu_allocated:<7} "
                            f"{svc.memory_allocated:<8} {svc.latency_ms}"
                        )
                    obs_text = "\n".join(lines)
                    reward_val = 0.1
                    reward_reason = "useful_diagnostic"

                    # Detect error for task2 grader
                    for svc in self.cloud.services.values():
                        if svc.status == "Error":
                            self._error_detected = True

            elif action.action_type == ActionType.GET_LOGS:
                svc = self.cloud.services[service_id]
                logs = "\n".join(svc.logs) if svc.logs else "(no logs)"
                obs_text = f"=== Logs: {service_id} ===\n{logs}"
                reward_val = 0.1
                reward_reason = "useful_diagnostic"

            elif action.action_type == ActionType.RESTART:
                svc = self.cloud.services[service_id]
                if svc.status == "Running":
                    obs_text = f"[WARN] {service_id} is already Running. Unnecessary restart triggered."
                    reward_val = -0.5
                    reward_reason = "unnecessary_restart"
                else:
                    svc.status = "Running"
                    svc.latency_ms = 35
                    svc.logs.append("[INFO] Container restarted successfully by SRE agent.")
                    obs_text = f"[OK] {service_id} restarted. Status: Running."
                    reward_val = 0.0
                    reward_reason = "restart_executed"

            elif action.action_type == ActionType.SCALE:
                cpu_value = action.cpu_value
                if cpu_value is None or cpu_value <= 0:
                    obs_text = "[ERROR] Invalid cpu_value. Must be a positive integer."
                    reward_val = -0.2
                    reward_reason = "invalid_cpu_value"
                    error_msg = "Invalid cpu_value"
                else:
                    svc = self.cloud.services[service_id]
                    old_cpu = svc.cpu_allocated
                    svc.cpu_allocated = cpu_value
                    if cpu_value >= 2048:
                        svc.latency_ms = 45
                    
                    # NEW: Add a hint if the service is still crashed
                    status_hint = ""
                    if svc.status == "Error":
                        status_hint = " Note: Service is still in Error state and needs a RESTART."
                    
                    obs_text = (
                        f"[OK] {service_id} scaled: CPU {old_cpu}m → {cpu_value}m.{status_hint}"
                    )
                    reward_val = 0.0
                    reward_reason = "scale_executed"

        # ---- Grader ----------------------------------------------------
        task_done, task_score = self._grade()
        if task_done:
            reward_val += task_score
            reward_reason += "+task_complete"
            self.done = True

        reward = Reward(value=max(-1.0, min(1.0, reward_val)), reason=reward_reason)
        self.cumulative_reward += reward.value
        self.reward_history.append(reward.value)

        obs = Observation(
            text_output=obs_text,
            structured_data=self.cloud.get_all_metrics(),
        )
        info: Dict[str, Any] = {
            "step": self.step_count,
            "scenario": self._scenario,
            "task": self.current_task,
            "error": error_msg,
        }
        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state snapshot."""
        return {
            "task": self.current_task,
            "scenario": self._scenario,
            "step": self.step_count,
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "rps": self.cloud.rps,
            "services": [s.to_metrics().dict() for s in self.cloud.services.values()],
        }

    # ------------------------------------------------------------------
    # Graders (deterministic — no LLMs)
    # ------------------------------------------------------------------

    def _grade(self) -> tuple[bool, float]:
        """
        Returns (done, score).
        Graders inspect MockCloud state directly — fully deterministic.
        """
        if self.current_task == "task1_status_audit":
            return self._grade_task1()
        elif self.current_task == "task2_self_healing":
            return self._grade_task2()
        elif self.current_task == "task3_latency_resolution":
            return self._grade_task3()
        return False, 0.0

    def _grade_task1(self) -> tuple[bool, float]:
        """Success: last action is GET_LOGS on the failing service."""
        if not self.action_history:
            return False, 0.0
        last = self.action_history[-1]
        if last.action_type != ActionType.GET_LOGS:
            return False, 0.0
        # Find which service was in error at start (payment-db for crash_loop)
        failing_id = self._get_failing_service_id()
        if failing_id and last.service_id == failing_id:
            return True, 1.0
        return False, 0.0

    def _grade_task2(self) -> tuple[bool, float]:
        """Success: previously-Error service is now Running."""
        if not self._error_detected:
            return False, 0.0
        failing_id = self._get_originally_failing_service()
        if failing_id and self.cloud.services[failing_id].status == "Running":
            return True, 1.0
        return False, 0.0

    def _grade_task3(self) -> tuple[bool, float]:
        """Success: auth-api cpu >= 2048 AND latency_ms < 100."""
        svc = self.cloud.services.get("auth-api")
        if svc and svc.cpu_allocated >= 2048 and svc.latency_ms < 100:
            return True, 1.0
        return False, 0.0

    def _get_failing_service_id(self) -> Optional[str]:
        for svc in self.cloud.services.values():
            if svc.status == "Error":
                return svc.id
        # payment-db was failing at start in crash_loop even if restarted
        if self._scenario == "crash_loop":
            return "payment-db"
        return None

    def _get_originally_failing_service(self) -> Optional[str]:
        """Return the service that was originally in Error state."""
        if self._scenario == "crash_loop":
            return "payment-db"
        return None


# ---------------------------------------------------------------------------
# OpenEnv HTTP shim  (serves /step, /reset, /state on port 8000)
# ---------------------------------------------------------------------------

def create_app():
    """Minimal WSGI-compatible HTTP wrapper so `openenv validate` can probe it."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI(title="CloudSREEnv", version="1.0.0")
        env = CloudSREEnv()

        @app.post("/reset")
        def api_reset(task_id: Optional[str] = None, scenario: Optional[str] = None):
            obs = env.reset(task_id=task_id, scenario=scenario)
            return JSONResponse(obs.dict())

        @app.post("/step")
        def api_step(action: Action):
            obs, reward, done, info = env.step(action)
            return JSONResponse({
                "observation": obs.dict(),
                "reward": reward.dict(),
                "done": done,
                "info": info,
            })

        @app.get("/state")
        def api_state():
            return JSONResponse(env.state())

        @app.get("/health")
        def health():
            return {"status": "ok", "env": "CloudSREEnv"}

        return app, uvicorn

    except ImportError:
        return None, None


if __name__ == "__main__":
    app, uvicorn = create_app()
    if app and uvicorn:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("[WARN] fastapi/uvicorn not installed. Running smoke test instead.")
        env = CloudSREEnv()
        obs = env.reset(task_id="task1_status_audit")
        print(obs.text_output)
        act = Action(action_type=ActionType.LIST_SERVICES)
        obs, rew, done, info = env.step(act)
        print(obs.text_output)
        print(f"Reward: {rew.value}  Done: {done}")
