# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kubernetes SRE Incident Responder — Core Environment Implementation.

Simulates a Kubernetes cluster with 3 nodes, 4-6 deployments, and 20-30 pods.
A single agent acts as an SRE engineer responding to incidents by diagnosing
issues and applying remediations.

Features:
  - 9 action types: 4 diagnostic (describe/logs/top/events) + 4 remediation
    (restart/scale/rollout/patch) + do_nothing.
  - Partial observability: agent must actively query to reveal details.
  - Realistic incident physics: OOM crashes, rollout failures, node pressure,
    cascading failures with deterministic + noisy progression.
  - Dense shaped rewards: diagnostic bonuses, downtime penalties, resolution
    bonuses, SLA violation costs.
  - Three graded tasks: Easy (single pod OOM), Medium (rollout failure),
    Hard (multi-node cascade).

Each step() advances simulated time by 30 seconds.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyK8sSreResponderAction, MyK8sSreResponderObservation
except ImportError:
    from models import MyK8sSreResponderAction, MyK8sSreResponderObservation


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

STEP_DURATION_S = 30  # Each step = 30 simulated seconds

# SLA thresholds
SLA_ERROR_RATE_THRESHOLD = 5.0  # error rate > 5% = violation
SLA_LATENCY_THRESHOLD = 500.0  # p99 > 500ms = violation

# Costs
DOWNTIME_COST_PER_MIN = 5000  # Rs 5,000 per minute of SLA violation
RESTART_COST = 200  # Small cost for restart actions
SCALE_COST = 500  # Cost for scaling
ROLLOUT_COST = 300  # Cost for rollout restart
PATCH_COST = 100  # Cost for config patch (cheapest = root cause)

# Recovery delays (in steps)
RESTART_DELAY = 1  # Pod restart takes 1 step
SCALE_DELAY = 3  # Scale takes 2-3 steps
ROLLOUT_DELAY = 2  # Rollout restart takes 1-2 steps
PATCH_DELAY = 2  # Config patch takes 1-2 steps


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Topology
# ═══════════════════════════════════════════════════════════════════════════

NODES = {
    "node-1": {"cpu_capacity": 4000, "mem_capacity_mb": 8192, "role": "worker"},
    "node-2": {"cpu_capacity": 4000, "mem_capacity_mb": 8192, "role": "worker"},
    "node-3": {"cpu_capacity": 4000, "mem_capacity_mb": 8192, "role": "worker"},
}

DEPLOYMENTS = {
    "prod-api": {
        "replicas": 3,
        "namespace": "production",
        "cpu_request": 250,
        "mem_request_mb": 256,
        "mem_limit_mb": 512,
        "node_affinity": None,
        "image": "prod-api:v2.1.0",
    },
    "prod-web": {
        "replicas": 2,
        "namespace": "production",
        "cpu_request": 200,
        "mem_request_mb": 128,
        "mem_limit_mb": 256,
        "node_affinity": None,
        "image": "prod-web:v1.8.3",
    },
    "payment-svc": {
        "replicas": 2,
        "namespace": "production",
        "cpu_request": 300,
        "mem_request_mb": 512,
        "mem_limit_mb": 1024,
        "node_affinity": None,
        "image": "payment-svc:v3.0.1",
    },
    "cache-redis": {
        "replicas": 1,
        "namespace": "infra",
        "cpu_request": 100,
        "mem_request_mb": 128,
        "mem_limit_mb": 256,
        "node_affinity": "node-2",
        "image": "redis:7.2",
    },
    "monitoring": {
        "replicas": 1,
        "namespace": "infra",
        "cpu_request": 150,
        "mem_request_mb": 256,
        "mem_limit_mb": 512,
        "node_affinity": None,
        "image": "prometheus:v2.48",
    },
    "log-collector": {
        "replicas": 3,
        "namespace": "infra",
        "cpu_request": 100,
        "mem_request_mb": 64,
        "mem_limit_mb": 128,
        "node_affinity": None,
        "image": "fluentd:v1.16",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Task Definitions
# ═══════════════════════════════════════════════════════════════════════════

TASKS: dict[str, dict[str, Any]] = {
    "task_1_easy": {
        "id": "task_1_easy",
        "name": "OOMKilled Pod Recovery",
        "difficulty": "easy",
        "description": (
            "A prod-api pod is OOMKilled and crash-looping. Diagnose the root "
            "cause (memory limit too low) and fix it. "
            "Goal: resolve within 8 steps, SLA violation < 60s."
        ),
        "episode_length": 20,
        "incident_type": "oom_crash",
        "affected_deployment": "prod-api",
        "affected_node": "node-1",
        "root_cause": "memory_limit",
        "root_cause_description": "Memory limit of 512Mi is too low for current workload",
        "success_criteria": {"max_mttr_steps": 10, "max_sla_violation_s": 240},
    },
    "task_2_medium": {
        "id": "task_2_medium",
        "name": "Failed Rollout with Latency Spike",
        "difficulty": "medium",
        "description": (
            "A prod-api deployment rollout introduced a bad config causing high "
            "latency and pod crashes. Diagnose (bad env config in new image) and "
            "either rollback or patch. "
            "Goal: resolve within 16 steps, SLA violation < 180s."
        ),
        "episode_length": 35,
        "incident_type": "rollout_failure",
        "affected_deployment": "prod-api",
        "affected_node": "node-1",
        "root_cause": "bad_config",
        "root_cause_description": "New image v2.2.0 has invalid DB_POOL_SIZE=0 causing connection failures",
        "success_criteria": {"max_mttr_steps": 16, "max_sla_violation_s": 360},
    },
    "task_3_hard": {
        "id": "task_3_hard",
        "name": "Node Pressure Cascade",
        "difficulty": "hard",
        "description": (
            "node-1 is under memory pressure causing pod evictions. This cascades "
            "to other nodes as evicted pods reschedule. Multiple services degrade. "
            "Fix requires: drain node, patch resource limits, rebalance. "
            "Goal: resolve within 24 steps, SLA violation < 300s, no recurrence."
        ),
        "episode_length": 50,
        "incident_type": "node_pressure_cascade",
        "affected_deployment": "prod-api",
        "affected_node": "node-1",
        "root_cause": "resource_contention",
        "root_cause_description": "Memory leak in log-collector + undersized limits on prod-api cause node pressure",
        "success_criteria": {"max_mttr_steps": 24, "max_sla_violation_s": 480},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Log / Event Templates
# ═══════════════════════════════════════════════════════════════════════════

OOM_LOGS = [
    "java.lang.OutOfMemoryError: Java heap space",
    "  at com.api.service.CacheManager.loadAll(CacheManager.java:142)",
    "  at com.api.handler.RequestHandler.process(RequestHandler.java:89)",
    "FATAL: memory allocation failed, current usage 498Mi / limit 512Mi",
    "Container killed by OOM (exit code 137)",
    "INFO: Graceful shutdown initiated...",
    "ERROR: Failed to flush write buffer - connection reset",
    "WARN: Health check endpoint /healthz returning 503",
]

ROLLOUT_LOGS = [
    "ERROR: Failed to connect to database pool",
    "FATAL: DB_POOL_SIZE=0 is invalid, must be > 0",
    "panic: runtime error: invalid pool configuration",
    "  goroutine 1 [running]:",
    "  main.initDB() at /app/db.go:45",
    "ERROR: Readiness probe failed: connection refused",
    "WARN: 5 consecutive probe failures, marking unhealthy",
    "INFO: Shutting down due to readiness failure",
]

NODE_PRESSURE_LOGS = [
    "kernel: [52413.2] Out of memory: Kill process 8192 (fluentd)",
    "kernel: [52413.3] Killed process 8192 total-vm:1048576kB",
    "kubelet: evicting pod log-collector-abc12 due to memory pressure",
    "kubelet: node memory pressure condition detected",
    "kubelet: taint node.kubernetes.io/memory-pressure applied",
    "WARN: prod-api-xyz89 failed liveness probe (timeout 3s)",
    "ERROR: cannot schedule pod: insufficient memory on all nodes",
    "WARN: payment-svc latency p99 increased to 1200ms",
]

NORMAL_LOGS = [
    "INFO: Request processed successfully in 45ms",
    "INFO: Health check passed",
    "DEBUG: Cache hit ratio: 94.2%",
    "INFO: Connection pool stats: active=12 idle=8 max=20",
    "INFO: Metrics exported successfully",
]

CLUSTER_EVENTS_OOM = [
    "0s    Warning  OOMKilling       pod/prod-api-7f8b-x1z2   Container killed (exit 137)",
    "30s   Warning  BackOff          pod/prod-api-7f8b-x1z2   Back-off restarting failed",
    "1m    Normal   Pulling          pod/prod-api-7f8b-x1z2   Pulling image prod-api:v2.1.0",
    "2m    Warning  Unhealthy        pod/prod-api-7f8b-x1z2   Readiness probe failed",
    "3m    Normal   Scheduled        pod/prod-api-7f8b-x1z2   Assigned to node-1",
]

CLUSTER_EVENTS_ROLLOUT = [
    "0s    Warning  Unhealthy        pod/prod-api-8a9c-q3r4   Readiness probe failed",
    "15s   Warning  BackOff          pod/prod-api-8a9c-q3r4   CrashLoopBackOff",
    "30s   Normal   RollingUpdate    deploy/prod-api           New ReplicaSet prod-api-8a9c",
    "1m    Warning  ProgressDeadline deploy/prod-api           Failed progressing",
    "2m    Normal   ScalingDown      deploy/prod-api           Old ReplicaSet scaled to 1",
]

CLUSTER_EVENTS_NODE = [
    "0s    Warning  EvictionThresholdMet   node/node-1          Memory pressure",
    "10s   Warning  Evicted          pod/log-collector-d5e6   Evicted due to memory",
    "20s   Warning  NodeNotReady     node/node-1              Node not ready",
    "40s   Warning  FailedScheduling pod/prod-api-7f8b-x1z2   Insufficient memory",
    "1m    Warning  Unhealthy        pod/payment-svc-c3d4     Liveness probe timeout",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def _gen_pod_name(deploy: str, rng: random.Random) -> str:
    """Generate a realistic pod name like 'prod-api-7f8b9c-x1z2q'."""
    rs = "".join(rng.choices("abcdef0123456789", k=6))
    pod = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
    return f"{deploy}-{rs}-{pod}"


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════


class MyK8sSreResponderEnvironment(Environment):
    """Kubernetes SRE Incident Responder RL Environment.

    A single-agent environment simulating K8s cluster incidents over 20-50
    time-steps (each step = 30 simulated seconds). The agent receives
    observations about cluster state, metrics, and alerts, then chooses
    diagnostic or remediation actions.

    Compatible with the OpenEnv framework (Environment base class).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        self._rng_seed: int = 42

        # Episode config
        self._task_id: str = "task_1_easy"
        self._task: dict[str, Any] = TASKS[self._task_id]
        self._episode_length: int = 20
        self._current_step: int = 0
        self._done: bool = False

        # Cluster simulation state
        self._pods: dict[str, dict[str, Any]] = {}  # pod_name -> state
        self._nodes: dict[str, dict[str, Any]] = {}  # node_name -> state
        self._deploy_state: dict[str, dict[str, Any]] = {}  # deploy -> state

        # Incident tracking
        self._incident_active: bool = True
        self._incident_resolved_step: Optional[int] = None
        self._root_cause_fixed: bool = False
        self._mitigation_applied: bool = False
        self._recurrence_count: int = 0

        # SLA tracking
        self._sla_violation_seconds: int = 0
        self._total_downtime_seconds: int = 0
        self._simulated_time_s: int = 0

        # Metrics (cluster-wide)
        self._cpu_usage_pct: float = 45.0
        self._mem_usage_pct: float = 55.0
        self._error_rate_pct: float = 0.0
        self._latency_p99_ms: float = 120.0

        # Action tracking
        self._actions_log: list[dict[str, Any]] = []
        self._diagnostic_targets: set[str] = set()  # resources agent has queried
        self._last_command_output: str = ""
        self._pending_remediations: list[dict[str, Any]] = []

        # Ensure ready even if reset() is skipped
        self.reset()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MyK8sSreResponderObservation:
        """Reset the environment for a new episode."""
        task_id = kwargs.get("task_id", "task_1_easy")
        if task_id not in TASKS:
            task_id = "task_1_easy"

        self._task_id = task_id
        self._task = TASKS[task_id]
        self._episode_length = self._task["episode_length"]

        # Seeding
        self._rng_seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(self._rng_seed)

        # State
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._current_step = 0
        self._done = False

        # Initialize cluster
        self._init_cluster()

        # Inject incident
        self._inject_incident()

        # Reset tracking
        self._incident_active = True
        self._incident_resolved_step = None
        self._root_cause_fixed = False
        self._mitigation_applied = False
        self._recurrence_count = 0

        self._sla_violation_seconds = 0
        self._total_downtime_seconds = 0
        self._simulated_time_s = 0

        self._actions_log = []
        self._diagnostic_targets = set()
        self._last_command_output = ""
        self._pending_remediations = []

        return self._make_observation(reward=0.0)

    # ------------------------------------------------------------------
    # Cluster initialisation
    # ------------------------------------------------------------------
    def _init_cluster(self) -> None:
        """Build initial healthy cluster state."""
        # Nodes
        self._nodes = {}
        for name, spec in NODES.items():
            self._nodes[name] = {
                "name": name,
                "status": "Ready",
                "cpu_used": 0,
                "mem_used_mb": 0,
                "cpu_capacity": spec["cpu_capacity"],
                "mem_capacity_mb": spec["mem_capacity_mb"],
                "conditions": ["Ready"],
                "taints": [],
            }

        # Deployments and pods
        self._pods = {}
        self._deploy_state = {}
        node_names = list(NODES.keys())

        for dname, dspec in DEPLOYMENTS.items():
            self._deploy_state[dname] = {
                "name": dname,
                "desired_replicas": dspec["replicas"],
                "ready_replicas": dspec["replicas"],
                "image": dspec["image"],
                "namespace": dspec["namespace"],
                "rollout_status": "healthy",
            }

            for i in range(dspec["replicas"]):
                pod_name = _gen_pod_name(dname, self._rng)
                if dspec["node_affinity"]:
                    assigned_node = dspec["node_affinity"]
                else:
                    assigned_node = node_names[i % len(node_names)]

                self._pods[pod_name] = {
                    "name": pod_name,
                    "deployment": dname,
                    "node": assigned_node,
                    "status": "Running",
                    "ready": True,
                    "restarts": 0,
                    "cpu_usage": dspec["cpu_request"]
                    * (0.5 + self._rng.random() * 0.3),
                    "mem_usage_mb": dspec["mem_request_mb"]
                    * (0.5 + self._rng.random() * 0.4),
                    "mem_limit_mb": dspec["mem_limit_mb"],
                    "namespace": dspec["namespace"],
                    "image": dspec["image"],
                    "start_time": "2026-04-08T10:00:00Z",
                    "last_state": None,
                }

                # Update node resource usage
                self._nodes[assigned_node]["cpu_used"] += int(
                    self._pods[pod_name]["cpu_usage"]
                )
                self._nodes[assigned_node]["mem_used_mb"] += int(
                    self._pods[pod_name]["mem_usage_mb"]
                )

        # Set baseline metrics
        self._cpu_usage_pct = 45.0 + self._rng.uniform(-5, 5)
        self._mem_usage_pct = 55.0 + self._rng.uniform(-5, 5)
        self._error_rate_pct = 0.1 + self._rng.uniform(0, 0.3)
        self._latency_p99_ms = 120.0 + self._rng.uniform(-20, 30)

    # ------------------------------------------------------------------
    # Incident injection
    # ------------------------------------------------------------------
    def _inject_incident(self) -> None:
        """Inject the incident scenario for the current task."""
        incident_type = self._task["incident_type"]
        affected_deploy = self._task["affected_deployment"]

        if incident_type == "oom_crash":
            self._inject_oom_crash(affected_deploy)
        elif incident_type == "rollout_failure":
            self._inject_rollout_failure(affected_deploy)
        elif incident_type == "node_pressure_cascade":
            self._inject_node_pressure()

    def _inject_oom_crash(self, deploy: str) -> None:
        """One pod of the deployment is OOMKilled and crash-looping."""
        target_pod = None
        for pname, pstate in self._pods.items():
            if (
                pstate["deployment"] == deploy
                and pstate["node"] == self._task["affected_node"]
            ):
                target_pod = pname
                break
        if target_pod is None:
            target_pod = next(
                p for p, s in self._pods.items() if s["deployment"] == deploy
            )

        self._pods[target_pod]["status"] = "CrashLoopBackOff"
        self._pods[target_pod]["ready"] = False
        self._pods[target_pod]["restarts"] = 3
        self._pods[target_pod]["last_state"] = "OOMKilled"
        self._pods[target_pod]["mem_usage_mb"] = self._pods[target_pod]["mem_limit_mb"]

        # Degrade deployment
        self._deploy_state[deploy]["ready_replicas"] -= 1

        # Spike metrics
        self._error_rate_pct = 8.0 + self._rng.uniform(0, 4)
        self._latency_p99_ms = 350.0 + self._rng.uniform(0, 100)
        self._mem_usage_pct = 75.0 + self._rng.uniform(0, 10)

    def _inject_rollout_failure(self, deploy: str) -> None:
        """Rollout introduced bad config; new pods crash, old pods degraded."""
        # Update deployment to show rollout in progress
        self._deploy_state[deploy]["image"] = "prod-api:v2.2.0"
        self._deploy_state[deploy]["rollout_status"] = "progressing_failed"

        # Crash 2 out of 3 pods (new version)
        count = 0
        for pname, pstate in self._pods.items():
            if pstate["deployment"] == deploy and count < 2:
                pstate["status"] = "CrashLoopBackOff"
                pstate["ready"] = False
                pstate["restarts"] = 5
                pstate["image"] = "prod-api:v2.2.0"
                pstate["last_state"] = "Error"
                count += 1

        self._deploy_state[deploy]["ready_replicas"] = 1

        # Severe metric degradation
        self._error_rate_pct = 15.0 + self._rng.uniform(0, 5)
        self._latency_p99_ms = 800.0 + self._rng.uniform(0, 200)
        self._cpu_usage_pct = 70.0 + self._rng.uniform(0, 10)

    def _inject_node_pressure(self) -> None:
        """Node-1 under memory pressure; pods get evicted, cascade begins."""
        node = self._task["affected_node"]
        self._nodes[node]["status"] = "NotReady"
        self._nodes[node]["conditions"] = ["MemoryPressure", "NotReady"]
        self._nodes[node]["taints"] = ["node.kubernetes.io/memory-pressure"]
        self._nodes[node]["mem_used_mb"] = int(
            self._nodes[node]["mem_capacity_mb"] * 0.95
        )

        # Evict pods on affected node
        evicted = 0
        for pname, pstate in self._pods.items():
            if pstate["node"] == node and evicted < 3:
                pstate["status"] = "Evicted"
                pstate["ready"] = False
                pstate["last_state"] = "Evicted"
                deploy = pstate["deployment"]
                self._deploy_state[deploy]["ready_replicas"] = max(
                    0, self._deploy_state[deploy]["ready_replicas"] - 1
                )
                evicted += 1

        # Cascade: overload other nodes
        self._nodes["node-2"]["mem_used_mb"] = int(
            self._nodes["node-2"]["mem_capacity_mb"] * 0.85
        )

        # Severe metric degradation
        self._error_rate_pct = 20.0 + self._rng.uniform(0, 8)
        self._latency_p99_ms = 1200.0 + self._rng.uniform(0, 300)
        self._mem_usage_pct = 90.0 + self._rng.uniform(0, 5)
        self._cpu_usage_pct = 80.0 + self._rng.uniform(0, 10)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(  # type: ignore[override]
        self,
        action: MyK8sSreResponderAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MyK8sSreResponderObservation:
        """Execute one time-step (30 simulated seconds)."""
        if self._done:
            return self._make_observation(reward=0.0)

        self._state.step_count += 1
        self._current_step += 1
        self._simulated_time_s += STEP_DURATION_S

        # ── Phase 0: Process pending remediations ─────────────────────
        self._process_pending_remediations()

        # ── Phase 1: Apply agent action ───────────────────────────────
        action_reward = self._apply_action(action)

        # ── Phase 2: Evolve incident (if still active) ────────────────
        self._evolve_incident()

        # ── Phase 3: Update metrics ───────────────────────────────────
        self._update_metrics()

        # ── Phase 4: Track SLA ────────────────────────────────────────
        sla_penalty = self._track_sla()

        # ── Phase 5: Check resolution ─────────────────────────────────
        resolution_bonus = self._check_resolution()

        # ── Phase 6: Compute total reward ─────────────────────────────
        reward = action_reward + sla_penalty + resolution_bonus

        # ── Phase 7: Check termination ────────────────────────────────
        if self._current_step >= self._episode_length:
            self._done = True

        # ── Log ───────────────────────────────────────────────────────
        self._actions_log.append(
            {
                "step": self._current_step,
                "action_type": action.action_type,
                "resource": f"{action.resource_type}/{action.resource_name}",
                "reward": round(reward, 4),
                "incident_active": self._incident_active,
                "error_rate": round(self._error_rate_pct, 2),
                "latency": round(self._latency_p99_ms, 1),
            }
        )

        return self._make_observation(reward=reward)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _apply_action(self, action: MyK8sSreResponderAction) -> float:
        """Apply the agent's action. Returns reward component."""
        at = action.action_type
        rt = action.resource_type
        rn = action.resource_name
        ep = action.extra_param
        reward = 0.0

        if at == "do_nothing":
            self._last_command_output = ""
            return 0.0

        # ── Diagnostic actions ────────────────────────────────────────
        if at == "kubectl_describe":
            reward = self._handle_describe(rt, rn)
        elif at == "kubectl_logs":
            reward = self._handle_logs(rn, ep)
        elif at == "kubectl_top":
            reward = self._handle_top(rt)
        elif at == "kubectl_get_events":
            reward = self._handle_events()

        # ── Remediation actions ───────────────────────────────────────
        elif at == "restart_pod":
            reward = self._handle_restart_pod(rn)
        elif at == "scale_deployment":
            reward = self._handle_scale(rn, ep)
        elif at == "rollout_restart":
            reward = self._handle_rollout_restart(rn)
        elif at == "apply_config_patch":
            reward = self._handle_config_patch(rn, ep)
        else:
            self._last_command_output = f"Unknown action type: {at}"
            reward = -0.5

        return reward

    # ------------------------------------------------------------------
    # Diagnostic handlers
    # ------------------------------------------------------------------
    def _handle_describe(self, resource_type: str, resource_name: str) -> float:
        """Handle kubectl describe."""
        reward = 0.0
        self._diagnostic_targets.add(f"{resource_type}/{resource_name}")

        if resource_type == "pod":
            pod = self._pods.get(resource_name)
            if pod is None:
                # Try fuzzy match on deployment name
                pod = self._find_pod_by_deploy(resource_name)
                if pod:
                    resource_name = pod["name"]

            if pod:
                lines = [
                    f"Name:         {pod['name']}",
                    f"Namespace:    {pod['namespace']}",
                    f"Node:         {pod['node']}",
                    f"Status:       {pod['status']}",
                    f"Ready:        {pod['ready']}",
                    f"Restarts:     {pod['restarts']}",
                    f"Image:        {pod['image']}",
                    f"CPU Usage:    {int(pod['cpu_usage'])}m",
                    f"Memory:       {int(pod['mem_usage_mb'])}Mi / {pod['mem_limit_mb']}Mi",
                ]
                if pod.get("last_state"):
                    lines.append(f"Last State:   {pod['last_state']}")
                    if pod["last_state"] == "OOMKilled":
                        lines.append("Reason:       OOMKilled (exit code 137)")
                        lines.append(
                            f"Message:      Container exceeded memory limit ({pod['mem_limit_mb']}Mi)"
                        )

                self._last_command_output = "\n".join(lines)

                # Reward for diagnosing affected resources
                if pod["deployment"] == self._task["affected_deployment"]:
                    reward = 0.3
                    if pod["status"] != "Running":
                        reward = 0.5  # Found the broken pod
            else:
                self._last_command_output = f'Error: pod "{resource_name}" not found'
                reward = -0.1

        elif resource_type == "deployment":
            deploy = self._deploy_state.get(resource_name)
            if deploy:
                lines = [
                    f"Name:               {deploy['name']}",
                    f"Namespace:          {deploy['namespace']}",
                    f"Replicas:           {deploy['ready_replicas']}/{deploy['desired_replicas']} ready",
                    f"Image:              {deploy['image']}",
                    f"RolloutStatus:      {deploy['rollout_status']}",
                ]
                if deploy["rollout_status"] != "healthy":
                    lines.append("Conditions:")
                    lines.append(
                        "  Available=False   Reason=MinimumReplicasUnavailable"
                    )
                    lines.append("  Progressing=False Reason=ProgressDeadlineExceeded")
                self._last_command_output = "\n".join(lines)

                if resource_name == self._task["affected_deployment"]:
                    reward = 0.3
            else:
                self._last_command_output = (
                    f'Error: deployment "{resource_name}" not found'
                )
                reward = -0.1

        elif resource_type == "node":
            node = self._nodes.get(resource_name)
            if node:
                cpu_pct = node["cpu_used"] / node["cpu_capacity"] * 100
                mem_pct = node["mem_used_mb"] / node["mem_capacity_mb"] * 100
                lines = [
                    f"Name:         {node['name']}",
                    f"Status:       {node['status']}",
                    f"Conditions:   {', '.join(node['conditions'])}",
                    f"CPU:          {node['cpu_used']}m / {node['cpu_capacity']}m ({cpu_pct:.0f}%)",
                    f"Memory:       {node['mem_used_mb']}Mi / {node['mem_capacity_mb']}Mi ({mem_pct:.0f}%)",
                    f"Taints:       {', '.join(node['taints']) if node['taints'] else 'none'}",
                ]
                self._last_command_output = "\n".join(lines)

                if resource_name == self._task["affected_node"]:
                    reward = 0.3
            else:
                self._last_command_output = f'Error: node "{resource_name}" not found'
                reward = -0.1

        elif resource_type == "service":
            self._last_command_output = (
                f"Name:      {resource_name}\n"
                f"Type:      ClusterIP\n"
                f"Endpoints: {self._count_ready_pods(resource_name)} ready endpoints"
            )
            reward = 0.1

        return reward

    def _handle_logs(self, pod_name: str, container: str) -> float:
        """Handle kubectl logs — returns truncated, noisy logs."""
        reward = 0.0
        self._diagnostic_targets.add(f"logs/{pod_name}")

        pod = self._pods.get(pod_name)
        if pod is None:
            pod = self._find_pod_by_deploy(pod_name)
            if pod:
                pod_name = pod["name"]

        if pod is None:
            self._last_command_output = f'Error: pod "{pod_name}" not found'
            return -0.1

        incident_type = self._task["incident_type"]
        deploy = pod["deployment"]

        # Select log template based on pod state and incident type
        if (
            pod["status"] in ("CrashLoopBackOff", "Error")
            or pod.get("last_state") == "OOMKilled"
        ):
            if (
                incident_type == "oom_crash"
                and deploy == self._task["affected_deployment"]
            ):
                logs = list(OOM_LOGS)
                reward = 0.5  # Critical diagnostic info
            elif (
                incident_type == "rollout_failure"
                and deploy == self._task["affected_deployment"]
            ):
                logs = list(ROLLOUT_LOGS)
                reward = 0.5
            elif incident_type == "node_pressure_cascade":
                logs = list(NODE_PRESSURE_LOGS)
                reward = 0.4
            else:
                logs = list(NORMAL_LOGS)
                reward = 0.1
        elif pod["status"] == "Evicted":
            logs = [
                "Error: pod has been evicted",
                "Previous logs not available for evicted pods",
                "Last state: Evicted due to node memory pressure",
            ]
            reward = 0.3
        else:
            logs = list(NORMAL_LOGS)
            reward = 0.1

        # Add noise: random INFO lines interspersed
        noisy_logs = []
        for line in logs:
            noisy_logs.append(line)
            if self._rng.random() < 0.3:
                noisy_logs.append(
                    f"INFO: [{self._rng.randint(100, 999)}] routine operation completed"
                )

        # Truncate to 10 lines
        self._last_command_output = "\n".join(noisy_logs[:10])
        return reward

    def _handle_top(self, resource_type: str) -> float:
        """Handle kubectl top pods/nodes."""
        self._diagnostic_targets.add(f"top/{resource_type}")

        if resource_type == "node":
            lines = ["NAME      CPU(cores)  CPU%   MEMORY(bytes)  MEMORY%"]
            for nname, nstate in self._nodes.items():
                cpu_pct = nstate["cpu_used"] / nstate["cpu_capacity"] * 100
                mem_pct = nstate["mem_used_mb"] / nstate["mem_capacity_mb"] * 100
                lines.append(
                    f"{nname:<10}{nstate['cpu_used']}m        {cpu_pct:.0f}%    "
                    f"{nstate['mem_used_mb']}Mi          {mem_pct:.0f}%"
                )
            self._last_command_output = "\n".join(lines)
            return 0.2

        elif resource_type == "pod":
            lines = ["NAME                          CPU(cores)  MEMORY(bytes)"]
            for pname, pstate in self._pods.items():
                if pstate["status"] in ("Running", "CrashLoopBackOff"):
                    lines.append(
                        f"{pname:<30}{int(pstate['cpu_usage'])}m         "
                        f"{int(pstate['mem_usage_mb'])}Mi"
                    )
            self._last_command_output = "\n".join(lines[:12])  # Truncate
            return 0.2

        self._last_command_output = "Usage: kubectl top [pods|nodes]"
        return 0.0

    def _handle_events(self) -> float:
        """Handle kubectl get events."""
        self._diagnostic_targets.add("events")
        incident_type = self._task["incident_type"]

        if incident_type == "oom_crash":
            events = list(CLUSTER_EVENTS_OOM)
        elif incident_type == "rollout_failure":
            events = list(CLUSTER_EVENTS_ROLLOUT)
        elif incident_type == "node_pressure_cascade":
            events = list(CLUSTER_EVENTS_NODE)
        else:
            events = ["No significant events"]

        # Add some noise events
        if self._rng.random() < 0.4:
            events.append(
                f"{self._rng.randint(5, 15)}m   Normal   Pulled   "
                f"pod/monitoring-{self._rng.randint(1000, 9999)}   "
                "Successfully pulled image"
            )

        self._last_command_output = "\n".join(events[:8])
        return 0.3

    # ------------------------------------------------------------------
    # Remediation handlers
    # ------------------------------------------------------------------
    def _handle_restart_pod(self, pod_name: str) -> float:
        """Restart (delete) a pod — it will be recreated by the deployment."""
        pod = self._pods.get(pod_name)
        if pod is None:
            pod = self._find_pod_by_deploy(pod_name)
            if pod:
                pod_name = pod["name"]

        if pod is None:
            self._last_command_output = f'Error: pod "{pod_name}" not found'
            return -0.3

        self._last_command_output = f'pod "{pod_name}" deleted'

        # Queue remediation with delay
        self._pending_remediations.append(
            {
                "type": "restart_pod",
                "target": pod_name,
                "deployment": pod["deployment"],
                "steps_remaining": RESTART_DELAY
                + (1 if self._rng.random() < 0.3 else 0),
            }
        )

        # Mark pod as terminating
        self._pods[pod_name]["status"] = "Terminating"
        self._pods[pod_name]["ready"] = False

        # Symptomatic fix flag — doesn't fix root cause
        if pod["deployment"] == self._task["affected_deployment"]:
            self._mitigation_applied = True

        return -RESTART_COST / 1000.0  # Small negative for cost

    def _handle_scale(self, deploy_name: str, extra_param: str) -> float:
        """Scale a deployment up or down."""
        deploy = self._deploy_state.get(deploy_name)
        if deploy is None:
            self._last_command_output = f'Error: deployment "{deploy_name}" not found'
            return -0.3

        try:
            target_replicas = (
                int(extra_param) if extra_param else deploy["desired_replicas"] + 1
            )
        except ValueError:
            target_replicas = deploy["desired_replicas"] + 1

        target_replicas = max(1, min(10, target_replicas))
        old = deploy["desired_replicas"]
        deploy["desired_replicas"] = target_replicas

        self._last_command_output = (
            f"deployment.apps/{deploy_name} scaled from {old} to {target_replicas}"
        )

        self._pending_remediations.append(
            {
                "type": "scale",
                "target": deploy_name,
                "deployment": deploy_name,
                "new_replicas": target_replicas,
                "old_replicas": old,
                "steps_remaining": SCALE_DELAY + (1 if self._rng.random() < 0.3 else 0),
            }
        )

        if deploy_name == self._task["affected_deployment"]:
            self._mitigation_applied = True

        return -SCALE_COST / 1000.0

    def _handle_rollout_restart(self, deploy_name: str) -> float:
        """Rollout restart a deployment."""
        deploy = self._deploy_state.get(deploy_name)
        if deploy is None:
            self._last_command_output = f'Error: deployment "{deploy_name}" not found'
            return -0.3

        self._last_command_output = f"deployment.apps/{deploy_name} restarted"

        self._pending_remediations.append(
            {
                "type": "rollout_restart",
                "target": deploy_name,
                "deployment": deploy_name,
                "steps_remaining": ROLLOUT_DELAY
                + (1 if self._rng.random() < 0.3 else 0),
            }
        )

        # Rollout restart on a rollout_failure task is a reasonable fix
        if (
            deploy_name == self._task["affected_deployment"]
            and self._task["incident_type"] == "rollout_failure"
        ):
            self._mitigation_applied = True

        return -ROLLOUT_COST / 1000.0

    def _handle_config_patch(self, resource_name: str, patch_json: str) -> float:
        """Apply a config patch — the root-cause fix action."""
        deploy = self._deploy_state.get(resource_name)
        if deploy is None:
            # Try interpreting as a pod name -> look up deployment
            pod = self._pods.get(resource_name)
            if pod:
                resource_name = pod["deployment"]
                deploy = self._deploy_state.get(resource_name)

        if deploy is None:
            self._last_command_output = f'Error: resource "{resource_name}" not found'
            return -0.3

        self._last_command_output = f"deployment.apps/{resource_name} patched"

        # Check if this is the correct root cause fix
        patch_lower = patch_json.lower() if patch_json else ""
        incident_type = self._task["incident_type"]
        is_root_cause_fix = False

        if resource_name == self._task["affected_deployment"]:
            if incident_type == "oom_crash":
                # Any mention of memory/limit in patch = root cause fix
                if any(
                    kw in patch_lower
                    for kw in ["memory", "limit", "mem", "1024", "1gi", "heap"]
                ):
                    is_root_cause_fix = True
                else:
                    # Even without the right patch content, patching the right
                    # deployment counts as root cause attempt
                    is_root_cause_fix = True

            elif incident_type == "rollout_failure":
                # Fix the bad config or rollback image
                if any(
                    kw in patch_lower
                    for kw in [
                        "db_pool",
                        "pool_size",
                        "image",
                        "rollback",
                        "v2.1",
                        "config",
                    ]
                ):
                    is_root_cause_fix = True
                else:
                    is_root_cause_fix = True

            elif incident_type == "node_pressure_cascade":
                # Fix resource limits
                if any(
                    kw in patch_lower
                    for kw in ["limit", "resource", "memory", "evict", "drain"]
                ):
                    is_root_cause_fix = True
                else:
                    is_root_cause_fix = True

        self._pending_remediations.append(
            {
                "type": "config_patch",
                "target": resource_name,
                "deployment": resource_name,
                "is_root_cause": is_root_cause_fix,
                "steps_remaining": PATCH_DELAY + (1 if self._rng.random() < 0.3 else 0),
            }
        )

        if is_root_cause_fix:
            self._root_cause_fixed = True
            self._mitigation_applied = True

        return -PATCH_COST / 1000.0

    # ------------------------------------------------------------------
    # Remediation processing (delayed effects)
    # ------------------------------------------------------------------
    def _process_pending_remediations(self) -> None:
        """Advance and apply pending remediations."""
        still_pending = []

        for rem in self._pending_remediations:
            rem["steps_remaining"] -= 1

            if rem["steps_remaining"] <= 0:
                self._apply_remediation(rem)
            else:
                still_pending.append(rem)

        self._pending_remediations = still_pending

    def _apply_remediation(self, rem: dict[str, Any]) -> None:
        """Apply a completed remediation."""
        rtype = rem["type"]
        deploy_name = rem["deployment"]

        if rtype == "restart_pod":
            target = rem["target"]
            if target in self._pods:
                pod = self._pods[target]
                # If root cause is NOT fixed, pod will crash again
                if self._root_cause_fixed or self._task["incident_type"] not in (
                    "oom_crash",
                    "rollout_failure",
                ):
                    pod["status"] = "Running"
                    pod["ready"] = True
                    pod["restarts"] += 1
                    pod["last_state"] = None
                    pod["mem_usage_mb"] = (
                        DEPLOYMENTS[deploy_name]["mem_request_mb"] * 0.6
                    )
                else:
                    # Recurrence — will crash again
                    pod["status"] = "CrashLoopBackOff"
                    pod["ready"] = False
                    pod["restarts"] += 1
                    self._recurrence_count += 1

                self._update_deploy_ready_count(deploy_name)

        elif rtype == "scale":
            deploy = self._deploy_state.get(deploy_name)
            if deploy:
                new_reps = rem["new_replicas"]
                old_reps = rem["old_replicas"]
                # Add new pods if scaling up
                if new_reps > old_reps:
                    nodes = list(NODES.keys())
                    for _ in range(new_reps - old_reps):
                        new_pod_name = _gen_pod_name(deploy_name, self._rng)
                        node = self._rng.choice(nodes)
                        dspec = DEPLOYMENTS.get(deploy_name, {})
                        self._pods[new_pod_name] = {
                            "name": new_pod_name,
                            "deployment": deploy_name,
                            "node": node,
                            "status": "Running",
                            "ready": True,
                            "restarts": 0,
                            "cpu_usage": dspec.get("cpu_request", 200) * 0.5,
                            "mem_usage_mb": dspec.get("mem_request_mb", 128) * 0.5,
                            "mem_limit_mb": dspec.get("mem_limit_mb", 256),
                            "namespace": dspec.get("namespace", "production"),
                            "image": deploy.get("image", "unknown"),
                            "start_time": f"step-{self._current_step}",
                            "last_state": None,
                        }
                self._update_deploy_ready_count(deploy_name)

        elif rtype == "rollout_restart":
            # Restart all pods in deployment
            for pname, pstate in self._pods.items():
                if pstate["deployment"] == deploy_name:
                    if (
                        self._root_cause_fixed
                        or self._task["incident_type"] == "oom_crash"
                    ):
                        pstate["status"] = "Running"
                        pstate["ready"] = True
                        pstate["restarts"] += 1
                        pstate["last_state"] = None
                    elif self._task["incident_type"] == "rollout_failure":
                        # Rollout restart with bad config still fails
                        if not self._root_cause_fixed:
                            pstate["status"] = "CrashLoopBackOff"
                            pstate["ready"] = False
                            pstate["restarts"] += 1
                            self._recurrence_count += 1
                        else:
                            pstate["status"] = "Running"
                            pstate["ready"] = True
                            pstate["restarts"] += 1
                            pstate["last_state"] = None
                    else:
                        pstate["status"] = "Running"
                        pstate["ready"] = True
                        pstate["restarts"] += 1
                        pstate["last_state"] = None

            self._update_deploy_ready_count(deploy_name)

        elif rtype == "config_patch":
            if rem.get("is_root_cause"):
                # Fix all pods in the deployment
                for pname, pstate in self._pods.items():
                    if pstate["deployment"] == deploy_name:
                        pstate["status"] = "Running"
                        pstate["ready"] = True
                        pstate["last_state"] = None
                        # Reduce resource usage to normal
                        dspec = DEPLOYMENTS.get(deploy_name, {})
                        pstate["mem_usage_mb"] = dspec.get("mem_request_mb", 128) * 0.6
                        pstate["cpu_usage"] = dspec.get("cpu_request", 200) * 0.5

                deploy = self._deploy_state.get(deploy_name)
                if deploy:
                    deploy["rollout_status"] = "healthy"

                self._update_deploy_ready_count(deploy_name)

                # Fix node if node_pressure task
                if self._task["incident_type"] == "node_pressure_cascade":
                    for nname, nstate in self._nodes.items():
                        nstate["status"] = "Ready"
                        nstate["conditions"] = ["Ready"]
                        nstate["taints"] = []
                        nstate["mem_used_mb"] = int(nstate["mem_capacity_mb"] * 0.55)

    # ------------------------------------------------------------------
    # Incident evolution
    # ------------------------------------------------------------------
    def _evolve_incident(self) -> None:
        """Evolve the incident state over time (degradation if unresolved)."""
        if not self._incident_active:
            return

        incident_type = self._task["incident_type"]
        noise = self._rng.uniform(-2, 2)

        if incident_type == "oom_crash":
            # Gradual degradation if not fixed
            if not self._root_cause_fixed:
                self._error_rate_pct = _clamp(
                    self._error_rate_pct + 0.5 + noise * 0.3, 5.0, 25.0
                )
                self._latency_p99_ms = _clamp(
                    self._latency_p99_ms + 10 + noise * 5, 200.0, 1500.0
                )

        elif incident_type == "rollout_failure":
            if not self._root_cause_fixed:
                self._error_rate_pct = _clamp(
                    self._error_rate_pct + 1.0 + noise * 0.5, 10.0, 35.0
                )
                self._latency_p99_ms = _clamp(
                    self._latency_p99_ms + 20 + noise * 10, 500.0, 2500.0
                )

        elif incident_type == "node_pressure_cascade":
            if not self._root_cause_fixed:
                # Cascade worsens: more pods affected
                self._error_rate_pct = _clamp(
                    self._error_rate_pct + 1.5 + noise * 0.8, 15.0, 40.0
                )
                self._latency_p99_ms = _clamp(
                    self._latency_p99_ms + 30 + noise * 15, 800.0, 3000.0
                )
                self._mem_usage_pct = _clamp(self._mem_usage_pct + 1.0, 85.0, 98.0)

                # Random pod crashes on overloaded nodes
                if self._current_step % 4 == 0:
                    for pname, pstate in self._pods.items():
                        if (
                            pstate["status"] == "Running"
                            and pstate["node"] in ("node-1", "node-2")
                            and self._rng.random() < 0.15
                        ):
                            pstate["status"] = "CrashLoopBackOff"
                            pstate["ready"] = False
                            pstate["last_state"] = "OOMKilled"
                            self._update_deploy_ready_count(pstate["deployment"])
                            break

    # ------------------------------------------------------------------
    # Metrics update
    # ------------------------------------------------------------------
    def _update_metrics(self) -> None:
        """Update cluster-wide metrics based on pod health."""
        # Count healthy pods for affected deployment
        affected = self._task["affected_deployment"]
        deploy = self._deploy_state.get(affected)
        if deploy is None:
            return

        ready_ratio = deploy["ready_replicas"] / max(deploy["desired_replicas"], 1)

        if self._incident_active:
            if ready_ratio >= 0.9:
                # All/most pods healthy → metrics recovering
                target_error = 0.5 + self._rng.uniform(0, 0.5)
                target_latency = 130 + self._rng.uniform(-20, 30)
                self._error_rate_pct = _clamp(
                    self._error_rate_pct * 0.7 + target_error * 0.3, 0.1, 40.0
                )
                self._latency_p99_ms = _clamp(
                    self._latency_p99_ms * 0.7 + target_latency * 0.3, 80.0, 3000.0
                )
                self._mem_usage_pct = _clamp(
                    self._mem_usage_pct * 0.9 + 55.0 * 0.1, 40.0, 98.0
                )
                self._cpu_usage_pct = _clamp(
                    self._cpu_usage_pct * 0.9 + 45.0 * 0.1, 30.0, 95.0
                )
        else:
            # Incident resolved — converge to healthy
            self._error_rate_pct = _clamp(
                self._error_rate_pct * 0.6 + 0.3 * 0.4, 0.1, 2.0
            )
            self._latency_p99_ms = _clamp(
                self._latency_p99_ms * 0.6 + 120.0 * 0.4, 80.0, 500.0
            )
            self._mem_usage_pct = _clamp(
                self._mem_usage_pct * 0.8 + 55.0 * 0.2, 40.0, 70.0
            )
            self._cpu_usage_pct = _clamp(
                self._cpu_usage_pct * 0.8 + 45.0 * 0.2, 30.0, 60.0
            )

    # ------------------------------------------------------------------
    # SLA tracking
    # ------------------------------------------------------------------
    def _track_sla(self) -> float:
        """Track SLA compliance, return penalty."""
        is_violated = (
            self._error_rate_pct > SLA_ERROR_RATE_THRESHOLD
            or self._latency_p99_ms > SLA_LATENCY_THRESHOLD
        )

        if is_violated:
            self._sla_violation_seconds += STEP_DURATION_S
            self._total_downtime_seconds += STEP_DURATION_S
            # Downtime cost penalty (normalised)
            cost = (STEP_DURATION_S / 60.0) * DOWNTIME_COST_PER_MIN
            return -cost / 5000.0  # Normalise to ~-1.0 per step of violation

        return 0.0

    # ------------------------------------------------------------------
    # Resolution check
    # ------------------------------------------------------------------
    def _check_resolution(self) -> float:
        """Check if the incident is resolved. Return resolution bonus."""
        if not self._incident_active:
            return 0.0

        # Check if all pods of affected deployment are healthy
        affected = self._task["affected_deployment"]
        deploy = self._deploy_state.get(affected)
        if deploy is None:
            return 0.0

        all_healthy = deploy["ready_replicas"] >= deploy["desired_replicas"]
        metrics_ok = (
            self._error_rate_pct <= SLA_ERROR_RATE_THRESHOLD
            and self._latency_p99_ms <= SLA_LATENCY_THRESHOLD
        )

        if all_healthy and metrics_ok:
            self._incident_active = False
            self._incident_resolved_step = self._current_step

            if self._root_cause_fixed:
                return 5.0  # Big bonus for root cause fix
            elif self._mitigation_applied:
                return 2.0  # Smaller bonus for mitigation only
            else:
                return 1.0  # Resolved by luck / natural recovery

        return 0.0

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    def grade(self) -> dict[str, Any]:
        """Programmatic grader — returns 0.0–1.0 overall score.

        Components:
          - mttr_score       (45 %) : time-to-resolution vs episode length
          - sla_compliance   (35 %) : fraction of time without SLA violation
          - root_cause_quality (20 %) : 1.0 root cause, 0.5 mitigation, 0.0 unresolved
        """
        task = self._task
        episode_len = task["episode_length"]
        max_time_s = episode_len * STEP_DURATION_S

        # MTTR score
        if self._incident_resolved_step is not None:
            mttr_steps = self._incident_resolved_step
            max_mttr = task["success_criteria"]["max_mttr_steps"]
            # Best: 0.0 (resolved instantly), Worst: 1.0 (never resolved)
            normalized_mttr = mttr_steps / episode_len
            mttr_score = max(0.0, 1.0 - normalized_mttr)
            # Bonus for resolving faster than target (capped at <1.0)
            if mttr_steps <= max_mttr:
                mttr_score = min(0.99, mttr_score * 1.2)
        else:
            mttr_score = 0.0  # Never resolved

        # SLA compliance
        if max_time_s > 0:
            sla_compliance = max(0.0, 1.0 - (self._sla_violation_seconds / max_time_s))
        else:
            sla_compliance = 1.0

        # Bonus for staying under violation target (capped at <1.0)
        max_violation = task["success_criteria"]["max_sla_violation_s"]
        if self._sla_violation_seconds <= max_violation:
            sla_compliance = min(0.99, sla_compliance * 1.1)

        # Root cause quality (capped at <1.0)
        if self._root_cause_fixed and self._recurrence_count == 0:
            root_cause_quality = 0.99
        elif self._root_cause_fixed:
            root_cause_quality = 0.7  # Fixed but had recurrence
        elif self._mitigation_applied:
            root_cause_quality = 0.5
        elif self._incident_resolved_step is not None:
            root_cause_quality = 0.3
        else:
            root_cause_quality = 0.0

        # Weighted total (capped at <1.0)
        overall = 0.45 * mttr_score + 0.35 * sla_compliance + 0.20 * root_cause_quality
        # Ensure strictly between 0 and 1
        overall = max(0.001, min(0.99, overall))

        # Success check
        criteria = task["success_criteria"]
        resolved_in_time = (
            self._incident_resolved_step is not None
            and self._incident_resolved_step <= criteria["max_mttr_steps"]
        )
        sla_ok = self._sla_violation_seconds <= criteria["max_sla_violation_s"]
        success = resolved_in_time and sla_ok

        return {
            "task_id": self._task_id,
            "task_name": task["name"],
            "difficulty": task["difficulty"],
            "overall_score": round(overall, 4),
            "success": success,
            "sub_scores": {
                "mttr_score": round(mttr_score, 4),
                "sla_compliance": round(sla_compliance, 4),
                "root_cause_quality": round(root_cause_quality, 4),
            },
            "metrics": {
                "mttr_steps": self._incident_resolved_step,
                "mttr_seconds": (
                    self._incident_resolved_step * STEP_DURATION_S
                    if self._incident_resolved_step
                    else None
                ),
                "sla_violation_seconds": self._sla_violation_seconds,
                "total_downtime_seconds": self._total_downtime_seconds,
                "root_cause_fixed": self._root_cause_fixed,
                "mitigation_applied": self._mitigation_applied,
                "recurrence_count": self._recurrence_count,
                "incident_resolved": self._incident_resolved_step is not None,
                "steps_completed": self._current_step,
                "diagnostics_run": len(self._diagnostic_targets),
                "actions_taken": len(self._actions_log),
            },
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _make_observation(self, reward: float) -> MyK8sSreResponderObservation:
        """Build the observation for the current state."""
        task = self._task
        affected_deploy = task["affected_deployment"]

        # Cluster summary
        nodes_ready = sum(1 for n in self._nodes.values() if n["status"] == "Ready")
        pods_running = sum(1 for p in self._pods.values() if p["status"] == "Running")
        pods_crashed = sum(
            1
            for p in self._pods.values()
            if p["status"] in ("CrashLoopBackOff", "Error")
        )
        pods_pending = sum(
            1
            for p in self._pods.values()
            if p["status"] in ("Pending", "Terminating", "Evicted")
        )

        cluster_summary = {
            "nodes_ready": nodes_ready,
            "nodes_total": len(self._nodes),
            "pods_running": pods_running,
            "pods_crashed": pods_crashed,
            "pods_pending": pods_pending,
            "pods_total": len(self._pods),
        }

        # Resource status (only show affected + key resources)
        resource_status: dict[str, Any] = {}

        # Always show affected deployment
        deploy = self._deploy_state.get(affected_deploy)
        if deploy:
            resource_status[affected_deploy] = {
                "type": "deployment",
                "ready": f"{deploy['ready_replicas']}/{deploy['desired_replicas']}",
                "status": deploy["rollout_status"],
                "image": deploy["image"],
            }

        # Show pods of affected deployment (short form)
        for pname, pstate in self._pods.items():
            if pstate["deployment"] == affected_deploy:
                resource_status[pname] = {
                    "type": "pod",
                    "status": pstate["status"],
                    "restarts": pstate["restarts"],
                    "node": pstate["node"],
                    "ready": pstate["ready"],
                }

        # Recent events (3-5 most recent)
        recent_events = self._get_recent_events()

        # SLA status
        max_time = self._episode_length * STEP_DURATION_S
        elapsed = self._current_step * STEP_DURATION_S
        is_violated = (
            self._error_rate_pct > SLA_ERROR_RATE_THRESHOLD
            or self._latency_p99_ms > SLA_LATENCY_THRESHOLD
        )
        compliance = max(0.0, 1.0 - (self._sla_violation_seconds / max(elapsed, 1)))

        sla_status = {
            "downtime_seconds": self._total_downtime_seconds,
            "violation_seconds": self._sla_violation_seconds,
            "compliance_pct": round(compliance * 100, 1),
            "is_violated": is_violated,
        }

        # Alert summary
        alert_summary = self._get_alert_summary()

        return MyK8sSreResponderObservation(
            current_step=self._current_step,
            total_steps=self._episode_length,
            incident_id=f"INC-{self._task_id.split('_')[1]}-{str(self._rng_seed)[-4:]}",
            alert_summary=alert_summary,
            cluster_summary=cluster_summary,
            resource_status=resource_status,
            command_output=self._last_command_output,
            metrics={
                "cpu_usage_pct": round(self._cpu_usage_pct, 1),
                "mem_usage_pct": round(self._mem_usage_pct, 1),
                "error_rate_pct": round(self._error_rate_pct, 1),
                "latency_p99_ms": round(self._latency_p99_ms, 0),
            },
            recent_events=recent_events,
            sla_status=sla_status,
            actions_taken=len(self._actions_log),
            task_id=self._task_id,
            task_description=task["description"],
            grade=self.grade() if self._done else None,
            done=self._done,
            reward=reward,
            metadata={
                "step": self._current_step,
                "incident_active": self._incident_active,
                "root_cause_fixed": self._root_cause_fixed,
                "pending_remediations": len(self._pending_remediations),
                "simulated_time_s": self._simulated_time_s,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_pod_by_deploy(self, name: str) -> Optional[dict[str, Any]]:
        """Find a pod that matches a deployment name prefix."""
        # Check if name is a deployment
        if name in self._deploy_state:
            for pname, pstate in self._pods.items():
                if pstate["deployment"] == name:
                    return pstate
        # Check partial match
        for pname, pstate in self._pods.items():
            if name in pname or name in pstate["deployment"]:
                return pstate
        return None

    def _count_ready_pods(self, deploy_or_svc: str) -> int:
        """Count ready pods for a deployment/service name."""
        count = 0
        for pstate in self._pods.values():
            if pstate["deployment"] == deploy_or_svc and pstate["ready"]:
                count += 1
        return count

    def _update_deploy_ready_count(self, deploy_name: str) -> None:
        """Recount ready replicas for a deployment."""
        deploy = self._deploy_state.get(deploy_name)
        if deploy:
            ready = sum(
                1
                for p in self._pods.values()
                if p["deployment"] == deploy_name and p["ready"]
            )
            deploy["ready_replicas"] = ready

    def _get_alert_summary(self) -> str:
        """Generate a concise alert summary."""
        if not self._incident_active:
            return "RESOLVED: All services healthy"

        incident_type = self._task["incident_type"]
        affected = self._task["affected_deployment"]

        if incident_type == "oom_crash":
            return f"ALERT: OOMKilled pod in {affected} — crash-looping, error rate {self._error_rate_pct:.0f}%"
        elif incident_type == "rollout_failure":
            return f"ALERT: Rollout failure in {affected} — pods crashing, latency {self._latency_p99_ms:.0f}ms"
        elif incident_type == "node_pressure_cascade":
            node = self._task["affected_node"]
            return (
                f"ALERT: Node {node} MemoryPressure — pod evictions, "
                f"cascading failures, error rate {self._error_rate_pct:.0f}%"
            )
        return "ALERT: Unknown incident"

    def _get_recent_events(self) -> list[str]:
        """Return 3-5 most recent events."""
        incident_type = self._task["incident_type"]

        if not self._incident_active:
            return ["Normal  All systems operational"]

        if incident_type == "oom_crash":
            events = CLUSTER_EVENTS_OOM[:3]
        elif incident_type == "rollout_failure":
            events = CLUSTER_EVENTS_ROLLOUT[:3]
        elif incident_type == "node_pressure_cascade":
            events = CLUSTER_EVENTS_NODE[:4]
        else:
            events = []

        # Add dynamic events based on step
        if self._pending_remediations:
            rem = self._pending_remediations[0]
            events.insert(
                0,
                f"0s    Normal   Applying         {rem['type']} on {rem['target']} ({rem['steps_remaining']} steps)",
            )

        return events[:5]

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Kubernetes SRE Incident Responder — Smoke Test")
    print("=" * 70)

    env = MyK8sSreResponderEnvironment()

    for task_id in TASKS:
        print(f"\n--- {task_id} ---")
        obs = env.reset(seed=42, task_id=task_id)
        print(f"  Reset: step={obs.current_step}/{obs.total_steps}")
        print(f"  Alert: {obs.alert_summary}")
        print(f"  Cluster: {obs.cluster_summary}")
        print(f"  Metrics: {obs.metrics}")

        steps = TASKS[task_id]["episode_length"]
        total_reward = 0.0

        # Simple baseline: describe → logs → patch
        for i in range(steps):
            if i == 0:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_get_events",
                )
            elif i == 1:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_describe",
                    resource_type="deployment",
                    resource_name=TASKS[task_id]["affected_deployment"],
                )
            elif i == 2:
                # Find a crashed pod
                crashed = [
                    p
                    for p, s in env._pods.items()
                    if s["status"] == "CrashLoopBackOff" or s["status"] == "Evicted"
                ]
                pod_name = crashed[0] if crashed else "prod-api"
                action = MyK8sSreResponderAction(
                    action_type="kubectl_logs",
                    resource_name=pod_name,
                )
            elif i == 3:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_top",
                    resource_type="node",
                )
            elif i == 4:
                action = MyK8sSreResponderAction(
                    action_type="apply_config_patch",
                    resource_type="deployment",
                    resource_name=TASKS[task_id]["affected_deployment"],
                    extra_param='{"memory_limit": "1024Mi"}',
                )
            elif i == 5:
                action = MyK8sSreResponderAction(
                    action_type="rollout_restart",
                    resource_name=TASKS[task_id]["affected_deployment"],
                )
            else:
                action = MyK8sSreResponderAction(action_type="do_nothing")

            obs = env.step(action)
            total_reward += obs.reward or 0.0

            if obs.command_output and i <= 5:
                preview = obs.command_output[:80].replace("\n", " | ")
                print(f"  Step {i}: [{action.action_type}] → {preview}...")

        print(f"  Final metrics: {obs.metrics}")
        print(f"  SLA: {obs.sla_status}")
        print(f"  Total reward: {total_reward:.2f}")

        grade_result = env.grade()
        print(
            f"  Grade: {grade_result['overall_score']:.3f} | "
            f"success={grade_result['success']}"
        )
        print(f"  Sub-scores: {grade_result['sub_scores']}")
        print(
            f"  MTTR: {grade_result['metrics']['mttr_steps']} steps | "
            f"SLA violation: {grade_result['metrics']['sla_violation_seconds']}s | "
            f"Root cause: {grade_result['metrics']['root_cause_fixed']}"
        )

    print("\n✅ Smoke test passed!")
