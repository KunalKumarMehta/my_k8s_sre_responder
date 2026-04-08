# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Kubernetes SRE Incident Responder Environment.

Single-agent environment where an autonomous SRE agent diagnoses and resolves
incidents in a simulated Kubernetes cluster, optimising for minimum MTTR,
minimum downtime cost, and maximum SLA compliance.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class MyK8sSreResponderAction(Action):
    """Agent action for the K8s SRE incident responder.

    Each step the agent chooses ONE action: either a diagnostic query or a
    remediation command.

    Diagnostic actions (information-gathering, low cost):
      - kubectl_describe : Inspect status of a pod/deployment/node/service.
      - kubectl_logs     : Fetch recent log lines from a pod (optionally a
                           specific container). Logs are truncated and noisy.
      - kubectl_top      : Show CPU/memory usage for pods or nodes.
      - kubectl_get_events : List recent cluster events.

    Remediation actions (state-changing, have cost and delayed effect):
      - restart_pod      : Delete and restart a specific pod (1-2 step delay).
      - scale_deployment : Scale a deployment up or down (3-4 step delay).
      - rollout_restart  : Rolling restart of a deployment (2-3 step delay).
      - apply_config_patch : Patch a resource config (root-cause fix, 2-3 steps).

    Noop:
      - do_nothing       : Wait and observe (time still advances).
    """

    action_type: Literal[
        "kubectl_describe",
        "kubectl_logs",
        "kubectl_top",
        "kubectl_get_events",
        "restart_pod",
        "scale_deployment",
        "rollout_restart",
        "apply_config_patch",
        "do_nothing",
    ] = Field(..., description="Type of action to perform")

    resource_type: Literal["pod", "deployment", "node", "service"] = Field(
        default="pod",
        description="Kubernetes resource type to target",
    )

    resource_name: str = Field(
        default="",
        description="Name of the target resource (e.g. 'prod-api-7f8b9c-x1z2q')",
    )

    extra_param: str = Field(
        default="",
        description=(
            "Optional parameter: container name for logs, replica count for "
            "scale, JSON patch for apply_config_patch"
        ),
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class MyK8sSreResponderObservation(Observation):
    """Observation returned by the K8s SRE environment each step.

    Designed for token efficiency: all fields are short strings or small dicts.
    The agent must actively query (diagnostic actions) to reveal detailed info.
    """

    # --- Time ---
    current_step: int = Field(default=0, description="Current step (0-indexed)")
    total_steps: int = Field(default=20, description="Total steps in the episode")

    # --- Incident ---
    incident_id: str = Field(default="", description="Unique incident identifier")
    alert_summary: str = Field(
        default="",
        description="Short alert description (e.g. 'OOMKilled: prod-api pod restarting')",
    )

    # --- Cluster state (always visible — partial) ---
    cluster_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "High-level cluster health: {nodes_ready, nodes_total, "
            "pods_running, pods_crashed, pods_pending}"
        ),
    )

    # --- Resource status (key resources only) ---
    resource_status: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Short status of key resources: {resource_name: {type, status, "
            "restarts, ready, ...}}"
        ),
    )

    # --- Command output (result of last diagnostic action) ---
    command_output: str = Field(
        default="",
        description=(
            "Output from the last diagnostic command. Truncated to ~10 lines. "
            "Empty if last action was remediation or do_nothing."
        ),
    )

    # --- Metrics (always visible) ---
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Current cluster metrics: {cpu_usage_pct, mem_usage_pct, "
            "error_rate_pct, latency_p99_ms}"
        ),
    )

    # --- Recent events (always visible, last 3-5) ---
    recent_events: List[str] = Field(
        default_factory=list,
        description="Last 3-5 cluster events (most recent first)",
    )

    # --- SLA status ---
    sla_status: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "SLA tracking: {downtime_seconds, violation_seconds, "
            "compliance_pct, is_violated}"
        ),
    )

    # --- Actions taken so far ---
    actions_taken: int = Field(default=0, description="Number of actions taken so far")

    # --- Task info ---
    task_id: str = Field(default="", description="Active task identifier")
    task_description: str = Field(
        default="", description="Human-readable task description"
    )

    # --- Grading (only on final step) ---
    grade: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Grade details (overall_score, sub_scores, metrics). "
            "Populated only on final step."
        ),
    )
