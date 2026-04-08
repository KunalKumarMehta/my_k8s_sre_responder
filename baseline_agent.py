#!/usr/bin/env python3
"""
Rule-Based Baseline Agent — Kubernetes SRE Incident Responder
=============================================================

A deterministic rule-based agent that follows a fixed SRE playbook:
  1. Get events → 2. Describe deployment → 3. Logs of crashed pod →
  4. Top nodes → 5. Apply config patch → 6. Rollout restart → 7. Wait

This agent does NOT use an LLM. It serves as a comparison baseline.

Usage:
    PYTHONPATH=. uv run python baseline_agent.py
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from server.my_k8s_sre_responder_environment import (
        MyK8sSreResponderEnvironment,
        TASKS,
    )
    from models import MyK8sSreResponderAction
except ImportError:
    from my_k8s_sre_responder.server.my_k8s_sre_responder_environment import (
        MyK8sSreResponderEnvironment,
        TASKS,
    )
    from my_k8s_sre_responder.models import MyK8sSreResponderAction


def _find_crashed_pod(env: MyK8sSreResponderEnvironment) -> str:
    """Find first crashed/evicted pod name."""
    for pname, pstate in env._pods.items():
        if pstate["status"] in ("CrashLoopBackOff", "Error", "Evicted"):
            return pname
    return ""


def run_baseline() -> dict[str, Any]:
    """Run the rule-based baseline on all 3 tasks."""
    env = MyK8sSreResponderEnvironment()
    all_results: dict[str, Any] = {}

    for task_id in TASKS:
        obs = env.reset(seed=42, task_id=task_id)
        affected_deploy = TASKS[task_id]["affected_deployment"]
        episode_length = TASKS[task_id]["episode_length"]

        total_reward = 0.0
        crashed_pod = _find_crashed_pod(env)

        for step in range(episode_length):
            # Fixed playbook
            if step == 0:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_get_events",
                )
            elif step == 1:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_describe",
                    resource_type="deployment",
                    resource_name=affected_deploy,
                )
            elif step == 2:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_logs",
                    resource_name=crashed_pod or affected_deploy,
                )
            elif step == 3:
                action = MyK8sSreResponderAction(
                    action_type="kubectl_top",
                    resource_type="node",
                )
            elif step == 4:
                action = MyK8sSreResponderAction(
                    action_type="apply_config_patch",
                    resource_type="deployment",
                    resource_name=affected_deploy,
                    extra_param='{"memory_limit":"1024Mi","DB_POOL_SIZE":"20"}',
                )
            elif step == 5:
                action = MyK8sSreResponderAction(
                    action_type="rollout_restart",
                    resource_name=affected_deploy,
                )
            else:
                action = MyK8sSreResponderAction(action_type="do_nothing")

            obs = env.step(action)
            total_reward += obs.reward or 0.0

        grade = env.grade()
        all_results[task_id] = {
            "overall_score": grade["overall_score"],
            "success": grade["success"],
            "sub_scores": grade["sub_scores"],
            "total_reward": round(total_reward, 4),
            "metrics": grade["metrics"],
        }

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  K8S SRE RESPONDER — RULE-BASED BASELINE")
    print("=" * 72)
    print(
        f"  {'Task':<22} {'Score':>8} {'OK?':>5} {'MTTR':>6} "
        f"{'SLA-Viol':>9} {'RootCause':>10} {'Reward':>10}"
    )
    print(f"  {'-' * 22} {'-' * 8} {'-' * 5} {'-' * 6} {'-' * 9} {'-' * 10} {'-' * 10}")

    scores = []
    for tid, r in all_results.items():
        scores.append(r["overall_score"])
        ok = "Y" if r["success"] else "N"
        mttr = r["metrics"].get("mttr_steps") or "-"
        rc = "Y" if r["metrics"].get("root_cause_fixed") else "N"
        sla_v = r["metrics"].get("sla_violation_seconds", 0)
        print(
            f"  {tid:<22} {r['overall_score']:>8.4f} {ok:>5} "
            f"{str(mttr):>6} {sla_v:>8}s "
            f"{'   ' + rc:>10} "
            f"{r['total_reward']:>+10.2f}"
        )

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'─' * 22} {'─' * 8}")
    print(f"  {'OVERALL AVERAGE':<22} {avg:>8.4f}")
    print("=" * 72)

    return all_results


if __name__ == "__main__":
    run_baseline()
