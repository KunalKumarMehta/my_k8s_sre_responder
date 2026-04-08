#!/usr/bin/env python3
"""
Baseline Inference Script — Kubernetes SRE Incident Responder
=============================================================

LLM agent (gpt-4o-mini by default) that plays all 3 tasks of the
K8s SRE Incident Responder OpenEnv environment.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_BASE_URL=http://localhost:8000   # or HF Space URL
    uv run python inference.py

Environment variables:
    API_BASE_URL      – LiteLLM proxy URL for LLM calls (set by grader)
    API_KEY           – LLM API key (set by grader)
    OPENAI_API_KEY    – OpenAI API key (fallback)
    ENV_BASE_URL      – Environment server URL (default: http://localhost:8000)
    MODEL_NAME        – OpenAI chat model (default: gpt-4o-mini)
    HF_TOKEN          – HuggingFace token (optional)
    LOCAL_IMAGE_NAME  – Docker image to launch locally (optional)

Hackathon: Meta PyTorch OpenEnv x Scaler (April 2026)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict

from openai import OpenAI

from my_k8s_sre_responder.models import MyK8sSreResponderAction
from my_k8s_sre_responder.client import MyK8sSreResponderEnv


# ─── Configuration ───────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TEMPERATURE = 0
SEED = 42


# ─── OpenAI Client ───────────────────────────────────────────────────────


def _make_openai_client() -> OpenAI:
    """Create OpenAI client using grader-compatible env vars."""
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        _log("WARN", {"message": "API_KEY not set. Using dummy key."})
        api_key = "dummy-key-for-validation"

    base_url = os.getenv("API_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


# ─── Logging (all to stderr) ────────────────────────────────────────────


def _log(event: str, data: Dict[str, Any]) -> None:
    parts = " ".join(f"{k}={v}" for k, v in data.items())
    print(f"[{event}] {parts}", flush=True, file=sys.stderr)


# ─── System Prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Kubernetes SRE incident responder. Each step choose ONE action to \
diagnose or fix cluster incidents. Minimize MTTR and SLA violations.

## Cluster
3 nodes (4CPU/8Gi each), 6 deployments, 12 pods.
SLA violated when error_rate>5% or p99>500ms. Cost: Rs5000/min violation.

## Actions (JSON)
Diagnostic (gather info first):
- kubectl_describe: inspect resource status
- kubectl_logs: fetch pod logs (shows root cause)
- kubectl_top: resource usage (node or pod)
- kubectl_get_events: recent cluster events

Remediation (fix issues):
- restart_pod: restart a crashed pod (1-2 step delay, symptomatic)
- scale_deployment: scale replicas (3-4 step delay)
- rollout_restart: rolling restart deployment (2-3 step delay)
- apply_config_patch: patch config/limits (2-3 step delay, ROOT CAUSE FIX)

## Strategy
1. FIRST: get_events, then describe affected deployment, then logs of crashed pod
2. IDENTIFY root cause from logs (OOM=memory limit, CrashLoop=bad config, Evicted=node pressure)
3. apply_config_patch on affected deployment with fix description (BEST action)
4. Then rollout_restart to apply changes
5. After fix applied, do_nothing and wait for recovery

## Response format (JSON only, no markdown):
{"action_type":"...", "resource_type":"pod|deployment|node|service", \
"resource_name":"...", "extra_param":"..."}\
"""


# ─── Observation → Text ─────────────────────────────────────────────────


def _obs_to_text(obs: Any) -> str:
    """Compact observation text for the LLM (<350 tokens)."""
    lines = [
        f"Step {obs.current_step}/{obs.total_steps} | Alert: {obs.alert_summary}",
    ]

    # Metrics
    m = obs.metrics
    lines.append(
        f"METRICS: cpu={m.get('cpu_usage_pct', 0)}% "
        f"mem={m.get('mem_usage_pct', 0)}% "
        f"err={m.get('error_rate_pct', 0)}% "
        f"p99={m.get('latency_p99_ms', 0)}ms"
    )

    # Cluster summary
    cs = obs.cluster_summary
    lines.append(
        f"CLUSTER: nodes={cs.get('nodes_ready', 0)}/{cs.get('nodes_total', 0)} "
        f"pods={cs.get('pods_running', 0)}run/{cs.get('pods_crashed', 0)}crash/"
        f"{cs.get('pods_pending', 0)}pend"
    )

    # Resource status (compact)
    if obs.resource_status:
        rs_parts = []
        for rname, rinfo in obs.resource_status.items():
            if rinfo.get("type") == "deployment":
                rs_parts.append(
                    f"{rname}: {rinfo.get('ready', '?')} {rinfo.get('status', '?')}"
                )
            elif rinfo.get("type") == "pod":
                rs_parts.append(
                    f"{rname}: {rinfo.get('status', '?')} "
                    f"restarts={rinfo.get('restarts', 0)}"
                )
        if rs_parts:
            lines.append("RESOURCES: " + " | ".join(rs_parts[:5]))

    # SLA status
    sla = obs.sla_status
    lines.append(
        f"SLA: violation={sla.get('violation_seconds', 0)}s "
        f"compliance={sla.get('compliance_pct', 100)}% "
        f"violated={sla.get('is_violated', False)}"
    )

    # Command output from last action
    if obs.command_output:
        # Truncate to first 5 lines for token efficiency
        cmd_lines = obs.command_output.strip().split("\n")[:5]
        lines.append("LAST_OUTPUT:\n" + "\n".join(cmd_lines))

    # Recent events (first 3)
    if obs.recent_events:
        lines.append("EVENTS: " + " | ".join(obs.recent_events[:3]))

    return "\n".join(lines)


# ─── LLM Decision ───────────────────────────────────────────────────────


def _llm_decide(
    client: OpenAI,
    obs: Any,
    task_id: str,
    step: int,
) -> MyK8sSreResponderAction:
    """Ask LLM for next action, return validated Action."""
    user_msg = _obs_to_text(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            seed=SEED,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
            content = content.rstrip("`").strip()

        # Try to extract JSON from response
        # Handle cases where LLM wraps in extra text
        if "{" in content:
            json_start = content.index("{")
            json_end = content.rindex("}") + 1
            content = content[json_start:json_end]

        parsed = json.loads(content)
        return MyK8sSreResponderAction(
            action_type=parsed.get("action_type", "do_nothing"),
            resource_type=parsed.get("resource_type", "pod"),
            resource_name=parsed.get("resource_name", ""),
            extra_param=parsed.get("extra_param", ""),
        )

    except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:
        _log(
            "WARN",
            {
                "task": task_id,
                "step": step,
                "message": "LLM parse error - fallback do_nothing",
                "error": str(exc)[:100],
            },
        )
        return MyK8sSreResponderAction(action_type="do_nothing")


# ─── Main Inference Loop ────────────────────────────────────────────────


async def run_inference() -> Dict[str, Any]:
    """Run the LLM agent on all 3 tasks sequentially."""
    openai_client = _make_openai_client()

    _log(
        "INFO",
        {
            "event": "startup",
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "hf_token_set": "yes" if HF_TOKEN else "no",
            "local_image": LOCAL_IMAGE_NAME or "none",
        },
    )

    tasks = ["task_1_easy", "task_2_medium", "task_3_hard"]
    all_results: Dict[str, Any] = {}

    try:
        if LOCAL_IMAGE_NAME:
            _log("INFO", {"message": f"Launching Docker image: {LOCAL_IMAGE_NAME}"})
            env: MyK8sSreResponderEnv = await MyK8sSreResponderEnv.from_docker_image(
                image=LOCAL_IMAGE_NAME,
                env_vars={"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
            )
            ctx_mgr = env
        else:
            env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
            ctx_mgr = MyK8sSreResponderEnv(base_url=env_url)

        async with ctx_mgr as env:
            for task_id in tasks:
                print(f"[START] task={task_id}", flush=True)

                t0 = time.time()
                try:
                    result = await env.reset(task_id=task_id, seed=SEED)
                    obs = result.observation

                    total_reward = 0.0
                    step = 0

                    while not result.done:
                        step += 1
                        action = _llm_decide(openai_client, obs, task_id, step)

                        _log(
                            "ACTION",
                            {
                                "task": task_id,
                                "step": step,
                                "type": action.action_type,
                                "resource": f"{action.resource_type}/{action.resource_name}",
                            },
                        )

                        result = await env.step(action)
                        obs = result.observation
                        reward = result.reward or 0.0
                        total_reward += reward

                        print(
                            f"[STEP] step={step} reward={round(reward, 4)}",
                            flush=True,
                        )

                    # ── Grade ───────────────────────────────────────
                    grade = obs.grade or {}
                    elapsed = round(time.time() - t0, 1)

                    task_result = {
                        "overall_score": grade.get("overall_score", 0.0),
                        "success": grade.get("success", False),
                        "sub_scores": grade.get("sub_scores", {}),
                        "total_reward": round(total_reward, 4),
                        "mttr_steps": grade.get("metrics", {}).get("mttr_steps"),
                        "sla_violation_s": grade.get("metrics", {}).get(
                            "sla_violation_seconds", 0
                        ),
                        "root_cause_fixed": grade.get("metrics", {}).get(
                            "root_cause_fixed", False
                        ),
                        "steps": step,
                        "elapsed_s": elapsed,
                        "metrics": grade.get("metrics", {}),
                    }
                    all_results[task_id] = task_result

                    score_val = round(task_result["overall_score"], 4)
                    print(
                        f"[END] task={task_id} score={score_val} steps={step}",
                        flush=True,
                    )

                except Exception as task_exc:
                    _log(
                        "ERROR",
                        {
                            "task": task_id,
                            "message": f"Task failed: {task_exc}",
                        },
                    )
                    print(
                        f"[END] task={task_id} score=0.0 steps=0",
                        flush=True,
                    )

    except Exception as env_exc:
        _log("ERROR", {"message": f"Environment connection failed: {env_exc}"})
        for task_id in tasks:
            print(f"[START] task={task_id}", flush=True)
            print(f"[END] task={task_id} score=0.0 steps=0", flush=True)

    # ─── Summary Table (stderr only) ─────────────────────────────────
    print("\n" + "=" * 78, file=sys.stderr)
    print("  K8S SRE INCIDENT RESPONDER — SCORE SUMMARY", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(
        f"  {'Task':<22} {'Score':>8} {'OK?':>5} {'MTTR':>6} "
        f"{'SLA-Viol':>9} {'RootCause':>10} {'Reward':>10} {'Time':>6}",
        file=sys.stderr,
    )
    print(
        f"  {'-' * 22} {'-' * 8} {'-' * 5} {'-' * 6} {'-' * 9} {'-' * 10} {'-' * 10} {'-' * 6}",
        file=sys.stderr,
    )

    scores = []
    for tid, r in all_results.items():
        scores.append(r["overall_score"])
        ok = "Y" if r["success"] else "N"
        mttr = r.get("mttr_steps") or "-"
        rc = "Y" if r.get("root_cause_fixed") else "N"
        print(
            f"  {tid:<22} {r['overall_score']:>8.4f} {ok:>5} "
            f"{str(mttr):>6} {r['sla_violation_s']:>8}s "
            f"{'   ' + rc:>10} "
            f"{r['total_reward']:>+10.2f} {r['elapsed_s']:>5.1f}s",
            file=sys.stderr,
        )

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'─' * 22} {'─' * 8}", file=sys.stderr)
    print(f"  {'OVERALL AVERAGE':<22} {avg:>8.4f}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    _log(
        "INFO",
        {
            "event": "inference_complete",
            "model": MODEL_NAME,
            "tasks": len(all_results),
            "avg_score": f"{avg:.4f}",
        },
    )

    return all_results


# ─── Entry Point ─────────────────────────────────────────────────────────


def main() -> None:
    try:
        asyncio.run(run_inference())
    except Exception as e:
        _log("ERROR", {"message": f"Unhandled exception: {e}"})


if __name__ == "__main__":
    main()
