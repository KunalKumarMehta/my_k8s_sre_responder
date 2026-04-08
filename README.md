---
title: My K8s Sre Responder Environment Server
emoji: 🔧
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Kubernetes SRE Incident Responder Environment

A single-agent OpenEnv environment where an autonomous SRE engineer diagnoses and resolves incidents in a simulated Kubernetes cluster. Optimises for **minimum MTTR**, **minimum downtime cost**, and **maximum SLA compliance**.

## Quick Start

```python
from my_k8s_sre_responder import MyK8sSreResponderAction, MyK8sSreResponderEnv

async with MyK8sSreResponderEnv(base_url="http://localhost:8000") as env:
    # Reset with a specific task
    result = await env.reset(task_id="task_1_easy", seed=42)
    obs = result.observation
    print(f"Alert: {obs.alert_summary}")
    print(f"Metrics: {obs.metrics}")

    # Diagnose: describe the affected deployment
    action = MyK8sSreResponderAction(
        action_type="kubectl_describe",
        resource_type="deployment",
        resource_name="prod-api",
    )
    result = await env.step(action)
    print(f"Output: {result.observation.command_output}")

    # Fix: apply config patch (root cause)
    action = MyK8sSreResponderAction(
        action_type="apply_config_patch",
        resource_name="prod-api",
        extra_param='{"memory_limit": "1024Mi"}',
    )
    result = await env.step(action)
```

## Environment Design

### Cluster Topology

- **3 worker nodes** (4 CPU, 8Gi memory each)
- **6 deployments**: `prod-api` (3 replicas), `prod-web` (2), `payment-svc` (2), `cache-redis` (1), `monitoring` (1), `log-collector` (3)
- **12 pods** total across the cluster

### Action Space (9 types)

| Action | Category | Description |
|--------|----------|-------------|
| `kubectl_describe` | Diagnostic | Inspect pod/deployment/node/service status |
| `kubectl_logs` | Diagnostic | Fetch container logs (truncated + noisy) |
| `kubectl_top` | Diagnostic | CPU/memory usage for pods or nodes |
| `kubectl_get_events` | Diagnostic | Recent cluster events |
| `restart_pod` | Remediation | Targeted pod restart (1-2 step delay) |
| `scale_deployment` | Remediation | Scale up/down (3-4 step delay) |
| `rollout_restart` | Remediation | Deployment rolling restart (2-3 step delay) |
| `apply_config_patch` | Remediation | Fix configs/limits — root cause fix (2-3 step delay) |
| `do_nothing` | Noop | Wait and observe |

### Observation Space

Each observation includes:
- **alert_summary**: Short incident description
- **cluster_summary**: Nodes ready, pods running/crashed/pending
- **resource_status**: Key resource states (affected deployment + pods)
- **command_output**: Result of last diagnostic action (truncated, noisy)
- **metrics**: CPU/mem usage, error rate, p99 latency
- **recent_events**: Last 3-5 cluster events
- **sla_status**: Downtime seconds, violation seconds, compliance %

**Partial observability**: The agent must actively query with diagnostic actions to reveal detailed information. Logs are truncated and contain noise.

### Simulation Physics

- Each `step()` advances **30 simulated seconds**
- Incidents evolve deterministically with small randomness
- Remediation actions have **realistic delayed effects** (1-4 steps)
- **SLA violation** occurs when error_rate > 5% or p99 latency > 500ms
- **Downtime cost**: ₹5,000 per minute of SLA violation

### Three Tasks

| Task | Steps | Incident | Root Cause | Success Criteria |
|------|-------|----------|------------|-----------------|
| Easy (20) | OOM pod crash | Memory limit too low | MTTR ≤ 8 steps, SLA violation ≤ 60s |
| Medium (35) | Rollout failure + latency | Bad env config in new image | MTTR ≤ 16 steps, SLA violation ≤ 180s |
| Hard (50) | Node pressure cascade | Resource contention + memory leak | MTTR ≤ 24 steps, SLA violation ≤ 300s |

### Reward Function (Dense + Shaped)

```
reward = -downtime_cost - fix_cost + diagnostic_bonus + resolution_bonus - sla_penalty
```

### Grading (0.0 – 1.0)

```
score = 0.45 × (1 - normalized_mttr) + 0.35 × sla_compliance + 0.20 × root_cause_quality
```

## Running the Server

```bash
# Local development
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Or with uv
uv run --project . server
```

## Running Inference

```bash
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:8000
uv run python inference.py
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Building Docker Image

```bash
docker build -t my_k8s_sre_responder-env:latest -f server/Dockerfile .
```

## Project Structure

```
my_k8s_sre_responder/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── models.py              # Action and Observation models
├── client.py              # MyK8sSreResponderEnv client
├── inference.py           # LLM agent (gpt-4o-mini)
├── baseline_agent.py      # Rule-based baseline agent
└── server/
    ├── __init__.py        # Server module exports
    ├── my_k8s_sre_responder_environment.py  # Core simulation engine
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
