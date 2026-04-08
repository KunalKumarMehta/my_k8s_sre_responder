# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kubernetes SRE Incident Responder Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyK8sSreResponderAction, MyK8sSreResponderObservation


class MyK8sSreResponderEnv(
    EnvClient[MyK8sSreResponderAction, MyK8sSreResponderObservation, State]
):
    """
    Client for the Kubernetes SRE Incident Responder Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> async with MyK8sSreResponderEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="task_1_easy", seed=42)
        ...     obs = result.observation
        ...     print(obs.alert_summary)
        ...
        ...     action = MyK8sSreResponderAction(
        ...         action_type="kubectl_describe",
        ...         resource_type="deployment",
        ...         resource_name="prod-api",
        ...     )
        ...     result = await env.step(action)

    Example with Docker:
        >>> env = await MyK8sSreResponderEnv.from_docker_image("my_k8s_sre_responder-env:latest")
        >>> try:
        ...     result = await env.reset(task_id="task_1_easy")
        ...     result = await env.step(MyK8sSreResponderAction(action_type="do_nothing"))
        ... finally:
        ...     await env.close()
    """

    def _step_payload(self, action: MyK8sSreResponderAction) -> Dict:
        """Convert action to JSON payload for step message."""
        return {
            "action_type": action.action_type,
            "resource_type": action.resource_type,
            "resource_name": action.resource_name,
            "extra_param": action.extra_param,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyK8sSreResponderObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = MyK8sSreResponderObservation(
            current_step=obs_data.get("current_step", 0),
            total_steps=obs_data.get("total_steps", 20),
            incident_id=obs_data.get("incident_id", ""),
            alert_summary=obs_data.get("alert_summary", ""),
            cluster_summary=obs_data.get("cluster_summary", {}),
            resource_status=obs_data.get("resource_status", {}),
            command_output=obs_data.get("command_output", ""),
            metrics=obs_data.get("metrics", {}),
            recent_events=obs_data.get("recent_events", []),
            sla_status=obs_data.get("sla_status", {}),
            actions_taken=obs_data.get("actions_taken", 0),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            grade=obs_data.get("grade"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
