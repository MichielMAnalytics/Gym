# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox observations built only from lifecycle facts an AsyncSandbox actually saw."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Literal

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.server_runtime.verifier_observability import observe_verifier_sandbox
from nemo_gym.trajectory.trajectory_record import SandboxObservation, SandboxUsageMeasurement


LOGGER = logging.getLogger(__name__)


def resource_usage_fields(sandbox: AsyncSandbox) -> dict[str, Any]:
    """The ``SandboxObservation`` usage fields measured for a sandbox ``sandbox`` started, or none.

    ``cpu_time_s``, ``peak_memory_mib`` and ``resource_usage_source`` are set only from a container-scope cgroup
    counter: a process-tree lower bound or a shared cgroup's total is never reported as the sandbox's own.
    """
    measurements = [SandboxUsageMeasurement.model_validate(asdict(summary)) for summary in sandbox.resource_usage]
    if not measurements:
        return {}
    fields: dict[str, Any] = {"resource_usage": measurements}
    container = next((item for item in measurements if item.scope == "container"), None)
    if container is not None:
        fields["resource_usage_source"] = f"{container.source}:{container.scope}"
        if container.cpu_kind == "counter" and container.cpu_usage_usec_total is not None:
            fields["cpu_time_s"] = container.cpu_usage_usec_total / 1_000_000
        if container.peak_kind == "counter" and container.memory_peak_bytes is not None:
            fields["peak_memory_mib"] = container.memory_peak_bytes / 2**20
    return fields


def record_resource_usage(observation: SandboxObservation, sandbox: AsyncSandbox) -> None:
    """Add the measured usage of ``sandbox`` to its observation, once ``stop()`` has taken the final sample."""
    if not isinstance(sandbox, AsyncSandbox):
        return
    if observation.sandbox_id is not None and observation.sandbox_id == sandbox.sandbox_id:
        for name, value in resource_usage_fields(sandbox).items():
            setattr(observation, name, value)


def lifecycle_observation(
    sandbox: AsyncSandbox | None, *, role: Literal["agent", "verifier", "environment"]
) -> dict[str, Any] | None:
    """The owner's observation of one created sandbox, or None when there is no AsyncSandbox to observe.

    ``outcome`` is ``sandbox_error`` only when every provider close failed; otherwise it stays ``unknown``: a
    confirmed teardown says nothing about the command or the grade. Cleanup failures stay listed after a retry,
    with the first as ``error_type``. Resource usage is the owner's measurement (see ``resource_usage_fields``);
    usage and remote lifetime are never estimated.
    """
    if not isinstance(sandbox, AsyncSandbox) or sandbox.sandbox_id is None:
        return None
    provider = getattr(getattr(sandbox, "_handle", None), "provider_name", None)
    closed = sandbox.teardown_confirmed
    errors = list(sandbox.cleanup_errors)
    return SandboxObservation(
        role=role,
        provider=provider if isinstance(provider, str) else None,
        sandbox_id=sandbox.sandbox_id,
        outcome="sandbox_error" if closed is False else "unknown",
        error_type=errors[0] if errors else None,
        observed_client_interval_s=sandbox.observed_client_interval_s,
        teardown_confirmed=closed,
        cleanup_errors=errors,
        **resource_usage_fields(sandbox),
    ).model_dump(mode="json")


def observe_lent_sandbox(sandbox: AsyncSandbox, *, role: Literal["agent", "verifier", "environment"]) -> None:
    """Record, from its owner, a stopped sandbox that another component borrowed through ``connect()``.

    A borrower measures nothing, so the server that created the sandbox binds its ``lifecycle_observation``,
    resource usage included, to the active verification. Outside a verification this does nothing. Best effort:
    an observation failure never replaces the stop or the grade.
    """
    try:
        observation = lifecycle_observation(sandbox, role=role)
        if observation is not None:
            observe_verifier_sandbox(observation)
    except Exception:
        LOGGER.warning("Lent sandbox observation unavailable", exc_info=True)
