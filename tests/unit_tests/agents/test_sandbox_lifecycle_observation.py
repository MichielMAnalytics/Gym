# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.sandbox_observation import (
    lifecycle_observation,
    observe_lent_sandbox,
    record_resource_usage,
    resource_usage_fields,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxUsageSample
from nemo_gym.trajectory.trajectory_record import SandboxObservation
from tests.unit_tests.sandbox.test_sandbox import FakeSandboxProvider
from tests.unit_tests.sandbox.test_sandbox_connect import FakeConnectableProvider


pytestmark = pytest.mark.sandbox


def _observed(sandbox: AsyncSandbox) -> SandboxObservation:
    return SandboxObservation.model_validate(lifecycle_observation(sandbox, role="verifier"))


async def test_the_interval_needs_both_the_provider_create_and_close_returns() -> None:
    sandbox = AsyncSandbox(FakeSandboxProvider())
    assert lifecycle_observation(sandbox, role="verifier") is None
    await sandbox.start(SandboxSpec(image="task"))
    started = _observed(sandbox)
    assert (started.observed_client_interval_s, started.teardown_confirmed, started.outcome) == (None, None, "unknown")
    await sandbox.stop()
    stopped = _observed(sandbox)
    assert stopped.observed_client_interval_s == sandbox.observed_client_interval_s >= 0
    assert (stopped.teardown_confirmed, stopped.cleanup_errors, stopped.outcome) == (True, [], "unknown")
    # A client-side interval is not remote lifetime or usage.
    assert (stopped.wall_time_s, stopped.cpu_time_s, stopped.peak_memory_mib) == (None, None, None)


@pytest.mark.parametrize("error", [RuntimeError("create failed"), asyncio.CancelledError()])
async def test_a_failed_or_cancelled_create_leaves_nothing_to_observe(error) -> None:
    provider = FakeSandboxProvider()
    provider.create = AsyncMock(side_effect=error)
    sandbox = AsyncSandbox(provider)
    with pytest.raises(type(error)):
        await sandbox.start(SandboxSpec(image="task"))
    assert lifecycle_observation(sandbox, role="verifier") is None


async def test_a_connected_handle_has_no_observed_create() -> None:
    owner = AsyncSandbox(FakeConnectableProvider())
    await owner.start(SandboxSpec(image="task"))
    borrowed = await AsyncSandbox.connect(await owner.serialize(), provider=FakeConnectableProvider())
    assert (_observed(borrowed).observed_client_interval_s, _observed(borrowed).teardown_confirmed) == (None, None)
    await borrowed.stop()
    # Teardown is confirmed, but this instance never saw the create, so no interval is reported.
    assert (_observed(borrowed).observed_client_interval_s, _observed(borrowed).teardown_confirmed) == (None, True)


@pytest.mark.parametrize("error", [TimeoutError(), asyncio.CancelledError()])
async def test_a_successful_retry_keeps_the_earlier_close_failure(error) -> None:
    provider = FakeSandboxProvider()
    provider.close = AsyncMock(side_effect=[error, None])
    sandbox = AsyncSandbox(provider)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(type(error)):
        await sandbox.stop()
    failed = _observed(sandbox)
    name = type(error).__name__
    assert (failed.outcome, failed.teardown_confirmed, failed.cleanup_errors) == (
        "sandbox_error",
        False,
        [f"close:{name}"],
    )
    assert failed.observed_client_interval_s is None
    await sandbox.stop()
    retried = _observed(sandbox)
    assert (retried.teardown_confirmed, retried.cleanup_errors, retried.error_type) == (
        True,
        [f"close:{name}"],
        f"close:{name}",
    )
    assert retried.outcome == "unknown" and provider.close.await_count == 2


async def test_a_client_release_failure_after_confirmed_teardown_is_kept_separately() -> None:
    provider = FakeSandboxProvider()
    provider.aclose = AsyncMock(side_effect=[OSError("client"), None])
    sandbox = AsyncSandbox(provider, owns_provider=True)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(OSError):
        await sandbox.stop()
    observed = _observed(sandbox)
    assert (observed.teardown_confirmed, observed.cleanup_errors, observed.outcome) == (
        True,
        ["aclose:OSError"],
        "unknown",
    )
    assert observed.observed_client_interval_s is not None


class _MeasuredProvider(FakeSandboxProvider):
    """Reports the given usage samples for each read."""

    def __init__(self, *samples: SandboxUsageSample) -> None:
        super().__init__()
        self.samples = samples

    async def usage(self, handle) -> tuple[SandboxUsageSample, ...]:
        return self.samples


def _usage(source: str, scope: str, kind: str, *, cpu: int, peak: int) -> SandboxUsageSample:
    return SandboxUsageSample(
        source=source,
        scope=scope,
        sampled_at=1.0,
        cpu_usage_usec_total=cpu,
        cpu_kind=kind,
        memory_peak_bytes=peak,
        peak_kind=kind,
    )


async def test_a_container_counter_fills_the_scalar_usage_fields() -> None:
    sandbox = AsyncSandbox(
        _MeasuredProvider(_usage("cgroup_v2", "container", "counter", cpu=2_500_000, peak=3 * 2**20))
    )
    await sandbox.start(SandboxSpec(image="task"))
    await sandbox.stop()

    observed = _observed(sandbox)
    assert (observed.cpu_time_s, observed.peak_memory_mib, observed.resource_usage_source) == (
        2.5,
        3.0,
        "cgroup_v2:container",
    )
    (measurement,) = observed.resource_usage
    assert (measurement.scope, measurement.cpu_kind, measurement.peak_kind, measurement.sample_count) == (
        "container",
        "counter",
        "counter",
        2,
    )


async def test_lower_bounds_and_shared_totals_never_become_the_sandbox_scalars() -> None:
    sandbox = AsyncSandbox(
        _MeasuredProvider(
            _usage("procfs_pidns", "process_tree", "sampled_lower_bound", cpu=1_000_000, peak=2**20),
            _usage("cgroup_v2", "shared_step", "counter", cpu=90_000_000, peak=2**30),
        )
    )
    await sandbox.start(SandboxSpec(image="task"))
    await sandbox.stop()

    observed = _observed(sandbox)
    assert (observed.cpu_time_s, observed.peak_memory_mib, observed.resource_usage_source) == (None, None, None)
    assert {(item.source, item.scope, item.peak_kind) for item in observed.resource_usage} == {
        ("procfs_pidns", "process_tree", "sampled_lower_bound"),
        ("cgroup_v2", "shared_step", "counter"),
    }


async def test_usage_is_recorded_on_the_creating_sandbox_observation_only() -> None:
    sandbox = AsyncSandbox(_MeasuredProvider(_usage("cgroup_v2", "container", "counter", cpu=1_000_000, peak=1)))
    await sandbox.start(SandboxSpec(image="task"))
    await sandbox.stop()

    stand_in = SandboxObservation(role="agent", sandbox_id=sandbox.sandbox_id)
    record_resource_usage(stand_in, object())
    assert stand_in.resource_usage is None

    other = SandboxObservation(role="agent", sandbox_id="another-sandbox")
    record_resource_usage(other, sandbox)
    assert other.resource_usage is None and other.cpu_time_s is None

    own = SandboxObservation(role="agent", sandbox_id=sandbox.sandbox_id)
    record_resource_usage(own, sandbox)
    assert own.cpu_time_s == 1.0 and len(own.resource_usage) == 1

    unmeasured = AsyncSandbox(FakeSandboxProvider())
    await unmeasured.start(SandboxSpec(image="task"))
    await unmeasured.stop()
    assert "resource_usage" not in resource_usage_fields(unmeasured)


async def test_a_lent_sandbox_reaches_the_exported_trajectory_once_through_its_owners_verification() -> None:
    from nemo_gym.artifacts.rollout_rows import _rollout_for_export
    from nemo_gym.core.episode_context import rollout_context
    from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
    from nemo_gym.server_runtime.verifier_observability import observe_verifier
    from nemo_gym.trajectory.trajectory_record import NG_AGENT_OBSERVATIONS_KEY, AgentObservationBundle

    sandbox = AsyncSandbox(_MeasuredProvider(_usage("cgroup_v2", "container", "counter", cpu=3_000_000, peak=1)))
    await sandbox.start(SandboxSpec(image="task"))
    observe_lent_sandbox(sandbox, role="agent")  # outside a verification: nothing to bind to, and no error

    async def verify() -> dict:
        # As a resources server does: it stops the task sandbox it lent while grading, then records it.
        await sandbox.stop()
        observe_lent_sandbox(sandbox, role="agent")
        observe_lent_sandbox(object(), role="agent")  # not an AsyncSandbox: no observation
        return {"response": {"id": "r", "output": []}, "reward": 1.0}

    with rollout_context("0-0"):
        result = await observe_verifier(verify, enabled=True)()
    # The borrowing agent's own record names the sandbox but measures nothing.
    borrowed = SandboxObservation(role="agent", sandbox_id=sandbox.sandbox_id)
    result[NG_AGENT_OBSERVATIONS_KEY] = AgentObservationBundle(source="test", records=[borrowed]).model_dump(
        mode="json"
    )
    trajectory = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, result)
    exported = _rollout_for_export({"ng_trajectory": trajectory.model_dump(mode="json")})["ng_trajectory"]

    (owner,) = exported["verifier_invocations"][0]["sandbox_observations"]
    assert (owner["role"], owner["sandbox_id"], owner["cpu_time_s"]) == ("agent", sandbox.sandbox_id, 3.0)
    assert [item["scope"] for item in owner["resource_usage"]] == ["container"]
    # The borrower's record in the saved agent observations names the sandbox without usage, so the sandbox is
    # measured once, by its owner.
    (agent_record,) = AgentObservationBundle.model_validate(result[NG_AGENT_OBSERVATIONS_KEY]).records
    assert (agent_record.sandbox_id, agent_record.resource_usage) == (sandbox.sandbox_id, None)


async def test_a_failing_lent_sandbox_observation_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.agents.utils import sandbox_observation

    def broken(sandbox, *, role):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(sandbox_observation, "lifecycle_observation", broken)
    sandbox_observation.observe_lent_sandbox(AsyncSandbox(FakeSandboxProvider()), role="agent")
