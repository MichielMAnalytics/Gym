# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/verify reports the eval sandbox's observed lifecycle without changing the grade it saved."""

from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle, SandboxSpec, SandboxUsageSample
from nemo_gym.trajectory import NG_VERIFIER_INVOCATIONS_KEY, SandboxObservation
from resources_servers.swebench_pro.tests.test_app import make_server, request_body
from resources_servers.swebench_pro.verification import VerificationResult


class StubProvider:
    """Test stub, not a real provider: create and close return or raise as configured; nothing runs."""

    name = "stub"

    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.closed = 0

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id="stub-1", provider_name=self.name, raw=None)

    async def close(self, handle: SandboxHandle) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error

    async def aclose(self) -> None:
        return None


def _verify(monkeypatch: MonkeyPatch, *, provider: StubProvider, result=None, error=None) -> dict:
    server = make_server(golden=True, inconclusive_verification_retries=0)

    async def create(body, *, files):
        return await AsyncSandbox(provider).start(SandboxSpec(image="task"))

    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.run_verification", AsyncMock(return_value=result, side_effect=error)
    )
    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())
    assert response.status_code == 200
    return response.json()


def _result(resolved: bool) -> VerificationResult:
    return VerificationResult(
        completed=True, resolved=resolved, patch_applied=True, test_results={"tests": []}, test_output="ok"
    )


@pytest.mark.parametrize("resolved,reward", [(True, 1.0), (False, 0.0)])
def test_a_graded_verify_reports_the_stopped_sandbox_and_keeps_its_reward(monkeypatch, resolved, reward) -> None:
    provider = StubProvider()
    body = _verify(monkeypatch, provider=provider, result=_result(resolved))
    observation = SandboxObservation.model_validate(body["verifier_sandbox_observations"][-1])
    assert body["reward"] == reward and body["evaluation_completed"] is True and provider.closed == 1
    assert (observation.role, observation.provider, observation.sandbox_id) == ("verifier", "stub", "stub-1")
    assert observation.observed_client_interval_s >= 0 and observation.teardown_confirmed is True
    # A confirmed teardown is not evidence about the tests or the grade, and no lifetime or usage is estimated.
    assert observation.outcome == "unknown" and observation.exit_code is None and observation.error_type is None
    assert (observation.wall_time_s, observation.cpu_time_s, observation.peak_memory_mib) == (None,) * 3


def test_a_failed_stop_after_the_answer_keeps_the_reward_and_reports_a_sandbox_error(monkeypatch) -> None:
    body = _verify(monkeypatch, provider=StubProvider(RuntimeError("close failed")), result=_result(True))
    observation = SandboxObservation.model_validate(body["verifier_sandbox_observations"][-1])
    assert body["reward"] == 1.0 and body["evaluation_completed"] is True and body["error"] is None
    assert (observation.outcome, observation.teardown_confirmed, observation.cleanup_errors) == (
        "sandbox_error",
        False,
        ["close:RuntimeError"],
    )
    assert observation.observed_client_interval_s is None


def test_a_verification_failure_keeps_its_error_and_still_reports_the_sandbox(monkeypatch) -> None:
    body = _verify(monkeypatch, provider=StubProvider(), error=RuntimeError("tests crashed"))
    observation = SandboxObservation.model_validate(body["verifier_sandbox_observations"][-1])
    assert body["reward"] == 0.0 and body["evaluation_completed"] is False
    assert "tests crashed" in body["error"]
    assert observation.outcome == "unknown" and observation.teardown_confirmed is True


def test_a_sandbox_that_was_never_created_is_not_observed(monkeypatch) -> None:
    server = make_server(golden=True, inconclusive_verification_retries=0)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=RuntimeError("sandbox unavailable")))
    body = TestClient(server.setup_webserver()).post("/verify", json=request_body()).json()
    assert body["verifier_sandbox_observation"] is None
    assert body["reward"] == 0.0 and "sandbox unavailable" in body["error"]


def test_cleanup_failure_does_not_replace_the_original_verification_error(monkeypatch) -> None:
    body = _verify(
        monkeypatch,
        provider=StubProvider(TimeoutError("close acknowledgement unavailable")),
        error=ValueError("original verification error"),
    )
    observation = SandboxObservation.model_validate(body["verifier_sandbox_observations"][-1])
    assert body["evaluation_completed"] is False and body["reward"] == 0.0
    assert "original verification error" in body["error"]
    assert "close acknowledgement unavailable" not in body["error"]
    assert observation.teardown_confirmed is False
    assert observation.cleanup_errors == ["close:TimeoutError"]


def test_observation_failure_cannot_replace_an_already_calculated_grade(monkeypatch) -> None:
    def broken_observation(*args, **kwargs):
        raise ValueError("controlled observation failure")

    monkeypatch.setattr("resources_servers.swebench_pro.app.lifecycle_observation", broken_observation)
    body = _verify(monkeypatch, provider=StubProvider(), result=_result(True))
    assert body["reward"] == 1.0 and body["evaluation_completed"] is True and body["error"] is None
    assert body["verifier_sandbox_observation"] is None
    assert body["verifier_sandbox_observations"] == [None]


async def test_retry_success_retains_the_earlier_cleanup_failure(monkeypatch) -> None:
    server = make_server(golden=True, inconclusive_verification_retries=1)
    providers = [StubProvider(TimeoutError("first teardown unconfirmed")), StubProvider()]

    async def create(body, *, files):
        return await AsyncSandbox(providers.pop(0)).start(SandboxSpec(image="task"))

    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.run_verification",
        AsyncMock(
            side_effect=[
                VerificationResult(completed=False, resolved=False, patch_applied=True, test_results=None),
                _result(True),
            ]
        ),
    )
    body = TestClient(server.setup_webserver()).post("/verify", json=request_body()).json()
    assert body["reward"] == 1.0 and body["verification_attempts"] == 2
    first, last = body["verifier_sandbox_observations"]
    assert first["cleanup_errors"] == ["close:TimeoutError"] and first["teardown_confirmed"] is False
    assert last["cleanup_errors"] == [] and last["teardown_confirmed"] is True
    legacy = body["verifier_sandbox_observation"]
    assert legacy == {
        key: value
        for key, value in last.items()
        if key not in {"observed_client_interval_s", "teardown_confirmed", "cleanup_errors", "resource_usage"}
    }
    assert not {"observed_client_interval_s", "teardown_confirmed", "cleanup_errors", "resource_usage"} & legacy.keys()


async def test_the_task_sandbox_the_agent_borrowed_is_recorded_by_this_server_when_it_stops_it(monkeypatch) -> None:
    server = make_server(golden=False)
    sandbox = await AsyncSandbox(StubProvider()).start(SandboxSpec(image="task"))
    server._session_id_to_sandbox["s"] = sandbox
    recorded = []
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.observe_lent_sandbox",
        lambda stopped, *, role: recorded.append((stopped, role, stopped.teardown_confirmed)),
    )

    await server._stop_session_sandbox("s")

    # Recorded after the stop, so the observation carries the final usage sample and the confirmed teardown.
    assert recorded == [(sandbox, "agent", True)] and "s" not in server._session_id_to_sandbox


class MeasuredTaskProvider(StubProvider):
    """A task sandbox that reports one container-scope usage sample and serves the patch capture."""

    name = "measured"

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id="task-1", provider_name=self.name, raw=None)

    async def exec(self, handle, command, **kwargs) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download_file(self, handle, source_path, target_path) -> None:
        target_path.write_text("")

    async def usage(self, handle) -> tuple[SandboxUsageSample, ...]:
        sample = SandboxUsageSample(
            source="cgroup_v2",
            scope="container",
            sampled_at=1.0,
            cpu_usage_usec_total=3_000_000,
            cpu_kind="counter",
            memory_peak_bytes=2**30,
            peak_kind="counter",
        )
        return (sample,)


def test_the_lent_task_sandbox_usage_reaches_the_persisted_verifier_invocation(monkeypatch) -> None:
    server = make_server(golden=False, inconclusive_verification_retries=0)
    server.server_client.global_config_dict = {"observability_enabled": True}

    async def create(body, *, files):
        return await AsyncSandbox(StubProvider()).start(SandboxSpec(image="eval"))

    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", AsyncMock(return_value=_result(True)))
    app = server.setup_webserver()

    @app.post("/test_seed")
    async def seed(request: Request) -> dict:
        task = await AsyncSandbox(MeasuredTaskProvider()).start(SandboxSpec(image="task"))
        server._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = task
        return {}

    client = TestClient(app)
    assert client.post("/test_seed").status_code == 200
    body = client.post("/ng-rollout/rollout-1/verify", json=request_body()).json()

    # The verify reply carries the verifier invocation that rollout collection writes into the trajectory.
    (invocation,) = body[NG_VERIFIER_INVOCATIONS_KEY]
    observations = [SandboxObservation.model_validate(item) for item in invocation["sandbox_observations"]]
    (agent,) = [item for item in observations if item.role == "agent"]
    assert (agent.sandbox_id, agent.teardown_confirmed, agent.cpu_time_s, agent.peak_memory_mib) == (
        "task-1",
        True,
        3.0,
        1024.0,
    )
    assert [measurement.scope for measurement in agent.resource_usage] == ["container"]
    assert [item.role for item in observations] == ["agent"]
    # The verifier's own sandbox stays in the reply's verifier observations; the task sandbox is not repeated there.
    assert [item["sandbox_id"] for item in body["verifier_sandbox_observations"]] == ["stub-1"]
