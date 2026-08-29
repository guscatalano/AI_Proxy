"""Distinguishing "vLLM is coming up" from "vLLM is gone".

Loading 42 GiB of weights takes ~9 minutes, during which /v1/models refuses connections exactly
like a container that isn't running. The dashboard showed "offline" for both, so a normal restart
looked identical to a failure.
"""
import asyncio

import pytest

import ai_proxy.proxy as P


class _FakeResp:
    def __init__(self, status):
        self.status_code = status
        self.text = ""          # /metrics is scraped as Prometheus text after /v1/models

    def json(self):
        return {"data": []}


class _FakeClient:
    """Only /v1/models matters here; everything else is unreachable."""
    def __init__(self, status=None, raise_exc=None):
        self.status, self.raise_exc = status, raise_exc

    async def get(self, url, *a, **kw):
        if self.raise_exc:
            raise self.raise_exc
        return _FakeResp(self.status)


@pytest.fixture
def _docker(monkeypatch):
    """Stub the docker calls so no test needs a real daemon."""
    calls = {"logs": "", "running": "true", "started": "2026-07-27T06:28:10.0Z",
             "container": "qwen-vllm", "inspect_code": 0}

    async def fake_container(*a, **k):
        return calls["container"]

    async def fake_run(args, timeout=120.0, max_chars=800, keep_tail=False):
        if "inspect" in args:
            return calls["inspect_code"], f"{calls['running']}\t{calls['started']}"
        if "logs" in args:
            calls["log_args"] = {"max_chars": max_chars, "keep_tail": keep_tail}
            text = calls["logs"]
            return 0, (text[-max_chars:] if keep_tail else text[:max_chars])
        return 1, ""

    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_vllm_container", fake_container)
    monkeypatch.setattr(P, "_run_cmd", fake_run)
    return calls


def test_running_container_that_is_not_answering_is_loading(_docker):
    _docker["logs"] = "Loading safetensors checkpoint shards:  70% Completed | 7/10 [03:53<01:45]"
    s = asyncio.run(P._vllm_boot_state())
    assert s["state"] == "loading"
    assert s["phase"] == "loading weights — shard 7 of 10 (70%)"
    assert s["container"] == "qwen-vllm"
    assert s["started_at"] == "2026-07-27T06:28:10.0Z"


def test_later_boot_phases_are_named(_docker):
    for line, expected in (
        ("Capturing CUDA graph shapes: 40%", "capturing CUDA graphs"),
        ("Available KV cache memory: 49.1 GiB", "allocating the KV cache"),
        ("Starting to load model saricles/Qwen3-Coder-Next", "starting to load the model"),
        ("Initializing a V1 LLM engine (v0.25.1)", "initialising the engine"),
    ):
        _docker["logs"] = line
        assert (asyncio.run(P._vllm_boot_state()))["phase"] == expected


def test_the_newest_phase_wins(_docker):
    # The log holds the whole boot; the last matching line is where it actually is.
    _docker["logs"] = "\n".join([
        "Starting to load model saricles/Qwen3-Coder-Next",
        "Loading safetensors checkpoint shards: 100% Completed | 10/10",
        "Capturing CUDA graph shapes: 12%",
    ])
    assert (asyncio.run(P._vllm_boot_state()))["phase"] == "capturing CUDA graphs"


def test_progress_survives_the_output_cap(_docker):
    # _run_cmd caps output at 800 chars from the *head* by default, which on a boot log returns
    # the oldest lines and drops the progress entirely — exactly what happened in production.
    _docker["logs"] = ("(APIServer) INFO some startup chatter line\n" * 60
                       + "Loading safetensors checkpoint shards:  50% Completed | 5/10")
    s = asyncio.run(P._vllm_boot_state())
    assert s["phase"] == "loading weights — shard 5 of 10 (50%)"
    assert _docker["log_args"]["keep_tail"] is True
    assert _docker["log_args"]["max_chars"] > 800


def test_stopped_container_is_not_reported_as_loading(_docker):
    _docker["running"] = "false"
    s = asyncio.run(P._vllm_boot_state())
    assert s["state"] == "stopped" and s["container"] == "qwen-vllm"
    assert "phase" not in s


def test_no_container_at_all(_docker, monkeypatch):
    async def none_container(*a, **k):
        return None
    monkeypatch.setattr(P, "_vllm_container", none_container)
    assert asyncio.run(P._vllm_boot_state()) == {"state": "absent"}


def test_no_docker_is_not_an_error(monkeypatch):
    monkeypatch.setattr(P, "_docker_bin", lambda: None)
    assert (asyncio.run(P._vllm_boot_state()))["state"] == "absent"


def test_loading_without_a_recognisable_phase_still_says_loading(_docker):
    _docker["logs"] = "some line nothing matches"
    s = asyncio.run(P._vllm_boot_state())
    assert s["state"] == "loading" and "phase" not in s


def test_snapshot_reports_ready_when_it_answers(monkeypatch, _docker):
    s = asyncio.run(P._vllm_snapshot(_FakeClient(status=200)))
    assert s["state"] == "ready" and s["reachable"] is True


def test_snapshot_falls_through_to_boot_state_on_refusal(_docker):
    import httpx
    _docker["logs"] = "Loading safetensors checkpoint shards:  30% Completed | 3/10"
    s = asyncio.run(P._vllm_snapshot(_FakeClient(raise_exc=httpx.ConnectError("refused"))))
    assert s["reachable"] is False
    assert s["state"] == "loading"
    assert "shard 3 of 10" in s["phase"]


def test_snapshot_handles_a_non_200_without_an_exception(_docker):
    # A 503 answers the socket but serves nothing; it must still be classified, not left blank.
    s = asyncio.run(P._vllm_snapshot(_FakeClient(status=503)))
    assert s["reachable"] is False and s["state"] in ("loading", "stopped", "absent")
