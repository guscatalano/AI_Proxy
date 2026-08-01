"""Downloading a model into Ollama from the dashboard.

Tens of gigabytes over minutes to hours, so it cannot be a request that blocks until done —
it is a background job with progress, like the database wipe.
"""
import asyncio
import json
import time

from ai_proxy import proxy as P


def _stub_pull(monkeypatch, events, status=200):
    """Stand in for Ollama's streaming /api/pull."""
    class _Stream:
        status_code = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aread(self):
            return b'{"error":"model not found"}'

        async def aiter_lines(self):
            for e in events:
                yield e if isinstance(e, str) else json.dumps(e)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            return _Stream()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda *a, **kw: _Client())


def _reset():
    P._PULL_JOB.clear()
    P._PULL_JOB.update({"state": "idle"})
    if P._PULL_LOCK.locked():
        P._PULL_LOCK.release()


def test_progress_sums_across_layers(client, monkeypatch):
    # Ollama reports per layer digest, so reading one layer's numbers restarts at zero on every
    # new layer and the bar goes backwards.
    _reset()
    _stub_pull(monkeypatch, [
        {"status": "pulling manifest"},
        {"status": "pulling a", "digest": "sha256:a", "total": 100, "completed": 50},
        {"status": "pulling a", "digest": "sha256:a", "total": 100, "completed": 100},
        {"status": "pulling b", "digest": "sha256:b", "total": 300, "completed": 150},
        {"status": "success"},
    ])
    asyncio.run(P._ollama_pull("qwen3:4b"))
    st = P._pull_status()
    assert st["state"] == "done"
    assert st["total_bytes"] == 400          # both layers, not just the last
    assert st["completed_bytes"] == 250
    assert st["percent"] == 62.5


def test_progress_only_moves_forward(client, monkeypatch):
    _reset()
    _stub_pull(monkeypatch, [
        {"digest": "sha256:a", "total": 100, "completed": 100},
        {"digest": "sha256:b", "total": 100, "completed": 10},
    ])
    asyncio.run(P._ollama_pull("m"))
    assert P._pull_status()["completed_bytes"] == 110


def test_an_ollama_error_event_fails_the_job(client, monkeypatch):
    _reset()
    _stub_pull(monkeypatch, [{"status": "pulling"}, {"error": "file does not exist"}])
    asyncio.run(P._ollama_pull("nope:latest"))
    st = P._pull_status()
    assert st["state"] == "error"
    assert "file does not exist" in st["error"]


def test_an_http_error_fails_the_job(client, monkeypatch):
    _reset()
    _stub_pull(monkeypatch, [], status=404)
    asyncio.run(P._ollama_pull("nope"))
    assert P._pull_status()["state"] == "error"
    assert "404" in P._pull_status()["error"]


def test_malformed_lines_are_skipped_not_fatal(client, monkeypatch):
    _reset()
    _stub_pull(monkeypatch, ["", "not json", '{"digest":"sha256:a","total":10,"completed":10}',
                             '{"status":"success"}'])
    asyncio.run(P._ollama_pull("m"))
    assert P._pull_status()["state"] == "done"


def test_the_lock_is_released_however_it_ends(client, monkeypatch):
    # A pull that dies holding the lock would wedge the endpoint until restart.
    for events in ([{"status": "success"}], [{"error": "boom"}]):
        _reset()
        P._PULL_LOCK.acquire()
        _stub_pull(monkeypatch, events)
        asyncio.run(P._ollama_pull("m"))
        assert not P._PULL_LOCK.locked(), f"lock held after {events}"


def test_endpoint_requires_a_model(client):
    _reset()
    r = client.post("/__proxy/api/control/models/pull", json={})
    assert r.status_code == 400 and "required" in r.json()["error"]


def test_a_second_pull_is_refused_while_one_runs(client):
    _reset()
    P._PULL_LOCK.acquire()
    P._PULL_JOB.update({"state": "running", "model": "big:latest"})
    try:
        r = client.post("/__proxy/api/control/models/pull", json={"model": "other"})
        assert r.status_code == 409
        assert "big:latest" in r.json()["error"]
    finally:
        _reset()


def test_status_reports_percent_and_elapsed(client):
    _reset()
    P._PULL_JOB.update({"state": "running", "model": "m", "started": time.time() - 5,
                        "total_bytes": 1000, "completed_bytes": 250})
    st = P._pull_status()
    assert st["percent"] == 25.0
    assert st["elapsed_s"] >= 5
    # A pull that has not reported sizes yet has no percentage to show, rather than 0%.
    P._PULL_JOB.update({"total_bytes": 0, "completed_bytes": 0})
    assert P._pull_status()["percent"] is None
    _reset()


def test_a_finished_pull_goes_stale_instead_of_sitting_there_forever(client):
    # A failed pull used to stay on the System tab indefinitely, redrawn on every refresh,
    # with nothing to dismiss it.
    _reset()
    P._PULL_JOB.update({"state": "error", "model": "m", "error": "boom",
                        "started": time.time() - 300, "finished": time.time() - 5})
    assert P._pull_status()["stale"] is False        # still news
    P._PULL_JOB["finished"] = time.time() - (P._PULL_RESULT_TTL_S + 10)
    assert P._pull_status()["stale"] is True         # now clutter
    _reset()


def test_a_running_pull_is_never_stale(client):
    _reset()
    P._PULL_JOB.update({"state": "running", "model": "m", "started": time.time() - 100000})
    assert P._pull_status()["stale"] is False
    _reset()


def test_dismiss_clears_a_finished_pull(client):
    _reset()
    P._PULL_JOB.update({"state": "error", "model": "m", "error": "boom",
                        "finished": time.time()})
    r = client.post("/__proxy/api/control/models/pull/clear")
    assert r.status_code == 200
    assert P._pull_status()["state"] == "idle"


def test_dismiss_refuses_while_a_pull_is_running(client):
    # Otherwise a click hides live work and the download continues invisibly.
    _reset()
    P._PULL_JOB.update({"state": "running", "model": "big:latest"})
    try:
        r = client.post("/__proxy/api/control/models/pull/clear")
        assert r.status_code == 409
        assert P._PULL_JOB["state"] == "running"
    finally:
        _reset()


def test_cancel_with_nothing_running(client):
    _reset()
    P._PULL_TASK = None
    r = client.post("/__proxy/api/control/models/pull/cancel")
    assert r.status_code == 409 and "no pull is running" in r.json()["error"]
