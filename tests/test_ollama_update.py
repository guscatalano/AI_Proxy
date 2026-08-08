"""Updating Ollama from the proxy: guarded, configured explicitly, honest about versions."""
import json
import time

from ai_proxy import proxy as P


def test_update_refuses_without_a_configured_updater(client, monkeypatch):
    monkeypatch.setattr(P, "_ollama_update_cmd", lambda: None)
    r = client.post("/__proxy/api/control/ollama/update")
    assert r.status_code == 501
    assert "ollama_update_cmd" in r.json()["error"]


def test_update_refuses_while_a_bench_runs(client, monkeypatch):
    """The installer restarts the service; mid-bench that turns every in-flight cell into a
    page of 502s attributed to whatever model was being measured."""
    monkeypatch.setattr(P, "_ollama_update_cmd", lambda: ["true"])
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                 "VALUES ('b_x', ?, 'm', '{}', 'running')", (time.time(),))
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/control/ollama/update")
    assert r.status_code == 409
    assert "benchmark" in r.json()["error"]


def test_update_runs_the_updater_and_reports_versions(client, monkeypatch):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    # Every invocation, not just the last: the background metrics collector shells out to
    # docker while the test runs, so a single-slot recorder is a race. It only ever fired
    # on Linux CI (where docker exists) and never locally, which is the worst way to find
    # out. Assert the configured updater WAS invoked, not that it was invoked last.
    ran = []

    async def fake_run(cmd, timeout, max_chars=2000, keep_tail=False, env=None):
        ran.append(list(cmd))
        return 0, "downloaded and installed"

    monkeypatch.setattr(P, "_ollama_update_cmd",
                        lambda: ["sudo", "-n", "/usr/local/sbin/ollama-update"])
    monkeypatch.setattr(P, "_run_cmd", fake_run)
    r = client.post("/__proxy/api/control/ollama/update")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True
    assert ["sudo", "-n", "/usr/local/sbin/ollama-update"] in ran, ran
    assert "installed" in d["output"]


def test_update_surfaces_updater_failure(client, monkeypatch):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()

    async def fake_run(cmd, timeout, max_chars=2000, keep_tail=False, env=None):
        return 1, "sudo: a password is required"

    monkeypatch.setattr(P, "_ollama_update_cmd", lambda: ["sudo", "-n", "x"])
    monkeypatch.setattr(P, "_run_cmd", fake_run)
    r = client.post("/__proxy/api/control/ollama/update")
    assert r.status_code == 502
    assert "password is required" in r.json()["error"]


def test_update_cmd_rejects_non_list_config(client, monkeypatch):
    monkeypatch.setattr(P, "load_rules_config",
                        lambda: {"model_control": {"ollama_update_cmd": "curl | sh"}})
    assert P._ollama_update_cmd() is None, "a shell string is not a command list"
    monkeypatch.setattr(P, "load_rules_config",
                        lambda: {"model_control": {"ollama_update_cmd": ["sudo", "-n", "/x"]}})
    assert P._ollama_update_cmd() == ["sudo", "-n", "/x"]


# ---- per-model default parameters (num_ctx etc.) --------------------------------------------

def test_model_params_validates_input(client):
    r = client.post("/__proxy/api/control/models/ollama-params", json={})
    assert r.status_code == 400 and "model" in r.json()["error"]
    r = client.post("/__proxy/api/control/models/ollama-params",
                    json={"model": "m", "params": {"num_gpu_layers": 5}})
    assert r.status_code == 400 and "unsupported" in r.json()["error"]
    r = client.post("/__proxy/api/control/models/ollama-params",
                    json={"model": "m", "num_ctx": 12})
    assert r.status_code == 400 and "num_ctx" in r.json()["error"]


def test_model_params_applies_through_ollama_create(client, monkeypatch):
    import asyncio
    applied = {}

    async def fake_apply(model, params):
        applied.update(model=model, params=params)
        return True, "ok"

    async def fake_kv(*a, **k):
        return None

    monkeypatch.setattr(P, "_ollama_apply_params", fake_apply)
    # keep the KV probe out of the network by making the estimate unavailable
    monkeypatch.setattr(P, "_bench_ollama_kv_mb", lambda info, tokens: None)
    r = client.post("/__proxy/api/control/models/ollama-params",
                    json={"model": "gemma4:26b", "num_ctx": 65536,
                          "params": {"temperature": 0.2}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True
    assert applied["model"] == "gemma4:26b"
    assert applied["params"]["num_ctx"] == 65536
    assert applied["params"]["temperature"] == 0.2
    assert "next load" in d["note"]


def test_model_params_refuses_an_oom_context_without_force(client, monkeypatch):
    async def fake_apply(model, params):
        raise AssertionError("must not apply an OOM context without force")

    class _R:
        status_code = 200
        def __init__(self, p): self._p = p
        def json(self): return self._p

    class _C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None):
            return _R({"model_info": {"general.architecture": "mistral3",
                                      "mistral3.block_count": 88,
                                      "mistral3.attention.head_count_kv": 8,
                                      "mistral3.attention.head_count": 96,
                                      "mistral3.attention.key_length": 128,
                                      "mistral3.attention.value_length": 128}})
        async def get(self, url):
            return _R({"models": [{"name": "big:latest", "size": 70 * 1024**3}]})

    monkeypatch.setattr(P, "_ollama_apply_params", fake_apply)
    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": 124610})
    r = client.post("/__proxy/api/control/models/ollama-params",
                    json={"model": "big:latest", "num_ctx": 262144})
    assert r.status_code == 400
    assert "OOM" in r.json()["error"] and "force:true" in r.json()["error"]
