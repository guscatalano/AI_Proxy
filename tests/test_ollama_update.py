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
    ran = {}

    async def fake_run(cmd, timeout, max_chars=2000, keep_tail=False, env=None):
        ran["cmd"] = cmd
        return 0, "downloaded and installed"

    monkeypatch.setattr(P, "_ollama_update_cmd",
                        lambda: ["sudo", "-n", "/usr/local/sbin/ollama-update"])
    monkeypatch.setattr(P, "_run_cmd", fake_run)
    r = client.post("/__proxy/api/control/ollama/update")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True
    assert ran["cmd"][0] == "sudo"
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
