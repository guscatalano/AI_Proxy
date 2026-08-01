"""Relaunching llama.cpp at a different context size.

llama.cpp fixes its context at launch, so a context sweep means rewriting the unit's command
line and restarting — the bench cannot do it per cell the way it can with Ollama.
"""
import asyncio

from ai_proxy import proxy as P


def _cfg(monkeypatch, tmp_path, **over):
    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["llamacpp"] = {"unit": "llamacpp.service", "scope": "user",
                      "binary": "/opt/llama-server", "model": "/m/x-00001-of-00003.gguf",
                      "host": "0.0.0.0", "port": 8080, "ngl": 999, "extra_args": [], **over}
    mc["services"] = {"llamacpp": {"unit": "llamacpp.service", "scope": "user"}}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P.Path, "home", classmethod(lambda cls: tmp_path))
    return cfg


def test_override_replaces_rather_than_appends(monkeypatch, tmp_path, client):
    # Without the bare `ExecStart=`, systemd appends and launches the server twice.
    _cfg(monkeypatch, tmp_path)
    ok, detail = P._write_llamacpp_override(131072, 4)
    assert ok, detail
    text = (tmp_path / ".config/systemd/user/llamacpp.service.d/override.conf").read_text()
    assert text.count("ExecStart=") == 2
    assert "\nExecStart=\nExecStart=/opt/llama-server" in text
    assert "--ctx-size 131072" in text and "--parallel 4" in text


def test_extra_args_are_carried_through(monkeypatch, tmp_path, client):
    _cfg(monkeypatch, tmp_path, extra_args=["--flash-attn", "--mlock"])
    P._write_llamacpp_override(65536, 2)
    text = (tmp_path / ".config/systemd/user/llamacpp.service.d/override.conf").read_text()
    assert "--flash-attn --mlock" in text


def test_refuses_without_a_binary_and_model(monkeypatch, tmp_path, client):
    # Guessing either would produce a unit that fails to start with a confusing message.
    _cfg(monkeypatch, tmp_path, binary="", model="")
    ok, msg = P._write_llamacpp_override(65536, 4)
    assert ok is False and "binary" in msg and "model" in msg


def test_load_endpoint_restarts_and_waits(monkeypatch, tmp_path, client):
    _cfg(monkeypatch, tmp_path)
    calls = []

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        calls.append(list(args))
        return 0, ""

    async def snap(c):
        return {"reachable": True, "n_ctx": 16384, "parallel": 4}

    async def ready(t):
        return True

    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "_llamacpp_snapshot", snap)
    monkeypatch.setattr(P, "_llamacpp_ready", ready)
    r = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "llamacpp", "context_length": 131072})
    d = r.json()
    assert d["ok"] is True and d["ready"] is True and d["context_length"] == 131072
    # daemon-reload takes no unit; passing one is "Too many arguments."
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "restart", "llamacpp.service"] in calls


def test_parallel_is_preserved_when_only_context_changes(monkeypatch, tmp_path, client):
    # Changing one must not silently reset the other to a default nobody asked for.
    _cfg(monkeypatch, tmp_path)

    async def run(args, timeout=120.0, **kw):
        return 0, ""

    async def snap(c):
        return {"reachable": True, "n_ctx": 16384, "parallel": 4}

    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "_llamacpp_snapshot", snap)
    monkeypatch.setattr(P, "_llamacpp_ready", lambda t: asyncio.sleep(0, result=True))
    client.post("/__proxy/api/control/models/load",
                json={"upstream": "llamacpp", "context_length": 262144})
    text = (tmp_path / ".config/systemd/user/llamacpp.service.d/override.conf").read_text()
    assert "--parallel 4" in text


def test_a_server_that_never_answers_is_reported_not_claimed(monkeypatch, tmp_path, client):
    _cfg(monkeypatch, tmp_path)

    async def run(args, timeout=120.0, **kw):
        return 0, ""

    async def snap(c):
        return {"reachable": False}

    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "_llamacpp_snapshot", snap)
    monkeypatch.setattr(P, "_llamacpp_ready", lambda t: asyncio.sleep(0, result=False))
    d = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "llamacpp", "context_length": 65536, "parallel": 1}).json()
    assert d["ok"] is False and d["ready"] is False
    assert "did not answer" in d["note"]
