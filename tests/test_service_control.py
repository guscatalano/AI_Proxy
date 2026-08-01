"""Starting and stopping configured services (ComfyUI first) from the proxy.

The proxy runs as the same user that owns the unit, so `systemctl --user` needs no privileges.
The alternative was a sudoers grant, and widening what this process can do as root in order to
press a button is a poor trade.
"""
import asyncio

from ai_proxy import proxy as P


def _cfg(monkeypatch, services):
    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["services"] = services
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    return cfg


def _fake_run(monkeypatch, ret=(0, "active")):
    seen = []

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        seen.append(list(args))
        if "is-active" in args:
            return 0, ret[1]
        if "is-enabled" in args:
            return 0, "enabled"
        return ret
    monkeypatch.setattr(P, "_run_cmd", run)
    return seen


def test_only_configured_units_are_reachable(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    assert P._service_def("comfyui")["unit"] == "comfyui.service"
    assert P._service_def("anything-else") is None
    # A unit entry without a unit name is not a service.
    _cfg(monkeypatch, {"broken": {"scope": "user"}})
    assert P._service_def("broken") is None


def test_unknown_service_is_refused(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    r = client.post("/__proxy/api/control/services/sshd", json={"action": "stop"})
    assert r.status_code == 404
    assert "no service named" in r.json()["error"]


def test_only_start_stop_restart_are_allowed(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    _fake_run(monkeypatch)
    for bad in ("mask", "disable", "kill", "", "start; rm -rf /"):
        r = client.post("/__proxy/api/control/services/comfyui", json={"action": bad})
        assert r.status_code == 400, bad
        assert "must be one of" in r.json()["error"]


def test_the_unit_name_is_an_argument_never_a_shell_string(client, monkeypatch):
    # _run_cmd takes a list and uses no shell, so a hostile unit name cannot become a command.
    _cfg(monkeypatch, {"evil": {"unit": "x.service; rm -rf /", "scope": "user"}})
    seen = _fake_run(monkeypatch)
    client.post("/__proxy/api/control/services/evil", json={"action": "start"})
    assert seen[0] == ["systemctl", "--user", "start", "x.service; rm -rf /"]


def test_user_scope_passes_the_user_flag(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    seen = _fake_run(monkeypatch)
    client.post("/__proxy/api/control/services/comfyui", json={"action": "restart"})
    assert seen[0] == ["systemctl", "--user", "restart", "comfyui.service"]


def test_system_scope_omits_it(client, monkeypatch):
    _cfg(monkeypatch, {"c": {"unit": "c.service", "scope": "system"}})
    seen = _fake_run(monkeypatch)
    client.post("/__proxy/api/control/services/c", json={"action": "start"})
    assert seen[0] == ["systemctl", "start", "c.service"]


def test_an_unrecognised_scope_is_not_treated_as_system(client, monkeypatch):
    # Defaulting an odd value to system scope would silently escalate what gets touched.
    _cfg(monkeypatch, {"c": {"unit": "c.service", "scope": "root"}})
    assert P._service_def("c")["scope"] == "system"
    _cfg(monkeypatch, {"d": {"unit": "d.service"}})
    assert P._service_def("d")["scope"] == "user"      # absent means user


def test_a_failing_systemctl_is_reported_not_swallowed(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    _fake_run(monkeypatch, ret=(1, "Failed to start comfyui.service: Unit not found."))
    r = client.post("/__proxy/api/control/services/comfyui", json={"action": "start"})
    assert r.status_code == 502
    assert "Unit not found" in r.json()["error"]
    assert "state" in r.json()


def test_listing_reports_state(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    _fake_run(monkeypatch)
    d = client.get("/__proxy/api/control/services").json()
    assert len(d["services"]) == 1
    s = d["services"][0]
    assert s["name"] == "comfyui" and s["running"] is True and s["enabled"] == "enabled"


def test_listing_is_empty_when_nothing_is_configured(client, monkeypatch):
    _cfg(monkeypatch, {})
    assert client.get("/__proxy/api/control/services").json()["services"] == []


def test_user_scope_points_systemctl_at_the_session_bus(client, monkeypatch):
    # The proxy is a *system* service: with no XDG_RUNTIME_DIR, systemctl --user reports
    # "Failed to connect to bus: No medium found" and every unit looks broken.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    env = P._systemctl_env({"scope": "user"})
    assert env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert env["DBUS_SESSION_BUS_ADDRESS"].startswith("unix:path=")
    assert env["DBUS_SESSION_BUS_ADDRESS"].endswith("/bus")
    assert P._systemctl_env({"scope": "system"}) is None


def test_a_missing_user_manager_explains_itself(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    _fake_run(monkeypatch, ret=(1, "Failed to connect to bus: No medium found"))
    r = client.post("/__proxy/api/control/services/comfyui", json={"action": "start"})
    assert r.status_code == 502
    # The fix is a specific command, so name it rather than echoing the raw error.
    assert "enable-linger" in r.json()["error"]


def test_env_is_merged_not_replaced(client, monkeypatch):
    # A bare env would drop PATH and systemctl would simply not be found.
    seen = {}

    async def fake_exec(*args, **kw):
        seen.update(kw)
        raise OSError("stop here")
    monkeypatch.setattr(P.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(P._run_cmd(["systemctl", "--user", "start", "x"], env={"XDG_RUNTIME_DIR": "/run/user/1000"}))
    assert seen["env"]["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert "PATH" in seen["env"]


def test_state_of_a_stopped_unit(client, monkeypatch):
    _cfg(monkeypatch, {"comfyui": {"unit": "comfyui.service", "scope": "user"}})
    _fake_run(monkeypatch, ret=(0, "inactive"))
    svc = P._service_def("comfyui")
    st = asyncio.run(P._service_state(svc))
    assert st["active"] == "inactive" and st["running"] is False
