"""Tool aliasing: models call tools by the name they expect, not the one that was declared."""
import json
import ai_proxy.proxy as p

REQ = {"tools": [{"type": "function", "function": {"name": "terminal",
                                                   "parameters": {"type": "object",
                                                                  "properties": {"cmd": {"type": "string"}}}}}]}


def _resp(name, args):
    return {"choices": [{"message": {"tool_calls": [
        {"function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def _cfg(monkeypatch, **over):
    base = {"enabled": True, "action": "rewrite", "map": {"run": "terminal"},
            "require_declared": True}
    base.update(over)
    monkeypatch.setattr(p, "load_rules_config", lambda: {"tool_aliases": base})


def test_renames_a_known_wrong_tool_name(monkeypatch):
    """The case this exists for: the model calls "run" against a client offering "terminal"."""
    _cfg(monkeypatch)
    r = _resp("run", {"cmd": "ls"})
    changes = p._apply_tool_aliases(r, REQ)
    assert changes == [{"from": "run", "to": "terminal", "renamed_args": {}}]
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "terminal"


def test_renames_arguments_too(monkeypatch):
    """run(command=…) against terminal(cmd=…) — renaming the tool alone would still fail."""
    _cfg(monkeypatch, map={"run": {"to": "terminal", "args": {"command": "cmd"}}})
    r = _resp("run", {"command": "ls -la"})
    changes = p._apply_tool_aliases(r, REQ)
    assert changes[0]["renamed_args"] == {"command": "cmd"}
    args = json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"cmd": "ls -la"}


def test_never_touches_a_call_that_would_have_worked(monkeypatch):
    """A declared name is correct by definition, even if it appears in the map."""
    _cfg(monkeypatch, map={"terminal": "something_else"})
    r = _resp("terminal", {"cmd": "ls"})
    assert p._apply_tool_aliases(r, REQ) == []
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "terminal"


def test_refuses_to_invent_an_undeclared_tool(monkeypatch):
    """A typo in the map would otherwise rename onto a tool the client has never heard of —
    worse than the original wrong name, because it looks deliberate."""
    _cfg(monkeypatch, map={"run": "shell"})       # 'shell' is not declared
    r = _resp("run", {"cmd": "ls"})
    assert p._apply_tool_aliases(r, REQ) == []
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "run"


def test_audit_action_reports_without_changing(monkeypatch):
    _cfg(monkeypatch, action="audit")
    r = _resp("run", {"cmd": "ls"})
    changes = p._apply_tool_aliases(r, REQ)
    assert changes and changes[0]["to"] == "terminal"
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "run"


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(p, "load_rules_config", lambda: {})
    assert p._apply_tool_aliases(_resp("run", {"cmd": "ls"}), REQ) == []
    assert p.DEFAULT_RULES_CONFIG["tool_aliases"]["enabled"] is False


def test_handles_ollama_native_shape(monkeypatch):
    _cfg(monkeypatch)
    r = {"message": {"tool_calls": [{"function": {"name": "run", "arguments": {"cmd": "ls"}}}]}}
    assert p._apply_tool_aliases(r, REQ)[0]["to"] == "terminal"
    assert r["message"]["tool_calls"][0]["function"]["name"] == "terminal"


def test_alias_prevents_the_hallucination_finding(monkeypatch):
    """Ordering is the point: renaming first means hallucinated_tool never sees a name that was
    always fixable."""
    monkeypatch.setattr(p, "load_rules_config", lambda: {
        "tool_aliases": {"enabled": True, "action": "rewrite", "map": {"run": "terminal"},
                         "require_declared": True},
        "hallucinated_tool": {"enabled": True},
    })
    r = _resp("run", {"cmd": "ls"})
    assert p._validate_response_tool_calls(r, REQ), "unfixed, this is a hallucination"
    r2 = _resp("run", {"cmd": "ls"})
    p._apply_tool_aliases(r2, REQ)
    assert p._validate_response_tool_calls(r2, REQ) == [], "renamed first, nothing to report"


def test_aliasing_requests_response_buffering(monkeypatch):
    """A streamed response is only reassembled when a post-flight rule asks for it. 854 of 872
    Hermes requests stream, so without this the rule would be inert exactly where it's needed."""
    body = {"tools": [{"type": "function", "function": {"name": "terminal"}}]}
    monkeypatch.setattr(p, "load_rules_config", lambda: {})
    assert p._post_flight_active(body) is False
    monkeypatch.setattr(p, "load_rules_config",
                        lambda: {"tool_aliases": {"enabled": True, "map": {"run": "terminal"}}})
    assert p._post_flight_active(body) is True
    # Enabled but with nothing to do shouldn't cost every streaming client its passthrough.
    monkeypatch.setattr(p, "load_rules_config",
                        lambda: {"tool_aliases": {"enabled": True, "map": {}}})
    assert p._post_flight_active(body) is False
