"""A router target of "auto" follows whichever model the upstream is serving.

Every model swap so far needed the catch-all rule rewritten by hand, and forgetting once left
traffic routed to a backend that had been stopped — opencode got "Could not open a connection to
vllm2" rather than an answer. "auto" removes the step that was being forgotten: start a
container, and traffic follows it.
"""
from ai_proxy import proxy as P


def _cfg(monkeypatch, then, upstream="vllm"):
    base = dict(P.load_rules_config())
    base["model_router"] = {"enabled": True, "aliases": {}, "advertise": {},
                            "rules": [{"if": {}, "then": then, "upstream": upstream}]}
    monkeypatch.setattr(P, "load_rules_config", lambda: base)


def _serving(monkeypatch, name):
    monkeypatch.setattr(P, "_live_served_model", lambda up: name)


def test_auto_resolves_to_the_live_model(client, monkeypatch):
    _cfg(monkeypatch, "auto")
    _serving(monkeypatch, "whatever-is-loaded")
    body = {"model": "gemma4", "messages": []}
    out = P.evaluate_router(body, {"upstream": "vllm"})
    assert out and out["to"] == "whatever-is-loaded"
    assert body["model"] == "whatever-is-loaded", "the outgoing body must carry the real name"


def test_swapping_the_container_moves_traffic_with_no_config_change(client, monkeypatch):
    _cfg(monkeypatch, "auto")
    _serving(monkeypatch, "model-a")
    b1 = {"model": "gemma4", "messages": []}
    P.evaluate_router(b1, {"upstream": "vllm"})
    _serving(monkeypatch, "model-b")          # the swap; rules untouched
    b2 = {"model": "gemma4", "messages": []}
    P.evaluate_router(b2, {"upstream": "vllm"})
    assert (b1["model"], b2["model"]) == ("model-a", "model-b")


def test_an_unreachable_upstream_leaves_the_model_alone(client, monkeypatch):
    """Rewriting to nothing would send a request naming no model at all."""
    _cfg(monkeypatch, "auto")
    _serving(monkeypatch, None)
    body = {"model": "gemma4", "messages": []}
    assert P.evaluate_router(body, {"upstream": "vllm"}) is None
    assert body["model"] == "gemma4"


def test_an_explicit_target_is_still_honoured(client, monkeypatch):
    """Pinning a specific model must keep working — that is how a bench targets one engine."""
    _cfg(monkeypatch, "qwen3-coder-next")
    _serving(monkeypatch, "something-else")
    body = {"model": "gemma4", "messages": []}
    out = P.evaluate_router(body, {"upstream": "vllm"})
    assert out["to"] == "qwen3-coder-next"


def test_auto_is_case_and_space_insensitive(client, monkeypatch):
    _cfg(monkeypatch, "  AUTO ")
    _serving(monkeypatch, "live-one")
    body = {"model": "gemma4", "messages": []}
    assert P.evaluate_router(body, {"upstream": "vllm"})["to"] == "live-one"


# --- keeping the model the client named -------------------------------------------------------
#
# A rule could change the model or the upstream, but not "keep this name, just send it to vLLM".
# So selecting a backend meant also rewriting the name, and every request came back served by
# whatever the catch-all pointed at — which is not what "reach the model I specified" means.


def _passthrough(monkeypatch, upstream="vllm"):
    base = dict(P.load_rules_config())
    base["model_router"] = {"enabled": True, "aliases": {}, "advertise": {},
                            "rules": [{"if": {}, "upstream": upstream}]}
    monkeypatch.setattr(P, "load_rules_config", lambda: base)


def test_the_named_model_is_kept_and_only_the_backend_changes(client, monkeypatch):
    _passthrough(monkeypatch)
    body = {"model": "some-candidate-model", "messages": []}
    out = P.evaluate_router(body, {"upstream": "ollama"})
    assert out and out["upstream"] == "vllm"
    assert body["model"] == "some-candidate-model", "the client's choice must survive"
    assert out["to"] == "some-candidate-model"


def test_two_different_names_stay_different(client, monkeypatch):
    """The symptom of the old behaviour: everything arrived as one model."""
    _passthrough(monkeypatch)
    a = {"model": "model-a", "messages": []}
    b = {"model": "model-b", "messages": []}
    P.evaluate_router(a, {"upstream": "ollama"})
    P.evaluate_router(b, {"upstream": "ollama"})
    assert (a["model"], b["model"]) == ("model-a", "model-b")
