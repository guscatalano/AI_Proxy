"""Ollama's metadata endpoints stay on Ollama.

Observed in production: vscode-copilot sent POST /api/show {"model": "qwen3-coder:tuned"}, the
model_router matched on the name and rewrote both the model and the upstream to vLLM, and the
request went to an OpenAI-compatible server that has no /api/show. 21 x 404 in one second.
"""
from ai_proxy import proxy as P


def test_metadata_endpoints_are_recognised():
    for p in ("/api/show", "/api/tags", "/api/ps", "/api/pull", "/api/delete",
              "/api/copy", "/api/create", "/api/version", "/api/blobs/sha256:abc"):
        assert P._is_ollama_native_only(p), p
    # With or without the leading slash — the proxy passes full_path unprefixed.
    assert P._is_ollama_native_only("api/show")


def test_inference_endpoints_are_not():
    # These must keep routing normally, or the router stops working at all.
    for p in ("/api/chat", "/api/generate", "/api/embeddings", "/v1/chat/completions",
              "/v1/completions", "/v1/models", "/v1/messages"):
        assert not P._is_ollama_native_only(p), p


def test_a_prefix_match_does_not_over_reach():
    # /api/shows-something is not /api/show.
    assert not P._is_ollama_native_only("/api/showcase")
    assert not P._is_ollama_native_only("/api/tagsearch")


def test_show_is_not_rerouted_by_a_matching_rule(client, monkeypatch):
    # The exact production config: qwen* -> qwen3-coder-next on vllm.
    cfg = dict(P.load_rules_config())
    cfg["model_router"] = {
        "enabled": True,
        "rules": [{"if": {"from_model_prefix": "qwen"}, "then": "qwen3-coder-next",
                   "upstream": "vllm"}],
    }
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)

    # The rule still fires for real inference...
    body = {"model": "qwen3-coder:tuned", "messages": [{"role": "user", "content": "hi"}]}
    verdict = P.evaluate_router(dict(body), {"path": "/api/chat", "upstream": "ollama"})
    assert verdict and verdict.get("to") == "qwen3-coder-next"
    assert verdict.get("upstream") == "vllm"

    # ...and the request path decides whether it is allowed to apply. /api/show is metadata:
    # answering it from vLLM is a 404, and answering about a different model is a wrong answer.
    assert P._is_ollama_native_only("/api/show")
    assert not P._is_ollama_native_only("/api/chat")
