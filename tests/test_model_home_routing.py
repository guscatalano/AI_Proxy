"""A model reaches the backend that actually serves it.

Passthrough routing sends a request to the path default — Ollama for /v1/chat — so naming a model
only vLLM serves produced "model 'gemma4-vllm' not found" in 100ms. Four models in one sweep
failed that way while their containers sat stopped and startable: the proxy knew where each one
lived and never looked.

Only an unambiguous answer is used. Two backends serving the same name is exactly what the
qualified "model/backend" form is for, and guessing there would pick the wrong build silently.
"""
import asyncio

from ai_proxy import proxy as P


def _index(monkeypatch, mapping):
    """mapping: {"<upstream>:<model>": upstream}"""
    async def fake_index():
        return {k: {"model": k.split(":", 1)[1], "upstream": k.split(":", 1)[0]}
                for k in mapping}
    monkeypatch.setattr(P, "_bench_model_index", fake_index)
    P._MODEL_HOME_CACHE.update(ts=0.0, map={})


def test_a_vllm_only_model_resolves_to_vllm(client, monkeypatch):
    _index(monkeypatch, {"vllm:gemma4-vllm": 1, "ollama:muse-glimmer:30b": 1})
    assert asyncio.run(P._backend_serving("gemma4-vllm")) == "vllm"


def test_an_ollama_only_model_resolves_to_ollama(client, monkeypatch):
    _index(monkeypatch, {"vllm:gemma4-vllm": 1, "ollama:muse-glimmer": 1})
    assert asyncio.run(P._backend_serving("muse-glimmer")) == "ollama"


def test_an_ambiguous_name_is_left_alone(client, monkeypatch):
    """Both backends serve it, so guessing would silently pick a build. That is what the
    qualified form exists to disambiguate."""
    _index(monkeypatch, {"vllm:qwen3-coder-next": 1, "ollama:qwen3-coder-next": 1})
    assert asyncio.run(P._backend_serving("qwen3-coder-next")) is None


def test_an_unknown_model_resolves_to_nothing(client, monkeypatch):
    _index(monkeypatch, {"vllm:gemma4-vllm": 1})
    assert asyncio.run(P._backend_serving("never-heard-of-it")) is None


def test_tags_do_not_defeat_the_match(client, monkeypatch):
    """Ollama tags its names; the catalogue normalises before comparing."""
    _index(monkeypatch, {"ollama:muse-glimmer:30b": 1})
    assert asyncio.run(P._backend_serving("muse-glimmer:30b")) == "ollama"


def test_the_answer_is_cached(client, monkeypatch):
    calls = {"n": 0}

    async def counting():
        calls["n"] += 1
        return {"vllm:gemma4-vllm": {"model": "gemma4-vllm", "upstream": "vllm"}}
    monkeypatch.setattr(P, "_bench_model_index", counting)
    P._MODEL_HOME_CACHE.update(ts=0.0, map={})
    asyncio.run(P._backend_serving("gemma4-vllm"))
    first = calls["n"]
    asyncio.run(P._backend_serving("gemma4-vllm"))
    assert calls["n"] == first, "this runs per request; it must not rebuild the index each time"
