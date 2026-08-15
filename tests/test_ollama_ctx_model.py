"""Per-model context length, delivered by a derived model.

Measured on the box: Ollama's OpenAI-compatible endpoint ignores num_ctx both at top level and
nested under options — only the native /api/chat honours it. A preload does not survive either,
because the OpenAI handler sends the server default on every request and reloads the model,
which is how a 65,536 preload was silently replaced by 262,144 mid-benchmark.

So the rule delivers num_ctx the way Ollama itself does: a derived model carrying the parameter,
which every endpoint respects because it lives in the model rather than the request.
"""
import pytest

from ai_proxy import proxy


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _fresh():
    proxy._CTX_MODELS_SEEN.clear()
    yield
    proxy._CTX_MODELS_SEEN.clear()


def _rules(monkeypatch, cfg):
    monkeypatch.setattr(proxy, "load_rules_config", lambda: {"ollama_options": cfg})


# --- naming ------------------------------------------------------------------------------


@pytest.mark.parametrize("model,ctx,want", [
    ("qwen3.8:27b", 524288, "qwen3.8-27b-ctx512k"),
    ("nemotron-3.5-lightning:30b-a3b", 1048576, "nemotron-3.5-lightning-30b-a3b-ctx1m"),
    ("gemma4:26b", 131072, "gemma4-26b-ctx128k"),
])
def test_derived_names_are_readable_in_ollama_list(model, ctx, want):
    assert proxy._ctx_derived_name(model, ctx) == want


def test_derived_names_avoid_characters_a_tag_cannot_hold():
    n = proxy._ctx_derived_name("hf.co/unsloth/Weird_Model:Q4_K_M", 65536)
    assert "/" not in n and ":" not in n


# --- the swap ----------------------------------------------------------------------------


class _FakeClient:
    """Stands in for Ollama: records what was created, answers /api/tags from that."""
    created: list = []
    existing: list = []
    fail: bool = False

    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def get(self, url, **k):
        class R:
            @staticmethod
            def json(): return {"models": [{"name": n} for n in _FakeClient.existing]}
        return R()

    caps: list = []

    async def post(self, url, json=None, **k):
        if url.endswith("/api/show"):
            class S:
                status_code = 200
                @staticmethod
                def json(): return {"capabilities": _FakeClient.caps}
            return S()

        class R:
            status_code = 500 if _FakeClient.fail else 200
            text = "boom"
        if not _FakeClient.fail:
            _FakeClient.created.append(json)
            _FakeClient.existing.append(json["model"])
        return R()


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    _FakeClient.created, _FakeClient.existing, _FakeClient.fail = [], [], False
    _FakeClient.caps = []
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeClient)


@pytest.mark.anyio
async def test_a_configured_model_is_swapped_for_its_pinned_variant(monkeypatch):
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 524288}}})
    body = {"model": "nemo", "messages": []}
    out = await proxy.apply_ollama_ctx_model(body, {"path": "v1/chat/completions"})
    assert out["to"] == "nemo-ctx512k" and out["num_ctx"] == 524288
    assert body["model"] == "nemo-ctx512k", "the forwarded request must name the variant"
    assert _FakeClient.created[0]["parameters"]["num_ctx"] == 524288


@pytest.mark.anyio
async def test_the_variant_is_created_once_not_per_request(monkeypatch):
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 65536}}})
    for _ in range(3):
        await proxy.apply_ollama_ctx_model({"model": "nemo"}, {"path": "v1/chat/completions"})
    assert len(_FakeClient.created) == 1, "creating on every request would add latency forever"


@pytest.mark.anyio
async def test_an_existing_variant_is_reused_without_creating(monkeypatch):
    _FakeClient.existing.append("nemo-ctx64k")
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 65536}}})
    body = {"model": "nemo"}
    await proxy.apply_ollama_ctx_model(body, {"path": "v1/chat/completions"})
    assert body["model"] == "nemo-ctx64k"
    assert _FakeClient.created == [], "it already existed"


@pytest.mark.anyio
async def test_the_native_path_is_left_alone(monkeypatch):
    """num_ctx works natively; swapping there would be a pointless model rename."""
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 65536}}})
    body = {"model": "nemo"}
    # router_ctx builds "/" + full_path, so the native path really does carry a leading slash.
    assert await proxy.apply_ollama_ctx_model(body, {"path": "/api/chat"}) is None
    assert body["model"] == "nemo"


@pytest.mark.anyio
async def test_models_without_a_configured_context_are_untouched(monkeypatch):
    _rules(monkeypatch, {"enabled": True, "per_model": {"other": {"num_ctx": 65536}}})
    body = {"model": "nemo"}
    assert await proxy.apply_ollama_ctx_model(body, {"path": "v1/chat/completions"}) is None
    assert body["model"] == "nemo"


@pytest.mark.anyio
async def test_a_disabled_rule_does_nothing(monkeypatch):
    _rules(monkeypatch, {"enabled": False, "per_model": {"nemo": {"num_ctx": 65536}}})
    assert await proxy.apply_ollama_ctx_model({"model": "nemo"}, {"path": "v1/chat"}) is None


@pytest.mark.anyio
async def test_a_failed_creation_serves_the_request_anyway(monkeypatch):
    """A tuning knob must never cost a request. Default window beats a 500."""
    _FakeClient.fail = True
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 65536}}})
    body = {"model": "nemo"}
    assert await proxy.apply_ollama_ctx_model(body, {"path": "v1/chat/completions"}) is None
    assert body["model"] == "nemo", "the original model must still be forwarded"


@pytest.mark.anyio
async def test_a_global_default_applies_when_no_per_model_entry(monkeypatch):
    _rules(monkeypatch, {"enabled": True, "defaults": {"num_ctx": 32768}})
    body = {"model": "anything"}
    out = await proxy.apply_ollama_ctx_model(body, {"path": "v1/chat/completions"})
    assert out["num_ctx"] == 32768 and body["model"] == "anything-ctx32k"


def test_openai_path_no_longer_claims_num_ctx_it_cannot_deliver(monkeypatch):
    """It used to report num_ctx as applied while changing nothing."""
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 65536,
                                                                "temperature": 0.2}}})
    body = {"model": "nemo", "messages": []}
    res = proxy.evaluate_ollama_options(body, {"path": "v1/chat/completions"})
    assert "num_ctx" not in (res or {}).get("applied", {})
    assert "num_ctx" not in body
    assert res["applied"]["temperature"] == 0.2, "other options still apply normally"


@pytest.mark.anyio
async def test_a_vision_model_is_never_derived(monkeypatch):
    """Deriving minicpm-v4.5 produced a variant that failed to load — the projector does not
    survive. A working default window beats a pinned model that cannot answer."""
    _FakeClient.caps = ["completion", "vision", "tools"]
    _rules(monkeypatch, {"enabled": True, "per_model": {"minicpm": {"num_ctx": 16384}}})
    body = {"model": "minicpm"}
    assert await proxy.apply_ollama_ctx_model(body, {"path": "/v1/chat/completions"}) is None
    assert body["model"] == "minicpm", "the base model must still be forwarded"
    assert _FakeClient.created == [], "nothing should have been created"


@pytest.mark.anyio
async def test_a_text_model_is_still_derived(monkeypatch):
    _FakeClient.caps = ["completion", "tools", "thinking"]
    _rules(monkeypatch, {"enabled": True, "per_model": {"nemo": {"num_ctx": 16384}}})
    body = {"model": "nemo"}
    out = await proxy.apply_ollama_ctx_model(body, {"path": "/v1/chat/completions"})
    assert out and body["model"] == "nemo-ctx16k"
