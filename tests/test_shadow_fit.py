"""A shadow run must not be able to kill the backend it borrows.

Ollama sizes its llama-server at OLLAMA_CONTEXT_LENGTH × OLLAMA_NUM_PARALLEL — 262,144 × 4 on
the box this was found on. A model that cannot hold that window does not fail the request; it
takes the whole Ollama service down, and every other model resident at the time with it.

Three shadow runs of a 2.5 GB model killed it three times:

    starting llama-server ... -c 1048576 -np 4
    common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce
                            device memory by 8615 MiB
    ollama.service: Failed with result 'signal'

The shadow path applied ollama_options but not the two things that actually protect the box:
the derived-model swap that carries num_ctx (Ollama's /v1 drops the field), and the fit gate.
"""
import asyncio
import json

from ai_proxy import proxy as P


def _target(model="qwen3:4b"):
    return {"target_model": model, "upstream_label": "ollama",
            "upstream_base": "http://localhost:11434"}


def _body():
    return {"model": "big-model", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def _no_send(monkeypatch):
    """Any attempt to reach the upstream is a failure of the test's premise."""
    sent = []

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            sent.append(url)
            raise AssertionError(f"the shadow must not have been sent: {url}")

    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    return sent


def test_a_shadow_that_cannot_fit_is_not_sent(client, monkeypatch):
    sent = _no_send(monkeypatch)

    async def refuse(model, assume_avail_mb=0, pin=None):
        return "qwen3:4b needs 82 GB of KV at the server default; 44 GB is free"
    monkeypatch.setattr(P, "_ollama_fit_refusal", refuse)

    asyncio.run(P._run_shadow("primary-1", _body(), "/v1/chat/completions",
                              _target(), "127.0.0.1", None))
    assert sent == []


def test_the_refusal_is_recorded_on_the_shadow_row(client, monkeypatch):
    """Silently dropping it would look identical to "shadows are not configured"."""
    _no_send(monkeypatch)

    async def refuse(model, assume_avail_mb=0, pin=None):
        return "would not fit"
    monkeypatch.setattr(P, "_ollama_fit_refusal", refuse)

    asyncio.run(P._run_shadow("primary-2", _body(), "/v1/chat/completions",
                              _target(), "127.0.0.1", None))
    conn = P.db()
    row = conn.execute(
        "SELECT shadow_of, status, error, model FROM requests WHERE shadow_of = ?",
        ("primary-2",)).fetchone()
    conn.close()
    assert row is not None, "the refused shadow left no trace at all"
    assert row["status"] == 0
    assert "would not fit" in (row["error"] or "")
    assert "shadow not sent" in (row["error"] or "")


def _recording_app(calls):
    """_run_shadow sends through app.state.client, not a fresh AsyncClient."""
    class _Client:
        def build_request(self, method, url, **k):
            calls.append(url)
            return ("req", url)

        async def send(self, req, **k):
            raise RuntimeError("stop here — it was already built, which is the assertion")

    return type("App", (), {"state": type("S", (), {"client": _Client()})()})()


def test_a_shadow_that_fits_still_goes(client, monkeypatch):
    """The gate must not become a blanket 'no shadows'."""
    calls = []

    async def allow(model, assume_avail_mb=0, pin=None):
        return None
    monkeypatch.setattr(P, "_ollama_fit_refusal", allow)

    asyncio.run(P._run_shadow("primary-3", _body(), "/v1/chat/completions",
                              _target(), "127.0.0.1", _recording_app(calls)))
    assert calls, "a shadow that fits must still be sent"
    assert calls[0].endswith("/v1/chat/completions")


def test_the_ctx_swap_reaches_the_shadow(client, monkeypatch):
    """num_ctx alone is a no-op on Ollama's /v1 — the derived model is what carries it."""
    swapped = []

    async def fake_swap(body, ctx):
        swapped.append(dict(body))
        body["model"] = body["model"].replace(":", "-") + "-ctx8k"
        return {"from": "x", "to": body["model"], "num_ctx": 8192}

    async def allow(model, assume_avail_mb=0, pin=None):
        return None
    monkeypatch.setattr(P, "apply_ollama_ctx_model", fake_swap)
    monkeypatch.setattr(P, "_ollama_fit_refusal", allow)

    seen = {}

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            seen["body"] = json.loads(k.get("content") or "{}") if k.get("content") else k.get("json")
            raise RuntimeError("stop")

    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    asyncio.run(P._run_shadow("primary-4", _body(), "/v1/chat/completions",
                              _target(), "127.0.0.1", None))
    assert swapped, "apply_ollama_ctx_model was never called for an Ollama shadow"


def test_a_non_ollama_shadow_skips_both(client, monkeypatch):
    """llama.cpp and LM Studio do not have Ollama's derived-model mechanism."""
    called = []

    async def refuse(model, assume_avail_mb=0, pin=None):
        called.append(model)
        return "nope"
    monkeypatch.setattr(P, "_ollama_fit_refusal", refuse)

    tgt = {**_target(), "upstream_label": "lmstudio"}

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            raise RuntimeError("stop")

    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    asyncio.run(P._run_shadow("primary-5", _body(), "/v1/chat/completions",
                              tgt, "127.0.0.1", None))
    assert called == [], "the Ollama fit gate must not gate other backends"


# --- the other way to kill Ollama: loading a model by hand -------------------------
# A zero-token generate is still a full load, so "make it resident" from the System tab
# reached the same 262,144 × 4 allocation the request path had already been taught to refuse.

def test_making_a_model_resident_is_gated_too(client, monkeypatch):
    posted = []

    async def refuse(model, assume_avail_mb=0, pin=None):
        return "needs 82 GB of KV at the server default; 44 GB is free"

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            posted.append(url)
            raise AssertionError(f"must not have loaded: {url}")

    monkeypatch.setattr(P, "_ollama_fit_refusal", refuse)
    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    res = asyncio.run(P.PROVIDERS["ollama"].load({}, "qwen3:4b"))
    assert posted == []
    assert getattr(res, "status_code", None) == 409
    assert b"82 GB" in bytes(res.body)


def test_force_still_loads_it(client, monkeypatch):
    """Same override the vLLM load path has — the guard is a default, not a wall."""
    posted = []

    async def refuse(model, assume_avail_mb=0, pin=None):
        return "would not fit"

    class _Resp:
        status_code = 200
        text = ""

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            posted.append(url)
            return _Resp()

    monkeypatch.setattr(P, "_ollama_fit_refusal", refuse)
    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    res = asyncio.run(P.PROVIDERS["ollama"].load({"force": True}, "qwen3:4b"))
    assert posted, "force:true must still load"
    assert res.get("ok") is True


def test_a_model_that_fits_loads_normally(client, monkeypatch):
    posted = []

    async def allow(model, assume_avail_mb=0, pin=None):
        return None

    class _Resp:
        status_code = 200
        text = ""

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            posted.append(url)
            return _Resp()

    monkeypatch.setattr(P, "_ollama_fit_refusal", allow)
    monkeypatch.setattr(P.httpx, "AsyncClient", _C)
    res = asyncio.run(P.PROVIDERS["ollama"].load({}, "qwen3:0.6b"))
    assert posted and res.get("ok") is True
