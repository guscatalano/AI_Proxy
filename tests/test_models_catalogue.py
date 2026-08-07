"""/v1/models: one catalogue across every backend the proxy fronts.

A client pointed at the proxy used to see only Ollama's list — the vLLM models the router
actually prefers were unselectable in any model picker. The union must include a stopped
container's configured model (the auto-loader boots it on first request), dedupe by id,
and stay best-effort when a backend is down.
"""
import json

from ai_proxy import proxy as P


class _Resp:
    def __init__(self, payload, code=200):
        self.status_code = code
        self._p = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _FakeClient:
    RESIDENT: list = []      # tag names ollama /api/ps reports as in-memory

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **k):
        if url.endswith("/v1/models"):
            return _Resp({"object": "list",
                          "data": [{"id": "gemma4:26b", "object": "model", "created": 123,
                                    "owned_by": "library"}]})
        if url.endswith("/api/ps"):
            return _Resp({"models": [{"name": n} for n in _FakeClient.RESIDENT]})
        return _Resp({}, 404)


def _patch_backends(monkeypatch, vllm=None, cfgs=None, lcpp=None, lms=None):
    # The endpoint caches its sweep for a couple of seconds; tests that swap the backends
    # out from under it must start from a cold cache or they assert on the last test's fleet.
    P._MODELS_CACHE.update(ts=0.0, data=None)
    async def _vllm(c):
        return vllm or {}

    async def _cfgs():
        return cfgs or []

    async def _lcpp(c):
        return lcpp or {}

    async def _lms(c):
        return lms or {}

    monkeypatch.setattr(P.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(P, "_vllm_snapshot", _vllm)
    monkeypatch.setattr(P, "_vllm_configs", _cfgs)
    monkeypatch.setattr(P, "_llamacpp_snapshot", _lcpp)
    monkeypatch.setattr(P, "_lmstudio_snapshot", _lms)


def test_catalogue_unions_every_backend(client, monkeypatch):
    _patch_backends(
        monkeypatch,
        vllm={"available": [{"id": "qwen3-coder-next", "max_context_length": 262144}]},
        cfgs=[{"container": "qwen-vllm", "model": "qwen3-coder-next", "state": "running"},
              {"container": "ornith-vllm", "model": "ornith", "state": "exited"}],
        lcpp={"available": [{"id": "ds4-flash"}], "n_ctx": 65536},
    )
    d = client.get("/v1/models").json()
    by_id = {m["id"]: m for m in d["data"]}
    assert set(by_id) == {"gemma4:26b", "qwen3-coder-next", "ornith", "ds4-flash"}
    assert by_id["gemma4:26b"]["owned_by"] == "ollama"
    assert by_id["gemma4:26b"]["created"] == 123
    assert by_id["qwen3-coder-next"]["owned_by"] == "vllm"
    assert by_id["qwen3-coder-next"]["max_model_len"] == 262144
    assert by_id["qwen3-coder-next"]["loaded"] is True
    # the stopped twin is listed AND says how it comes back
    assert by_id["ornith"]["proxy_state"] == "stopped — loads on first request"
    assert by_id["ornith"]["loaded"] is False
    assert by_id["ds4-flash"]["context_length"] == 65536
    # an ollama model on disk but not resident says so, both ways
    assert by_id["gemma4:26b"]["loaded"] is False
    assert by_id["gemma4:26b"]["proxy_state"] == "available — loads on request"


def test_a_resident_ollama_model_reports_loaded(client, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(_FakeClient, "RESIDENT", ["gemma4:26b"])
    d = client.get("/v1/models").json()
    m = next(x for x in d["data"] if x["id"] == "gemma4:26b")
    assert m["loaded"] is True and m["proxy_state"] == "loaded"


def test_catalogue_dedupes_serving_over_config(client, monkeypatch):
    """The running container's served id (with its context) must win over the bare config
    entry for the same model, and a duplicate must not appear twice."""
    _patch_backends(
        monkeypatch,
        vllm={"available": [{"id": "qwen3-coder-next", "max_context_length": 262144}]},
        cfgs=[{"container": "qwen-vllm", "model": "qwen3-coder-next", "state": "running"}],
    )
    d = client.get("/v1/models").json()
    hits = [m for m in d["data"] if m["id"] == "qwen3-coder-next"]
    assert len(hits) == 1
    assert hits[0].get("max_model_len") == 262144
    assert "proxy_state" not in hits[0] or hits[0]["proxy_state"] == "loaded"


def test_catalogue_survives_every_backend_down(client, monkeypatch):
    P._MODELS_CACHE.update(ts=0.0, data=None)   # answer from the dead fleet, not the cache

    class _DeadClient(_FakeClient):
        async def get(self, url, **k):
            raise P.httpx.ConnectError("down")

    async def _boom(*a):
        raise P.httpx.ConnectError("down")

    monkeypatch.setattr(P.httpx, "AsyncClient", _DeadClient)
    monkeypatch.setattr(P, "_vllm_snapshot", _boom)
    monkeypatch.setattr(P, "_vllm_configs", _boom)
    monkeypatch.setattr(P, "_llamacpp_snapshot", _boom)
    monkeypatch.setattr(P, "_lmstudio_snapshot", _boom)
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": []}


def test_the_catalogue_caches_its_sweep(client, monkeypatch):
    """Model pickers poll this. Each miss fans out to five backends, one of them through
    `docker inspect`, so a polling client must not turn a listing into load."""
    calls = {"n": 0}

    async def counting_cfgs():
        calls["n"] += 1
        return [{"container": "qwen-vllm", "model": "qwen3-coder-next", "state": "running"}]

    _patch_backends(monkeypatch, cfgs=[])
    monkeypatch.setattr(P, "_vllm_configs", counting_cfgs)
    first = client.get("/v1/models").json()
    for _ in range(4):
        assert client.get("/v1/models").json() == first
    assert calls["n"] == 1, f"swept the fleet {calls['n']} times for 5 requests"
    # ...and a cache that never expires would hide a container starting up.
    P._MODELS_CACHE.update(ts=0.0, data=None)
    client.get("/v1/models")
    assert calls["n"] == 2
