"""llama.cpp as an upstream in its own right.

Both llama-server and LM Studio speak the same OpenAI shape, so running llama.cpp on LM
Studio's port works — and then every log row, bench result and report attributes the numbers to
"lmstudio". Mislabelled measurements are worse than no measurements, which is why this gets its
own slot instead.
"""
import asyncio

from ai_proxy import proxy as P


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Client:
    def __init__(self, models=None, props=None, fail=False):
        self.models, self.props, self.fail = models, props, fail

    async def get(self, url, *a, **kw):
        if self.fail:
            raise P.httpx.RequestError("refused")
        if url.endswith("/v1/models"):
            return _Resp(200 if self.models is not None else 503, self.models or {})
        if url.endswith("/props"):
            return _Resp(200 if self.props is not None else 404, self.props or {})
        return _Resp(404, {})


def test_it_is_a_routable_upstream(client):
    # Without this the router can name it but the request has nowhere to go.
    assert P.LLAMACPP_URL
    assert P.LLAMACPP_URL != P.LMSTUDIO_URL, "must not share LM Studio's port by default"


def test_snapshot_reports_the_served_model(client):
    snap = asyncio.run(P._llamacpp_snapshot(_Client(
        models={"data": [{"id": "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"}]},
        props={"model_path": "/models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf",
               "total_slots": 4,
               "default_generation_settings": {"n_ctx": 65536}})))
    assert snap["reachable"] is True
    assert snap["loaded"][0]["id"] == "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"
    assert snap["n_ctx"] == 65536
    assert snap["parallel"] == 4          # the serial-vs-parallel question, answered by the server
    assert "IQ2_XXS" in (snap["model_path"] or "")


def test_quantisation_is_recovered_from_the_filename(client):
    # llama-server's API never states the quant; the path is the only place it appears.
    snap = asyncio.run(P._llamacpp_snapshot(_Client(
        models={"data": [{"id": "model"}]},
        props={"model_path": "/m/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf"})))
    assert snap["loaded"][0]["quant"]


def test_unreachable_is_quiet(client):
    snap = asyncio.run(P._llamacpp_snapshot(_Client(fail=True)))
    assert snap["reachable"] is False and snap["loaded"] == []


def test_missing_props_still_yields_the_model(client):
    # /props is a bonus; a server without it must not vanish from the index.
    snap = asyncio.run(P._llamacpp_snapshot(_Client(models={"data": [{"id": "m"}]}, props=None)))
    assert snap["reachable"] is True and snap["loaded"][0]["id"] == "m"


def test_it_counts_toward_decode_rate_stats(client):
    # Left out, 100% of its traffic would be invisible in the throughput figures — exactly the
    # bug vLLM had when it was added as an upstream and this set was not updated. The set is a
    # local inside stats(), so this guards the literal rather than an importable name.
    import inspect
    src = inspect.getsource(P.stats)
    line = next(l for l in src.splitlines() if "GENERATION_RATE_UPSTREAMS = " in l)
    assert "llamacpp" in line, line


def test_bench_index_includes_llamacpp_models(client, monkeypatch):
    async def fake_now():
        return {"ollama": {}, "lmstudio": {}, "vllm": {},
                "llamacpp": {"reachable": True, "n_ctx": 65536, "parallel": 4,
                             "model_path": "/m/x-UD-IQ2_XXS-00001-of-00003.gguf",
                             "available": [{"id": "ds4-flash", "arch": "llama.cpp",
                                            "quant": "IQ2_XXS"}]}}
    monkeypatch.setattr(P, "system_now", fake_now)
    idx = asyncio.run(P._bench_model_index())
    key = next(k for k in idx if k.startswith("llamacpp:"))
    rec = idx[key]
    assert rec["loaded"] is True
    assert rec["max_context"] == 65536 and rec["parallel"] == 4
    assert "IQ2_XXS" in (rec.get("quant") or "")


def test_bench_lists_it_as_a_backend(client, monkeypatch):
    # The bench's backend chips come from _BENCH_LOAD_MODES, not from the model index, so a
    # backend missing from that table is invisible in the bench UI even when it is serving.
    assert "llamacpp" in P._BENCH_LOAD_MODES
    # "fixed": one model per process, chosen on the command line — the bench can measure what
    # is up but cannot swap models there, same as vLLM.
    assert P._BENCH_LOAD_MODES["llamacpp"] == "fixed"

    async def fake_now():
        return {"llamacpp": {"reachable": True, "available": [{"id": "ds4"}]}}
    monkeypatch.setattr(P, "system_now", fake_now)
    d = asyncio.run(P.bench_models())
    names = {u["upstream"] for u in d["upstreams"]}
    assert "llamacpp" in names
    assert next(u for u in d["upstreams"] if u["upstream"] == "llamacpp")["reachable"] is True


def test_info_advertises_the_slot(client):
    d = client.get("/__proxy/api/info").json()
    assert d.get("llamacpp") == P.LLAMACPP_URL
