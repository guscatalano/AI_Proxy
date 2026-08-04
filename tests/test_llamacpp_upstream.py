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


def test_model_id_is_the_name_not_the_home_dir_path(client):
    # llama-server sometimes reports the model id as the full GGUF PATH (under the user's home dir, with a
    # -NNNNN-of-NNNNN shard suffix). The snapshot must surface just the model NAME — not the path; the full
    # path stays in model_path for quant/context inference.
    p = r"C:\Users\crimson\models\DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf"
    snap = asyncio.run(P._llamacpp_snapshot(_Client(
        models={"data": [{"id": p}]}, props={"model_path": p})))
    assert snap["loaded"][0]["id"] == "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"
    assert "crimson" not in snap["loaded"][0]["id"], "the model name must not leak the home directory"
    assert "IQ2_XXS" in (snap["model_path"] or ""), "the full path is still available for quant/context"


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
    assert P.PROVIDERS["llamacpp"].measures_decode is True


def test_bench_index_includes_llamacpp_models(client, monkeypatch):
    def fake_now():          # sync, like the real handler — an async fake hid a real bug
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
    # The bench's backend chips are derived from the registry, so registering a backend is
    # the single act that makes it appear — no second list to remember.
    assert "llamacpp" in P.PROVIDERS
    # "fixed": one model per process, chosen on the command line — the bench can measure what
    # is up but cannot swap models there, same as vLLM.
    assert P.PROVIDERS["llamacpp"].load_mode == "fixed"

    def fake_now():          # sync, like the real handler
        return {"llamacpp": {"reachable": True, "available": [{"id": "ds4"}]}}
    monkeypatch.setattr(P, "system_now", fake_now)
    d = asyncio.run(P.bench_models())
    names = {u["upstream"] for u in d["upstreams"]}
    assert "llamacpp" in names
    assert next(u for u in d["upstreams"] if u["upstream"] == "llamacpp")["reachable"] is True


def test_vllm_container_is_found_while_stopped(client, monkeypatch):
    """The whole point of a Start button is that it works when the thing is not running.

    docker's {{.Ports}} column is the live port map and is empty for a stopped container, so
    matching on it meant the proxy could stop vLLM and then report "no local vLLM container
    publishing this port was found" when asked to start it again.
    """
    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        if "ps" in args:
            return 0, "qwen-vllm\nornith-vllm\ndockpeek\n"
        if "inspect" in args:
            # Exactly what a stopped container reports: bindings present, live ports absent,
            # State.Running false.
            return 0, ('/qwen-vllm\tfalse\t{"8000/tcp":[{"HostIp":"","HostPort":"8001"}]}\n'
                       '/ornith-vllm\tfalse\t{"8000/tcp":[{"HostIp":"","HostPort":"8002"}]}\n'
                       '/dockpeek\tfalse\t{"8000/tcp":[{"HostIp":"","HostPort":"3420"}]}\n')
        return 1, ""
    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "VLLM_URL", "http://localhost:8001")
    assert asyncio.run(P._vllm_container()) == "qwen-vllm"

    # And it must not grab a container that merely happens to be listed.
    monkeypatch.setattr(P, "VLLM_URL", "http://localhost:9999")
    assert asyncio.run(P._vllm_container()) is None


def test_info_advertises_the_slot(client):
    d = client.get("/__proxy/api/info").json()
    assert d.get("llamacpp") == P.LLAMACPP_URL


def test_a_running_llamacpp_keys_the_index_by_its_checkpoint_path(client, monkeypatch):
    """The stopped fallback keys on the configured PATH and sweeps submit that path. When the
    display id was cleaned for the System tab, a running llama.cpp stopped matching its own
    cells: eleven server-context cells failed preflight in zero seconds each with
    "does not serve <path> — it has <clean name>"."""
    import asyncio
    path = "/models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf"

    def fake_sysinfo():
        return {"llamacpp": {"available": [{"id": "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"}],
                             "model_path": path, "n_ctx": 32768, "parallel": 1},
                "ollama": {}, "lmstudio": {}, "vllm": {}}

    monkeypatch.setattr(P, "system_now", fake_sysinfo)
    monkeypatch.setattr(P, "_llamacpp_cfg", lambda: {"model": path})
    async def no_containers():
        return []
    monkeypatch.setattr(P, "_vllm_configs", no_containers)
    index = asyncio.run(P._bench_model_index())
    key = f"llamacpp:{path}"
    assert key in index, sorted(k for k in index if k.startswith("llamacpp"))
    assert index[key]["loaded"] is True
    meta = P._bench_resolve_model(index, path, "llamacpp")
    assert meta and meta.get("loaded"), "the sweep's own model string must resolve while up"
