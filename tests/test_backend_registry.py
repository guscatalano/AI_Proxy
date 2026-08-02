"""One list of backends instead of ten.

Every backend used to be wired by hand into the metrics gather, the INSERT columns, the INSERT
*values*, system_now, GENERATION_RATE_UPSTREAMS, _BENCH_LOAD_MODES, the bench index, the
router's upstream table, a provider tab and a panel. Every UI bug in this area was a missed
site, not a logic error — including a forgotten INSERT value that killed telemetry for an hour.
"""
from ai_proxy import proxy as P


def test_every_provider_is_routable():
    # The router names a backend; without a base URL the request has nowhere to go.
    for name, prov in P.PROVIDERS.items():
        assert prov.base_url, f"{name} has no base_url"
        assert prov.base_url.startswith("http"), name


def test_load_modes_are_derived_not_repeated():
    modes = P._bench_load_modes()
    assert set(modes) == set(P.PROVIDERS)
    assert modes["ollama"] == "on-demand"     # loads whatever is asked for
    assert modes["lmstudio"] == "jit"         # loads on use, evicts on its own schedule
    assert modes["vllm"] == "fixed"           # one model per process, chosen at launch
    assert modes["llamacpp"] == "fixed"


def test_registering_a_backend_is_the_only_step(monkeypatch):
    """The property the refactor exists for: one registration, and every derived site follows."""
    async def probe(client):
        return {"reachable": True, "available": [{"id": "m"}]}

    extra = P._FnProvider("newthing", "New Thing", probe, lambda: "http://localhost:9",
                          load_mode="jit")
    monkeypatch.setitem(P.PROVIDERS, "newthing", extra)

    assert P._bench_load_modes()["newthing"] == "jit"          # bench backend list
    assert "newthing" in {n for n, p in P.PROVIDERS.items() if p.measures_decode}
    assert {n: p.base_url for n, p in P.PROVIDERS.items()}["newthing"] == "http://localhost:9"


def test_a_backend_can_opt_out_of_decode_stats(monkeypatch):
    # Attributing decode rate to a backend that does not stream would corrupt the figure.
    quiet = P._FnProvider("batchy", "Batchy", None, lambda: "http://x", measures_decode=False)
    monkeypatch.setitem(P.PROVIDERS, "batchy", quiet)
    assert "batchy" not in {n for n, p in P.PROVIDERS.items() if p.measures_decode}


def test_side_services_are_not_providers():
    # ComfyUI is startable and probeable but serves no models; treating it as a provider would
    # put it in the bench's backend list with nothing to benchmark.
    assert "comfyui" in P.SIDE_SERVICES
    assert "comfyui" not in P.PROVIDERS
    assert not hasattr(P.SIDE_SERVICES["comfyui"], "load_mode")


def test_lookup_spans_both_kinds():
    assert P.backend("llamacpp") is P.PROVIDERS["llamacpp"]
    assert P.backend("comfyui") is P.SIDE_SERVICES["comfyui"]
    assert P.backend("nope") is None


def test_control_mechanism_is_declared_per_backend():
    # Three different mechanisms exist; the backend says which, rather than each call site
    # guessing from the name.
    assert P.PROVIDERS["vllm"].control == "docker"
    assert P.PROVIDERS["llamacpp"].control == "unit"
    assert P.SIDE_SERVICES["comfyui"].control == "unit"
    assert P.PROVIDERS["ollama"].control == "none"


def _row(**cols):
    """A metrics row as sqlite3.Row would present it."""
    class R(dict):
        def keys(self):
            return list(super().keys())
    return R(cols)


def test_snapshots_are_stored_as_one_blob_per_sample(client):
    # The INSERT no longer names backends, so adding one cannot desync its column list from
    # its value list — the failure that stopped telemetry for an hour.
    import inspect
    # The INSERT moved off the event loop into its own function; the property is about the
    # write, so follow it there rather than pinning where it happens to live.
    src = inspect.getsource(P._collect_once) + inspect.getsource(P._write_metrics_sample)
    assert "backends_json" in src
    for legacy in ("ollama_json", "vllm_json", "llamacpp_json"):
        assert legacy not in src, f"{legacy} still written by hand"


def test_reading_prefers_the_blob(client):
    row = _row(backends_json='{"ollama": {"reachable": true}, "vllm": {"reachable": false}}')
    out = P._backends_from_row(row)
    assert out["ollama"]["reachable"] is True
    assert out["vllm"]["reachable"] is False


def test_rows_written_before_the_registry_still_read(client):
    # History must survive the storage change, or every chart loses its past.
    row = _row(ollama_json='{"reachable": true}', vllm_json='{"reachable": true}',
               backends_json=None)
    out = P._backends_from_row(row)
    assert out["ollama"]["reachable"] is True
    assert out["vllm"]["reachable"] is True


def test_every_registered_backend_gets_a_key_even_when_absent(client):
    # Consumers index these directly; a missing key would be an AttributeError in the UI.
    out = P._backends_from_row(_row(backends_json=None))
    for name in list(P.PROVIDERS) + list(P.SIDE_SERVICES):
        assert name in out and out[name] == {}


def test_corrupt_json_does_not_take_the_sample_down(client):
    out = P._backends_from_row(_row(backends_json="not json", ollama_json='{"reachable": true}'))
    assert out["ollama"]["reachable"] is True     # the legacy column still read


def test_one_failing_probe_does_not_lose_the_whole_sample(client, monkeypatch):
    """return_exceptions=True: losing every metric because a single backend raised is how the
    collector became a single point of failure."""
    import inspect
    src = inspect.getsource(P._collect_once)
    assert "return_exceptions=True" in src


def test_one_place_knows_how_each_backend_starts(client, monkeypatch):
    """The three mechanisms were re-derived from the backend's name at every call site — the
    residency handshake, the load endpoint and the UI each knew vLLM means docker and ComfyUI
    means systemd. Adding llama.cpp meant teaching all of them again."""
    import asyncio
    calls = []

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        calls.append(list(args))
        return 0, ""

    async def container():
        return "qwen-vllm"

    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["services"] = {"comfyui": {"unit": "comfyui.service", "scope": "user"},
                      "llamacpp": {"unit": "llamacpp.service", "scope": "user"}}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "_vllm_container", container)
    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")

    assert asyncio.run(P.PROVIDERS["vllm"].stop())["via"] == "docker"
    assert ["/usr/bin/docker", "stop", "qwen-vllm"] in calls
    assert asyncio.run(P.PROVIDERS["llamacpp"].start())["via"] == "systemctl"
    assert ["systemctl", "--user", "start", "llamacpp.service"] in calls
    assert asyncio.run(P.SIDE_SERVICES["comfyui"].stop())["via"] == "systemctl"


def test_an_unmanaged_backend_says_so_rather_than_pretending(client):
    import asyncio
    res = asyncio.run(P.PROVIDERS["ollama"].start())
    assert res["ok"] is False and res["via"] == "none"
    assert "not managed" in res["detail"]


def test_the_load_endpoint_does_not_branch_on_backend_names(client):
    """The tenth and last hand-wired site.

    `control_model_load` used to be four `if upstream == "..."` branches, each with its own
    mechanism inline — which is how llama.cpp's exemption from the model-name check ended up
    written into a condition here instead of stated on the backend it describes.
    """
    import ast
    import inspect
    src = inspect.getsource(P.control_model_load)
    fn = ast.parse(src).body[0]
    doc = fn.body[0]                                   # prose may name backends; code may not
    src = "\n".join(src.splitlines()[doc.end_lineno:])
    for name in P.PROVIDERS:
        # A default upstream is a default, not a branch, so only comparisons count.
        assert f'== "{name}"' not in src, f"{name} still branched on in the load endpoint"
        assert f'"{name}",' not in src, f"{name} still listed in the load endpoint"
    assert "upstream ==" not in src and " in (" not in src


def test_whether_a_name_is_needed_is_the_backend_s_own_answer(client):
    # vLLM and llama.cpp are launched with one model, so "load" means "start this server".
    assert P.PROVIDERS["ollama"].requires_model_name is True
    assert P.PROVIDERS["lmstudio"].requires_model_name is True
    assert P.PROVIDERS["vllm"].requires_model_name is False
    assert P.PROVIDERS["llamacpp"].requires_model_name is False


def test_an_unknown_upstream_is_rejected_rather_than_sent_to_ollama(client):
    # It used to fall through the branches into the Ollama path, so a typo'd backend name
    # quietly asked Ollama to load a model it had never heard of and reported its 404 instead.
    r = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "vlmm", "model": "qwen"})
    assert r.status_code == 404
    body = r.json()
    assert "vlmm" in body["error"]
    assert "vllm" in body["known"], "the reply should say what it would have accepted"


def test_a_registered_backend_is_loadable_without_touching_the_endpoint(monkeypatch, client):
    """Completes the property: registration alone reaches the load path too."""
    class Newthing(P._FnProvider):
        async def load(self, payload, name):
            return {"ok": True, "model": name, "upstream": "newthing"}

    async def probe(c):
        return {"reachable": True}

    monkeypatch.setitem(P.PROVIDERS, "newthing",
                        Newthing("newthing", "New Thing", probe, lambda: "http://localhost:9"))
    d = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "newthing", "model": "m"}).json()
    assert d == {"ok": True, "model": "m", "upstream": "newthing"}


def test_a_backend_with_no_load_mechanism_says_so(monkeypatch, client):
    # Better a 501 than the silent fall-through to Ollama that the branch chain gave.
    async def probe(c):
        return {"reachable": True}

    monkeypatch.setitem(P.PROVIDERS, "inert",
                        P._FnProvider("inert", "Inert", probe, lambda: "http://localhost:9"))
    r = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "inert", "model": "m"})
    assert r.status_code == 501
    assert "cannot load" in r.json()["error"]
