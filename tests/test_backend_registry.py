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
