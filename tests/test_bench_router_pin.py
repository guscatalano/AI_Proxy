"""A benchmark measures the model it names, even when a router rule says otherwise.

With a catch-all model_router rule pointing every request at one model, an unpinned benchmark
sends its requests to that model instead of the one under test. Observed both ways on 2026-08-17
with the catch-all aimed at gemma4-vllm:

  loud   — a nemotron run died after 5 of 357 units. Auto-load saw a request for the other vLLM
           twin, tried to swap mid-benchmark, and refused because a benchmark was running (its
           own), surfacing as "a benchmark owns the box right now" — which reads like contention.
  quiet  — where the rewrite target is already loaded, the run simply completes, and the results
           table records one model's scores under another model's name. That one leaves no trace.

The UI has always sent x-proxy-no-router; only the API defaulted the other way, so scripted runs
were the exposed ones.
"""
from ai_proxy import proxy as P


def _headers(**cfg):
    return P._bench_headers(cfg)


# --- the header the pin rides on -----------------------------------------------------------


def test_a_pinned_bench_sends_the_no_router_header():
    assert _headers(bypass_router=True).get("x-proxy-no-router") == "1"


def test_an_unpinned_bench_omits_it():
    assert "x-proxy-no-router" not in _headers(bypass_router=False)


def test_the_bench_client_name_is_what_panic_mode_whitelists():
    """Exclusive runs arm panic mode, which 503s everything except this name — so a benchmark
    that lost it would be blocked by its own quiesce."""
    assert _headers().get("x-client-name") == "ai-proxy-bench"


def test_the_upstream_pin_rides_along(client):
    h = _headers(upstream="vllm")
    assert any(v == "vllm" for k, v in h.items() if "upstream" in k.lower()), h


# --- the default the API applies -----------------------------------------------------------


def _cfg_from_api(client, **payload):
    """Submit a run, read back the config the API stored, then remove the row.

    The row has to go rather than merely be cancelled: only one bench may be in flight, so a
    leftover running/pending row makes the next submission 409 and the second test in this file
    fail for a reason that has nothing to do with what it checks.
    """
    import json

    body = {"model": "nemotron-vllm", "suite": "coding-v1", "runs": 1,
            "max_tokens": 256, "upstream": "vllm", "exclusive": False, **payload}
    r = client.post("/__proxy/api/bench/run", json=body)
    assert r.status_code == 200, r.text
    rid = r.json().get("id")
    assert rid
    conn = P.db()
    try:
        row = conn.execute("SELECT config_json FROM bench_runs WHERE id=?", (rid,)).fetchone()
        cfg = json.loads(row["config_json"])
        conn.execute("DELETE FROM bench_runs WHERE id=?", (rid,))
        conn.commit()
    finally:
        conn.close()
    return cfg


def test_the_api_pins_by_default(client):
    """The regression this file exists for: the default used to be False."""
    assert _cfg_from_api(client)["bypass_router"] is True


def test_opting_out_is_still_possible(client):
    """Measuring the routed path is a legitimate thing to want — it just has to be asked for."""
    assert _cfg_from_api(client, bypass_router=False)["bypass_router"] is False
