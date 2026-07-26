"""End-to-end tests of the proxy's own API surface (no upstream required)."""
import ai_proxy


def test_info_reports_version(client):
    r = client.get("/__proxy/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == ai_proxy.__version__
    assert "upstream" in body and "port" in body


def test_health_ok(client):
    r = client.get("/__proxy/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "db" in body
    assert "path" in body["db"]


def test_ui_is_served(client):
    r = client.get("/__proxy/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "AI Proxy" in r.text


def test_requests_endpoint_returns_json(client):
    r = client.get("/__proxy/api/requests")
    assert r.status_code == 200
    r.json()  # must be valid JSON


def test_stats_endpoint_ok(client):
    r = client.get("/__proxy/api/stats")
    assert r.status_code == 200
    r.json()


def test_requests_client_filter_and_pagination(client):
    from ai_proxy import proxy as P

    conn = P.db()
    import time as _t
    now = _t.time()
    conn.executemany(
        "INSERT INTO requests (id, ts, method, path, upstream_url, client_app) "
        "VALUES (?, ?, 'POST', '/v1/messages', 'http://x', ?)",
        [(f"rf-{i}", now - i, "claude-code" if i % 2 == 0 else "openai-sdk") for i in range(10)],
    )
    conn.commit()
    conn.close()

    # Unfiltered: clients list surfaces both apps.
    d = client.get("/__proxy/api/requests?limit=100").json()
    apps = {c["app"] for c in d["clients"]}
    assert {"claude-code", "openai-sdk"} <= apps

    # Filtered by client: only that app's rows come back.
    f = client.get("/__proxy/api/requests?limit=100&client=claude-code").json()
    assert f["total"] >= 5
    assert all(it["client_app"] == "claude-code" for it in f["items"])

    # Pagination: page 2 (offset) returns different ids than page 1.
    p1 = client.get("/__proxy/api/requests?limit=3&offset=0&client=claude-code").json()
    p2 = client.get("/__proxy/api/requests?limit=3&offset=3&client=claude-code").json()
    assert {i["id"] for i in p1["items"]}.isdisjoint({i["id"] for i in p2["items"]})


def test_unknown_proxy_api_404(client):
    # A path under the reserved __proxy namespace that isn't a route should 404,
    # not get forwarded upstream.
    r = client.get("/__proxy/api/definitely-not-a-real-endpoint")
    assert r.status_code == 404


def test_generation_rate_includes_every_streaming_local_upstream():
    """vLLM streams token-by-token like Ollama and LM Studio. It was missing from this set,
    which silently emptied the decode metric on any box whose daily driver had moved to vLLM —
    the dashboard then fell back to a completion-over-total-time figure roughly 6x lower, which
    reads as a broken model rather than a missing filter."""
    import inspect
    import ai_proxy.proxy as p
    src = inspect.getsource(p.stats)
    assert "GENERATION_RATE_UPSTREAMS" in src
    line = next(l for l in src.splitlines() if "GENERATION_RATE_UPSTREAMS = " in l)
    for engine in ("ollama", "lmstudio", "vllm"):
        assert engine in line, f"{engine} missing from the decode-rate upstreams"
    # Anthropic batches its SSE, so (duration - ttft) is transfer time there, not decode time.
    assert "anthropic" not in line
