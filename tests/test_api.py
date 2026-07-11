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


def test_unknown_proxy_api_404(client):
    # A path under the reserved __proxy namespace that isn't a route should 404,
    # not get forwarded upstream.
    r = client.get("/__proxy/api/definitely-not-a-real-endpoint")
    assert r.status_code == 404
