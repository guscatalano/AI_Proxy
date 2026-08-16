"""Filtering the requests list by source address.

The client filter answered "which app", but one box runs several agents and one agent name
arrives from several boxes — hermes and hermes-safety both come from 192.168.15.10, while
ai-proxy-bench has come from three different machines. "Which app" and "which machine" are
different questions and are often asked together, so the two filters AND rather than replace
each other.
"""
import ai_proxy
from ai_proxy import proxy


def _seed(rows):
    conn = proxy.db()
    conn.execute("DELETE FROM requests WHERE id LIKE 'ipf-%'")
    for i, (app, ip) in enumerate(rows):
        conn.execute(
            "INSERT INTO requests (id, ts, method, path, upstream_url, model, is_stream, "
            "client_ip, client_app, upstream, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"ipf-{i}", 1_800_000_000 + i, "POST", "/v1/chat/completions",
             "http://localhost:11434/v1/chat/completions", "m", 0, ip, app, "ollama", 200))
    conn.commit()
    conn.close()


def _cleanup():
    conn = proxy.db()
    conn.execute("DELETE FROM requests WHERE id LIKE 'ipf-%'")
    conn.commit()
    conn.close()


def _ids(client, params=""):
    return {i["id"] for i in client.get(f"/__proxy/api/requests?limit=200{params}").json()["items"]
            if i["id"].startswith("ipf-")}


def test_filtering_by_source_address(client):
    _seed([("hermes", "10.0.0.1"), ("hermes", "10.0.0.2"), ("opencode", "10.0.0.1")])
    try:
        assert _ids(client, "&ip=10.0.0.1") == {"ipf-0", "ipf-2"}
    finally:
        _cleanup()


def test_the_two_filters_narrow_together_rather_than_replacing_each_other(client):
    """hermes AND from 10.0.0.2 is one request, not the three that match either."""
    _seed([("hermes", "10.0.0.1"), ("hermes", "10.0.0.2"), ("opencode", "10.0.0.2")])
    try:
        assert _ids(client, "&client=hermes&ip=10.0.0.2") == {"ipf-1"}
    finally:
        _cleanup()


def test_no_filter_still_returns_everything(client):
    _seed([("hermes", "10.0.0.1"), ("opencode", "10.0.0.2")])
    try:
        assert _ids(client) == {"ipf-0", "ipf-1"}
    finally:
        _cleanup()


def test_an_address_nobody_used_returns_nothing_rather_than_everything(client):
    _seed([("hermes", "10.0.0.1")])
    try:
        assert _ids(client, "&ip=10.9.9.9") == set()
    finally:
        _cleanup()


def test_the_dropdown_gets_addresses_with_counts_and_the_apps_behind_them(client):
    """A bare 192.168.x.y is not memorable; which box is 'the hermes one' is what is being
    picked, so the apps ride along with the address."""
    _seed([("hermes", "10.0.0.1"), ("hermes-safety", "10.0.0.1"), ("opencode", "10.0.0.2")])
    try:
        body = client.get("/__proxy/api/requests?limit=200").json()
        rows = {r["ip"]: r for r in body["ips"]}
        assert "10.0.0.1" in rows and rows["10.0.0.1"]["count"] >= 2
        apps = set((rows["10.0.0.1"]["apps"] or "").split(","))
        assert {"hermes", "hermes-safety"} <= apps, apps
    finally:
        _cleanup()


def test_the_total_reflects_the_filter_so_the_pager_is_not_lying(client):
    """The pager pages through `total`; if it counted unfiltered rows it would offer pages
    that come back empty."""
    _seed([("hermes", "10.0.0.1"), ("hermes", "10.0.0.2"), ("hermes", "10.0.0.2")])
    try:
        both = client.get("/__proxy/api/requests?limit=200").json()["total"]
        one = client.get("/__proxy/api/requests?limit=200&ip=10.0.0.2").json()["total"]
        assert one == 2 and one < both
    finally:
        _cleanup()
