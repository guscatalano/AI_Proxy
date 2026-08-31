"""Live view for in-flight requests.

Two defects behind "live doesn't always work": a non-streaming request produced no live
text ever — streamer() is the only caller of _live_update_from_chunk and never runs for one —
yet the endpoint still reported active, so the pane claimed to be live and rendered blank for
the full 30s of a hindsight call. And chunk parsing had no memory across chunk boundaries, so
any SSE line split by the socket was dropped from the reconstructed text.
"""
import ai_proxy
from ai_proxy import proxy


def _seed(req_id, *, stream):
    proxy._LIVE_STREAMS[req_id] = {"prompt": None, "completion": None,
                                   "est_prompt": 1234, "stream": stream, "buf": ""}


def _feed(req_id, *chunks):
    for c in chunks:
        proxy._live_update_from_chunk(req_id, c)
    return (proxy._LIVE_STREAMS[req_id].get("text") or "")


def _sse(text):
    return 'data: {"choices":[{"delta":{"content":"%s"}}]}\n' % text


def test_a_split_sse_line_is_not_lost():
    """The socket splits mid-event constantly; each half alone parses as nothing."""
    _seed("live-split", stream=True)
    whole = _sse("hello world")
    half = len(whole) // 2
    assert _feed("live-split", whole[:half], whole[half:]) == "hello world"


def test_a_line_split_three_ways_still_arrives():
    _seed("live-thirds", stream=True)
    w = _sse("abc")
    assert _feed("live-thirds", w[:10], w[10:20], w[20:]) == "abc"


def test_chunks_that_land_on_boundaries_still_work():
    """The common case must not regress while fixing the split case."""
    _seed("live-clean", stream=True)
    assert _feed("live-clean", _sse("one"), _sse("two")) == "onetwo"


def test_a_trailing_partial_line_is_held_not_parsed():
    _seed("live-partial", stream=True)
    whole = _sse("later")
    assert _feed("live-partial", whole[:12]) == ""          # nothing complete yet
    assert proxy._LIVE_STREAMS["live-partial"]["buf"]        # ...but it was kept
    assert _feed("live-partial", whole[12:]) == "later"


def test_a_stream_with_no_newlines_cannot_grow_the_buffer_forever():
    _seed("live-flood", stream=True)
    proxy._live_update_from_chunk("live-flood", "x" * 70000)
    assert len(proxy._LIVE_STREAMS["live-flood"]["buf"]) <= 65536


def test_live_reports_whether_output_is_even_coming(client):
    """active=True with empty text is indistinguishable from a broken pane unless the
    endpoint says the request was never going to stream.

    No row is inserted: the endpoint already treats a missing row as in-flight, and writing
    one here contends with the app's background writers for the full db() timeout."""
    _seed("live-nonstream", stream=False)

    body = client.get("/__proxy/api/requests/live-nonstream/live").json()
    assert body["active"] is True
    assert body["streaming"] is False, "the UI needs this to explain the empty pane"
    assert body["est_prompt_tokens"] == 1234, "something truthful to show meanwhile"
    assert body["text"] == "", "and there is genuinely nothing to show"


def test_a_streaming_request_still_says_so(client):
    _seed("live-stream", stream=True)
    _feed("live-stream", _sse("partial"))

    body = client.get("/__proxy/api/requests/live-stream/live").json()
    assert body["streaming"] is True
    assert body["text"] == "partial"


def test_seeding_records_the_stream_flag():
    """_save_pending is where the flag is known; the live state is useless without it."""
    import inspect
    src = inspect.getsource(proxy._save_pending)
    assert '"stream": bool(is_stream)' in src


# --- pending requests, before the upstream has said anything --------------------------------


def _pending(req_id, *, ts=None, upstream="ollama"):
    """An entry in the shape the proxy now registers BEFORE client.send() returns: no
    upstream_resp yet, because the upstream has not produced response headers."""
    import time
    proxy._INFLIGHT_REQUESTS[req_id] = {"ts": ts or time.time(), "upstream_resp": None,
                                        "upstream": upstream, "cancelled": False}


def _row(req_id, ts_offset=-300.0):
    """A saved request row older than the live query's 32-second recency window, so the tile
    can only come from the in-flight registry."""
    import time
    conn = proxy.db()
    conn.execute("INSERT INTO requests (id, ts, method, path, upstream_url, model, is_stream, "
                 "client_ip, client_app, upstream) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (req_id, time.time() + ts_offset, "POST", "/v1/chat/completions",
                  "http://localhost:11434/v1/chat/completions", "nemotron", 1,
                  "127.0.0.1", "needle-test", "ollama"))
    conn.commit()
    conn.close()


def _cleanup(req_id):
    proxy._INFLIGHT_REQUESTS.pop(req_id, None)
    conn = proxy.db()
    conn.execute("DELETE FROM requests WHERE id=?", (req_id,))
    conn.commit()
    conn.close()


def test_a_request_still_waiting_on_the_upstream_shows_in_the_live_view(client):
    """The whole bug: a 300k-token prefill takes minutes during which client.send() has not
    returned. Registered only afterwards, the tile appeared only once generation began — so
    the Live view was blind to exactly the requests worth watching."""
    _row("live-pending")
    _pending("live-pending")
    try:
        tiles = client.get("/__proxy/api/live").json()["tiles"]
        mine = [t for t in tiles if t.get("req_id") == "live-pending"]
        assert mine, f"a pending request is missing from the live view: {tiles}"
        assert not mine[0]["done"], "it is in flight, not finished"
        assert mine[0]["state"] == "THINKING", "no tokens have come back yet"
    finally:
        _cleanup("live-pending")


def test_the_tile_reports_how_long_it_has_been_waiting(client):
    """A prefill that has been running for four minutes must say four minutes, not zero —
    the elapsed clock is the only thing that distinguishes 'slow' from 'wedged'."""
    import time
    _row("live-slow")
    _pending("live-slow", ts=time.time() - 240)
    try:
        t = [x for x in client.get("/__proxy/api/live").json()["tiles"]
             if x.get("req_id") == "live-slow"][0]
        assert t["elapsed_ms"] > 200_000, t["elapsed_ms"]
    finally:
        _cleanup("live-slow")


def test_a_pending_request_can_be_killed_rather_than_reported_missing(client):
    """It used to answer 404 'not in-flight' for a request that was very much in flight —
    just not far enough along to have a socket to close."""
    _row("live-kill")
    _pending("live-kill")
    try:
        r = client.post("/__proxy/api/control/cancel/live-kill")
        assert r.status_code == 200, r.text
        assert proxy._INFLIGHT_REQUESTS["live-kill"]["cancelled"] is True
    finally:
        _cleanup("live-kill")


def test_the_inflight_list_counts_a_pending_request(client):
    _row("live-count")
    _pending("live-count")
    try:
        ids = [i["id"] for i in client.get("/__proxy/api/control/inflight").json()["items"]]
        assert "live-count" in ids
    finally:
        _cleanup("live-count")


def test_the_request_is_registered_before_the_upstream_answers(client):
    """The defect itself, at the seam. Registration used to happen on the line AFTER
    client.send() returned, and send(stream=True) does not return until the upstream produces
    response headers. So this asserts from inside send(): by the time the proxy is waiting on
    the upstream, the request must already be visible as in-flight.

    Everything above this test seeds the registry by hand and would have passed against the
    broken code.
    """
    import httpx

    seen = {}
    real = proxy.app.state.client

    class _WatchingClient:
        def build_request(self, *a, **k):
            return real.build_request(*a, **k)

        async def send(self, req, **k):
            # This is the moment a long prefill sits in. What does the live view show now?
            seen["registry"] = dict(proxy._INFLIGHT_REQUESTS)
            # The view function directly, not over HTTP: a re-entrant TestClient request while
            # this one is still being handled does not resolve.
            stub = type("R", (), {"headers": {}, "client": type("C", (), {"host": "127.0.0.1"})()})()
            seen["tiles"] = proxy.live_view(stub)["tiles"]
            raise httpx.ConnectError("stop here — the answer is already decided")

        def __getattr__(self, name):
            return getattr(real, name)

    proxy.app.state.client = _WatchingClient()
    try:
        client.post("/v1/chat/completions",
                    json={"model": "nemotron", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"x-client-name": "pending-probe"})
    finally:
        proxy.app.state.client = real

    assert seen.get("registry"), \
        "nothing was in-flight while the proxy was blocked waiting on the upstream"
    rid = next(iter(seen["registry"]))
    assert seen["registry"][rid]["upstream_resp"] is None, \
        "there is no response yet — that is the entire point of registering early"
    assert any(t.get("req_id") == rid and not t.get("done") for t in seen["tiles"]), \
        f"the live view did not show the pending request: {seen['tiles']}"

    # And the failed send must not leave the entry behind to haunt the registry.
    assert rid not in proxy._INFLIGHT_REQUESTS, "a failed send leaked its in-flight entry"
