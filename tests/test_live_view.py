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
