"""Thinking is recorded on the row, not re-derived from the body.

It used to be knowable only by parsing the stored request body in the browser, which meant
nothing could count or filter by it — and the stored body is the client's original, captured
before the model quirks run, so it could not show a decision the proxy made itself. Measured on
the box before this landed: hermes sent reasoning_effort=none on 256 of 258 chat requests and
opencode sent nothing at all, and neither fact was visible anywhere in the UI.

Six dialects say "think" or "don't", and clients here use most of them.
"""
import json

from ai_proxy import proxy as P


class _StubRequest:
    """Enough Request for the write path. The real endpoint would reach for an upstream that
    is not running in CI; what is under test is what lands in the row."""

    def __init__(self, headers=None):
        self.headers = headers or {"content-type": "application/json"}
        self.method = "POST"
        self.client = type("C", (), {"host": "10.1.2.3"})()


def _intent(body):
    return P._thinking_intent(body)


# --- the dialects ------------------------------------------------------------------------


def test_openai_reasoning_effort_is_recorded_with_its_level():
    assert _intent({"reasoning_effort": "high"}) == ("high", "on", "reasoning_effort")
    assert _intent({"reasoning_effort": "medium"}) == ("medium", "on", "reasoning_effort")


def test_effort_none_is_thinking_off_not_a_missing_value():
    """hermes sends this on nearly every request; reading it as 'unset' would have made the
    single most common signal on this proxy invisible."""
    assert _intent({"reasoning_effort": "none"}) == (None, "off", "reasoning_effort")


def test_minimal_is_folded_into_low():
    assert _intent({"reasoning_effort": "minimal"})[0] == "low"


def test_the_nested_reasoning_object_is_read_too():
    assert _intent({"reasoning": {"effort": "low"}}) == ("low", "on", "reasoning_effort")


def test_enable_thinking_is_read_at_both_depths():
    assert _intent({"enable_thinking": False})[1] == "off"
    assert _intent({"chat_template_kwargs": {"enable_thinking": True}}) == (None, "on",
                                                                           "enable_thinking")


def test_ollamas_native_think_field_including_its_level_form():
    assert _intent({"think": False})[1] == "off"
    assert _intent({"think": "high"}) == ("high", "on", "think")


def test_the_anthropic_thinking_block_maps_budget_to_a_level():
    assert _intent({"thinking": {"type": "disabled"}})[1] == "off"
    assert _intent({"thinking": {"type": "enabled", "budget_tokens": 12000}})[0] == "high"
    assert _intent({"thinking": {"type": "enabled", "budget_tokens": 3000}})[0] == "medium"
    assert _intent({"thinking": {"type": "enabled", "budget_tokens": 500}})[0] == "low"


def test_qwens_in_band_switches_are_read_out_of_the_prompt():
    """/no_think never appears in the body's parameters — it is inside the user's text."""
    body = {"messages": [{"role": "user", "content": "count to ten /no_think"}]}
    assert _intent(body) == (None, "off", "/no_think")
    body = {"messages": [{"role": "user", "content": "/think about this one"}]}
    assert _intent(body)[1] == "on"


def test_the_switch_is_read_from_the_last_user_turn_not_the_first():
    body = {"messages": [
        {"role": "user", "content": "first /think"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now be quick /no_think"},
    ]}
    assert _intent(body)[1] == "off"


def test_multipart_user_content_is_flattened_before_looking():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe this /no_think"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    assert _intent(body)[1] == "off"


def test_a_body_that_says_nothing_records_nothing():
    """Most traffic. Recording a guess here would make 'off' unfalsifiable."""
    assert _intent({"model": "x", "messages": [{"role": "user", "content": "hi"}]}) == (
        None, None, None)


def test_a_non_dict_body_does_not_raise():
    assert _intent(None) == (None, None, None)
    assert _intent("not json") == (None, None, None)


# --- precedence --------------------------------------------------------------------------


def test_an_explicit_enable_thinking_beats_reasoning_effort():
    """Both present means the template read enable_thinking; the effort was decoration."""
    eff, mode, src = _intent({"reasoning_effort": "high",
                              "chat_template_kwargs": {"enable_thinking": False}})
    assert mode == "off" and src == "enable_thinking"
    assert eff == "high", "what the client asked for is still worth keeping"


def test_the_prompt_switch_beats_the_body():
    body = {"reasoning_effort": "high",
            "messages": [{"role": "user", "content": "go /no_think"}]}
    assert _intent(body)[1] == "off"


# --- what lands in the row ----------------------------------------------------------------


def _row(req_id):
    conn = P.db()
    r = conn.execute("SELECT reasoning_effort, thinking, thinking_src FROM requests WHERE id=?",
                     (req_id,)).fetchone()
    conn.execute("DELETE FROM requests WHERE id=?", (req_id,))
    conn.commit()
    conn.close()
    return dict(r) if r else None


def _save(req_id, body):
    P._save_pending(req_id, _StubRequest(), "v1/chat/completions",
                    "http://localhost:11434/v1/chat/completions",
                    json.dumps(body), body, body.get("model"), False, upstream="ollama")


def test_the_columns_are_written_at_request_time(client):
    _save("r_think_1", {"model": "qwen3:4b", "reasoning_effort": "high",
                        "messages": [{"role": "user", "content": "hi"}]})
    assert _row("r_think_1") == {"reasoning_effort": "high", "thinking": "on",
                                 "thinking_src": "reasoning_effort"}


def test_a_silent_request_leaves_the_columns_null(client):
    _save("r_think_2", {"model": "qwen3:4b", "messages": [{"role": "user", "content": "hi"}]})
    assert _row("r_think_2") == {"reasoning_effort": None, "thinking": None,
                                 "thinking_src": None}


def test_the_list_api_returns_them(client):
    _save("r_think_3", {"model": "qwen3:4b", "reasoning_effort": "none",
                        "messages": [{"role": "user", "content": "hi"}]})
    try:
        items = client.get("/__proxy/api/requests?limit=50").json()["items"]
        row = next((i for i in items if i["id"] == "r_think_3"), None)
        assert row is not None, "the request is not in the list at all"
        assert row["thinking"] == "off" and row["thinking_src"] == "reasoning_effort", \
            "the badge has nothing to render from"
    finally:
        _row("r_think_3")


def test_the_detail_api_returns_them(client):
    _save("r_think_4", {"model": "qwen3:4b", "think": "medium",
                        "messages": [{"role": "user", "content": "hi"}]})
    try:
        d = client.get("/__proxy/api/requests/r_think_4").json()
        assert d.get("reasoning_effort") == "medium" and d.get("thinking") == "on"
    finally:
        _row("r_think_4")


def test_a_quirk_decision_overwrites_the_row_and_says_so(client):
    """The one fact the stored body can never carry: the proxy switched thinking off for a
    model that ignores reasoning_effort, and the client never knew."""
    _save("r_think_5", {"model": "ornith", "messages": [{"role": "user", "content": "hi"}]})
    conn = P.db()
    conn.execute("UPDATE requests SET thinking=?, thinking_src=? WHERE id=?",
                 ("off", "quirk:force_off", "r_think_5"))
    conn.commit()
    conn.close()
    assert _row("r_think_5") == {"reasoning_effort": None, "thinking": "off",
                                 "thinking_src": "quirk:force_off"}
