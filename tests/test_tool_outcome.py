"""Join a tool call to the result it produced, one turn later.

A call and its outcome arrive in different requests: the model asks in turn N, the client
reports what happened in turn N+1. Nothing joined them, so the request list could show that Edit
had been called twelve times and never that all twelve failed with "String to replace not found"
— the reason for the loop was only visible by opening the next request's body and reading the
transcript by hand.

Matching is by tool_call_id, which is why ids are stored and not just names.
"""
import json

from ai_proxy import proxy as P


def _sse(*objs):
    return "".join("data: %s\n\n" % json.dumps(o) for o in objs) + "data: [DONE]\n\n"


# --- reading the call out of a response ------------------------------------------------------


def test_a_streamed_openai_call_yields_its_id():
    got = P._extract_tool_call_ids(None, _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "Edit", "arguments": "{"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "}"}}]}}]}))
    assert got == [{"id": "call_1", "name": "Edit"}], "argument deltas must not add a second call"


def test_a_streamed_anthropic_call_yields_its_id():
    got = P._extract_tool_call_ids(None, _sse(
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_9", "name": "Read"}}))
    assert got == [{"id": "toolu_9", "name": "Read"}]


def test_a_response_without_calls_yields_nothing():
    assert P._extract_tool_call_ids(None, _sse({"choices": [{"delta": {"content": "hi"}}]})) == []


# --- reading the result out of the next request ----------------------------------------------


def test_an_openai_error_result_is_classified():
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "Edit"}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "Error: String to replace not found in file."}]}
    r = P._tool_results_with_ids(body)
    assert r[0]["id"] == "call_1" and r[0]["name"] == "Edit"
    assert r[0]["ok"] is False and "String to replace" in r[0]["excerpt"]


def test_an_anthropic_is_error_flag_is_trusted():
    """Anthropic states it outright; that beats sniffing the text for error-ish words."""
    body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True,
             "content": "file went missing"}]}]}
    r = P._tool_results_with_ids(body)
    assert r[0]["ok"] is False and r[0]["name"] == "Read"


def test_a_successful_result_reads_as_ok():
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "127.0.0.1 localhost"}]}
    assert P._tool_results_with_ids(body)[0]["ok"] is True


# --- the join ---------------------------------------------------------------------------------


def _seed(rid, conv, calls):
    conn = P.db()
    conn.execute("INSERT INTO requests (id, ts, method, path, upstream_url, model, is_stream, "
                 "client_ip, client_app, conversation_id, tool_call_ids, tool_outcome) "
                 "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (rid, 1_800_000_000, "POST", "/v1/chat/completions", "http://x", "m", 1,
                  "127.0.0.1", "test", conv, json.dumps(calls), "pending"))
    conn.commit()
    conn.close()


def _read(rid):
    conn = P.db()
    r = conn.execute("SELECT tool_outcome, tool_outcome_detail FROM requests WHERE id=?",
                     (rid,)).fetchone()
    conn.execute("DELETE FROM requests WHERE id=?", (rid,))
    conn.commit()
    conn.close()
    return dict(r) if r else {}


def test_a_failed_result_marks_the_request_that_asked(client):
    _seed("to_err", "conv_a", [{"id": "call_1", "name": "Edit"}])
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "Edit"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Error: String to replace not found"}]}
    assert P._resolve_tool_outcomes("conv_a", body) == 1
    row = _read("to_err")
    assert row["tool_outcome"] == "error"
    assert "String to replace" in json.loads(row["tool_outcome_detail"])[0]["excerpt"]


def test_a_successful_result_marks_it_ok(client):
    _seed("to_ok", "conv_b", [{"id": "c9", "name": "Read"}])
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "c9", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "c9", "content": "all good"}]}
    P._resolve_tool_outcomes("conv_b", body)
    assert _read("to_ok")["tool_outcome"] == "ok"


def test_one_failure_among_several_is_partial(client):
    """Flattening this to "error" would hide which call actually broke."""
    _seed("to_part", "conv_c", [{"id": "a", "name": "Read"}, {"id": "b", "name": "Edit"}])
    body = {"messages": [
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "Read"}}, {"id": "b", "function": {"name": "Edit"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "fine"},
        {"role": "tool", "tool_call_id": "b", "content": "Error: nope"}]}
    P._resolve_tool_outcomes("conv_c", body)
    assert _read("to_part")["tool_outcome"] == "partial"


def test_an_unanswered_call_stays_pending(client):
    """The agent asked and walked away — worth seeing, so it must not be silently marked ok."""
    _seed("to_pend", "conv_d", [{"id": "never", "name": "Bash"}])
    body = {"messages": [{"role": "user", "content": "unrelated"}]}
    assert P._resolve_tool_outcomes("conv_d", body) == 0
    assert _read("to_pend")["tool_outcome"] == "pending"


def test_a_resolved_request_is_not_rewritten(client):
    """Later turns replay the whole transcript, so the same result appears again and again."""
    _seed("to_once", "conv_e", [{"id": "z", "name": "Read"}])
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "z", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "z", "content": "ok"}]}
    assert P._resolve_tool_outcomes("conv_e", body) == 1
    assert P._resolve_tool_outcomes("conv_e", body) == 0, "must not rewrite settled history"
    _read("to_once")


def test_another_conversation_is_untouched(client):
    _seed("to_other", "conv_f", [{"id": "shared", "name": "Read"}])
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "shared", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "shared", "content": "ok"}]}
    assert P._resolve_tool_outcomes("conv_ELSEWHERE", body) == 0
    assert _read("to_other")["tool_outcome"] == "pending"
