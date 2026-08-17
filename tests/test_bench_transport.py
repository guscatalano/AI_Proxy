"""The suite that would have caught the Ollama tool-call loss.

Ollama's /v1 endpoint drops tool calls from streamed replies — about 2.5% of claude-code turns
on this box arrived empty and read as the agent giving up. The benchmark suite scored every
model as healthy throughout, because it had no path that offered tools over a stream: single-turn
tasks stream without tools, agent episodes carry tools with stream:False. transport-v1 is that
missing combination, and these tests hold the wiring in place, since the failure of a test that
never runs is indistinguishable from a pass.
"""
import asyncio
import json

from ai_proxy import proxy as P
from ai_proxy.bench_graders import _bench_lang_available


# --- the wiring, which is where this kind of suite dies quietly ---------------------------


def test_the_suite_is_registered():
    assert "transport-v1" in P._BENCH_SUITES
    assert len(P._BENCH_SUITES["transport-v1"]) >= 8, "too few attempts to see a ~2.5% loss"


def test_its_grading_mode_is_not_silently_skipped():
    """A task whose lang is missing from the gate is DROPPED, not failed — two whole suites
    once vanished from a full run exactly this way, reported only as skipped_languages."""
    assert _bench_lang_available("toolstream") is True


def test_full_v2_composition_is_unchanged():
    """Deliberately not folded in: full-v2 is the axis every cross-model comparison here is
    expressed in, and changing its composition would make new totals quietly incomparable
    with every run already recorded."""
    assert len(P._BENCH_SUITES["full-v2"]) == 119


def test_every_task_offers_tools_and_asks_to_stream():
    for t in P._BENCH_SUITES["transport-v1"]:
        assert t.get("stream_tools") is True, t["id"]
        assert t.get("tools"), t["id"]
        assert t.get("prompt"), t["id"]


def test_task_ids_are_unique():
    ids = [t["id"] for t in P._BENCH_SUITES["transport-v1"]]
    assert len(ids) == len(set(ids))


# --- the request that actually goes out ----------------------------------------------------


def test_tools_reach_the_request_body_and_it_still_streams():
    task = P._BENCH_SUITES["transport-v1"][0]
    body = P._bench_build_body("m", "p", 256, {}, task["tools"])
    assert body["tools"] == task["tools"]
    assert body["stream"] is True, "a non-streaming request cannot see this bug at all"


def test_an_ordinary_task_still_sends_no_tools():
    assert "tools" not in P._bench_build_body("m", "p", 256, {})


# --- reading the stream --------------------------------------------------------------------


def _sse(*objs):
    return "".join("data: %s\n\n" % json.dumps(o) for o in objs) + "data: [DONE]\n\n"


class _FakeStream:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for line in self.text.splitlines():
            yield line

    async def aread(self):
        return self.text.encode()


class _FakeClient:
    def __init__(self, text, status=200):
        self.text, self.status = text, status

    def stream(self, *a, **k):
        return _FakeStream(self.text, self.status)


def _run(text, status=200):
    """asyncio.run rather than pytest.mark.asyncio: this repo has no pytest-asyncio, and an
    unregistered marker makes the test silently not run at all."""
    return asyncio.run(P._bench_run_one(_FakeClient(text, status), "http://x", "m", 64,
                                        "prompt", 1, cfg={}, tools=[{"type": "function"}]))


def test_a_delivered_tool_call_is_counted():
    row = _run(_sse(
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\":"}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "\"/etc/hosts\"}"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ))
    assert row["tool_calls"] == 1, "argument deltas carry no name and must not count again"
    assert not row["error"], "a tool call is output; this is not an empty completion"


def test_the_production_failure_is_counted_as_a_loss():
    """Verbatim shape of the Ollama failure: empty content, a clean stop, no call."""
    row = _run(_sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ))
    assert row["tool_calls"] == 0


def test_two_distinct_calls_are_both_counted():
    row = _run(_sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "read_file", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "function": {"name": "search", "arguments": "{}"}}]}}]},
    ))
    assert row["tool_calls"] == 2


def test_prose_without_a_call_is_a_loss_not_a_pass():
    """The model answering in words when a tool was the point still delivered no call."""
    row = _run(_sse(
        {"choices": [{"delta": {"content": "I would read the file for you."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ))
    assert row["tool_calls"] == 0
