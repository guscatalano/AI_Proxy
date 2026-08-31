"""Reasoning reaches the client as a thinking block instead of vanishing.

The Anthropic translator built exactly two block types — `text` from `content` and `tool_use`
from `tool_calls`. There was no branch for `reasoning` at all, in any of the three paths, so a
model that answered in its reasoning field produced a well-formed EMPTY turn.

Measured in production before the fix: a claude-code turn generated 1,952 tokens over 100
seconds, of which 4,888 characters were reasoning and zero were content. The client waited a
minute and a half and got nothing, which reads as the model giving up for no reason. Anthropic
has a `thinking` block type; nobody had wired it.
"""
import json

from ai_proxy import proxy as P


def _sse(*objs):
    return "".join("data: %s\n\n" % json.dumps(o) for o in objs) + "data: [DONE]\n\n"


def _events(blob):
    out = []
    for line in (blob.decode() if isinstance(blob, bytes) else blob).splitlines():
        if line.startswith("data: "):
            try:
                out.append(json.loads(line[6:]))
            except ValueError:
                pass
    return out


def _blocks(evts):
    """Block types in the order the client would see them open."""
    return [e.get("content_block", {}).get("type")
            for e in evts if e.get("type") == "content_block_start"]


def _thinking_text(evts):
    return "".join(e["delta"].get("thinking", "")
                   for e in evts if e.get("type") == "content_block_delta"
                   and (e.get("delta") or {}).get("type") == "thinking_delta")


# --- non-streaming ---------------------------------------------------------------------------


def test_a_reasoning_only_reply_is_no_longer_empty(client):
    """The production failure: all reasoning, no content. It must not translate to nothing."""
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "gemma4:26b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "", "reasoning": "Rayleigh scattering, because..."}}],
    })
    assert [b["type"] for b in out["content"]] == ["thinking"]
    assert "Rayleigh" in out["content"][0]["thinking"]


def test_thinking_comes_before_the_answer(client):
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Blue.", "reasoning": "Let me think."}}],
    })
    assert [b["type"] for b in out["content"]] == ["thinking", "text"]


def test_a_reply_without_reasoning_is_unchanged(client):
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Blue."}}],
    })
    assert [b["type"] for b in out["content"]] == ["text"]


def test_whitespace_reasoning_does_not_open_a_block(client):
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Blue.", "reasoning": "   \n  "}}],
    })
    assert [b["type"] for b in out["content"]] == ["text"]


def test_the_openai_alternate_field_name_is_read(client):
    """Engines disagree: some send `reasoning`, others `reasoning_content`."""
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "", "reasoning_content": "thought"}}],
    })
    assert out["content"][0]["thinking"] == "thought"


# --- streaming, incremental: the path claude-code actually uses ------------------------------


def _feed(chunks):
    b = P.IncrementalAnthropicBridge("claude-sonnet-4")
    out = b.feed(_sse(*chunks).encode())
    tail = b.flush()
    return _events((out or b"") + (tail or b""))


def test_streamed_reasoning_becomes_thinking_deltas(client):
    evts = _feed([
        {"choices": [{"index": 0, "delta": {"reasoning": "First I "}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "consider."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    assert "thinking" in _blocks(evts)
    assert _thinking_text(evts) == "First I consider."


def test_streamed_reasoning_then_content_opens_two_blocks(client):
    evts = _feed([
        {"choices": [{"index": 0, "delta": {"reasoning": "hmm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Blue."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    assert _blocks(evts) == ["thinking", "text"]


def test_the_thinking_block_is_closed_before_the_text_block(client):
    """Anthropic's protocol requires strict open/close nesting; a client that sees a second
    content_block_start without a stop will reject the stream."""
    evts = _feed([
        {"choices": [{"index": 0, "delta": {"reasoning": "hmm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Blue."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    kinds = [e.get("type") for e in evts
             if e.get("type") in ("content_block_start", "content_block_stop")]
    assert kinds == ["content_block_start", "content_block_stop",
                     "content_block_start", "content_block_stop"]


def test_a_stream_without_reasoning_is_unchanged(client):
    evts = _feed([
        {"choices": [{"index": 0, "delta": {"content": "Blue."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    assert _blocks(evts) == ["text"]


def test_reasoning_before_a_tool_call_keeps_both(client):
    evts = _feed([
        {"choices": [{"index": 0, "delta": {"reasoning": "I should read it"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": "{}"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ])
    assert _blocks(evts) == ["thinking", "tool_use"]


# --- streaming, buffered: used when a post-flight transform needed the whole stream ----------


def test_the_buffered_translator_also_emits_thinking(client):
    evts = _events(P._openai_sse_to_anthropic_events(_sse(
        {"choices": [{"index": 0, "delta": {"reasoning": "because scattering"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Blue."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ), fallback_model="m"))
    assert _blocks(evts) == ["thinking", "text"]
    assert _thinking_text(evts) == "because scattering"


def test_the_buffered_translator_is_unchanged_without_reasoning(client):
    evts = _events(P._openai_sse_to_anthropic_events(_sse(
        {"choices": [{"index": 0, "delta": {"content": "Blue."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ), fallback_model="m"))
    assert _blocks(evts) == ["text"]


# --- the way back out ------------------------------------------------------------------------


def test_translation_can_be_switched_off_without_a_deploy(client, monkeypatch):
    """It changes which block types a client sees, so a client that cannot cope needs a lever."""
    cfg = dict(P.load_rules_config())
    cfg["protocol_bridge"] = {"enabled": True, "translate_reasoning": False}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    out = P._openai_to_anthropic_response({
        "id": "c1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Blue.", "reasoning": "hmm"}}],
    })
    assert [b["type"] for b in out["content"]] == ["text"]
