"""The Anthropic thinking block, translated for the OpenAI side.

The bridge dropped it silently. Measured on gemma4 through /v1/messages: time to the first
character of the answer was 15.80s with thinking explicitly disabled, against 0.97s once the
translated request actually carried reasoning_effort=none. Every client on that endpoint —
Claude Code, the Anthropic SDK, opencode — had no working way to turn thinking off.
"""
from ai_proxy import proxy as P


def _t(body):
    return P._anthropic_to_openai_request({"model": "m", "messages": [], **body})


def test_disabled_becomes_effort_none():
    assert _t({"thinking": {"type": "disabled"}})["reasoning_effort"] == "none"


def test_a_budget_maps_onto_a_level():
    """budget_tokens is the only dial Anthropic gives; the OpenAI side understands levels."""
    assert _t({"thinking": {"type": "enabled", "budget_tokens": 12000}})["reasoning_effort"] == "high"
    assert _t({"thinking": {"type": "enabled", "budget_tokens": 3000}})["reasoning_effort"] == "medium"
    assert _t({"thinking": {"type": "enabled", "budget_tokens": 500}})["reasoning_effort"] == "low"


def test_enabled_without_a_budget_still_asks_for_thinking():
    assert _t({"thinking": {"type": "enabled"}})["reasoning_effort"] == "low"


def test_a_request_that_says_nothing_stays_silent():
    """Most traffic. Inventing an effort here would override the model's own default and the
    proxy's per-model quirks, which is a decision the client did not make."""
    assert "reasoning_effort" not in _t({})


def test_a_malformed_thinking_block_is_ignored_rather_than_raising():
    for bad in ("disabled", 3, [], {"type": "nonsense"}):
        out = _t({"thinking": bad})
        assert isinstance(out, dict)
        if bad == {"type": "nonsense"}:
            assert "reasoning_effort" not in out


def test_the_rest_of_the_translation_is_untouched():
    out = _t({"thinking": {"type": "disabled"}, "temperature": 0.5, "stream": True,
              "stop_sequences": ["X"]})
    assert out["temperature"] == 0.5 and out["stop"] == ["X"]
    assert out["stream_options"] == {"include_usage": True}
