"""A nudge must not cost more than the problem it warns about.

The compaction reminder appended to the FIRST system message, at the very front of the prompt.
Because it fires on every request once a conversation is long, every request presented a
different prefix, the engine's prefix cache missed, and each turn paid a full prefill. Measured
on this box the hour it was switched on:

    large-context turns   before: p50 8s, 70% under 30s
                          after:  p50 235s, 0% under 30s

235s is past the 180s Claude Code waits before hanging up, so the nudge intended to make long
conversations cheaper instead made every long turn time out. Sixteen of twenty-two disconnects
that hour landed at 180.5-180.9s.

Two properties keep it honest: the prefix is never touched, and the text does not change between
firings — an exact token count would re-prefill the tail on every turn.
"""
from ai_proxy import proxy as P


def _conv():
    return {"messages": [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]}


# --- the cache property ----------------------------------------------------------------------


def test_the_prefix_is_left_byte_identical():
    """Everything before the tail must be untouched, or the whole transcript re-prefills."""
    b = _conv()
    before = [dict(m) for m in b["messages"][:-1]]
    assert P._inject_trailing_reminder(b, "NUDGE") is True
    assert b["messages"][:-1] == before


def test_it_lands_on_the_last_user_message():
    b = _conv()
    P._inject_trailing_reminder(b, "NUDGE")
    assert "NUDGE" in b["messages"][-1]["content"]
    assert b["messages"][-1]["role"] == "user"


def test_repeating_it_does_not_stack():
    """It fires on every request once a conversation is long; stacking would grow the tail
    without bound and change it every time."""
    b = _conv()
    assert P._inject_trailing_reminder(b, "NUDGE") is True
    assert P._inject_trailing_reminder(b, "NUDGE") is False
    assert b["messages"][-1]["content"].count("NUDGE") == 1


def test_a_list_content_message_is_appended_to():
    b = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert P._inject_trailing_reminder(b, "NUDGE") is True
    assert b["messages"][-1]["content"][-1]["text"] == "NUDGE"


def test_a_conversation_with_no_user_turn_still_gets_it():
    b = {"messages": [{"role": "system", "content": "S"}]}
    assert P._inject_trailing_reminder(b, "NUDGE") is True
    assert b["messages"][-1]["content"] == "NUDGE"


# --- the stability property ---------------------------------------------------------------------


def test_nearby_percentages_produce_identical_text():
    """An exact figure changed on every request, so even a tail append re-prefilled each turn."""
    assert P._pct_bucket(91) == P._pct_bucket(94) == "90%"


def test_buckets_still_move_across_a_conversation():
    assert P._pct_bucket(71) != P._pct_bucket(91)


def test_the_rule_text_carries_no_exact_token_count(client, monkeypatch):
    cfg = dict(P.load_rules_config())
    cfg["compaction_nudge"] = {"enabled": True, "threshold_pct": 1, "chars_per_token": 3.5,
                               "assumed_default_num_ctx": 1000,
                               "default_strategy": "system_reminder_plain",
                               "client_strategies": {}}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    v = P.evaluate_compaction_nudge({"messages": [{"role": "user", "content": "x" * 4000}]},
                                    {"path": "/v1/chat/completions", "client_app": "any"})
    assert v and v["action"] == "system_reminder"
    assert str(v["estimated_tokens"]) not in v["text"], \
        "an exact token count makes the reminder unique per request"
