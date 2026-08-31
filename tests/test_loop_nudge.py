"""Nudge a repeating model before blocking it.

Blocking outright ends the turn and hands the problem back to the human. A model that repeats
itself has usually just failed to notice, and the only thing that can change its behaviour is a
change in its context — which a note provides and a 400 does not.

Observed on this box: gemma4 called Edit with byte-identical arguments six times. The edit failed
with "String to replace not found", the model tried to recover by re-reading the file, and Claude
Code's own dedup answered "Wasted call — file unchanged since your last Read". Nothing in its
context ever changed, so it re-issued the same call forever — 745 messages, ~600 tokens added per
cycle, ~31s of GPU each.

Escalation, not leniency: the nudge fires below max_repeats, the block at or above it. And
because the signature is name+arguments, a model that actually changes its call resets the count
— only another byte-identical repeat is blocked.
"""
from ai_proxy import proxy as P

CFG = {"enabled": True, "action": "block", "max_repeats": 4,
       "window": 10, "tail_consecutive": 3, "nudge_at": 3}


def _call(name, args):
    return {"role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def _body(n, name="Edit", args='{"file_path":"a.cs","old_string":"x"}'):
    """One user turn, then n identical assistant tool calls."""
    return {"messages": [{"role": "user", "content": "go"}] + [_call(name, args) for _ in range(n)]}


def _verdict(n, **over):
    cfg = dict(CFG, **over)
    return P._rule_loop_detector(_body(n), cfg)


# --- the two tiers ---------------------------------------------------------------------------


def test_two_repeats_is_not_a_loop_yet():
    assert _verdict(2) is None


def test_the_third_repeat_nudges_rather_than_blocks():
    r = _verdict(3)
    assert r is not None
    assert r["details"]["nudge"] is True
    assert r["details"]["soft_unblock"] is True, "a nudge must not block the request"


def test_the_nudge_tells_it_what_to_do_instead():
    note = _verdict(3)["details"]["note"]
    assert "Edit" in note
    assert "identical" in note.lower()
    assert "blocked" in note.lower(), "the escalation has to be stated or it is not a warning"


def test_repeating_after_the_nudge_blocks():
    r = _verdict(4)
    assert r is not None
    assert not (r["details"] or {}).get("soft_unblock"), "this one must actually block"
    assert "4" in r["reason"]


def test_a_soft_unblock_is_allowed_through_as_a_warn(client, monkeypatch):
    """The verdict layer is what makes a nudge non-blocking; the rule only flags it."""
    cfg = dict(P.load_rules_config())
    cfg["loop_detector"] = CFG
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    assert P.evaluate_rules(_body(3))["verdict"] == "warn"
    assert P.evaluate_rules(_body(4))["verdict"] == "block"


# --- what must NOT trigger it ------------------------------------------------------------------


def test_changing_the_arguments_resets_the_count():
    """The escape route. A model that adapts is not looping, and must not be punished for
    having repeated an earlier call before it adapted."""
    msgs = [{"role": "user", "content": "go"}]
    for i in range(6):
        msgs.append(_call("Edit", '{"file_path":"a.cs","old_string":"v%d"}' % i))
    assert P._rule_loop_detector({"messages": msgs}, CFG) is None


def test_different_tools_are_not_a_loop():
    msgs = [{"role": "user", "content": "go"}]
    for name in ("Read", "Edit", "Bash", "Read", "Edit", "Bash"):
        msgs.append(_call(name, "{}"))
    assert P._rule_loop_detector({"messages": msgs}, CFG) is None


def test_a_new_user_turn_resets_the_window():
    """The user intervening is the strongest possible signal that the situation changed."""
    msgs = [{"role": "user", "content": "go"}] + [_call("Edit", "{}") for _ in range(5)]
    msgs.append({"role": "user", "content": "stop, try something else"})
    msgs.append(_call("Edit", "{}"))
    r = P._rule_loop_detector({"messages": msgs}, CFG)
    # A loop before the last user turn is the pre-existing soft-unblock path, never a block.
    assert r is None or (r["details"] or {}).get("soft_unblock")


def test_nudging_can_be_switched_off_to_get_the_old_behaviour():
    r = _verdict(3, nudge_at=0)
    assert r is None or not (r["details"] or {}).get("nudge")
