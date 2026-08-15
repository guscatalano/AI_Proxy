"""Why a task failed, not just that it did.

A score says a model lost 35 of 119. It cannot say whether it was wrong, silent, cut off or
unwilling — and those need four different responses: a better model, a bigger token budget,
a different prompt, or a policy decision. Every signal below was already being recorded by
the graders and discarded at render time.

Classification is first-match-wins, so a truncated answer counts as truncated rather than as
the wrong answer it necessarily also was — the root cause is the actionable one.
"""
from ai_proxy import bench_report as R


def row(text="def f(): pass", tokens=100, error=None):
    return {"text": text, "completion_tokens": tokens, "error": error}


def test_a_backend_error_is_not_counted_as_a_wrong_answer():
    assert R._bench_failure_reason(row(error="HTTP 400"), {"passed": 0, "total": 1}) == "backend"


def test_an_empty_answer_is_a_backend_failure_not_a_wrong_one():
    """A reasoning model that spent its whole budget thinking returns nothing to grade."""
    assert R._bench_failure_reason(row(text="   "), {"passed": 0, "total": 1}) == "backend"


def test_code_that_did_not_compile_is_distinguished_from_code_that_was_wrong():
    g = {"build": False, "error": "gcc: expected ';'", "passed": 0, "total": 6}
    assert R._bench_failure_reason(row(), g) == "build"


def test_hitting_the_token_ceiling_is_reported_as_truncation():
    g = {"cases": [{"ok": False}], "passed": 0, "total": 3, "truncated": True}
    assert R._bench_failure_reason(row(), g) == "truncated"


def test_truncation_is_inferred_from_the_budget_when_not_flagged():
    """46% of one nemotron cell hit the ceiling; nothing in the report said so."""
    g = {"cases": [{"ok": False}], "passed": 0, "total": 3}
    assert R._bench_failure_reason(row(tokens=2048), g, max_tokens=2048) == "truncated"
    assert R._bench_failure_reason(row(tokens=300), g, max_tokens=2048) == "wrong"


def test_truncation_outranks_wrongness():
    """An answer cut off mid-sentence is also wrong; the budget is the actionable cause."""
    g = {"cases": [{"ok": False, "got": "expected 7, got 3"}], "passed": 0, "total": 1,
         "truncated": True}
    assert R._bench_failure_reason(row(), g) == "truncated"


def test_agent_episode_signals_are_separated():
    base = {"cases": [{"ok": False}], "passed": 0, "total": 2}
    assert R._bench_failure_reason(row(), {**base, "exhausted": True}) == "exhausted"
    assert R._bench_failure_reason(row(), {**base, "malformed": 2}) == "malformed"
    assert R._bench_failure_reason(row(), {**base, "repeats": 4}) == "looped"


def test_over_refusal_is_its_own_category():
    """A policy problem, not a capability one — it needs a different fix entirely."""
    g = {"cases": [{"ok": False, "got": "declined a request it should have answered"}],
         "passed": 0, "total": 1}
    assert R._bench_failure_reason(row(), g) == "refused"


def test_prose_where_code_was_wanted_is_not_a_wrong_answer():
    g = {"cases": [{"ok": False, "got": "no code in any language could be identified"}],
         "passed": 0, "total": 1}
    assert R._bench_failure_reason(row(), g) == "nocode"


def test_an_ordinary_wrong_answer_falls_through():
    g = {"cases": [{"ok": False, "got": "27 words, limit 25"}], "passed": 0, "total": 2}
    assert R._bench_failure_reason(row(), g) == "wrong"


# --- the rendered table ------------------------------------------------------------------


def _run(model, grades, max_tokens=2048):
    return {"id": "r-" + model, "model": model, "ts": 1786600000,
            "label": "%s · short · think=off" % model,
            "config": {"suite": "full-v2", "max_tokens": max_tokens},
            "results": {"rows": [{"task": "t%d" % i, "text": g.pop("_text", "code"),
                                  "completion_tokens": g.pop("_tok", 100),
                                  "error": g.pop("_err", None), "grade": g}
                                 for i, g in enumerate(grades)]}}


def test_the_table_counts_reasons_per_model():
    runs = [_run("alpha", [
        {"cases": [{"ok": False}], "passed": 0, "total": 1, "truncated": True},
        {"cases": [{"ok": False}], "passed": 0, "total": 1, "truncated": True},
        {"cases": [{"ok": False, "got": "declined a request it should have answered"}],
         "passed": 0, "total": 1},
        {"cases": [{"ok": True}], "passed": 1, "total": 1},          # a pass: not counted
    ])]
    html = R._bench_failure_taxonomy_html(runs, [{}], ["alpha · short · think=off"])
    assert "Why it failed" in html
    assert "ran out of tokens" in html and "declined to engage" in html
    assert "total failures" in html


def test_a_flawless_run_renders_nothing():
    runs = [_run("alpha", [{"cases": [{"ok": True}], "passed": 1, "total": 1}])]
    assert R._bench_failure_taxonomy_html(runs, [{}], ["alpha · short"]) == ""


def test_reasons_nobody_hit_are_not_shown():
    runs = [_run("alpha", [{"cases": [{"ok": False}], "passed": 0, "total": 1}])]
    html = R._bench_failure_taxonomy_html(runs, [{}], ["alpha · short"])
    assert "wrong answer" in html
    assert "malformed tool calls" not in html, "empty rows are noise"
