"""project-v1: build the thing the spec describes, not the thing the examples show.

The coding suite saturated — qwen3.8 scored 45/47 and gemma4 44/47, one task apart, which is
inside noise. These tasks exist to separate them, and the mechanism is that each spec prints a
few of its test cases and hides the rest. Passing a printed example proves only that a model
can copy it; the gap between visible and hidden is the measurement.
"""
import pytest

from ai_proxy import bench_graders as G
from ai_proxy import bench_project as P
from ai_proxy import proxy


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- the shape of a task ------------------------------------------------------------------


def test_every_task_shows_some_cases_and_hides_others():
    """A task with no visible cases cannot show the gap; one with no hidden cases measures
    nothing the examples did not already give away."""
    for t in P.PROJECT_TASKS:
        vis = [c for c in t["cases"] if c.get("visible")]
        hid = [c for c in t["cases"] if not c.get("visible")]
        assert vis, f"{t['id']} shows no examples"
        assert hid, f"{t['id']} hides nothing"


def test_the_entry_point_is_named_in_the_spec():
    """A model cannot implement a function whose name it was never told."""
    for t in P.PROJECT_TASKS:
        assert t["entry"] in t["prompt"], f"{t['id']} never names {t['entry']}"


def test_hidden_cases_carry_a_label_naming_the_requirement():
    """'rounds half UP, not to even' names the requirement that was skipped. A bare
    expected-versus-got does not."""
    for t in P.PROJECT_TASKS:
        for c in t["cases"]:
            if not c.get("visible"):
                assert c.get("label"), f"{t['id']} has an unlabelled hidden case: {c['args']}"


def test_the_tiers_and_both_axes_are_represented():
    tiers = {t["tier"] for t in P.PROJECT_TASKS}
    assert {"build", "gap", "refactor"} <= tiers, tiers
    assert {True, False} == {t["lang_named"] for t in P.PROJECT_TASKS}, "no unnamed-language task"
    assert {"short", "long"} <= {t["spec_size"] for t in P.PROJECT_TASKS}


def test_the_long_spec_buries_a_requirement_past_the_examples():
    """A requirement stated before the examples is one a skimmer still sees."""
    t = next(x for x in P.PROJECT_TASKS if x["spec_size"] == "long")
    body = t["prompt"]
    assert body.count("\n") > 25, "the long spec is not long"
    assert body.index("14.") > len(body) * 0.4, "requirement 14 is not buried"
    hidden14 = [c for c in t["cases"] if "14" in (c.get("label") or "")]
    assert hidden14 and not hidden14[0].get("visible"), "requirement 14 is not hidden-tested"


# --- the visible / hidden split -----------------------------------------------------------


def _res(oks):
    return {"cases": [{"ok": bool(o)} for o in oks], "passed": sum(oks), "total": len(oks)}


def test_visible_and_hidden_are_counted_apart():
    cases = [{"visible": True}, {"visible": True}, {}, {}, {}]
    r = G._bench_split_visible(_res([1, 1, 0, 0, 0]), cases)
    assert (r["visible_passed"], r["visible_total"]) == (2, 2)
    assert (r["hidden_passed"], r["hidden_total"]) == (0, 3)


def test_copying_the_examples_is_named_as_such():
    """Every printed example reproduced, every requirement behind them missed. A single
    percentage averages exactly this away."""
    cases = [{"visible": True}, {"visible": True}, {}, {}]
    assert G._bench_split_visible(_res([1, 1, 0, 0]), cases)["copied_examples"] is True


def test_a_model_that_read_the_spec_is_not_flagged_as_copying():
    cases = [{"visible": True}, {"visible": True}, {}, {}]
    assert G._bench_split_visible(_res([1, 1, 1, 1]), cases)["copied_examples"] is False


def test_a_model_that_failed_everything_is_not_flagged_as_copying():
    """Failing the examples too is a different diagnosis — it did not build, or misread
    entirely."""
    cases = [{"visible": True}, {"visible": True}, {}, {}]
    assert G._bench_split_visible(_res([0, 0, 0, 0]), cases)["copied_examples"] is False


def test_labels_survive_the_subprocess_boundary():
    """The executed grader returns {ok, got}; the label lives in the task definition."""
    cases = [{"visible": True}, {"label": "rounds half up", "visible": False}]
    r = G._bench_split_visible(_res([1, 0]), cases)
    assert r["cases"][1]["label"] == "rounds half up"
    assert r["cases"][0]["visible"] is True and r["cases"][1]["visible"] is False


# --- the traps have to actually trip ------------------------------------------------------

_NAIVE_TOTAL = """```python
def total(items):
    out = 0
    for it in items:
        if it.get("qty", 0) <= 0 or "cents" not in it:
            continue
        line = it["qty"] * it["cents"]
        if "discount" in it:
            line = round(line * (100 - it["discount"]) / 100)
        out += line
    return out
```"""

_GOOD_TOTAL = """```python
def total(items):
    out = 0
    for it in items:
        if it.get("qty", 0) <= 0 or "cents" not in it:
            continue
        line = it["qty"] * it["cents"]
        d = it.get("discount")
        if d:
            line = (line * (100 - d) + 50) // 100
        out += line
    return out
```"""


@pytest.mark.anyio
async def test_the_naive_implementation_fails_a_hidden_requirement(client):
    """The lazy answer a model writes from the examples alone — Python's round() is banker's
    rounding and returns 50 where the spec asks for 51. If this passed, the task would be
    measuring nothing."""
    task = next(t for t in P.PROJECT_TASKS if t["id"] == "proj_refactor_total")
    r = await proxy._bench_grade(_NAIVE_TOTAL, task, 30.0)
    assert r["visible_passed"] == r["visible_total"], "the printed examples should pass"
    assert r["hidden_passed"] < r["hidden_total"], "a hidden requirement must catch it"
    missed = [c.get("label") for c in r["cases"] if not c.get("ok")]
    assert any("half UP" in (m or "") for m in missed), missed


@pytest.mark.anyio
async def test_a_correct_implementation_passes_everything(client):
    """The traps must be passable, or the task measures the trap rather than the model."""
    task = next(t for t in P.PROJECT_TASKS if t["id"] == "proj_refactor_total")
    r = await proxy._bench_grade(_GOOD_TOTAL, task, 30.0)
    assert r["passed"] == r["total"], [c for c in r["cases"] if not c.get("ok")]


# --- the task that names no language ------------------------------------------------------


def test_an_unnamed_language_is_detected_from_the_reply():
    assert G._bench_detect_lang("```python\ndef f():\n    pass\n```") == "python"
    assert G._bench_detect_lang("```js\nfunction f() {}\n```") == "js"


def test_an_unreadable_reply_is_detected_as_nothing():
    """An ungradeable answer says nothing about correctness — the same contract as a missing
    toolchain, so it is skipped rather than scored zero."""
    assert G._bench_detect_lang("I would reach for Rust here.") is None


@pytest.mark.anyio
async def test_the_free_language_task_grades_whatever_the_model_chose(client):
    task = next(t for t in P.PROJECT_TASKS if t["lang"] == "auto")
    js = """```js
function window_max(nums, k) {
  if (k <= 0 || k > nums.length) return [];
  const out = [];
  for (let i = 0; i + k <= nums.length; i++) out.push(Math.max.apply(null, nums.slice(i, i + k)));
  return out;
}
```"""
    r = await proxy._bench_grade(js, task, 30.0)
    if r.get("skipped"):
        pytest.skip("node is not installed on this machine")
    assert r.get("chose_lang") == "js", r.get("chose_lang")
    assert r["passed"] == r["total"], [c for c in r["cases"] if not c.get("ok")]


@pytest.mark.anyio
async def test_a_reply_with_no_code_is_skipped_not_failed(client):
    task = next(t for t in P.PROJECT_TASKS if t["lang"] == "auto")
    r = await proxy._bench_grade("I would use Rust, but here is the idea.", task, 30.0)
    assert r.get("skipped") is True and "no recognisable code" in r.get("error", "")


# --- the seams ----------------------------------------------------------------------------


def test_the_suite_is_registered_and_out_of_the_aggregates():
    assert "project-v1" in proxy._BENCH_SUITES
    ids = {t["id"] for t in proxy._BENCH_SUITES["full-v2"]}
    assert not any(t["id"] in ids for t in P.PROJECT_TASKS), \
        "an episode here is minutes; full-v2 runs in seconds per task"


def test_every_task_has_a_description_and_a_note():
    for t in P.PROJECT_TASKS:
        assert proxy._BENCH_TASK_DESC.get(t["id"]), t["id"]
        assert proxy._BENCH_TASK_NOTES.get(t["id"]), t["id"]


def test_auto_is_listed_in_the_availability_gate():
    """A lang missing from the gate is SKIPPED silently, not failed — two whole suites vanished
    from a full run that way once."""
    assert G._bench_lang_available("auto") is True
