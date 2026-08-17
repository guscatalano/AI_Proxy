"""The project section of the report.

Rendered from a real four-model run, the generic tables put gemma4, qwen3.8 and
qwen3-coder-next all at "86% fully correct" — while the suite had separated them at 75/78
against 72/78 on hidden cases, and one of the four repeated completed tool calls until its step
budget expired on five of six episodes. A single percentage averages all of that away.
"""
from ai_proxy import bench_report as R


def _unit(task, vis=(3, 3), hid=(5, 5), copied=False, built=None, clean=None, ms=9000):
    g = {"visible_passed": vis[0], "visible_total": vis[1],
         "hidden_passed": hid[0], "hidden_total": hid[1],
         "passed": vis[0] + hid[0], "total": vis[1] + hid[1]}
    if copied:
        g["copied_examples"] = True
    if built is not None:
        g["cases"] = [{"ok": built}, {"ok": clean}]
    return {"task": task, "grade": g, "total_ms": ms}


def _run(label, units):
    return {"label": label + " · project-v1", "results": {"rows": units}}


def test_the_section_is_absent_when_no_project_tasks_ran():
    other = [{"label": "x", "results": {"rows": [{"task": "mid_floor", "grade": {}}]}}]
    assert R._bench_project_html(other) == ""


def test_visible_and_hidden_are_shown_apart():
    """The whole point: passing a printed example proves a model can copy it."""
    html = R._bench_project_html([_run("m", [_unit("proj_calc", vis=(3, 3), hid=(1, 5))])])
    assert ">Visible<" in html and ">Hidden<" in html
    assert "3/3" in html and "1/5" in html


def test_a_model_that_copied_the_examples_is_flagged():
    html = R._bench_project_html([
        _run("copier", [_unit("proj_calc", vis=(3, 3), hid=(0, 5), copied=True)])])
    assert ">Copied<" in html
    import re
    row = re.search(r"<tr data-m=\"copier\">.*?</tr>", html, re.S).group(0)
    assert ">1<" in row, "the copied count is not on the model's row"


def test_episodes_report_building_and_conduct_apart():
    """A model that writes a working file through a mess of repeated calls is not the same as
    one that writes nothing cleanly."""
    html = R._bench_project_html([_run("looper", [
        _unit("projep_calc_build", built=True, clean=False),
        _unit("projep_slugify_fix", built=True, clean=False),
    ])])
    assert ">Built<" in html and ">Clean<" in html
    import re
    row = re.search(r"<tr data-m=\"looper\">.*?</tr>", html, re.S).group(0)
    assert "2/2" in row and "0/2" in row, row


def test_single_turn_tasks_do_not_count_toward_the_episode_columns():
    """Only the episodes hand the model tools; counting the rest would dilute the signal."""
    html = R._bench_project_html([_run("m", [
        _unit("proj_calc"), _unit("projep_calc_build", built=True, clean=True)])])
    import re
    row = re.search(r"<tr data-m=\"m\">.*?</tr>", html, re.S).group(0)
    assert "1/1" in row, "the episode columns counted a single-turn task"


def test_models_are_ordered_by_hidden_cases_then_conduct():
    """Hidden cases are the measurement; conduct breaks a tie, because two models that read the
    spec equally well are separated by whether they behave while using tools."""
    html = R._bench_project_html([
        _run("worse", [_unit("proj_calc", hid=(3, 5)),
                       _unit("projep_calc_build", built=True, clean=True)]),
        _run("better", [_unit("proj_calc", hid=(5, 5)),
                        _unit("projep_calc_build", built=True, clean=True)]),
    ])
    assert html.index("better") < html.index("worse")


def test_the_per_task_table_names_which_requirement_was_missed():
    html = R._bench_project_html([_run("m", [
        _unit("proj_calc", hid=(2, 5)), _unit("proj_report", hid=(5, 5))])])
    assert ">calc<" in html and ">report<" in html
    assert "2/5" in html and "5/5" in html


def test_episode_tasks_are_labelled_as_episodes_in_the_per_task_table():
    html = R._bench_project_html([_run("m", [
        _unit("projep_calc_build", built=True, clean=True)])])
    assert "ep:calc_build" in html, "an episode should read differently from a single-turn task"


def test_a_run_with_no_episodes_does_not_divide_by_zero():
    html = R._bench_project_html([_run("m", [_unit("proj_calc")])])
    assert html and "Built" in html
