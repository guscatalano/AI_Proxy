"""The grading browser: every request, every case, statically.

The page exists to answer "it scored 89% — what exactly is the other 11% and WHY?", so the
contract is completeness and static rendering: every request appears, passing cases render
what-was-asked → expected, failing cases add what actually came back, truncation is named,
the full graded code is inline, failures sort first, and nothing on the page loads
dynamically — a saved copy answers the same questions offline.
"""
import json
import time

from ai_proxy import bench_report as BR
from ai_proxy import proxy as P


def _cell_run():
    return {
        "id": "b_cell", "label": "m1 · @ollama · short · cached",
        "model": "m1",
        "config": {"suite": "coding-v1", "upstream": "ollama", "max_tokens": 1024},
        "results": {"rows": [
            {"seq": 1, "task": "roman",
             "text": "```python\ndef to_roman(n):\n    return 'IIII'\n```",
             "grade": {"passed": 4, "total": 6,
                       "cases": [{"ok": True}, {"ok": True},
                                 {"ok": False, "got": "IIII"},
                                 {"ok": False, "got": "MDCCCC"},
                                 {"ok": True}, {"ok": True}]}},
            {"seq": 2, "task": "roman", "completion_tokens": 1024,
             "text": "```python\ndef to_roman(n):\n    pass",
             "grade": {"passed": 0, "total": 6, "truncated": True,
                       "error": "SyntaxError: invalid syntax"}},
            {"seq": 3, "task": "binary_search",
             "text": "```python\ndef binary_search(items, target):\n    return -1\n```",
             "grade": {"passed": 6, "total": 6,
                       "cases": [{"ok": True}] * 6}},
        ]},
    }


def test_every_request_appears_with_its_cases(client):
    html = BR._bench_grades_html(_cell_run())
    assert html.count('class="greq"') == 3, "all three requests must render"
    assert "✓" in html and "✗" in html
    assert "to_roman(" in html, "passing cases still show what was asked"
    assert "IIII" in html and "MDCCCC" in html, "failures show what came back"
    assert "hit the token cap mid-answer" in html
    assert "the code the grader extracted" in html
    assert "return &#x27;IIII&#x27;" in html or "return 'IIII'" in html


def test_failures_sort_before_clean_tasks(client):
    html = BR._bench_grades_html(_cell_run())
    assert html.index('id="t-roman"') < html.index('id="t-binary_search"'), \
        "the imperfect task leads; that is the question the page answers"


def test_the_page_is_static(client):
    html = BR._bench_grades_html(_cell_run())
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    assert '<script src=' not in html
    assert "<details" in html, "collapsing is native HTML, not script"


def test_the_prompt_is_browsable(client):
    html = BR._bench_grades_html(_cell_run())
    assert "the prompt every run got" in html
    assert "Roman numeral" in html or "to_roman" in html


def test_route_serves_cell_and_parent_index(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    run = _cell_run()
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, label, config_json, results_json, status, "
        "parent_id) VALUES (?,?,?,?,?,?,?,?)",
        ("b_cell", time.time(), "m1", run["label"], json.dumps(run["config"]),
         json.dumps(run["results"]), "done", "b_parent"))
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,?,?)",
        ("b_parent", time.time(), "m1", "{}", "done"))
    conn.commit()
    conn.close()
    r = client.get("/__proxy/api/bench/runs/b_cell/grades")
    assert r.status_code == 200 and "greq" in r.text
    r2 = client.get("/__proxy/api/bench/runs/b_parent/grades")
    assert r2.status_code == 200 and "/__proxy/api/bench/runs/b_cell/grades" in r2.text
    assert client.get("/__proxy/api/bench/runs/nope/grades").status_code == 404


def test_the_report_links_every_row_to_its_grades(client):
    from test_bench_report_layout import _run, _render
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert html.count('class="glink"') == 2
    assert '/__proxy/api/bench/runs/a/grades' in html
