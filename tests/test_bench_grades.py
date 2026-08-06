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
    assert "TOKEN CAP" in html and "SYNTAX ERROR" in html, "badges must shout"
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


def test_grader_errors_are_loud_and_complete(client):
    html = BR._bench_grades_html(_cell_run())
    assert 'class="gbadge"' in html
    assert 'class="gerr"' in html, "the full error renders as its own block, not a grey flag"
    assert "SyntaxError: invalid syntax" in html


def test_grades_pages_link_back_to_the_report(client):
    run = _cell_run()
    run["parent_id"] = "b_parent"
    html = BR._bench_grades_html(run)
    assert "/__proxy/api/bench/report?format=html&ids=b_parent" in html
    assert "/__proxy/api/bench/runs/b_parent/grades" in html
    idx = BR._bench_grades_index_html({"id": "b_parent"}, [])
    assert "/__proxy/api/bench/report?format=html&ids=b_parent" in idx


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


def test_task_notes_explain_the_trap_for_imperfect_tasks(client):
    html = BR._bench_grades_html(_cell_run())
    assert "Why this fails when it fails" in html
    assert "Subtractive pairs" in html, "roman's authored note must render"
    # A clean task keeps its page quiet — the note only appears where it answers something.
    assert html.split('id="t-binary_search"')[1].count("Why this fails") == 0


def test_compile_errors_carry_their_build_context(client):
    run = _cell_run()
    run["config"]["suite"] = "coding-v2"
    run["results"]["rows"] = [
        {"seq": 1, "task": "csv_escape",
         "text": "```cpp\nstd::string csv_escape(std::string f) { return f\n```",
         "grade": {"passed": 0, "total": 6,
                   "error": "compile error: expected ';' before '}' token",
                   "build": {"cmd": "g++ -std=c++17 -O0 task.cpp -o task.exe",
                             "compiler": "g++ (Ubuntu 13.2.0) 13.2.0",
                             "harness": "#include <iostream>\nint main() { /* calls */ }"}}},
    ]
    html = BR._bench_grades_html(run)
    assert "compiled with" in html and "g++ -std=c++17" in html
    assert "g++ (Ubuntu 13.2.0)" in html
    assert "the model saw only the task prompt" in html
    assert "the full source as compiled" in html and "#include &lt;iostream&gt;" in html


def test_every_task_in_every_suite_has_a_note(client):
    from ai_proxy.bench_suites import SUITES, TASK_NOTES
    missing = [t["id"] for name in ("coding-v1", "coding-v2", "coding-v3")
               for t in SUITES[name] if t["id"] not in TASK_NOTES]
    assert not missing, missing


def test_the_report_names_its_graders(client):
    from test_bench_report_layout import _run, _render
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    runs[0]["env"]["toolchains"] = {"c": "gcc (Ubuntu 13.2.0) 13.2.0", "python": "3.12.3"}
    html = _render(runs)
    assert "Graded with" in html
    assert "gcc (Ubuntu 13.2.0)" in html
    assert "read from the host at report time" not in html.split("Graded with")[1][:600], \
        "run-recorded versions must not carry the backfill caveat"


def test_grades_page_names_its_graders_when_recorded(client):
    run = _cell_run()
    run["env"] = {"toolchains": {"python": "3.12.3", "go": "go version go1.26.5"}}
    html = BR._bench_grades_html(run)
    assert "Graded with:" in html and "go1.26.5" in html
