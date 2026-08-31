"""The guide: what every benchmark task asks, and what right and wrong look like.

Written because the suites had accumulated 177 tasks whose point was legible only from the
source. Every task already carried a description and a note explaining why it exists; nothing
surfaced them, and nothing showed what a passing answer actually looks like.
"""
from ai_proxy import proxy


def test_every_task_appears_exactly_once(client):
    """Grouping by suite drew 431 cards for 177 tasks: full-v2 contains coding-v3 plus agent-v2
    plus security-v1 plus the rest, so parse_query was rendered four times and the page read as
    three times longer than it is."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    ids = [t["id"] for c in body["categories"] for t in c["tasks"]]
    distinct = {t["id"] for s in proxy._BENCH_SUITES.values() for t in s}
    assert len(ids) == len(set(ids)) == len(distinct), (len(ids), len(set(ids)), len(distinct))


def test_a_task_names_the_suites_that_include_it(client):
    """Which suites contain a task is a property of the task, not a reason to draw it again."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    t = next(t for c in body["categories"] for t in c["tasks"] if t["id"] == "parse_query")
    assert len(t["suites"]) >= 3, t["suites"]


def test_categories_cover_every_task(client):
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    cats = {c["name"] for c in body["categories"]}
    assert {"coding", "security", "agentic", "longcontext"} <= cats, cats
    assert sum(len(c["tasks"]) for c in body["categories"]) == len(
        {t["id"] for s in proxy._BENCH_SUITES.values() for t in s})


def test_every_task_carries_its_description_and_its_why(client):
    """The description says what it does; the note says why it is in the suite. The note is the
    half that cannot be reconstructed from the code."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    for s in body["categories"]:
        for t in s["tasks"]:
            assert t["desc"], f"{t['id']} has no description"
            assert t["note"], f"{t['id']} has no note"


def test_a_passing_answer_is_shown_for_each_task(client):
    """Derived from the task's own cases, in the same vocabulary the report's failure examples
    use, so a reader who has seen one recognises the other."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    empty = [t["id"] for s in body["categories"] for t in s["tasks"] if not t["good"]]
    assert not empty, f"no passing example derivable for: {empty[:8]}"


def test_a_single_suite_can_still_be_asked_for(client):
    body = client.get("/__proxy/api/bench/guide?suite=longctx-lite&format=json").json()
    ids = [t["id"] for c in body["categories"] for t in c["tasks"]]
    assert len(ids) == 6 and all(i.startswith("longctx_") for i in ids)


def test_an_unknown_suite_says_what_exists(client):
    r = client.get("/__proxy/api/bench/guide?suite=nope&format=json")
    assert r.status_code == 404 and "available" in r.json()["error"]


def test_the_html_page_loads_nothing_from_outside(client):
    """Same contract as the report: it survives being saved or mailed. Checked as resource
    LOADS rather than as any occurrence of a URL — the security tasks legitimately contain
    http://2130706433/ and https://api.stripe.com/... as the text a model is asked to review,
    and a test that forbids the substring fails on the content it is meant to document."""
    import re as _re
    html = client.get("/__proxy/api/bench/guide").text
    assert html.startswith("<!doctype html>")
    for pattern in (r"<script[^>]+src\s*=", r"<link[^>]+href\s*=",
                    r"<img[^>]+src\s*=\s*[\"']https?:", r"@import", r"url\(\s*https?:"):
        m = _re.search(pattern, html, _re.I)
        assert not m, f"the page loads an external resource: {m.group(0)[:60]}"


def test_the_needle_tasks_explain_their_depths(client):
    """A reader should learn what the ladder measures from the page, not from the source."""
    html = client.get("/__proxy/api/bench/guide?suite=longctx-lite").text
    assert "planted at the start" in html and "50% of the way through" in html
    assert "CRIMSON-4417" in html, "the expected answer should be concrete"


def test_real_failures_are_used_rather_than_invented_ones(client):
    """An invented failure teaches the shape of a mistake nobody made."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    assert isinstance(body.get("failures"), dict)
    html = client.get("/__proxy/api/bench/guide").text
    assert "Real failures" in html
    assert "nothing has failed this yet" in html, "tasks with a clean record must say so"


def test_all_three_failure_channels_are_surfaced():
    """Measured across recent runs on the box: 135 failures carried a per-case `got`, 18 were
    compile errors in grade.error, and 6 were runtime errors in the case's own `error` with
    got=null. Reading only `got` hid every task that never built — the loudest failure a coding
    suite can have. Driven directly rather than through the endpoint, because a fresh test
    database has no stored runs to draw a real compile error from."""
    from ai_proxy import bench_report as R
    suites = [{"name": "coding-v3", "tasks": [
        {"id": "go_rle_decode", "lang": "go", "desc": "d", "note": "n", "good": [("f(1)", "2")],
         "stats": {"runs": 4, "perfect": 1}},
    ]}]
    examples = {"go_rle_decode": [
        ("build", 'compile error: ./task.go:7:2: "errors" imported and not used'),
        ("crash", "UnboundLocalError: cannot access local variable"),
        ("wrong", "f(1) -> 3, expected 2"),
    ]}
    html = R._bench_guide_html(suites, examples)
    for label in ("did not build", "crashed", "wrong answer"):
        assert label in html, f"the {label!r} failure kind is never shown"
    assert "imported and not used" in html, "the compiler's own words are the useful part"


def test_the_endpoint_collects_build_errors_not_only_case_results():
    """The gap was in collection, not rendering: grade.error was never read."""
    import inspect
    from ai_proxy import proxy as P
    src = inspect.getsource(P.bench_guide)
    assert 'g.get("error")' in src, "compile failures are still dropped"
    assert 'c.get("error")' in src, "per-case crashes are still dropped"
    assert 'c.get("got")' in src


def test_each_card_carries_a_pass_rate(client):
    """The stat line is the point of a card. A task nobody has run says so rather than
    showing a zero, which would read as a task everything fails."""
    html = client.get("/__proxy/api/bench/guide").text
    assert 'class="stat' in html
    assert "runs passed" in html or "never run" in html


def test_the_page_is_searchable_and_groups_collapse(client):
    """177 tasks is too many to scroll. Search filters the cards, and a group whose cards are
    all filtered out hides itself rather than leaving an empty header behind."""
    html = client.get("/__proxy/api/bench/guide").text
    assert 'id="q"' in html and "type=\"search\"" in html
    assert 'class="tab' in html and 'class="panel' in html
    assert "Only tasks with failures" in html


def test_each_category_is_a_tab_with_its_count(client):
    """Fifteen suite groups was the overwhelming part; eight categories with counts is the
    shape a reader can hold."""
    html = client.get("/__proxy/api/bench/guide").text
    import re as _re
    tabs = _re.findall(r'<button class="tab[^"]*" data-cat="([^"]+)"', html)
    assert {"coding", "security", "agentic"} <= set(tabs), tabs
    assert html.count('class="tcount"') == len(tabs), "every tab needs its task count"
    assert html.count('class="panel') == len(tabs)


def test_search_spans_every_category(client):
    """A task you cannot name is usually one you cannot place in a category either, so scoping
    the search to the open tab would hide the answer."""
    html = client.get("/__proxy/api/bench/guide").text
    assert "var scope = term ? cards : visible()" in html
