"""The guide: what every benchmark task asks, and what right and wrong look like.

Written because the suites had accumulated 177 tasks whose point was legible only from the
source. Every task already carried a description and a note explaining why it exists; nothing
surfaced them, and nothing showed what a passing answer actually looks like.
"""
from ai_proxy import proxy


def test_the_guide_covers_every_suite(client):
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    names = {s["name"] for s in body["suites"]}
    assert names == set(proxy._BENCH_SUITES), names ^ set(proxy._BENCH_SUITES)


def test_every_task_carries_its_description_and_its_why(client):
    """The description says what it does; the note says why it is in the suite. The note is the
    half that cannot be reconstructed from the code."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    for s in body["suites"]:
        for t in s["tasks"]:
            assert t["desc"], f"{t['id']} has no description"
            assert t["note"], f"{t['id']} has no note"


def test_a_passing_answer_is_shown_for_each_task(client):
    """Derived from the task's own cases, in the same vocabulary the report's failure examples
    use, so a reader who has seen one recognises the other."""
    body = client.get("/__proxy/api/bench/guide?format=json").json()
    empty = [t["id"] for s in body["suites"] for t in s["tasks"] if not t["good"]]
    assert not empty, f"no passing example derivable for: {empty[:8]}"


def test_a_single_suite_can_be_asked_for(client):
    body = client.get("/__proxy/api/bench/guide?suite=longctx-lite&format=json").json()
    assert [s["name"] for s in body["suites"]] == ["longctx-lite"]
    assert len(body["suites"][0]["tasks"]) == 6


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
    assert "no failure on record" in html or "A real failure" in html
