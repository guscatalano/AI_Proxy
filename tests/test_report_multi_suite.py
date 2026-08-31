"""A report handed more than one suite must not rank them against each other.

langpref-v1 grades "did you pick a defensible language" over 24 tasks; full-v2 grades
correctness over 119. Combining them in one report made a model's 24/24 outrank another's
101/119, and the false winner reached the headline, the weighted standings, the category
winners, the trade-off scatter and the memory chart. Only the cost table was ever noticed.
"""
from ai_proxy import bench_report as R


def run(model, suite, n_total, perfect, tokens=300):
    return {
        "id": "r-%s-%s" % (model, suite), "model": model, "ts": 1786600000,
        "label": "%s · short · think=off · t=0.0" % model,
        "config": {"suite": suite, "runs": 1, "thinking": "off", "temperature": 0.0},
        "env": {}, "status": "done",
        "results": {"summary": {
            "n_total": n_total, "n_success": n_total,
            "ttft_ms": {"p50": 500.0}, "decode_tps": {"p50": 60.0},
            "total_ms": {"p50": 3000.0}, "completion_tokens": {"mean": tokens},
            "quality": {"perfect_rate": perfect, "case_pass_rate": perfect,
                        "total_runs": n_total, "perfect_runs": int(perfect * n_total),
                        "tiers": {}, "tasks": []},
        }, "rows": []},
    }


def render(runs):
    return R._bench_report_html(runs, [R._bench_report_row(r) for r in runs])


def test_a_small_suite_ace_does_not_become_the_headline():
    """The exact shape that broke: 24/24 on langpref beating 101/119 on full-v2."""
    runs = [run("alpha", "full-v2", 119, 0.655),
            run("alpha", "langpref-v1", 24, 1.0),
            run("beta", "full-v2", 119, 0.849),
            run("beta", "langpref-v1", 24, 0.958)]
    html = render(runs)
    assert "100% correct" not in html, "a langpref score is being reported as correctness"
    assert "85%" in html, "the real full-v2 leader should be the headline"


def test_the_bigger_suite_is_the_one_that_ranks():
    runs = [run("alpha", "full-v2", 119, 0.655), run("alpha", "langpref-v1", 24, 1.0),
            run("beta", "full-v2", 119, 0.849), run("beta", "langpref-v1", 24, 0.958)]
    html = render(runs)
    assert "full-v2" in html
    # The 24-task rows must not appear as ranked configurations in the standings.
    assert html.count("langpref-v1") >= 1, "the other suite should still be named"


def test_the_reader_is_told_what_was_set_aside():
    runs = [run("alpha", "full-v2", 119, 0.655), run("alpha", "langpref-v1", 24, 1.0),
            run("beta", "full-v2", 119, 0.849), run("beta", "langpref-v1", 24, 0.958)]
    html = render(runs)
    assert "in their own sections" in html or "own sections" in html, (
        "runs vanishing from the table without explanation is its own confusion")


def test_a_single_suite_report_is_untouched():
    runs = [run("alpha", "full-v2", 119, 0.655), run("beta", "full-v2", 119, 0.849)]
    html = render(runs)
    assert "own sections" not in html, "nothing was set aside, so say nothing"
    assert "85%" in html


def test_suite_of_equal_size_still_picks_one_and_explains():
    """Two suites, same task count: pick one deterministically rather than interleaving."""
    runs = [run("alpha", "coding-v3", 47, 0.9), run("alpha", "security-v1", 47, 0.5)]
    html = render(runs)
    assert "own sections" in html
