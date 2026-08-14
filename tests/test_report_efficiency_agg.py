"""Cost-per-solved-task must describe everything a model was measured on.

The table used to sort rows by perfect_rate and keep the FIRST one per model, so a model
appearing in two runs was reported at whichever flattered it. A run scoring 65% on full-v2
and 100% on the 24-task langpref suite printed as "100% correct" — a number that described
neither suite and made every model look better than it was.
"""
from ai_proxy import bench_report as R


def row(name, n_total, perfect_rate, mean_tokens):
    return {"_name": name, "label": name, "n_total": n_total,
            "perfect_rate": perfect_rate, "mean_tokens": mean_tokens}


def pct_for(html, model):
    """The Correct% cell rendered for a model."""
    import re
    m = re.search(r'data-m="%s".*?<td class="n">(\d+)%%</td>' % re.escape(model), html, re.S)
    return int(m.group(1)) if m else None


def test_two_runs_for_one_model_are_combined_not_cherry_picked():
    html = R._bench_efficiency_html([
        row("alpha", 119, 0.655, 279),      # the real suite
        row("alpha", 24, 1.0, 1172),        # the small one it aced
        row("beta", 119, 0.849, 238),
        row("beta", 24, 0.958, 1267),
    ])
    # 78 + 24 solved of 143 tasks = 71%, not 100%.
    assert pct_for(html, "alpha") == 71, "reported its best suite instead of its whole record"
    assert pct_for(html, "beta") == 87


def test_a_single_run_per_model_is_unchanged():
    html = R._bench_efficiency_html([row("alpha", 119, 0.655, 279),
                                     row("beta", 119, 0.849, 238)])
    assert pct_for(html, "alpha") == 66
    assert pct_for(html, "beta") == 85


def test_tokens_per_solved_uses_pooled_totals():
    """Spend and solves both pool, so the ratio stays a real cost-per-answer."""
    html = R._bench_efficiency_html([
        row("alpha", 100, 0.5, 100),        # 10,000 tokens, 50 solved
        row("alpha", 100, 0.5, 300),        # 30,000 tokens, 50 solved
        row("beta", 100, 0.5, 200),
    ])
    assert "400" in html, "40,000 tokens over 100 solved should read as 400 per solved"


def test_a_model_that_solved_nothing_is_dropped_not_divided_by_zero():
    html = R._bench_efficiency_html([row("alpha", 50, 0.0, 500),
                                     row("beta", 50, 0.5, 200),
                                     row("gamma", 50, 0.4, 300)])
    assert "alpha" not in html
    assert "beta" in html
