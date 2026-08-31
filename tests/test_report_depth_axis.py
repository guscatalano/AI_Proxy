"""Prompt depth must survive as a comparison axis.

A long-context sweep compares depth 0 against depth N. Depth 0 was recorded as absent rather
than as a value, so the axis held one distinct entry, was judged constant, and no column was
drawn — a 0-vs-32k run rendered as four rows labelled only by model name, with nothing
indicating which was the baseline. The measurement was there; the report could not show it.
"""
from ai_proxy import bench_report as R


def row(model, depth, tps=60.0, perfect=0.9):
    return {"model": model, "served": model, "thinking": "off", "temperature": 0.0,
            "prompt_tokens": depth, "n_total": 47, "perfect_rate": perfect,
            "mean_tokens": 250, "decode_p50": tps, "ttft_p50": 500.0, "label": model}


def cfg():
    return {"config": {"suite": "coding-v3", "upstream": "ollama", "concurrency": 1}}


def test_zero_depth_is_a_value_not_an_absence():
    v = R._bench_axis_values(row("alpha", 0), {})
    assert v["prompt"] is not None, "the baseline of a depth sweep must be representable"


def test_a_depth_sweep_produces_a_varying_axis():
    rows = [row("alpha", 0), row("alpha", 32000)]
    runs = [cfg(), cfg()]
    varying, constant, _vals = R._bench_axis_split(rows, runs)
    assert "prompt" in varying, "0 vs 32k must read as two values, not one"
    assert "prompt" not in constant


def test_two_models_at_two_depths_name_every_cell_distinctly():
    """Four rows, two models, two depths — every cell needs its own identity."""
    rows = [row("alpha", 0), row("alpha", 32000), row("beta", 0), row("beta", 32000)]
    runs = [cfg()] * 4
    varying, _c, vals = R._bench_axis_split(rows, runs)
    names = [R._bench_cell_name(v, varying) for v in vals]
    assert len(set(names)) == 4, f"cells are indistinguishable: {names}"
    assert any("32,000" in n for n in names)
    assert any("none" in n for n in names)


def test_an_unpadded_comparison_does_not_grow_a_depth_column():
    """Every run at depth 0 means depth is constant — it should not become a column."""
    rows = [row("alpha", 0), row("beta", 0)]
    varying, constant, _v = R._bench_axis_split(rows, [cfg(), cfg()])
    assert "prompt" not in varying
    assert constant.get("prompt") == "none"


def test_a_run_with_no_depth_setting_at_all_is_still_absent():
    r = row("alpha", 0)
    del r["prompt_tokens"]
    assert R._bench_axis_values(r, {})["prompt"] is None


def test_depth_values_are_formatted_readably():
    assert R._bench_axis_values(row("a", 128000), {})["prompt"] == "128,000"
    assert R._bench_axis_values(row("a", 0), {})["prompt"] == "none"
