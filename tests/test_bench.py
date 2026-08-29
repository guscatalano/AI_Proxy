"""Tests for the benchmark runner: measurement correctness, matrix expansion, grading.

No upstream is required — the pieces exercised here are pure functions plus the submit
endpoint's validation. The one part that genuinely runs a subprocess is the grader, which is
the point: it executes model-written code, so its containment behavior is worth asserting.
"""
import asyncio

import pytest

import ai_proxy.proxy as p


# ---- measurement ------------------------------------------------------------------------

def test_summarize_separates_reasoning_from_content_latency():
    """TTFT is first-token-of-any-kind; TTFC is first *content* token. Conflating them (the
    original bug) reports a thinking model's time-to-end-of-reasoning as its TTFT."""
    rows = [
        {"seq": 1, "ttft_ms": 100.0, "ttfc_ms": 900.0, "total_ms": 2000.0,
         "completion_tokens": 500, "reasoning_tokens": 300, "decode_tps": 26.3, "error": None},
        {"seq": 2, "ttft_ms": 120.0, "ttfc_ms": 1100.0, "total_ms": 2200.0,
         "completion_tokens": 520, "reasoning_tokens": 310, "decode_tps": 25.0, "error": None},
    ]
    s = p._bench_summarize(rows)
    assert s["n_success"] == 2
    assert s["ttft_ms"]["p50"] == pytest.approx(110.0)
    assert s["ttfc_ms"]["p50"] == pytest.approx(1000.0)
    # The reasoning phase is the gap between the two.
    assert s["reasoning_ms"]["p50"] == pytest.approx(890.0)
    assert s["reasoning_tokens"]["p50"] == pytest.approx(305.0)


def test_summarize_omits_reasoning_block_for_non_thinking_runs():
    rows = [{"seq": 1, "ttft_ms": 50.0, "ttfc_ms": 50.0, "total_ms": 500.0,
             "completion_tokens": 100, "reasoning_tokens": None, "decode_tps": 222.0,
             "error": None}]
    s = p._bench_summarize(rows)
    assert "reasoning_tokens" not in s
    # ttfc == ttft means a zero-length reasoning phase, which is still reported (as 0).
    assert s["reasoning_ms"]["p50"] == 0


def test_summarize_collapses_repeated_errors():
    rows = [{"seq": i, "ttft_ms": None, "ttfc_ms": None, "total_ms": 10.0,
             "completion_tokens": 0, "decode_tps": None, "error": "HTTP 500: boom"}
            for i in range(3)]
    s = p._bench_summarize(rows)
    assert s["n_success"] == 0
    assert s["errors"] == [{"message": "HTTP 500: boom", "count": 3}]


# ---- thinking control -------------------------------------------------------------------

def test_thinking_auto_leaves_body_untouched():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    p._bench_apply_thinking(body, "auto")
    assert "chat_template_kwargs" not in body
    assert "reasoning_effort" not in body


def test_thinking_off_sets_both_engine_switches():
    """Qwen-lineage engines read chat_template_kwargs; ds4 reads reasoning_effort. Setting
    both means one config works across engines."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    p._bench_apply_thinking(body, "off")
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["reasoning_effort"] == "none"


def test_thinking_off_prefill_appends_spent_think_block():
    """LM Studio / llama.cpp drops chat_template_kwargs entirely, so the only lever there is
    putting an already-closed <think> block in the model's mouth."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    p._bench_apply_thinking(body, "off_prefill")
    last = body["messages"][-1]
    assert last["role"] == "assistant"
    assert "</think>" in last["content"]


def test_thinking_on_requests_reasoning():
    body = {"model": "m", "messages": []}
    p._bench_apply_thinking(body, "on")
    assert body["chat_template_kwargs"]["enable_thinking"] is True
    assert body["reasoning_effort"] == "high"


def test_sampling_only_sets_provided_knobs():
    body = {}
    p._bench_apply_sampling(body, {"temperature": 0.2, "top_k": None, "extra_body": {"x": 1}})
    assert body == {"temperature": 0.2, "x": 1}


def test_headers_pin_routing_when_asked():
    h = p._bench_headers({"bypass_router": True, "upstream": "vllm", "no_nudge": True})
    assert h["x-client-name"] == "ai-proxy-bench"   # panic-mode whitelist
    assert h["x-proxy-no-router"] == "1"
    assert h["x-proxy-upstream"] == "vllm"
    assert h["x-proxy-no-nudge"] == "1"
    assert "x-proxy-upstream" not in p._bench_headers({})


# ---- matrix -----------------------------------------------------------------------------

def test_matrix_expands_the_cross_product():
    cells = p._bench_expand_matrix(
        ["a", "b"], {"prompt_tokens": [0, 32000], "thinking": ["off", "on"]})
    assert len(cells) == 8
    assert {c["model"] for c in cells} == {"a", "b"}
    assert {c["thinking"] for c in cells} == {"off", "on"}


def test_matrix_scalars_collapse_to_one_cell():
    cells = p._bench_expand_matrix(["a"], {"prompt_tokens": 4096, "thinking": "off"})
    assert cells == [{"model": "a", "prompt_tokens": 4096, "thinking": "off"}]


def test_cell_label_is_human_readable():
    label = p._bench_cell_label(
        {"model": "ornith", "prompt_tokens": 32000, "thinking": "off", "temperature": 0.2})
    assert "ornith" in label and "32k" in label and "off" in label and "0.2" in label


# ---- grading ----------------------------------------------------------------------------

def _grade(text, task_id="binary_search", timeout=15):
    task = next(t for t in p._BENCH_SUITES["coding-v1"] if t["id"] == task_id)
    return asyncio.run(p._bench_grade(text, task, timeout))


def test_grader_scores_a_correct_implementation():
    res = _grade("""Sure:
```python
def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```
""")
    assert res["score"] == 1.0
    assert res["passed"] == res["total"]


def test_grader_catches_a_wrong_implementation():
    res = _grade("```python\ndef binary_search(items, target):\n    return 0\n```")
    assert 0 < res["score"] < 1


def test_grader_reports_missing_code_as_such():
    """Prose must not be handed to the compiler — 'SyntaxError' would read as broken code
    rather than as a model that never answered."""
    res = _grade("I would rather not.")
    assert res["score"] == 0.0
    assert res["error"] == "no code in response"


def test_grader_survives_an_infinite_loop():
    """Model-written code runs in a subprocess with a hard timeout; a runaway loop must not
    wedge the bench."""
    res = _grade("```python\ndef binary_search(items, target):\n    while True:\n        pass\n```",
                 timeout=3)
    assert res["score"] == 0.0
    assert "timeout" in res["error"]


def test_grader_treats_tuples_and_lists_as_equal():
    """JSON can't distinguish them, and [('a',1)] vs [['a',1]] is not a correctness issue."""
    res = _grade("""```python
def word_freq(text, n):
    import re
    from collections import Counter
    words = re.findall(r"[a-z]+", text.lower())
    counts = Counter(words)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]
```""", task_id="word_freq")
    assert res["score"] == 1.0


def test_code_extraction_prefers_the_longest_block():
    """Models often emit a short usage example beside the real implementation."""
    text = "```python\nprint(binary_search([1],1))\n```\nand the impl:\n```python\ndef binary_search(a,b):\n    return -1\n```"
    assert "def binary_search" in p._bench_extract_code(text)


def test_quality_summary_separates_perfect_from_partial():
    """A model right on 3 of 6 cases everywhere and one perfect on half the tasks can share a
    case-pass-rate; only the perfect-run rate tells them apart."""
    suite = p._BENCH_SUITES["coding-v1"][:2]
    rows = [
        {"task": suite[0]["id"], "grade": {"passed": 6, "total": 6}},
        {"task": suite[1]["id"], "grade": {"passed": 2, "total": 4}},
    ]
    q = p._bench_quality_summary(rows, suite)
    assert q["passed_cases"] == 8 and q["total_cases"] == 10
    assert q["perfect_runs"] == 1 and q["total_runs"] == 2
    assert q["perfect_rate"] == pytest.approx(0.5)


# ---- endpoint validation ----------------------------------------------------------------

def test_submit_rejects_unknown_thinking_mode(client):
    r = client.post("/__proxy/api/bench/run", json={"model": "m", "thinking": "sideways"})
    assert r.status_code == 400
    assert "thinking" in r.json()["error"]


def test_submit_rejects_unknown_suite(client):
    r = client.post("/__proxy/api/bench/run", json={"model": "m", "suite": "nope"})
    assert r.status_code == 400
    assert "unknown suite" in r.json()["error"]


def test_submit_rejects_unknown_upstream(client):
    r = client.post("/__proxy/api/bench/run", json={"model": "m", "upstream": "bedrock"})
    assert r.status_code == 400


def test_submit_requires_a_model(client):
    assert client.post("/__proxy/api/bench/run", json={}).status_code == 400


def test_suites_endpoint_lists_tasks(client):
    r = client.get("/__proxy/api/bench/suites")
    assert r.status_code == 200
    body = r.json()
    names = [s["name"] for s in body["suites"]]
    assert "coding-v1" in names
    assert "off_prefill" in body["thinking_modes"]


def test_report_requires_ids(client):
    assert client.get("/__proxy/api/bench/report").status_code == 400


def test_models_endpoint_reports_upstream_and_load_state(client):
    """The upstream is derivable from the model, which is why the bench form doesn't ask for
    both. `loaded` drives the cold-model warning and the warm-up."""
    r = client.get("/__proxy/api/bench/models")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    for item in body["items"]:
        assert item["upstream"] in ("ollama", "lmstudio", "vllm")
        assert isinstance(item["loaded"], bool)


def test_model_index_reads_the_snapshot_keys_the_snapshots_actually_emit(monkeypatch):
    """Regression: the index read `lmstudio.models` / `vllm.models`, but both snapshots emit
    `loaded` / `available`. The result was that LM Studio and vLLM models never appeared in the
    bench picker at all — silently, since an empty list looks like 'nothing is running'."""
    def fake_system_now():
        return {
            "ollama": {"tags": [{"name": "llama3:8b"}], "ps": [{"name": "llama3:8b"}]},
            "lmstudio": {
                "reachable": True,
                "available": [
                    {"id": "qwen/qwen3-coder-next", "state": "not-loaded",
                     "quant": "Q4_K_M", "max_context_length": 262144},
                    {"id": "ornith-35b", "state": "loaded", "quant": "Q8_0",
                     "max_context_length": 32768},
                ],
                "loaded": [{"id": "ornith-35b", "state": "loaded"}],
            },
            "vllm": {
                "reachable": True,
                "available": [{"id": "ornith-nvfp4", "state": "loaded",
                               "max_context_length": 40960}],
                "loaded": [{"id": "ornith-nvfp4", "state": "loaded"}],
            },
        }

    monkeypatch.setattr(p, "system_now", fake_system_now)
    index = asyncio.run(p._bench_model_index())

    assert index["lmstudio:qwen/qwen3-coder-next"]["upstream"] == "lmstudio"
    assert index["lmstudio:qwen/qwen3-coder-next"]["loaded"] is False
    assert index["lmstudio:qwen/qwen3-coder-next"]["quant"] == "Q4_K_M"
    assert index["lmstudio:ornith-35b"]["loaded"] is True
    assert index["vllm:ornith-nvfp4"]["upstream"] == "vllm"
    assert index["ollama:llama3:8b"]["loaded"] is True


def test_model_index_records_per_backend_load_semantics(monkeypatch):
    """vLLM can't load on demand, so an unloaded model there is a guaranteed failure rather
    than a slow first request. The UI needs that distinction to warn correctly."""
    def fake_system_now():
        return {
            "ollama": {"tags": [{"name": "m1"}], "ps": []},
            "lmstudio": {"available": [{"id": "m2", "state": "not-loaded"}]},
            "vllm": {"available": [{"id": "m3", "state": "loaded"}]},
        }

    monkeypatch.setattr(p, "system_now", fake_system_now)
    index = asyncio.run(p._bench_model_index())
    assert index["ollama:m1"]["load_mode"] == "on-demand"
    assert index["lmstudio:m2"]["load_mode"] == "jit"
    assert index["vllm:m3"]["load_mode"] == "fixed"


def test_loaded_entry_wins_over_catalogue_entry(monkeypatch):
    """Ollama lists a model in both /api/tags and /api/ps; the running one must not be
    overwritten by the catalogue entry and reported as cold."""
    def fake_system_now():
        return {"ollama": {"tags": [{"name": "dup"}], "ps": [{"name": "dup"}]},
                "lmstudio": {}, "vllm": {}}

    monkeypatch.setattr(p, "system_now", fake_system_now)
    assert asyncio.run(p._bench_model_index())["ollama:dup"]["loaded"] is True


def test_upstream_pin_header_does_not_corrupt_the_router_verdict(client):
    """Regression: x-proxy-upstream used to be folded into the model_router `rewrite` dict,
    producing a partial dict with no from/to/via. The audit path reads those keys unguarded, so
    any pinned request 500'd as soon as another transform fired and triggered an audit write —
    which is every bench request, since the bench pins its upstream."""
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent-test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-proxy-upstream": "vllm", "x-proxy-no-router": "1",
                 "x-client-name": "ai-proxy-bench"},
    )
    # No upstream is running in tests, so 502 (unreachable) is the expected outcome. What must
    # NOT happen is a 500 from the proxy's own audit code.
    assert r.status_code != 500, r.text
    assert r.status_code in (502, 503, 504), r.status_code


# ---- report rendering --------------------------------------------------------------------

def test_report_html_is_self_contained(client, monkeypatch):
    """The report has to survive being saved to a file, mailed, or printed to PDF, so it must
    not reference anything it doesn't carry."""
    run = {
        "id": "b_x", "ts": 1785000000.0, "label": "m · short", "model": "m",
        "config": {"runs": 3, "suite": "coding-v1", "upstream": "ollama", "thinking": "auto"},
        "env": {"proxy_version": "0.2.0", "gpus": [{"name": "GB10", "mem_total_mb": 131072}]},
        "results": {"summary": {
            "n_total": 3, "n_success": 3, "served_models": ["m"],
            "ttft_ms": {"p50": 70.0}, "ttfc_ms": {"p50": 240.0},
            "decode_tps": {"p50": 160.0}, "total_ms": {"p50": 640.0},
            "warmup_ms": 280.0,
            "quality": {"perfect_rate": 0.83, "case_pass_rate": 0.81,
                        "tasks": [{"task": "roman", "perfect_rate": 0.0}]},
        }},
    }
    html = p._bench_report_html([run], [p._bench_report_row(run)])
    assert html.startswith("<!doctype html>")
    for external in ("src=\"http", "href=\"http", "@import", "fonts.googleapis"):
        assert external not in html, f"report pulls in {external}"
    assert "GB10" in html            # environment captured
    assert "roman" in html           # per-task breakdown
    assert "@media print" in html    # printable to PDF


def test_report_html_escapes_untrusted_names():
    """Model names, labels and upstreams come from outside and land in the page. None of them is
    trusted markup, in either the single-configuration or the comparison layout."""
    def mk(i):
        return {
            "id": f"b_{i}", "ts": 1785000000.0, "label": "<script>alert(1)</script>",
            "model": "<img onerror=x>", "config": {"upstream": "<b>x</b>"}, "env": {},
            "results": {"summary": {"n_total": 1, "n_success": 1,
                                    "served_models": ["<svg onload=y>"]}},
        }
    for runs in ([mk(1)], [mk(1), mk(2)]):          # single-cell and comparison paths
        html = p._bench_report_html(runs, [p._bench_report_row(r) for r in runs])
        for raw in ("<script>alert(1)</script>", "<img onerror=x>", "<svg onload=y>"):
            assert raw not in html, raw
        assert "&lt;" in html, "nothing was escaped at all"


# ---- cache / concurrency axes -------------------------------------------------------------

def test_cache_axis_expands_and_maps_to_randomize():
    """cold = salted prompts so nothing can be reused; cached = one identical prompt. Comparing
    the pair is the only way to see whether a backend's prefix caching is doing anything —
    caching silently disabled otherwise reads as ordinary slowness."""
    cells = p._bench_expand_matrix([{"model": "m", "upstream": "vllm"}],
                                   {"cache": ["cold", "cached"]})
    assert [c["cache"] for c in cells] == ["cold", "cached"]
    assert all(c["upstream"] == "vllm" for c in cells)


def test_concurrency_axis_expands():
    cells = p._bench_expand_matrix(["m"], {"concurrency": [1, 4]})
    # concurrency 1 is the default and stays off the label; 4 is recorded.
    assert [c.get("concurrency") for c in cells] == [None, 4]


def test_full_report_matrix_size_is_what_the_ui_claims():
    """3 contexts x 2 thinking x 2 cache x 2 concurrency = 24 cells per model."""
    cells = p._bench_expand_matrix(
        [{"model": "m", "upstream": "ollama"}],
        {"prompt_tokens": [0, 8000, 32000], "thinking": ["off", "on"],
         "cache": ["cold", "cached"], "concurrency": [1, 4]})
    assert len(cells) == 24


def test_cell_label_carries_the_distinguishing_axes():
    label = p._bench_cell_label({"model": "m", "upstream": "vllm", "prompt_tokens": 32000,
                                 "thinking": "off", "cache": "cached", "concurrency": 4})
    for bit in ("m", "vllm", "32k", "off", "cached", "4"):
        assert bit in label


def test_submit_rejects_an_unknown_cache_value(client):
    r = client.post("/__proxy/api/bench/run", json={"model": "m", "cache": "warm-ish"})
    assert r.status_code == 400
    assert "cache" in r.json()["error"]


def test_suite_has_a_core_and_a_hard_tier():
    """The core tier matches the original study's 12 tasks and saturates for any capable model —
    it can confirm a model works but cannot rank two that both do. The hard tier exists to
    separate them; without it a report reads 100% for everything and decides nothing."""
    suite = p._BENCH_SUITES["coding-v1"]
    core = [t for t in suite if t.get("tier", "core") == "core"]
    hard = [t for t in suite if t.get("tier") == "hard"]
    assert len(core) == 12
    assert len(hard) >= 6
    assert len({t["id"] for t in suite}) == len(suite), "task ids must be unique"
    assert all(t["cases"] and t["entry"] for t in suite)


def test_quality_summary_scores_each_tier_separately():
    """A blended rate hides the thing you want: two models at 100% core are indistinguishable
    until you look at hard."""
    suite = [
        {"id": "easy", "tier": "core", "entry": "e", "cases": [1, 2]},
        {"id": "tough", "tier": "hard", "entry": "t", "cases": [1, 2]},
    ]
    rows = [
        {"task": "easy", "grade": {"passed": 2, "total": 2}},
        {"task": "tough", "grade": {"passed": 1, "total": 2}},
    ]
    q = p._bench_quality_summary(rows, suite)
    assert q["tiers"]["core"]["perfect_rate"] == 1.0
    assert q["tiers"]["hard"]["perfect_rate"] == 0.0
    # The blended number sits between them and tells you neither.
    assert 0 < q["perfect_rate"] < 1


def test_chart_labels_elide_the_middle_not_the_tail():
    """Sweep labels share a long prefix and differ at the end. Tail-truncation renders every row
    of a sweep identically, which is the one thing a comparison chart must not do."""
    rows = [{"label": "fake-model · @ollama · 4k ctx · think=off · cold", "v": 10},
            {"label": "fake-model · @ollama · 4k ctx · think=off · cached", "v": 12}]
    svg = p._bench_bar_svg(rows, "v", "X", "u")
    assert "cold" in svg and "cached" in svg


def test_report_pairs_cold_and_cached():
    def mk(cache, ttft):
        return {"id": "b_" + cache, "ts": 1785000000.0, "label": "m · " + cache, "model": "m",
                "config": {"cache": cache, "prompt_tokens": 32000, "thinking": "off",
                           "upstream": "vllm"},
                "env": {}, "results": {"summary": {"n_total": 1, "n_success": 1,
                                                   "ttft_ms": {"p50": ttft}}}}
    runs = [mk("cold", 6146.0), mk("cached", 300.0)]
    html = p._bench_report_html(runs, [p._bench_report_row(r) for r in runs])
    assert "Prompt cache" in html
    assert "cache is working" in html      # 20x speed-up, as in the vLLM prefix-caching finding


def test_report_flags_a_backend_with_no_cache_reuse():
    def mk(cache, ttft):
        return {"id": "b_" + cache, "ts": 1785000000.0, "label": "m · " + cache, "model": "m",
                "config": {"cache": cache, "prompt_tokens": 32000, "thinking": "off"},
                "env": {}, "results": {"summary": {"n_total": 1, "n_success": 1,
                                                   "ttft_ms": {"p50": ttft}}}}
    runs = [mk("cold", 320.0), mk("cached", 300.0)]
    html = p._bench_report_html(runs, [p._bench_report_row(r) for r in runs])
    assert "no measurable reuse" in html


# ---- sweep robustness ---------------------------------------------------------------------

def test_every_cell_carries_a_scalar_concurrency():
    """The bug that broke a 24-cell sweep on spark: concurrency was only copied into the child
    config when it differed from the default, so every concurrency-1 cell inherited the parent's
    [1, 4] LIST. int() on a list raises, the first child died before it could be marked running,
    the exception escaped a fire-and-forget task, and all 24 cells sat 'pending' forever with no
    error recorded anywhere."""
    cfg = {"prompt_tokens": [0, 8000], "thinking": ["off", "on"],
           "cache": ["cold", "cached"], "concurrency": [1, 4]}
    cells = p._bench_expand_matrix([{"model": "m", "upstream": "vllm"}], cfg)
    assert len(cells) == 16
    # Reproduce what the suite executor writes for each child.
    for axes in cells:
        child = dict(cfg)
        child["prompt_tokens"] = axes["prompt_tokens"]
        child["thinking"] = axes["thinking"]
        child["concurrency"] = axes.get("concurrency") or 1
        assert isinstance(child["concurrency"], int), axes
        int(child["concurrency"])   # must not raise


def test_scalar_coercion_survives_a_list_that_slips_through():
    """Defence in depth: a future axis that forgets to flatten shouldn't crash the cell, because
    a crashed cell explains nothing to whoever is waiting on it."""
    def _scalar(cfg, key, default):
        v = cfg.get(key, default)
        return (v[0] if v else default) if isinstance(v, list) else v

    assert _scalar({"concurrency": [1, 4]}, "concurrency", 1) == 1
    assert _scalar({"concurrency": []}, "concurrency", 1) == 1
    assert _scalar({"concurrency": 4}, "concurrency", 1) == 4
    assert _scalar({}, "concurrency", 1) == 1


def test_startup_fails_runs_left_queued_by_a_restart(tmp_path, monkeypatch):
    """A bench only exists as an in-memory task, so a restart strands it. Rows that can never
    advance must not keep claiming they're about to."""
    import sqlite3 as sq
    dbf = tmp_path / "b.db"
    conn = sq.connect(dbf)
    conn.executescript(p.SCHEMA_TABLE)
    for bid, st in (("b_1", "pending"), ("b_2", "running"), ("b_3", "done")):
        conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                     "VALUES (?, ?, 'm', '{}', ?)", (bid, 1.0, st))
    conn.commit()
    conn.close()

    monkeypatch.setattr(p, "DB_PATH", str(dbf))
    p.init_db()

    conn = sq.connect(dbf)
    rows = dict(conn.execute("SELECT id, status FROM bench_runs").fetchall())
    errs = dict(conn.execute("SELECT id, error FROM bench_runs").fetchall())
    conn.close()
    assert rows["b_1"] == "failed" and rows["b_2"] == "failed"
    assert rows["b_3"] == "done", "a finished run must be left alone"
    assert "restarted" in (errs["b_1"] or "")


# ---- report theming / usage report ---------------------------------------------------------

def test_reports_are_whitepapers_in_both_themes():
    """Reports became documents rather than dashboard panels: paper-light by default, dark via
    the viewer's OS preference, printable either way. Both report types share the one shell,
    and the charts take their colours from the same tokens — including the label halo, which
    strokes in --bg so text stays legible on top of data on either ground."""
    run = {"id": "b", "ts": 1785000000.0, "label": "m", "model": "m", "config": {}, "env": {},
           "results": {"summary": {"n_total": 1, "n_success": 1}}}
    bench = p._bench_report_html([run], [p._bench_report_row(run)])
    usage = p._stats_report_html({"overall": {"count": 0}}, {})
    for html in (bench, usage):
        assert "--bg:#FFFFFF" in html, "light paper default missing"
        assert "prefers-color-scheme: dark" in html, "no dark counterpart"
        assert "--bg:#121418" in html, "dark tokens missing"
        assert "@media print" in html, "not printable"
        # Self-contained: no external fetches of any kind.
        for external in ('src="http', 'href="http', "@import", "fonts.googleapis"):
            assert external not in html
    assert 'stroke="var(--bg)"' in p._SVG_HALO, "halo must stroke in the page colour"


def test_the_theme_toggle_and_the_media_query_cannot_disagree():
    """The OS preference applies a theme via the media query; the button stamps data-theme,
    which must win in both directions. Both paths are emitted from the same constants, and the
    toggle ships as the page's one self-contained script."""
    css = p._REPORT_CSS
    assert css.count(p._REPORT_TOKENS_DARK) == 2, "media-query and data-theme dark drifted"
    assert css.count(p._REPORT_TOKENS_LIGHT) == 2
    assert ':root[data-theme="dark"]' in css and ':root[data-theme="light"]' in css
    head = p._report_head("t", "e")
    assert 'id="themeflip"' in head
    assert "localStorage" in head
    assert "prefers-color-scheme: dark" in head


def test_usage_report_survives_an_empty_database():
    """A fresh install opening the report must not 500 on divide-by-zero or missing keys."""
    html = p._stats_report_html({"overall": {"count": 0, "prompt_tokens": 0,
                                             "completion_tokens": 0, "errors": 0}}, {})
    assert "Usage report" in html


def test_status_zero_is_not_shown_as_an_http_code():
    """0 is what a request that never completed records. Rendering it as a status sends people
    looking for a status-0 response that doesn't exist."""
    assert p._status_label(0) == p._status_label(None)
    assert "aborted" in p._status_label(0)
    assert p._status_label(200) == "200"


def test_tier_section_is_omitted_when_a_run_predates_tiers():
    """Older runs carry no tier data; a heading plus a paragraph describing an absent table is
    worse than no section at all."""
    run = {"id": "b", "ts": 1785000000.0, "label": "m", "model": "m",
           "config": {"suite": "coding-v1"}, "env": {},
           "results": {"summary": {"n_total": 1, "n_success": 1,
                                   "quality": {"perfect_rate": 1.0, "tasks": [{"task": "x", "perfect_rate": 1.0}]}}}}
    html = p._bench_report_html([run], [p._bench_report_row(run)])
    assert "Per-task correctness" in html
    assert "Correctness by tier" not in html


def test_long_context_cells_that_cannot_fit_are_skipped(monkeypatch):
    """A prompt larger than the window returns 200 with no tokens — indistinguishable from a fast
    success without the empty-completion check, and it burns a full cold prefill to learn
    nothing. At 256K that is minutes per request."""
    cells = p._bench_expand_matrix(
        [{"model": "small", "upstream": "vllm"}],
        {"prompt_tokens": [32000, 131072, 262144], "cache": ["cold", "cached"]})
    assert len(cells) == 6
    index = {"vllm:small": {"model": "small", "upstream": "vllm", "loaded": True,
                            "max_context": 40960}}
    # Same arithmetic the suite executor applies: reply budget plus tokeniser headroom.
    fits = []
    for c in cells:
        meta = p._bench_resolve_model(index, c["model"], c.get("upstream", ""))
        window = meta.get("loaded_context") or meta.get("max_context")
        budget = (window - 512) * 0.9 if window else None
        if not (budget and c["prompt_tokens"] > budget):
            fits.append(c)
    assert {c["prompt_tokens"] for c in fits} == {32000}, "128K/256K must not run on a 40K window"


def test_resolve_model_prefers_the_exact_backend_pair():
    index = {
        "vllm:m": {"model": "m", "upstream": "vllm", "loaded": True, "max_context": 262144},
        "lmstudio:m": {"model": "m", "upstream": "lmstudio", "loaded": False, "max_context": 32768},
    }
    assert p._bench_resolve_model(index, "m", "lmstudio")["max_context"] == 32768
    assert p._bench_resolve_model(index, "m", "vllm")["max_context"] == 262144
    # Without an upstream, a loaded copy wins over an unloaded one.
    assert p._bench_resolve_model(index, "m")["upstream"] == "vllm"


def test_eviction_spares_the_model_about_to_run(monkeypatch):
    """Evicting the target and then immediately reloading it would add a cold load to the very
    measurement the eviction exists to clean up."""
    calls = []

    class FakeResp:
        def json(self):
            return {"models": [{"name": "a"}, {"name": "target"}, {"name": "b"}]}

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return FakeResp()
        async def post(self, url, json=None): calls.append(json["model"])

    monkeypatch.setattr(p.httpx, "AsyncClient", lambda **kw: FakeClient())
    unloaded = asyncio.run(p._bench_evict_ollama(keep="target"))
    assert "target" not in calls, "the model about to be measured must not be evicted"
    assert set(calls) == {"a", "b"}
    assert set(unloaded) == {"a", "b"}


def test_eviction_reports_what_it_unloaded():
    """Silent eviction of someone else's model is not acceptable; the run records it."""
    import inspect
    src = inspect.getsource(p._bench_execute)
    assert "evicted_before_run" in src, "eviction must be recorded in the run environment"


# ---- model residency control ---------------------------------------------------------------

def test_lms_control_refuses_a_remote_lm_studio(monkeypatch):
    """The lms CLI drives the LM Studio on *this* machine. With a remote LMSTUDIO_URL that would
    load models on the wrong host — quietly, and with the proxy reporting success."""
    monkeypatch.setattr(p, "LMSTUDIO_URL", "http://192.168.6.50:1234")
    assert p._lms_is_local() is False
    ok, why = p._lms_available()
    assert ok is False and "only controls a local instance" in why


def test_lms_control_accepts_a_local_lm_studio(monkeypatch):
    monkeypatch.setattr(p, "LMSTUDIO_URL", "http://127.0.0.1:1234")
    assert p._lms_is_local() is True


def test_capabilities_report_per_backend(client):
    """The UI must not offer buttons that cannot work, and must say why when it can't."""
    caps = client.get("/__proxy/api/control/models/capabilities").json()
    assert caps["ollama"]["load"] and caps["ollama"]["unload"]
    # vLLM depends on finding a local container; either way it explains itself.
    v = caps["vllm"]
    assert isinstance(v["load"], bool)
    assert v["reason"] or v["how"], "vLLM must say how it works or why it cannot"
    assert "one model for the life of the server" in v["note"]


def test_vllm_control_refuses_a_remote_server(monkeypatch):
    """Stopping a *local* container because a remote vLLM is unreachable would be both wrong and
    destructive — it would kill a different service than the one being asked about."""
    monkeypatch.setattr(p, "VLLM_URL", "http://192.168.6.50:8001")
    assert p._upstream_is_local(p.VLLM_URL) is False
    assert asyncio.run(p._vllm_container()) is None


def test_model_control_rejects_vllm_without_a_container(client, monkeypatch):
    async def no_container(*a, **k):
        return None
    monkeypatch.setattr(p, "_vllm_container", no_container)
    for path in ("load", "unload"):
        r = client.post(f"/__proxy/api/control/models/{path}",
                        json={"model": "x", "upstream": "vllm"})
        assert r.status_code == 501, path


def test_model_control_requires_a_name(client):
    assert client.post("/__proxy/api/control/models/load", json={}).status_code == 400
    assert client.post("/__proxy/api/control/models/unload", json={}).status_code == 400


def test_starting_vllm_does_not_require_a_model_name(client, monkeypatch):
    """Starting the vLLM container is the whole operation — the server already knows its model.
    Requiring a name here made it impossible to start vLLM through the proxy at all, which was
    only discovered after stopping it."""
    async def fake_container(*a, **k):
        return "qwen-vllm"
    async def fake_run(args, timeout=120.0):
        return 0, "qwen-vllm"
    async def fake_ready(t, *a, **k):
        return True
    monkeypatch.setattr(p, "_vllm_container", fake_container)
    monkeypatch.setattr(p, "_run_cmd", fake_run)
    monkeypatch.setattr(p, "_vllm_ready", fake_ready)
    monkeypatch.setattr(p, "_docker_bin", lambda: "/usr/bin/docker")
    r = client.post("/__proxy/api/control/models/load", json={"upstream": "vllm"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["started_container"] == "qwen-vllm" and body["ready"] is True
    # And a name is still required where one actually means something.
    assert client.post("/__proxy/api/control/models/load",
                       json={"upstream": "ollama"}).status_code == 400


def test_vllm_switch_refuses_a_container_that_does_not_publish_our_port(client, monkeypatch):
    """Starting a container that binds a different port would succeed at the Docker level and
    then never be reachable — a success message for a server the proxy cannot talk to."""
    async def configs(*a, **k):
        return [{"container": "other-vllm", "running": False, "serves_port": False,
                 "model": "x", "checkpoint": "x", "quant": None, "max_model_len": None,
                 "kv_cache_dtype": None, "prefix_caching": False, "args": ""}]
    monkeypatch.setattr(p, "_vllm_configs", configs)
    monkeypatch.setattr(p, "_docker_bin", lambda: "/usr/bin/docker")
    r = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "vllm", "container": "other-vllm"})
    assert r.status_code == 409
    assert "would not make it reachable" in r.json()["error"]


def test_vllm_switch_reports_unknown_containers_with_the_alternatives(client, monkeypatch):
    async def configs(*a, **k):
        return [{"container": "qwen-vllm", "running": True, "serves_port": True, "model": "q",
                 "checkpoint": "q", "quant": None, "max_model_len": None, "kv_cache_dtype": None,
                 "prefix_caching": False, "args": ""}]
    monkeypatch.setattr(p, "_vllm_configs", configs)
    monkeypatch.setattr(p, "_docker_bin", lambda: "/usr/bin/docker")
    r = client.post("/__proxy/api/control/models/load",
                    json={"upstream": "vllm", "container": "nope"})
    assert r.status_code == 404
    assert r.json()["available"] == ["qwen-vllm"]


def test_quant_falls_back_when_the_checkpoint_is_a_bind_mount():
    """--model /model names nothing. The quant is still recoverable from the served name, the
    container name, or the host path that was mounted."""
    assert p._infer_quant("/model", "ornith-nvfp4", "ornith-vllm", []) == "NVFP4"
    assert p._infer_quant("/model", None, "x", ["/home/u/models/ornith-nvfp4"]) == "NVFP4"
    assert p._infer_quant("/model", "plain", "plain", ["/data/plain"]) is None


def test_explicit_container_list_excludes_everything_else(monkeypatch):
    """Auto-discovery matches on the name or image containing "vllm", which could sweep up a
    container the proxy has no business stopping. An explicit list means exactly those."""
    async def fake_run(args, timeout=120.0):
        if "ps" in args:
            return 0, ("qwen-vllm\tvllm/vllm-openai\trunning\t0.0.0.0:8001->8000/tcp\n"
                       "ornith-vllm\tvllm/vllm-openai\texited\t\n"
                       "someone-elses-vllm\tvllm/vllm-openai\trunning\t0.0.0.0:9999->8000/tcp")
        if "Config.Cmd" in args[-1]:
            return 0, '["--served-model-name","m"]'
        if "PortBindings" in args[-1]:
            return 0, '{"8000/tcp":[{"HostPort":"8001"}]}'
        return 0, "[]"
    monkeypatch.setattr(p, "_run_cmd", fake_run)
    monkeypatch.setattr(p, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(p, "VLLM_URL", "http://localhost:8001")

    monkeypatch.setattr(p, "load_rules_config", lambda: {})
    assert {c["container"] for c in asyncio.run(p._vllm_configs())} == {
        "qwen-vllm", "ornith-vllm", "someone-elses-vllm"}, "auto-discovery takes all of them"

    monkeypatch.setattr(p, "load_rules_config",
                        lambda: {"model_control": {"vllm_containers": ["qwen-vllm", "ornith-vllm"]}})
    assert {c["container"] for c in asyncio.run(p._vllm_configs())} == {"qwen-vllm", "ornith-vllm"}


def test_vllm_container_prefers_the_running_one(monkeypatch):
    """qwen-vllm and ornith-vllm are both configured for port 8001, one running at a time.
    Returning the first match made "stop vLLM" target whichever sorted first — the bench once
    stopped the already-exited container, reported success, and left the running one's ~45 GB
    resident while a 123B model tried to load beside it."""
    monkeypatch.setattr(p, "VLLM_URL", "http://localhost:8001")
    monkeypatch.setattr(p, "_docker_bin", lambda: "docker")

    async def fake_run(cmd, timeout, max_chars=200000):
        if cmd[1] == "ps":
            return 0, "qwen-vllm\nornith-vllm\n"
        return 0, ('/qwen-vllm\tfalse\t{"8000/tcp":[{"HostPort":"8001"}]}\n'
                   '/ornith-vllm\ttrue\t{"8000/tcp":[{"HostPort":"8001"}]}\n')

    monkeypatch.setattr(p, "_run_cmd", fake_run)
    assert asyncio.run(p._vllm_container()) == "ornith-vllm"


def test_vllm_container_falls_back_to_first_configured_when_none_runs(monkeypatch):
    monkeypatch.setattr(p, "VLLM_URL", "http://localhost:8001")
    monkeypatch.setattr(p, "_docker_bin", lambda: "docker")

    async def fake_run(cmd, timeout, max_chars=200000):
        if cmd[1] == "ps":
            return 0, "qwen-vllm\nornith-vllm\n"
        return 0, ('/qwen-vllm\tfalse\t{"8000/tcp":[{"HostPort":"8001"}]}\n'
                   '/ornith-vllm\tfalse\t{"8000/tcp":[{"HostPort":"8001"}]}\n')

    monkeypatch.setattr(p, "_run_cmd", fake_run)
    assert asyncio.run(p._vllm_container()) == "qwen-vllm"
