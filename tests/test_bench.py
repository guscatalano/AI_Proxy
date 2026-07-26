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
    async def fake_system_now():
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
    async def fake_system_now():
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
    async def fake_system_now():
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


def test_report_html_escapes_model_names():
    """Model names come from upstreams and end up in the page; they are not trusted markup."""
    run = {
        "id": "b_y", "ts": 1785000000.0, "label": "<script>alert(1)</script>",
        "model": "<img onerror=x>", "config": {}, "env": {},
        "results": {"summary": {"n_total": 1, "n_success": 1}},
    }
    html = p._bench_report_html([run], [p._bench_report_row(run)])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


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


def test_suite_has_twelve_tasks():
    """The original study graded 12 tasks; a 6-task suite is a weaker signal for the same cost
    in wall-clock, since latency dominates."""
    suite = p._BENCH_SUITES["coding-v1"]
    assert len(suite) == 12
    assert len({t["id"] for t in suite}) == 12
    assert all(t["cases"] and t["entry"] for t in suite)


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
