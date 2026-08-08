"""The comparison table shows what differs, and its headers match its cells.

Two failures this pins:

- Seventeen columns, five of which held the same value in every row, next to a Configuration
  column that restated most of the others. The two numbers a reader wanted were off the
  right-hand edge.
- The header inserted the graded pair before "vs best" while every row appended it after, so
  the ratio printed under "Fully correct" and each quality figure sat one column left of its
  name. Assembled separately, they had drifted.
"""
import re

from ai_proxy import proxy as P


def _run(bench_id, *, prompt=32000, cache="cold", model="/m/big-model-00001-of-00003.gguf",
         graded=True, upstream="llamacpp"):
    return {
        "id": bench_id, "ts": 1_780_000_000, "model": model,
        "label": f"{model} · @{upstream} · {prompt // 1000}k prompt · {cache}",
        "config": {"upstream": upstream, "prompt_tokens": prompt, "cache": cache,
                   "concurrency": 1, "suite": "coding-v1" if graded else None},
        "env": {"gpus": []},
        "results": {"summary": {
            "n_success": 36, "n_total": 36, "served_models": [model],
            "ttft_ms": {"p50": 305.0}, "decode_tps": {"p50": 17.6},
            "total_ms": {"p50": 8516.0}, "completion_tokens": {"mean": 166},
            "quality": ({"perfect_rate": 0.94, "case_pass_rate": 0.93,
                         # Real per-task rows: the block that renders these reassigned a name
                         # the shared-settings block also used, and an empty list skipped it
                         # entirely — so the tests passed while the page 500'd.
                         "tasks": [{"task": "binary_search", "perfect_rate": 1.0},
                                   {"task": "roman", "perfect_rate": 0.5}]}
                        if graded else {}),
        }},
    }


def _render(runs):
    rows = [P._bench_report_row(r) for r in runs]
    return P._bench_report_html(runs, rows)


def _first_table(html):
    t = re.search(r"<table.*?</table>", html.split("<h2>Results</h2>")[-1], re.S).group(0)
    head = [re.sub(r"<[^>]+>", "", x).strip()
            for x in re.findall(r"<th[^>]*>(.*?)</th>", t.split("</thead>")[0], re.S)]
    first = t.split("<tbody>")[1]
    cells = [re.sub(r"<[^>]+>", "", x).strip()
             for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", first.split("</tr>")[0], re.S)]
    return head, cells


def test_header_and_row_have_the_same_shape():
    """The alignment bug: every figure was one column away from its name."""
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    head, cells = _first_table(_render(runs))
    assert len(head) == len(cells), f"{len(head)} headers vs {len(cells)} cells"


def test_the_ratio_sits_under_vs_best():
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    head, cells = _first_table(_render(runs))
    assert cells[head.index("vs best")].endswith("x")
    assert "%" in cells[head.index("Fully correct")]


def test_the_axes_are_the_identity_of_a_row():
    """A Configuration column beside the axis columns restated them word for word: the row read
    "cold · 32,000 | cold | 32,000 | ...". The axes are what names a cell."""
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    head, cells = _first_table(_render(runs))
    assert "Configuration" not in head
    assert head[0] in ("Cache", "Prompt")
    assert cells[0] in ("cold", "32,000")


def test_a_comparison_with_no_varying_axis_still_names_its_rows():
    head, _ = _first_table(_render([_run("a"), _run("b")]))
    assert head[0] == "Configuration"


def test_settings_shared_by_every_cell_leave_the_table():
    # Model, quant, backend and thinking are identical across a single-model sweep; printing
    # them once is the difference between 7 columns and 17.
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    head, _ = _first_table(_render(runs))
    for gone in ("Model", "Backend", "Think"):
        assert gone not in head, f"{gone} is the same in every row and still has a column"
    assert len(head) <= 10, f"{len(head)} columns is a horizontal scrollbar, not a table"
    assert "Prompt" in head, "the axis that actually varies must be shown"


def test_what_left_the_table_is_stated_above_it():
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    held = html.split("<h2>Results</h2>")[0]
    assert "Held constant" in held
    assert "llamacpp" in held, "the shared backend has to be recorded somewhere"


def test_a_sweep_whose_axes_all_collapsed_says_so():
    """Six cells with identical settings is what a prompt sweep plus a graded suite produced.
    Silently printing six near-identical rows invites reading noise as a finding."""
    html = _render([_run("a"), _run("b"), _run("c")])
    assert "identical settings" in html


def test_a_real_axis_is_not_reported_as_inert():
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "identical settings" not in html


def test_no_full_paths_survive_into_the_page():
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "/m/big-model" not in html, "a raw path forces every table wider than the page"
    assert "big-model" in html, "...but the model still has to be identifiable"


def test_the_warm_up_footer_uses_short_names_too():
    """The last place a raw path survived: six of them in one paragraph at the foot of the page."""
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    for r in runs:
        r["results"]["summary"]["warmup_ms"] = 900_000.0
    html = _render(runs)
    assert "Cold-start cost" in html
    assert "/m/big-model" not in html
    # And it ranks: the slower cold start must come first in the table.
    seg = html.split("Cold-start cost")[-1]
    assert seg.index("Cold start") < seg.index("</table>")


def test_a_single_run_still_renders():
    html = _render([_run("solo")])
    assert "Configuration" in html and "vs best" not in html


def test_the_per_task_table_renders_alongside_the_summary():
    """This path reassigned `labels`, which the Held-constant block also read. With no tasks it
    never ran, so every test passed and the deployed page returned 500."""
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "binary_search" in html and "Held constant" in html


def test_a_starved_cell_is_named_under_the_table(client):
    """A cell that ran short of memory still produces numbers; they are just numbers about the
    machine. Marking it in the table with a symbol does not say loudly enough that it should not
    be compared with the rest."""
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    runs[1]["env"]["memory_warning"] = "only 28 GB free after stopping others; 44 GB wanted"
    html = _render(runs)
    assert "Measured under memory pressure" in html
    assert "28 GB free" in html
    assert "not comparable" in html


def test_a_clean_run_says_nothing_about_memory(client):
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "memory pressure" not in html


def test_the_same_weights_on_two_engines_compare_as_one_model(client):
    """Ollama names the default tag qwen3-coder-next:latest; the vLLM container serving the same
    checkpoint calls it qwen3-coder-next. Compared as written, the report says model AND backend
    both vary, and the difference cannot be attributed to either."""
    a = _run("a", model="qwen3-coder-next:latest", upstream="ollama")
    b = _run("b", model="qwen3-coder-next", upstream="vllm")
    rows = [P._bench_report_row(r) for r in (a, b)]
    varying, constant, _ = P._bench_axis_split(rows, [a, b])
    assert varying == ["backend"], varying
    assert constant["model"] == "qwen3-coder-next"


def test_genuinely_different_tags_stay_apart(client):
    """:latest means "the default"; :30b and :tuned are different weights and must not collapse."""
    assert P._bench_model_identity("qwen3-coder:30b") == "qwen3-coder:30b"
    assert P._bench_model_identity("qwen3-coder:tuned") == "qwen3-coder:tuned"
    assert P._bench_model_identity("qwen3-coder-next:latest") == "qwen3-coder-next"
    assert P._bench_model_identity(
        "/m/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf"
    ) == "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"


def test_the_picker_says_where_else_a_model_can_run(client):
    index = {
        "ollama:qwen3-coder-next:latest": {"model": "qwen3-coder-next:latest", "upstream": "ollama"},
        "vllm:qwen3-coder-next": {"model": "qwen3-coder-next", "upstream": "vllm"},
        "ollama:gemma4:26b": {"model": "gemma4:26b", "upstream": "ollama"},
    }
    P._bench_annotate_engines(index)
    assert index["ollama:qwen3-coder-next:latest"]["also_on"] == ["vllm"]
    assert index["vllm:qwen3-coder-next"]["also_on"] == ["ollama"]
    assert "also_on" not in index["ollama:gemma4:26b"]


# ---- what it costs to have the model, not just to use it ------------------------------------

def _with_env(run, **env):
    run["env"].update(env)
    return run


def test_load_and_footprint_get_their_own_columns(client):
    """A model that decodes quickly but takes seven minutes to load is a different proposition
    from one ready in forty seconds, and the decode column cannot say so."""
    a = _with_env(_run("a", prompt=32000), load_ms=41_000, resident_mb=18_600)
    b = _with_env(_run("b", prompt=131072), load_ms=418_000, resident_mb=91_000)
    head, cells = _first_table(_render([a, b]))
    assert "Load" in head and "Resident" in head
    assert cells[head.index("Load")] == "41 s"
    assert cells[head.index("Resident")] == "18.2 GB"


def test_a_run_that_loaded_nothing_grows_no_columns(client):
    """Otherwise every speed-only comparison sprouts two columns of dashes."""
    head, _ = _first_table(_render([_run("a", prompt=32000), _run("b", prompt=131072)]))
    assert "Load" not in head and "Resident" not in head


def test_the_columns_survive_a_cell_that_missed_them(client):
    """One backend reports resident size per model and the others do not; a gap must not shift
    every figure one column left."""
    a = _with_env(_run("a", prompt=32000), load_ms=41_000, resident_mb=18_600)
    b = _run("b", prompt=131072)          # no env at all
    head, cells = _first_table(_render([a, b]))
    assert len(head) == len(cells)


# ---- a 38-cell sweep has to stay readable ---------------------------------------------------

def _graded(bench_id, model, upstream, perfect, decode, tasks):
    r = _run(bench_id, model=model, upstream=upstream)
    q = r["results"]["summary"]["quality"]
    q["perfect_rate"] = perfect
    q["tasks"] = [{"task": t, "perfect_rate": v} for t, v in tasks.items()]
    r["results"]["summary"]["decode_tps"]["p50"] = decode
    return r


def test_the_per_task_table_does_not_grow_a_column_per_cell(client):
    """One column per cell put 38 columns and a 60-character heading in each; the table could
    not render. Cells are rows everywhere; columns are for metrics."""
    runs = [_graded(f"r{i}", f"model-{i}", "ollama", 0.9, 60 - i,
                    {"calculator": 0.0 if i % 2 else 1.0, "binary_search": 1.0})
            for i in range(12)]
    html = _render(runs)
    # Just that table: the method section further down legitimately names every task.
    after = html.split("Per-task correctness")[-1]
    tbl = re.search(r"<table.*?</table>", after, re.S).group(0)
    head = [x for x in re.findall(r"<th[^>]*>(.*?)</th>", tbl.split("</thead>")[0], re.S)]
    assert len(head) <= 4, f"{len(head)} columns in the per-task table"
    # ...and the task nothing failed is summarised away rather than given a row of 100%s.
    assert "binary_search" not in tbl.split("<tbody>")[1]
    assert "calculator" in tbl


def test_a_task_nobody_missed_is_not_given_a_row(client):
    runs = [_graded(f"r{i}", f"m{i}", "ollama", 1.0, 60, {"easy": 1.0}) for i in range(4)]
    html = _render(runs)
    assert "separates nothing" in html


def test_the_tier_table_puts_cells_in_rows(client):
    runs = []
    for i in range(8):
        r = _graded(f"t{i}", f"m{i}", "ollama", 0.9, 50, {"x": 1.0})
        r["results"]["summary"]["quality"]["tiers"] = {
            "core": {"perfect_rate": 1.0}, "hard": {"perfect_rate": 0.5}}
        runs.append(r)
    html = _render(runs)
    if "Correctness by tier" in html:
        tbl = html.split("Correctness by tier")[-1]
        head = re.findall(r"<th[^>]*>(.*?)</th>", tbl.split("</thead>")[0], re.S)
        assert len(head) <= 4, f"{len(head)} columns in the tier table"


def test_the_report_leads_with_the_answer(client):
    """Thirty-eight rows in run order make the reader do the ranking by hand."""
    runs = [_graded("a", "slow-but-right", "ollama", 1.0, 18.2, {"x": 1.0}),
            _graded("b", "fast-but-wrong", "ollama", 0.3, 300.0, {"x": 0.0}),
            _graded("c", "best", "ollama", 1.0, 63.7, {"x": 1.0})]
    html = _render(runs)
    lede = html.split("<h2>")[0]
    assert "best" in lede and "leads" in lede
    assert "63.7" in lede
    head, cells = _first_table(html)
    assert "best" in " ".join(cells), "the table is not sorted to match the lede"


def test_a_field_only_some_backends_report_is_not_an_axis(client):
    """Absent is not a value. Size counted as varying because one backend reports it, giving a
    column of dashes with two real entries."""
    a = _run("a", model="x", upstream="ollama")
    b = _run("b", model="y", upstream="ollama")
    a["results"]["summary"]["size_mb"] = None
    rows = [P._bench_report_row(r) for r in (a, b)]
    rows[0]["size_mb"] = 16800
    rows[1]["size_mb"] = None
    varying, _c, _v = P._bench_axis_split(rows, [a, b])
    assert "size" not in varying


def test_many_axes_collapse_into_one_column(client):
    """Six axis columns plus eight metrics pushed the numbers off the right edge. Past about
    three axes the compound name reads better than six narrow columns."""
    runs = []
    for i in range(6):
        r = _run(f"m{i}", model=f"model-{i}", upstream="ollama" if i % 2 else "vllm")
        r["config"]["cache"] = "cold" if i % 2 else "cached"
        r["config"]["thinking"] = "off" if i < 3 else "on"
        r["config"]["server_context"] = 32768 * (1 + i % 3)
        r["results"]["summary"]["quality"]["perfect_rate"] = 1.0 - i / 20
        runs.append(r)
    head, cells = _first_table(_render(runs))
    assert head[0] == "Configuration", head
    assert len(head) <= 10, f"{len(head)} columns is still a scrollbar"
    assert len(head) == len(cells)


def test_few_axes_still_get_their_own_columns(client):
    head, _ = _first_table(_render([_run("a", cache="cold"), _run("b", cache="cached")]))
    assert "Configuration" not in head
    assert "Cache" in head


def test_a_metric_identical_on_every_row_leaves_the_table(client):
    """Reply length was 166 tokens on all 38 cells because they answered the same suite. That
    is a property of the run, not a measurement, and it belongs stated once."""
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    html = _render(runs)
    head, _ = _first_table(html)
    assert "Tokens" not in head
    assert "Reply length" in html.split("<h2>Results</h2>")[0]


def test_a_metric_that_differs_keeps_its_column(client):
    a, b = _run("a", prompt=32000), _run("b", prompt=131072)
    b["results"]["summary"]["completion_tokens"] = {"mean": 900}
    head, _ = _first_table(_render([a, b]))
    assert "Tokens" in head


def test_the_report_says_what_was_actually_tested(client):
    """Six months on, nobody remembers what coding-v1 contained, and a number is only
    interpretable if you know what produced it."""
    runs = [_graded("a", "m1", "ollama", 1.0, 60, {"calculator": 1.0}),
            _graded("b", "m2", "ollama", 0.5, 30, {"calculator": 0.0})]
    html = _render(runs)
    assert "What was tested" in html
    assert "coding-v1" in html
    assert "Fully correct" in html and "every" in html          # what the score means
    assert "not a sandbox" in html                              # the honest caveat
    assert "calculator" in html.split("What was tested")[-1]    # the task list


def test_the_method_section_is_absent_without_a_suite(client):
    html = _render([_run("a", graded=False), _run("b", graded=False)])
    assert "What was tested" not in html


def test_a_graded_comparison_carries_the_trade_off_chart(client):
    """A stray `scatter_html = ""` after the charts block once dropped the scatter from every
    multi-cell report — silently, because nothing referenced it again before the f-string."""
    runs = [_graded("a", "m1", "ollama", 1.0, 60, {"x": 1.0}),
            _graded("b", "m2", "ollama", 0.5, 30, {"x": 0.0}),
            _graded("c", "m3", "ollama", 0.8, 90, {"x": 0.5})]
    html = _render(runs)
    assert "The trade-off" in html
    assert 'aria-label="Correctness against output rate"' in html
    # The frontier and the hover names are what make it worth having.
    assert "stroke-dasharray" in html
    # Anchor on the chart's own h2 — "The trade-off — explore" now also exists further down.
    assert "<title>" in html.split("<h2>The trade-off</h2>")[-1].split("</svg>")[0]
    # And it sits before the results table, not buried after the bars.
    assert html.index("The trade-off") < html.index("<h2>Results</h2>")


def test_the_speed_chart_is_seconds_and_stops_at_the_leaders(client):
    """Tokens-per-second rankings became seconds-to-a-finished-answer: the only speed figure
    in units a person feels, capped at the leaders with the table carrying the field."""
    runs = [_graded(f"r{i}", f"model-{i}", "ollama", 0.9, 20 + i, {"x": 1.0})
            for i in range(20)]
    html = _render(runs)
    bars = [s for s in re.findall(r"<svg.*?</svg>", html, re.S)
            if "Seconds to a complete answer" in s]
    assert bars, "the answer-time chart is gone"
    assert bars[0].count("<rect") <= 12
    assert "the full field is in the table" in bars[0]
    assert "SECONDS UNTIL THE FULL ANSWER" in bars[0]


# ---- the whitepaper charts adapt to whatever the run contains -------------------------------

def test_the_scatter_annotates_the_actual_winner(client):
    """The annotations are computed, not remembered: a different run names a different model."""
    runs = [_graded("a", "underdog:7b", "ollama", 1.0, 55.0, {"x": 1.0}),
            _graded("b", "famous:70b", "ollama", 0.8, 20.0, {"x": 0.5}),
            _graded("c", "third:1b", "ollama", 0.4, 30.0, {"x": 0.0})]
    html = _render(runs)
    sc = [s for s in re.findall(r"<svg.*?</svg>", html, re.S)
          if "Correctness against output rate" in s][0]
    assert "underdog:7b" in sc
    assert "100% correct at 55 tok/s" in sc


def test_the_fast_but_wrong_callout_only_fires_when_it_is_true(client):
    fixture = [_graded("a", "steady", "ollama", 1.0, 50.0, {"x": 1.0}),
               _graded("b", "midfield", "ollama", 0.9, 40.0, {"x": 1.0}),
               _graded("c", "sprinter", "ollama", 0.2, 300.0, {"x": 0.0})]
    sc = [s for s in re.findall(r"<svg.*?</svg>", _render(fixture), re.S)
          if "Correctness against" in s][0]
    assert "the winner" in sc and "wrong 80% of the time" in sc

    # No configuration is dramatically faster than the winner: no callout, no stale story.
    calm = [_graded("a", "steady", "ollama", 1.0, 50.0, {"x": 1.0}),
            _graded("b", "close", "ollama", 0.9, 55.0, {"x": 1.0}),
            _graded("c", "slower", "ollama", 0.8, 30.0, {"x": 0.5})]
    sc2 = [s for s in re.findall(r"<svg.*?</svg>", _render(calm), re.S)
           if "Correctness against" in s][0]
    assert "the winner's speed" not in sc2


def test_scorecards_lead_with_the_recommendation(client):
    runs = [_graded("a", "pick-me:26b", "ollama", 1.0, 63.5, {"x": 1.0}),
            _graded("b", "second", "ollama", 1.0, 50.0, {"x": 1.0}),
            _graded("c", "fastwrong", "ollama", 0.1, 300.0, {"x": 0.0})]
    html = _render(runs)
    cards = html.split('class="cards"')[1].split("</div>\n")[0]
    assert "Run this" in cards and "pick-me:26b" in cards
    assert "Runner-up" in cards
    assert "fooled" in cards and "fastwrong" in cards


def test_the_fooled_card_stays_home_when_nothing_earns_it(client):
    runs = [_graded("a", "best", "ollama", 1.0, 60.0, {"x": 1.0}),
            _graded("b", "near", "ollama", 0.9, 55.0, {"x": 1.0})]
    html = _render(runs)
    assert "fooled" not in html


def test_bubbles_need_sizes_and_note_the_absent(client):
    """No sizes, no chart — a bubble chart of unknowns would be an invention. With sizes, the
    section renders and says why some models are missing rather than plotting them at zero."""
    runs = [_graded(f"m{i}", f"model-{i}", "ollama", 1.0 - i / 10, 60 - i, {"x": 1.0})
            for i in range(4)]
    assert "What memory buys" not in _render(runs)

    def sized(rs):
        rows = [P._bench_report_row(r) for r in rs]
        for i, r in enumerate(rows):
            r["size_mb"] = (i + 1) * 10_000
        return P._bench_report_html(rs, rows)

    html = sized(runs)
    assert "What memory buys" in html
    assert "are absent, not zero" in html


def test_the_engine_section_appears_only_with_a_real_pair(client):
    solo = [_graded("a", "m1", "ollama", 1.0, 60, {"x": 1.0}),
            _graded("b", "m2", "ollama", 0.9, 50, {"x": 1.0})]
    assert "Same weights, two engines" not in _render(solo)

    paired = [_graded("a", "twin:latest", "ollama", 1.0, 60, {"x": 1.0}),
              _graded("b", "twin", "vllm", 1.0, 62, {"x": 1.0}),
              _graded("c", "other", "ollama", 0.9, 50, {"x": 1.0})]
    for r in paired:
        r["config"]["cache"] = "cold"
    html = _render(paired)
    assert "Same weights, two engines" in html
    assert "TIME TO FIRST TOKEN" in html


def test_cold_start_gets_a_chart_when_more_than_one_model_loaded(client):
    runs = [_graded("a", "m1", "ollama", 1.0, 60, {"x": 1.0}),
            _graded("b", "m2", "ollama", 0.9, 50, {"x": 1.0}),
            _graded("c", "m3", "ollama", 0.8, 40, {"x": 1.0})]
    for i, r in enumerate(runs):
        r["results"]["summary"]["warmup_ms"] = (i + 1) * 40_000.0
    html = _render(runs)
    seg = html.split("Cold-start cost")[-1]
    assert "Seconds of loading before the first useful token" in seg
    assert seg.index("aria-label") < seg.index("<table")


def test_the_scatter_and_bubbles_name_both_axes(client):
    """Bare percentages up an unlabelled axis: the x-axes were titled, the y-axes were not."""
    runs = [_graded(f"m{i}", f"model-{i}", "ollama", 1.0 - i / 10, 60 - i, {"x": 1.0})
            for i in range(4)]
    rows = [P._bench_report_row(r) for r in runs]
    for i, r in enumerate(rows):
        r["size_mb"] = (i + 1) * 10_000
    html = P._bench_report_html(runs, rows)
    for aria in ("Correctness against output rate", "Memory spent against correctness bought"):
        svg = [s for s in re.findall(r"<svg.*?</svg>", html, re.S) if aria in s][0]
        assert "TASKS FULLY CORRECT" in svg, f"y-axis unnamed on {aria!r}"


# ---- a concurrency sweep has to read as one --------------------------------------------------

def _conc(bench_id, model, conc, decode, q=1.0):
    r = _graded(bench_id, model, "ollama", q, decode, {"x": 1.0})
    r["config"]["concurrency"] = conc
    return r


def test_concurrency_is_an_axis_when_it_varies(client):
    """Concurrency 1 emitted None to keep single-level runs quiet, so a 1-vs-4 sweep never
    registered as varying: twenty pairs of identically named rows above a Held-constant block
    claiming Parallel 4 — false for half the table."""
    runs = [_conc("a", "m1", 1, 60.0), _conc("b", "m1", 4, 41.0),
            _conc("c", "m2", 1, 50.0), _conc("d", "m2", 4, 30.0)]
    head, cells = _first_table(_render(runs))
    assert "Parallel" in head, head
    rows_html = _render(runs)
    held = rows_html.split("<h2>Results</h2>")[0]
    assert "Parallel" not in held.split("Held constant")[-1].split("</div>")[0]


def test_uniform_concurrency_is_held_constant_not_hidden(client):
    runs = [_conc("a", "m1", 4, 41.0), _conc("b", "m2", 4, 30.0)]
    html = _render(runs)
    assert "Parallel" not in _first_table(html)[0]
    held = html.split("<h2>Results</h2>")[0]
    assert "Parallel" in held


def test_engine_pairs_never_mix_concurrency_levels(client):
    """An engine comparison holds everything but the engine constant. Keyed on model and cache
    alone, vLLM at concurrency 4 could pair against Ollama at concurrency 1 and present the
    difference as the engine's."""
    runs = []
    for eng, model in (("ollama", "twin:latest"), ("vllm", "twin")):
        for conc in (1, 4):
            r = _graded(f"{eng}{conc}", model, eng, 1.0, 60.0, {"x": 1.0})
            r["config"]["concurrency"] = conc
            r["config"]["cache"] = "cached"
            runs.append(r)
    rows = [P._bench_report_row(r) for r in runs]
    pairs = P._bench_engine_pair_data(rows, runs)
    assert len(pairs) == 2, f"expected one pair per concurrency level, got {len(pairs)}"
    for key, v in pairs:
        assert len(key) > 2, "concurrency missing from the pair key"
        concs = {(run.get("config") or {}).get("concurrency")
                 for run in runs for u, r in v.items() if r is rows[runs.index(run)]}
        assert len(concs) == 1, "a pair mixed concurrency levels"


# ---- weighted standings: the trade-off made explicit -----------------------------------------

def _wfield():
    """The user's own example: 1 point more correct does not buy half the speed."""
    return [_graded("a", "slowbetter", "ollama", 0.87, 17.7, {"x": 0.9}),
            _graded("b", "fastclose", "ollama", 0.86, 36.0, {"x": 0.9}),
            _graded("c", "third", "ollama", 0.50, 20.0, {"x": 0.5})]


def test_weighted_standings_rank_speed_adjusted(client):
    data = P._bench_weighted_data([P._bench_report_row(r) for r in _wfield()])
    ranked = P._bench_weighted_rows(data, P._BENCH_WEIGHT_DEFAULT / 100.0)
    assert ranked[0][0]["m"] == "fastclose", \
        "at 70/30 the model 1 point behind at twice the speed must lead"
    quality_only = P._bench_weighted_rows(data, 0.0)
    assert quality_only[0][0]["m"] == "slowbetter", "at w=0 correctness alone must rule"


def test_weighted_section_declares_its_weights(client):
    html = _render(_wfield())
    assert "Weighted standings" in html
    assert "% × correctness +" in html and "% × relative speed" in html, \
        "the formula must be printed, not implied"
    assert 'id="wrange"' in html and 'type="range"' in html, "the weighting must be adjustable"
    assert f'value="{P._BENCH_WEIGHT_DEFAULT}"' in html
    body = html.split('id="wbody"')[1]
    assert body.index("fastclose") < body.index("slowbetter"), \
        "the server-rendered default order must already be speed-adjusted"


def test_scorecard_stat_lines_match_between_cards(client):
    html = _render(_wfield())
    cards = re.findall(r'<div class="card[^"]*"><p class="k">(Run this|Runner-up)[^<]*</p>.*?'
                       r'<p class="d">(.*?)</p>', html, re.S)
    lines = dict(cards)
    assert "Run this" in lines and "Runner-up" in lines
    for k in lines:
        assert "% correct" in lines[k], f"{k} must label its correctness"
        assert "answers in" in lines[k], f"{k} must state time to an answer"


# ---- task descriptions: the id alone says nothing --------------------------------------------

def test_the_method_section_describes_every_task(client):
    html = _render(_wfield())
    sect = html.split("What was tested")[-1]
    assert '<ul class="tl">' in sect
    assert P._BENCH_TASK_DESC["binary_search"] in sect


def test_imperfect_tasks_carry_their_description(client):
    runs = [_graded("a", "m1", "ollama", 0.5, 30, {"roman": 0.5, "binary_search": 1.0}),
            _graded("b", "m2", "ollama", 0.6, 40, {"roman": 0.7, "binary_search": 1.0})]
    html = _render(runs)
    sect = html.split("Per-task correctness")[-1].split("What was tested")[0]
    assert P._BENCH_TASK_DESC["roman"] in sect


# ---- chart labels must not print through each other ------------------------------------------

def _svg_text_collisions(svg):
    boxes = []
    for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"([^>]*)>(.*?)</text>', svg):
        x, y, attrs, txt = float(m[1]), float(m[2]), m[3], re.sub(r"<[^>]+>", "", m[4])
        anchor = (re.search(r'text-anchor="(\w+)"', attrs) or (None, "start"))[1]
        w = 6.6 * len(txt)
        x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        boxes.append((x0, y, x0 + w, txt))
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[3] and b[3] and abs(a[1] - b[1]) < 11 and a[0] < b[2] and b[0] < a[2]:
                hits.append((a[3], b[3]))
    return hits


def test_scatter_labels_do_not_collide_in_a_crowded_field(client):
    """Twenty models with half of them sharing the top band is the real shape of a full sweep;
    the labels must dodge, drop, or defer to the annotations — never overprint."""
    runs = [_graded(f"r{i}", f"model-with-a-name-{i}", "ollama",
                    1.0 if i < 6 else 0.85 + (i % 5) * 0.02, 12 + i * 9, {"x": 1.0})
            for i in range(16)]
    rows = [P._bench_report_row(r) for r in runs]
    svg = P._bench_scatter_svg(rows)
    hits = _svg_text_collisions(svg)
    assert not hits, f"overlapping chart text: {hits[:4]}"


def test_the_scorecard_verdict_is_the_weighted_one(client):
    """"Run this" and the Weighted standings must agree — one verdict, stated twice. Raw
    correctness crowned the model 1 point better at half the speed."""
    html = _render(_wfield())
    win = re.search(r'Run this[^<]*</p><p class="v">([^<]+)</p>', html)
    assert win and win.group(1) == "fastclose", win and win.group(1)
    assert re.search(r'Run this — \d+/\d+ weighted', html), \
        "the card must say it is a weighted verdict"
    for card in re.findall(r'<p class="k">(?:Run this[^<]*|Runner-up)</p>.*?<p class="d">([^<]*)</p>',
                           html, re.S):
        assert "score" in card, "every ranked card carries its score"


# ---- failure examples: what a percentage point is made of ------------------------------------

def test_failure_examples_show_the_call_and_expected_vs_got(client):
    runs = [_graded("a", "m1", "ollama", 0.9, 30, {"roman": 0.5}),
            _graded("b", "m2", "ollama", 1.0, 40, {"roman": 1.0})]
    runs[0]["results"]["rows"] = [
        {"seq": 1, "task": "roman",
         "grade": {"passed": 5, "total": 6,
                   "cases": [{"ok": True}] * 3 + [{"ok": False, "got": "IIII"}]
                            + [{"ok": True}] * 2}},
        {"seq": 2, "task": "roman", "grade": {"passed": 6, "total": 6}},
    ]
    runs[1]["results"]["rows"] = []
    html = _render(runs)
    assert "What the failures actually looked like" in html
    sect = html.split("What the failures actually looked like")[-1]
    assert "IIII" in sect and "expected" in sect, "got and expected must both appear"
    assert "to_roman(" in sect, "the failing call is reconstructed from the suite's own case"
    assert "m2" not in sect.split("</details>")[0], "clean configurations are not examples"


def test_failure_examples_surface_grader_errors_verbatim(client):
    runs = [_graded("a", "m1", "ollama", 0.5, 30, {"csv_escape": 0.0})]
    runs[0]["config"]["suite"] = "coding-v2"
    runs[0]["results"]["rows"] = [
        {"seq": 1, "task": "csv_escape",
         "grade": {"passed": 0, "total": 6, "error": "compile error: missing terminating"}},
    ]
    html = _render(runs)
    assert "compile error: missing terminating" in html


def test_failure_examples_include_the_code_that_failed(client):
    runs = [_graded("a", "m1", "ollama", 0.9, 30, {"roman": 0.5})]
    runs[0]["results"]["rows"] = [
        {"seq": 1, "task": "roman",
         "text": "Here you go:\n```python\ndef to_roman(n):\n    return 'IIII'\n```",
         "grade": {"passed": 5, "total": 6,
                   "cases": [{"ok": True}] * 3 + [{"ok": False, "got": "IIII"}]
                            + [{"ok": True}] * 2}},
    ]
    html = _render(runs)
    sect = html.split("What the failures actually looked like")[-1]
    assert "the code it wrote" in sect
    assert "return &#x27;IIII&#x27;" in sect or "return 'IIII'" in sect, \
        "the graded code must appear, escaped, inside the example"


def test_a_reply_with_no_code_block_shows_the_raw_reply(client):
    runs = [_graded("a", "m1", "ollama", 0.5, 30, {"roman": 0.0})]
    runs[0]["results"]["rows"] = [
        {"seq": 1, "task": "roman",
         "text": "I would be happy to help! First, let me explain Roman numerals...",
         "grade": {"passed": 0, "total": 6, "error": "no python code block in the response"}},
    ]
    html = _render(runs)
    sect = html.split("What the failures actually looked like")[-1]
    assert "no code block — raw reply" in sect
    assert "happy to help" in sect


def test_the_hardware_section_names_the_machine(client):
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    runs[0]["env"] = {"gpus": [{"name": "NVIDIA GB10", "mem_total_mb": 124610}],
                      "mem": {"total_mb": 124610},
                      "hw": {"cpu_model": "Grace C1", "cpu_cores": 20,
                             "os": "Ubuntu 24.04 (aarch64)", "kernel": "6.8.0"},
                      "ollama_version": "0.32.5", "proxy_version": "0.2.0"}
    html = _render(runs)
    sect = html.split("<h2>Hardware</h2>")[-1]
    for fact in ("NVIDIA GB10", "122 GB", "Grace C1", "20", "Ubuntu 24.04", "6.8.0", "0.32.5"):
        assert fact in sect, f"hardware section missing {fact}"
    assert "read from the host at report time" not in sect, \
        "run-time facts must not carry the backfill caveat"


def test_old_runs_backfill_static_hardware_with_a_caveat(client):
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "<h2>Hardware</h2>" in html
    assert "read from the host at report time" in html


# ---- the interactive layer: enhancement, never replacement -----------------------------------

def test_marks_carry_their_model_identity(client):
    html = _render(_wfield())
    assert html.count('data-m="') > 10, "charts and tables must tag their model marks"
    assert 'data-m="fastclose"' in html


def test_the_report_embeds_its_dataset_and_layer(client):
    html = _render(_wfield())
    assert 'id="report-data"' in html and '"rows":' in html
    assert "linked highlighting" in html or "setHL" in html
    assert 'classList.add("ix-on")' in html


def test_d3_is_inlined_not_fetched(client):
    html = _render(_wfield())
    assert "<script src=" not in html, "the report must stay a single offline file"
    if P._d3_source():
        assert "https://d3js.org v7" in html
        assert 'id="ix-scatter"' in html


def test_print_hides_the_interactive_extras(client):
    html = _render(_wfield())
    assert ".ix-only, .ixtip, tr.drill { display:none !important; }" in html


def test_per_task_rows_are_clickable_with_data(client):
    runs = [_graded("a", "m1", "ollama", 0.5, 30, {"roman": 0.5, "binary_search": 1.0}),
            _graded("b", "m2", "ollama", 0.6, 40, {"roman": 0.7, "binary_search": 1.0})]
    html = _render(runs)
    assert 'class="taskrow" data-task="roman"' in html
    assert '"tasks":' in html.split('id="report-data"')[1][:20000]


# ---- category winners: best correctness / speed / parallel ----------------------------------

def _cell(bench_id, model, upstream, *, conc=1, decode=60.0, ttft=400.0, perfect=0.9):
    r = _run(bench_id, model=model, upstream=upstream)
    r["label"] = f"{model} · @{upstream} · short · " + (f"{conc}×parallel" if conc > 1 else "t=0.0")
    r["config"]["concurrency"] = conc
    s = r["results"]["summary"]
    s["decode_tps"]["p50"] = decode
    s["ttft_ms"]["p50"] = ttft
    s["quality"]["perfect_rate"] = perfect
    return r


def _parallel_field():
    """The fleet in miniature, numbers shaped like the real coding-v3 matrix: gemma batches
    (TTFT flat, per-stream drops), qwen-on-ollama queues (decode identical, TTFT explodes)."""
    return [
        _cell("g1", "gemma", "ollama", conc=1, decode=63.0, ttft=500, perfect=0.90),
        _cell("g4", "gemma", "ollama", conc=4, decode=38.0, ttft=800, perfect=0.91),
        _cell("q1", "qwen", "ollama", conc=1, decode=60.0, ttft=400, perfect=0.72),
        _cell("q4", "qwen", "ollama", conc=4, decode=60.0, ttft=6200, perfect=0.72),
    ]


def test_parallel_groups_tell_batching_from_queueing(client):
    rows = [P._bench_report_row(r) for r in _parallel_field()]
    groups = {g["model"]: g for g in P._bench_parallel_groups(rows)}
    assert groups["gemma"]["serialized"] is False
    assert groups["gemma"]["agg"] == 38.0 * 4
    assert groups["qwen"]["serialized"] is True, "flat decode + 15x TTFT is a queue"
    assert groups["qwen"]["agg"] == 60.0, "a queue's throughput is one stream"


def test_category_winners_crown_the_right_models(client):
    html = _render(_parallel_field())
    sec = html.split("Category winners")[-1].split("<h2>")[0]
    assert "Most correct" in sec and "Fastest single stream" in sec
    assert "Best under 4× load" in sec
    # gemma wins all three here: correctness (91%), single stream (63 > 60), parallel (152)
    assert '<p class="k">Most correct</p><p class="v">gemma</p>' in sec
    assert '<p class="k">Fastest single stream</p><p class="v">gemma</p>' in sec
    assert '<p class="k">Best under 4× load</p><p class="v">gemma</p>' in sec
    assert "queues" in sec and "batches" in sec
    assert "152" in sec, "aggregate = per-stream × streams for the batcher"


def test_no_parallel_cells_means_no_parallel_card(client):
    runs = [_cell("a", "m1", "ollama", conc=1), _cell("b", "m2", "ollama", conc=1, decode=50)]
    html = _render(runs)
    assert "Best under" not in html
    assert "Category winners" in html, "correctness and speed cards still render"


# ---- category breakdown (full-v1 merges coding + agentic + security) -----------------------

def _mixed_run(bench_id, model, rates):
    """One run whose per-task rates span all three categories."""
    r = _run(bench_id, model=model, upstream="ollama")
    q = r["results"]["summary"]["quality"]
    q["tasks"] = [{"task": t, "perfect_rate": v} for t, v in rates.items()]
    q["perfect_rate"] = sum(rates.values()) / len(rates)
    return r


def test_category_table_splits_the_merged_suite(client):
    """The reason for merging: one number would average a model that codes well and obeys
    injected instructions into a single misleading score."""
    rates = {"roman": 1.0, "csv_line": 1.0,                       # coding
             "agent_chain": 0.5, "agent_bisect": 0.5,             # agentic
             "sec_fix_sqli": 0.4, "sec_exploit_sqli": 0.2,        # security (blue, red)
             "sec_agent_injection": 0.0}                          # security (blue)
    html = _render([_mixed_run("a", "m1", rates),
                    _mixed_run("b", "m2", {k: min(1.0, v + 0.3) for k, v in rates.items()})])
    sec = html.split("Results by category")[-1].split("<h2>")[0]
    assert "Coding" in sec and "Agentic" in sec and "Security" in sec
    assert "red" in sec and "blue" in sec, "security splits into finding vs fixing"
    assert "2 tasks" in sec        # per-category task counts are stated
    # the spread sentence names both ends rather than leaving the reader to diff columns
    assert "would describe a model that does not exist" in sec


def test_category_table_is_absent_for_a_single_category_run(client):
    html = _render([_graded("a", "m1", "ollama", 0.9, 60, {"roman": 0.5}),
                    _graded("b", "m2", "ollama", 0.8, 50, {"roman": 1.0})])
    assert "Results by category" not in html


# ---- cold start: booting the server vs the first request ------------------------------------

def test_cold_start_counts_the_boot_not_just_the_warm_up(client):
    """The measurement bug the first full-v2 sweep exposed: env.load_ms held only the warm-up
    request, so a vLLM cell that spent ~6 minutes loading weights before that request reported
    16 s — and the report compared it against an Ollama cold load as if that were the same
    thing."""
    ollama = _run("a", model="gemma4:26b", upstream="ollama")
    ollama["env"].update(load_ms=27_000, warmup_request_ms=27_000)
    vllm = _run("b", model="qwen3-coder-next", upstream="vllm")
    vllm["env"].update(load_ms=372_000, backend_start_ms=356_000, warmup_request_ms=16_000)
    rows = [P._bench_report_row(r) for r in (ollama, vllm)]
    assert rows[1]["backend_start_ms"] == 356_000
    html = P._bench_coldstart_split_html(rows)
    assert "Boot the server" in html and "First request" in html
    assert "356 s" in html and "16.0 s" in html and "372 s" in html
    # An Ollama-only field has no server to boot, so the split says nothing and is omitted.
    assert P._bench_coldstart_split_html([rows[0]]) == ""


def test_cold_start_section_explains_the_correction(client):
    ollama = _run("a", model="gemma4:26b", upstream="ollama")
    ollama["env"].update(load_ms=27_000, warmup_request_ms=27_000)
    ollama["results"]["summary"]["warmup_ms"] = 27_000.0
    vllm = _run("b", model="qwen3-coder-next", upstream="vllm")
    vllm["env"].update(load_ms=372_000, backend_start_ms=356_000, warmup_request_ms=16_000)
    vllm["results"]["summary"]["warmup_ms"] = 16_000.0
    html = _render([ollama, vllm])
    seg = html.split("Cold-start cost")[-1]
    assert "weight load" in seg and "understated" in seg
