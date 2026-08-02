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
