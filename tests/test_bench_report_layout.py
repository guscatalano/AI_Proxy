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
            "total_ms": {"p50": 8516.0}, "mean_tokens": {"p50": 166},
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


def test_settings_shared_by_every_cell_leave_the_table():
    # Model, quant, backend and thinking are identical across a single-model sweep; printing
    # them once is the difference between 7 columns and 17.
    runs = [_run("a", prompt=32000), _run("b", prompt=131072)]
    head, _ = _first_table(_render(runs))
    for gone in ("Model", "Backend", "Think"):
        assert gone not in head, f"{gone} is the same in every row and still has a column"
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


def test_a_single_run_still_renders():
    html = _render([_run("solo")])
    assert "Configuration" in html and "vs best" not in html


def test_the_per_task_table_renders_alongside_the_summary():
    """This path reassigned `labels`, which the Held-constant block also read. With no tasks it
    never ran, so every test passed and the deployed page returned 500."""
    html = _render([_run("a", prompt=32000), _run("b", prompt=131072)])
    assert "binary_search" in html and "Held constant" in html
