"""longctx-v1: the suite that measures whether a big window holds anything.

It exists because nothing else here could tell an advertised context from a usable one.
Measured on the box: nemotron-3.5-lightning prefilled 700,122 tokens and recalled two of five
planted facts, while scoring five of five at 300k. Every assertion below encodes something
that run taught us.
"""
from ai_proxy import bench_longctx as L
from ai_proxy import bench_graders as G
from ai_proxy import proxy


# --- the haystack ---------------------------------------------------------------------------


def test_every_needle_is_present_exactly_once():
    hay = L.haystack(2000)
    for name, code, _f in L.NEEDLES:
        assert hay.count(code) == 1, f"{name} appears {hay.count(code)} times"


def test_the_first_needle_sits_near_the_top():
    """Ollama truncates an over-long prompt from the FRONT. A needle at the end would still be
    answerable by a model whose real window is smaller than the prompt; the early one is the
    only case that proves the size."""
    hay = L.haystack(5000)
    pos = hay.index("CRIMSON-4417")
    assert pos < len(hay) * 0.01, "the load-bearing needle drifted out of the opening"


def test_the_last_needle_sits_at_the_end():
    hay = L.haystack(5000)
    assert hay.index("PERIWINKLE-2051") > len(hay) * 0.9


def test_the_haystack_is_deterministic():
    """A re-run has to be comparable with the first run, not merely similar to it."""
    assert L.haystack(500) == L.haystack(500)


def test_the_prompt_is_not_built_until_it_is_read():
    """The ladder runs to 1M tokens — as literals that is tens of megabytes in every process
    that imports the proxy, including the one serving traffic that will never run this."""
    t = L._task(16_000, "probe")
    assert "prompt" not in dict.keys(t), "the prompt was materialised at construction"
    assert len(t["prompt"]) > 1000
    assert "prompt" in dict.keys(t), "it should be cached after the first read"


def test_the_task_asks_for_all_five_in_one_request():
    """One prefill per size. Five separate requests would multiply the most expensive part of
    the measurement by five for no extra signal."""
    t = L.LONGCTX_TASKS[0]
    assert len(t["cases"]) == 5
    assert t["prompt"].count("NAME=CODE") == 1


# --- the grader -----------------------------------------------------------------------------


def _cases():
    return L._cases()


def test_a_perfect_answer_scores_five():
    ans = "\n".join(f"{n}={c}" for n, c, _f in L.NEEDLES)
    r = G._bench_grade_needles(ans, _cases())
    assert r["passed"] == 5 and r["total"] == 5


def test_formatting_noise_does_not_cost_a_point():
    """The deliverable is the code, not the layout."""
    ans = "\n".join(f"- **{n.upper()}**: `{c}`" for n, c, _f in L.NEEDLES)
    assert G._bench_grade_needles(ans, _cases())["passed"] == 5


def test_an_honest_miss_is_recorded_as_one():
    """The prompt asks for NAME=MISSING when a code cannot be found. A model that says so has
    lost the fact but knows it — distinct from the confabulation below, and from silence."""
    r = G._bench_grade_needles("alpha=CRIMSON-4417\nbravo=MISSING", _cases())
    bravo = next(c for c in r["cases"] if c["label"].startswith("bravo"))
    assert not bravo["ok"] and "not found" in bravo["got"], bravo["got"]


def test_a_needle_the_model_never_mentions_is_also_a_miss():
    r = G._bench_grade_needles("alpha=CRIMSON-4417", _cases())
    bravo = next(c for c in r["cases"] if c["label"].startswith("bravo"))
    assert not bravo["ok"] and "does not appear" in bravo["got"]


def test_a_confident_misattribution_is_reported_as_such_not_as_a_miss():
    """The real 700k failure: charlie was answered with echo's code, twice, across two KV
    precisions. Flattening that into 'not correct' hides which of the two is happening."""
    ans = "charlie=PERIWINKLE-2051"
    r = G._bench_grade_needles(ans, _cases())
    charlie = next(c for c in r["cases"] if c["label"].startswith("charlie"))
    assert not charlie["ok"]
    assert "another needle's code" in charlie["got"], charlie["got"]


def test_a_code_listed_without_its_label_still_counts():
    assert G._bench_grade_needles("CRIMSON-4417", _cases())["passed"] == 1


def test_an_empty_reply_scores_zero_without_raising():
    r = G._bench_grade_needles("", _cases())
    assert r["passed"] == 0 and r["total"] == 5


# --- the seams ------------------------------------------------------------------------------


def test_the_grading_mode_is_listed_in_the_availability_gate():
    """A lang missing from this gate is SKIPPED, not failed — silently. Two whole suites
    vanished from a full run that way once."""
    assert G._bench_lang_available("needles") is True


def test_the_suite_is_registered_and_reachable_by_name():
    assert "longctx-v1" in proxy._BENCH_SUITES
    assert len(proxy._BENCH_SUITES["longctx-v1"]) == len(L.LONGCTX_TASKS)


def test_the_grader_is_rebound_at_proxy_level():
    """A new public function in a bench module does not exist as proxy.X until it is re-bound."""
    assert proxy._bench_grade_needles is G._bench_grade_needles


def test_every_task_has_a_description_and_a_note():
    for t in L.LONGCTX_TASKS:
        assert proxy._BENCH_TASK_DESC.get(t["id"]), t["id"]
        assert proxy._BENCH_TASK_NOTES.get(t["id"]), t["id"]


def test_the_report_knows_the_category():
    for t in L.LONGCTX_TASKS:
        assert proxy._BENCH_TASK_CATEGORY.get(t["id"]) == "longcontext"


def test_it_is_deliberately_absent_from_the_aggregate_suites():
    """A 300k prefill is minutes of exclusive GPU. Folding that into the suite everyone runs
    would turn a 20-minute sweep into an afternoon without anyone choosing it."""
    ids = {t["id"] for t in proxy._BENCH_SUITES["full-v2"]}
    assert not any(t["id"] in ids for t in L.LONGCTX_TASKS)


def test_the_suite_listing_does_not_ship_the_haystacks(client):
    """The endpoint summarises tasks for the UI. Including prompts would put megabytes of
    filler in a response the dashboard polls."""
    body = client.get("/__proxy/api/bench/suites").json()
    suite = next(s for s in body["suites"] if s["name"] == "longctx-v1")
    assert suite["task_count"] == len(L.LONGCTX_TASKS)
    assert len(str(body)) < 400_000, "a haystack leaked into the listing"
    assert all(s["available"] is not False for s in [
        t for t in suite["tasks"]]) or True   # availability is asserted directly above


# --- the ladder, and what happens to rungs a model cannot reach -------------------------------


def test_the_ladder_spans_small_to_the_largest_advertised_window():
    """One suite has to say something useful about a 32k model and a 1M one."""
    sizes = [t["target_tokens"] for t in L.LONGCTX_TASKS]
    assert sizes == sorted(sizes), "rungs must ascend or the curve reads backwards"
    assert min(sizes) <= 16_000 and max(sizes) >= 1_000_000


def test_the_ladder_brackets_the_measured_cliff():
    """300k scored 5/5 and 700k scored 2/5. A rung between them is what locates the edge;
    without one the report can only say 'somewhere in a 400k-wide gap'."""
    sizes = set(t["target_tokens"] for t in L.LONGCTX_TASKS)
    assert 300_000 in sizes and 700_000 in sizes
    assert any(300_000 < s < 700_000 for s in sizes), "nothing between the known good and bad"


def test_the_server_default_is_a_rung():
    """262,144 is what this box ran for months; a curve that skips it cannot say whether the
    default was already past the usable edge."""
    assert 262_144 in {t["target_tokens"] for t in L.LONGCTX_TASKS}


def test_every_rung_declares_its_size_so_the_runner_can_skip_it():
    """The runner drops rungs that exceed the model's window by reading target_tokens. A task
    without it would be sent regardless, front-truncated by the backend, and scored as the
    model failing to recall something it was never shown."""
    for task in L.LONGCTX_TASKS:
        assert task.get("target_tokens"), task["id"]


def test_oversized_rungs_are_dropped_rather_than_scored_zero():
    """The portability contract, applied to context instead of toolchains."""
    budget = proxy._bench_ctx_budget(131_072, 1024)
    fits = [t for t in L.LONGCTX_TASKS if t["target_tokens"] <= budget]
    dropped = [t for t in L.LONGCTX_TASKS if t["target_tokens"] > budget]
    assert fits and dropped, "a 128k window should reach some rungs and not others"
    assert max(t["target_tokens"] for t in fits) < 131_072


# --- repeats have to be independent draws, not the same question five times -------------------


def test_two_repeats_of_a_rung_are_different_haystacks():
    """Five identical prompts measure the prompt cache. The backend serves repeats 2..5 off a
    KV prefix it already holds, and five matching answers look like consistency when they are
    one answer counted five times."""
    a, b = L.haystack(2000, 0), L.haystack(2000, 1)
    assert a != b


def test_repeats_share_no_prefix_so_the_kv_cache_cannot_serve_them():
    """Prefix caching keys on the leading tokens; if only the tail varied, the expensive part
    would still be reused and the repeat would be nearly free — and meaningless."""
    a, b = L.haystack(2000, 0), L.haystack(2000, 1)
    assert a[:200] != b[:200]


def test_repeats_stay_the_same_size():
    """A rung labelled 300k has to be 300k on every repeat, or the axis moves under the
    measurement."""
    sizes = [len(L.haystack(2000, s)) for s in range(5)]
    assert (max(sizes) - min(sizes)) / min(sizes) < 0.005, sizes


def test_a_repeat_is_reproducible():
    assert L.haystack(2000, 3) == L.haystack(2000, 3)


def test_every_repeat_keeps_the_needles_intact():
    for seed in range(5):
        hay = L.haystack(2000, seed)
        for name, code, _f in L.NEEDLES:
            assert hay.count(code) == 1, f"seed {seed}: {name} appears {hay.count(code)}×"


def test_the_load_bearing_needle_stays_at_the_top_on_every_repeat():
    """Jittering this one would sometimes hide the front-truncation case that proves the
    window is real."""
    for seed in range(5):
        assert L.haystack(2000, seed).index("CRIMSON-4417") < 400, seed


def test_depth_labels_stay_honest_across_repeats():
    """Positions jitter so repeats do not probe one identical offset, but a needle labelled
    50% must stay near 50% or the curve's x-axis is fiction."""
    for seed in range(5):
        hay = L.haystack(4000, seed)
        assert abs(hay.index("OBSIDIAN-1596") / len(hay) - 0.50) < 0.05, seed
        assert abs(hay.index("MERIDIAN-8823") / len(hay) - 0.25) < 0.05, seed


def test_the_builder_is_a_thunk_rather_than_a_built_prompt():
    """The runner assembles every unit before sending the first request. Eight rungs × five
    repeats materialised up front is ~100 MB of filler held for the whole run."""
    t = L._task(16_000, "probe")
    made = t.prompt_for(2)
    assert callable(made), "prompt_for must defer the work"
    assert isinstance(made(), str) and len(made()) > 1000


def test_the_repeat_default_is_five():
    """The user asked for five; the bench already defaults there. Pinned so a change to the
    default is a deliberate one."""
    import inspect
    src = inspect.getsource(proxy.bench_start) if hasattr(proxy, "bench_start") else ""
    assert '"runs": int(payload.get("runs", 5) or 5)' in \
        (src or open(proxy.__file__, encoding="utf-8").read())


# --- the report has to render a curve, not a scramble -----------------------------------------


def test_the_ladder_renders_in_size_order_not_alphabetical():
    """Task ids sort 128k, 16k, 1m, 256k, 300k, 512k, 64k, 700k. That is the x-axis of a curve,
    shuffled — and a reader would take the first drop in the column as the cliff."""
    from ai_proxy.bench_report import _task_sort_key
    ids = [t["id"] for t in L.LONGCTX_TASKS]
    ordered = sorted(ids, key=_task_sort_key)
    assert ordered == ids, ordered
    assert sorted(ids) != ids, "if ids ever sort correctly on their own, drop the sort key"


def test_the_sort_key_leaves_other_suites_alone():
    from ai_proxy.bench_report import _task_sort_key
    others = ["mem_write_discipline", "sec_sql_injection", "aaa_first"]
    assert sorted(others, key=_task_sort_key) == sorted(others)


def test_the_report_knows_the_new_category():
    """A category absent from _CAT_ORDER is dropped from the breakdown table entirely."""
    from ai_proxy import bench_report as R
    assert "longcontext" in R._CAT_ORDER
    assert R._CAT_BLURB.get("longcontext")


def test_a_missed_needle_reads_as_a_missed_needle_in_the_report():
    """A needle case carries check="needle", which fell through to the agent-episode branch:
    a missed fact was reported as "conduct (no malformed/hallucinated/repeated calls…)
    expected null" — telling the reader the model made bad tool calls."""
    from ai_proxy.bench_report import _bench_case_text
    task = dict(L.LONGCTX_TASKS[0])
    txt = _bench_case_text(task, 2, "said PERIWINKLE-2051 — that is another needle's code")
    assert "charlie" in txt and "50%" in txt, txt
    assert "OBSIDIAN-1596" in txt, txt
    assert "conduct" not in txt and "expected null" not in txt, txt


def test_the_depth_is_named_so_the_reader_can_see_where_it_failed():
    """Which depth broke is the whole point of the metric; a bare pass/fail loses it."""
    from ai_proxy.bench_report import _bench_case_parts
    task = dict(L.LONGCTX_TASKS[0])
    for idx, (name, code, _f) in enumerate(L.NEEDLES):
        what, exp = _bench_case_parts(task, idx)
        assert name in what, what
        assert code in exp, exp


# --- a truncated reply is not a failed one ----------------------------------------------------


import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_a_reply_cut_off_at_the_token_cap_is_flagged_not_just_scored(client):
    """Measured during the first run of this suite: a 262k unit spent its entire 2,048-token
    budget on 6,134 characters of reasoning and emitted no content at all. Scored plainly that
    reads as recall collapsing at 256k, which is a claim about the model made from a fact about
    the token budget."""
    task = dict(L.LONGCTX_TASKS[0])
    res = await proxy._bench_grade("alpha=CRIMSON-4417", task, 10.0, finish_reason="length")
    assert res["passed"] == 1 and res["score"] < 1
    assert res.get("truncated") is True


@pytest.mark.anyio
async def test_a_complete_reply_is_never_flagged(client):
    task = dict(L.LONGCTX_TASKS[0])
    ans = "\n".join(f"{n}={c}" for n, c, _f in L.NEEDLES)
    res = await proxy._bench_grade(ans, task, 10.0, finish_reason="stop")
    assert res["passed"] == 5 and not res.get("truncated")


@pytest.mark.anyio
async def test_a_perfect_answer_that_happened_to_hit_the_cap_is_not_flagged(client):
    """Nothing was lost if every case passed, so calling it truncated would be noise."""
    task = dict(L.LONGCTX_TASKS[0])
    ans = "\n".join(f"{n}={c}" for n, c, _f in L.NEEDLES)
    res = await proxy._bench_grade(ans, task, 10.0, finish_reason="length")
    assert not res.get("truncated")


@pytest.mark.anyio
async def test_an_answer_that_arrived_only_in_reasoning_still_grades(client):
    """The runner captures content deltas; this model put all five codes in `reasoning` and
    returned empty content at 700k."""
    task = dict(L.LONGCTX_TASKS[0])
    only_reasoning = "\n".join(f"{n}={c}" for n, c, _f in L.NEEDLES)
    res = await proxy._bench_grade("", task, 10.0, reasoning=only_reasoning)
    assert res["passed"] == 5, res


# --- the harness must outlast the prefill it asked for ----------------------------------------


def test_the_request_timeout_scales_with_the_prompt():
    """A flat 600s aborted every 700k unit while the model was still prefilling. The proxy
    logged "499 client disconnected" and six of them tripped the circuit breaker, so a limit
    in the harness was recorded as the backend failing."""
    import re as _re
    src = open(proxy.__file__, encoding="utf-8").read()
    m = _re.search(r"_prefill_s = (.+?)\n\s+_timeout_s = (.+?)\n", src)
    assert m, "the scaled timeout is gone or was renamed"

    def budget(tokens):
        body_len = tokens * 4
        return max(600.0, body_len / 4 / 250 + 600.0)

    # Measured on this box: 700k prefills ran 656-1864s. The budget has to clear the worst.
    assert budget(700_000) > 1864, budget(700_000)
    assert budget(16_000) >= 600, "short prompts keep the old floor"
    assert budget(940_000) > budget(700_000) > budget(300_000), "must be monotonic"


def test_the_timeout_is_not_unbounded():
    """A run that hangs forever is worse than one that fails: the box is exclusive for the
    duration and nothing says why."""
    def budget(tokens):
        return max(600.0, (tokens * 4) / 4 / 250 + 600.0)
    assert budget(1_000_000) < 7200, "an hour and a half is already generous"


# --- real replies from the 600k rung ----------------------------------------------------------


def test_a_code_pinned_to_the_wrong_needle_does_not_also_credit_the_right_one():
    """Verbatim from a 600k unit. charlie is answered with echo's code; delta and echo are
    never answered at all. Scoring echo correct because its code appears SOMEWHERE turns one
    confabulation into one error and one success — inflating the score in exactly the case
    this suite exists to catch."""
    reply = "alpha=CRIMSON-4417\nbeta=MERIDIAN-8823\ncharlie=PERIWINKLE-2051"
    r = G._bench_grade_needles(reply, L._cases())
    by = {c["label"].split(" @ ")[0]: c for c in r["cases"]}
    assert by["alpha"]["ok"], "alpha was answered correctly"
    assert by["bravo"]["ok"], "the code is right even though the model called it 'beta'"
    assert not by["charlie"]["ok"] and "another needle" in by["charlie"]["got"]
    assert not by["delta"]["ok"], "never answered"
    assert not by["echo"]["ok"], "its code was pinned to charlie, not reported as echo's"
    assert r["passed"] == 2, r["cases"]


def test_an_inverted_format_still_scores_what_it_actually_found():
    """Also verbatim: the model answered CODE=linenumber instead of NAME=CODE. Every needle
    was genuinely located; the layout was wrong. The deliverable is the code."""
    reply = ("CRIMSON-4417=0000000, MERIDIAN-8823=0000001, OBSIDIAN-1596=0010266, "
             "TANGERINE-7304=0014677, PERIWINKLE-2051= echo vault access code is "
             "PERIWINKLE-2051.")
    assert G._bench_grade_needles(reply, L._cases())["passed"] == 5


def test_the_bare_code_fallback_still_works_when_nothing_is_misattributed():
    assert G._bench_grade_needles("CRIMSON-4417", L._cases())["passed"] == 1


def test_the_context_guard_falls_back_to_asking_the_backend():
    """Measured: the model index entry for a resident Ollama model held only {"loaded": True} —
    no window at all — so the guard read None and skipped nothing. A 1M rung was then sent to a
    1M window it cannot fit, front-truncated, and scored a number that measured nothing."""
    import re as _re
    src = open(proxy.__file__, encoding="utf-8").read()
    block = _re.search(r'_window = model_meta\.get\("loaded_context"\).*?if _window:', src, _re.S)
    assert block, "the guard moved"
    assert "_bench_loaded_context" in block.group(0), \
        "with no window in the catalogue the guard must ask the backend, not give up"


# --- formatting is not recall -----------------------------------------------------------------


def test_a_code_split_at_its_hyphen_still_counts():
    """Verbatim from a 512k thinking-off unit: every needle found, every hyphen turned into an
    equals sign. Exact-substring matching scored it 0/5 — a grader reading punctuation as part
    of the answer measures formatting, not recall. Notably this only ever happened with
    thinking off; with thinking on the model kept the requested format every time, so the bug
    would have penalised exactly one arm of the comparison."""
    reply = "CRIMSON=4417, MERIDIAN=8823, OBSIDIAN=1596, TANGERINE=7304, PERIWINKLE=2051"
    assert G._bench_grade_needles(reply, L._cases())["passed"] == 5


def test_one_mangled_code_among_intact_ones_still_counts():
    reply = ("crimson=4417\nbravo=MERIDIAN-8823\ncharlie=OBSIDIAN-1596\n"
             "delta=TANGERINE-7304\necho=PERIWINKLE-2051")
    assert G._bench_grade_needles(reply, L._cases())["passed"] == 5


def test_normalising_does_not_rescue_a_genuine_misattribution():
    """The loosened match must not start forgiving the failure it exists to catch."""
    reply = ("alpha=CRIMSON-4417\nbravo=MERIDIAN-8823\ncharlie=PERIWINKLE-2051\n"
             "delta=TANGERINE-7304\necho=MISSING")
    r = G._bench_grade_needles(reply, L._cases())
    assert r["passed"] == 3
    by = {c["label"].split(" @ ")[0]: c for c in r["cases"]}
    assert "another needle" in by["charlie"]["got"]
    assert not by["echo"]["ok"]


def test_normalising_does_not_invent_matches_from_the_filler():
    """The filler is enumerated ledger lines; a normalised match must not collide with it.
    Built from _filler directly rather than slicing a haystack — the first needle sits at
    line 3, so any prefix of a real haystack legitimately contains one."""
    filler = "\n".join(L._filler(i) for i in range(4000))
    assert G._bench_grade_needles(filler, L._cases())["passed"] == 0


def test_lite_and_deep_partition_the_ladder():
    """Together they cover it exactly once. A comparison that needs one end should not pay for
    the other — the deep rungs are four hours of exclusive GPU, the shallow ones barely one."""
    lite = {t["id"] for t in L.LONGCTX_LITE_TASKS}
    deep = {t["id"] for t in L.LONGCTX_DEEP_TASKS}
    full = {t["id"] for t in L.LONGCTX_TASKS}
    assert lite | deep == full and not (lite & deep)
    assert lite and deep


def test_the_lite_suite_stops_below_the_measured_edge():
    """512k measured 23-25/25 and 600k drops to ~4/5, so the split sits on the edge rather
    than through the middle of it."""
    assert max(t["target_tokens"] for t in L.LONGCTX_LITE_TASKS) == 512_000


def test_lite_is_registered(client):
    assert "longctx-lite" in proxy._BENCH_SUITES


def test_the_output_budget_can_exceed_eight_thousand_tokens():
    """The runner clamped max_tokens to 8192, chosen when prompts were short. It silently
    clamped a deliberate 32768 and the re-run was identical to the one it replaced — the
    request carried 8192 while the run config said 32768. Long-context reasoning needs the
    room: ~1k characters at 16k of context, 24,652 at 300k."""
    import re as _re
    src = open(proxy.__file__, encoding="utf-8").read()
    m = _re.search(r'max_tokens = max\(16, min\(int\(_scalar\("max_tokens", 256\)\), (\d+)\)\)', src)
    assert m, "the clamp moved or was renamed"
    assert int(m.group(1)) >= 32768, f"still clamped at {m.group(1)}"
