"""The report-only views: conversation depth, prefill/decode split, tool-call hygiene, and the
daily token/cost ledger.

None of these run on the dashboard poll, so nothing else exercises them. Two need care: the
hygiene pass walks untrusted request bodies, where one malformed blob must not take the report
down, and the ledger puts a dollar figure on the page, where an arithmetic slip is a lie rather
than a glitch.
"""
import json
import time

import pytest

from ai_proxy import proxy as P


def _seed(rows, blobs=(), stream=()):
    conn = P.db()
    conn.executemany(
        """INSERT INTO requests (id, ts, method, path, upstream_url, client_app, status,
                                 turn_index, prompt_tokens, est_prompt_tokens, duration_ms,
                                 ttft_ms, conversation_id)
           VALUES (:id, :ts, 'POST', '/v1/messages', 'http://x', :app, :status,
                   :turn, :ptok, :est, :dur, :ttft, :conv)""",
        [{"id": r["id"], "ts": r.get("ts", time.time()), "app": r.get("app", "test-app"),
          "status": r.get("status", 200), "turn": r.get("turn"), "ptok": r.get("ptok"),
          "est": r.get("est"), "dur": r.get("dur"), "ttft": r.get("ttft"),
          "conv": r.get("conv")} for r in rows])
    if blobs:
        conn.executemany(
            "INSERT INTO request_blobs (id, request_body, response_body) VALUES (?, ?, ?)", blobs)
    if stream:
        conn.executemany(
            "INSERT INTO request_blobs (id, request_body, stream_chunks) VALUES (?, ?, ?)", stream)
    conn.commit()
    conn.close()


def _tool_call(name, args=None):
    return {"type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


@pytest.fixture
def _clean(client):
    """`client` builds the app against a temp DB; empty the tables between cases."""
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.commit()
    conn.close()
    return client


def test_turn_depth_buckets_and_order(_clean):
    _seed([{"id": f"t{i}", "turn": turn, "ptok": ptok, "dur": 1000, "ttft": 100}
           for i, (turn, ptok) in enumerate(
               [(1, 1000), (3, 2000), (7, 50000), (20, 90000), (60, 120000), (500, 200000)])])
    ex = P._usage_extras()
    buckets = [b["bucket"] for b in ex["by_turn"]]
    # Shallow first: the section reads as a progression, so the order is part of the meaning.
    assert buckets == ["1–4", "5–14", "15–39", "40–99", "100+"]
    first = ex["by_turn"][0]
    assert first["n"] == 2 and first["prompt"] == pytest.approx(1500)


def test_turn_depth_prefers_reported_tokens_over_estimate(_clean):
    _seed([{"id": "a", "turn": 1, "ptok": 8000, "est": 100, "dur": 1, "ttft": 1},
           {"id": "b", "turn": 2, "ptok": None, "est": 4000, "dur": 1, "ttft": 1}])
    ex = P._usage_extras()
    assert ex["by_turn"][0]["prompt"] == pytest.approx(6000)


def test_turn_depth_skips_failures_and_untracked_turns(_clean):
    _seed([{"id": "ok", "turn": 1, "ptok": 100, "dur": 10, "ttft": 1},
           {"id": "err", "turn": 1, "ptok": 100, "dur": 10, "ttft": 1, "status": 500},
           {"id": "noturn", "turn": None, "ptok": 100, "dur": 10, "ttft": 1}])
    ex = P._usage_extras()
    assert sum(b["n"] for b in ex["by_turn"]) == 1


def test_time_split_and_abandoned(_clean):
    _seed([{"id": "s1", "turn": 1, "dur": 1000, "ttft": 250},
           {"id": "s2", "turn": 1, "dur": 1000, "ttft": 250},
           {"id": "gone", "turn": 1, "dur": 4000, "ttft": 100, "status": 499}])
    ex = P._usage_extras()
    ts = ex["time_split"]
    # 600 of 6000ms. The abandoned request counts: the GPU did that work whether or not the
    # client stayed to collect it.
    assert ts["prefill_pct"] == pytest.approx(10.0)
    assert ts["decode_pct"] == pytest.approx(90.0)
    assert ex["abandoned"] == {"n": 1, "wasted_ms": 4000}


def test_time_split_empty_is_not_a_division_by_zero(_clean):
    ex = P._usage_extras()
    assert ex["time_split"]["prefill_pct"] is None
    assert ex["undeclared_tools"] == []


def test_hygiene_finds_undeclared_tool_names(_clean):
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [
        _tool_call("run", {"command": "ls"}), _tool_call("terminal", {"cmd": "ls"})]}}]})
    _seed([{"id": "h1", "turn": 1, "app": "hermes"}], [("h1", req, resp)])
    ex = P._usage_extras()
    assert [u["tool"] for u in ex["undeclared_tools"]] == ["run"]   # terminal was declared
    u = ex["undeclared_tools"][0]
    assert u["calls"] == 1 and u["clients"] == ["hermes"]


def test_hygiene_reads_streamed_tool_calls(_clean):
    # Agentic clients stream, so the call arrives as SSE deltas and response_body is empty.
    # Reading only response_body is how this section silently reports nothing.
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    chunks = "\n".join([
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "run", "arguments": ""}}]}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"command":"ls"}'}}]}}]}),
        "data: [DONE]",
    ])
    _seed([{"id": "sd", "ts": time.time() - 60, "turn": 1, "app": "hermes"},
           {"id": "sc", "turn": 2, "app": "hermes"}],
          blobs=[("sd", req, "{}")], stream=[("sc", "{}", chunks)])
    assert [u["tool"] for u in P._usage_extras()["undeclared_tools"]] == ["run"]


def test_hygiene_reads_ollama_native_shape(_clean):
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"message": {"tool_calls": [_tool_call("run")]}})
    _seed([{"id": "h2", "turn": 1}], [("h2", req, resp)])
    assert [u["tool"] for u in P._usage_extras()["undeclared_tools"]] == ["run"]


def test_hygiene_ranks_by_call_count(_clean):
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [
        _tool_call("run"), _tool_call("run"), _tool_call("bash")]}}]})
    _seed([{"id": "h3", "turn": 1}], [("h3", req, resp)])
    assert [u["tool"] for u in P._usage_extras()["undeclared_tools"]] == ["run", "bash"]


def test_hygiene_takes_declarations_from_the_client_not_the_request(_clean):
    # The reply is matched against what that client declares generally, so the request carrying
    # the bad call never has to be parsed — that is what keeps this affordable.
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run")]}}]})
    now = time.time()
    _seed([{"id": "decl", "ts": now - 60, "turn": 1, "app": "hermes"},
           {"id": "call", "ts": now, "turn": 2, "app": "hermes"}],
          [("decl", req, "{}"), ("call", "{}", resp)])
    assert [u["tool"] for u in P._usage_extras()["undeclared_tools"]] == ["run"]


def test_hygiene_keeps_clients_apart(_clean):
    # run() is legitimate for a client that declares it; flagging it there would be noise.
    declares_run = json.dumps({"tools": [{"type": "function", "function": {"name": "run"}}]})
    declares_terminal = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run")]}}]})
    _seed([{"id": "a", "turn": 1, "app": "fine"}, {"id": "b", "turn": 1, "app": "broken"}],
          [("a", declares_run, resp), ("b", declares_terminal, resp)])
    und = P._usage_extras()["undeclared_tools"]
    assert len(und) == 1 and und[0]["clients"] == ["broken"] and und[0]["calls"] == 1


def test_hygiene_marks_names_an_alias_already_handles(_clean, monkeypatch):
    # Blobs record what the model emitted, so an aliased name still shows up. Without this
    # column the panel reads as "74 broken calls" when the rewrite is already catching them.
    cfg = dict(P.load_rules_config())
    cfg["tool_aliases"] = {"enabled": True, "map": {"run": "terminal",
                                                    "task": {"to": "delegate_task"}}}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [
        _tool_call("run"), _tool_call("task"), _tool_call("invented")]}}]})
    _seed([{"id": "al", "turn": 1, "app": "hermes"}], [("al", req, resp)])
    by_name = {u["tool"]: u["aliased_to"] for u in P._usage_extras()["undeclared_tools"]}
    assert by_name == {"run": "terminal", "task": "delegate_task", "invented": None}


def test_hygiene_ignores_aliases_when_the_rule_is_off(_clean, monkeypatch):
    cfg = dict(P.load_rules_config())
    cfg["tool_aliases"] = {"enabled": False, "map": {"run": "terminal"}}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run")]}}]})
    _seed([{"id": "off", "turn": 1}], [("off", req, resp)])
    assert P._usage_extras()["undeclared_tools"][0]["aliased_to"] is None


def test_hygiene_ignores_requests_declaring_no_tools(_clean):
    # The `tools` key present but empty means the client offered nothing, so "undeclared" is
    # every name — reporting them would bury the real mismatches.
    req = json.dumps({"tools": [], "messages": []})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run")]}}]})
    _seed([{"id": "h4", "turn": 1}], [("h4", req, resp)])
    assert P._usage_extras()["undeclared_tools"] == []


def test_hygiene_survives_unparseable_bodies(_clean):
    good_req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    good_resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run")]}}]})
    _seed([{"id": "bad", "turn": 1, "ts": time.time() - 1}, {"id": "good", "turn": 1}],
          [("bad", '{"tools": [truncated', "not json either"), ("good", good_req, good_resp)])
    # One corrupt row must not cost the whole section.
    assert [u["tool"] for u in P._usage_extras()["undeclared_tools"]] == ["run"]


def test_extras_respect_the_window(_clean):
    old = time.time() - 90000
    _seed([{"id": "old", "ts": old, "turn": 1, "ptok": 10, "dur": 100, "ttft": 10}])
    assert P._usage_extras(window_s=3600)["by_turn"] == []
    assert P._usage_extras(window_s=172800)["by_turn"] != []


def _price(monkeypatch, **over):
    cfg = dict(P.load_rules_config())
    cfg["pricing"] = {
        "enabled": True,
        "tiers": [{"name": "premium tier", "input": 3.0, "cached_input": 0.30, "output": 15.0},
                  {"name": "budget tier", "input": 0.30, "cached_input": 0.03, "output": 0.60}],
        "electricity": {"usd_per_kwh": 0.0, "watts": None, "watts_idle": 0},
        **over}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    return cfg


def test_daily_ledger_groups_and_totals(_clean, monkeypatch):
    _price(monkeypatch)
    day = 86400
    now = time.time()
    # No conversation id, so each is a one-shot and its whole prompt is new.
    _seed([{"id": "d1", "ts": now, "turn": 1, "ptok": 1_000_000},
           {"id": "d2", "ts": now, "turn": 2, "ptok": 1_000_000},
           {"id": "d3", "ts": now - day, "turn": 1, "ptok": 500_000}])
    conn = P.db()
    conn.executemany("UPDATE requests SET completion_tokens = ? WHERE id = ?",
                     [(1000, "d1"), (1000, "d2"), (2000, "d3")])
    conn.commit()
    conn.close()

    daily = P._usage_by_day()
    assert len(daily["by_day"]) == 2
    assert daily["by_day"][-1]["n"] == 2                     # today, sorted last
    assert daily["total"]["input"] == 2_500_000
    assert daily["total"]["output"] == 4000
    # 2.5M uncached input at $3/M + 4k output at $15/M
    assert daily["total"]["costs"] == [pytest.approx(7.5 + 0.06),
                                      pytest.approx(0.75 + 0.0024)]


def test_daily_ledger_charges_only_what_a_turn_added(_clean, monkeypatch):
    # The whole point: turn 2 re-sends turn 1's prompt, and only its growth is a new token.
    _price(monkeypatch)
    now = time.time()
    _seed([{"id": "c1", "ts": now - 10, "turn": 1, "ptok": 100_000, "conv": "x"},
           {"id": "c2", "ts": now, "turn": 2, "ptok": 1_000_000, "conv": "x"}])
    tot = P._usage_by_day()["total"]
    assert tot["input"] == 1_000_000 and tot["cached_input"] == 100_000
    # Charging both prompts in full would read $3.30 instead.
    assert tot["costs"][0] == pytest.approx(3.0 + 0.03)


def test_daily_ledger_handles_a_prompt_that_shrank(_clean, monkeypatch):
    # Compaction cuts the prompt; the reusable prefix is what is left, not the longer old one.
    _price(monkeypatch)
    now = time.time()
    _seed([{"id": "s1", "ts": now - 10, "turn": 1, "ptok": 1_000_000, "conv": "y"},
           {"id": "s2", "ts": now, "turn": 2, "ptok": 200_000, "conv": "y"}])
    tot = P._usage_by_day()["total"]
    assert tot["input"] == 1_000_000 and tot["cached_input"] == 200_000


def test_daily_ledger_keeps_conversations_apart(_clean, monkeypatch):
    # Two conversations interleaved in time: one's prompt must never discount the other's.
    _price(monkeypatch)
    now = time.time()
    _seed([{"id": "a1", "ts": now - 30, "turn": 1, "ptok": 500_000, "conv": "a"},
           {"id": "b1", "ts": now - 20, "turn": 1, "ptok": 500_000, "conv": "b"},
           {"id": "a2", "ts": now - 10, "turn": 2, "ptok": 600_000, "conv": "a"}])
    tot = P._usage_by_day()["total"]
    assert tot["input"] == 1_100_000 and tot["cached_input"] == 500_000


def test_daily_ledger_prices_every_tier(_clean, monkeypatch):
    # One rate is an argument about what a local model is worth; a bracket is a measurement.
    _price(monkeypatch)
    _seed([{"id": "m1", "turn": 1, "ptok": 1_000_000}])
    conn = P.db()
    conn.execute("UPDATE requests SET completion_tokens = 0 WHERE id = 'm1'")
    conn.commit()
    conn.close()
    daily = P._usage_by_day()
    assert daily["tiers"] == ["premium tier", "budget tier"]
    assert daily["total"]["costs"] == [pytest.approx(3.0), pytest.approx(0.30)]


def test_daily_ledger_can_be_turned_off(_clean, monkeypatch):
    _price(monkeypatch, enabled=False)
    _seed([{"id": "p1", "turn": 1, "ptok": 1_000_000}])
    daily = P._usage_by_day()
    assert daily["priced"] is False
    assert daily["total"]["costs"] == []        # tokens still counted, money not claimed
    assert daily["total"]["input"] == 1_000_000


def test_daily_ledger_excludes_shadow_traffic(_clean, monkeypatch):
    # Shadow requests are the proxy's own duplicate of a real one. Counting them would inflate
    # both the token total and the cost by whatever share of traffic is being shadowed.
    _price(monkeypatch)
    _seed([{"id": "real", "turn": 1, "ptok": 1_000_000},
           {"id": "shadow", "turn": 1, "ptok": 1_000_000}])
    conn = P.db()
    conn.execute("UPDATE requests SET shadow_of = 'real', completion_tokens = 0 WHERE id = 'shadow'")
    conn.execute("UPDATE requests SET completion_tokens = 0 WHERE id = 'real'")
    conn.commit()
    conn.close()
    assert P._usage_by_day()["total"]["n"] == 1


def test_daily_ledger_counts_days_with_no_traffic(_clean, monkeypatch):
    _price(monkeypatch)
    now = time.time()
    _seed([{"id": "g1", "ts": now, "turn": 1, "ptok": 10, "dur": 10, "ttft": 5},
           {"id": "g2", "ts": now - 3 * 86400, "turn": 1, "ptok": 10, "dur": 10, "ttft": 5}])
    assert P._usage_by_day()["missing_days"] == 2


def test_busy_time_merges_concurrent_requests(_clean):
    # Summing durations is the obvious mistake and it overstates busy time severalfold: four
    # requests served at once occupy the GPU for the length of the longest, not their sum.
    base = time.mktime(time.strptime("2026-03-05 12:00:00", "%Y-%m-%d %H:%M:%S"))
    rows = [{"ts": base, "duration_ms": 10_000},
            {"ts": base + 2, "duration_ms": 4_000},      # wholly inside the first
            {"ts": base + 8, "duration_ms": 10_000},     # overlaps, extends to +18
            {"ts": base + 30, "duration_ms": 5_000}]     # separate
    busy = P._busy_seconds_by_day(rows)
    assert sum(busy.values()) == pytest.approx(18 + 5)   # not 10 + 4 + 10 + 5


def test_busy_time_splits_at_midnight(_clean):
    # A generation running over midnight belongs to both days, or one day's energy lands on the
    # wrong bill.
    midnight = time.mktime(time.strptime("2026-03-06 00:00:00", "%Y-%m-%d %H:%M:%S"))
    busy = P._busy_seconds_by_day([{"ts": midnight - 60, "duration_ms": 120_000}])
    assert busy["2026-03-05"] == pytest.approx(60)
    assert busy["2026-03-06"] == pytest.approx(60)


def test_busy_time_ignores_requests_that_never_finished(_clean):
    assert P._busy_seconds_by_day([{"ts": time.time(), "duration_ms": None},
                                   {"ts": time.time(), "duration_ms": 0}]) == {}


def test_electricity_costs_the_clock_not_the_requests(_clean, monkeypatch):
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500, "watts_idle": 0})
    now = time.time()
    # Two hours of wall-clock work, served two-at-a-time. At 500W that is 1 kWh, not 2.
    _seed([{"id": "e1", "ts": now - 7200, "turn": 1, "ptok": 10, "dur": 7_200_000},
           {"id": "e2", "ts": now - 7200, "turn": 1, "ptok": 10, "dur": 7_200_000}])
    daily = P._usage_by_day()
    assert daily["power"]["watts"] == 500
    assert daily["power"]["source"] == "whole system, configured"
    assert daily["total"]["kwh"] == pytest.approx(1.0, rel=0.01)
    assert daily["total"]["power_cost"] == pytest.approx(0.20, rel=0.01)


def test_idle_draw_is_charged_for_the_hours_between_requests(_clean, monkeypatch):
    # The box draws power whether or not it is answering, and on bursty traffic those hours are
    # most of the day. Charging only busy time would understate the real cost several times.
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500, "watts_idle": 100})
    now = time.time()
    _seed([{"id": "i1", "ts": now - 3600, "turn": 1, "ptok": 10, "dur": 3_600_000}])
    # Totals, not by_day[0]: an hour-long request started before midnight is split across two
    # day rows, and asserting on the first one only passes for most of the day.
    tot = P._usage_by_day(days=1)["total"]
    assert tot["busy_s"] == pytest.approx(3600, rel=0.01)
    assert tot["idle_s"] > 0
    # Stated rather than assumed: the total is the two draws over their own hours.
    assert tot["kwh"] == pytest.approx(
        (tot["busy_s"] / 3600) * 0.5 + (tot["idle_s"] / 3600) * 0.1, rel=1e-6)
    assert tot["kwh"] > (tot["busy_s"] / 3600) * 0.5     # idle actually moved the number


def test_power_is_not_lost_when_a_request_runs_past_midnight(_clean, monkeypatch):
    # The row is filed under the day it started, so the day it spilled into has no request of
    # its own. Looking that day up instead of creating it dropped the power on the floor.
    _price(monkeypatch, electricity={"usd_per_kwh": 1.0, "watts": 3600, "watts_idle": 0})
    midnight = time.mktime(time.strptime("2026-07-25 00:00:00", "%Y-%m-%d %H:%M:%S"))
    _seed([{"id": "mid", "ts": midnight - 1800, "turn": 1, "ptok": 10, "dur": 3_600_000}])
    daily = P._usage_by_day(days=30)
    dates = {d["date"]: d for d in daily["by_day"]}
    assert "2026-07-24" in dates and "2026-07-25" in dates
    assert dates["2026-07-24"]["busy_s"] == pytest.approx(1800, rel=0.01)
    assert dates["2026-07-25"]["busy_s"] == pytest.approx(1800, rel=0.01)
    # 3600 W for one hour at $1/kWh = $3.60, whichever side of midnight it fell on.
    assert daily["total"]["power_cost"] == pytest.approx(3.6, rel=0.01)
    # The spilled-into day carries power but no tokens, and must not invent traffic.
    assert dates["2026-07-25"]["n"] == 0 and dates["2026-07-25"]["input"] == 0


def test_idle_draw_is_assumed_from_load_when_not_measured(_clean, monkeypatch):
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500})
    _seed([{"id": "i2", "turn": 1, "ptok": 10, "dur": 1000}])
    power = P._usage_by_day()["power"]
    assert power["watts_idle"] == pytest.approx(500 * P._IDLE_DRAW_FRACTION)
    assert power["idle_source"] == "assumed"    # the page has to say so, so the data must too


def test_idle_hours_stop_at_the_window_edge(_clean, monkeypatch):
    # A one-day window must not bill the first day for hours before the window opened.
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500, "watts_idle": 100})
    _seed([{"id": "i3", "turn": 1, "ptok": 10, "dur": 1000}])
    for d in P._usage_by_day(days=1)["by_day"]:
        assert d["busy_s"] + d["idle_s"] <= 86400 + 1


def test_report_says_so_when_the_wattage_is_only_the_gpu(_clean, monkeypatch):
    # Falling back to the GPU's draw understates the machine twice over. Printing it without
    # saying so would pass a floor off as the cost of running the box.
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": None})
    monkeypatch.setattr(P, "_gpu_watts", lambda: 40.0)
    _seed([{"id": "g1", "turn": 1, "ptok": 10, "dur": 3_600_000}])
    assert P._usage_by_day()["power"]["source"].startswith("GPU rail")
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "is a floor rather than a measurement" in html
    assert "power-supply losses" in html


def test_electricity_is_omitted_without_a_wattage(_clean, monkeypatch):
    # No configured watts and no GPU to ask: say nothing rather than invent a number.
    _price(monkeypatch, electricity={"usd_per_kwh": 0.17, "watts": None})
    monkeypatch.setattr(P, "_gpu_watts", lambda: None)
    _seed([{"id": "n1", "turn": 1, "ptok": 10, "dur": 1000}])
    daily = P._usage_by_day()
    assert "power" not in daily
    assert daily["total"]["power_cost"] == 0


def test_report_renders_electricity_against_the_tiers(_clean, monkeypatch):
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500, "watts_idle": 0})
    now = time.time()
    _seed([{"id": "w1", "ts": now - 3600, "turn": 1, "ptok": 2_000_000, "dur": 3_600_000}])
    conn = P.db()
    conn.execute("UPDATE requests SET completion_tokens = 0 WHERE id = 'w1'")
    conn.commit()
    conn.close()
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "GPU hrs" in html and "Electricity" in html
    assert "$0.10" in html               # 1h at 500W = 0.5 kWh at $0.20
    assert "Cost to produce" in html         # the headline figures are cards now
    assert "500 W under load (whole system, configured)" in html   # names its sources
    assert "0 W idle (configured)" in html
    # A configured wattage is the machine's own; no floor caveat belongs on it.
    assert "is a floor rather than a measurement" not in html


def test_cost_is_not_at_the_top_of_the_page(_clean, monkeypatch):
    # The ledger's card markup was assigned to `cards`, the name holding the page's opening
    # hero. That replaced the top of the report with its cost figures and printed them twice.
    _price(monkeypatch, electricity={"usd_per_kwh": 0.20, "watts": 500, "watts_idle": 0})
    _seed([{"id": "top", "turn": 1, "ptok": 2_000_000, "dur": 9000, "ttft": 200}])
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert html.count("Cost to produce") == 1, "cost cards rendered more than once"
    assert "tokens and wrote" in html, "the page's own opening hero went missing"
    # Whatever money appears, it appears after the traffic tables.
    assert html.index("tokens and wrote") < html.index("Cost to produce")
    assert html.index("Day by day") > html.index("Models")


def test_opening_does_not_repeat_the_meta_strip(_clean):
    # Period, request count and token total are already in the header. The hero cards are the
    # most valuable space on the page and shouldn't spend it restating them.
    _seed([{"id": "o1", "turn": 1, "ptok": 5000, "dur": 900, "ttft": 200}])
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    opening = html[:html.index("Models")]
    assert "Decode rate" in opening and "Reply length" in opening
    # "mean" is what it computes; calling it a median was simply wrong.
    assert "mean across every request" in opening
    assert "median request wrote this much" not in html


def test_report_renders_the_daily_ledger(_clean, monkeypatch):
    _price(monkeypatch)
    _seed([{"id": "rr", "turn": 1, "ptok": 2_000_000}])
    conn = P.db()
    conn.execute("UPDATE requests SET completion_tokens = 0 WHERE id = 'rr'")
    conn.commit()
    conn.close()
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "Day by day" in html
    assert "would have cost bought elsewhere" in html
    assert "<ul class=\"fn\">" in html      # caveats are a list, not a slab of grey prose
    assert "premium tier" in html and "budget tier" in html
    assert "$6.00" in html and "$0.60" in html   # 2M uncached input at $3/M and at $0.30/M
    assert 'class="sum"' in html                 # the total row reads as a total


def test_report_renders_the_new_sections(_clean):
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run", {"command": "ls"})]}}]})
    _seed([{"id": "r1", "turn": 2, "ptok": 5000, "dur": 900, "ttft": 200}], [("r1", req, resp)])
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "Conversation depth" in html
    assert "Where the time goes" in html
    assert "Tool calls the client never offered" in html
    assert "<code>run</code>" in html


def test_report_streams_a_building_notice_before_the_content(_clean):
    # Assembling this takes seconds; a browser with nothing to render shows a blank tab that
    # looks like a hang. The notice must arrive first and hide itself once the content lands.
    _seed([{"id": "s1", "turn": 1, "ptok": 100, "dur": 10, "ttft": 5}])
    r = _clean.get("/__proxy/api/stats/report?format=html")
    assert r.status_code == 200
    html = r.text
    assert html.index('id="building"') < html.index("Usage report</title>") + len(html)
    assert html.index('id="building"') < html.index("<footer")
    # The hiding rule is sent last, so a saved copy shows no leftover spinner.
    assert "#building{display:none}" in html
    assert html.index("#building{display:none}") > html.index('id="building"')
    assert r.headers.get("cache-control") == "no-store"


def test_report_streams_an_error_into_the_page(_clean, monkeypatch):
    # The head is already on the wire by then, so a failure cannot be a status code.
    def boom():
        raise RuntimeError("stats exploded")
    monkeypatch.setattr(P, "stats", boom)
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "could not be built" in html
    assert "RuntimeError: stats exploded" in html
    assert "<footer" in html          # the document is still closed properly


def test_report_omits_sections_with_nothing_to_say(_clean):
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "Tool calls the client never offered" not in html
    assert "Conversation depth" not in html
