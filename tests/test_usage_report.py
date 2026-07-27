"""The three report-only views: conversation depth, prefill/decode split, tool-call hygiene.

These don't run on the dashboard poll, so nothing else exercises them — and the hygiene pass in
particular walks untrusted request bodies, where one malformed blob must not take the report down.
"""
import json
import time

import pytest

from ai_proxy import proxy as P


def _seed(rows, blobs=(), stream=()):
    conn = P.db()
    conn.executemany(
        """INSERT INTO requests (id, ts, method, path, upstream_url, client_app, status,
                                 turn_index, prompt_tokens, est_prompt_tokens, duration_ms, ttft_ms)
           VALUES (:id, :ts, 'POST', '/v1/messages', 'http://x', :app, :status,
                   :turn, :ptok, :est, :dur, :ttft)""",
        [{"id": r["id"], "ts": r.get("ts", time.time()), "app": r.get("app", "test-app"),
          "status": r.get("status", 200), "turn": r.get("turn"), "ptok": r.get("ptok"),
          "est": r.get("est"), "dur": r.get("dur"), "ttft": r.get("ttft")} for r in rows])
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


def test_report_renders_the_new_sections(_clean):
    req = json.dumps({"tools": [{"type": "function", "function": {"name": "terminal"}}]})
    resp = json.dumps({"choices": [{"message": {"tool_calls": [_tool_call("run", {"command": "ls"})]}}]})
    _seed([{"id": "r1", "turn": 2, "ptok": 5000, "dur": 900, "ttft": 200}], [("r1", req, resp)])
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "Conversation depth" in html
    assert "Where the time goes" in html
    assert "Tool calls the client never offered" in html
    assert "<code>run</code>" in html


def test_report_omits_sections_with_nothing_to_say(_clean):
    html = _clean.get("/__proxy/api/stats/report?format=html").text
    assert "Tool calls the client never offered" not in html
    assert "Conversation depth" not in html
