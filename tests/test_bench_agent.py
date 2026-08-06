"""agent-v1: the episode machinery, graded on completion AND conduct.

The suite exists because single-shot benches cannot see what kills models in long agentic
sessions (gemma4, 32 turns deep at a saturated context, answering in 15 tokens). The tests
pin the deterministic worlds, the two-case partial-credit grading, and the episode runner's
loop against a scripted fake model.
"""
import asyncio
import json

from ai_proxy import bench_agent as A
from ai_proxy import proxy as P


def _task(tid):
    return next(t for t in A.AGENT_TASKS if t["id"] == tid)


def test_worlds_are_deterministic_and_stateful():
    w = A.AgentWorld(_task("agent_chain"))
    assert json.loads(w.execute("lookup", '{"name": "start"}'))["value"] == "see:brass"
    assert json.loads(w.execute("lookup", '{"name": "harbor"}'))["value"] == "417"
    kv = A.AgentWorld(_task("agent_update_verify"))
    kv.execute("set_value", '{"key": "mode", "value": "ready"}')
    assert json.loads(kv.execute("get_value", '{"key": "mode"}'))["value"] == "ready"


def test_the_flaky_tool_fails_exactly_twice():
    w = A.AgentWorld(_task("agent_flaky"))
    assert "error" in json.loads(w.execute("get_balance", '{"account": "ax9"}'))
    assert "error" in json.loads(w.execute("get_balance", '{"account": "ax9"}'))
    assert json.loads(w.execute("get_balance", '{"account": "ax9"}'))["balance"] == "2044"


def test_conduct_accounting():
    w = A.AgentWorld(_task("agent_sum_files"))
    w.execute("list_files", "not json {")
    assert w.malformed == 1
    w.execute("teleport", "{}")
    assert w.unknown == 1
    w.execute("read_file", '{"name": "a.num"}')
    w.execute("read_file", '{"name": "a.num"}')
    assert w.repeats == 1


def test_retrying_after_an_error_is_not_a_repeat():
    """agent_flaky DEMANDS retrying the identical call; the first conduct rule failed every
    model that did the right thing — caught by the suite's own first run."""
    w = A.AgentWorld(_task("agent_flaky"))
    w.execute("get_balance", '{"account": "ax9"}')      # error 1
    w.execute("get_balance", '{"account": "ax9"}')      # error 2 — retry, not a repeat
    w.execute("get_balance", '{"account": "ax9"}')      # success
    assert w.repeats == 0
    w.execute("get_balance", '{"account": "ax9"}')      # repeat after SUCCESS — that counts
    assert w.repeats == 1


def test_grading_gives_partial_credit():
    t = _task("agent_chain")
    clean = A.AgentWorld(t)
    g = A.grade_episode(t, clean, "417", steps=9, exhausted=False)
    assert g["passed"] == 2 and g["total"] == 2
    messy = A.AgentWorld(t)
    messy.malformed = 2
    g2 = A.grade_episode(t, messy, "417", steps=9, exhausted=False)
    assert g2["passed"] == 1, "right answer, dirty conduct — one case each"
    assert "malformed" in g2["cases"][1]["got"]
    g3 = A.grade_episode(t, A.AgentWorld(t), "the answer is see:brass maybe", 3, False)
    assert g3["cases"][0]["ok"] is False
    g4 = A.grade_episode(t, A.AgentWorld(t), None, 14, exhausted=True)
    assert g4["passed"] == 0 and "budget" in g4["cases"][1]["got"]


class _FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _ScriptedModel:
    """Plays the model side: emits scripted turns, receives the growing conversation."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_bodies = []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.seen_bodies.append(json)
        msg = self.turns.pop(0)
        return _FakeResp({"choices": [{"message": msg}],
                         "usage": {"completion_tokens": 7}})


def _tc(name, args, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_episode_runner_walks_the_chain(client):
    t = _task("agent_update_verify")
    model = _ScriptedModel([
        {"tool_calls": [_tc("set_value", {"key": "mode", "value": "ready"})]},
        {"tool_calls": [_tc("get_value", {"key": "mode"}, "c2")]},
        {"content": "ready"},
    ])
    row = asyncio.run(P._bench_run_agent(model, "http://x", "m", t, {"max_tokens": 512}))
    assert row["grade"]["passed"] == 2, row["grade"]
    assert row["steps"] == 3
    assert "TOOL CALL: set_value" in row["text"]
    assert "ASSISTANT: ready" in row["text"]
    # the tool results flowed back into the conversation the model saw
    last_body = model.seen_bodies[-1]
    assert any(m.get("role") == "tool" for m in last_body["messages"])


def test_episode_runner_counts_exhaustion(client):
    t = dict(_task("agent_chain"), max_steps=2)
    model = _ScriptedModel([
        {"tool_calls": [_tc("lookup", {"name": "start"})]},
        {"tool_calls": [_tc("lookup", {"name": "brass"}, "c2")]},
    ])
    row = asyncio.run(P._bench_run_agent(model, "http://x", "m", t, {}))
    assert row["grade"]["exhausted"] is True
    assert row["grade"]["passed"] == 0
    assert "budget" in row["text"]


def test_agent_suite_is_registered_like_any_other(client):
    assert "agent-v1" in P._BENCH_SUITES
    for t in P._BENCH_SUITES["agent-v1"]:
        assert P._BENCH_TASK_DESC.get(t["id"]), t["id"]
        assert P._BENCH_TASK_NOTES.get(t["id"]), t["id"]
        assert len(t["cases"]) == 2
    d = client.get("/__proxy/api/bench/suites").json()
    names = {s["name"] for s in d["suites"]}
    assert "agent-v1" in names


def test_episode_cases_render_in_the_grades_vocabulary(client):
    from ai_proxy import bench_report as BR
    t = _task("agent_chain")
    call, exp = BR._bench_case_parts(t, 0)
    assert call == "final answer" and "417" in exp
    call2, _ = BR._bench_case_parts(t, 1)
    assert "conduct" in call2
