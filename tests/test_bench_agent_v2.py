"""agent-v2: the hard episode set. Every task targets a specific agentic failure mode; the
tests prove each world is solvable cleanly (a competent agent scores 2/2), that the trap
each task is built around actually fires, and that the suite is registered like any other.
"""
import asyncio
import json

from ai_proxy import bench_agent as A
from ai_proxy import proxy as P


def _task(tid):
    return next(t for t in A.AGENT2_TASKS if t["id"] == tid)


def _grade(task, world, answer, steps):
    return A.grade_episode(task, world, answer, steps=steps, exhausted=False)


def test_vault_protocol_is_taught_by_errors():
    t = _task("agent_vault")
    w = A.AgentWorld(t)
    sealed = json.loads(w.execute("vault_read", '{"key": "launch"}'))
    assert "vault_open" in sealed["error"], "the error must name the missing step"
    bad = json.loads(w.execute("vault_open", '{"token": "guess"}'))
    assert "get_token" in bad["error"]
    tok = json.loads(w.execute("get_token", "{}"))["token"]
    assert json.loads(w.execute("vault_open", json.dumps({"token": tok})))["ok"] is True
    assert json.loads(w.execute("vault_read", '{"key": "launch"}'))["value"] == "9412"
    g = _grade(t, w, "9412", steps=5)
    assert g["passed"] == 2, g


def test_reconcile_join_solves_to_270():
    t = _task("agent_reconcile")
    w = A.AgentWorld(t)
    orders = json.loads(w.execute("orders_for", '{"customer": "acme"}'))["orders"]
    total, prices = 0, {}
    for oid in orders:
        o = json.loads(w.execute("order_get", json.dumps({"id": oid})))
        if o["sku"] not in prices:   # re-asking a price you already know is a conduct repeat
            prices[o["sku"]] = json.loads(
                w.execute("price_of", json.dumps({"sku": o["sku"]})))["unit_price"]
        total += o["qty"] * prices[o["sku"]]
    assert total == 270
    assert _grade(t, w, str(total), steps=8)["passed"] == 2


def test_bisect_rewards_planning_and_punishes_scanning():
    t = _task("agent_bisect")
    w = A.AgentWorld(t)
    lo, hi, probes = 1, 100, 0
    while True:
        mid = (lo + hi) // 2
        probes += 1
        r = json.loads(w.execute("probe", json.dumps({"n": mid})))["result"]
        if r == "match":
            break
        lo, hi = (mid + 1, hi) if r == "too low" else (lo, mid - 1)
    assert probes <= 7, "binary search must fit the budget"
    assert _grade(t, w, "61", steps=probes + 1)["passed"] == 2
    # a linear scanner burns the budget: exhaustion fails conduct even with a lucky answer
    g = A.grade_episode(t, A.AgentWorld(t), None, steps=t["max_steps"], exhausted=True)
    assert g["passed"] == 0 and "budget" in g["cases"][1]["got"]


def test_deps_requires_dedup():
    t = _task("agent_deps")
    w = A.AgentWorld(t)
    seen, frontier = set(), ["app"]
    while frontier:
        pkg = frontier.pop()
        for d in json.loads(w.execute("deps_of", json.dumps({"package": pkg})))["deps"]:
            if d not in seen:
                seen.add(d)
                frontier.append(d)
    assert len(seen) == 6, "shared 'log' must count once"
    assert _grade(t, w, "6", steps=8)["passed"] == 2


def test_authority_db_beats_cache_and_file():
    t = _task("agent_authority")
    w = A.AgentWorld(t)
    vals = {s: json.loads(w.execute("config_read", json.dumps({"source": s, "key": "timeout"})))["value"]
            for s in ("cache", "db", "file")}
    assert vals == {"cache": "30", "db": "45", "file": "60"}, "sources must disagree"
    assert _grade(t, w, "45", steps=4)["passed"] == 2
    # trusting the cache is the trap this task exists to catch
    assert _grade(t, A.AgentWorld(t), "30", steps=4)["cases"][0]["ok"] is False


def test_migrate_full_cycle_sums_to_40():
    t = _task("agent_migrate")
    w = A.AgentWorld(t)
    missing = json.loads(w.execute("new_read", '{"key": "alpha"}'))
    assert "error" in missing, "reading the new store before writing must fail"
    total = 0
    for k in json.loads(w.execute("old_list", "{}"))["keys"]:
        v = json.loads(w.execute("old_read", json.dumps({"key": k})))["value"]
        w.execute("new_write", json.dumps({"key": k, "value": v}))
        rb = json.loads(w.execute("new_read", json.dumps({"key": k})))["value"]
        assert rb == v
        total += int(v)
    assert total == 40
    assert _grade(t, w, "40", steps=12)["passed"] == 2


def test_assemble_passphrase_order_is_shuffled_on_disk():
    t = _task("agent_assemble")
    w = A.AgentWorld(t)
    parts = {}
    for name in json.loads(w.execute("list_files", "{}"))["files"]:
        txt = json.loads(w.execute("read_file", json.dumps({"name": name})))["contents"]
        n, frag = txt.replace("part ", "").split(": ")
        parts[int(n)] = frag
    phrase = "".join(parts[i] for i in sorted(parts))
    assert phrase == "QF7NXK2PLM9TRV4D"
    # file order must NOT equal part order, or the task tests nothing (checked against the
    # task data directly — re-reading through the world would count as conduct repeats)
    disk_frags = [t["world"]["files"][f].split(": ")[1] for f in sorted(t["world"]["files"])]
    assert disk_frags != [parts[i] for i in sorted(parts)]
    assert _grade(t, w, phrase, steps=10)["passed"] == 2


def test_unstable_blind_retry_never_works():
    t = _task("agent_unstable")
    w = A.AgentWorld(t)
    for _ in range(3):
        assert "error" in json.loads(w.execute("fetch_report", "{}"))
    assert json.loads(w.execute("service_restart", "{}"))["ok"] is True
    rep = json.loads(w.execute("fetch_report", "{}"))["report"]
    assert rep["errors"] == 3
    assert w.repeats == 0, "retrying an ERRORING call is not a conduct repeat"
    assert _grade(t, w, "3", steps=6)["passed"] == 2


def test_capstone_single_discrepancy():
    t = _task("agent_capstone")
    w = A.AgentWorld(t)
    bad = []
    for sku in json.loads(w.execute("wh_list", "{}"))["skus"]:
        wh = json.loads(w.execute("wh_count", json.dumps({"sku": sku})))["count"]
        lg = json.loads(w.execute("ledger_count", json.dumps({"sku": sku})))["count"]
        if wh != lg:
            bad.append(sku)
    assert bad == ["crate"]
    assert _grade(t, w, "crate", steps=15)["passed"] == 2


def test_guessing_without_evidence_fails_conduct_on_every_v2_task():
    for t in A.AGENT2_TASKS:
        g = A.grade_episode(t, A.AgentWorld(t), str(t["expect"]), steps=1, exhausted=False)
        assert g["cases"][1]["ok"] is False, t["id"]
        assert "answered without evidence" in g["cases"][1]["got"], t["id"]


def test_agent_v2_registered_like_any_other(client):
    assert "agent-v2" in P._BENCH_SUITES
    for t in P._BENCH_SUITES["agent-v2"]:
        assert P._BENCH_TASK_DESC.get(t["id"]), t["id"]
        assert P._BENCH_TASK_NOTES.get(t["id"]), t["id"]
        assert len(t["cases"]) == 2
        assert t.get("require_tools"), t["id"]
    names = {s["name"] for s in client.get("/__proxy/api/bench/suites").json()["suites"]}
    assert "agent-v2" in names


class _FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)

    async def post(self, url, headers=None, json=None, timeout=None):
        msg = self.turns.pop(0)
        return _FakeResp({"choices": [{"message": msg}],
                          "usage": {"completion_tokens": 7}})


def _tc(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_episode_runner_recovers_through_vault_errors(client):
    """End to end through the HTTP loop: the model reads two errors, adapts, and the
    grade sees a clean 2/2 — errors it recovered from are not conduct violations."""
    t = _task("agent_vault")
    model = _ScriptedModel([
        {"tool_calls": [_tc("vault_read", {"key": "launch"}, "c1")]},      # sealed → error
        {"tool_calls": [_tc("get_token", {}, "c2")]},
        {"tool_calls": [_tc("vault_open", {"token": "tk-88a1"}, "c3")]},
        {"tool_calls": [_tc("vault_read", {"key": "launch"}, "c4")]},
        {"content": "9412"},
    ])
    row = asyncio.run(P._bench_run_agent(model, "http://x", "m", t, {"max_tokens": 512}))
    assert row["grade"]["passed"] == 2, row["grade"]
    assert row["steps"] == 5
