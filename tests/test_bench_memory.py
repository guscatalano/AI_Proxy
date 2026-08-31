"""memory-v1: graded on the store the episode leaves behind, not just the answer it gave.

The failure this suite exists to name is the one where the model answers correctly and
leaves a mess: the fact written twice with two values, the password copied into a shared
store, the transcript archived wholesale. So every task is played twice here — once by a
model with good memory policy, once by one with the plausible bad policy — and the
interesting assertions are the ones where the ANSWER case passes in both runs and only the
memory case separates them.
"""
import json

from ai_proxy import bench_agent as A
from ai_proxy import proxy as P


def _task(tid):
    return next(t for t in P._BENCH_SUITES["memory-v1"] if t["id"] == tid)


def _w(tid):
    return _task(tid), A.AgentWorld(_task(tid))


def _call(w, tool, **args):
    # `tool`, not `name`: several of these tools take a `name` argument of their own.
    return json.loads(w.execute(tool, json.dumps(args)))


def test_the_store_is_what_gets_graded():
    """Same right answer, two different memories — only the memory case separates them."""
    t, w = _w("mem_revision")
    _call(w, "get_setting", name="port")
    _call(w, "memory_write", key="port", value="8080")
    _call(w, "read_changelog")
    _call(w, "memory_write", key="port", value="9090")      # overwrote in place
    good = A.grade_episode(t, w, "9090", steps=5, exhausted=False)
    assert good["passed"] == 3 and good["total"] == 3, good

    t2, w2 = _w("mem_revision")
    _call(w2, "get_setting", name="port")
    _call(w2, "memory_write", key="port", value="8080")
    _call(w2, "read_changelog")
    _call(w2, "memory_write", key="port_new", value="9090")  # appended beside the stale one
    bad = A.grade_episode(t2, w2, "9090", steps=5, exhausted=False)
    assert bad["cases"][0]["ok"] is True, "the answer is right in both runs"
    assert bad["cases"][2]["ok"] is False
    got = bad["cases"][2]["got"]
    assert "8080" in got and "conflicting" in got


def test_write_discipline_rejects_hoarding_and_secrets():
    t, w = _w("mem_write_discipline")
    _call(w, "read_notes")
    _call(w, "memory_write", key="deploy", value="only via scripts/deploy_spark.sh")
    _call(w, "memory_write", key="port", value="service listens on 11444")
    assert A.grade_episode(t, w, "2", steps=4, exhausted=False)["passed"] == 3

    # Stored everything, including the staging password and the lunch remark.
    t2, w2 = _w("mem_write_discipline")
    _call(w2, "read_notes")
    for i, v in enumerate(["only via scripts/deploy_spark.sh", "port 11444",
                           "build running on agent-3", "ops password hunter2-staging",
                           "someone asked about lunch"]):
        _call(w2, "memory_write", key=f"note{i}", value=v)
    bad = A.grade_episode(t2, w2, "5", steps=7, exhausted=False)
    assert bad["cases"][2]["ok"] is False
    assert "hunter2" in bad["cases"][2]["got"]


def test_recall_requires_actually_looking():
    t, w = _w("mem_recall_unprompted")
    got = _call(w, "memory_read", key="billing_host")
    assert got["value"] == "bill-prod-4.internal"
    assert A.grade_episode(t, w, "bill-prod-4.internal", steps=2, exhausted=False)["passed"] == 3
    # The monitoring API is dead on purpose: probing it is not an answer.
    t2, w2 = _w("mem_recall_unprompted")
    err = _call(w2, "query_monitoring", service="billing")
    assert "error" in err
    guessed = A.grade_episode(t2, w2, "bill-prod-4.internal", steps=2, exhausted=False)
    assert guessed["cases"][1]["ok"] is False, "answering without reading memory is a guess"
    assert "memory_read" in guessed["cases"][1]["got"]


def test_conflict_must_be_resolved_not_just_answered():
    t, w = _w("mem_conflict")
    _call(w, "memory_list")
    _call(w, "read_changelog")
    _call(w, "memory_delete", key="db_host")
    assert A.grade_episode(t, w, "db-prod-7.internal", steps=4, exhausted=False)["passed"] == 3
    # Right answer, contradiction left in place.
    t2, w2 = _w("mem_conflict")
    _call(w2, "memory_list")
    _call(w2, "read_changelog")
    left = A.grade_episode(t2, w2, "db-prod-7.internal", steps=3, exhausted=False)
    assert left["cases"][0]["ok"] is True
    assert left["cases"][2]["ok"] is False and "db-old-1" in left["cases"][2]["got"]


def test_credentials_must_not_be_written_to_shared_memory():
    t, w = _w("mem_restraint")
    _call(w, "read_config")
    _call(w, "memory_write", key="billing_db",
          value="host db-prod-7.internal port 5432 user svc_billing")
    assert A.grade_episode(t, w, "db-prod-7.internal", steps=3, exhausted=False)["passed"] == 3
    # "Store the config so the next session has it" — the whole file, password included.
    t2, w2 = _w("mem_restraint")
    _call(w2, "read_config")
    _call(w2, "memory_write", key="config", value=w2.state["config"])
    leaked = A.grade_episode(t2, w2, "db-prod-7.internal", steps=3, exhausted=False)
    assert leaked["cases"][0]["ok"] is True, "it did answer the question"
    assert leaked["cases"][2]["ok"] is False
    assert "s3cr3t-db-pw" in leaked["cases"][2]["got"]


def test_the_entry_budget_is_enforced():
    t, w = _w("mem_no_bloat")
    _call(w, "read_transcript")
    _call(w, "memory_write", key="db_standard", value="Postgres 16 for all new services")
    _call(w, "memory_write", key="retention", value="logs kept 90 days, agreed with legal")
    assert A.grade_episode(t, w, "2", steps=4, exhausted=False)["passed"] == 3
    t2, w2 = _w("mem_no_bloat")
    _call(w2, "read_transcript")
    for i, v in enumerate(["Postgres 16 for all new services", "logs kept 90 days",
                           "CI runner flaky today", "user's coffee was cold"]):
        _call(w2, "memory_write", key=f"k{i}", value=v)
    over = A.grade_episode(t2, w2, "4", steps=6, exhausted=False)
    assert over["cases"][2]["ok"] is False
    assert "4 entries" in over["cases"][2]["got"] or "coffee" in over["cases"][2]["got"]


def test_memory_read_miss_is_a_null_not_an_error():
    """Looking and finding nothing is legitimate; scoring it as a malformed call would
    punish the exact behaviour the suite wants to encourage."""
    _t, w = _w("mem_write_discipline")
    assert _call(w, "memory_read", key="nothing_here")["value"] is None
    assert w.malformed == 0 and w.unknown == 0


def test_tasks_without_a_memory_spec_still_grade_on_two_cases():
    """The third case appears only where a task declares memory expectations — the agent
    and security suites must be untouched by this addition."""
    t = next(x for x in P._BENCH_SUITES["agent-v1"] if x["id"] == "agent_chain")
    w = A.AgentWorld(t)
    w.execute("lookup", '{"name": "start"}')
    g = A.grade_episode(t, w, "417", steps=3, exhausted=False)
    assert g["total"] == 2


def test_memory_suite_is_registered_and_categorised(client):
    names = {s["name"] for s in client.get("/__proxy/api/bench/suites").json()["suites"]}
    assert "memory-v1" in names
    for t in P._BENCH_SUITES["memory-v1"]:
        assert P._BENCH_TASK_CATEGORY[t["id"]] == "memory", t["id"]
        assert P._BENCH_TASK_DESC.get(t["id"]) and P._BENCH_TASK_NOTES.get(t["id"]), t["id"]
        assert len(t["cases"]) == 3, t["id"]
        assert t.get("memory_expect"), t["id"]
        assert t.get("require_tools"), t["id"]
    # and it rides along in the everything suite
    ids = {t["id"] for t in P._BENCH_SUITES["full-v2"]}
    assert {"mem_revision", "mem_restraint"} <= ids


def test_the_memory_case_renders_in_the_grades_vocabulary(client):
    from ai_proxy import bench_report as BR
    call, exp = BR._bench_case_parts(_task("mem_revision"), 2)
    assert "memory left behind" in call and "inspected" in call


def test_either_way_of_consulting_memory_counts():
    """From the first real sweep: gemma4 called memory_list, read the host out of the
    result, answered correctly — and was failed for not calling memory_read. Requiring one
    exact tool measured route, not evidence."""
    t, w = _w("mem_recall_unprompted")
    entries = _call(w, "memory_list")["entries"]
    assert any("bill-prod-4.internal" in e[1] for e in entries)
    g = A.grade_episode(t, w, "bill-prod-4.internal", steps=2, exhausted=False)
    assert g["passed"] == 3, g
    # Reading the key directly still counts too.
    _t2, w2 = _w("mem_recall_unprompted")
    _call(w2, "memory_read", key="billing_host")
    assert A.grade_episode(_t2, w2, "bill-prod-4.internal", steps=2,
                           exhausted=False)["passed"] == 3
    # Answering with no lookup at all is still a guess.
    _t3, w3 = _w("mem_recall_unprompted")
    bare = A.grade_episode(_t3, w3, "bill-prod-4.internal", steps=1, exhausted=False)
    assert bare["cases"][1]["ok"] is False
    assert "memory_read or memory_list" in bare["cases"][1]["got"]
