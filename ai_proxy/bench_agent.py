"""agent-v1: long-running multi-tool episodes, graded on completion AND discipline.

Single-shot suites cannot see what kills models in real agentic use: holding state across
many tool calls, recovering from a tool error, iterating pagination to the end, not
re-issuing identical calls, and still producing a clean final answer late in a long
conversation. Each task here is an EPISODE: the bench plays the tool side with a
deterministic simulated world, the model drives, and the grade is two cases — did it reach
the right answer, and did it behave (no malformed calls, no hallucinated tools, no exact
repeats, within the step budget).

The world is pure data per episode; tools are pure functions over it. Nothing here talks to
the network — the engine owns the HTTP loop.
"""
import json


def _tool(name, desc, params):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": params,
                                        "required": list(params)}}}


AGENT_TASKS = [
    {
        "id": "agent_chain",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 14,
        "prompt": "Start with the ledger entry named 'start'. Each entry's value names "
                  "another entry, until one holds a number. Follow the chain with the "
                  "lookup tool and reply with ONLY that final number.",
        "tools": [_tool("lookup", "Return a ledger entry's value.", {"name": {"type": "string"}})],
        "world": {"ledger": {"start": "see:brass", "brass": "see:copper", "copper": "see:dawn",
                             "dawn": "see:ember", "ember": "see:flint", "flint": "see:gale",
                             "gale": "see:harbor", "harbor": "417"}},
        "expect": "417",
        "cases": [{"check": "answer", "expect": "417"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_sum_files",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 12,
        "prompt": "Using the tools, list the files, read every .num file, and reply with "
                  "ONLY the sum of the numbers they contain.",
        "tools": [_tool("list_files", "List file names.", {}),
                  _tool("read_file", "Read a file's contents.", {"name": {"type": "string"}})],
        "world": {"files": {"a.num": "12", "notes.txt": "ignore me", "b.num": "30",
                            "c.num": "5", "readme.md": "docs", "d.num": "111"}},
        "expect": "158",
        "cases": [{"check": "answer", "expect": "158"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_flaky",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 8,
        "prompt": "Fetch the balance of account 'ax9' with the get_balance tool and reply "
                  "with ONLY the number. The service is unreliable; if a call errors, try "
                  "again.",
        "tools": [_tool("get_balance", "Return an account's balance.",
                        {"account": {"type": "string"}})],
        "world": {"balances": {"ax9": "2044"}, "flaky_failures": 2},
        "expect": "2044",
        "cases": [{"check": "answer", "expect": "2044"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_paginate",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 12,
        "prompt": "Count the orders with the list_orders tool. Pages are numbered from 1; "
                  "keep requesting the next page until one comes back empty, then reply "
                  "with ONLY the total number of orders.",
        "tools": [_tool("list_orders", "Return one page of order ids (may be empty).",
                        {"page": {"type": "integer"}})],
        "world": {"pages": [["o1", "o2", "o3"], ["o4", "o5", "o6"], ["o7", "o8"], ["o9"]]},
        "expect": "9",
        "cases": [{"check": "answer", "expect": "9"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_needle",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 16,
        "prompt": "Exactly one of the files contains the phrase 'launch code'. Find it "
                  "with the tools and reply with ONLY the number on that line.",
        "tools": [_tool("list_files", "List file names.", {}),
                  _tool("read_file", "Read a file's contents.", {"name": {"type": "string"}})],
        "world": {"files": {f"log{i}.txt": f"routine entry {i}\nnothing here"
                            for i in range(1, 9)}
                  | {"log5.txt": "routine entry 5\nlaunch code 7731\ntrailing noise"}},
        "expect": "7731",
        "cases": [{"check": "answer", "expect": "7731"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_update_verify",
        "tier": "hard",
        "entry": "episode",
        "max_steps": 10,
        "prompt": "Set the key 'mode' to 'ready' with set_value, then read it back with "
                  "get_value to confirm, and reply with ONLY the confirmed value.",
        "tools": [_tool("set_value", "Store a value under a key.",
                        {"key": {"type": "string"}, "value": {"type": "string"}}),
                  _tool("get_value", "Read a stored value.", {"key": {"type": "string"}})],
        "world": {"kv": {"mode": "idle"}},
        "expect": "ready",
        "cases": [{"check": "answer", "expect": "ready"},
                  {"check": "protocol", "expect": "clean"}],
    },
]

AGENT_TASK_DESC = {
    "agent_chain": "Follow an 8-link pointer chain with a lookup tool",
    "agent_sum_files": "List, filter, read many files, aggregate",
    "agent_flaky": "A tool that errors twice — retry, don't give up",
    "agent_paginate": "Iterate pages until empty, count the total",
    "agent_needle": "Search many files for one fact",
    "agent_update_verify": "Write state, read it back, confirm",
}

AGENT_TASK_NOTES = {
    "agent_chain": "Endurance and state: eight dependent calls with nothing to parallelize. "
        "Failures are usually giving up mid-chain, re-looking-up an earlier link, or "
        "answering with an intermediate 'see:' value.",
    "agent_sum_files": "Selection plus aggregation: the .txt and .md files are decoys. "
        "Wrong sums usually mean a skipped file or a decoy read as data.",
    "agent_flaky": "The first two calls FAIL by design. Answering from the error message, "
        "or giving up after one attempt, is the failure this task exists to catch.",
    "agent_paginate": "The stop condition is an EMPTY page, not a fixed count. Stopping at "
        "the first short page (3 items → 1 item) undercounts; never stopping burns the "
        "step budget.",
    "agent_needle": "Patience under repetition: eight near-identical files. Failures are "
        "early guessing or losing track of which files were already read.",
    "agent_update_verify": "Write-then-verify ordering: answering 'ready' without the "
        "read-back also passes the answer check but trips protocol if set_value was never "
        "called — the transcript shows which happened.",
}


class AgentWorld:
    """Deterministic tool execution over an episode's world state."""

    def __init__(self, task: dict):
        self.task = task
        self.state = json.loads(json.dumps(task.get("world") or {}))
        self.calls: list = []
        self.malformed = 0
        self.unknown = 0
        self.repeats = 0
        self.flaky_left = int(self.state.get("flaky_failures") or 0)
        self._last_ok: dict = {}

    def execute(self, name: str, raw_args) -> str:
        """Run one tool call, returning the tool message content (always a string)."""
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError) as e:
            self.malformed += 1
            return json.dumps({"error": f"malformed arguments: {e}"})
        sig = (name, json.dumps(args, sort_keys=True))
        # A repeat only counts against conduct when the previous identical call SUCCEEDED —
        # retrying after an error is what a competent agent does (agent_flaky demands it),
        # and the first version of this rule failed every model that did the right thing.
        if self._last_ok.get(sig):
            self.repeats += 1
        self.calls.append(sig)
        try:
            out = self._dispatch(name, args)
        except KeyError as e:
            out = json.dumps({"error": f"not found: {e}"})
        except Exception as e:
            self.malformed += 1
            out = json.dumps({"error": f"{type(e).__name__}: {e}"})
        try:
            self._last_ok[sig] = "error" not in (json.loads(out) or {})
        except (json.JSONDecodeError, TypeError):
            self._last_ok[sig] = True
        return out

    def _dispatch(self, name: str, a: dict) -> str:
        st = self.state
        if name == "lookup":
            return json.dumps({"value": st["ledger"][str(a.get("name"))]})
        if name == "list_files":
            return json.dumps({"files": sorted(st["files"])})
        if name == "read_file":
            return json.dumps({"contents": st["files"][str(a.get("name"))]})
        if name == "get_balance":
            if self.flaky_left > 0:
                self.flaky_left -= 1
                return json.dumps({"error": "service busy, please retry"})
            return json.dumps({"balance": st["balances"][str(a.get("account"))]})
        if name == "list_orders":
            page = int(a.get("page") or 0)
            pages = st["pages"]
            return json.dumps({"orders": pages[page - 1] if 1 <= page <= len(pages) else []})
        if name == "set_value":
            st.setdefault("kv", {})[str(a.get("key"))] = str(a.get("value"))
            return json.dumps({"ok": True})
        if name == "get_value":
            return json.dumps({"value": st["kv"][str(a.get("key"))]})
        self.unknown += 1
        return json.dumps({"error": f"no such tool: {name}"})


def grade_episode(task: dict, world: AgentWorld, final_text: str | None,
                  steps: int, exhausted: bool) -> dict:
    """Two cases: the answer, and the conduct. Partial credit is the point — a model that
    reaches the right number by hammering malformed calls is not the same as one that
    walks there cleanly, and a single pass/fail would hide the difference."""
    expect = str(task.get("expect"))
    ans = (final_text or "").strip()
    answer_ok = expect in ans and len(ans) < 20 + len(expect) * 3
    problems = []
    if world.malformed:
        problems.append(f"{world.malformed} malformed call(s)")
    if world.unknown:
        problems.append(f"{world.unknown} hallucinated tool(s)")
    if world.repeats:
        problems.append(f"{world.repeats} exact repeat call(s)")
    if exhausted:
        problems.append(f"step budget exhausted ({task.get('max_steps')})")
    protocol_ok = not problems
    cases = [{"ok": answer_ok, **({} if answer_ok else {"got": ans[:120] or None})},
             {"ok": protocol_ok, **({} if protocol_ok else {"got": "; ".join(problems)})}]
    return {"passed": sum(1 for c in cases if c["ok"]), "total": 2, "cases": cases,
            "steps": steps, "malformed": world.malformed, "repeats": world.repeats,
            "exhausted": exhausted}
