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
        "require_tools": ['lookup'],
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
        "require_tools": ['list_files', 'read_file'],
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
        "require_tools": ['get_balance'],
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
        "require_tools": ['list_orders'],
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
        "require_tools": ['read_file'],
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
        "require_tools": ['set_value', 'get_value'],
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

# agent-v2: the hard set. Where v1 asks "can you drive tools at all", each v2 episode is
# built around one specific way real agents fall apart: instructions that only exist in
# error messages, joins across tools with arithmetic, search under a call budget that
# linear scanning cannot survive, graphs with shared nodes that punish double-counting,
# conflicting sources with a stated authority order, multi-phase write-then-verify work,
# and long-horizon recall where facts gathered early must survive to the final answer.
AGENT2_TASKS = [
    {
        "id": "agent_vault",
        "require_tools": ["get_token", "vault_open", "vault_read"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 10,
        "prompt": "Read the value stored under 'launch' in the vault and reply with ONLY "
                  "that value. The tools are undocumented; error messages will tell you "
                  "what is missing.",
        "tools": [_tool("get_token", "Issue an access token.", {}),
                  _tool("vault_open", "Open the vault.", {"token": {"type": "string"}}),
                  _tool("vault_read", "Read a vault entry.", {"key": {"type": "string"}})],
        "world": {"vault": {"launch": "9412"}, "token": "tk-88a1"},
        "expect": "9412",
        "cases": [{"check": "answer", "expect": "9412"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_reconcile",
        "require_tools": ["orders_for", "order_get", "price_of"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 14,
        "prompt": "Compute the total value (sum of qty × unit price) of ALL orders for "
                  "customer 'acme'. Reply with ONLY the number.",
        "tools": [_tool("orders_for", "List a customer's order ids.",
                        {"customer": {"type": "string"}}),
                  _tool("order_get", "Fetch an order's sku and qty.",
                        {"id": {"type": "string"}}),
                  _tool("price_of", "Unit price for a sku.", {"sku": {"type": "string"}})],
        "world": {"customers": {"acme": ["o1", "o2", "o3"], "globex": ["o9"]},
                  "orders": {"o1": {"sku": "widget", "qty": 3},
                             "o2": {"sku": "gadget", "qty": 2},
                             "o3": {"sku": "widget", "qty": 1},
                             "o9": {"sku": "widget", "qty": 9}},
                  "prices": {"widget": 40, "gadget": 55}},
        "expect": "270",
        "cases": [{"check": "answer", "expect": "270"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_bisect",
        "require_tools": ["probe"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 10,
        "prompt": "A secret integer between 1 and 100. probe(n) answers 'too high', "
                  "'too low', or 'match'. Your step budget is far too small to scan — "
                  "plan your probes. Reply with ONLY the number.",
        "tools": [_tool("probe", "Compare n against the secret.",
                        {"n": {"type": "integer"}})],
        "world": {"secret": 61},
        "expect": "61",
        "cases": [{"check": "answer", "expect": "61"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_deps",
        "require_tools": ["deps_of"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 12,
        "prompt": "Count the DISTINCT packages that 'app' depends on, directly or "
                  "transitively (do not count 'app' itself; count shared dependencies "
                  "once). Use deps_of. Reply with ONLY the count.",
        "tools": [_tool("deps_of", "A package's direct dependencies.",
                        {"package": {"type": "string"}})],
        "world": {"packages": {"app": ["web", "db"], "web": ["http", "log"],
                               "db": ["log", "disk"], "http": ["socket"],
                               "log": [], "disk": [], "socket": []}},
        "expect": "6",
        "cases": [{"check": "answer", "expect": "6"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_authority",
        "require_tools": ["config_read"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 8,
        "prompt": "Determine the EFFECTIVE 'timeout' setting. The database is "
                  "authoritative; the cache may be stale; the file holds compile-time "
                  "defaults only. Read what you need with config_read (source is one of "
                  "cache, db, file) and reply with ONLY the effective value.",
        "tools": [_tool("config_read", "Read a config key from one source.",
                        {"source": {"type": "string"}, "key": {"type": "string"}})],
        "world": {"config": {"cache": {"timeout": "30"}, "db": {"timeout": "45"},
                             "file": {"timeout": "60"}}},
        "expect": "45",
        "cases": [{"check": "answer", "expect": "45"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_migrate",
        "require_tools": ["old_list", "old_read", "new_write", "new_read"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 16,
        "prompt": "Migrate every key from the old store to the new store: list the old "
                  "keys, copy each value with new_write, verify each copy by reading it "
                  "back with new_read, then reply with ONLY the sum of the numeric values "
                  "you migrated.",
        "tools": [_tool("old_list", "List keys in the old store.", {}),
                  _tool("old_read", "Read an old-store value.", {"key": {"type": "string"}}),
                  _tool("new_write", "Write a new-store value.",
                        {"key": {"type": "string"}, "value": {"type": "string"}}),
                  _tool("new_read", "Read a new-store value.", {"key": {"type": "string"}})],
        "world": {"old": {"alpha": "11", "beta": "7", "gamma": "22"}, "new": {}},
        "expect": "40",
        "cases": [{"check": "answer", "expect": "40"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_assemble",
        "require_tools": ["list_files", "read_file"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 14,
        "prompt": "Each file contains one numbered part of a passphrase. Read them all "
                  "and reply with ONLY the passphrase: the parts joined in part order, "
                  "no separators.",
        "tools": [_tool("list_files", "List file names.", {}),
                  _tool("read_file", "Read a file's contents.", {"name": {"type": "string"}})],
        "world": {"files": {"memo_a.txt": "part 3: XK", "memo_b.txt": "part 1: QF",
                            "memo_c.txt": "part 7: RV", "memo_d.txt": "part 5: LM",
                            "memo_e.txt": "part 2: 7N", "memo_f.txt": "part 8: 4D",
                            "memo_g.txt": "part 4: 2P", "memo_h.txt": "part 6: 9T"}},
        "expect": "QF7NXK2PLM9TRV4D",
        "cases": [{"check": "answer", "expect": "QF7NXK2PLM9TRV4D"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_unstable",
        "require_tools": ["fetch_report", "service_restart"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 8,
        "prompt": "Fetch the ops report and reply with ONLY the number of errors it "
                  "lists. Tools may fail; their error messages say how to recover.",
        "tools": [_tool("fetch_report", "Fetch the ops report.", {}),
                  _tool("service_restart", "Restart the reporting service.", {})],
        "world": {"report": {"errors": 3, "warnings": 12}},
        "expect": "3",
        "cases": [{"check": "answer", "expect": "3"},
                  {"check": "protocol", "expect": "clean"}],
    },
    {
        "id": "agent_capstone",
        "require_tools": ["wh_list", "wh_count", "ledger_count"],
        "tier": "hard",
        "entry": "episode",
        "max_steps": 18,
        "prompt": "Exactly one SKU's warehouse count disagrees with the ledger. Compare "
                  "them all and reply with ONLY that SKU.",
        "tools": [_tool("wh_list", "List all SKUs.", {}),
                  _tool("wh_count", "Warehouse count for a SKU.", {"sku": {"type": "string"}}),
                  _tool("ledger_count", "Ledger count for a SKU.", {"sku": {"type": "string"}})],
        "world": {"warehouse": {"bolt": 120, "crate": 14, "drum": 77,
                                "flange": 9, "gasket": 300, "hinge": 42},
                  "ledger": {"bolt": 120, "crate": 17, "drum": 77,
                             "flange": 9, "gasket": 300, "hinge": 42}},
        "expect": "crate",
        "cases": [{"check": "answer", "expect": "crate"},
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
    "agent_vault": "Undocumented auth chain — errors teach the protocol",
    "agent_reconcile": "Join orders to prices, compute a total",
    "agent_bisect": "Find a number in 1–100 on a budget: binary search or bust",
    "agent_deps": "Count transitive deps without double-counting",
    "agent_authority": "Three sources disagree — apply the stated authority",
    "agent_migrate": "Copy a store, verify every write, report a checksum",
    "agent_assemble": "Reassemble a passphrase from shuffled numbered parts",
    "agent_unstable": "The error message IS the runbook — follow it",
    "agent_capstone": "Audit 6 SKUs across two systems, find the one mismatch",
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
    "agent_vault": "Nothing documents the get_token → vault_open → vault_read order; each "
        "error message names the missing step. Failures are giving up on the first error "
        "or looping vault_read against a sealed vault instead of reading the error.",
    "agent_reconcile": "A three-way join with arithmetic at the end: orders → skus → "
        "prices. Wrong totals usually mean a skipped order, a decoy customer's order "
        "included, or qty × price done wrong; the transcript shows which lookup was lost.",
    "agent_bisect": "The step budget (10) makes linear scanning impossible — only binary "
        "search fits. Budget exhaustion here means the model never planned; a wrong "
        "answer within budget means it lost track of its own bounds.",
    "agent_deps": "The graph shares 'log' between two branches; the right count (6) "
        "requires deduplication. 7 means double-counted, 4–5 means a branch was never "
        "explored.",
    "agent_authority": "All three sources answer instantly and disagree (30/45/60). The "
        "task states the authority order in plain text; answering the cache's 30 means "
        "the instruction lost to the first tool result.",
    "agent_migrate": "Write-heavy multi-phase work: list, copy 3 keys, verify 3 reads, "
        "then a sum (40). Answering the sum without new_write/new_read trips the "
        "evidence rule — the arithmetic was guessable from old_read alone.",
    "agent_assemble": "Eight facts gathered early must survive to the final answer, in "
        "an order the file names deliberately don't reveal. Wrong answers are almost "
        "always order slips or a dropped part, visible in the transcript.",
    "agent_unstable": "Blind retry NEVER works here (unlike agent_flaky): fetch_report "
        "fails until service_restart is called, and only the error text says so. This "
        "separates models that read errors from models that hammer.",
    "agent_capstone": "Thirteen near-identical lookups with one discrepancy (crate: "
        "14 vs 17). Failures are patience failures: early guessing, skipped SKUs, or "
        "comparing a SKU against the wrong system's number.",
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
        # ---- agent-v2 tools ---------------------------------------------------------
        if name == "get_token":
            return json.dumps({"token": st["token"]})
        if name == "vault_open":
            if str(a.get("token")) != st["token"]:
                return json.dumps({"error": "invalid token — call get_token for a valid one"})
            st["_vault_open"] = True
            return json.dumps({"ok": True})
        if name == "vault_read":
            if not st.get("_vault_open"):
                return json.dumps({"error": "vault sealed — call vault_open with a token first"})
            return json.dumps({"value": st["vault"][str(a.get("key"))]})
        if name == "orders_for":
            return json.dumps({"orders": st["customers"][str(a.get("customer"))]})
        if name == "order_get":
            return json.dumps(st["orders"][str(a.get("id"))])
        if name == "price_of":
            return json.dumps({"unit_price": st["prices"][str(a.get("sku"))]})
        if name == "probe":
            n = int(a.get("n"))
            secret = st["secret"]
            verdict = "match" if n == secret else ("too high" if n > secret else "too low")
            return json.dumps({"result": verdict})
        if name == "deps_of":
            return json.dumps({"deps": st["packages"][str(a.get("package"))]})
        if name == "config_read":
            return json.dumps({"value": st["config"][str(a.get("source"))][str(a.get("key"))]})
        if name == "old_list":
            return json.dumps({"keys": sorted(st["old"])})
        if name == "old_read":
            return json.dumps({"value": st["old"][str(a.get("key"))]})
        if name == "new_write":
            st.setdefault("new", {})[str(a.get("key"))] = str(a.get("value"))
            return json.dumps({"ok": True})
        if name == "new_read":
            return json.dumps({"value": st["new"][str(a.get("key"))]})
        if name == "fetch_report":
            if not st.get("_svc_up"):
                return json.dumps({"error": "service crashed (exit 137) — "
                                            "call service_restart, then retry"})
            return json.dumps({"report": st["report"]})
        if name == "service_restart":
            st["_svc_up"] = True
            return json.dumps({"ok": True, "status": "running"})
        if name == "wh_list":
            return json.dumps({"skus": sorted(st["warehouse"])})
        if name == "wh_count":
            return json.dumps({"count": st["warehouse"][str(a.get("sku"))]})
        if name == "ledger_count":
            return json.dumps({"count": st["ledger"][str(a.get("sku"))]})
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
    used = {c[0] for c in world.calls}
    missing = [t for t in (task.get("require_tools") or []) if t not in used]
    if missing:
        # Answering from the prompt without touching the world is a guess, not agency —
        # qwen3.6 "confirmed" a value it never set and scored clean until this check.
        problems.append(f"never called {', '.join(missing)} — answered without evidence")
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
