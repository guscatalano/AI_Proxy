"""coding-v2 must be solvable, and hard for the right reason.

coding-v1 is saturated: seventeen of its eighteen tasks were solved perfectly in all twenty-three
recorded runs, including its entire "hard" tier. Only `calculator` ever discriminated, and it is
the one task where a *plausible* solution fails rather than an unfamiliar one.

So v2's difficulty is specification traps and adversarial cases, not exotic algorithms. Which
makes this file essential: an expected value that is simply wrong would fail every model and be
indistinguishable from a task that discriminates. Every task here carries a reference solution,
and each is run through the real grader against the real cases.
"""
import pytest

from ai_proxy import proxy as P

REFERENCE = {}


def ref(name):
    def take(fn):
        REFERENCE[name] = fn.__doc__
        return fn
    return take


@ref("semver_cmp")
def _semver():
    '''
def semver_cmp(a, b):
    def parse(v):
        v = v.split("+", 1)[0]
        core, _, pre = v.partition("-")
        nums = [int(x) for x in core.split(".")]
        return nums, (pre.split(".") if pre else None)

    an, ap = parse(a)
    bn, bp = parse(b)
    if an != bn:
        return -1 if an < bn else 1
    if ap is None and bp is None:
        return 0
    if ap is None:
        return 1
    if bp is None:
        return -1
    for x, y in zip(ap, bp):
        xd, yd = x.isdigit(), y.isdigit()
        if xd and yd:
            if int(x) != int(y):
                return -1 if int(x) < int(y) else 1
        elif xd != yd:
            return -1 if xd else 1
        elif x != y:
            return -1 if x < y else 1
    if len(ap) != len(bp):
        return -1 if len(ap) < len(bp) else 1
    return 0
'''


@ref("csv_line")
def _csv():
    '''
def csv_line(line):
    out, cur, i, n = [], [], 0, len(line)
    while i < n:
        if not cur and line[i] == '"':
            i += 1
            while i < n:
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        cur.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                cur.append(line[i])
                i += 1
            while i < n and line[i] != ",":
                cur.append(line[i])
                i += 1
            continue
        if line[i] == ",":
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(line[i])
        i += 1
    out.append("".join(cur))
    return out
'''


@ref("glob_match")
def _glob():
    '''
def glob_match(pattern, text):
    m, n = len(pattern), len(text)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(1, m + 1):
        if pattern[i - 1] == "*":
            dp[i][0] = dp[i - 1][0]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = pattern[i - 1]
            if c == "*":
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif c == "?" or c == text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
'''


@ref("roman_strict")
def _roman():
    '''
def roman_strict(s):
    if not s:
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if any(c not in vals for c in s):
        return None
    import re
    pat = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    if not re.match(pat, s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = vals[c]
        total += v if v >= prev else -v
        prev = max(prev, v)
    return total if 1 <= total <= 3999 else None
'''


@ref("topo_lex")
def _topo():
    '''
def topo_lex(n, edges):
    import heapq
    adj = {i: set() for i in range(n)}
    indeg = [0] * n
    seen = set()
    for u, v in edges:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        adj[u].add(v)
        indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(h)
    out = []
    while h:
        u = heapq.heappop(h)
        out.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(h, v)
    return out if len(out) == n else None
'''


@ref("lru_ops")
def _lru():
    '''
def lru_ops(capacity, ops):
    from collections import OrderedDict
    cache = OrderedDict()
    out = []
    for op in ops:
        if op[0] == "put":
            _, k, v = op
            if k in cache:
                cache.move_to_end(k)
            cache[k] = v
            if len(cache) > capacity:
                cache.popitem(last=False)
        else:
            k = op[1]
            if k in cache:
                cache.move_to_end(k)
                out.append(cache[k])
            else:
                out.append(-1)
    return out
'''


@ref("justify")
def _justify():
    '''
def justify(words, width):
    lines, cur, ln = [], [], 0
    for w in words:
        if cur and ln + len(cur) + len(w) > width:
            lines.append(cur)
            cur, ln = [], 0
        cur.append(w)
        ln += len(w)
    if cur:
        lines.append(cur)
    out = []
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        if last or len(line) == 1:
            s = " ".join(line)
            out.append(s + " " * (width - len(s)))
            continue
        chars = sum(len(w) for w in line)
        gaps = len(line) - 1
        base, extra = divmod(width - chars, gaps)
        s = ""
        for j, w in enumerate(line):
            s += w
            if j < gaps:
                s += " " * (base + (1 if j < extra else 0))
        out.append(s)
    return out
'''


@ref("path_norm")
def _path():
    '''
def path_norm(path):
    parts = []
    for seg in path.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/" + "/".join(parts)
'''


@ref("json_pointer")
def _ptr():
    '''
def json_pointer(doc, pointer):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        return None
    cur = doc
    for raw in pointer[1:].split("/"):
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            if not tok.isdigit() or (len(tok) > 1 and tok[0] == "0"):
                return None
            i = int(tok)
            if i >= len(cur):
                return None
            cur = cur[i]
        elif isinstance(cur, dict):
            if tok not in cur:
                return None
            cur = cur[tok]
        else:
            return None
    return cur
'''


@ref("base_convert")
def _base():
    '''
def base_convert(s, frm, to):
    if not (2 <= frm <= 36 and 2 <= to <= 36):
        return None
    if not s:
        return None
    neg = s[0] == "-"
    body = s[1:] if neg else s
    if not body:
        return None
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    val = 0
    for ch in body.lower():
        d = digits.find(ch)
        if d < 0 or d >= frm:
            return None
        val = val * frm + d
    if val == 0:
        return "0"
    out = ""
    while val:
        out = digits[val % to] + out
        val //= to
    return ("-" + out) if neg else out
'''


@ref("interval_intersect")
def _ivl():
    '''
def interval_intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out
'''


@ref("tokenize_expr")
def _tok():
    '''
def tokenize_expr(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/()":
            out.append(c)
            i += 1
            continue
        if c.isdigit() or c == ".":
            j = i
            dots = 0
            while j < n and (s[j].isdigit() or s[j] == "."):
                if s[j] == ".":
                    dots += 1
                j += 1
            chunk = s[i:j]
            if dots > 1:
                return None
            out.append(float(chunk) if dots else int(chunk))
            i = j
            continue
        return None
    return out
'''


def _tasks():
    return {t["id"]: t for t in P._BENCH_SUITES["coding-v2"]}


def test_every_task_has_a_reference_solution():
    """A task nobody has solved is a task whose expected values nobody has checked."""
    assert set(_tasks()) == set(REFERENCE), (
        f"missing references: {set(_tasks()) - set(REFERENCE)}")


@pytest.mark.parametrize("task_id", sorted(REFERENCE))
def test_the_reference_solution_passes_every_case(task_id):
    """Run through the real grader, not a local call: this proves the cases are right *and*
    that the grader can execute and score them."""
    task = _tasks()[task_id]
    res = P._bench_grade_sync(REFERENCE[task_id], task["entry"], task["cases"], 20.0)
    assert res.get("passed") == len(task["cases"]), (
        f"{task_id}: {res.get('passed')}/{len(task['cases'])} — {res.get('failures') or res}")


def test_the_suite_is_registered_and_selectable(client):
    d = client.get("/__proxy/api/bench/suites").json()
    names = {s["name"] for s in d["suites"]}
    assert "coding-v2" in names and "coding-v1" in names, "v1 must survive for comparability"


def test_v1_is_left_untouched():
    """Reuse keys on the suite name, so changing v1 would silently invalidate every recorded
    result rather than obviously replacing it."""
    assert len(P._BENCH_SUITES["coding-v1"]) == 18


def test_the_traps_are_actually_traps():
    """Each task carries at least one case that the obvious wrong implementation fails. Without
    this the suite is just longer, not harder."""
    t = _tasks()
    # A permissive Roman parser returns a number for these.
    assert {"IIII", "VV", "IC", "MCMC"} <= {c["args"][0] for c in t["roman_strict"]["cases"]}
    # A pre-release must rank below its release.
    assert {"expect": -1, "args": ["1.0.0-alpha", "1.0.0"]} in t["semver_cmp"]["cases"]
    # Greedy glob matching fails this.
    assert {"args": ["*a*b", "aaab"], "expect": True} in t["glob_match"]["cases"]
    # '..' must not escape the root.
    assert {"args": ["/../"], "expect": "/"} in t["path_norm"]["cases"]
    # ~1 decodes before ~0.
    assert {"args": [{"~1": 9}, "/~01"], "expect": 9} in t["json_pointer"]["cases"]


# ---- the naive solutions must fail ---------------------------------------------------------
# Solvable is half the requirement. If the obvious wrong implementation also scores 100%, the
# task is longer than v1's but no harder, which is exactly the failure being corrected.

NAIVE = {
    # Splits on commas with no notion of quoting.
    "csv_line": 'def csv_line(line):\n    return line.split(",")\n',
    # Classic additive/subtractive parse with no validation: "IIII" happily returns 4.
    "roman_strict": (
        'def roman_strict(s):\n'
        '    v = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}\n'
        '    t = 0\n'
        '    for i, c in enumerate(s):\n'
        '        if i + 1 < len(s) and v[c] < v[s[i+1]]:\n'
        '            t -= v[c]\n'
        '        else:\n'
        '            t += v[c]\n'
        '    return t\n'
    ),
    # Greedy '*' consumption: fails "*a*b" against "aaab".
    "glob_match": (
        'def glob_match(pattern, text):\n'
        '    i = j = 0\n'
        '    while i < len(pattern) and j < len(text):\n'
        '        c = pattern[i]\n'
        '        if c == "*":\n'
        '            i += 1\n'
        '            if i == len(pattern):\n'
        '                return True\n'
        '            while j < len(text) and text[j] != pattern[i]:\n'
        '                j += 1\n'
        '        elif c == "?" or c == text[j]:\n'
        '            i += 1\n'
        '            j += 1\n'
        '        else:\n'
        '            return False\n'
        '    return i == len(pattern) and j == len(text)\n'
    ),
    # Compares core numbers only, so a pre-release ties with its release.
    "semver_cmp": (
        'def semver_cmp(a, b):\n'
        '    pa = [int(x) for x in a.split("+")[0].split("-")[0].split(".")]\n'
        '    pb = [int(x) for x in b.split("+")[0].split("-")[0].split(".")]\n'
        '    return (pa > pb) - (pa < pb)\n'
    ),
    # Decodes ~0 before ~1, so "~01" becomes "/" instead of "~1".
    "json_pointer": (
        'def json_pointer(doc, pointer):\n'
        '    if pointer == "":\n'
        '        return doc\n'
        '    cur = doc\n'
        '    for raw in pointer[1:].split("/"):\n'
        '        tok = raw.replace("~0", "~").replace("~1", "/")\n'
        '        try:\n'
        '            cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]\n'
        '        except Exception:\n'
        '            return None\n'
        '    return cur\n'
    ),
    # Distributes the extra spaces on the right gaps instead of the left.
    "justify": (
        'def justify(words, width):\n'
        '    lines, cur, ln = [], [], 0\n'
        '    for w in words:\n'
        '        if cur and ln + len(cur) + len(w) > width:\n'
        '            lines.append(cur); cur, ln = [], 0\n'
        '        cur.append(w); ln += len(w)\n'
        '    if cur: lines.append(cur)\n'
        '    out = []\n'
        '    for i, line in enumerate(lines):\n'
        '        if i == len(lines) - 1 or len(line) == 1:\n'
        '            s = " ".join(line); out.append(s + " " * (width - len(s))); continue\n'
        '        chars = sum(len(w) for w in line); gaps = len(line) - 1\n'
        '        base, extra = divmod(width - chars, gaps)\n'
        '        s = ""\n'
        '        for j, w in enumerate(line):\n'
        '            s += w\n'
        '            if j < gaps:\n'
        '                s += " " * (base + (1 if j >= gaps - extra else 0))\n'
        '        out.append(s)\n'
        '    return out\n'
    ),
}


@pytest.mark.parametrize("task_id", sorted(NAIVE))
def test_the_obvious_wrong_answer_is_caught(task_id):
    task = _tasks()[task_id]
    res = P._bench_grade_sync(NAIVE[task_id], task["entry"], task["cases"], 20.0)
    assert res.get("passed", 0) < len(task["cases"]), (
        f"{task_id}: the naive implementation scored full marks — the task is not a trap")
