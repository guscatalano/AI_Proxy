"""coding-v3's new wave must be solvable, and hard for the right reason.

Same contract as the v2 battery: every new task carries a reference solution run through the
REAL grader against the real cases (a wrong expected value fails every model and reads as a
discriminating task), and a naive solution that falls into the task's trap. Expected values
in bench_suites.py were COMPUTED by executing oracles, never hand-typed.

Three languages are new here: SQL grades in-process on sqlite (available everywhere), bash
and Go grade through subprocess/compile and skip when the toolchain is absent — the same
portability contract as every other language.
"""
import pytest

from ai_proxy import proxy as P

REFERENCE = {
    "quoted_split": '''
def quoted_split(s):
    out, cur, i, n = [], [], 0, len(s)
    in_q = has = False
    while i < n:
        ch = s[i]
        if in_q:
            if ch == "\\\\" and i + 1 < n and s[i + 1] in ('"', "\\\\"):
                cur.append(s[i + 1]); i += 2; continue
            if ch == '"':
                in_q = False; i += 1; continue
            cur.append(ch); i += 1; continue
        if ch == '"':
            in_q = has = True; i += 1; continue
        if ch in (" ", "\\t"):
            if has:
                out.append("".join(cur)); cur, has = [], False
            i += 1; continue
        cur.append(ch); has = True; i += 1
    if in_q:
        return None
    if has:
        out.append("".join(cur))
    return out
''',
    "ring_buffer": '''
def ring(capacity, ops):
    buf, out = [], []
    for op in ops:
        if op[0] == "push":
            if len(buf) == capacity:
                buf.pop(0)
            buf.append(op[1])
        else:
            out.append(buf.pop(0) if buf else None)
    return out
''',
    "expand_ranges": '''
def expand_ranges(s):
    if not s or not s.strip():
        return None
    seen = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            return None
        if "-" in part:
            a, _, b = part.partition("-")
            a, b = a.strip(), b.strip()
            if not (a.isdigit() and b.isdigit()) or int(a) > int(b):
                return None
            seen.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            seen.add(int(part))
        else:
            return None
    return sorted(seen)
''',
    "dedent_text": '''
def dedent_text(s):
    lines = s.split("\\n")
    solid = [l for l in lines if l.strip()]
    if not solid:
        return "\\n".join("" for _ in lines)
    prefix = None
    for l in solid:
        ws = l[:len(l) - len(l.lstrip(" \\t"))]
        if prefix is None:
            prefix = ws
        else:
            keep = []
            for a, b in zip(prefix, ws):
                if a == b:
                    keep.append(a)
                else:
                    break
            prefix = "".join(keep)
    k = len(prefix or "")
    return "\\n".join("" if not l.strip() else l[k:] for l in lines)
''',
    "deep_get": '''
function deepGet(obj, path) {
  if (!path) return null;
  let cur = obj;
  for (const part of path.split(".")) {
    const m = /^([^\\[\\]]+)((?:\\[\\d+\\])*)$/.exec(part);
    if (!m) return null;
    if (cur === null || typeof cur !== "object" || Array.isArray(cur)
        || !(m[1] in cur)) return null;
    cur = cur[m[1]];
    for (const i of (m[2].match(/\\d+/g) || [])) {
      const k = Number(i);
      if (!Array.isArray(cur) || k >= cur.length) return null;
      cur = cur[k];
    }
  }
  return cur === undefined ? null : cur;
}
''',
    "sql_top_spenders": '''
SELECT customer, SUM(amount) AS total FROM orders
GROUP BY customer HAVING SUM(amount) >= 100
ORDER BY total DESC, customer ASC
''',
    "sql_missing_users": '''
SELECT name FROM users u
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
ORDER BY name ASC
''',
    "bash_dedup_lines": "awk '!seen[$0]++'\n",
    "bash_sort_versions": "sort -t. -k1,1n -k2,2n -k3,3n\n",
    "go_rle_decode": '''
import "strings"

func RleDecode(s string) string {
    var b strings.Builder
    i := 0
    for i < len(s) {
        ch := s[i]
        if !(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') {
            return ""
        }
        i++
        j := i
        for j < len(s) && s[j] >= '0' && s[j] <= '9' {
            j++
        }
        if j == i || s[i] == '0' {
            return ""
        }
        n := 0
        for _, d := range s[i:j] {
            n = n*10 + int(d-'0')
        }
        for k := 0; k < n; k++ {
            b.WriteByte(ch)
        }
        i = j
    }
    return b.String()
}
''',
    "go_ipv4_valid": '''
import "strings"

func IPv4Valid(s string) bool {
    parts := strings.Split(s, ".")
    if len(parts) != 4 {
        return false
    }
    for _, p := range parts {
        if p == "" || len(p) > 3 {
            return false
        }
        for _, c := range p {
            if c < '0' || c > '9' {
                return false
            }
        }
        if len(p) > 1 && p[0] == '0' {
            return false
        }
        n := 0
        for _, c := range p {
            n = n*10 + int(c-'0')
        }
        if n > 255 {
            return false
        }
    }
    return true
}
''',
    "c_bit_count_range": '''
long long bit_count_range(int a, int b) {
    long long total = 0;
    for (long long i = a; i <= (long long)b; i++) {
        unsigned int v = (unsigned int)i;
        while (v) { total += v & 1u; v >>= 1; }
    }
    return total;
}
''',
    "cpp_wrap_count": '''
int wrap_count(const std::string& s, int width) {
    std::vector<std::string> words;
    std::string w;
    for (char c : s) {
        if (c == ' ') { if (!w.empty()) { words.push_back(w); w.clear(); } }
        else w += c;
    }
    if (!w.empty()) words.push_back(w);
    if (words.empty()) return 0;
    int lines = 1;
    int cur = (int)words[0].size();
    for (size_t i = 1; i < words.size(); i++) {
        int need = cur + 1 + (int)words[i].size();
        if (need <= width) cur = need;
        else { lines++; cur = (int)words[i].size(); }
    }
    return lines;
}
''',
    "rust_kv_get": '''
fn kv_get(s: &str, key: &str) -> String {
    let mut val = String::new();
    for entry in s.split(';') {
        if let Some(eq) = entry.find('=') {
            if &entry[..eq] == key {
                val = entry[eq + 1..].to_string();
            }
        }
    }
    val
}
''',
    "cs_round_half": '''
public static class Sol {
    public static int RoundHalf(double v) {
        return v >= 0 ? (int)System.Math.Floor(v + 0.5)
                      : (int)System.Math.Ceiling(v - 0.5);
    }
}
''',
    "php_flatten_keys": '''
function flatten_keys(array $a, string $prefix = ""): array {
    $out = [];
    foreach ($a as $k => $v) {
        $kk = $prefix === "" ? (string)$k : $prefix . "." . $k;
        if (is_array($v)) {
            $out = array_merge($out, flatten_keys($v, $kk));
        } else {
            $out[$kk] = $v;
        }
    }
    return $out;
}
''',
    "html_nav_current": '''
<nav>
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/about" aria-current="page">About</a></li>
    <li><a href="/contact">Contact</a></li>
  </ul>
</nav>
''',
    "css_sticky_footer": '''
body { display: flex; flex-direction: column; min-height: 100vh; }
footer { margin-top: auto; }
''',
}

NAIVE = {
    "quoted_split": "def quoted_split(s):\n    return s.split()\n",
    "ring_buffer": ('def ring(capacity, ops):\n'
                    '    buf, out = [], []\n'
                    '    for op in ops:\n'
                    '        if op[0] == "push":\n'
                    '            if len(buf) < capacity:\n'
                    '                buf.append(op[1])\n'
                    '        else:\n'
                    '            out.append(buf.pop(0) if buf else None)\n'
                    '    return out\n'),
    "expand_ranges": ('def expand_ranges(s):\n'
                      '    seen = set()\n'
                      '    for part in s.split(","):\n'
                      '        part = part.strip()\n'
                      '        if "-" in part:\n'
                      '            a, _, b = part.partition("-")\n'
                      '            seen.update(range(int(a), int(b) + 1))\n'
                      '        else:\n'
                      '            seen.add(int(part))\n'
                      '    return sorted(seen)\n'),
    "dedent_text": ('def dedent_text(s):\n'
                    '    return "\\n".join(l.strip() for l in s.split("\\n"))\n'),
    "deep_get": ('function deepGet(obj, path) {\n'
                 '  let cur = obj;\n'
                 '  for (const k of path.split(".")) {\n'
                 '    if (cur == null || !(k in cur)) return null;\n'
                 '    cur = cur[k];\n'
                 '  }\n'
                 '  return cur === undefined ? null : cur;\n'
                 '}\n'),
    "sql_top_spenders": ("SELECT customer, SUM(amount) AS total FROM orders\n"
                         "WHERE amount >= 100 GROUP BY customer\n"
                         "ORDER BY total DESC, customer ASC\n"),
    "sql_missing_users": ("SELECT name FROM users\n"
                          "WHERE id NOT IN (SELECT user_id FROM orders)\n"
                          "ORDER BY name ASC\n"),
    "bash_dedup_lines": "sort -u\n",
    "bash_sort_versions": "sort\n",
    "go_ipv4_valid": '''
import (
    "strconv"
    "strings"
)

func IPv4Valid(s string) bool {
    parts := strings.Split(s, ".")
    if len(parts) != 4 {
        return false
    }
    for _, p := range parts {
        n, err := strconv.Atoi(p)
        if err != nil || n < 0 || n > 255 {
            return false
        }
    }
    return true
}
''',
    "c_bit_count_range": '''
long long bit_count_range(int a, int b) {
    long long total = 0;
    for (int i = a; i <= b; i++) {
        unsigned int v = (unsigned int)i;
        while (v) { total += v & 1u; v >>= 1; }
    }
    return total;
}
''',
    "cpp_wrap_count": '''
int wrap_count(const std::string& s, int width) {
    if (s.empty()) return 0;
    int lines = 1, cur = 0;
    std::string w;
    std::vector<std::string> words;
    for (char c : s) {
        if (c == ' ') { if (!w.empty()) { words.push_back(w); w.clear(); } }
        else w += c;
    }
    if (!w.empty()) words.push_back(w);
    for (auto& word : words) {
        int len = (int)word.size();
        while (len > width) { lines++; len -= width; }
        if (cur == 0) cur = len;
        else if (cur + 1 + len <= width) cur += 1 + len;
        else { lines++; cur = len; }
    }
    return lines;
}
''',
    "rust_kv_get": '''
fn kv_get(s: &str, key: &str) -> String {
    for entry in s.split(';') {
        let parts: Vec<&str> = entry.split('=').collect();
        if parts.len() >= 2 && parts[0] == key {
            return parts[1].to_string();
        }
    }
    String::new()
}
''',
    "cs_round_half": ('public static class Sol {\n'
                      '    public static int RoundHalf(double v) {\n'
                      '        return (int)System.Math.Round(v);\n'
                      '    }\n'
                      '}\n'),
    "php_flatten_keys": ('function flatten_keys(array $a): array {\n'
                         '    $out = [];\n'
                         '    foreach ($a as $k => $v) {\n'
                         '        if (is_array($v)) {\n'
                         '            foreach ($v as $k2 => $v2) { $out[$k . "." . $k2] = $v2; }\n'
                         '        } else { $out[$k] = $v; }\n'
                         '    }\n'
                         '    return $out;\n'
                         '}\n'),
    "html_nav_current": ('<nav><a href="/">Home</a><a href="/about">About</a>'
                         '<a href="/contact">Contact</a></nav>'),
    "css_sticky_footer": ("footer { position: fixed; bottom: 0; }\n"),
}


def _tasks():
    return {t["id"]: t for t in P._BENCH_SUITES["coding-v3"]}


def _runtime_missing(lang):
    return not P._bench_lang_available(lang)


@pytest.mark.parametrize("task_id", sorted(REFERENCE))
def test_v3_reference_passes_every_case(task_id):
    task = _tasks()[task_id]
    lang = task.get("lang") or "python"
    if _runtime_missing(lang):
        pytest.skip(f"{lang} runtime not on this host")
    res = P._bench_grade_sync(REFERENCE[task_id], task["entry"], task["cases"], 30.0, lang)
    assert res.get("passed") == res.get("total"), f"{task_id}: {res}"


@pytest.mark.parametrize("task_id", sorted(NAIVE))
def test_v3_naive_answers_fall_into_the_trap(task_id):
    task = _tasks()[task_id]
    lang = task.get("lang") or "python"
    if _runtime_missing(lang):
        pytest.skip(f"{lang} runtime not on this host")
    res = P._bench_grade_sync(NAIVE[task_id], task["entry"], task["cases"], 30.0, lang)
    assert res.get("passed", 0) < res.get("total", 1), f"{task_id}: naive scored full marks"


def test_v3_is_a_superset_of_v2():
    v2 = {t["id"] for t in P._BENCH_SUITES["coding-v2"]}
    v3 = {t["id"] for t in P._BENCH_SUITES["coding-v3"]}
    assert v2 < v3
    assert len(P._BENCH_SUITES["coding-v3"]) == 47


def test_v3_spans_twelve_languages():
    langs = {t.get("lang") or "python" for t in P._BENCH_SUITES["coding-v3"]}
    assert langs == {"python", "js", "c", "cpp", "rust", "csharp", "php", "html", "css",
                     "go", "sql", "bash"}


def test_every_v3_task_has_a_description():
    missing = [t["id"] for t in P._BENCH_SUITES["coding-v3"]
               if not P._BENCH_TASK_DESC.get(t["id"])]
    assert not missing, missing


def test_sql_grades_without_any_toolchain():
    assert P._bench_lang_available("sql") is True
    bad = P._bench_grade_sql("SELECT nope FROM missing",
                             [{"setup": "CREATE TABLE t(x);", "expect": []}])
    assert bad["passed"] == 0 and bad["cases"][0].get("got")


def test_graded_suite_defaults_carry_more_tokens(client):
    """The 512-token era truncated long answers mid-function; graded runs must default to a
    budget the long tasks can finish in. An explicit max_tokens still wins."""
    import json as _json
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/bench/run",
                    json={"model": "qwen3:4b", "suite": "coding-v3", "runs": 1})
    assert r.status_code == 200, r.text
    assert (r.json().get("config") or {}).get("max_tokens") == 1024
    r2 = client.post("/__proxy/api/bench/run",
                     json={"model": "qwen3:4b", "runs": 1})
    if r2.status_code == 200:        # a concurrent-bench refusal is fine; the default isn't it
        assert (r2.json().get("config") or {}).get("max_tokens") == 256
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()


def test_a_present_but_broken_toolchain_reads_as_unavailable(monkeypatch):
    """A rustc with no linker compiles nothing. 'Available' must mean the grader end-to-end,
    or every task in that language zeroes on the box instead of skipping."""
    monkeypatch.setattr(P, "_bench_tool", lambda name: "/fake/" + name)
    monkeypatch.setattr(P, "_bench_grade_sync",
                        lambda *a, **k: {"passed": 0, "total": 1, "error": "link failed"})
    monkeypatch.setitem(P.__dict__, "_BENCH_TOOLCHAIN_OK", {})
    assert P._bench_lang_available("rust") is False
    monkeypatch.setattr(P, "_bench_grade_sync",
                        lambda *a, **k: {"passed": 1, "total": 1})
    monkeypatch.setitem(P.__dict__, "_BENCH_TOOLCHAIN_OK", {})
    assert P._bench_lang_available("rust") is True
