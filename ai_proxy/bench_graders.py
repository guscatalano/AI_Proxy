"""Grading: languages, code extraction, toolchain probes, per-language graders.

Everything here is synchronous, self-contained, and importable without the proxy app: the
subprocess runners, the compile-and-run harnesses, the structural HTML/CSS checkers, and the
availability probes that make "this box can grade language X" a verified fact rather than a
PATH guess. The async wrapper (_bench_grade) stays with the engine.

SECURITY: this executes model-generated code — separate process, hard timeout, scratch cwd,
stripped env. That contains accidents and runaway loops; it is NOT a sandbox against
deliberately hostile code. Grading is opt-in per bench run.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

try:
    from .bench_suites import SUITES
except ImportError:          # flat-script launch: modules sit beside each other
    from bench_suites import SUITES

_BENCH_CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9+]*)[ \t]*\n(.*?)```", re.DOTALL)

# What each grading language answers to in a fence tag, and the shape its unfenced code takes.
_BENCH_LANGS = {
    "python": {"tags": {"python", "py", ""}, "bare": re.compile(r"\bdef\s+\w+")},
    "js": {"tags": {"js", "javascript", ""}, "bare": re.compile(r"\bfunction\s+\w+|=>")},
    "c": {"tags": {"c", ""}, "bare": re.compile(r"\b(?:int|long|unsigned)\s+\w+\s*\(")},
    "cpp": {"tags": {"cpp", "c++", "cxx", ""},
            "bare": re.compile(r"std::|\b(?:int|std::string)\s+\w+\s*\(")},
    "rust": {"tags": {"rust", "rs", ""}, "bare": re.compile(r"\bfn\s+\w+")},
    "csharp": {"tags": {"csharp", "cs", "c#", ""},
               "bare": re.compile(r"\bpublic\s+static\b|\bclass\s+\w+")},
    "php": {"tags": {"php", ""}, "bare": re.compile(r"<\?php|\bfunction\s+\w+")},
    "html": {"tags": {"html", ""}, "bare": re.compile(r"<[a-zA-Z][^>]*>")},
    "css": {"tags": {"css", ""}, "bare": re.compile(r"[.#:\w\s,>-]+\{[^}]*\}")},
    "go": {"tags": {"go", "golang", ""}, "bare": re.compile(r"\bfunc\s+\w+")},
    "sql": {"tags": {"sql", "sqlite", ""}, "bare": re.compile(r"(?i)\bselect\b")},
    "bash": {"tags": {"bash", "sh", "shell", ""},
             "bare": re.compile(r"^#!|\b(?:awk|sed|sort|uniq|while\s+read)\b")},
}

# Toolchains may be user-local installs the service's PATH has never heard of.
_BENCH_TOOL_DIRS = ("~/.cargo/bin", "~/.dotnet", "~/.local/php", "~/.local/bin",
                    "~/.local/go/bin")


def _bench_tool(name: str):
    import shutil as _sh
    found = _sh.which(name)
    if found:
        return found
    for d in _BENCH_TOOL_DIRS:
        cand = os.path.expanduser(f"{d}/{name}")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _compiler_version(binary: str) -> str:
    """One line of version output, however the toolchain spells the flag."""
    for args in ([binary, "--version"], [binary, "version"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
            if out.strip():
                return out.strip().splitlines()[0][:120]
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def _build_context(comp_cmd: list, src: str) -> dict:
    """What a compile failure needs to be judged: the exact command, the toolchain version,
    and the harness the grader wrapped around the model's code — so a reader can tell a
    model error from a harness error, and knows the model never saw the harness."""
    return {"cmd": " ".join(str(a) for a in comp_cmd),
            "compiler": _compiler_version(str(comp_cmd[0])),
            "harness": src[:2000]}


def _bench_lit(v, lang: str) -> str:
    """One argument as a source literal. JSON string escaping is valid in C, C++, Rust and C#
    for the ASCII cases these tasks use, which is exactly why the cases stay ASCII."""
    if isinstance(v, bool):
        return {"c": "1" if v else "0"}.get(lang, "true" if v else "false")
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v)




# One line per task saying what it actually tests, shown in the report's method section and
# beside per-task results. The id alone ("mid_floor") tells a reader nothing; the trap the
# task was built around is the whole reason it is in the suite.


# Hello-world programs per language, run once through the REAL grader to prove the toolchain
# works end to end. A binary on PATH is not a working toolchain — a rustc with no linker
# compiles nothing, and "available" would have zeroed every rust task on that box instead of
# skipping them, which is exactly what the portability contract forbids.
_BENCH_TOOLCHAIN_PROBES = {
    "js": ("function probe() { return 7; }", "probe"),
    "c": ("int probe(void) { return 7; }", "probe"),
    "cpp": ("int probe(const std::string& s) { return (int)s.size() + 6; }", "probe"),
    "rust": ("fn probe() -> i64 { 7 }", "probe"),
    "csharp": ("public static class Sol { public static int Probe() { return 7; } }",
               "Sol.Probe"),
    "php": ("function probe(): int { return 7; }", "probe"),
    "go": ("func Probe() int { return 7 }", "Probe"),
}
_BENCH_TOOLCHAIN_OK: dict = {}


def _bench_lang_available(lang: str) -> bool:
    """Whether this machine can grade a language — verified, not inferred. Portability
    contract: a task whose tooling is absent or broken is SKIPPED — dropped from the run and
    recorded as such — never scored as a zero, because a zero would punish the model for the
    box. The probe result is cached for the life of the process."""
    if lang in ("python", "html", "css", "sql", None, ""):
        return True                    # stdlib-only grading; nothing to probe
    tool = {"js": "node", "c": "gcc", "cpp": "g++", "rust": "rustc", "csharp": "dotnet",
            "php": "php", "bash": "bash", "go": "go"}.get(lang)
    if tool is None or _bench_tool(tool) is None:
        return False
    if lang == "csharp" and not os.path.isdir(os.path.expanduser("~/.cache/ai_proxy_cs")):
        return False
    if lang not in _BENCH_TOOLCHAIN_OK:
        if lang == "bash":
            res = _bench_grade_bash("echo 7", [{"stdin": "", "expect": "7"}], 20.0)
        else:
            code, entry = _BENCH_TOOLCHAIN_PROBES[lang]
            args_case = [{"args": ["x"], "expect": 7}] if lang == "cpp" \
                else [{"args": [], "expect": 7}]
            res = _bench_grade_sync(code, entry, args_case, 30.0, lang)
        _BENCH_TOOLCHAIN_OK[lang] = (res.get("passed") == res.get("total"))
    return _BENCH_TOOLCHAIN_OK[lang]


def _bench_suite_tasks(suite_name: str) -> tuple:
    """The tasks this machine can actually grade, and the ones it cannot, by language."""
    tasks = SUITES.get(suite_name) or []
    runnable, skipped = [], {}
    for t in tasks:
        lang = t.get("lang") or "python"
        if _bench_lang_available(lang):
            runnable.append(t)
        else:
            skipped.setdefault(lang, []).append(t["id"])
    return runnable, skipped


# Attributes whose VALUES HTML itself treats as case-insensitive (enumerated keywords):
# method="POST" submits identically to method="post", and failing a model over that casing
# is grader pedantry, not a wrong answer. href stays strict — paths are case-sensitive.
_HTML_CI_ATTRS = {"method", "type", "enctype", "autocomplete", "scope", "aria-current",
                  "dir", "inputmode"}


def _html_attr_eq(name: str, got, want) -> bool:
    if got is None or want is None:
        return got == want
    if name.lower() in _HTML_CI_ATTRS:
        return str(got).lower() == str(want).lower()
    return got == want


class _BenchHTML(HTMLParser):
    """Just enough DOM to check structure: tags, attributes, text, parentage."""

    VOID = {"input", "br", "img", "meta", "hr", "link", "source", "wbr", "col", "area", "base"}

    def __init__(self):
        super().__init__()
        self.root = {"tag": "#root", "attrs": {}, "children": [], "text": []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag.lower(), "attrs": {k.lower(): ("" if v is None else v)
                                              for k, v in attrs},
                "children": [], "text": []}
        self.stack[-1]["children"].append(node)
        if tag.lower() not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self.VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i]["tag"] == tag.lower():
                del self.stack[i:]
                break

    def handle_data(self, data):
        if data.strip():
            self.stack[-1]["text"].append(data)


def _bench_html_select(root: dict, selector: str) -> list:
    """A deliberately small selector: tag, #id, .class, [attr] and [attr=value], compounded,
    with descendant combination by whitespace. Enough to check structure; nothing more."""
    def parse_simple(tok):
        m = re.match(r"^([a-zA-Z][\w-]*|\*)?(#[\w-]+)?((?:\.[\w-]+)*)"
                     r"((?:\[[^\]]+\])*)$", tok)
        if not m:
            return None
        tag = (m.group(1) or "*").lower()
        want_id = m.group(2)[1:] if m.group(2) else None
        classes = [c for c in (m.group(3) or "").split(".") if c]
        attrs = []
        for am in re.finditer(r"\[([^\]=]+)(?:=([^\]]*))?\]", m.group(4) or ""):
            attrs.append((am.group(1).strip().lower(),
                          None if am.group(2) is None else am.group(2).strip("'\"")))
        return tag, want_id, classes, attrs

    def matches(node, simple):
        tag, want_id, classes, attrs = simple
        if tag != "*" and node["tag"] != tag:
            return False
        if want_id and node["attrs"].get("id") != want_id:
            return False
        node_classes = (node["attrs"].get("class") or "").split()
        if any(c not in node_classes for c in classes):
            return False
        for name, val in attrs:
            if name not in node["attrs"]:
                return False
            if val is not None and not _html_attr_eq(name, node["attrs"][name], val):
                return False
        return True

    def walk(node):
        for ch in node["children"]:
            yield ch
            yield from walk(ch)

    parts = [parse_simple(t) for t in selector.split()]
    if any(p is None for p in parts):
        return []
    pools = [root]
    for part in parts:
        nxt = []
        for scope in pools:
            for n in walk(scope):
                if matches(n, part) and n not in nxt:
                    nxt.append(n)
        pools = nxt
    return pools


def _bench_html_text(node: dict) -> str:
    out = list(node["text"])

    def rec(n):
        for ch in n["children"]:
            out.extend(ch["text"])
            rec(ch)
    rec(node)
    return " ".join(" ".join(out).split())


def _bench_check_html(code: str, cases: list) -> dict:
    """Structural grading: the fragment is parsed and each case asserts one fact about it.
    This checks the markup's structure — bindings, attributes, counts, text — not how a
    browser would paint it, and the report's method section says exactly that."""
    parser = _BenchHTML()
    try:
        parser.feed(code or "")
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"unparseable HTML: {e}"}
    root = parser.root
    results = []
    for c in cases:
        op = c.get("op")
        ok, got = False, None
        try:
            if op == "count":
                got = len(_bench_html_select(root, c["sel"]))
                ok = got == c["expect"]
            elif op == "attr":
                m = _bench_html_select(root, c["sel"])
                got = m[0]["attrs"].get(c["name"].lower()) if m else None
                ok = _html_attr_eq(c["name"], got, c["expect"])
            elif op == "text":
                m = _bench_html_select(root, c["sel"])
                got = _bench_html_text(m[0]) if m else None
                ok = got == c["expect"]
            elif op == "labels_bound":
                ids = {n["attrs"].get("id") for n in _bench_html_select(root, "input")
                       if n["attrs"].get("id")}
                got = sum(1 for lb in _bench_html_select(root, "label")
                          if lb["attrs"].get("for") in ids)
                ok = got == c["expect"]
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        r = {"ok": bool(ok)}
        if not ok:
            r["got"] = got
        results.append(r)
    return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
            "cases": results}


def _bench_check_css(code: str, cases: list) -> dict:
    """Structural CSS grading: rules are parsed (one level of @media supported), and each case
    asserts that a selector declares a property with a value — in the right context. The last
    matching declaration wins, which is the cascade's own rule for equal specificity."""
    text = re.sub(r"/\*.*?\*/", "", code or "", flags=re.S)
    rules = []          # (media_or_None, selector, prop, value) in source order

    def parse_block(body, media):
        for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", body):
            sel = " ".join(m.group(1).split()).lower()
            for decl in m.group(2).split(";"):
                if ":" in decl:
                    prop, _, val = decl.partition(":")
                    rules.append((media, sel, prop.strip().lower(),
                                  " ".join(val.split()).lower()))

    pos = 0
    while pos < len(text):
        at = text.find("@media", pos)
        if at < 0:
            parse_block(text[pos:], None)
            break
        parse_block(text[pos:at], None)
        brace = text.find("{", at)
        if brace < 0:
            break
        cond = " ".join(text[at + 6:brace].split()).lower()
        depth, i = 1, brace + 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        parse_block(text[brace + 1:i - 1], cond)
        pos = i

    results = []
    for c in cases:
        sel = " ".join(c["sel"].split()).lower()
        want_media = (c.get("media") or "").lower() or None
        got = None
        for media, rsel, prop, val in rules:
            sels = [x.strip() for x in rsel.split(",")]
            media_ok = (media is None) if want_media is None else \
                (media is not None and want_media in media)
            if media_ok and sel in sels and prop == c["prop"].lower():
                got = val
        want = " ".join(str(c["expect"]).split()).lower()
        ok = got == want
        r = {"ok": ok}
        if not ok:
            r["got"] = got
        results.append(r)
    return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
            "cases": results}


def _bench_extract_code(text: str, lang: str = "python") -> str:
    """Pull the task's language out of a model response.

    A polyglot answer often carries more than one fence — a Python usage example beside the
    JavaScript being graded — so blocks tagged with the task's language win outright, untagged
    blocks are the fallback, and a block tagged as some *other* language is never graded: a
    Python function handed to node fails as a syntax error and reads as a broken
    implementation rather than a mis-extraction.
    """
    cfg = _BENCH_LANGS.get(lang) or _BENCH_LANGS["python"]
    blocks = _BENCH_CODE_BLOCK_RE.findall(text or "")
    tagged = [b for t, b in blocks if t.lower() in cfg["tags"] and t]
    if tagged:
        return max(tagged, key=len)
    untagged = [b for t, b in blocks if not t]
    if untagged:
        return max(untagged, key=len)
    # An OPENED fence that never closes is what a response cut off at the token limit looks
    # like. Grading everything after the fence keeps the diagnosis honest: the code fails on
    # its real truncation point, not on the fence line itself — 67 rows once read
    # "SyntaxError: invalid syntax (line 1)" when the model had written a perfect opening.
    if not blocks:
        m = re.search(r"```([A-Za-z0-9+]*)[ \t]*\n(.*)\Z", text or "", re.DOTALL)
        if m:
            tag = m.group(1).lower()
            if tag in cfg["tags"]:
                return m.group(2)
            return ""      # a fence for some other language: never grade it as this one
    if not blocks and cfg["bare"].search(text or ""):
        return text
    return ""


# Runner executed in the child process: import the model's code, call the entry point on each
# case, and report what came back. Kept as a string so it can be fed to `python -c` without
# needing a file on disk that the parent has to clean up.
_BENCH_GRADER_SRC = r'''
import json, sys
payload = json.loads(sys.stdin.read())
ns = {}
out = []
try:
    exec(compile(payload["code"], "<model>", "exec"), ns)
except Exception as e:
    print(json.dumps({"fatal": "%s: %s" % (type(e).__name__, e)}))
    sys.exit(0)
fn = ns.get(payload["entry"])
if not callable(fn):
    print(json.dumps({"fatal": "no callable named %r" % payload["entry"]}))
    sys.exit(0)
for case in payload["cases"]:
    try:
        got = fn(*case["args"])
        # Tuples and lists are interchangeable for grading: JSON can't tell them apart, and a
        # model returning [("a",1)] vs [["a",1]] is not a correctness difference.
        def norm(v):
            if isinstance(v, tuple):
                return [norm(x) for x in v]
            if isinstance(v, list):
                return [norm(x) for x in v]
            return v
        out.append({"got": norm(got), "ok": norm(got) == norm(case["expect"])})
    except Exception as e:
        out.append({"got": None, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})
print(json.dumps({"results": out}))
'''


# Same protocol as the Python grader: payload on stdin, one JSON line out. eval twice in the
# same scope — the model's declarations, then the entry name — because direct eval scopes
# function declarations to the enclosing call, which is exactly what makes them reachable.
_BENCH_JS_GRADER_SRC = (
    'const ch=[];process.stdin.on("data",d=>ch.push(d));process.stdin.on("end",()=>{'
    'const p=JSON.parse(ch.join(""));let out={results:[]};try{eval(p.code);'
    'const fn=eval(p.entry);if(typeof fn!=="function")throw new Error("entry not defined: "+p.entry);'
    'for(const c of p.cases){const r={};try{const got=fn(...c.args);'
    'r.ok=JSON.stringify(got)===JSON.stringify(c.expect);'
    'if(!r.ok)r.got=got===undefined?null:got;}catch(e){r.ok=false;r.err=String(e).slice(0,150)}'
    'out.results.push(r)}}catch(e){out={fatal:String(e).slice(0,250)}}'
    'console.log(JSON.stringify(out))});')


def _bench_grade_c(code: str, entry: str, cases: list, timeout_s: float) -> dict:
    """Compile-and-run grading for C, integer signatures only.

    JSON-comparing arbitrary C values would need a serialiser the model would have to fight,
    so C tasks keep an int contract and the harness is a generated main() that prints one
    result per line. -fwrapv pins signed overflow to wraparound, so a naive implementation
    fails deterministically instead of by undefined behaviour.
    """
    import shutil as _sh
    gcc = _sh.which("gcc")
    if not gcc:
        return {"passed": 0, "total": len(cases), "error": "gcc is not installed on this host"}
    calls = "".join(
        '    printf("%d\\n", {e}({args}));\n'.replace("{e}", entry)
        # _bench_lit, not int(): count_words takes a const char*, and JSON string escaping
        # is valid C for the ASCII these cases use. Caught by the on-box battery — the dev
        # machine has no gcc, so the local tests skip C and could not see it.
        .replace("{args}", ", ".join(_bench_lit(a, "c") for a in c["args"]))
        for c in cases)
    src = ("#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
           "#include <limits.h>\n\n" + code +
           "\n\nint main(void) {\n" + calls + "    return 0;\n}\n")
    workdir = tempfile.mkdtemp(prefix="bench_c_")
    try:
        cpath = os.path.join(workdir, "task.c")
        epath = os.path.join(workdir, "task.exe")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(src)
        env = {"PATH": os.environ.get("PATH", ""),
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        comp_cmd = [gcc, "-O0", "-fwrapv", cpath, "-o", epath]
        comp = subprocess.run(comp_cmd,
                              capture_output=True, text=True, timeout=20, env=env, cwd=workdir)
        if comp.returncode != 0:
            return {"passed": 0, "total": len(cases),
                    "error": ("compile error: " + (comp.stderr or "")[-600:]).strip(),
                    "build": _build_context(comp_cmd, src)}
        run = subprocess.run([epath], capture_output=True, text=True,
                             timeout=timeout_s, env=env, cwd=workdir)
        got = (run.stdout or "").strip().splitlines()
        results = []
        for i, c in enumerate(cases):
            ok = i < len(got) and got[i].strip() == str(int(c["expect"]))
            r = {"ok": ok}
            if not ok:
                r["got"] = got[i].strip() if i < len(got) else None
            results.append(r)
        return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
                "cases": results}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(cases), "error": f"timeout after {timeout_s}s"}
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            import shutil as _sh2
            _sh2.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def _bench_grade_compiled(lang: str, code: str, entry: str, cases: list,
                          timeout_s: float) -> dict:
    """C++, Rust and C# share the C grader's shape: generate a main that prints one result per
    line, compile, run, compare lines. Only the syntax of the harness differs."""
    workdir = tempfile.mkdtemp(prefix=f"bench_{lang}_")
    env = {"PATH": os.environ.get("PATH", ""),
           "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
           "HOME": os.path.expanduser("~"),
           "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"}
    try:
        if lang == "cpp":
            gxx = _bench_tool("g++")
            if not gxx:
                return {"passed": 0, "total": len(cases), "error": "g++ is not installed"}
            calls = "".join(
                f'    std::cout << {entry}({", ".join(_bench_lit(a, "cpp") for a in c["args"])})'
                ' << "\\n";\n' for c in cases)
            src = ("#include <iostream>\n#include <string>\n#include <vector>\n"
                   "#include <algorithm>\n#include <cstdint>\n#include <climits>\n\n"
                   + code + "\n\nint main() {\n" + calls + "    return 0;\n}\n")
            spath = os.path.join(workdir, "task.cpp")
            epath = os.path.join(workdir, "task.exe")
            comp_cmd = [gxx, "-std=c++17", "-O0", spath, "-o", epath]
            run_cmd = [epath]
        elif lang == "rust":
            rustc = _bench_tool("rustc")
            if not rustc:
                return {"passed": 0, "total": len(cases), "error": "rustc is not installed"}
            calls = "".join(
                f'    println!("{{}}", {entry}({", ".join(_bench_lit(a, "rust") for a in c["args"])}));\n'
                for c in cases)
            src = ("#![allow(dead_code)]\n" + code
                   + "\n\nfn main() {\n" + calls + "}\n")
            spath = os.path.join(workdir, "task.rs")
            epath = os.path.join(workdir, "task.exe")
            comp_cmd = [rustc, "--edition", "2021", "-o", epath, spath]
            run_cmd = [epath]
        elif lang == "csharp":
            dotnet = _bench_tool("dotnet")
            runner = os.path.expanduser("~/.cache/ai_proxy_cs")
            if not dotnet or not os.path.isdir(runner):
                return {"passed": 0, "total": len(cases),
                        "error": "the dotnet runner is not installed"}
            calls = "".join(
                f'        System.Console.WriteLine({entry}('
                f'{", ".join(_bench_lit(a, "csharp") for a in c["args"])}));\n'
                for c in cases)
            src = (code + "\n\npublic static class Program {\n"
                   "    public static void Main() {\n" + calls + "    }\n}\n")
            spath = os.path.join(runner, "Program.cs")
            comp_cmd = [dotnet, "build", "-v", "q", "--nologo", runner]
            dll = os.path.join(runner, "bin", "Debug", "net8.0", "csrunner.dll")
            run_cmd = [dotnet, dll]
        elif lang == "go":
            gobin = _bench_tool("go")
            if not gobin:
                return {"passed": 0, "total": len(cases), "error": "go is not installed"}
            # The harness aliases fmt so a model that imports fmt itself doesn't collide,
            # and a model that pasted its own `package main` line loses it rather than
            # failing on a duplicate declaration.
            body = re.sub(r"^\s*package\s+\w+\s*$", "", code, count=1, flags=re.M)
            calls = "".join(
                f'    __fmt.Println({entry}({", ".join(_bench_lit(a, "go") for a in c["args"])}))\n'
                for c in cases)
            src = ('package main\n\nimport __fmt "fmt"\n\n' + body
                   + "\n\nfunc main() {\n" + calls + "}\n")
            spath = os.path.join(workdir, "task.go")
            epath = os.path.join(workdir, "task.exe")
            env["GOCACHE"] = os.path.join(workdir, "gocache")
            env["GOPATH"] = os.path.join(workdir, "gopath")
            comp_cmd = [gobin, "build", "-o", epath, spath]
            run_cmd = [epath]
        else:
            return {"passed": 0, "total": len(cases), "error": f"no compiler for {lang!r}"}

        with open(spath, "w", encoding="utf-8") as f:
            f.write(src)
        comp = subprocess.run(comp_cmd, capture_output=True, text=True, timeout=90,
                              env=env, cwd=workdir)
        if comp.returncode != 0:
            msg = (comp.stderr or comp.stdout or "")[-600:]
            return {"passed": 0, "total": len(cases),
                    "error": ("compile error: " + msg).strip(),
                    "build": _build_context(comp_cmd, src)}
        run = subprocess.run(run_cmd, capture_output=True, text=True,
                             timeout=timeout_s, env=env, cwd=workdir)
        # splitlines on the raw output, not on .strip() of it: stripping ate trailing empty
        # lines, so a function correctly returning "" lost its output line and every case
        # after an empty result shifted — rust_kv_get's empty-string sentinel found this.
        got = (run.stdout or "").splitlines()
        results = []
        for i, c in enumerate(cases):
            want = str(c["expect"]) if not isinstance(c["expect"], bool) else \
                ("true" if c["expect"] else "false")
            ok = i < len(got) and got[i].rstrip("\r") == want
            r = {"ok": ok}
            if not ok:
                r["got"] = got[i].rstrip("\r") if i < len(got) else None
            results.append(r)
        return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
                "cases": results}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(cases), "error": f"timeout after {timeout_s}s"}
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            import shutil as _sh2
            _sh2.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def _bench_grade_php(code: str, entry: str, cases: list, timeout_s: float) -> dict:
    """PHP grading over a JSON line protocol. Cases travel base64d so no quoting rules of any
    of the three languages involved can interfere; results compare as parsed JSON, which is
    what lets a case distinguish a missing key from an explicit null."""
    php = _bench_tool("php")
    if not php:
        return {"passed": 0, "total": len(cases), "error": "php is not installed"}
    import base64
    body = code.strip()
    if body.startswith("<?php"):
        body = body[5:]
    b64 = base64.b64encode(json.dumps([c["args"] for c in cases]).encode()).decode()
    src = ("<?php\n" + body + "\n"
           '$__cases = json_decode(base64_decode("' + b64 + '"), true);\n'
           'foreach ($__cases as $__c) { echo json_encode(' + entry + '(...$__c)), "\n"; }\n')
    workdir = tempfile.mkdtemp(prefix="bench_php_")
    try:
        spath = os.path.join(workdir, "task.php")
        with open(spath, "w", encoding="utf-8") as f:
            f.write(src)
        env = {"PATH": os.environ.get("PATH", ""),
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        run = subprocess.run([php, spath], capture_output=True, text=True,
                             timeout=timeout_s, env=env, cwd=workdir)
        got = (run.stdout or "").strip().splitlines()
        if not got and run.stderr:
            return {"passed": 0, "total": len(cases),
                    "error": (run.stderr or "")[-300:].strip()}
        results = []
        for i, c in enumerate(cases):
            ok = False
            got_v = None
            if i < len(got):
                try:
                    got_v = json.loads(got[i])
                    ok = got_v == c["expect"]
                except (json.JSONDecodeError, TypeError):
                    got_v = got[i][:120]
            r = {"ok": ok}
            if not ok:
                r["got"] = got_v
            results.append(r)
        return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
                "cases": results}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(cases), "error": f"timeout after {timeout_s}s"}
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            import shutil as _sh2
            _sh2.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def _bench_grade_sql(code: str, cases: list) -> dict:
    """SQL grading on the stdlib sqlite3 — the one language that needs no toolchain at all.

    Each case carries its own `setup` script (schema + data) and the expected rows. A fresh
    in-memory database per case means no state leaks between cases, and rows compare as
    lists so column order and row order are both part of the contract."""
    import sqlite3 as _sq
    sql = (code or "").strip().rstrip(";")
    results = []
    for c in cases:
        ok, got = False, None
        try:
            conn = _sq.connect(":memory:")
            conn.executescript(c.get("setup") or "")
            got = [list(r) for r in conn.execute(sql).fetchall()]
            ok = got == c.get("expect")
            conn.close()
        except _sq.Error as e:
            got = f"{type(e).__name__}: {e}"
        r = {"ok": bool(ok)}
        if not ok:
            r["got"] = got
        results.append(r)
    return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
            "cases": results}


def _bench_grade_bash(code: str, cases: list, timeout_s: float) -> dict:
    """Bash grading: the script gets each case's stdin and must print the expected stdout.
    Pure text-in text-out — no filesystem contract, so nothing to sandbox beyond the same
    subprocess + timeout every other language gets."""
    bash = _bench_tool("bash")
    if not bash:
        return {"passed": 0, "total": len(cases), "error": "bash is not installed"}
    workdir = tempfile.mkdtemp(prefix="bench_bash_")
    try:
        spath = os.path.join(workdir, "task.sh")
        with open(spath, "w", encoding="utf-8", newline="\n") as f:
            f.write(code or "")
        env = {"PATH": os.environ.get("PATH", ""),
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "LC_ALL": "C"}          # sorting must not depend on the box's locale
        results = []
        for c in cases:
            ok, got = False, None
            try:
                run = subprocess.run([bash, spath], input=c.get("stdin") or "",
                                     capture_output=True, text=True,
                                     timeout=timeout_s, env=env, cwd=workdir)
                got = (run.stdout or "").rstrip("\n")
                if run.returncode != 0 and not got:
                    got = (run.stderr or f"exit {run.returncode}")[-160:]
                ok = got == str(c.get("expect")).rstrip("\n")
            except subprocess.TimeoutExpired:
                got = f"timeout after {timeout_s}s"
            r = {"ok": bool(ok)}
            if not ok:
                r["got"] = got
            results.append(r)
        return {"passed": sum(1 for r in results if r["ok"]), "total": len(cases),
                "cases": results}
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            import shutil as _sh2
            _sh2.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def _bench_grade_sync(code: str, entry: str, cases: list, timeout_s: float,
                      lang: str = "python") -> dict:
    """Run one task's code against its cases in a subprocess. Blocking — call via to_thread.

    HTML and CSS are the exception: nothing executes, so they grade in-process — the answer is
    parsed and checked against required structure. That is honest static grading, not a claim
    about how the page renders.
    """
    if lang == "c":
        return _bench_grade_c(code, entry, cases, timeout_s)
    if lang in ("cpp", "rust", "csharp", "go"):
        return _bench_grade_compiled(lang, code, entry, cases, timeout_s)
    if lang == "php":
        return _bench_grade_php(code, entry, cases, timeout_s)
    if lang == "html":
        return _bench_check_html(code, cases)
    if lang == "css":
        return _bench_check_css(code, cases)
    if lang == "sql":
        return _bench_grade_sql(code, cases)
    if lang == "bash":
        return _bench_grade_bash(code, cases, timeout_s)
    payload = json.dumps({"code": code, "entry": entry, "cases": cases})
    env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    if lang == "js":
        import shutil as _sh
        node = _sh.which("node")
        if not node:
            return {"passed": 0, "total": len(cases),
                    "error": "node is not installed on this host"}
        cmd = [node, "-e", _BENCH_JS_GRADER_SRC]
    else:
        cmd = [sys.executable, "-I", "-c", _BENCH_GRADER_SRC]
    try:
        proc = subprocess.run(
            cmd,
            input=payload, capture_output=True, text=True,
            timeout=timeout_s, cwd=tempfile.gettempdir(), env=env,
        )
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(cases), "error": f"timeout after {timeout_s}s"}
    except Exception as e:
        return {"passed": 0, "total": len(cases), "error": f"{type(e).__name__}: {e}"}
    raw = (proc.stdout or "").strip().splitlines()
    if not raw:
        return {"passed": 0, "total": len(cases),
                "error": (proc.stderr or "no output from grader")[:300]}
    try:
        parsed = json.loads(raw[-1])
    except (json.JSONDecodeError, TypeError):
        return {"passed": 0, "total": len(cases), "error": "grader output was not JSON"}
    if parsed.get("fatal"):
        return {"passed": 0, "total": len(cases), "error": parsed["fatal"]}
    results = parsed.get("results") or []
    passed = sum(1 for r in results if r.get("ok"))
    return {"passed": passed, "total": len(cases), "cases": results}
