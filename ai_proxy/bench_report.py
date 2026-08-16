"""The whitepaper report: page shell, charts, scorecards, failure examples.

Pure rendering — rows and run dicts in, self-contained HTML out. Inline CSS and SVG, no
external requests, so a saved copy renders identically a month later. Imports only the data
and grader modules; never the app.
"""
import datetime
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

try:
    from .bench_suites import SUITES, TASK_DESC, TASK_NOTES
    from .bench_graders import _bench_extract_code, toolchain_versions
except ImportError:          # flat-script launch: modules sit beside each other
    from bench_suites import SUITES, TASK_DESC, TASK_NOTES
    from bench_graders import _bench_extract_code, toolchain_versions

def _fmt_n(v, digits=0, suffix=""):
    return "—" if v is None else f"{v:,.{digits}f}{suffix}"


def _host_hw_facts() -> dict:
    """The static facts about this machine: CPU, cores, kernel, OS. Gathered at snapshot
    time for new runs and at render time as a fallback for old ones — a CPU model does not
    change between the two, which is what makes the fallback honest."""
    import platform as _pf
    facts: dict = {"cpu_cores": os.cpu_count(), "kernel": _pf.release(),
                   "os": f"{_pf.system()} {_pf.machine()}", "python": _pf.python_version()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware", "cpu part")):
                    facts["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if not facts.get("cpu_model") or facts["cpu_model"].startswith("0x"):
        # ARM /proc/cpuinfo gives a hex part code ("0xd87"), which names nothing. lscpu
        # resolves it to the actual core ("Cortex-X925"); the hex stays as the fallback.
        try:
            out = subprocess.run(["lscpu"], capture_output=True, text=True,
                                 timeout=5).stdout
            m = re.search(r"^Model name:\s*(.+)$", out, re.M)
            if m and m.group(1).strip() != "-":
                facts["cpu_model"] = m.group(1).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)', os_release, re.M)
        if m:
            facts["os"] = f"{m.group(1)} ({_pf.machine()})"
    except OSError:
        pass
    return facts


def _bench_model_display(name: str) -> str:
    """A model identifier short enough for a table cell.

    llama.cpp names a model by its full path, and a split GGUF adds a shard suffix on top. The
    raw string is wider than the table it sits in, every row repeats it twice, and because it
    contains no spaces nothing can wrap it — one cell then forces the whole table sideways.
    """
    s = str(name or "").strip()
    if not s:
        return "?"
    if "/" in s or "\\" in s:
        s = re.split(r"[\\/]", s)[-1]
    s = re.sub(r"\.gguf$", "", s, flags=re.I)
    # "-00001-of-00003": one shard of a split file, never part of the model's identity.
    s = re.sub(r"-\d{4,5}-of-\d{4,5}$", "", s)
    return s or str(name)


def _bench_model_identity(name: str) -> str:
    """The same weights under whatever each backend calls them.

    Ollama names the default tag `qwen3-coder-next:latest`; the vLLM container serving the same
    checkpoint calls it `qwen3-coder-next`. Compared as written, an engine comparison reads as
    two different models on two different engines and the difference cannot be attributed to
    either. Only the `:latest` tag is dropped — it means "the default", where `:30b` and
    `:tuned` are genuinely different weights and must stay apart.
    """
    s = _bench_model_display(name)
    return s[:-7] if s.endswith(":latest") else s


def _bench_label_display(label: str) -> str:
    """Cell labels lead with the model, so they inherit the same problem. Shortened at render
    time rather than at write time so runs recorded before this existed also read properly."""
    parts = str(label or "").split(" · ")
    if parts:
        parts[0] = _bench_model_display(parts[0])
    return " · ".join(parts)


def _bench_fmt(v, digits=1, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}{suffix}"
    return f"{v}{suffix}"


_D3_CACHE: dict = {}


def _d3_source() -> str:
    """The vendored d3 bundle, inlined so the report stays a single offline file. An absent
    vendor file degrades gracefully: highlighting and the animated re-rank are vanilla JS,
    and the d3-only extras simply do not build."""
    if "src" not in _D3_CACHE:
        try:
            _D3_CACHE["src"] = (Path(__file__).parent / "static"
                                / "d3.v7.min.js").read_text(encoding="utf-8")
        except OSError:
            _D3_CACHE["src"] = ""
    return _D3_CACHE["src"]


# The interactive layer. Screen-only enhancements over the server-rendered truth: the static
# SVGs and tables remain the print/PDF/no-JS version, and everything here degrades to them.
_REPORT_IX_JS = '\n(function () {\n  var root = document.documentElement;\n  root.classList.add("ix-on");\n  var dataEl = document.getElementById("report-data");\n  var DATA = null;\n  try { DATA = dataEl ? JSON.parse(dataEl.textContent) : null; } catch (e) {}\n  var byName = {};\n  if (DATA && DATA.rows) DATA.rows.forEach(function (r) { byName[r.name] = r; });\n\n  // ---- linked highlighting: hover a model anywhere, see it everywhere ----\n  // Marks are queried LIVE on every change: the weighted-standings slider rebuilds its rows,\n  // and a list captured at load kept pointing at orphaned elements — the standings went dark\n  // to highlighting after the first drag.\n  function setHL(m) {\n    var marks = document.querySelectorAll("[data-m]");\n    if (m == null) {\n      root.classList.remove("hl");\n      Array.prototype.forEach.call(marks, function (e) { e.classList.remove("lit"); });\n      return;\n    }\n    root.classList.add("hl");\n    Array.prototype.forEach.call(marks, function (e) {\n      e.classList.toggle("lit", e.getAttribute("data-m") === m);\n    });\n  }\n\n  // ---- shared tooltip ----\n  var tip = document.createElement("div");\n  tip.className = "ixtip";\n  tip.style.opacity = 0;\n  document.body.appendChild(tip);\n  function showTip(html, x, y) {\n    tip.innerHTML = html;\n    tip.style.left = Math.min(x + 14, window.innerWidth - 260) + "px";\n    tip.style.top = (y + 14) + "px";\n    tip.style.opacity = 1;\n  }\n  function hideTip() { tip.style.opacity = 0; }\n\n  function esc(t) {\n    var d = document.createElement("i");\n    d.textContent = t == null ? "" : String(t);\n    return d.innerHTML;\n  }\n  function blurb(r) {\n    var bits = [];\n    if (r.q != null) bits.push(Math.round(r.q) + "% correct");\n    if (r.d) bits.push(r.d.toFixed(1) + " tok/s");\n    if (r.g) bits.push(r.g.toFixed(1) + " GB");\n    if (r.t) bits.push("answers in " + (r.t / 1000).toFixed(1) + "s");\n    if (r.l) bits.push("loads in " + Math.round(r.l / 1000) + "s");\n    return "<b>" + esc(r.name) + "</b><br>" + bits.join(" \\u00b7 ");\n  }\n\n  // One hover handler for everything: highlight by model, and when the mark names its exact\n  // cell (data-name) or its model has dataset rows, show the results blurb too.\n  document.addEventListener("mouseover", function (ev) {\n    var t = ev.target.closest ? ev.target.closest("[data-m]") : null;\n    if (!t) return;\n    setHL(t.getAttribute("data-m"));\n    var nm = t.getAttribute("data-name");\n    var r = nm && byName[nm];\n    if (!r && DATA && DATA.rows) {\n      var mine = DATA.rows.filter(function (q) {\n        return q.m === t.getAttribute("data-m");\n      });\n      if (mine.length === 1) r = mine[0];\n    }\n    if (r) showTip(blurb(r), ev.clientX, ev.clientY);\n  });\n  document.addEventListener("mousemove", function (ev) {\n    if (tip.style.opacity == 1) {\n      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 260) + "px";\n      tip.style.top = (ev.clientY + 14) + "px";\n    }\n  });\n  document.addEventListener("mouseout", function (ev) {\n    var t = ev.target.closest ? ev.target.closest("[data-m]") : null;\n    if (t) { setHL(null); hideTip(); }\n  });\n\n  // ---- explorable scatter (d3): brush to zoom, double-click resets ----\n  if (window.d3 && DATA && DATA.rows && DATA.rows.length > 2) {\n    var host = document.getElementById("ix-scatter");\n    if (host) {\n      var pts = DATA.rows.filter(function (r) { return r.d && r.q != null; });\n      var W = 860, H = 460, mL = 54, mR = 24, mT = 16, mB = 44;\n      var svg = d3.select(host).append("svg")\n        .attr("viewBox", "0 0 " + W + " " + H).attr("width", "100%");\n      var xExt = d3.extent(pts, function (r) { return r.d; });\n      var yMin = Math.min(40, d3.min(pts, function (r) { return r.q; }));\n      var X0 = [xExt[0] * 0.8, xExt[1] * 1.3], Y0 = [yMin, 100.5];\n      var x = d3.scaleLog().domain(X0).range([mL, W - mR]);\n      var y = d3.scaleLinear().domain(Y0).range([H - mB, mT]);\n      var gGrid = svg.append("g"), gDots = svg.append("g");\n      svg.append("text").attr("x", (mL + W - mR) / 2).attr("y", H - 8)\n        .attr("class", "ct").attr("text-anchor", "middle")\n        .text("output tokens/sec (log) \\u2014 drag a box to zoom, double-click to reset");\n      svg.append("text").attr("x", mL).attr("y", mT - 4).attr("class", "ct")\n        .text("TASKS FULLY CORRECT \\u2191");\n      function redraw(animate) {\n        var gt = gGrid.selectAll("g.tick-y").data(y.ticks(6), String);\n        gt.exit().remove();\n        var ge = gt.enter().append("g").attr("class", "tick-y");\n        ge.append("line");\n        ge.append("text");\n        gGrid.selectAll("g.tick-y").each(function (v) {\n          var g = d3.select(this);\n          g.select("line").attr("x1", mL).attr("x2", W - mR)\n            .attr("y1", y(v)).attr("y2", y(v))\n            .attr("stroke", "currentColor").attr("stroke-opacity", 0.1);\n          g.select("text").attr("x", mL - 7).attr("y", y(v) + 4)\n            .attr("class", "ct").attr("text-anchor", "end").text(Math.round(v) + "%");\n        });\n        var sel = gDots.selectAll("circle").data(pts, function (r) { return r.name; });\n        sel = sel.enter().append("circle")\n          .attr("r", 4.5).attr("fill", "var(--accent)").attr("fill-opacity", 0.75)\n          .attr("data-m", function (r) { return r.m; })\n          .attr("data-name", function (r) { return r.name; })\n          .merge(sel);\n        (animate ? sel.transition().duration(350) : sel)\n          .attr("cx", function (r) { return x(r.d); })\n          .attr("cy", function (r) { return y(r.q); })\n          .attr("display", function (r) {\n            var dx = x(r.d), dy = y(r.q);\n            return (dx < mL || dx > W - mR || dy < mT || dy > H - mB) ? "none" : null;\n          });\n      }\n      var brush = d3.brush().extent([[mL, mT], [W - mR, H - mB]])\n        .on("end", function (ev) {\n          if (!ev.selection) return;\n          var s0 = ev.selection;\n          x.domain([x.invert(s0[0][0]), x.invert(s0[1][0])]);\n          y.domain([y.invert(s0[1][1]), y.invert(s0[0][1])]);\n          svg.select(".brush").call(brush.move, null);\n          redraw(true);\n        });\n      svg.append("g").attr("class", "brush").call(brush);\n      svg.on("dblclick", function () { x.domain(X0); y.domain(Y0); redraw(true); });\n      redraw(false);\n    }\n  }\n\n  // ---- per-task drill-down: click a missed task, see every configuration ----\n  Array.prototype.forEach.call(document.querySelectorAll("tr.taskrow"), function (tr) {\n    tr.addEventListener("click", function () {\n      var tid = tr.getAttribute("data-task");\n      var open = tr.nextElementSibling && tr.nextElementSibling.classList.contains("drill");\n      Array.prototype.forEach.call(document.querySelectorAll("tr.drill"),\n                                   function (d) { d.remove(); });\n      if (open || !DATA || !DATA.tasks || !DATA.tasks[tid]) return;\n      var info = DATA.tasks[tid];\n      var d = document.createElement("tr");\n      d.className = "drill";\n      var td = document.createElement("td");\n      td.colSpan = tr.children.length;\n      d.appendChild(td);\n      tr.parentNode.insertBefore(d, tr.nextSibling);\n      var W = 720, H = 30 + 12 * 5, mL = 12, mR = 44;\n      var html = ["<p class=\\"k\\">" + esc(tid) + " \\u2014 " + esc(info.desc || "") +\n                  " \\u00b7 share of runs fully correct, per configuration</p>"];\n      html.push("<svg viewBox=\\"0 0 " + W + " " + H + "\\" width=\\"100%\\">");\n      info.rates.forEach(function (pair, i) {\n        var cx = mL + (pair[1] || 0) * (W - mL - mR);\n        var cy = 16 + (i % 5) * 12;\n        var m = String(pair[0]).split(" \\u00b7 ")[0];\n        html.push("<circle cx=\\"" + cx.toFixed(1) + "\\" cy=\\"" + cy + "\\" r=\\"5\\" " +\n                  "fill=\\"var(--accent)\\" fill-opacity=\\"0.65\\" data-m=\\"" + esc(m) +\n                  "\\" data-name=\\"" + esc(pair[0]) + "\\"></circle>");\n      });\n      html.push("<text x=\\"" + mL + "\\" y=\\"" + (H - 4) + "\\" class=\\"ct\\">0%</text>");\n      html.push("<text x=\\"" + (W - mR) + "\\" y=\\"" + (H - 4) +\n                "\\" class=\\"ct\\">100%</text>");\n      html.push("</svg>");\n      td.innerHTML = html.join("");\n    });\n  });\n})();\n'


def _bench_report_row(run: dict) -> dict:
    """Flatten one run into the fields a comparison table needs."""
    res = run.get("results") or {}
    s = res.get("summary") or {}
    q = s.get("quality") or {}
    cfg = run.get("config") or {}
    return {
        "id": run.get("id"),
        "label": run.get("label") or run.get("model"),
        "model": run.get("model"),
        "served": ", ".join(s.get("served_models") or []) or None,
        "thinking": cfg.get("thinking"),
        "temperature": cfg.get("temperature"),
        "prompt_tokens": cfg.get("prompt_tokens"),
        # What the server was serving, as distinct from what the run sent it.
        "server_context": cfg.get("server_context") or (run.get("env") or {}).get("loaded_context"),
        # A cell that ran short of memory still produced numbers; they are just numbers about
        # memory pressure. Carried through to the report so that is visible rather than inferred.
        "memory_warning": (run.get("env") or {}).get("memory_warning"),
        # The cost of having the model, as distinct from the cost of using it. A model that
        # decodes quickly but takes seven minutes to load is a different proposition from one
        # ready in forty seconds, and the decode column cannot say so.
        "load_ms": (run.get("env") or {}).get("load_ms"),
        # The two halves of a cold start, kept apart: booting the server (container start
        # plus weight load) and the first request against it. A container backend spends
        # nearly all of it in the first half, an on-demand backend all of it in the second.
        "backend_start_ms": (run.get("env") or {}).get("backend_start_ms"),
        "warmup_request_ms": (run.get("env") or {}).get("warmup_request_ms"),
        "unload_ms": (run.get("env") or {}).get("unload_ms"),
        "resident_mb": (run.get("env") or {}).get("resident_mb"),
        "n_success": s.get("n_success"),
        "n_total": s.get("n_total"),
        "ttft_p50": (s.get("ttft_ms") or {}).get("p50"),
        "ttfc_p50": (s.get("ttfc_ms") or {}).get("p50"),
        "decode_p50": (s.get("decode_tps") or {}).get("p50"),
        "total_p50": (s.get("total_ms") or {}).get("p50"),
        "reasoning_tok_p50": (s.get("reasoning_tokens") or {}).get("p50"),
        "perfect_rate": q.get("perfect_rate"),
        "case_pass_rate": q.get("case_pass_rate"),
        "suite": cfg.get("suite"),
        # Mean output length: the report's thinking-vs-not chapter turned on this number as much
        # as on latency — ~18x more tokens generated for no quality gain.
        "mean_tokens": (s.get("completion_tokens") or {}).get("mean"),
        "cache": cfg.get("cache"),
        "concurrency": cfg.get("concurrency") or 1,
        "quant": (run.get("env") or {}).get("quant"),
        "size_mb": (run.get("env") or {}).get("size_mb"),
        "checkpoint": (run.get("env") or {}).get("checkpoint"),
        "prefix_caching": (run.get("env") or {}).get("prefix_caching"),
        "kv_cache_dtype": (run.get("env") or {}).get("kv_cache_dtype"),
        "warmup_ms": s.get("warmup_ms"),
        "warmup_ttft_ms": s.get("warmup_ttft_ms"),
        "tiers": q.get("tiers") or {},
        # Full distributions, used when there is a single configuration and a comparison table
        # would have nothing to compare against.
        "ttft": s.get("ttft_ms") or {},
        "decode": s.get("decode_tps") or {},
        "total": s.get("total_ms") or {},
    }



# Reports use the dashboard's palette rather than a document look of their own — they're read
# next to the UI they came from, so a different skin reads as a different tool. Colour lives in
# CSS variables with a light counterpart, because a dark page printed to PDF wastes a cartridge
# and reads badly on paper; print forces the light set.
# One definition per theme. The viewer's OS preference applies a theme via the media query;
# the toggle stamps data-theme on the root, which must win in both directions — so each block
# is emitted twice from the same constant and the two application paths cannot drift.
_REPORT_TOKENS_LIGHT = """
    --bg:#FFFFFF; --panel:#FFFFFF; --panel-2:#F5F5F2; --border:#E4E4E1;
    --ink:#121212; --ink-dim:#454B52; --ink-faint:#727272;
    --accent:#C4321F; --accent-deep:#8F2416; --good:#1F7A47; --warn:#8A6D1F; --bad:#B3261E;
    --ghost:#B9B9B9; --grid:#EBEBE8; --blue:#3E6FA8;
    color-scheme: light;
"""
_REPORT_TOKENS_DARK = """
    --bg:#121418; --panel:#15181D; --panel-2:#1B1F26; --border:#2A2F37;
    --ink:#E8EAED; --ink-dim:#C4CAD2; --ink-faint:#98A0AA;
    --accent:#E0604E; --accent-deep:#C4321F; --good:#5FBF8A; --warn:#D9C37A; --bad:#F07178;
    --ghost:#565E6A; --grid:#242931; --blue:#7AA7DA;
    color-scheme: dark;
"""

_REPORT_CSS = """
  :root {
    --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
    --sans:"Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;
    --serif:Georgia,"Times New Roman",serif;
""" + _REPORT_TOKENS_LIGHT + """  }
  @media (prefers-color-scheme: dark) { :root {""" + _REPORT_TOKENS_DARK + """  } }
  :root[data-theme="dark"] {""" + _REPORT_TOKENS_DARK + """  }
  :root[data-theme="light"] {""" + _REPORT_TOKENS_LIGHT + """  }
  #themeflip { position:fixed; top:14px; right:14px; z-index:5; font-family:var(--sans);
    font-size:12px; padding:5px 13px; border:1px solid var(--border); border-radius:999px;
    background:var(--panel); color:var(--ink-dim); cursor:pointer; }
  #themeflip:hover { color:var(--ink); border-color:var(--ink-faint); }
  #themeflip:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }
  @media print { #themeflip { display:none; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--serif);
         font-size:16.5px; line-height:1.62; padding:clamp(18px,4vw,44px) clamp(14px,4vw,34px);
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1040px; margin:0 auto; }
  .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.18em; text-transform:uppercase;
             color:var(--accent); margin:0 0 10px; }
  h1 { font-size:clamp(22px,3.4vw,32px); line-height:1.1; margin:0 0 6px; letter-spacing:-.02em;
       font-weight:680; }
  .sub { color:var(--ink-faint); font-size:13px; margin:0 0 20px; }
  h2 { font-size:20px; font-family:var(--serif); color:var(--ink); margin:38px 0 12px;
       padding-top:12px; border-top:1px solid var(--ink); font-weight:700;
       letter-spacing:-.01em; }
  .meta { display:flex; flex-wrap:wrap; gap:6px 24px; font-family:var(--mono); font-size:12px;
          color:var(--ink-faint); border-top:1px solid var(--border);
          border-bottom:1px solid var(--border); padding:10px 0; margin-bottom:6px; }
  .meta b { color:var(--ink-dim); font-weight:500; }
  .note { color:var(--ink-faint); font-size:12.5px; margin:0 0 12px; max-width:76ch; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;
           margin:16px 0 4px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
  .card .k { font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
             color:var(--ink-faint); margin:0 0 8px; }
  .card .v { font-family:var(--mono); font-size:24px; font-weight:600; letter-spacing:-.02em;
             line-height:1.05; color:var(--ink); }
  .card .v small { font-size:13px; color:var(--ink-faint); font-weight:400; }
  .card .d { font-size:12px; color:var(--ink-faint); margin:7px 0 0; }
  .card.hi { border-color:var(--accent-deep); } .card.hi .v { color:var(--accent); }
  /* Category winners: the parallel-verdict chips and the engine tag beside a model name. */
  .pv { font-family:var(--mono); font-size:10.5px; letter-spacing:.05em; text-transform:uppercase;
        padding:2px 7px; border-radius:5px; white-space:nowrap; }
  .pv.good { color:var(--good); border:1px solid var(--good); }
  .pv.bad { color:var(--bad); border:1px solid var(--bad); }
  .eng { font-family:var(--mono); font-size:11px; color:var(--ink-faint); }
  .tbl { overflow-x:auto; border:1px solid var(--border); border-radius:10px; background:var(--panel); }
  table { border-collapse:collapse; width:100%; font-size:12.5px;
          font-family:var(--sans); }
  th, td { padding:7px 11px; text-align:left; border-bottom:1px solid var(--border); }
  thead th { font-family:var(--mono); font-size:10.5px; letter-spacing:.06em; text-transform:uppercase;
             color:var(--ink-faint); font-weight:500; background:var(--panel-2); }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom:none; }
  tbody th { font-weight:600; color:var(--ink-dim); }
  td.n, th.n { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums;
              white-space:nowrap; }
  /* A ledger's total is a different kind of row from the days above it, and has to read as one. */
  tbody tr.sum th, tbody tr.sum td { border-top:2px solid var(--border); background:var(--panel-2);
                                     font-weight:600; color:var(--ink); }
  /* Nine columns of numbers read as noise until they're grouped. The spanning row names what
     each block of them is, and a rule down the seam keeps the blocks from bleeding together. */
  thead tr.grp th { text-align:center; font-size:9.5px; letter-spacing:.12em; padding-bottom:4px;
                    color:var(--ink-faint); border-bottom:none; }
  thead tr.grp th.blank { background:var(--panel-2); }
  th.seam, td.seam { border-left:1px solid var(--border); }
  thead th.wrap { white-space:normal; max-width:96px; line-height:1.25; }
  /* A model identifier has no spaces to wrap at, so one long name used to set the width of
     every table on the page. Bound the two columns that carry names and let them break
     anywhere; .tbl still scrolls if a row genuinely needs more room. */
  th.cfg { max-width:30ch; white-space:normal; overflow-wrap:anywhere; line-height:1.35; }
  code.mdl { max-width:26ch; display:inline-block; overflow-wrap:anywhere; line-height:1.35;
             vertical-align:top; }
  /* Settings columns are context, not findings: keep them present but visually behind the
     measurements, so the eye lands on the numbers that answer the question. */
  td.ax { font-family:var(--mono); font-size:11.5px; color:var(--ink-faint); white-space:nowrap; }
  /* The list of configurations that missed a task is the point of that row, so it wraps
     rather than truncating — it is prose, not a figure. */
  td.fails { font-size:11.5px; color:var(--ink-dim); white-space:normal; line-height:1.45;
             overflow-wrap:anywhere; }
  .warnbox { border-left:3px solid var(--warn); background:var(--panel);
             padding:10px 14px; border-radius:0 8px 8px 0; color:var(--ink-dim); }
  /* Footnotes belong under the thing they qualify — above it they're just a wall to climb.
     A list, not paragraphs: four separate caveats set as prose read as one grey slab. */
  /* Sent before any query runs; a rule at the end of the stream hides it once the real
     content has arrived. No script, so it still works in a saved copy. */
  #building { display:flex; align-items:center; gap:9px; margin:20px 0 0; padding:13px 16px;
              border:1px solid var(--border); border-radius:10px; background:var(--panel);
              color:var(--ink-faint); font-size:13px; }
  #building .spin { color:var(--accent); font-size:15px; animation:reportspin 1.1s linear infinite; }
  @keyframes reportspin { to { transform:rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { #building .spin { animation:none; } }
  @media print { #building { display:none; } }
  ul.fn { color:var(--ink-faint); font-size:11.5px; line-height:1.6; max-width:86ch;
          margin:11px 0 0; padding-left:17px; }
  ul.fn li { margin:0 0 4px; }
  td.win { color:var(--accent); font-weight:700; }
  td.slow { color:var(--bad); font-weight:600; }
  td.ok { color:var(--good); } td.bad { color:var(--bad); font-weight:600; }
  .unit { color:var(--ink-faint); font-weight:400; font-size:11.5px; }
  code { font-family:var(--mono); font-size:.88em; background:var(--panel-2);
         border:1px solid var(--border); border-radius:4px; padding:1px 5px; color:var(--accent); }
  .bar { display:block; height:8px; background:var(--panel-2); border-radius:4px;
         overflow:hidden; border:1px solid var(--border); min-width:60px; }
  .bar i { display:block; height:100%; background:linear-gradient(90deg,var(--accent-deep),var(--accent)); }
  svg { margin:4px 0 16px; display:block; }
  /* Language profile: one stacked bar per model. Colours are assigned per language so the
     same language keeps its colour down the column and the eye can compare rows directly. */
  .langbars { margin:6px 0 16px; }
  .lrow { display:flex; align-items:center; gap:12px; margin:0 0 3px; }
  .lname { flex:0 0 auto; min-width:15ch; text-align:right; }
  .lbar { flex:1 1 auto; display:flex; height:26px; border-radius:5px; overflow:hidden;
          border:1px solid var(--border); background:var(--panel-2); min-width:220px; }
  .lseg { display:flex; align-items:center; justify-content:center; font-family:var(--mono);
          font-size:11px; color:#05080c; font-weight:700; white-space:nowrap; overflow:hidden;
          box-shadow:inset -1px 0 0 rgba(0,0,0,.28); letter-spacing:-.2px; }
  .lleg { display:flex; flex-wrap:wrap; gap:4px 14px; font-family:var(--mono); font-size:11px;
          color:var(--ink-dim); margin:0 0 14px calc(15ch + 12px); }
  .lchip { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
  .lchip i { width:9px; height:9px; border-radius:2px; display:inline-block; flex:none;
             border:1px solid rgba(0,0,0,.35); }
  /* The failure taxonomy leads with the reason and explains it underneath, so the table can
     be read without carrying a legend in your head. */
  .fr-d { font-weight:400; font-size:11px; color:var(--ink-faint); margin-top:2px;
          max-width:44ch; }
  .lseg-none { background:repeating-linear-gradient(45deg,var(--panel-2),var(--panel-2) 5px,
               var(--border) 5px,var(--border) 10px); color:var(--ink-dim); }
  details.lnum { margin:0 0 16px; }
  details.lnum summary { cursor:pointer; color:var(--ink-faint); font-size:12.5px; }
  .lang-python { background:#5b9bd5; } .lang-javascript { background:#e6c84f; }
  .lang-typescript { background:#4b8bbe; } .lang-c { background:#8fb7d9; }
  .lang-cpp { background:#a06fc4; } .lang-csharp { background:#7bc47b; }
  .lang-java { background:#e08a4a; } .lang-go { background:#57d1e0; }
  .lang-rust { background:#d97757; } .lang-kotlin { background:#b58ae0; }
  .lang-bash { background:#8aa06a; } .lang-powershell { background:#6d8ed6; }
  .lang-html { background:#d9736a; } .lang-css { background:#6ab0d9; }
  .lang-lua { background:#7f8ce0; } .lang-sql { background:#c9a35e; }
  .lang-php { background:#9a8fd0; } .lang-ruby { background:#d96a7f; }
  .lang-swift { background:#e09a5a; } .lang-scala { background:#d95f5f; }
  .ct { font-size:10.5px; fill:var(--ink-faint); font-weight:600; text-transform:uppercase;
        letter-spacing:.6px; font-family:var(--sans); }
  .cl { font-size:11.5px; fill:var(--ink-dim); font-family:var(--sans); }
  .cv { font-size:11.5px; fill:var(--ink); font-weight:600; font-family:var(--sans); }
  .ann { font-size:12.5px; font-family:var(--sans); }
  /* The one loud element on the page. Everything else stays quiet so this reads first. */
  .hero { background:linear-gradient(180deg,var(--panel),var(--panel-2));
          border:1px solid var(--border); border-left:3px solid var(--accent);
          border-radius:12px; padding:20px 22px; margin:18px 0 4px; }
  .hero .lede { font-size:clamp(17px,2.2vw,21px); line-height:1.45; color:var(--ink);
                margin:0; letter-spacing:-.01em; }
  .hero .lede b { color:var(--accent); font-family:var(--mono); font-weight:600;
                  font-size:1.06em; letter-spacing:-.02em; }
  .hero .why { color:var(--ink-faint); font-size:13px; margin:10px 0 0; max-width:74ch; }
  /* Minor tables sit side by side: full width would give them an authority they haven't earned. */
  /* A single run is a record: read down a list of properties, not across a 14-column row. */
  .spec { display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:1px;
          background:var(--border); border:1px solid var(--border); border-radius:10px;
          overflow:hidden; margin:16px 0 4px; }
  .spec div { background:var(--panel); padding:11px 14px; }
  .spec .k { font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
             color:var(--ink-faint); margin:0 0 5px; }
  .spec .v { font-family:var(--mono); font-size:14px; color:var(--ink); margin:0;
             overflow-wrap:anywhere; }
  .spec .v.big { font-size:19px; font-weight:600; color:var(--accent); letter-spacing:-.02em; }
  .band { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:0 26px; }
  .band > section > h2 { margin-top:26px; }
  section { min-width:0; }
  footer { margin-top:38px; padding-top:14px; border-top:1px solid var(--border);
           color:var(--ink-faint); font-size:11.5px; }
  /* The task catalogue in the method section: id + one line on what it tests. Full width,
     below the spec grid — a wall of 29 descriptions inside one grid cell dwarfed the
     one-word cells beside it. */
  .catalog { margin-top:10px; }
  .catalog .tierblk { background:var(--panel); border:1px solid var(--border);
                      border-radius:10px; padding:11px 14px; margin-top:8px; }
  .catalog .k { font-family:var(--mono); font-size:10px; letter-spacing:.1em;
                text-transform:uppercase; color:var(--ink-faint); margin:0 0 6px; }
  .tl { list-style:none; margin:4px 0 0; padding:0; column-width:330px; column-gap:26px; }
  .tl li { margin:2px 0; font-size:12.5px; break-inside:avoid; }
  .tl span { color:var(--ink-faint); }
  /* Failure examples: collapsed by default — evidence on demand, not a wall of stack traces. */
  details.fx { margin:7px 0; border:1px solid var(--border); border-radius:8px;
               padding:8px 12px; background:var(--panel); }
  details.fx summary { cursor:pointer; font-size:13px; }
  details.fx summary span { color:var(--ink-faint); }
  details.fx ul { list-style:none; margin:8px 0 2px; padding:0; }
  details.fx li { margin:5px 0; font-size:12.5px; overflow-wrap:anywhere; }
  details.fx li code { color:var(--ink-dim); }
  details.fxc { margin:4px 0 2px 14px; }
  details.fxc summary { cursor:pointer; font-size:11.5px; color:var(--ink-faint); }
  details.fxc pre { margin:6px 0 4px; padding:9px 12px; font-size:11.5px; line-height:1.45;
                    background:var(--panel-2); border:1px solid var(--border);
                    border-radius:6px; overflow:auto; max-height:340px;
                    white-space:pre-wrap; overflow-wrap:anywhere; }
  .tdesc { display:block; font-family:var(--sans, inherit); font-weight:400;
           font-size:11.5px; color:var(--ink-faint); margin-top:1px; }
  /* Weighted standings: the two-segment bar IS the weighting made visible. */
  .wslider { display:flex; align-items:center; gap:12px; margin:10px 0 4px;
             font-size:12px; color:var(--ink-faint); }
  .wslider input { flex:0 1 260px; accent-color:var(--accent); }
  .wslider b { color:var(--ink); font-variant-numeric:tabular-nums; }
  .wbar { display:inline-block; width:150px; height:11px; border-radius:3px;
          background:var(--panel-2); overflow:hidden; vertical-align:middle; }
  .wbar i { display:inline-block; height:100%; float:left; }
  .wbar .q { background:var(--accent); }
  .wbar .s { background:var(--blue); }
  .wtab td b { font-variant-numeric:tabular-nums; margin-left:8px; }
  .wkey i { display:inline-block; width:10px; height:10px; border-radius:2px;
            vertical-align:-1px; margin-right:4px; }
  /* Interactive layer: dim everything but the hovered model; screen-only extras. */
  html.hl [data-m] { opacity:.22; transition:opacity .12s; }
  html.hl [data-m].lit { opacity:1; }
  tr.taskrow { cursor:pointer; }
  tr.taskrow:hover th code { text-decoration:underline; }
  tr.drill td { background:var(--panel); padding:10px 14px; }
  tr.drill .k { font-family:var(--mono); font-size:10.5px; letter-spacing:.08em;
                text-transform:uppercase; color:var(--ink-faint); margin:0 0 6px; }
  .ixtip { position:fixed; pointer-events:none; background:var(--panel);
           border:1px solid var(--border); border-radius:6px; padding:6px 10px;
           font-size:12px; line-height:1.5; z-index:60; max-width:240px;
           box-shadow:0 6px 18px rgba(0,0,0,.28); transition:opacity .12s; }
  #ix-scatter svg { display:block; }
  .brush .selection { stroke:var(--accent); fill:var(--accent); fill-opacity:.07; }
  @media print {
    .ix-only, .ixtip, tr.drill { display:none !important; }
    :root { --bg:#fff; --panel:#fff; --panel-2:#f5f5f2; --border:#e4e4e1;
            --ink:#121212; --ink-dim:#454b52; --ink-faint:#727272;
            --accent:#c4321f; --accent-deep:#8f2416; --good:#1f7a47; --bad:#b3261e;
            --ghost:#b9b9b9; --grid:#ebebe8; --blue:#3e6fa8;
            color-scheme: light; }
    body { padding:0; }
    h2 { page-break-after:avoid; }
    table, svg, .cards { page-break-inside:avoid; }
  }
"""


def _report_head(title: str, eyebrow: str) -> str:
    """Everything that can be sent before a single row has been counted."""
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{_h(title)}</title><style>{_REPORT_CSS}</style></head><body>"
        # The one script on the page. Self-contained, so a saved copy keeps its toggle;
        # localStorage remembers the choice, and with nothing stored the OS preference rules.
        "<button id=\"themeflip\" type=\"button\" aria-label=\"Switch colour theme\"></button>"
        "<script>(function(){var r=document.documentElement,k=\"ai-proxy-report-theme\","
        "b=document.getElementById(\"themeflip\");"
        "function eff(){return r.getAttribute(\"data-theme\")||"
        "(window.matchMedia&&matchMedia(\"(prefers-color-scheme: dark)\").matches"
        "?\"dark\":\"light\")}"
        "function lab(){b.textContent=eff()===\"dark\"?\"\u2600 Light\":\"\u263e Dark\"}"
        "try{var s=localStorage.getItem(k);if(s)r.setAttribute(\"data-theme\",s)}catch(e){}"
        "b.addEventListener(\"click\",function(){var n=eff()===\"dark\"?\"light\":\"dark\";"
        "r.setAttribute(\"data-theme\",n);try{localStorage.setItem(k,n)}catch(e){}lab()});"
        "lab()})();</script>"
        "<div class=\"wrap\">"
        f"<p class=\"eyebrow\">{_h(eyebrow)}</p>"
        f"<h1>{_h(title)}</h1>"
    )


# Shown the instant the page opens and hidden by a rule sent at the very end, so it disappears
# on its own when the real content has arrived. No script: this has to survive being saved.
_REPORT_BUILDING = (
    '<div id="building"><span class="spin">\u25d0</span> Building the report \u2014 reading every request '
    'the proxy has recorded. A few seconds.</div>'
)
_REPORT_BUILT = "<style>#building{display:none}</style>"


def _report_foot() -> str:
    return (_REPORT_BUILT
            + "<footer>Generated by AI Proxy \u00b7 every request behind these numbers passed "
              "through the proxy and is individually inspectable in the dashboard.</footer>"
              "</div></body></html>")


def _report_page(title: str, eyebrow: str, sub: str, meta: list, body: str) -> str:
    """Shared chrome for every generated report."""
    meta_html = "".join(f"<div>{_h(k)} <b>{_h(v)}</b></div>" for k, v in meta if v is not None)
    return (_report_head(title, eyebrow)
            + f"<p class=\"sub\">{_h(sub)}</p>"
            + f"<div class=\"meta\">{meta_html}</div>"
            + body
            + _report_foot())



# A label that crosses a dot or a gridline is unreadable without this: a stroke in the page
# colour painted under the glyphs. Strokes in var(--bg), so it works in both themes.
_SVG_HALO = ('paint-order="stroke" stroke="var(--bg)" stroke-width="3.5" '
             'stroke-linejoin="round"')


_BENCH_WEIGHT_DEFAULT = 30      # % of the score that is speed; the rest is correctness


def _bench_weighted_data(rows: list) -> list:
    """One entry per model — its best cell's correctness and decode rate, speed normalised to
    the fastest model in the field so the two axes share a 0..1 scale."""
    bpm = [r for r in _bench_best_per_model(rows)
           if r.get("decode_p50") and r.get("perfect_rate") is not None]
    if len(bpm) < 3:
        return []
    dmax = max(r["decode_p50"] for r in bpm)
    return [{"m": (r.get("_name") or _bench_label_display(r.get("label") or "")
                   ).split(" · ")[0],
             "q": round(r["perfect_rate"], 4),
             "d": round(r["decode_p50"], 1),
             "s": round(r["decode_p50"] / dmax, 4)} for r in bpm]


def _bench_weighted_rows(data: list, w: float) -> list:
    """(entry, score, q_share, s_share) sorted by score. Mirrored exactly by the page's JS —
    the server renders the default so the section survives with scripts off."""
    out = []
    for e in data:
        qc, sc = (1 - w) * e["q"], w * e["s"]
        out.append((e, qc + sc, qc, sc))
    return sorted(out, key=lambda t: -t[1])


def _bench_weighted_html(rows: list) -> str:
    """A ranking that says out loud what it trades: N% correctness, M% relative speed.

    The raw table sorts by correctness alone, which crowns a model 1 point more correct and
    half the speed — an exchange nobody would actually make. The weights are printed in the
    heading, drawn as the two segments of every score bar, and adjustable live; there is no
    hidden judgement to disagree with, only a slider."""
    data = _bench_weighted_data(rows)
    if not data:
        return ""
    w = _BENCH_WEIGHT_DEFAULT / 100.0
    dmax = max(e["d"] for e in data)
    fastest = next(e["m"] for e in data if e["d"] == dmax)
    trs = []
    for rank, (e, score, qc, sc) in enumerate(_bench_weighted_rows(data, w), start=1):
        trs.append(
            f'<tr data-m="{_h(e["m"])}"><td class="n">{rank}</td>'
            f'<th scope="row"><code class="mdl">{_h(e["m"])}</code></th>'
            f'<td class="n">{e["q"] * 100:.0f}%</td>'
            f'<td class="n">{e["d"]:,.1f}</td>'
            f'<td><div class="wbar"><i class="q" style="width:{qc * 100:.1f}%"></i>'
            f'<i class="s" style="width:{sc * 100:.1f}%"></i></div>'
            f'<b>{score * 100:.0f}</b></td></tr>')
    payload = json.dumps(data)
    return f"""<h2>Weighted standings</h2>
<p class="note">Correctness alone is not a ranking — one point of correctness is not worth half
the speed. Each model's best cell scores
<b><span id="wq">{100 - _BENCH_WEIGHT_DEFAULT}</span>% × correctness +
<span id="ws">{_BENCH_WEIGHT_DEFAULT}</span>% × relative speed</b>, where relative speed is
decode rate against the fastest model in this report ({_h(fastest)}, {dmax:,.1f} tok/s = 1.0).
The bar is the weighting made visible:
<span class="wkey"><i style="background:var(--accent)"></i>correctness</span> ·
<span class="wkey"><i style="background:var(--blue)"></i>speed</span>.
Drag to change what you value; the ranking recomputes.</p>
<div class="wslider"><span>all correctness</span>
<input id="wrange" type="range" min="0" max="100" step="5"
 value="{_BENCH_WEIGHT_DEFAULT}" aria-label="Speed weight percent">
<span>all speed</span><b id="wshow">{100 - _BENCH_WEIGHT_DEFAULT} / {_BENCH_WEIGHT_DEFAULT}</b></div>
<div class="tbl"><table class="wtab"><thead><tr><th class="n">#</th><th>Model</th>
<th class="n">Correct</th><th class="n">tok/s</th><th>Score</th></tr></thead>
<tbody id="wbody">{"".join(trs)}</tbody></table></div>
<script>(function(){{var D={payload};
var r=document.getElementById("wrange"),b=document.getElementById("wbody");
function esc(t){{var d=document.createElement("i");d.textContent=t;return d.innerHTML}}
function render(){{var w=r.value/100;
document.getElementById("wq").textContent=Math.round(100-w*100);
document.getElementById("ws").textContent=Math.round(w*100);
document.getElementById("wshow").textContent=Math.round(100-w*100)+" / "+Math.round(w*100);
var rows=D.map(function(e){{var qc=(1-w)*e.q,sc=w*e.s;
return{{e:e,score:qc+sc,qc:qc,sc:sc}}}}).sort(function(a,c){{return c.score-a.score}});
var prev={{}};Array.prototype.forEach.call(b.children,function(tr){{
prev[tr.getAttribute("data-m")]=tr.getBoundingClientRect().top}});
b.innerHTML=rows.map(function(t,i){{
return '<tr data-m="'+esc(t.e.m)+'"><td class="n">'+(i+1)+'</td><th scope="row"><code class="mdl">'+esc(t.e.m)
+'</code></th><td class="n">'+Math.round(t.e.q*100)+'%</td><td class="n">'
+t.e.d.toLocaleString()+'</td><td><div class="wbar"><i class="q" style="width:'
+(t.qc*100).toFixed(1)+'%"></i><i class="s" style="width:'+(t.sc*100).toFixed(1)
+'%"></i></div><b>'+Math.round(t.score*100)+'</b></td></tr>'}}).join("");
Array.prototype.forEach.call(b.children,function(tr){{
var m=tr.getAttribute("data-m");if(!(m in prev))return;
var d=prev[m]-tr.getBoundingClientRect().top;if(!d)return;
tr.style.transform="translateY("+d+"px)";tr.style.transition="none";
requestAnimationFrame(function(){{tr.style.transition="transform .35s ease";
tr.style.transform=""}})}})}}
r.addEventListener("input",render)}})();</script>"""


def _bench_parallel_groups(rows: list) -> list:
    """Pair each (model, engine)'s best sequential cell with its most concurrent cell, and
    judge whether concurrency actually overlapped.

    The tell is time-to-first-token: an engine that batches holds TTFT roughly flat under
    load, one that queues multiplies it (qwen3.6 went 0.6s → 35s at 4× with decode speed
    unchanged — four requests taking turns). A serialized cell's aggregate throughput is
    its single-stream rate, whatever the concurrency setting claimed; crediting conc ×
    decode there would award the queue a 4× it never delivered."""
    groups: dict = {}
    for r in rows:
        if not r.get("decode_p50"):
            continue
        name = r.get("_name") or _bench_label_display(r.get("label") or "")
        parts = [p.strip() for p in name.split(" · ")]
        engine = next((p for p in parts[1:] if p.startswith("@")), "")
        g = groups.setdefault((parts[0], engine),
                              {"model": parts[0], "engine": engine, "seq": None, "par": None})
        conc = int(r.get("concurrency") or 1)
        if conc <= 1:
            if g["seq"] is None or r["decode_p50"] > g["seq"]["decode_p50"]:
                g["seq"] = r
        else:
            cur = g["par"]
            if cur is None or ((int(cur.get("concurrency") or 1), cur["decode_p50"])
                               < (conc, r["decode_p50"])):
                g["par"] = r
    out = []
    for g in groups.values():
        if not (g["seq"] and g["par"]):
            continue
        seq, par, conc = g["seq"], g["par"], int(g["par"].get("concurrency") or 1)
        ts, tp = seq.get("ttft_p50"), par.get("ttft_p50")
        serialized = bool(ts and tp and tp > 3 * ts + 1000)
        agg = par["decode_p50"] * (1 if serialized else conc)
        out.append({**g, "conc": conc, "serialized": serialized, "agg": agg,
                    "scale": (agg / seq["decode_p50"]) if seq["decode_p50"] else None})
    return sorted(out, key=lambda g: -g["agg"])


def _bench_category_winners_html(rows: list) -> str:
    """Three titles, one line each: most correct, fastest single stream, best under
    parallel load — with the parallel evidence table underneath. The weighted standings
    answer "what should I run"; this answers "who wins each event", and the parallel
    column is the one nothing else in the report shows."""
    ok = [r for r in rows if r.get("decode_p50")]
    if len(ok) < 2:
        return ""

    def nm(r):
        return (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]

    cards = []
    graded = [r for r in ok if r.get("perfect_rate") is not None]
    if graded:
        c = max(graded, key=lambda r: (r["perfect_rate"], r["decode_p50"]))
        cards.append(('Most correct', nm(c),
                      f'{c["perfect_rate"] * 100:.0f}% of tasks fully correct'))
    seq = [r for r in ok if int(r.get("concurrency") or 1) <= 1]
    if seq:
        f = max(seq, key=lambda r: r["decode_p50"])
        cards.append(('Fastest single stream', nm(f),
                      f'{f["decode_p50"]:,.1f} tok/s sequential'))
    pgroups = _bench_parallel_groups(rows)
    batching = [g for g in pgroups if not g["serialized"]]
    if batching:
        b = batching[0]
        cards.append((f'Best under {b["conc"]}× load', b["model"],
                      f'{b["agg"]:,.0f} tok/s aggregate '
                      f'({b["par"]["decode_p50"]:,.1f}/stream{" " + b["engine"] if b["engine"] else ""}), '
                      f'first token in {b["par"]["ttft_p50"] / 1000:.1f}s'))
    if not cards:
        return ""
    cards_html = "".join(
        f'<div class="card" data-m="{_h(m)}"><p class="k">{_h(k)}</p>'
        f'<p class="v">{_h(m)}</p><p class="d">{_h(d)}</p></div>'
        for k, m, d in cards)

    table_html = ""
    if pgroups:
        trs = []
        for g in pgroups:
            ttfts = (f'{g["seq"]["ttft_p50"] / 1000:.1f}s → {g["par"]["ttft_p50"] / 1000:.1f}s'
                     if g["seq"].get("ttft_p50") and g["par"].get("ttft_p50") else "—")
            verdict = ('<span class="pv bad">queues — requests wait in line</span>'
                       if g["serialized"] else
                       f'<span class="pv good">batches · {g["scale"]:.1f}×</span>')
            trs.append(
                f'<tr data-m="{_h(g["model"])}">'
                f'<th scope="row"><code class="mdl">{_h(g["model"])}</code>'
                f'{" <span class=eng>" + _h(g["engine"]) + "</span>" if g["engine"] else ""}</th>'
                f'<td class="n">{g["seq"]["decode_p50"]:,.1f}</td>'
                f'<td class="n">{g["par"]["decode_p50"]:,.1f} × {g["conc"]}</td>'
                f'<td class="n"><b>{g["agg"]:,.0f}</b></td>'
                f'<td class="n">{ttfts}</td>'
                f'<td>{verdict}</td></tr>')
        table_html = (
            '<p class="note">Aggregate = per-stream decode × streams, credited only when '
            'time-to-first-token stays flat under load — TTFT exploding while decode holds '
            'steady means the engine queued the requests, and its real throughput is one '
            'stream, whatever the concurrency was set to.</p>'
            '<div class="tbl"><table><thead><tr><th>Model</th><th class="n">Solo tok/s</th>'
            '<th class="n">Per stream</th><th class="n">Aggregate</th>'
            '<th class="n">TTFT solo → loaded</th><th>Verdict</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')

    return (f'<h2>Category winners</h2><div class="cards">{cards_html}</div>{table_html}')


# Populated at registration time by proxy.py, which owns the suite definitions — the report
# must not re-derive which task is which category, or the two definitions drift.
TASK_CATEGORY: dict = {}
TASK_SIDE: dict = {}
# task id -> language its prompt demanded (directed langpref variants only).
TASK_REQUESTED_LANG: dict = {}

_CAT_ORDER = ("coding", "agentic", "security", "instruct", "refusal", "memory",
              "longcontext")
_CAT_BLURB = {
    "coding": "write code that passes its tests",
    "agentic": "drive tools across many turns and finish",
    "security": "defend code, and find the hole in it",
    "instruct": "produce the shape that was asked for",
    "refusal": "engage with security work, decline the harmful end",
    "memory": "keep a store a future session can inherit",
    "longcontext": "still find a fact after a very long prompt",
}


def _task_sort_key(task_id: str):
    """Order tasks for display. Plain alphabetical is right for every suite except the
    long-context ladder, whose ids sort 128k, 16k, 1m, 256k, 300k, 512k, 64k, 700k — the axis
    of a curve, scrambled. A rung is placed by the size it measures instead.
    """
    if task_id.startswith("longctx_"):
        suffix = task_id[len("longctx_"):].lower()
        try:
            if suffix.endswith("m"):
                return (0, float(suffix[:-1]) * 1_000_000, task_id)
            if suffix.endswith("k"):
                return (0, float(suffix[:-1]) * 1_000, task_id)
        except ValueError:
            pass
    return (1, 0.0, task_id)


def _bench_category_html(tasks: dict, axis_names: list) -> str:
    """Correctness per category, per configuration — the point of merging the suites.

    One aggregate number hides the thing you most need to know: a model can write clean
    code and still take its orders from a hostile tool result. These are different skills
    with different failure modes, so they get different columns, and the security column is
    split red/blue because finding a hole and closing one are also not the same skill.
    """
    if not tasks or not TASK_CATEGORY:
        return ""
    cats = [c for c in _CAT_ORDER
            if any(TASK_CATEGORY.get(t) == c for t in tasks)]
    if len(cats) < 2:
        return ""      # a single-category run is the plain per-task table's job
    sides = sorted({TASK_SIDE.get(t) for t in tasks
                    if TASK_CATEGORY.get(t) == "security" and TASK_SIDE.get(t)})

    def mean_for(name, pred):
        vals = [rates[name] for tid, rates in tasks.items()
                if pred(tid) and rates.get(name) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def cell(v):
        if v is None:
            return '<td class="n">—</td>'
        cls = " win" if v >= 0.9 else (" bad" if v < 0.6 else "")
        return f'<td class="n{cls}">{v * 100:.0f}%</td>'

    rows_out = []
    for name in axis_names:
        per = {c: mean_for(name, lambda t, c=c: TASK_CATEGORY.get(t) == c)[0] for c in cats}
        if all(v is None for v in per.values()):
            continue
        side_cells = "".join(
            cell(mean_for(name, lambda t, s=s: TASK_CATEGORY.get(t) == "security"
                          and TASK_SIDE.get(t) == s)[0]) for s in sides)
        overall = [v for v in per.values() if v is not None]
        rows_out.append((name, per, side_cells,
                         sum(overall) / len(overall) if overall else 0))
    if not rows_out:
        return ""
    rows_out.sort(key=lambda r: -r[3])
    body = "".join(
        f'<tr data-m="{_h(name.split(" · ")[0])}">'
        f'<th scope="row"><code class="mdl">{_h(name.split(" · ")[0])}</code></th>'
        + "".join(cell(per[c]) for c in cats) + side_cells + "</tr>"
        for name, per, side_cells, _ in rows_out)
    counts = {c: sum(1 for t in tasks if TASK_CATEGORY.get(t) == c) for c in cats}
    head = "".join(f'<th class="n">{_h(c.title())}<br>'
                   f'<span class="ct">{counts[c]} tasks</span></th>' for c in cats)
    head += "".join(f'<th class="n">— {_h(s)}<br><span class="ct">team</span></th>'
                    for s in sides)

    # The sentence a table cannot say: where the same model is strong and weak.
    note = ""
    top = rows_out[0]
    spread = [(c, top[1][c]) for c in cats if top[1][c] is not None]
    if len(spread) > 1:
        best_c, best_v = max(spread, key=lambda kv: kv[1])
        worst_c, worst_v = min(spread, key=lambda kv: kv[1])
        if best_v - worst_v >= 0.15:
            note = (f'<p class="note"><b>{_h(top[0].split(" · ")[0])}</b> leads overall on '
                    f'{best_v * 100:.0f}% {_h(best_c)} — and {worst_v * 100:.0f}% '
                    f'{_h(worst_c)}. Averaging those into one score would describe a model '
                    f'that does not exist.</p>')
    legend = " · ".join(f"<b>{_h(c.title())}</b> {_CAT_BLURB[c]}" for c in cats)
    return (f'<h2>Results by category</h2><p class="note">{legend}.</p>'
            f'<div class="tbl"><table><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>{note}')


def _bench_efficiency_html(rows: list) -> str:
    """Tokens spent per task actually solved — the cost side of the scorecard.

    Decode rate says how fast tokens arrive; it cannot say how many were needed. A model
    that reasons for 900 tokens to reach the same answer another reaches in 120 is three
    quarters waste at identical tok/s, and on a shared box that waste is someone else's
    latency. Tokens per SOLVED task (not per task) is the honest denominator: spending
    fewer tokens to fail is not efficiency.
    """
    usable = [r for r in rows
              if r.get("mean_tokens") and r.get("perfect_rate") is not None
              and r.get("n_total")]
    if len(usable) < 2:
        return ""

    def nm(r):
        return (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]

    # Aggregate every row a model has, rather than keeping one of them. The previous version
    # sorted by perfect_rate and kept the FIRST row per model — which silently reported each
    # model at its best suite. A model with a 65% full-v2 row and a 100% langpref row was
    # printed as "100% correct", flattering everything and comparing nothing.
    acc: dict = {}
    for r in usable:
        a = acc.setdefault(nm(r), {"tasks": 0, "solved": 0.0, "spent": 0.0, "think": []})
        a["tasks"] += r["n_total"]
        a["solved"] += (r["perfect_rate"] or 0) * r["n_total"]
        a["spent"] += r["mean_tokens"] * r["n_total"]
        if r.get("reasoning_tok_p50"):
            a["think"].append(r["reasoning_tok_p50"])
    entries = []
    for name, a in acc.items():
        if a["solved"] <= 0:
            continue          # nothing solved: a per-solved figure would be a divide by zero
        entries.append({"m": name, "per": a["spent"] / a["solved"],
                        "mean": a["spent"] / a["tasks"], "q": a["solved"] / a["tasks"],
                        "think": (sum(a["think"]) / len(a["think"])) if a["think"] else None})
    if len(entries) < 2:
        return ""
    entries.sort(key=lambda e: e["per"])
    best = entries[0]["per"]
    trs = "".join(
        f'<tr data-m="{_h(e["m"])}"><td class="n">{i}</td>'
        f'<th scope="row"><code class="mdl">{_h(e["m"])}</code></th>'
        f'<td class="n">{e["q"] * 100:.0f}%</td>'
        f'<td class="n">{e["mean"]:,.0f}</td>'
        f'<td class="n">{"—" if not e["think"] else format(e["think"], ",.0f")}</td>'
        f'<td class="n{" win" if i == 1 else ""}"><b>{e["per"]:,.0f}</b></td>'
        f'<td class="n">{e["per"] / best:.1f}×</td></tr>'
        for i, e in enumerate(entries, start=1))
    worst = entries[-1]
    note = ""
    if worst["per"] >= 2 * best:
        note = (f'<p class="note"><b>{_h(worst["m"])}</b> spends '
                f'{worst["per"] / best:.1f}× the tokens of <b>{_h(entries[0]["m"])}</b> per '
                f'task it gets right. At equal decode rates that is the same answer for '
                f'{worst["per"] / best:.1f}× the wall-clock and {worst["per"] / best:.1f}× '
                f'the KV cache.</p>')
    return ('<h2>Cost per correct answer</h2>'
            '<p class="note">Output tokens spent per task fully solved — reasoning tokens '
            'included, because the box pays for them whether or not you read them. Failed '
            'tasks still cost their tokens; they just buy nothing.</p>'
            '<div class="tbl"><table><thead><tr><th class="n">#</th><th>Model</th>'
            '<th class="n">Correct</th><th class="n">Tokens/answer</th>'
            '<th class="n">of which thinking</th><th class="n">Tokens/SOLVED</th>'
            '<th class="n">vs best</th></tr></thead>'
            f'<tbody>{trs}</tbody></table></div>{note}')


def _bench_variance_html(runs: list, rows: list, axis_names: list) -> str:
    """Which tasks a model gets right only sometimes.

    Averages hide instability, and instability is what breaks agents: a task passing 3 runs
    in 5 is not "60% correct", it is a coin flip that will land wrong inside a 30-step
    episode. Only visible with repeats, so the section is absent for single-run cells.
    """
    flaky: dict = {}
    cells = 0
    for run, nm in zip(runs, axis_names):
        cfg = run.get("config") or {}
        if int(cfg.get("runs") or 1) < 2:
            continue
        cells += 1
        q = (((run.get("results") or {}).get("summary") or {}).get("quality") or {})
        for t in (q.get("tasks") or []):
            rate = t.get("perfect_rate")
            if rate is None or rate in (0.0, 1.0):
                continue          # always-right and always-wrong are stable, not flaky
            flaky.setdefault(t["task"], []).append((nm.split(" · ")[0], rate,
                                                    int(cfg.get("runs") or 1)))
    if not cells:
        return ""
    if not flaky:
        return ('<h2>Determinism</h2><p class="note">Every task scored identically across '
                'all repeats in every cell — no task flipped between runs. At temperature 0 '
                'that is what you want, and it means the correctness figures above are '
                'measurements rather than samples.</p>')
    trs = []
    for tid, hits in sorted(flaky.items(), key=lambda kv: -len(kv[1])):
        for model, rate, n in sorted(hits, key=lambda h: h[1]):
            trs.append(f'<tr data-m="{_h(model)}"><th scope="row"><code>{_h(tid)}</code>'
                       f'</th><td>{_h(TASK_DESC.get(tid) or "")}</td>'
                       f'<td><code class="mdl">{_h(model)}</code></td>'
                       f'<td class="n">{round(rate * n)}/{n}</td>'
                       f'<td class="n">{rate * 100:.0f}%</td></tr>')
    return ('<h2>Determinism</h2>'
            f'<p class="note"><b>{len(flaky)}</b> task'
            f'{"s" if len(flaky) != 1 else ""} did not settle: the same model, the same '
            'prompt, a different outcome between repeats. Treat these scores as samples, '
            'not measurements — and expect the coin to land wrong somewhere inside a long '
            'agent run.</p>'
            '<div class="tbl"><table><thead><tr><th>Task</th><th>What it asks</th>'
            '<th>Model</th><th class="n">Passed</th><th class="n">Rate</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


# Short codes for bar segments too narrow for the full name. One task out of sixteen is ~6%
# of the bar — enough for "ps1", not for "powershell".
_LANG_SHORT = {
    "python": "py", "javascript": "js", "typescript": "ts", "csharp": "c#", "cpp": "c++",
    "powershell": "ps1", "kotlin": "kt", "rust": "rs", "ruby": "rb", "bash": "sh",
    "shell": "sh", "java": "java", "swift": "swift", "scala": "scala", "html": "html",
    "css": "css", "sql": "sql", "go": "go", "lua": "lua", "php": "php", "c": "c",
    "perl": "pl", "haskell": "hs", "elixir": "ex", "clojure": "clj", "dart": "dart",
}


# Why a task failed, in priority order — the first match wins, so a truncated answer is
# counted as truncated rather than as the wrong answer it necessarily also was. Every key
# here is something the graders already recorded and the report used to discard: the page
# said 84/119 and left the other 35 as an undifferentiated deficit, when a third of them
# were the model running out of tokens and a third were it refusing to engage.
_FAIL_REASONS = [
    ("backend",    "backend or harness error", "the request never produced a gradeable answer"),
    ("build",      "did not compile",          "code was returned but the toolchain rejected it"),
    ("truncated",  "ran out of tokens",        "hit the max_tokens ceiling mid-answer"),
    ("exhausted",  "agent ran out of steps",   "the episode ended before the task was done"),
    ("malformed",  "malformed tool calls",     "emitted tool calls the dispatcher could not parse"),
    ("looped",     "repeated itself",          "the same call often enough to be a loop"),
    ("nocode",     "produced no code",         "prose or a diagram where a program was asked for"),
    ("refused",    "declined to engage",       "refused a request the suite expects answered"),
    ("wrong",      "wrong answer",             "ran, returned, and did not match"),
]


def _bench_failure_reason(row: dict, grade: dict, max_tokens=None) -> str:
    """Classify one failed task. Grounded in fields the graders already emit."""
    if row.get("error") or not (row.get("text") or "").strip():
        return "backend"
    if grade.get("build") is False or (grade.get("error") and "cases" not in grade):
        return "build"
    if grade.get("truncated") or (max_tokens and (row.get("completion_tokens") or 0) >= max_tokens - 8):
        return "truncated"
    if grade.get("exhausted"):
        return "exhausted"
    if grade.get("malformed"):
        return "malformed"
    if (grade.get("repeats") or 0) > 0:
        return "looped"
    gots = " ".join(str(c.get("got") or "") for c in (grade.get("cases") or []) if not c.get("ok"))
    if "no code in any language" in gots or "no code" in gots:
        return "nocode"
    if "declined" in gots or "refused" in gots:
        return "refused"
    return "wrong"


def _bench_failure_taxonomy_html(runs: list, rows_meta: list, axis_names: list) -> str:
    """Counts of WHY each model failed, not just how often.

    A score says a model lost 35 tasks. It cannot say whether it was wrong, silent, cut off,
    or unwilling — and those call for four different responses: a better model, a bigger token
    budget, a different prompt, or a policy change. The information was always in the grades.
    """
    per: dict = {}
    for run, nm, meta in zip(runs, axis_names, rows_meta):
        model = nm.split(" · ")[0]
        mt = (run.get("config") or {}).get("max_tokens")
        for row in ((run.get("results") or {}).get("rows") or []):
            g = row.get("grade") or {}
            if not g:
                continue
            if (g.get("passed") or 0) >= (g.get("total") or 1):
                continue
            per.setdefault(model, {})
            k = _bench_failure_reason(row, g, mt)
            per[model][k] = per[model].get(k, 0) + 1
    if not per or not any(sum(v.values()) for v in per.values()):
        return ""
    models = [m for m in dict.fromkeys(nm.split(" · ")[0] for nm in axis_names) if m in per]
    live = [(k, lbl, desc) for k, lbl, desc in _FAIL_REASONS
            if any(per[m].get(k) for m in models)]
    if not live:
        return ""
    head = "".join(f'<th class="n">{_h(m)}</th>' for m in models)
    body = []
    for k, lbl, desc in live:
        cells = ""
        for m in models:
            n = per[m].get(k, 0)
            tot = sum(per[m].values()) or 1
            cells += (f'<td class="n">{n}'
                      + (f' <span class="ct">({n / tot * 100:.0f}%)</span>' if n else "") + "</td>")
        body.append(f'<tr><th scope="row">{_h(lbl)}<div class="fr-d">{_h(desc)}</div></th>{cells}</tr>')
    tot_row = "".join(f'<td class="n"><b>{sum(per[m].values())}</b></td>' for m in models)
    body.append(f'<tr><th scope="row">total failures</th>{tot_row}</tr>')
    return ('<h2>Why it failed</h2>'
            '<p class="note">Every failed task, classified by the first thing that went wrong. '
            'A model that is cut off, silent, unwilling or simply incorrect has four different '
            'problems, and only one of them is answered by picking a different model — the '
            'others are a token budget, a prompt, and a policy decision.</p>'
            f'<div class="tbl"><table><thead><tr><th>Reason</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _bench_language_profile_html(runs: list, axis_names: list) -> str:
    """What each model reached for when the task did not say.

    A score cannot carry this: the answer to "which language" is a distribution, not a
    number. Two views, because they answer different questions — the profile (how often
    each model chose each language) says what a model IS, and the per-task grid says where
    two models disagree, which is where the interesting arguments live.
    """
    picks: dict = {}          # model -> {task -> language}, free-choice tasks only
    directed: dict = {}       # model -> {task -> language}, where a language was demanded
    for run, nm in zip(runs, axis_names):
        model = nm.split(" · ")[0]
        for row in ((run.get("results") or {}).get("rows") or []):
            g = row.get("grade") or {}
            lang = g.get("picked")
            if not lang:
                continue
            tid = row.get("task")
            if TASK_REQUESTED_LANG.get(tid):
                directed.setdefault(model, {})[tid] = lang
            else:
                picks.setdefault(model, {})[tid] = lang
    if not picks and not directed:
        return ""
    models = list(dict.fromkeys(list(picks) + list(directed)))
    tasks = sorted({t for m in picks.values() for t in m})
    langs = sorted({l for m in picks.values() for l in m.values()})
    if len(tasks) < 2 and not directed:
        return ""

    head = "".join(f'<th class="n">{_h(m)}</th>' for m in models)

    # One stacked bar per model instead of a language×model grid. The grid was mostly dashes
    # — thirteen rows, four columns, and the answer to "what does this model reach for" was
    # spread across all of them. A bar puts each model's disposition on one line, in order,
    # with the dominant language first and unmistakable.
    order = sorted(langs, key=lambda l: -sum(
        1 for m in models for t in tasks if picks[m].get(t) == l))
    bars = []
    for m in models:
        counts = [(l, sum(1 for t in tasks if picks[m].get(t) == l)) for l in order]
        counts = [(l, n) for l, n in counts if n]
        counts.sort(key=lambda p: -p[1])
        undet = len(tasks) - sum(n for _l, n in counts)
        segs = []
        for l, n in counts:
            pct = n / len(tasks) * 100
            # Every segment gets a label. A one-task segment is ~6% of the bar, far too narrow
            # for "powershell", so narrow segments fall back to a short code — readable at a
            # glance, where an unlabelled block was readable only by hovering it.
            label = _h(l) if pct >= 15 else _h(_LANG_SHORT.get(l, l[:3]))
            segs.append(f'<span class="lseg lang-{_h(l)}" style="width:{pct:.4f}%" '
                        f'title="{_h(l)}: {n} of {len(tasks)} ({pct:.0f}%)">{label}</span>')
        if undet:
            pct = undet / len(tasks) * 100
            segs.append(f'<span class="lseg lseg-none" style="width:{pct:.4f}%" '
                        f'title="no identifiable code: {undet} of {len(tasks)}">—</span>')
        # Colour-keyed legend: the swatch is what ties a narrow "ps1" back to powershell.
        chips = "".join(f'<span class="lchip"><i class="lang-{_h(l)}"></i>{_h(l)} {n}</span>'
                        for l, n in counts)
        if undet:
            chips += f'<span class="lchip"><i class="lseg-none"></i>no code {undet}</span>'
        bars.append(f'<div class="lrow"><div class="lname"><code class="mdl">{_h(m)}</code></div>'
                    f'<div class="lbar">{"".join(segs)}</div></div>'
                    f'<div class="lleg">{chips}</div>')
    prof_html = f'<div class="langbars">{"".join(bars)}</div>'

    prof = []
    for lang in order:
        cells = []
        for m in models:
            n = sum(1 for t in tasks if picks[m].get(t) == lang)
            share = n / len(tasks) * 100 if tasks else 0
            # Parenthesised, because "1 6%" and "7 44%" were unreadable — the count and the
            # share ran together into what looked like one number (16%, 744%).
            cells.append(f'<td class="n">{n or "—"}'
                         + (f' <span class="ct">({share:.0f}%)</span>' if n else "") + "</td>")
        prof.append(f'<tr><th scope="row"><code>{_h(lang)}</code></th>{"".join(cells)}</tr>')

    grid = []
    for t in tasks:
        cells = []
        vals = [picks[m].get(t) for m in models]
        disagree = len({v for v in vals if v}) > 1
        for v in vals:
            cells.append(f'<td><code>{_h(v or "—")}</code></td>')
        grid.append(f'<tr{" class=win" if disagree else ""}>'
                    f'<th scope="row"><code>{_h(t)}</code></th>'
                    f'<td>{_h(TASK_DESC.get(t) or "")}</td>{"".join(cells)}</tr>')

    # The one-line reading: a model that answers everything in one language has a reflex,
    # not a preference, and that is worth saying out loud.
    notes = []
    for m in models:
        counts: dict = {}
        for t in tasks:
            v = picks[m].get(t)
            if v:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        top, n = max(counts.items(), key=lambda kv: kv[1])
        if n / len(tasks) >= 0.5:
            notes.append(f'<b>{_h(m)}</b> answered {n} of {len(tasks)} tasks in '
                         f'{_h(top)} — including ones whose domain points elsewhere')
    note_html = (f'<p class="note">{" · ".join(notes)}.</p>' if notes else "")

    comply_html = ""
    if directed:
        dtasks = sorted({t for m in directed.values() for t in m})
        rows_out, totals = [], {m: [0, 0] for m in models}
        for t_id in dtasks:
            want = TASK_REQUESTED_LANG.get(t_id)
            cells = []
            for m in models:
                got = directed.get(m, {}).get(t_id)
                if got is None:
                    cells.append('<td class="n">—</td>')
                    continue
                ok = got == want
                totals[m][1] += 1
                totals[m][0] += 1 if ok else 0
                cells.append(f'<td class="n{"" if ok else " bad"}">'
                             + ("✓" if ok else f"<code>{_h(got)}</code>") + "</td>")
            rows_out.append(f'<tr><th scope="row"><code>{_h(t_id)}</code></th>'
                            f'<td><code>{_h(want)}</code></td>{"".join(cells)}</tr>')
        tot = "".join(
            f'<td class="n"><b>{totals[m][0]}/{totals[m][1]}</b></td>' for m in models)
        ignored = [f"{_h(m)} ignored it {totals[m][1] - totals[m][0]} time"
                   f"{'s' if totals[m][1] - totals[m][0] != 1 else ''}"
                   for m in models if totals[m][1] and totals[m][0] < totals[m][1]]
        comply_html = (
            '<p class="note">The other half of the suite names the language outright, and '
            'names one the free-choice answers suggest the model would not have picked — '
            'compliance is only a test when it costs something. A preference a model can '
            'set aside on request is a preference; one it cannot is a reflex.'
            + (" Here, " + ", ".join(ignored) + "." if ignored else
               " Every instruction was followed.") + '</p>'
            f'<div class="tbl"><table><thead><tr><th>Task</th><th>Asked for</th>{head}'
            f'</tr></thead><tbody>{"".join(rows_out)}'
            f'<tr><th scope="row">followed</th><td></td>{tot}</tr>'
            f'</tbody></table></div>')

    if not tasks:
        return f'<h2>Language preference</h2>{comply_html}'
    return ('<h2>Language preference</h2>'
            '<p class="note">None of these prompts names a language. What came back is the '
            'model\'s disposition: what it reaches for when the choice is left open. '
            'Nothing here is graded on whether the code runs.</p>'
            f'<p class="note">Each bar is one model across the {len(tasks)} free-choice '
            f'tasks, widest language first. <b>none</b> means no identifiable code came '
            f'back — a diagram or prose rather than a program.</p>'
            f'{prof_html}'
            '<details class="lnum"><summary>The same thing as numbers</summary>'
            f'<div class="tbl"><table><thead><tr><th>Language</th>{head}</tr></thead>'
            f'<tbody>{"".join(prof)}</tbody></table></div></details>'
            f'{note_html}'
            '<p class="note">Task by task — highlighted rows are where the models '
            'disagreed, which is where the choice was actually a judgement call.</p>'
            f'<div class="tbl"><table><thead><tr><th>Task</th><th>What it asks</th>'
            f'{head}</tr></thead><tbody>{"".join(grid)}</tbody></table></div>'
            + comply_html)


def _bench_coldstart_split_html(rows: list) -> str:
    """Where a cold start actually goes: booting the server, or the first request.

    Only rendered when at least one cell had a server to boot — for an all-Ollama field the
    split is meaningless (every millisecond is the warm-up) and the table would be a column
    of dashes next to a column of totals.
    """
    have = [r for r in rows if r.get("backend_start_ms")]
    if not have:
        return ""

    def nm(r):
        return (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]

    seen, trs = set(), []
    for r in sorted(rows, key=lambda r: -(r.get("load_ms") or 0)):
        name = nm(r)
        if name in seen or not r.get("load_ms"):
            continue
        seen.add(name)
        boot = (r.get("backend_start_ms") or 0) / 1000
        warm = (r.get("warmup_request_ms") or r.get("load_ms") or 0) / 1000
        total = (r.get("load_ms") or 0) / 1000
        trs.append(
            f'<tr data-m="{_h(name)}"><th scope="row"><code class="mdl">{_h(name)}</code></th>'
            f'<td class="n">{("%.0f s" % boot) if boot else "—"}</td>'
            f'<td class="n">{warm:.1f} s</td>'
            f'<td class="n"><b>{total:.0f} s</b></td></tr>')
    if not trs:
        return ""
    return ('<div class="tbl"><table><thead><tr><th>Model</th>'
            '<th class="n">Boot the server</th><th class="n">First request</th>'
            '<th class="n">Total cold start</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def _bench_place_labels(cands: list, est: float = 6.6) -> list:
    """Greedy label placement: sort by x; when a label would collide with one already on its
    row, bump it up a row. Several dots on one line cannot be labelled without this — the
    quadrant of 100%-correct models is exactly that shape."""
    placed, out = [], []
    for x, name in sorted(cands, key=lambda c: c[0]):
        w = est * len(name) + 12
        row = 0
        while any(r == row and x - w / 2 < ox + ow / 2 for ox, ow, r in placed):
            row += 1
        placed.append((x, w, row))
        out.append((x, row, name))
    return out


def _bench_size_by_model(rows: list) -> dict:
    """On-disk size per model, looked up across all of a model's cells: sizes are recorded on
    cold cells and the cell a chart wants is usually the cached one."""
    out: dict = {}
    for r in rows:
        if r.get("size_mb"):
            k = (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]
            out[k] = max(out.get(k, 0), r["size_mb"])
    return out


def _bench_best_per_model(rows: list) -> list:
    """One representative cell per model — highest quality, then speed — for charts where a
    dot per cell would drown the story in duplicates."""
    best: dict = {}
    for r in rows:
        if not r.get("decode_p50"):
            continue
        k = (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]
        key = ((r.get("perfect_rate") or 0), r["decode_p50"])
        if k not in best or key > ((best[k].get("perfect_rate") or 0), best[k]["decode_p50"]):
            best[k] = r
    return sorted(best.values(), key=lambda r: (-(r.get("perfect_rate") or 0),
                                                -(r.get("decode_p50") or 0)))


def _bench_bubbles_svg(rows: list, width: int = 820, height: int = 470) -> str:
    """What memory buys: position is footprint against correctness, bubble area is speed.

    Empty unless at least three models have a known size — a bubble chart of two points is a
    sentence, and vLLM checkpoints (inside containers) legitimately cannot be sized.
    """
    sizes = _bench_size_by_model(rows)
    bpm = [r for r in _bench_best_per_model(rows)
           if sizes.get((r.get("_name") or "").split(" · ")[0])
           and r.get("perfect_rate") is not None]
    if len(bpm) < 3:
        return ""
    import math as _m
    pad_l, pad_r, pad_t, pad_b = 46, 24, 72, 46
    x0, x1, y0, y1 = pad_l, width - pad_r, pad_t, height - pad_b
    gb_of = {id(r): sizes[(r.get("_name") or "").split(" · ")[0]] / 1024 for r in bpm}
    xmax = max(gb_of.values()) * 1.12
    smax = max(r["decode_p50"] for r in bpm)
    winner = bpm[0]

    def px(gb):
        return x0 + gb / xmax * (x1 - x0)

    def py(v):
        return y1 - v / 100 * (y1 - y0)

    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         'aria-label="Memory spent against correctness bought">']
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" '
                 'stroke="var(--grid)"/>')
        o.append(f'<text x="{x0-8}" y="{py(v)+4:.1f}" class="ct" text-anchor="end">{v}%</text>')
    for gb in (10, 20, 40, 60, 80):
        if gb < xmax:
            o.append(f'<text x="{px(gb):.1f}" y="{y1+16}" class="ct" '
                     f'text-anchor="middle">{gb}</text>')
    o.append(f'<text x="{x0}" y="{height-6}" class="ct">GIGABYTES THE MODEL OCCUPIES → '
             '<tspan fill="var(--ghost)">bubble area = output speed</tspan></text>')
    o.append(f'<text x="{x0}" y="{pad_t - 8}" class="ct" {_SVG_HALO}>'
             f'TASKS FULLY CORRECT ↑</text>')
    groups: dict = {}
    for r in sorted(bpm, key=lambda r: -gb_of[id(r)]):
        gb = gb_of[id(r)]
        qv = (r.get("perfect_rate") or 0) * 100
        rad = 5 + 17 * _m.sqrt(r["decode_p50"] / smax)
        col = ("var(--accent)" if r is winner
               else "var(--blue)" if qv == 100 else "var(--ghost)")
        nm = (r.get("_name") or "").split(" · ")[0]
        o.append(f'<circle cx="{px(gb):.1f}" cy="{py(qv):.1f}" r="{rad:.1f}" fill="{col}" '
                 f'data-m="{_h(nm)}" '
                 f'data-name="{_h(r.get("_name") or nm)}" '
                 f'opacity="{.92 if r is winner else .55}">'
                 f'<title>{_h(nm)} — {gb:.1f} GB, {qv:.0f}%, '
                 f'{r["decode_p50"]:,.1f} tok/s</title></circle>')
        if r is winner or gb > xmax * 0.6 or (qv == 100 and gb > xmax * 0.35):
            key = (round(gb), round(qv))
            g = groups.setdefault(key, {"x": px(gb), "y": py(qv) - rad - 7, "names": [],
                                        "win": False})
            g["names"].append(nm[:22])
            g["win"] = g["win"] or r is winner
    # Real-box collision, not row bumping: labels anchor to bubbles at different heights, so
    # "same row" is meaningless — devstral's label sat straight through the winner's until the
    # boxes themselves were compared.
    placed: list = []
    for x, base_y, text in sorted(
            ((g["x"], g["y"], " & ".join(sorted(set(g["names"])))
              + (" — the buy" if g["win"] else "")) for g in groups.values()),
            key=lambda c: c[0]):
        w = 6.2 * len(text)
        bx0 = x - w / 2
        for dy in (0, -15, 15, -30, 30):
            ly = max(14, base_y + dy)
            if not any(abs(ly - py_) < 13 and bx0 < p1 and p0 < bx0 + w
                       for p0, p1, py_ in placed):
                placed.append((bx0, bx0 + w, ly))
                o.append(f'<text x="{x:.0f}" y="{ly:.0f}" class="cl" {_SVG_HALO} '
                         f'text-anchor="middle">{_h(text)}</text>')
                break
    o.append("</svg>")
    return "".join(o)


def _bench_answer_time_svg(rows: list, width: int = 820, limit: int = 12) -> str:
    """Seconds until the complete answer has arrived — the wait a person actually feels,
    which tokens-per-second only implies. One bar per model, correctness riding along."""
    bpm = [r for r in _bench_best_per_model(rows) if r.get("total_p50")]
    if len(bpm) < 2:
        return ""
    shown = sorted(bpm, key=lambda r: r["total_p50"])[:limit]
    bar_h, gap, pad_l = 21, 8, 190
    height = len(shown) * (bar_h + gap) + 34 + (16 if len(bpm) > len(shown) else 0)
    top = max(r["total_p50"] for r in shown) / 1000
    graded = any(r.get("perfect_rate") is not None for r in shown)
    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         'aria-label="Seconds to a complete answer">']
    o.append(f'<text x="{pad_l}" y="12" class="ct">SECONDS UNTIL THE FULL ANSWER HAS '
             'ARRIVED (median task) →</text>')
    for i, r in enumerate(shown):
        y = 24 + i * (bar_h + gap)
        v = r["total_p50"] / 1000
        w = max(2, v / top * (width - pad_l - 96))
        q = r.get("perfect_rate")
        col = ("var(--accent)" if r is bpm[0]
               else "var(--ink)" if q == 1 else "var(--ghost)")
        nm = (r.get("_name") or "").split(" · ")[0]
        o.append(f'<text x="{pad_l-8}" y="{y+15}" class="cl" text-anchor="end">'
                 f'{_h(nm[:24])}</text>')
        o.append(f'<rect x="{pad_l}" y="{y}" width="{w:.0f}" height="{bar_h}" fill="{col}" '
                 f'opacity="{1 if col != "var(--ghost)" else .6}"/>')
        tail = f' <tspan fill="var(--ink-faint)">· {q * 100:.0f}%</tspan>' if graded and q is not None else ""
        o.append(f'<text x="{pad_l+w+7:.0f}" y="{y+15}" class="cv" {_SVG_HALO}>'
                 f'{v:.1f}s{tail}</text>')
    if len(bpm) > len(shown):
        o.append(f'<text x="{pad_l}" y="{height-5}" class="ct">top {len(shown)} of '
                 f'{len(bpm)} models — the full field is in the table</text>')
    o.append("</svg>")
    return "".join(o)


def _bench_engine_pair_data(rows: list, runs: list) -> list:
    """The same weights reachable through more than one engine, paired by cache state. This is
    the only controlled engine comparison a run can contain, and most runs contain none."""
    ident: dict = {}
    for r, run in zip(rows, runs):
        if not r.get("decode_p50"):
            continue
        up = (run.get("config") or {}).get("upstream") or ""
        key = (_bench_model_identity(r.get("model") or ""), r.get("cache") or "-",
               (run.get("config") or {}).get("concurrency") or 1)
        ident.setdefault(key, {})[up] = r
    return sorted((k, v) for k, v in ident.items() if len(v) > 1)


def _bench_engine_pairs_svg(pairs: list, width: int = 760) -> str:
    if not pairs:
        return ""
    row_h, pad_l = 44, 230
    height = len(pairs) * row_h + 58
    allv = [r.get("ttft_p50") or 0 for _k, v in pairs for r in v.values()]
    top = (max(allv) or 1) * 1.15
    ups = sorted({u for _k, v in pairs for u in v})
    colour = {u: ("var(--accent)" if i == 0 else "var(--blue)" if i == 1 else "var(--ghost)")
              for i, u in enumerate(ups)}

    def px(v):
        return pad_l + v / top * (width - pad_l - 90)

    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         'aria-label="Same model, two engines">']
    o.append(f'<text x="{pad_l}" y="14" class="ct">TIME TO FIRST TOKEN, MS →</text>')
    for i, (key, v) in enumerate(pairs):
        name, cache = key[0], key[1]
        conc = key[2] if len(key) > 2 else 1
        y = 36 + i * row_h
        tail = f"· {cache}" + (f" · {conc}×" if conc != 1 else "")
        o.append(f'<text x="{pad_l-10}" y="{y+4}" class="cl" text-anchor="end">'
                 f'{_h(name[:24])} <tspan fill="var(--ink-faint)">{_h(tail)}</tspan></text>')
        xs = {u: px(r.get("ttft_p50") or 0) for u, r in v.items()}
        if len(xs) > 1:
            a, b = min(xs.values()), max(xs.values())
            o.append(f'<line x1="{a:.0f}" y1="{y}" x2="{b:.0f}" y2="{y}" '
                     'stroke="var(--grid)" stroke-width="2"/>')
        xs_sorted = sorted(xs.items(), key=lambda kv: kv[1])
        crowded = len(xs_sorted) > 1 and xs_sorted[-1][1] - xs_sorted[0][1] < 46
        for idx, (u, x) in enumerate(xs_sorted):
            o.append(f'<circle cx="{x:.0f}" cy="{y}" r="6" fill="{colour[u]}" '
                     f'data-m="{_h(name)}">'
                     f'<title>{_h(u)}: {v[u].get("ttft_p50"):,.0f} ms</title></circle>')
            # Two dots nearly on top of each other put both centred values through each
            # other; push the values outward to the sides of the pair instead. A pair close
            # to the left edge has no left side — its value would print through the row
            # label — so that one drops below its dot instead.
            if crowded and idx in (0, len(xs_sorted) - 1):
                val_txt = f'{v[u].get("ttft_p50"):,.0f}'
                if idx == 0 and (x - 9 - 6.6 * len(val_txt)) < pad_l + 4:
                    anchor, tx, ty = "middle", x, y + 19
                else:
                    anchor = "end" if idx == 0 else "start"
                    tx = x - 9 if idx == 0 else x + 9
                    ty = y + 4
            else:
                anchor, tx, ty = "middle", x, y - 11
            o.append(f'<text x="{tx:.0f}" y="{ty}" class="cv" {_SVG_HALO} '
                     f'text-anchor="{anchor}">{v[u].get("ttft_p50"):,.0f}</text>')
    lx, ly = pad_l, height - 10
    for u in ups:
        o.append(f'<circle cx="{lx}" cy="{ly-4}" r="5" fill="{colour[u]}"/>')
        o.append(f'<text x="{lx+10}" y="{ly}" class="cl">{_h(u)}</text>')
        lx += 26 + 7 * len(u)
    o.append("</svg>")
    return "".join(o)


def _bench_coldstart_svg(rows: list, width: int = 820, limit: int = 10) -> str:
    """The switching tax as lollipops: seconds of loading before the first useful token."""
    warm = sorted((r for r in rows if r.get("warmup_ms")),
                  key=lambda r: -r["warmup_ms"])[:limit]
    if len(warm) < 2:
        return ""
    row_h, pad_l = 27, 230
    height = len(warm) * row_h + 36
    top = warm[0]["warmup_ms"] / 1000

    def px(v):
        return pad_l + v / top * (width - pad_l - 70)

    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         'aria-label="Seconds of loading before the first useful token">']
    o.append(f'<text x="{pad_l}" y="12" class="ct">SECONDS TO LOAD BEFORE THE FIRST '
             'USEFUL TOKEN →</text>')
    for i, r in enumerate(warm):
        y = 30 + i * row_h
        v = r["warmup_ms"] / 1000
        nm = r.get("_name") or _bench_label_display(r.get("label") or "")
        o.append(f'<text x="{pad_l-10}" y="{y+4}" class="cl" text-anchor="end">'
                 f'{_h(nm[:30])}</text>')
        o.append(f'<line x1="{pad_l}" y1="{y}" x2="{px(v):.0f}" y2="{y}" '
                 'stroke="var(--grid)" stroke-width="2"/>')
        o.append(f'<circle cx="{px(v):.0f}" cy="{y}" r="6" '
                 f'fill="{"var(--accent)" if i == 0 else "var(--blue)"}"/>')
        o.append(f'<text x="{px(v)+11:.0f}" y="{y+4}" class="cv">{v:,.0f}s</text>')
    o.append("</svg>")
    return "".join(o)


def _bench_scorecards(rows: list) -> str:
    """The briefing cards: the recommendation and the traps, computed from whatever this run
    actually contains — a different winner, no quality data, or no failures all render."""
    ok = [r for r in rows if r.get("decode_p50")]
    if len(ok) < 2:
        return ""
    graded = any(r.get("perfect_rate") is not None for r in ok)
    bpm = _bench_best_per_model(rows)
    sizes = _bench_size_by_model(rows)

    def nm(r):
        return (r.get("_name") or _bench_label_display(r.get("label") or "")).split(" · ")[0]

    # The recommendation IS the weighted ranking, not raw correctness: quality-first crowned
    # a model 1 point more correct at half the speed. Same data, same default weights, same
    # order as the Weighted standings section below — one verdict, stated twice.
    wdata = _bench_weighted_data(rows) if graded else []
    scores: dict = {}
    if wdata:
        ranked = _bench_weighted_rows(wdata, _BENCH_WEIGHT_DEFAULT / 100.0)
        scores = {e["m"]: sc for e, sc, _q, _s in ranked}
        by_name = {nm(r): r for r in bpm}
        ordered = [by_name[e["m"]] for e, _sc, _q, _s in ranked if e["m"] in by_name]
        bpm = ordered or bpm
    win = bpm[0]

    def gbtxt(r):
        gb = sizes.get(nm(r))
        return f" · {gb / 1024:.1f} GB" if gb else ""

    def dline(r):
        # One format for every card's stat line. The runner-up used to drop "answers in Ns"
        # and shorten "% correct" to "%", which read as different data rather than the same
        # measurements of a different model.
        q = (f'{(r.get("perfect_rate") or 0) * 100:.0f}% correct · ' if graded else "")
        t = (f' · answers in {r["total_p50"] / 1000:.1f}s' if r.get("total_p50") else "")
        s = (f' · score {scores[nm(r)] * 100:.0f}' if nm(r) in scores else "")
        return f'{q}{r["decode_p50"]:,.1f} tok/s{gbtxt(r)}{t}{s}'

    wtag = (f' — {100 - _BENCH_WEIGHT_DEFAULT}/{_BENCH_WEIGHT_DEFAULT} weighted'
            if scores else "")
    cards = []
    cards.append(f'<div class="card hi"><p class="k">Run this{wtag}</p>'
                 f'<p class="v">{_h(nm(win))}</p>'
                 f'<p class="d">{dline(win)}</p></div>')
    if len(bpm) > 1:
        ru = bpm[1]
        cards.append(f'<div class="card"><p class="k">Runner-up</p><p class="v">{_h(nm(ru))}</p>'
                     f'<p class="d">{dline(ru)}</p></div>')
    fastest = max(ok, key=lambda r: r["decode_p50"])
    if graded and fastest.get("perfect_rate") is not None             and fastest["decode_p50"] >= 2 * win["decode_p50"]             and fastest["perfect_rate"] < 0.6:
        cards.append(f'<div class="card"><p class="k">Don\'t be fooled by</p>'
                     f'<p class="v">{_h(nm(fastest))}</p>'
                     f'<p class="d">{fastest["decode_p50"]:,.0f} tok/s and only '
                     f'{fastest["perfect_rate"] * 100:.0f}% correct</p></div>')
    n_fail = len(rows) - len(ok)
    if n_fail:
        cards.append(f'<div class="card"><p class="k">Not measured</p>'
                     f'<p class="v">{n_fail} cell{"s" if n_fail > 1 else ""}</p>'
                     f'<p class="d">listed in the table as failures, not omitted</p></div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def _bench_pareto(points: list) -> set:
    """Indices not beaten on both axes at once.

    A configuration that is slower *and* less correct than another is strictly dominated: there
    is no reason to run it, whatever you value. That set is the shortlist, and it is the one
    thing a ranked bar chart cannot show — ranking one metric at a time hides the trade.
    """
    keep = set()
    for i, (_n, x, y) in enumerate(points):
        if not any(ox >= x and oy >= y and (ox > x or oy > y)
                   for j, (_m, ox, oy) in enumerate(points) if j != i):
            keep.add(i)
    return keep


def _bench_scatter_svg(rows: list, width: int = 840, height: int = 440) -> str:
    """Correctness against output rate, with the frontier drawn.

    The bar charts rank one metric at a time, so the actual question — which of these is worth
    running at all — takes three charts and a pencil. Here a dominated point is visibly below
    and to the left of something better, and the frontier is the shortlist.

    Log x because the range is genuinely two orders of magnitude on this hardware (5.7 tok/s to
    309); linear would put every serious model in the left tenth of the plot.
    """
    pts = [(r.get("_name") or _bench_label_display(r.get("label") or ""),
            float(r.get("decode_p50") or 0), (r.get("perfect_rate") or 0) * 100.0)
           for r in rows
           if r.get("decode_p50") and r.get("perfect_rate") is not None]
    if len(pts) < 3:
        return ""          # a scatter of two points is a table with extra steps

    import math
    # The right-hand gutter belongs to the winner's annotation; the plot never enters it.
    pad_l, pad_r, pad_t, pad_b = 46, 186, 20, 40
    x0, x1 = pad_l, width - pad_r
    y0, y1 = pad_t, height - pad_b
    lo = min(math.log10(max(p[1], 0.1)) for p in pts)
    hi = max(math.log10(max(p[1], 0.1)) for p in pts)
    if hi - lo < 0.05:
        lo, hi = lo - 0.25, hi + 0.25
    ylo = min(min(p[2] for p in pts), 100.0)
    ylo = 0.0 if ylo < 40 else 40.0        # zoom in when everything is good, never mislead

    def px(v):
        return x0 + (math.log10(max(v, 0.1)) - lo) / (hi - lo) * (x1 - x0)

    def py(v):
        return y1 - (v - ylo) / max(100.0 - ylo, 1) * (y1 - y0)

    front = _bench_pareto(pts)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="Correctness against output rate">']
    # Grid: horizontal only. Vertical lines on a log axis read as data.
    step = 20 if ylo == 0 else 10
    v = ylo
    while v <= 100.001:
        y = py(v)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'stroke="currentColor" stroke-opacity=".12"/>')
        out.append(f'<text x="{x0 - 7}" y="{y + 4:.1f}" class="ct" {_SVG_HALO} '
                   f'text-anchor="end">{v:.0f}%</text>')
        v += step
    for tick in (1, 3, 10, 30, 100, 300, 1000):
        if lo <= math.log10(tick) <= hi:
            out.append(f'<text x="{px(tick):.1f}" y="{y1 + 16}" class="ct" '
                       f'text-anchor="middle">{tick}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 6}" class="ct" '
               f'text-anchor="middle">output tokens/sec (log)</text>')
    out.append(f'<text x="{x1 - 6}" y="{y0 + 4}" class="ct" {_SVG_HALO} text-anchor="end" '
               f'fill="var(--accent)" fill-opacity=".8">better ↗</text>')
    out.append(f'<text x="{x0}" y="{y0 - 6}" class="ct" {_SVG_HALO}>'
               f'TASKS FULLY CORRECT ↑</text>')

    # The frontier as a staircase: from each frontier point you trade rate for correctness only
    # by stepping, never smoothly, so a curve would be a fiction.
    fp = sorted((pts[i] for i in front), key=lambda p: p[1])
    if len(fp) > 1:
        d = [f"M {px(fp[0][1]):.1f} {py(fp[0][2]):.1f}"]
        for a, b in zip(fp, fp[1:]):
            d.append(f"L {px(b[1]):.1f} {py(a[2]):.1f} L {px(b[1]):.1f} {py(b[2]):.1f}")
        out.append(f'<path d="{" ".join(d)}" fill="none" stroke="var(--accent)" '
                   f'stroke-opacity=".45" stroke-width="1.5" stroke-dasharray="4 3"/>')

    # The annotations are the findings, computed from this run rather than remembered from the
    # last one: a different winner, a different trap, or no trap at all each render correctly.
    def _clamp(v, a, b):
        return max(a, min(b, v))

    def _note(cx, cy, tx, ty, lines, colour="var(--ink)", anchor="start"):
        tx = _clamp(tx, pad_l + 4, width - 10)
        # Never below the plot: the x-axis tick row lives at y1+16, and an annotation clamped
        # into it printed "and wrong 95% of the time" straight through the "100" tick.
        ty = _clamp(ty, y0 + 12, y1 - 8 - 17 * (len(lines) - 1))
        edge = tx - 4 if anchor == "start" else tx + 4
        out.append(f'<path d="M {cx:.0f} {cy:.0f} L {edge:.0f} {ty - 5:.0f}" '
                   f'stroke="var(--ink-faint)" stroke-opacity=".55" fill="none" '
                   f'stroke-width="1"/>')
        for j, ln in enumerate(lines):
            w = "700" if j == 0 else "400"
            fill = colour if j == 0 else "var(--ink-faint)"
            out.append(f'<text x="{tx:.0f}" y="{ty + j * 17:.0f}" class="ann" {_SVG_HALO} '
                       f'text-anchor="{anchor}" font-weight="{w}" '
                       f'fill="{fill}">{_h(ln)}</text>')

    def _seg(nm):
        return nm.split(" · ")[0]

    # Decide which points get a full annotation BEFORE labelling: an annotated point also
    # carrying its frontier label printed its own name twice, once through the other.
    win_i = max(range(len(pts)), key=lambda i: (pts[i][2], pts[i][1]))
    wn, wx, wy = pts[win_i]
    sizes = _bench_size_by_model(rows)
    fast_i = max(range(len(pts)), key=lambda i: pts[i][1])
    fn, fx, fy = pts[fast_i]
    show_fast = fast_i != win_i and fx >= 2 * wx and fy < 60
    giants = [(i, sizes[_seg(pts[i][0])]) for i in range(len(pts))
              if _seg(pts[i][0]) in sizes]
    gi = ggb = None
    show_giant = False
    if giants:
        gi, ggb = max(giants, key=lambda t: t[1])
        gn, gx, gy = pts[gi]
        beats = [i for i, gb2 in giants
                 if gb2 <= ggb / 2 and pts[i][2] >= gy and pts[i][1] >= gx and i != gi]
        show_giant = gi not in (win_i, fast_i) and bool(beats)
    annotated = {win_i} | ({fast_i} if show_fast else set()) \
        | ({gi} if show_giant else set())

    labelled = 0
    placed_boxes: list = []       # (x0, x1, y) of every label already on the plot
    for i, (name, xv, yv) in enumerate(pts):
        on = i in front
        cx, cy = px(xv), py(yv)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{4.5 if on else 3}" '
                   f'data-m="{_h(_seg(name))}" data-name="{_h(name)}" '
                   f'fill="{"var(--accent)" if on else "currentColor"}" '
                   f'fill-opacity="{1 if on else .28}">'
                   f'<title>{_h(name)} — {yv:.0f}% correct at {xv:.1f} tok/s</title></circle>')
        # Only the frontier is named. Thirty-eight labels is a smear, and the dominated points
        # are precisely the ones nobody needs to identify.
        if on and i not in annotated and labelled < 8:
            anchor = "end" if cx > x1 - 90 else "start"
            dx = -7 if anchor == "end" else 7
            # First and last segment of the compound name: the model, and the axis value that
            # distinguishes this cell from its siblings. The middle is shared context.
            _segs = name.split(" · ")
            short = (_segs[0] if len(_segs) < 3 else f"{_segs[0]} · {_segs[-1]}")[:34]
            est_w = 6.6 * len(short)
            lx0 = cx + dx - (est_w if anchor == "end" else 0)
            # Frontier points cluster in the top band, so labels at dot height collide with
            # their neighbours. Try the dot row first, then rows above and below; a label
            # with no free row is dropped — its tooltip still identifies the point.
            for dy in (0, -13, 13, -26, 26):
                ly = cy + 3.5 + dy
                if y0 + 8 < ly < y1 - 2 and not any(
                        abs(ly - py_) < 12 and lx0 < px1 and px0 < lx0 + est_w
                        for px0, px1, py_ in placed_boxes):
                    placed_boxes.append((lx0, lx0 + est_w, ly))
                    labelled += 1
                    out.append(f'<text x="{cx + dx:.1f}" y="{ly:.1f}" class="cl" {_SVG_HALO} '
                               f'data-m="{_h(_seg(name))}" '
                               f'text-anchor="{anchor}">{_h(short)}</text>')
                    break

    win_lines = [_seg(wn), f"{wy:.0f}% correct at {wx:,.0f} tok/s"]
    wgb = sizes.get(_seg(wn))
    if wgb:
        win_lines.append(f"on {wgb / 1024:.1f} GB of memory")
    _note(px(wx) + 8, py(wy), x1 + 16, py(wy) + 5, win_lines, "var(--accent)")

    if show_fast:
        wrong = 1 - fy / 100
        _note(px(fx) - 8, py(fy) + 6, px(fx) - 16, py(fy) + 52,
              [_seg(fn), f"{fx / wx:,.0f}× the winner's speed —",
               f"and wrong {wrong * 100:.0f}% of the time"], "var(--ink)", "end")

    # The dominated giant: the largest sized model, if something at half its size or less
    # matches its correctness and beats its speed.
    if show_giant:
        small = min(sizes[_seg(pts[i][0])] for i in beats)
        _note(px(gx) - 6, py(gy) + 6, px(gx) - 14, py(gy) + 58,
              [f"{_seg(gn)} — {ggb / 1024:.0f} GB",
               "beaten on both axes by a model",
               f"{small / ggb:.0%} of its size"], "var(--ink)", "end")

    out.append("</svg>")
    return "".join(out)


def _bench_bar_svg(rows, key, label, unit, better="high", width=680, limit=12):
    """Horizontal bar chart as inline SVG — no script, no fonts, survives being saved to a file
    or printed. Charts exist to make the ordering obvious at a glance; the table carries the
    full field. Capped at the leaders: three charts of thirty-eight bars each was three
    thousand pixels of the table repeated, and rank 31 vs rank 34 is not a question anyone
    brings to a chart."""
    vals = [(r.get("_name") or _bench_label_display(r["label"]), r.get(key))
            for r in rows if r.get(key) is not None]
    if not vals:
        return ""
    total_n = len(vals)
    # Each chart ranks its own metric; the table's quality-first order is a different question.
    vals.sort(key=(lambda nv: -nv[1]) if better == "high" else (lambda nv: nv[1]))
    vals = vals[:limit]
    top = max(v for _n, v in vals) or 1
    best = max(v for _n, v in vals) if better == "high" else min(v for _n, v in vals)
    bar_h, gap, pad_l = 20, 8, 250
    height = len(vals) * (bar_h + gap) + 26 + (18 if total_n > len(vals) else 0)

    def elide(name, limit=42):
        """Elide the MIDDLE, not the tail. Cell labels share a long prefix (the model) and
        differ in their last few segments (the axis values), so tail-truncation renders every
        row of a sweep identically — which is the one thing a chart must not do."""
        if len(name) <= limit:
            return name
        head = limit // 3
        return name[:head] + "…" + name[-(limit - head - 1):]

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'role="img" aria-label="{_h(label)}">',
             f'<text x="0" y="12" class="ct">{_h(label)} ({_h(unit)})</text>']
    for i, (name, v) in enumerate(vals):
        y = 26 + i * (bar_h + gap)
        w = max(2, int((v / top) * (width - pad_l - 60)))
        fill = "#57d1e0" if v == best else "#33566b"
        _m = _h(name.split(" · ")[0])
        parts.append(f'<text x="0" y="{y + 14}" class="cl" data-m="{_m}">{_h(elide(name))}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y}" width="{w}" height="{bar_h}" rx="3" '
                     f'fill="{fill}" data-m="{_m}"/>')
        parts.append(f'<text x="{pad_l + w + 6}" y="{y + 14}" class="cv">{v:,.1f}</text>')
    if total_n > len(vals):
        parts.append(f'<text x="0" y="{height - 5}" class="ct">'
                     f'top {len(vals)} of {total_n} — the full field is in the table</text>')
    parts.append("</svg>")
    return "".join(parts)


def _h(v) -> str:
    """Escape for HTML text/attribute context."""
    return (str("" if v is None else v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


_BENCH_AXIS_LABELS = [
    ("model", "Model"), ("quant", "Quant"), ("size", "Size"), ("backend", "Backend"),
    ("think", "Think"), ("cache", "Cache"), ("prompt", "Prompt"), ("ctx", "Server ctx"),
    ("temp", "Temp"), ("conc", "Parallel"),
]


def _bench_axis_values(r: dict, cfg: dict) -> dict:
    """Everything about a cell that is a *setting* rather than a measurement."""
    return {
        # Identity, not display: two backends spelling the same checkpoint differently must
        # collapse, or "which engine is faster" is unanswerable from the table.
        "model": _bench_model_identity(r.get("served") or r.get("model") or "?"),
        "quant": r.get("quant"),
        "size": (f"{round(r['size_mb'] / 1024, 1)} GB" if r.get("size_mb") else None),
        "backend": cfg.get("upstream"),
        "think": r.get("thinking") or "auto",
        "cache": r.get("cache"),
        # `is not None`, not truthiness: depth 0 is the BASELINE of a long-context sweep, not
        # an absent setting. Treating it as absent left the axis with one distinct value, so
        # it was judged constant and no column was drawn — a 0-vs-32k comparison rendered as
        # four rows labelled only by model, with nothing saying which was which.
        "prompt": ("none" if (r.get("prompt_tokens") or 0) == 0
                   else f"{r['prompt_tokens']:,}") if r.get("prompt_tokens") is not None else None,
        "ctx": (f"{r['server_context']:,}" if r.get("server_context") else None),
        "temp": (str(r["temperature"]) if r.get("temperature") is not None else None),
        "conc": str(cfg.get("concurrency") or 1),
    }


def _bench_axis_split(rows: list[dict], runs: list[dict]) -> tuple[list, dict, list]:
    """Split a comparison's settings into what differs and what every cell shares.

    A comparison table should carry what varies and nothing else. This report used to print
    seventeen columns, five of which held the same value in every row, beside a Configuration
    column that restated most of the others — so the two numbers a reader actually wanted were
    somewhere off the right-hand edge. Constants are facts about the run as a whole and belong
    above the table, once.
    """
    vals = [_bench_axis_values(r, run.get("config") or {}) for r, run in zip(rows, runs)]
    if not vals:
        return [], {}, []
    keys = [k for k, _lbl in _BENCH_AXIS_LABELS]
    def spread(k):
        # Absent is not a value. A field only some backends report was being counted as an
        # axis, giving a column of dashes with two real entries in it.
        return {str(v[k]) for v in vals if v.get(k) not in (None, "")}
    varying = [k for k in keys if len(spread(k)) > 1]
    constant = {k: vals[0][k] for k in keys
                if k not in varying and vals[0].get(k) not in (None, "")}
    return varying, constant, vals


def _bench_cell_name(v: dict, varying: list) -> str:
    """Name a cell by what makes it different from the others, and nothing more."""
    bits = [str(v[k]) for k in varying if v.get(k) not in (None, "")]
    return " · ".join(bits) or str(v.get("model") or "run")


def _bench_case_parts(task: dict, idx: int) -> tuple:
    """(what-was-asked, expected) for one case — the vocabulary both the failure examples and
    the grading browser share. Function tasks render as a call; HTML/CSS as the structural
    check; SQL and bash by their inputs."""
    cases = task.get("cases") or []
    c = cases[idx] if idx < len(cases) else {}
    exp = json.dumps(c.get("expect"))
    if "args" in c:
        return f'{task["entry"]}({", ".join(json.dumps(a) for a in c["args"])})', exp
    if "op" in c:                       # HTML structural check
        what = {"count": f'count of "{c.get("sel")}"',
                "attr": f'{c.get("name")} of "{c.get("sel")}"',
                "text": f'text of "{c.get("sel")}"',
                "labels_bound": "labels bound to inputs"}.get(c["op"], c["op"])
        return what, exp
    if "prop" in c:                     # CSS declaration check
        scope = f' inside @media {c["media"]}' if c.get("media") else ""
        return f'{c.get("sel")} {{ {c["prop"]} }}{scope}', exp
    if "setup" in c:                    # SQL: the query ran against this case's data
        return "query result", exp
    if "stdin" in c:                    # bash: stdin in, stdout compared
        stdin_short = (c.get("stdin") or "").replace("\n", "⏎")[:48]
        return f'stdin "{stdin_short}"', exp
    if "check" in c:                    # agent episode: answer + conduct (+ memory)
        if c["check"] == "answer":
            return "final answer", exp
        if c["check"] == "memory":
            return "the memory left behind (inspected, not inferred)", exp
        if c["check"] == "needle":      # long-context recall: which fact, and how deep
            # Without this the case fell through to the conduct line below and a missed needle
            # was reported as the model making malformed tool calls, expecting null.
            depth = str(c.get("depth") or "")
            where = (f"at the {depth}" if depth in ("start", "end")
                     else f"{depth} of the way through" if depth else "in the haystack")
            return f'the {c.get("name")} code, planted {where}', json.dumps(c.get("code"))
        return "conduct (no malformed/hallucinated/repeated calls, within budget)", exp
    if "expect_any" in c:               # analysis answer: graded on what it names
        wanted = ", ".join(str(w) for w in (c.get("expect_any") or [])[:4])
        banned = c.get("expect_not") or []
        what = c.get("label") or "the answer must name it"
        return (f"{what} (text-graded)",
                f'says any of: {wanted}' + (f'; never: {", ".join(banned[:3])}'
                                            if banned else ""))
    return f"case {idx + 1}", exp


def _bench_case_text(task: dict, idx: int, got) -> str:
    """One failing case as a sentence: what was asked, what came back, what should have."""
    call, exp = _bench_case_parts(task, idx)
    gtxt = json.dumps(got) if not isinstance(got, str) or len(got) < 60 else json.dumps(got[:60] + "…")
    return f"{call} → {gtxt}, expected {exp}"


# ---- The grading browser: every request, every case, statically ------------------------------

_GRADES_CSS = """
  .greq { border:1px solid var(--border); border-radius:9px; background:var(--panel);
          padding:10px 14px; margin:8px 0; }
  .greq summary { cursor:pointer; font-size:13px; }
  .greq summary b.bad { color:var(--bad); }
  .greq summary b.ok { color:var(--good); }
  .gcase { list-style:none; margin:8px 0 2px; padding:0; }
  .gcase li { margin:3px 0; font-size:12.5px; font-family:var(--mono);
              overflow-wrap:anywhere; }
  .gcase .mark { display:inline-block; width:1.2em; }
  .gcase li.bad .mark { color:var(--bad); }
  .gcase li.ok .mark { color:var(--good); }
  .gcase li.bad { background:color-mix(in srgb, var(--bad) 7%, transparent);
                  border-radius:4px; padding:2px 6px; }
  .gresp pre { margin:6px 0 4px; padding:9px 12px; font-size:11.5px; line-height:1.45;
               background:var(--panel-2); border:1px solid var(--border); border-radius:6px;
               overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
  .gtoc { columns:2; column-gap:26px; margin:8px 0 20px; font-size:13px; }
  .gtoc a { color:var(--ink-dim); text-decoration:none; }
  .gtoc a:hover { text-decoration:underline; }
  .gtoc .fx { color:var(--bad); font-weight:600; }
  .gprompt pre { white-space:pre-wrap; font-size:12px; background:var(--panel-2);
                 padding:9px 12px; border-radius:6px; border:1px solid var(--border); }
  .gnote { border-left:3px solid var(--accent); background:color-mix(in srgb,
           var(--accent) 6%, var(--panel-2)); border-radius:0 6px 6px 0;
           padding:8px 12px; margin:6px 0 10px; font-size:12.5px; color:var(--ink-dim); }
  .gbuild { font-size:12px; color:var(--ink-dim); margin:6px 0 2px; }
  .gbuild code { overflow-wrap:anywhere; }
  .gbadge { display:inline-block; background:var(--bad); color:#fff; border-radius:4px;
            padding:1px 8px; font-family:var(--mono); font-size:10.5px; letter-spacing:.06em;
            margin-left:8px; vertical-align:1px; }
  .gbadge.warn { background:var(--amber, #9a6b16); }
  .gerr { border-left:3px solid var(--bad); background:color-mix(in srgb, var(--bad) 6%,
          var(--panel-2)); border-radius:0 6px 6px 0; padding:8px 12px; margin:8px 0; }
  .gerr b { color:var(--bad); font-family:var(--mono); font-size:11px; letter-spacing:.08em; }
  .gerr pre { margin:6px 0 2px; white-space:pre-wrap; overflow-wrap:anywhere;
              font-size:11.5px; line-height:1.5; }
"""


def _grade_error_kind(err: str) -> str:
    """Name the failure class so a compile error cannot hide inside small grey text."""
    low = err.lower()
    if low.startswith("compile error"):
        return "COMPILE ERROR"
    if "timeout" in low:
        return "TIMEOUT"
    if "syntaxerror" in low:
        return "SYNTAX ERROR"
    if "no code" in low:
        return "NO CODE BLOCK"
    return "GRADER ERROR"


def _bench_grades_html(run: dict) -> str:
    """One cell, fully accounted for: every request, its full graded response, and every case
    with what was asked, what was expected, and what came back. Static HTML — native
    <details> for collapsing, no scripts — so it can be saved, printed, and browsed offline.
    This page is the answer to "it scored 89% — what exactly is the other 11% and WHY?"."""
    cfg = run.get("config") or {}
    res = run.get("results") or {}
    rows = res.get("rows") or []
    suite_name = str(cfg.get("suite") or "")
    tasks = {t["id"]: t for t in (SUITES.get(suite_name) or [])}
    label = _bench_label_display(run.get("label") or run.get("model") or run.get("id") or "")

    by_task: dict = {}
    for i, rr in enumerate(rows):
        by_task.setdefault(rr.get("task") or "—", []).append((i, rr))

    def frac(items):
        tot = ok = 0
        for _i, rr in items:
            g = rr.get("grade") or {}
            tot += g.get("total") or 0
            ok += g.get("passed") or 0
        return ok, tot

    # Failures first: that is the question this page exists to answer.
    ordered = sorted(by_task.items(),
                     key=lambda kv: (frac(kv[1])[0] >= frac(kv[1])[1], kv[0]))

    toc = []
    sections = []
    for tid, items in ordered:
        ok, tot = frac(items)
        task = tasks.get(tid) or {}
        imperfect = ok < tot
        toc.append(f'<div><a href="#t-{_h(tid)}" class="{"fx" if imperfect else ""}">'
                   f'{_h(tid)}</a> <span style="color:var(--ink-faint)">{ok}/{tot}</span></div>')
        blocks = [f'<h2 id="t-{_h(tid)}">{_h(tid)}'
                  f' <span style="color:var(--ink-faint);font-size:13px;font-weight:400">— '
                  f'{_h(TASK_DESC.get(tid) or "")} · {ok}/{tot} cases</span></h2>']
        note = TASK_NOTES.get(tid)
        if note and imperfect:
            blocks.append(f'<p class="gnote"><b>Why this fails when it fails:</b> '
                          f'{_h(note)}</p>')
        if task.get("prompt"):
            blocks.append(f'<details class="gprompt"><summary>the prompt every run got'
                          f'</summary><pre>{_h(task["prompt"])}</pre></details>')
        # failed runs first inside the task, then by sequence
        items = sorted(items, key=lambda p: (
            (p[1].get("grade") or {}).get("passed", 0)
            >= (p[1].get("grade") or {}).get("total", 1), p[0]))
        for run_no, (i, rr) in enumerate(items, start=1):
            g = rr.get("grade") or {}
            passed, total = g.get("passed") or 0, g.get("total") or 0
            ok_run = passed >= total and total > 0
            badges = []
            err_blocks = []
            if g.get("truncated"):
                badges.append('<span class="gbadge warn">TOKEN CAP</span>')
            if rr.get("error"):
                badges.append('<span class="gbadge">REQUEST FAILED</span>')
                err_blocks.append(f'<div class="gerr"><b>REQUEST FAILED</b>'
                                  f'<pre>{_h(str(rr["error"]))}</pre></div>')
            if g.get("error"):
                kind = _grade_error_kind(str(g["error"]))
                badges.append(f'<span class="gbadge">{kind}</span>')
                # The FULL stored error — compiler output is the diagnosis, and truncating
                # it to a grey flag was how compile errors stayed invisible.
                build = g.get("build") or {}
                build_html = ""
                if build:
                    build_html = (
                        f'<div class="gbuild">compiled with <code>{_h(build.get("cmd") or "?")}'
                        f'</code>'
                        + (f' — {_h(build["compiler"])}' if build.get("compiler") else "")
                        + '<br><span style="color:var(--ink-faint)">the model saw only the '
                        'task prompt above; the harness below is the grader\'s wrapper, '
                        'added after the fact.</span></div>'
                        + (f'<details class="gresp"><summary>the full source as compiled '
                           f'(harness + model code)</summary>'
                           f'<pre>{_h(build.get("harness") or "")}</pre></details>'
                           if build.get("harness") else ""))
                err_blocks.append(f'<div class="gerr"><b>{kind}</b>'
                                  f'<pre>{_h(str(g["error"]))}</pre>{build_html}</div>')
            head = (f'<b class="{"ok" if ok_run else "bad"}">{passed}/{total}</b> · '
                    f'request #{i + 1}{"".join(badges)}')
            cases_html = ""
            gcases = g.get("cases") or []
            if gcases:
                lis = []
                for ci, cres in enumerate(gcases):
                    call, exp = _bench_case_parts(task, ci) if task else (f"case {ci+1}", "?")
                    cok = bool(cres.get("ok"))
                    if cok:
                        line = f'{_h(call)} → {_h(exp)}'
                    else:
                        got = cres.get("got")
                        gtxt = json.dumps(got) if not isinstance(got, str) or len(got) < 80                             else json.dumps(got[:80] + "…")
                        line = f'{_h(call)} → got {_h(gtxt)}, expected {_h(exp)}'
                    lis.append(f'<li class="{"ok" if cok else "bad"}">'
                               f'<span class="mark">{"✓" if cok else "✗"}</span>{line}</li>')
                cases_html = f'<ul class="gcase">{"".join(lis)}</ul>'
            resp_html = ""
            text = rr.get("text")
            if text:
                lang = task.get("lang") or "python"
                code = _bench_extract_code(text, lang)
                shown = code or text
                kind = "the code the grader extracted" if code else "raw reply (no code block)"
                resp_html = (f'<details class="gresp"><summary>{kind}</summary>'
                             f'<pre>{_h(shown[:6000])}</pre></details>')
            blocks.append(f'<details class="greq"{"" if ok_run else " open"}>'
                          f'<summary>{head}</summary>{"".join(err_blocks)}'
                          f'{cases_html}{resp_html}</details>')
        sections.append("".join(blocks))

    n_req = len(rows)
    ok_all, tot_all = frac([(i, r) for i, r in enumerate(rows)])
    _parent = run.get("parent_id")
    _nav = (f'<p style="margin:0 0 14px"><a href="/__proxy/api/bench/report?format=html'
            f'&ids={_h(_parent or run.get("id") or "")}">← back to the report</a>'
            + (f' · <a href="/__proxy/api/bench/runs/{_h(_parent)}/grades">all cells</a>'
               if _parent else "") + '</p>')
    body = (_nav + f'<style>{_GRADES_CSS}</style>'
            f'<p class="note">Every request this cell made, every case it was graded on, and '
            f'what came back — failures first, passing runs collapsed. Static page: save it, '
            f'print it, it keeps working.</p>'
            f'<div class="gtoc">{"".join(toc)}</div>'
            + "".join(sections))
    _tc = (run.get("env") or {}).get("toolchains") if isinstance(run.get("env"), dict) else None
    _tc_backfilled = not _tc
    if not _tc:
        try:
            _tc = toolchain_versions()
        except Exception:
            _tc = {}
    if _tc:
        body += ('<p class="note" style="margin-top:22px">Graded with: '
                 + " · ".join(f"{_h(k)} {_h(str(v))}" for k, v in sorted(_tc.items()))
                 + (" — a census of the host as of viewing; this run predates toolchain "
                    "recording." if _tc_backfilled else "")
                 + "</p>")
    return _report_page(
        title=f"Grades — {label}",
        eyebrow="AI Proxy · benchmark · grading browser",
        sub=f'run {run.get("id")} · suite {suite_name or "—"}',
        meta=[("Requests", n_req), ("Cases passed", f"{ok_all}/{tot_all}"),
              ("Model", label.split(" · ")[0])],
        body=body,
    )


def _bench_grades_index_html(parent: dict, children: list) -> str:
    """A parent sweep's doorway into per-cell grading pages. Static links only."""
    items = []
    for c in sorted(children, key=lambda c: ((c.get("label") or ""), c.get("id") or "")):
        res = (c.get("results") or {})
        s2 = res.get("summary") or {}
        q = (s2.get("quality") or {}).get("perfect_rate")
        qtxt = f'{q * 100:.0f}% fully correct' if q is not None else (c.get("status") or "")
        items.append(f'<li><a href="/__proxy/api/bench/runs/{_h(c.get("id") or "")}/grades">'
                     f'{_h(_bench_label_display(c.get("label") or c.get("id") or ""))}</a> '
                     f'<span style="color:var(--ink-faint)">— {_h(qtxt)}</span></li>')
    body = (f'<p style="margin:0 0 14px"><a href="/__proxy/api/bench/report?format=html'
            f'&ids={_h(parent.get("id") or "")}">← back to the report</a></p>'
            '<p class="note">Pick a cell to browse every request it made and how each was '
            'graded.</p><ul style="line-height:2">' + "".join(items) + "</ul>")
    return _report_page(
        title=f'Grades — {parent.get("id")}',
        eyebrow="AI Proxy · benchmark · grading browser",
        sub=f'{len(children)} cells',
        meta=[("Cells", len(children))],
        body=body,
    )


def _bench_failure_examples(runs: list, rows: list, per_task: int = 4) -> dict:
    """task id → concrete failures, one per configuration: the first failing case with what
    came back, or the grader's error (compile failure, timeout, no code block). This is the
    difference between "87%" and knowing the model writes ordinals like 111st."""
    suite_name = next((r.get("suite") for r in rows if r.get("suite")), None)
    tasks = {t["id"]: t for t in (SUITES.get(suite_name) or [])}
    if not tasks:
        return {}
    out: dict = {}
    seen: set = set()
    for run, row in zip(runs, rows):
        name = (row.get("_name") or _bench_label_display(row.get("label") or ""))
        for rr in ((run.get("results") or {}).get("rows") or []):
            t, g = rr.get("task"), rr.get("grade")
            if not t or not isinstance(g, dict) or t not in tasks:
                continue
            if (g.get("passed") or 0) >= (g.get("total") or 0):
                continue
            if (t, name) in seen:       # one example per configuration per task
                continue
            seen.add((t, name))
            if g.get("error"):
                detail = str(g["error"])[:180]
            else:
                detail = None
                for i, c in enumerate(g.get("cases") or []):
                    if not c.get("ok"):
                        detail = _bench_case_text(tasks[t], i, c.get("got"))
                        break
            if detail:
                if g.get("truncated"):
                    detail = f"hit the token cap mid-answer — {detail}"
                ex = {"cell": name, "detail": detail}
                # The code that produced the failure, exactly as the grader extracted it.
                # A wrong answer without the code is a verdict; with it, it's a diagnosis.
                lang = tasks[t].get("lang") or "python"
                code = _bench_extract_code(rr.get("text") or "", lang)
                if code:
                    # Stored replies are truncated at 4 KB, which can cut a closing fence;
                    # the extractor then falls back to the whole text, opening fence and
                    # all. The grader saw the untruncated reply, so only display needs this.
                    code = re.sub(r"^```[\w+]*[ \t]*\n", "", code)
                    ex["code"] = code[:1600] + ("\n… (truncated)" if len(code) > 1600 else "")
                elif rr.get("text"):
                    # No extractable block IS the failure mode for some responses; showing
                    # the raw reply excerpt is the only way to see what happened.
                    ex["raw"] = rr["text"][:600]
                out.setdefault(t, []).append(ex)
    # Most-missed first; a handful of examples per task is illustration, not a log dump.
    return {t: v[:per_task] for t, v in
            sorted(out.items(), key=lambda kv: -len(kv[1]))}


def _bench_longctx_html(runs: list) -> str:
    """The long-context view, rendered instead of guessing from the generic tables.

    A needle ladder is not a coding suite and does not read like one. The generic report calls
    this run "93% fully correct" because it counts two units that hit the output cap as
    failures — they are not failures, they are units where the model never finished answering,
    and the truth is 140 of 140. It also has no place to show the two things the metric exists
    to expose: where recall breaks as the prompt grows, and which DEPTH the lost facts were at.

    Rendered only when the run actually contains needle tasks, so nothing changes for every
    other suite.
    """
    units = []
    for r in runs:
        for u in ((r.get("results") or {}).get("rows") or []):
            if str(u.get("task") or "").startswith("longctx_"):
                units.append(u)
    if not units:
        return ""

    by_rung: dict = {}
    for u in units:
        by_rung.setdefault(u["task"], []).append(u)

    def size_of(task):
        z = task.replace("longctx_", "")
        try:
            return float(z[:-1]) * (1_000_000 if z.endswith("m") else 1000)
        except ValueError:
            return 0.0

    depths = ["start", "25%", "50%", "75%", "end"]
    rows_html = []
    dep_tot: dict = {d: [0, 0] for d in depths}
    kinds: dict = {}
    grand_ok = grand_n = grand_cut = 0

    for task in sorted(by_rung, key=size_of):
        us = by_rung[task]
        clean = [u for u in us if not (u.get("grade") or {}).get("truncated")]
        cut = len(us) - len(clean)
        ok = sum((u.get("grade") or {}).get("passed", 0) for u in clean)
        n = len(clean) * 5
        grand_ok += ok
        grand_n += n
        grand_cut += cut
        # Real prompt size as the upstream counted it, not the size we asked for.
        pts = sorted(u.get("prompt_tokens") or 0 for u in us) or [0]
        med_pt = pts[len(pts) // 2]
        # Prefill rate: tokens the model had to read, over the time before the first one came
        # back. The only throughput number that means anything on a prompt this size.
        rates = [(u.get("prompt_tokens") or 0) / ((u.get("ttft_ms") or 0) / 1000.0)
                 for u in us if (u.get("ttft_ms") or 0) > 0 and (u.get("prompt_tokens") or 0)]
        rate = sorted(rates)[len(rates) // 2] if rates else None
        per_depth = []
        for d in depths:
            hit = tot = 0
            for u in clean:
                for c in ((u.get("grade") or {}).get("cases") or []):
                    if str(c.get("label") or "").endswith("@ " + d):
                        tot += 1
                        hit += 1 if c.get("ok") else 0
            dep_tot[d][0] += hit
            dep_tot[d][1] += tot
            cls = "" if not tot else (" win" if hit == tot else (" bad" if hit * 2 < tot else ""))
            per_depth.append(f'<td class="n{cls}">{hit}/{tot}</td>' if tot else '<td class="n">—</td>')
        for u in clean:
            for c in ((u.get("grade") or {}).get("cases") or []):
                if c.get("ok"):
                    continue
                got = c.get("got") or ""
                k = ("misattributed to another needle"
                     if ("another needle" in got or "appears only as" in got)
                     else "reported as not found" if "not found" in got
                     else "absent from the reply")
                kinds[k] = kinds.get(k, 0) + 1
        pct = (ok / n) if n else None
        cls = "" if pct is None else (" win" if pct >= 0.999 else (" bad" if pct < 0.8 else ""))
        rate_cell = f'<td class="n">{rate:,.0f}</td>' if rate else '<td class="n">&mdash;</td>'
        cut_cell = f'<td class="n">{cut}</td>' if cut else '<td class="n">&mdash;</td>'
        rows_html.append(
            f'<tr><th scope="row"><code>{_h(task.replace("longctx_", ""))}</code></th>'
            f'<td class="n">{med_pt:,}</td>'
            f'<td class="n{cls}">{ok}/{n}</td>'
            + "".join(per_depth) + rate_cell + cut_cell + "</tr>")

    # class="n" on a numeric header, matching td.n — without it the header sits left while the
    # figures under it sit right, and every column reads as offset by one.
    dep_head = "".join(f'<th class="n">{_h(d)}</th>' for d in depths)
    dep_foot = "".join(
        f'<td class="n">{dep_tot[d][0]}/{dep_tot[d][1]}</td>' if dep_tot[d][1] else '<td class="n">—</td>'
        for d in depths)
    kinds_txt = " · ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))         or "none — every planted fact was recalled"
    cut_note = ""
    if grand_cut:
        cut_note = (f'<p class="note"><b>{grand_cut}</b> unit(s) excluded: the reply hit the '
                    f'output cap before it finished answering. A truncated reply is not a '
                    f'recall failure, and scoring it as one measures the token budget rather '
                    f'than the model. Reasoning length here grows with the size of the prompt, '
                    f'not the difficulty of the task.</p>')

    return f"""
    <section>
      <h2>Long context — what the window actually holds</h2>

      <p>A model's advertised context is the prompt it will <em>accept</em>. It says nothing
      about whether the model can still find anything in there. Those are different properties
      and they part company well before the advertised number: a backend will happily prefill
      a prompt the model cannot actually read, return a confident answer, and report success at
      every layer. Nothing in a coding or agentic suite can catch that, because every one of
      their prompts fits in a few thousand tokens.</p>

      <p>So this measures the second property directly. Five facts — short codes — are planted
      at fixed depths through a haystack of known size, and all five are asked for in a single
      request. The haystack is deliberately repetitive, enumerated ledger lines that all look
      alike, which is the hard case for binding a fact to a position: the number below is a
      lower bound, and a real document of the same size would score better.</p>

      <p>Each rung is run five times, and each repeat is a <em>different</em> draw — the filler
      shifts from the first line and the needles jitter around their depths. Repeating one
      identical prompt would measure the backend's prompt cache instead, which serves repeats
      two through five off a prefix it already holds and returns five matching answers that
      look like consistency.</p>

      <p><b>Read the depth columns before the totals.</b> The first needle sits near the
      <b>start</b> on every repeat, because a backend that front-truncates an over-long prompt
      discards the beginning first — so a healthy start column is the evidence that the window
      is genuinely allocated, and a failing one means the prompt was silently cut rather than
      badly recalled. When start holds and the middle columns fall, that is a retrieval limit
      in the model. When start falls, it is a configuration problem in the stack.</p>

      <p><b>Prefill tok/s</b> is the prompt size over the time before the first token returned.
      At these sizes it is the only throughput figure that means anything — the answer is five
      short lines, so decode speed is noise next to the cost of reading the haystack.</p>

      <div class="tbl"><table>
        <thead><tr><th>Rung</th><th class="n">Prompt tokens</th><th class="n">Recalled</th>{dep_head}
          <th class="n">Prefill tok/s</th><th class="n">Excluded</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
        <tfoot><tr><th scope="row">all</th><td class="n">&mdash;</td>
          <td class="n">{grand_ok}/{grand_n}</td>{dep_foot}
          <td class="n">&mdash;</td><td class="n">{grand_cut or '&mdash;'}</td></tr></tfoot>
      </table></div>
      {cut_note}
      <p class="note"><b>Failures by kind:</b> {_h(kinds_txt)}. These are not
      interchangeable. A model that writes NAME=MISSING has lost the fact and knows it — it can
      tell you it does not know. A model that answers with another needle's code has lost it and
      does not know, and will state the wrong answer with the same confidence as a right one.
      Both score zero; only one is safe to build on.</p>
    </section>"""


def _bench_report_html(runs: list[dict], rows: list[dict]) -> str:
    """Self-contained comparison report: environment, per-cell table, charts, quality breakdown.

    Deliberately one file with inline CSS/SVG and no external requests, so it can be saved,
    mailed, or printed to PDF and still render exactly the same a month later.
    """
    graded = any(r.get("perfect_rate") is not None for r in rows)
    env = next((r.get("env") for r in runs if r.get("env")), {}) or {}
    gpus = env.get("gpus") or []
    gpu_txt = ", ".join(
        f"{g.get('name') or 'GPU'}"
        + (f" · {round((g.get('mem_total_mb') or 0) / 1024)} GB" if g.get("mem_total_mb") else "")
        for g in gpus) or "not reported"
    when = runs[0].get("ts")
    when_txt = datetime.datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M") if when else "—"
    title = ((rows[0].get("served") or rows[0]["model"]) if len(rows) == 1
             else f"{len(rows)} configurations")

    def fmt(v, digits=0, suffix=""):
        return "—" if v is None else f"{v:,.{digits}f}{suffix}"

    def pct(v):
        return "—" if v is None else f"{v * 100:.0f}%"

    # Suites are not comparable, and every ranking below assumes they are ------------------
    #
    # A report handed both a 119-task full suite and a 24-task language-preference suite used
    # to rank their rows against each other. Language preference grades "did you pick a
    # defensible language", so a model scoring 24/24 there outranked one scoring 101/119 on
    # the real thing — and that false winner propagated into the headline, the standings, the
    # category winners, the scatter and the memory chart. Rank inside the largest suite only;
    # the other suites keep their own sections, which is what they were for.
    _by_suite: dict = {}
    for _r, _run in zip(rows, runs):
        _by_suite.setdefault(_r.get("suite") or "—", []).append((_r, _run))
    _primary = max(_by_suite, key=lambda s: (
        max((x[0].get("n_total") or 0) for x in _by_suite[s]), len(_by_suite[s])))
    _other_suites = [s for s in _by_suite if s != _primary]
    # The language section is about the language suite, so it keeps every run. Its own naming
    # only needs the model, which is the head of the label.
    _lang_pairs = [(run, (r.get("label") or r.get("model") or ""))
                   for s in _by_suite for (r, run) in _by_suite[s]]
    if _other_suites:
        rows = [r for (r, _run) in _by_suite[_primary]]
        runs = [run for (_r, run) in _by_suite[_primary]]

    # Table -------------------------------------------------------------------------------
    varying, constant, axis_vals = _bench_axis_split(rows, runs)
    # Best first. Run order is an implementation detail of the sweep, and a reader scanning
    # thirty-eight rows for the winner is doing work the report should have done.
    _order = sorted(range(len(rows)),
                    key=lambda i: (-(rows[i].get("perfect_rate") if rows[i].get("perfect_rate")
                                     is not None else -1),
                                   -(rows[i].get("decode_p50") or 0)))
    rows = [rows[i] for i in _order]
    runs = [runs[i] for i in _order]
    axis_vals = [axis_vals[i] for i in _order]
    axis_names = [_bench_cell_name(v, varying) for v in axis_vals]
    for _r, _nm in zip(rows, axis_names):
        _r["_name"] = _nm
    _ds_tasks: dict = {}
    labels = dict(_BENCH_AXIS_LABELS)
    # Built in the same order the body appends its cells, from the same two conditions. They
    # used to be assembled separately — head by slice-index, body by append — and had drifted:
    # the graded pair was inserted before "vs best" in the header and after it in every row, so
    # the ratio printed under "Fully correct" and each quality figure sat one column left of
    # its name.
    # Splitting the axes into columns is right for two or three and wrong for six: it cost
    # more width than the metrics had, and the reader lost the numbers off the right edge.
    # Past that, the compound name in one column reads better than six narrow ones.
    split_axes = 0 < len(varying) <= 3
    show_vs = len(rows) > 1
    # Only when something was actually measured, so a run that loaded nothing does not sprout
    # columns of dashes.
    show_load = any(r.get("load_ms") for r in rows)
    show_res = any(r.get("resident_mb") for r in rows)
    # A metric identical on every row is not a measurement, it is a property of the run. Reply
    # length is 166 tokens for all 38 cells here, because they answered the same suite.
    def _same(key):
        vals = {r.get(key) for r in rows if r.get(key) is not None}
        return len(vals) <= 1
    show_tokens = not _same("mean_tokens")
    # No separate Configuration column when the axes are shown: it restated them word for word.
    head = ((["Configuration"] if not split_axes else [labels[k] for k in varying])
            + (["Load"] if show_load else []) + (["Resident"] if show_res else [])
            + ["TTFT p50", "Decode p50"] + (["Tokens"] if show_tokens else []) + ["Total p50"]
            + (["Fully correct", "Cases"] if graded else [])
            + (["vs best"] if show_vs else [])
            + ["OK", ""])   # the trailing blank heads the grades link column
    # Slowdown against the fastest configuration in the set. A raw latency column doesn't make
    # "16x slower for no quality gain" jump out; a ratio does.
    fastest = min((r["total_p50"] for r in rows if r["total_p50"]), default=None)
    best_dec = max((r["decode_p50"] for r in rows if r["decode_p50"] is not None), default=None)
    best_ttft = min((r["ttft_p50"] for r in rows if r["ttft_p50"] is not None), default=None)
    best_q = max((r["perfect_rate"] for r in rows if r["perfect_rate"] is not None), default=None)

    body_rows = []
    for r, run, av, nm in zip(rows, runs, axis_vals, axis_names):
        cfg = run.get("config") or {}
        slow = (r["total_p50"] / fastest) if (fastest and r["total_p50"]) else None
        if split_axes:
            cells = [f'<th scope="row" class="ax">{_h(av.get(varying[0]) or "—")}</th>']
            _row_m = _h((r.get("_name") or nm or "").split(" · ")[0])
            cells += [f'<td class="ax">{_h(av.get(k) or "—")}</td>' for k in varying[1:]]
        else:
            cells = [f'<th scope="row" class="cfg">{_h(nm)}</th>']
            _row_m = _h((r.get("_name") or nm or "").split(" · ")[0])
        if show_load:
            cells.append('<td class="n">%s</td>' % (
                fmt((r.get("load_ms") or 0) / 1000, 0, " s") if r.get("load_ms") else "—"))
        if show_res:
            cells.append('<td class="n">%s</td>' % (
                fmt((r.get("resident_mb") or 0) / 1024, 1, " GB") if r.get("resident_mb") else "—"))
        cells += [
            f'<td class="n{" win" if r["ttft_p50"] == best_ttft else ""}">{fmt(r["ttft_p50"], 0, " ms")}</td>',
            f'<td class="n{" win" if r["decode_p50"] == best_dec else ""}">{fmt(r["decode_p50"], 1)}</td>',
        ]
        if show_tokens:
            cells.append(f'<td class="n">{fmt(r.get("mean_tokens"), 0)}</td>')
        cells.append(f'<td class="n">{fmt(r["total_p50"], 0, " ms")}</td>')
        if graded:
            cells.append(f'<td class="n{" win" if r["perfect_rate"] == best_q else ""}">{pct(r["perfect_rate"])}</td>')
            cells.append(f'<td class="n">{pct(r["case_pass_rate"])}</td>')
        if show_vs:
            cells.append(
                f'<td class="n{" slow" if (slow or 0) >= 2 else ""}">'
                f'{("1.0x" if slow and slow < 1.05 else fmt(slow, 1, "x")) if slow else "—"}</td>')
        ok = r["n_success"] == r["n_total"]
        cells.append(f'<td class="n {"ok" if ok else "bad"}">{r["n_success"]}/{r["n_total"]}</td>')
        # Deep link into the grading browser: every request and case behind this row's
        # numbers, statically rendered. Hidden in print, where a link is just clutter.
        cells.append(f'<td class="n ix-only"><a class="glink" '
                     f'href="/__proxy/api/bench/runs/{_h(r.get("id") or "")}/grades">'
                     f'grades</a></td>')
        body_rows.append(f'<tr data-m="{_row_m}">' + "".join(cells) + "</tr>")

    # One configuration has nothing to compare against: a chart with a single full-width bar
    # conveys no scale, "best in column" marks everything, and "vs best" is always 1.0x. The
    # only real variation in a single cell is run-to-run spread, so show that instead.
    single = len(rows) == 1
    scatter_html = ""          # filled by the multi-cell branch; referenced by its results
    cards_html = mem_html = engine_html = ""   # likewise; a single run has none of them
    if single:
        r0 = rows[0]

        def spread(label, dist, unit, digits=0):
            if not dist or dist.get("p50") is None:
                return ""
            cells = "".join(f'<td class="n">{fmt(dist.get(k), digits)}</td>'
                            for k in ("min", "p50", "p90", "max"))
            return f'<tr><th scope="row">{_h(label)} <span class="unit">{_h(unit)}</span></th>{cells}</tr>'

        charts = (
            f'<p class="note">A single configuration, so there is nothing to rank \u2014 what matters '
            f'here is consistency across the {r0["n_total"]} requests. A wide gap between p50 and max '
            f'means the number is not repeatable.</p>'
            '<table><thead><tr><th>Metric</th><th>min</th><th>p50</th><th>p90</th><th>max</th></tr></thead><tbody>'
            + spread("Time to first token", r0.get("ttft"), "ms")
            + spread("Decode rate", r0.get("decode"), "tokens/sec", 1)
            + spread("Total", r0.get("total"), "ms")
            + '</tbody></table>')
    else:
        _sc = _bench_scatter_svg(rows) if graded else ""
        scatter_html = ""
        if _sc:
            scatter_html = (
                '<h2>The trade-off</h2>'
                '<p class="note">Every configuration, placed by correctness and output rate. '
                'A point below and to the left of another is beaten on both counts at once, so '
                'the dashed frontier is the shortlist — everything off it is dominated by '
                'something on it. Hover any point for its name.</p>' + _sc)
        # Seconds to a complete answer replaces the tok/s and TTFT rankings: it is the
        # only speed figure in units a person feels, and correctness rides on each bar.
        charts = _bench_answer_time_svg(rows)
        cards_html = _bench_scorecards(rows)
        _mem = _bench_bubbles_svg(rows)
        mem_html = ((
            '<h2>What memory buys</h2>'
            '<p class="note">Position is footprint against correctness; bubble area is output '
            'speed. Models whose size cannot be read — vLLM checkpoints live inside their '
            'containers — are absent, not zero.</p>' + _mem) if _mem else "")
        _eng = _bench_engine_pairs_svg(_bench_engine_pair_data(rows, runs))
        engine_html = ((
            '<h2>Same weights, two engines</h2>'
            '<p class="note">The only controlled engine comparison a run can contain: '
            'identical weights reachable through more than one backend, paired by cache '
            'state. When output rates tie, the wait for the first token is what an engine '
            'buys.</p>' + _eng) if _eng else "")

    # Per-task quality --------------------------------------------------------------------
    task_html = ""
    if graded:
        tasks = {}
        for r, run, nm in zip(rows, runs, axis_names):
            q = (((run.get("results") or {}).get("summary") or {}).get("quality") or {})
            for t in (q.get("tasks") or []):
                tasks.setdefault(t["task"], {})[nm] = t.get("perfect_rate")
        if tasks:
            for _tid, _per in tasks.items():
                _ds_tasks[_tid] = {"desc": TASK_DESC.get(_tid) or "",
                                   "rates": sorted(_per.items(),
                                                   key=lambda kv: kv[1] if kv[1] is not None
                                                   else -1)}
            col_labels = list(axis_names)
            th = "".join(f"<th>{_h(l)}</th>" for l in col_labels)
            # With one configuration, a row per task is a column of identical 100%s. Only the
            # tasks that lost a case carry information, so list those and count the rest.
            items = sorted(tasks.items(), key=lambda kv: _task_sort_key(kv[0]))
            if len(rows) == 1:
                lab = col_labels[0]
                imperfect = [(t, per) for t, per in items if (per.get(lab) or 0) < 1]
                clean = len(items) - len(imperfect)
                if not imperfect:
                    task_summary = (f'<p class="note">All <b>{clean}</b> tasks fully correct on '
                                    "every run — nothing to single out.</p>")
                    trs = []
                    th = ""
                else:
                    task_summary = (f'<p class="note"><b>{clean}</b> of {len(items)} tasks were '
                                    "fully correct on every run; the ones that were not are "
                                    "listed below.</p>")
                    trs = [f'<tr class="taskrow" data-task="{_h(t)}">'
                           f'<th scope="row"><code>{_h(t)}</code>'
                           f'<span class="tdesc">{_h(TASK_DESC.get(t) or "")}</span></th>'
                           f'<td class="n">{pct(per.get(lab))}</td></tr>' for t, per in imperfect]
                    th = '<th>Task</th><th class="n">Perfect</th>'
            else:
                # One column per cell put 38 columns and a 60-character compound heading in
                # each; the table could not render, let alone be read. Cells are rows here and
                # everywhere else — columns are for metrics, and there are always few of those.
                #
                # Inverted as well as transposed: a task everything solved carries no
                # information, so the useful axis is which cells failed which task.
                task_summary = ""
                trs = []
                clean_tasks = []
                for tname, per in items:
                    failed = sorted(nm for nm in col_labels if (per.get(nm) or 0) < 1)
                    if not failed:
                        clean_tasks.append(tname)
                        continue
                    trs.append(
                        f'<tr class="taskrow" data-task="{_h(tname)}">'
                        f'<th scope="row"><code>{_h(tname)}</code>'
                        f'<span class="tdesc">{_h(TASK_DESC.get(tname) or "")}</span></th>'
                        f'<td class="n">{len(col_labels) - len(failed)} of {len(col_labels)}</td>'
                        f'<td class="fails">{_h(", ".join(failed))}</td></tr>')
                if clean_tasks:
                    task_summary = (
                        f'<p class="note"><b>{len(clean_tasks)}</b> of {len(items)} tasks were '
                        f"solved perfectly by every configuration and are not listed: "
                        f"<code>{_h(', '.join(clean_tasks))}</code>. A task nothing fails "
                        f"separates nothing.</p>")
                th = ('<th>Task</th><th class="n">Perfect in</th>'
                      '<th>Configurations that missed it</th>')
            tier_rows = ""
            if any(r.get("tiers") for r in rows):
                names = {"core": "Core", "hard": "Hard"}
                tiers_present = [t for t in ("core", "hard")
                                 if any((r.get("tiers") or {}).get(t) for r in rows)]

                def _tier_pr(r, t):
                    return ((r.get("tiers") or {}).get(t) or {}).get("perfect_rate")

                # Same inversion as the per-task table: a row reading 100% / 100% separates
                # nothing, and most rows here read exactly that.
                shown = [(r, nm) for r, nm in zip(rows, axis_names)
                         if any((_tier_pr(r, t) or 0) < 1 for t in tiers_present
                                if _tier_pr(r, t) is not None)]
                clean_n = len(rows) - len(shown)
                ttrs = []
                for r, nm in shown:
                    tds = "".join(f'<td class="n">{pct(_tier_pr(r, t))}</td>'
                                  for t in tiers_present)
                    ttrs.append(f'<tr><th scope="row" class="cfg">{_h(nm)}</th>{tds}</tr>')
                tier_head = "".join(f'<th class="n">{_h(names.get(t, t))}</th>'
                                    for t in tiers_present)
                clean_note = (f'<p class="note"><b>{clean_n}</b> configuration'
                              f'{"s" if clean_n != 1 else ""} cleared both tiers in full and '
                              f'are not listed.</p>') if clean_n else ""
                tier_rows = (clean_note + (
                    '<div class="tbl"><table><thead><tr><th>Configuration</th>'
                    + tier_head + '</tr></thead><tbody>'
                    + "".join(ttrs) + '</tbody></table></div>' if ttrs else ""))
            # Runs recorded before the hard tier existed carry no tier data; an empty heading
            # with a paragraph explaining a table that isn't there is worse than no section.
            tier_html = ("<h2>Correctness by tier</h2>"
                         '<p class="note">The core tier confirms a model is not broken; it '
                         "saturates for anything capable, which is exactly why the hard tier "
                         "exists. Compare two models on the hard row when both score 100% on "
                         "core.</p>" + tier_rows) if tier_rows else ""
            fx = _bench_failure_examples(runs, rows)
            fx_html = ""
            if fx:
                blocks = []
                for t, exs in fx.items():
                    desc = TASK_DESC.get(t) or ""
                    lis = ""
                    for e in exs:
                        codeblk = ""
                        if e.get("code"):
                            codeblk = (f'<details class="fxc"><summary>the code it wrote'
                                       f'</summary><pre>{_h(e["code"])}</pre></details>')
                        elif e.get("raw"):
                            codeblk = (f'<details class="fxc"><summary>no code block — raw '
                                       f'reply</summary><pre>{_h(e["raw"])}</pre></details>')
                        lis += (f'<li><b>{_h(e["cell"])}</b> — '
                                f'<code>{_h(e["detail"])}</code>{codeblk}</li>')
                    blocks.append(
                        f'<details class="fx"><summary><code>{_h(t)}</code>'
                        f'<span> — {_h(desc)} · {len(exs)} example'
                        f'{"s" if len(exs) > 1 else ""}</span></summary>'
                        f'<ul>{lis}</ul></details>')
                fx_html = ('<h3>What the failures actually looked like</h3>'
                           '<p class="note">The first failing case per configuration: the '
                           'call that was made, what came back, and what should have — or '
                           'the compile error or timeout that stopped it. This is what a '
                           'percentage point of correctness is made of.</p>'
                           + "".join(blocks))
            task_html = tier_html + f"""<h2>Per-task correctness</h2>
<p class="note">Share of responses that passed every case for that task. A model strong
everywhere except one task and a model mediocre throughout can share an overall average.</p>
{task_summary}
{f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>' if trs else ""}
{fx_html}"""

    # Cold vs cached, paired by everything except the cache axis. This is the comparison that
    # exposes a backend serving every repeated prompt as a fresh prefill.
    cache_html = ""
    pairs: dict = {}
    for r in rows:
        if not r.get("cache"):
            continue
        key = (r["model"], r["prompt_tokens"], r["thinking"], r.get("concurrency"))
        pairs.setdefault(key, {})[r["cache"]] = r
    both = {k: v for k, v in pairs.items() if "cold" in v and "cached" in v}
    if both:
        trs = []
        for (model, ctx, think, _cc), v in both.items():
            cold, cached = v["cold"]["ttft_p50"], v["cached"]["ttft_p50"]
            speedup = (cold / cached) if (cold and cached) else None
            verdict = ("cache is working" if speedup and speedup >= 1.5 else
                       "no measurable reuse" if speedup else "—")
            trs.append(
                f'<tr><th scope="row"><code class="mdl">{_h(_bench_model_display(model))}</code></th>'
                f'<td class="n">{fmt(ctx)}</td><td>{_h(think)}</td>'
                f'<td class="n">{fmt(cold, 0, " ms")}</td>'
                f'<td class="n">{fmt(cached, 0, " ms")}</td>'
                f'<td class="n{" win" if speedup and speedup >= 1.5 else ""}">{fmt(speedup, 1, "x") if speedup else "—"}</td>'
                f'<td>{_h(verdict)}</td></tr>')
        cache_html = f"""<h2>Prompt cache — cold vs cached cells</h2>
<p class="note">Cold sends a uniquely salted prompt every time so nothing can be reused; cached
repeats one identical prompt after a priming request. A backend whose prefix caching is off or
unsupported shows roughly the same first-token latency in both columns — which looks like
ordinary slowness rather than a misconfiguration.</p>
<table><thead><tr><th>Model</th><th class="n">Prompt</th><th>Think</th><th class="n">Cold TTFT</th>
<th>Cached TTFT</th><th>Speed-up</th><th></th></tr></thead><tbody>{"".join(trs)}</tbody></table>"""

    # The warm-up in cached mode sends the prompt the measured runs will send, so its TTFT is
    # the cold prefill of that exact prompt. Against the measured (cached) TTFT that is a direct
    # read on whether the prefix cache is working, and it costs nothing extra to report.
    cold_html = ""
    cold_rows = []
    for r in rows:
        cold_t, warm_t = r.get("warmup_ttft_ms"), r.get("ttft_p50")
        if r.get("cache") == "cached" and cold_t and warm_t and cold_t > warm_t * 1.5:
            cold_rows.append((r, cold_t, warm_t, cold_t / warm_t))
    if cold_rows:
        trs = "".join(
            f'<tr><th scope="row" class="cfg">{_h(r.get("_name") or r["label"])}</th>'
            f'<td class="n">{fmt(r["prompt_tokens"])}</td>'
            f'<td class="n">{fmt(c, 0, " ms")}</td><td class="n">{fmt(w, 0, " ms")}</td>'
            f'<td class="n win">{fmt(ratio, 0, "x")}</td></tr>'
            for r, c, w, ratio in cold_rows)
        cold_html = (
            "<h2>Prompt cache — warm-up vs measured</h2>"
            '<p class="note">The warm-up sends the same prompt the measured runs use, so its '
            "first-token time is that prompt\u2019s <em>cold</em> prefill; everything after it is "
            "served warm. A backend whose prefix caching is off or unsupported shows no gap "
            "between these two columns.</p>"
            "<table><thead><tr><th>Configuration</th><th>Prompt</th><th>Cold TTFT</th>"
            "<th>Cached TTFT</th><th>Faster by</th></tr></thead><tbody>" + trs + "</tbody></table>")

    _wvals = [(r.get("_name") or _bench_label_display(r["label"]),
               ((run.get("results") or {}).get("summary") or {}).get("warmup_ms"))
              for r, run in zip(rows, runs)
              if ((run.get("results") or {}).get("summary") or {}).get("warmup_ms")]
    if len(rows) == 1:
        warm = [fmt(v, 0, " ms") for _n, v in _wvals]
    elif _wvals:
        # Ranked, slowest first: the question this answers is which models are expensive to
        # make resident, and a comma-run of thirty-eight "name: ms" pairs answers nothing.
        _wvals.sort(key=lambda nv: -nv[1])
        _wtop = _wvals[:10]
        warm = ['<div class="tbl"><table><thead><tr><th>Configuration</th>'
                '<th class="n">Cold start</th></tr></thead><tbody>'
                + "".join(f'<tr><th scope="row" class="cfg">{_h(n)}</th>'
                          f'<td class="n">{fmt(v / 1000, 1, " s")}</td></tr>'
                          for n, v in _wtop)
                + '</tbody></table></div>'
                + (f'<p class="note">+ {len(_wvals) - len(_wtop)} more under '
                   f'{_wtop[-1][1] / 1000:.1f} s.</p>' if len(_wvals) > len(_wtop) else "")]
    else:
        warm = []

    if single:
        r0 = rows[0]
        cfg0 = (runs[0].get("config") or {})
        q0 = r0.get("perfect_rate")
        # The verdict first. A wide table asks you to reconstruct it from fifteen cells.
        verdict = (f"scored <b>{q0 * 100:.0f}%</b> fully correct" if q0 is not None
                   else f"completed <b>{r0['n_success']}</b> of {r0['n_total']} requests")
        results = f"""
  <div class="hero">
    <p class="lede"><b>{_h(r0["served"] or r0["model"])}</b> {verdict} at
      <b>{_fmt_n(r0["decode_p50"], 1)}</b> tok/s, first token in <b>{_fmt_n(r0["ttft_p50"], 0)}</b> ms.</p>
    <p class="why">{r0["n_success"]} of {r0["n_total"]} requests succeeded.</p>
  </div>
  <h2>Configuration</h2>
  <div class="spec">
    <div><p class="k">Backend</p><p class="v">{_h(cfg0.get("upstream") or "—")}</p></div>
    <div><p class="k">Context</p><p class="v">{_fmt_n(r0["prompt_tokens"]) if r0["prompt_tokens"] else "short"}</p></div>
    <div><p class="k">Thinking</p><p class="v">{_h(r0["thinking"] or "auto")}</p></div>
    <div><p class="k">Prompt cache</p><p class="v">{_h(r0.get("cache") or "—")}</p></div>
    <div><p class="k">Temperature</p><p class="v">{"—" if r0.get("temperature") is None else _h(str(r0["temperature"]))}</p></div>
    <div><p class="k">Quantisation</p><p class="v">{_h(r0.get("quant") or "not reported")}</p></div>
    <div><p class="k">Prefix caching</p><p class="v">{"on" if r0.get("prefix_caching") else ("off" if r0.get("prefix_caching") is False else "—")}</p></div>
    <div><p class="k">KV cache dtype</p><p class="v">{_h(r0.get("kv_cache_dtype") or "—")}</p></div>
    <div><p class="k">Decode rate</p><p class="v big">{_fmt_n(r0["decode_p50"], 1)}</p></div>
    <div><p class="k">Time to first token</p><p class="v big">{_fmt_n(r0["ttft_p50"], 0)} ms</p></div>
    <div><p class="k">Reply length</p><p class="v">{_fmt_n(r0.get("mean_tokens"), 0)} tok</p></div>
    <div><p class="k">Total per request</p><p class="v">{_fmt_n(r0["total_p50"], 0)} ms</p></div>
  </div>
"""
    else:
        # Everything every cell shares, stated once above the table instead of repeated down it.
        shared = "".join(
            f'<div><p class="k">{_h(labels[k])}</p><p class="v">{_h(constant[k])}</p></div>'
            for k in [k for k, _l in _BENCH_AXIS_LABELS] if k in constant)
        if not show_tokens and rows and rows[0].get("mean_tokens"):
            shared += (f'<div><p class="k">Reply length</p>'
                       f'<p class="v">{fmt(rows[0]["mean_tokens"], 0)} tok</p></div>')
        held = (f'<h2>Held constant</h2><div class="spec">{shared}</div>' if shared else "")
        # A sweep whose axes all collapsed measured one thing N times. Say so at the top rather
        # than leaving the reader to notice that every row matches.
        # Cells that could not get the memory they wanted. Named under the table rather than
        # marked in it: the point is that these particular numbers should not be compared with
        # the others, and a superscript does not say that loudly enough.
        starved = [(nm, r["memory_warning"]) for r, nm in zip(rows, axis_names)
                   if r.get("memory_warning")]
        starved_html = ""
        if starved:
            starved_html = (
                '<p class="note warnbox"><b>Measured under memory pressure.</b> '
                + "; ".join(f"{_h(nm)} — {_h(w)}" for nm, w in starved)
                + ". A model that does not fit is partly offloaded, so these figures describe "
                  "the machine as much as the model and are not comparable with the rest.</p>")
        inert = ""
        if not varying and len(rows) > 1:
            inert = ('<p class="note warnbox">Every cell in this comparison used identical '
                     'settings, so the rows below differ only by run-to-run noise.</p>')
        # The finding, before the evidence. A comparison whose answer is only recoverable by
        # scanning thirty-eight rows has buried it. Best quality first, and among equals the
        # quickest — the same order the table is now sorted in, so the lede names its top row.
        # (scatter_html was assigned in the charts block above; a stray reset here once
        # clobbered it and silently dropped the chart from every multi-cell report.)
        _lede = ""
        if len(rows) > 1 and rows[0].get("decode_p50"):
            _b, _bn = rows[0], axis_names[0]
            _q = _b.get("perfect_rate")
            _tied = [n for r2, n in zip(rows, axis_names)
                     if r2.get("perfect_rate") == _q and r2 is not _b]
            _lede = f"""
  <div class="hero">
    <p class="lede"><b>{_h(_bn)}</b> leads: {
      f'<b>{pct(_q)}</b> fully correct at ' if _q is not None else ''}<b>{
      _fmt_n(_b.get("decode_p50"), 1)}</b> tok/s out{
      f', {_fmt_n((_b.get("load_ms") or 0) / 1000, 0)} s to load' if _b.get("load_ms") else ''}.</p>
    <p class="why">{
      f'{len(_tied)} other configuration{"s" if len(_tied) != 1 else ""} scored the same and were slower. '
      if _tied else ''}Ranked by correctness first, then output rate. {len(rows)} configurations
      measured{f' across {len({v.get("backend") for v in axis_vals if v.get("backend")})} backends'
      if len({v.get("backend") for v in axis_vals if v.get("backend")}) > 1 else ''}.</p>
  </div>"""
        results = f"""
  {_lede}
  {cards_html}
  {scatter_html}
  {held}
  <h2>Results</h2>
  <p class="note">TTFT is the first token of any kind; TTFC the first <em>content</em> token —
  the gap between them is time the model spent reasoning. Decode rate is measured from the first
  token onward, so reasoning tokens count as generated work. Best value in each column is
  highlighted.</p>
  {inert}
  {starved_html}
  <div class="tbl"><table>
    <thead><tr>{"".join(f'<th class="{"n" if h.endswith(("p50", "best", "OK", "Cases", "correct")) else ""}">{_h(h)}</th>' for h in head)}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table></div>
"""

    # Method, last. A number is only interpretable if you know what produced it, and six
    # months from now nobody remembers what coding-v1 contained.
    method_html = ""
    _suite_name = rows[0].get("suite") if rows else None
    _suite = SUITES.get(_suite_name) if _suite_name else None
    if _suite:
        _skipped = (env.get("skipped_languages") or {}) if isinstance(env, dict) else {}
        _skip_html = ""
        if _skipped:
            _skip_html = ('<p class="note warnbox">Skipped on this machine — toolchain not '
                          'installed: ' + "; ".join(
                              f"<b>{_h(k)}</b> ({_h(', '.join(v))})"
                              for k, v in _skipped.items()) + ".</p>")
        _tiers: dict = {}
        for _t in _suite:
            _tiers.setdefault(_t.get("tier") or "untiered", []).append(_t)
        _blocks = "".join(
            f'<div class="tierblk"><p class="k">{_h(_tn.title())} — {len(_ts)} tasks, '
            f'{sum(len(t["cases"]) for t in _ts)} cases</p>'
            '<ul class="tl">' + "".join(
                f'<li><code>{_h(t["id"])}</code> <span>'
                f'{_h(TASK_DESC.get(t["id"]) or "")}'
                f'{" · " + _h(t["lang"]) if t.get("lang") and t["lang"] != "python" else ""}'
                f'</span></li>' for t in _ts) + '</ul></div>'
            for _tn, _ts in sorted(_tiers.items()))
        _cfg0 = (runs[0].get("config") or {}) if runs else {}
        # The graders themselves are part of the method: "graded with gcc 13" and "graded
        # with gcc 15" are different experiments. Recorded at run time; older runs backfill
        # from the host at render time and say so, same honesty rule as the hardware panel.
        _tc = env.get("toolchains")
        _tc_backfilled = not isinstance(_tc, dict)
        if _tc_backfilled:
            try:
                _tc = toolchain_versions()
            except Exception:
                _tc = {}
        _tc_html = ""
        if _tc:
            _tc_html = ('<div class="catalog"><div class="tierblk">'
                        '<p class="k">Graded with</p><ul class="tl">'
                        + "".join(f'<li><code>{_h(k)}</code> <span>{_h(str(v))}</span></li>'
                                  for k, v in sorted(_tc.items()))
                        + '</ul>'
                        + ('<p class="note" style="margin:6px 0 0">Versions are a census of '
                           'the host as of report time — this run predates toolchain '
                           'recording.</p>' if _tc_backfilled else "")
                        + '</div></div>')
        method_html = f"""
  <h2>What was tested</h2>
  <p class="note">Every task asks for one answer in the task's language — Python,
  JavaScript under node, C and C++ under gcc, Rust, C#, or PHP, each run in a separate process
  under a timeout with the return value compared against the expected one. HTML and CSS tasks
  are graded structurally: the answer is parsed and checked against required structure
  (bindings, attributes, declarations in the right context) — a claim about the markup, not
  about how a browser renders it. Tasks whose toolchain is absent on the machine are skipped
  and listed here, never scored as zero. <b>Fully correct</b>
  counts only responses where <em>every</em> case for that task passed; <b>cases</b> is the
  share of individual cases that passed, so a near-miss still scores there. A response with no
  extractable code block scores zero — that measures instruction-following, not coding.</p>
  <div class="spec">
    <div><p class="k">Suite</p><p class="v">{_h(_suite_name)}</p></div>
    <div><p class="k">Tasks</p><p class="v">{len(_suite)}</p></div>
    <div><p class="k">Cases</p><p class="v">{sum(len(t["cases"]) for t in _suite)}</p></div>
    <div><p class="k">Repeats</p><p class="v">{_h(str(_cfg0.get("runs") or 1))} per task</p></div>
    <div><p class="k">Languages</p><p class="v">{len({t.get("lang") or "python"
      for t in _suite})}</p></div>
  </div>
  <div class="catalog">{_blocks}</div>
  {_tc_html}
  {_skip_html}
  <ul class="fn">
    <li>Grading executes model-written code in a subprocess with a hard timeout, a scratch
      working directory and a stripped environment. That contains accidents and runaway loops;
      it is not a sandbox against deliberately hostile code.</li>
    <li>Requests go through this proxy so every one is logged and attributed, and carry
      <code>x-client-name: ai-proxy-bench</code>.</li>
  </ul>"""

    # The machine, fully described. Facts recorded at run time win; static facts a snapshot
    # missed (CPU model, kernel — they don't change) are filled from the host at render time
    # and say so, the same honesty rule as the GPU backfill before it.
    hw = _host_hw_facts()
    hw_backfilled = not isinstance(env.get("hw"), dict)
    if not hw_backfilled:
        hw = {**hw, **env["hw"]}
    _mem_gb = ((env.get("mem") or {}).get("total_mb") or 0) / 1024
    _hw_rows = [
        ("GPU", gpu_txt if gpu_txt != "not reported" else None),
        ("Unified memory", f"{_mem_gb:.0f} GB" if _mem_gb else None),
        ("CPU", hw.get("cpu_model")),
        ("Cores", hw.get("cpu_cores")),
        ("OS", hw.get("os")),
        ("Kernel", hw.get("kernel")),
        ("Ollama", env.get("ollama_version")),
        ("Proxy", env.get("proxy_version")),
    ]
    hw_html = ('<h2>Hardware</h2><div class="spec">'
               + "".join(f'<div><p class="k">{_h(k)}</p><p class="v">{_h(str(v))}</p></div>'
                         for k, v in _hw_rows if v is not None)
               + '</div>'
               + ('<p class="note">CPU, OS and kernel read from the host at report time — '
                  'this run predates their capture, and they do not change between runs.</p>'
                  if hw_backfilled else ""))

    weighted_html = _bench_weighted_html(rows) if graded else ""
    winners_html = _bench_category_winners_html(rows)
    # Only meaningful once a run spans more than one category — full-v1 does, the older
    # single-purpose suites do not, and the renderer returns "" for them.
    category_html = _bench_category_html(tasks, axis_names) if graded else ""
    failure_html = _bench_failure_taxonomy_html(runs, rows, axis_names) if graded else ""
    efficiency_html = _bench_efficiency_html(rows) if graded else ""
    langpref_html = _bench_language_profile_html([p[0] for p in _lang_pairs],
                                                 [p[1] for p in _lang_pairs])
    # Say what was set aside, so a reader counting runs in the URL against rows in the table
    # is not left wondering which ones went missing.
    if _other_suites:
        _n_other = sum(len(_by_suite[s]) for s in _other_suites)
        langpref_html = (
            f'<p class="note">Scores, standings and charts above describe '
            f'<b>{_h(_primary)}</b> only. {_n_other} further run'
            f'{"s" if _n_other != 1 else ""} from '
            f'{", ".join("<b>" + _h(s) + "</b>" for s in _other_suites)} '
            f'{"are" if _n_other != 1 else "is"} reported in their own sections rather than '
            f'ranked alongside — a 24-task preference suite and a 119-task correctness suite '
            f'produce numbers that do not mean the same thing.</p>') + langpref_html
    variance_html = _bench_variance_html(runs, rows, axis_names) if graded else ""
    # Placed directly under the results table rather than among the coding sections: for a
    # needle ladder this IS the result, and the generic tables above it are the supporting
    # detail. Empty for every other suite, so nothing else moves.
    longctx_html = _bench_longctx_html(runs)
    body = f"""
  {results}

  {longctx_html}

  {weighted_html}

  {category_html}

  {winners_html}

  {failure_html}

  {langpref_html}

  {efficiency_html}

  {variance_html}

  <h2>{"Consistency" if single else "Time to a finished answer"}</h2>
  {charts}

  {mem_html}

  {engine_html}

  {cache_html}

  {cold_html}

  {task_html}

  {hw_html}

  {method_html}

  {"<h2>Cold-start cost</h2><p class='note'>The price of making the model answerable, excluded from every measurement above: booting the server where one has to be booted (container start plus weight load) plus the discarded warm-up request. Counting only the warm-up understated a vLLM start twenty-fold — the weights are already in memory by the time that request arrives.</p>" + _bench_coldstart_split_html(rows) + _bench_coldstart_svg(rows) + (warm[0] if len(rows) > 1 else "<p class='note'>" + _h(warm[0]) + "</p>") if warm else ""}
"""
    _ds_rows = [{"name": (r.get("_name") or _bench_label_display(r.get("label") or "")),
                 "m": (r.get("_name") or _bench_label_display(r.get("label") or "")
                       ).split(" · ")[0],
                 "d": r.get("decode_p50"), "q": (r.get("perfect_rate") or 0) * 100
                 if r.get("perfect_rate") is not None else None,
                 "t": r.get("total_p50"), "c": r.get("concurrency"),
                 "g": round(r["size_mb"] / 1024, 1) if r.get("size_mb") else None,
                 "l": r.get("load_ms")}
                for r in rows if r.get("decode_p50")]
    _ds_json = json.dumps({"rows": _ds_rows, "tasks": _ds_tasks}).replace("</", "<\\/")
    _explore = ""
    if graded and len(_ds_rows) > 2 and _d3_source():
        _explore = (
            '<div class="ix-only"><h2>The trade-off — explore</h2>'
            '<p class="note">Screen-only companion to the annotated chart above: hover a dot '
            'for its numbers, drag a box to zoom into the crowded band, double-click to '
            'reset. Hovering a model anywhere on this page highlights it everywhere. The '
            'printed report keeps the annotated version.</p>'
            '<div id="ix-scatter"></div></div>')
    body += (
        _explore
        + f'<script id="report-data" type="application/json">{_ds_json}</script>'
        + (f"<script>{_d3_source()}</script>" if _d3_source() else "")
        + f"<script>{_REPORT_IX_JS}</script>")
    return _report_page(
        title=f"Benchmark — {title}",
        eyebrow="AI Proxy · benchmark",
        sub=f"{when_txt} · proxy {env.get('proxy_version') or ''}",
        meta=[("GPU", gpu_txt), ("Configurations", len(rows)),
              ("Requests", f"{sum(r['n_total'] or 0 for r in rows):,}"),
              ("Graded suite", rows[0].get("suite") if graded else None)],
        body=body,
    )
