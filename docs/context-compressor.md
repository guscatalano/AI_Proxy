# Future feature: Native context compressor + content-capture transparency

Status: **v1 shipped** (2026-07-19) — the `context_compressor` rule below is live: deterministic tool-output + JSON squeeze, shadow/live modes, savings at `/__proxy/api/compress-stats` and on the Stats page. v2/v3 (code-AST, before/after diff view, history compaction, prose summarization) remain future work.

## Motivation
- Local coding agents routinely blow past the context window (qwen3-coder-next, 256k) and waste tokens on verbose tool outputs, JSON blobs, file dumps, and accreting history.
- The proxy already sits in the exact choke point between every client and every model, and already captures every request/response — so it's the natural place to (a) **compress** what the model receives and (b) make the **before/after transformation fully visible**.
- Inspired by [`headroom-ai`](https://pypi.org/project/headroom-ai/) (Apache-2.0 context-compression tool: ~60–95% fewer tokens on JSON, 15–20% for coding agents). We want to **replicate the deterministic parts natively** (no heavy dependency) rather than take the dep.

## Two halves of the feature
1. **Compress** the outbound prompt (opt-in, configurable) to cut tokens / relieve context pressure.
2. **Capture & show** exactly what content was sent vs received — including the compression diff — so nothing is a black box.

## What headroom does, and how hard each part is to rebuild natively

| Piece | What it is | Native replication |
|---|---|---|
| Content routing (Magika) | Detect JSON / code / prose / logs to pick a compressor | **Easy** — heuristics on role, structure, path/extension. No Magika needed. |
| SmartCrusher (JSON) | Minify, dedupe keys/values, truncate huge arrays/strings with markers, drop nulls | **Easy–medium**, deterministic. Where the big JSON savings live. |
| Tool-output squeeze | Strip ANSI, head/tail-truncate long command output, collapse repeated lines | **Easy**, high value for coding agents. |
| CodeCompressor (AST) | Strip comments/blank lines; keep signatures, elide bodies | **Medium** — regex per language is quick; tree-sitter is better but more work. |
| History compaction | Compress/drop older turns | **Medium — hooks already exist** in the pipeline (prune / trim / ctx-cap / compaction-reminder). Extend, don't restart. |
| Kompress-v2 (neural prose) | Trained ONNX model that semantically compresses prose | **Hard — can't truly clone.** Options: skip, heuristic-truncate, or optional "summarize old turns with a local model" (adds an LLM call headroom avoids). |
| CCR reversal cache | Store originals; model fetches full context on demand | **Easy to store**, but **near-useless here** — local models won't call a retrieve tool. Our version is effectively one-way. Keep originals for the inspector instead. |
| Savings dashboard | Token before/after metrics | **Medium** — already track `est_prompt_tokens`; add compressed-vs-original + a Stats view. |

**Takeaway:** natively rebuild ~80% of the practical value (deterministic JSON + tool-output + code compressors, plus history compaction) with **zero new heavy deps, deterministic + fast, fully configurable**. The neural prose model is the only thing we can't clone; leave prose alone or offer optional local-model summarization.

## Proposed design (v1)
A new pipeline step beside the existing prune/trim, driven by the **rules config**, per-content-type with thresholds:

```yaml
compress:
  tool_outputs: { enabled: true,  max_chars: 4000, strategy: head_tail }
  json:         { enabled: true,  min_chars: 2000, drop_nulls: true, max_array: 50 }
  code:         { enabled: false, strip_comments: true, elide_bodies: false }
  history:      { enabled: false, keep_last_turns: 6 }
  prose:        { enabled: false, mode: none }   # none | truncate | summarize_local
```

- Applies **only where a rule opts in** (by model / client / `prompt_chars_gt`) — never global-by-default, because it's lossy.
- **Original always preserved** in `request_blobs`, so the inspector can show the true payload; only the *forwarded* copy is compressed.
- Records **original vs compressed tokens** per request; surfaces **tokens saved / % reduction** per request and aggregated in Stats.
- Ships with a **shadow/measure-only mode** so you can see the savings on real traffic *before* it changes any model behavior.

## Content-capture / transparency half
- A **before → after view** in the request detail: original payload, the compressed payload actually sent, and a diff/summary of what was dropped/shrunk per content block.
- Per-block annotations ("tool result: 18 KB → 3.9 KB, head/tail", "JSON array: 412 → 50 items").
- This makes the compression trustworthy and doubles as a general "what exactly did the model see vs what the client sent" inspector (useful even with compression off, since the proxy already rewrites/injects/prunes).

## Phasing
1. **v1:** JSON + tool-output compressors + config + savings readout + shadow mode. Deterministic, no deps. *(shippable first slice)*
2. **v2:** code-AST compressor, before/after diff view.
3. **v3:** history compaction integration; optional local-model prose summarization behind a flag.

## Open decisions
- Live vs shadow-first (recommend shadow-first).
- Whether to ever touch prose (recommend: no, or opt-in local summarize only).
- tree-sitter vs regex for code.

## References
- headroom-ai (PyPI): https://pypi.org/project/headroom-ai/ — `from headroom import compress`
- Repo: https://github.com/headroomlabs-ai/headroom
- Techniques to mirror: SmartCrusher (JSON), CodeCompressor (AST), Kompress-v2-base (neural prose — not replicated), CCR reversal.
