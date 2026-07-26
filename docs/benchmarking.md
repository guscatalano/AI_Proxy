# Benchmarking models through the proxy

The Bench tab measures how fast a model is **and** whether it is still correct, across as many
models, context sizes, thinking modes and temperatures as you want to compare. Everything runs
through the proxy, so every bench request is logged and inspectable like any other traffic.

## Why measurement is not obvious

### Reasoning models break naive TTFT

A thinking model emits its reasoning as `reasoning_content` deltas *before* any `content`
arrives. If you time "first token" by watching only `content` — the obvious implementation —
you report time-to-**end**-of-reasoning as TTFT and drop every reasoning token from the decode
rate. The model looks far slower to first token and slower to generate than it is.

The runner records both boundaries:

| Metric | Meaning |
|---|---|
| **TTFT** | first token of *any* kind — the real prefill latency |
| **TTFC** | first *content* token — when the user starts seeing the answer |
| **Reasoning phase** | TTFC − TTFT, plus the reasoning token count when the engine reports it |

For a non-thinking model the two collapse and the reasoning rows are hidden.

### Thinking is disabled three different ways

There is no portable switch. Each engine has its own, and the wrong one is silently ignored:

| Mechanism | Works on | Notes |
|---|---|---|
| `chat_template_kwargs.enable_thinking: false` | Qwen3-lineage on vLLM | These models ignore OpenAI's `reasoning_effort` entirely |
| `reasoning_effort: "none"` | ds4 / DwarfStar | Its knob is binary — `low`/`medium`/`high` all mean "think" |
| Empty `<think></think>` assistant prefill | LM Studio / llama.cpp | Drops `chat_template_kwargs` on the floor, so pre-closing the block in the model's mouth is the only lever |

The `thinking` setting maps onto these:

- `auto` — change nothing, so the proxy's own per-model quirks apply. This is what production
  traffic actually sees.
- `off` — sets both `enable_thinking: false` and `reasoning_effort: "none"`.
- `off_prefill` — as `off`, plus the pre-closed `<think>` block. Use for LM Studio.
- `on` — `enable_thinking: true` and `reasoning_effort: "high"`.

### Picking a model does not load it

The picker lists every model the proxy can reach, grouped by the backend that serves it and
marked ● loaded / ○ not loaded. Selecting one only names it in the request — what happens next
depends on the backend:

- **Ollama** pulls an unloaded model into VRAM on first use. Without a warm-up, that load time —
  tens of seconds for a large model — lands inside the first measured request while the rest run
  warm, which makes the run's min/max meaningless.
- **LM Studio and vLLM** don't auto-load. They only serve what is already resident, so an
  unlisted model simply errors.

**Warm up first** (on by default) sends one throwaway request before measuring. It is excluded
from every statistic, but reported separately as `warmup_ms` — if the warm-up took 40 s and the
measured runs took 2 s, that gap *is* the model-load cost. Turn it off only when cold start is
what you want to measure.

### The backend is picked with the model, not separately

There is no "which backend" field to keep in sync with your model choice. The backend picker is a
filter over the model list, showing how much of each backend is resident (`ollama 1/14 loaded`),
and each selected model carries its own backend into the run. That's what lets a single sweep mix
an Ollama model and a vLLM model — each cell routes itself.

Models are identified by the **(backend, model) pair**, so the same name served by two backends is
two distinct, separately benchmarkable entries. Filtering is a view, not a reset: selections on
other backends stay in the sweep, and the Models label says how many are selected and how many are
currently hidden.

### Routing silently substitutes the model

A `model_router` rule can rewrite the model *and* the upstream. Benchmark `qwen` with a rule in
place and you may actually measure a different model on a different backend, with the result
filed under the name you asked for. Two controls prevent this, both on by default in the UI:

- **Pin model** (`bypass_router`) — sends `x-proxy-no-router`, so the request goes to the model
  you named.
- **No system nudge** (`no_nudge`) — sends `x-proxy-no-nudge`, suppressing quirk-injected system
  prompt text. That injection is a real confound when you are measuring the model rather than
  using it.

Either way, the summary reports `served_models` — the model name the upstream echoed back — so
you can always see what actually answered.

### An empty completion is a failure, not a fast success

A prompt that overflows the model's context window typically returns HTTP 200 with zero tokens,
which naive timing records as an extremely fast run. Any run producing zero completion tokens is
marked as an error.

Related trap: prompt sizing is estimated at ~3.5 chars/token, but tokenizer density varies by
roughly 15% between model families. A prompt that fits one model's window can overflow another's.
The summary reports the upstream's own `prompt_tokens` so you can see what was really sent.

## Simple mode

The Bench tab opens in **Simple** mode: pick the models, choose a preset, press run.

| Preset | Shape | Cost per model |
|---|---|---|
| **Quick** | 32K context, reasoning off, cache warm | 12 tasks × 3 runs = **36 requests** |
| **Full report** | 3 contexts (short/8K/32K) × reasoning off·on × cold·cached × 1·4 parallel | 24 cells × 36 = **864 requests** |

Both grade on `coding-v1` at temperature 0, with a warm-up, routing pinned, and **other proxy
clients held** for the duration.

Holding is on by default because a benchmark sharing the model with live traffic measures the
traffic as much as the model. Other clients keep their connections and resume as soon as the run
finishes — they wait, they don't fail — and the gate is released between cells of a sweep, so a
long run leaves gaps rather than blocking solid for hours. Uncheck **Hold other proxy clients**
if you'd rather not delay anyone; the run bar says which mode you're in either way.

Freeing the GPU (unloading Ollama models) stays opt-in: it evicts models other people may be
mid-conversation with, which is a bigger imposition than waiting.

Quick ranks models on speed and correctness. **Full report** is the shape of a real
investigation — it answers what context depth costs, what reasoning costs, whether the prompt
cache is doing anything, and how throughput holds up under load, in one submission instead of a
dozen collated by hand. The form shows the request count before you start; a full sweep is a
long job, not a click.

The presets are fixed on purpose: a report is only worth comparing against another report if both
were produced the same way, and choosing sizes by hand each time guarantees they weren't.

Two things they deliberately leave off: holding other clients, and freeing the GPU. Both disrupt
everyone else using the box, which is not a reasonable default for a button labelled "run".

**Advanced** exposes every control described below. The choice is remembered.

## The cache axis

`cold` salts every prompt so nothing can be served from the prefix cache; `cached` repeats one
identical prompt after a priming request. Reported as a pair, with the speed-up and a verdict.

This exists because a backend whose prefix caching is off doesn't look broken — it looks *slow*.
A single number can't tell "this engine is slower" apart from "this engine re-prefills every
repeated prompt", and those have very different fixes. Comparing cold against cached separates
them: a working cache shows a large gap, a disabled one shows none.

In cached mode the warm-up sends the prompt the measured runs will send. A short throwaway would
prime nothing, and the first "cached" request would actually be a cold prefill — exactly the
confound the axis exists to expose.

## Reports

A report with **one configuration** drops the comparison furniture — no charts (a single bar
conveys no scale), no "best in column" highlighting, no "vs best" ratio — and shows the
min/p50/p90/max spread instead, since run-to-run consistency is the only real variation a single
cell contains.

When a cached run has a warm-up, the report also derives **cold vs cached** from it for free: the
warm-up sends the prompt the measured runs will send, so its first-token time is that prompt's
cold prefill. On spark that reads 14,200 ms cold against 134 ms cached — a 106× gap, and direct
evidence the prefix cache is working.

Any finished run has a **Report ↗** link, and a comparison of several runs has **Open report ↗**.
It renders a standalone HTML page: environment, a per-configuration table with the best value in
each column highlighted, bar charts for decode rate / TTFT / correctness, and a per-task
correctness breakdown.

The page carries its own CSS and inline SVG and makes no external requests, so it can be saved,
mailed, or archived and still render identically later. It has print styles — use the browser's
**Print → Save as PDF** for a PDF.

Same data at `GET /__proxy/api/bench/report?ids=<comma-separated>&format=html`.

## Matrix runs

Any of **models**, **prompt size**, **thinking**, **temperature**, **cache** and **concurrency**
can take several values, and the submission expands into one child run per combination — 3 models
× 2 context sizes × 2 thinking modes is 12 cells. The form shows the cell and request count before you start.

Cells run **one at a time**. Running them concurrently would have them contend for the same GPU
and corrupt every number in the sweep.

Each cell is a full run with its own percentiles, and the parent aggregates them. Deleting the
parent deletes its cells.

## Graded suites

Selecting a suite replaces the synthetic prompt with real tasks. Each asks for one named Python
function and carries deterministic input → expected cases. The model's code block is extracted
and executed, and two rates are reported:

- **Fully correct** — the fraction of responses that passed *every* case. This is the headline
  number: partially-right code is still broken code.
- **Cases passed** — partial credit, useful for telling "close" from "hopeless".

A per-task breakdown comes with it, because a model that is perfect everywhere except one task
and a model that is mediocre throughout can share an identical average.

> **Security.** Grading executes model-generated Python. It runs in a separate process with a
> hard timeout, an isolated interpreter (`-I`), a scratch working directory and a stripped
> environment. That contains accidents and runaway loops — it is **not** a sandbox against
> deliberately hostile code. Grading is opt-in per run, and the UI confirms before starting.

### Tiers

`coding-v1` has two tiers, both graded in the same run:

- **core** (12 tasks) — any usable coding model clears these. It confirms a model isn't broken,
  and saturates at 100% for anything capable, so it cannot rank two models that both pass.
- **hard** (6 tasks) — edit distance, longest increasing subsequence, Unix path canonicalisation,
  an expression evaluator with integer division truncating toward zero, word break, and a
  topological sort with a lexicographic tie-break. Each has an edge case a plausible-looking
  implementation gets wrong, so this is the tier that separates models.

Read the hard row when comparing two models that both score 100% on core.

`coding-v1` ships with 12 core tasks / 49 cases: binary search, interval merging, word frequency,
Roman numerals, bracket balancing, list flattening, two-sum, an LRU cache simulation, anagram
grouping, run-length encoding, version comparison, and spiral matrix traversal. Every task is
verified solvable by a reference implementation — a task its own reference can't pass would cap
every model's score forever.

## Getting clean numbers

Competing traffic ruins timings. Two levels of isolation:

- **Exclusive mode** — other proxy clients queue for the duration. Bench requests carry
  `x-client-name: ai-proxy-bench`, which is also what panic mode whitelists.
- **Quiesce GPU** — additionally enables panic mode and unloads every loaded Ollama model, then
  restores them afterward. Exclusive mode alone only gates traffic *through the proxy*; Ollama
  still answers anything that asks it directly on its own port, and an idle loaded model still
  holds VRAM.

Also relevant:

- **Randomize prompt** defeats the prompt cache. Without it, repeated identical prompts hit the
  KV cache and report a prefill time that has nothing to do with a cold request.
- **Drain** waits before starting so in-flight requests finish first.
- The exclusive-mode safety cap scales with the workload, so a long sweep can't have the gate
  expire underneath it and let traffic back in halfway through.

## Reproducibility

Every run stores an environment snapshot: proxy version, GPU model / utilization / VRAM, system
memory, loaded Ollama models and server config, and the model lists from LM Studio and vLLM.
Without it, a number from three weeks ago is uninterpretable — you can't tell which quant was
loaded or whether the GPU was already half-full when the run started.

## Comparing and exporting

Select any number of runs to compare. Two produce a delta table; more produce a ranked table with
the best value in each column highlighted. **Copy Markdown** puts a pasteable table on the
clipboard; a suite parent expands to one row per cell.

The same data is available at `GET /__proxy/api/bench/report?ids=<comma-separated>&format=markdown`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/__proxy/api/bench/run` | POST | Queue a run or a matrix |
| `/__proxy/api/bench/runs` | GET | History |
| `/__proxy/api/bench/runs/{id}` | GET | Full detail, incl. suite children |
| `/__proxy/api/bench/runs/{id}` | DELETE | Delete (cascades to cells) |
| `/__proxy/api/bench/suites` | GET | Available graded suites and thinking modes |
| `/__proxy/api/bench/models` | GET | Models across Ollama, LM Studio and vLLM |
| `/__proxy/api/bench/report` | GET | Comparison report: `format=json` \| `markdown` \| `html` |

Submit body: `{model | models[], runs, max_tokens, prompt_tokens, concurrency, randomize,
exclusive, drain_seconds, thinking, temperature, top_p, top_k, min_p, seed, extra_body, upstream,
bypass_router, no_nudge, suite, grade_timeout, quiesce}`. `models`, `prompt_tokens`, `thinking`
and `temperature` accept lists.
