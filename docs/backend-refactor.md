# Backend registry: what's done, what's left

Written at the end of the session that introduced `Provider` / `SideService`. Everything here
is deployed and green (316 tests) on `feature/local-backends-and-usage-reporting`.

## Why this exists

Adding a backend used to mean ten coordinated hand-edits. Every UI and telemetry bug in that
area was a *missed site*, never a logic error:

- The metrics `INSERT` gained a column and a placeholder for `llamacpp_json` but never its
  **value**. Every tick raised `Incorrect number of bindings` into `except Exception: pass`.
  Telemetry was dead for **62 minutes** while the dashboard served an hour-old sample as
  current — which is why vLLM showed "ready" after it had stopped, and why its Stop button
  looked broken.
- `GENERATION_RATE_UPSTREAMS` once omitted vLLM. 100% of its traffic vanished from the decode
  figures and the dashboard fell back to a number ~6× lower, reading as a broken model.
- The System tab's provider bar was a hand-written list of five. llama.cpp was invisible twice
  despite serving correctly.
- `_bench_model_index` did `await system_now()` after `system_now` became sync. The `TypeError`
  went into a blanket `except`, silently dropping **every** backend's enrichment and falling
  through to an Ollama-only path. The bench listed 15 Ollama models and zero of everything
  else, for an unknown length of time.

## What landed

| Commit | Change |
|---|---|
| `a9d335a` | `Provider` / `SideService` + `PROVIDERS` / `SIDE_SERVICES` registry. `_bench_load_modes()`, `GENERATION_RATE_UPSTREAMS`, `_UPSTREAM_BASES` and the bench's upstream validation all derived from it. |
| `db30f4a` | Metrics stored as one `backends_json` blob per sample. No backend name appears in the `INSERT` any more, so the desync is unrepresentable. Probes gather with `return_exceptions=True`. Reads fall back to legacy columns so history survives. |
| `a39c6a1` | `SideService.start()` / `.stop()`; `_control_backend` implements docker / systemd / unmanaged **once**. The bench residency handshake iterates the registry instead of carrying a field per backend. |
| `417fe3c` | Tab bar built from `backends_meta`; generic panel for any backend without a bespoke one. `Provider.ready()` replaces the identical `_vllm_ready` / `_llamacpp_ready` polls. |

Adding a backend today is **one registry entry**. It then appears in the tab bar, gets a panel,
is routable, benchmarkable, counted in decode stats, stopped/restored by the bench, and stored
in metrics. `tests/test_backend_registry.py::test_registering_a_backend_is_the_only_step` pins
that property.

## What's left: `control_model_load`

The tenth site. `control_model_load` still branches per upstream instead of calling
`provider.load(...)`. It is the only place left that maps a backend name to a mechanism by hand.

**Target shape**

```python
prov = PROVIDERS.get(upstream)
if prov is None:
    return JSONResponse({"error": f"unknown upstream {upstream!r}",
                         "known": sorted(PROVIDERS)}, status_code=404)
return await prov.load(payload, name)
```

**The four branches, and why this isn't mechanical.** Each takes a genuinely different parameter
shape, so `load()` cannot be one signature with one body:

| Backend | Mechanism | Parameters |
|---|---|---|
| `lmstudio` | `lms` CLI | `--context-length`, `--parallel`, `--gpu`, `--ttl`; needs `_lms_available()` gate first |
| `vllm` | docker start | optional `container` chosen from `_vllm_configs()`, validated against `serves_port`; then `ready()` |
| `llamacpp` | systemd drop-in rewrite | `context_length`, `parallel`; carries over unspecified values from the running server; `daemon-reload` (**no unit argument**) then restart, then `ready()` |
| `ollama` | HTTP keep-alive warm | model name only |

Note `vllm` and `llamacpp` are exempt from the `'model' is required` guard — they take a
container or a context, not a name.

**Do it by hand, not by script.** Two attempts to relocate these with text substitution failed
in this session:

1. Replacing `_vllm_ready` by slicing between two anchors deleted everything between them,
   taking `_lms_bin`, `_lms_is_local`, `_lms_available`, `_lms_run` and `VLLM_ARG_SUMMARY`.
2. A regex dedent of the branch bodies handled 8-space indentation but not deeper levels and
   produced an `IndentationError`.

Both were caught by the suite in seconds and nothing broken reached production, but the lesson
is that this file does not tolerate blind structural surgery. Read the branches, move them
deliberately, run the suite after each one.

## Follow-ups not related to the refactor

- **`sudo loginctl enable-linger crimson`** — ComfyUI and llama.cpp are `systemctl --user`
  units. `Linger=no`, so they will not start at boot and the proxy's buttons report
  "Failed to connect to bus" after a reboot until someone logs in.
- **llama.cpp context sizing.** `--parallel N` divides one KV pool across slots, so
  `--ctx-size 131072 --parallel 4` serves **32,768 per slot**. Benchmarking DeepSeek V4 Flash
  against qwen's 262K at 32K would be misleading; decide the per-slot target before running
  anything you intend to trust.
- **DeepSeek V4 Flash is loaded and unbenchmarked.** 90.9 GB `UD-IQ2_XXS` at
  `~/models/ds4-flash/`, arch verified `deepseek4`, `parallel=4`. The recorded verdict
  ("serial, one request at a time, not viable") was about antirez's bespoke `ds4-server`, not
  the model — mainline llama.cpp reports 4 slots. Worth re-testing, along with DeepSWE
  7.3 → 54.4 on the `0731` checkpoint.
- **`Exited (137)` on vLLM stop** — Docker SIGKILLs it after the 180s `docker stop` timeout
  rather than a clean shutdown. Harmless today; raise the timeout if graceful matters.
- **`request_blobs` is ~6.4 GB of a ~7.8 GB database.** Archiving is enabled at
  `after_days: 7`; the main file stops growing but does not shrink without a `VACUUM`.
- **Blanket `except` audit.** Three separate outages this session were a live failure swallowed
  by `except Exception: pass` — the metrics collector, the archive delete, and the bench index.
  Worth a sweep for the pattern where the handler has no logging and the loop continues.
