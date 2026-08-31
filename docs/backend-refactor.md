# Backend registry

Written at the end of the session that introduced `Provider` / `SideService`, and updated when
the last hand-wired site was closed. The refactor is **complete**: all ten fan-out sites now
read from the registry. Everything here is deployed and green (321 tests) on
`feature/local-backends-and-usage-reporting`.

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
| _this commit_ | `Provider.load()` — the tenth and last site. `control_model_load` went from 118 lines of four-way branching to 22 with no backend named in it. |

Adding a backend today is **one registry entry**. It then appears in the tab bar, gets a panel,
is routable, loadable, benchmarkable, counted in decode stats, stopped/restored by the bench,
and stored in metrics. `tests/test_backend_registry.py::test_registering_a_backend_is_the_only_step`
and `::test_a_registered_backend_is_loadable_without_touching_the_endpoint` pin that property.

## The last site: `control_model_load`

`load()` is the one method that genuinely differs per backend rather than being shared, because
the four mechanisms have nothing in common:

| Backend | Mechanism | Parameters |
|---|---|---|
| `lmstudio` | `lms` CLI | `--context-length`, `--parallel`, `--gpu`, `--ttl`; `_lms_available()` gate first |
| `vllm` | docker start | optional `container` from `_vllm_configs()`, validated against `serves_port`; stops rivals on the port; then `ready()` |
| `llamacpp` | systemd drop-in rewrite | `context_length`, `parallel`; carries over unspecified values from the running server; `daemon-reload` (**no unit argument**) then restart, then `ready()` |
| `ollama` | HTTP keep-alive warm | model name only |

Two behaviours moved out of the handler and onto the backends:

- **`requires_model_name`.** vLLM and llama.cpp are launched with one model, so "load" means
  "start this server". That exemption used to be a literal `upstream not in ("vllm",
  "llamacpp")` in the handler — a fact about two backends, written somewhere neither of them
  could see.
- **Unknown upstreams now 404** instead of falling through the branch chain into the Ollama
  path, where a typo'd backend name quietly asked Ollama to load a model it had never heard of.
  A registered backend with no `load()` gets a 501 that names itself.

`_llamacpp_ready` / `_vllm_ready` are still called as module-level functions inside `load()`
rather than `self.ready()`. They already delegate to the registry, and they are the seam the
tests patch.

**It was done by hand.** Two earlier attempts to relocate these branches with text substitution
failed:

1. Replacing `_vllm_ready` by slicing between two anchors deleted everything between them,
   taking `_lms_bin`, `_lms_is_local`, `_lms_available`, `_lms_run` and `VLLM_ARG_SUMMARY`.
2. A regex dedent of the branch bodies handled 8-space indentation but not deeper levels and
   produced an `IndentationError`.

What worked: the branch bodies sit at 8 spaces inside `if upstream == ...:` and at 8 spaces
again inside a method, so they transfer verbatim with **no dedent at all**. Move them into
subclasses first, leave the originals in a dead function, delete that function by line number
once the suite is green, and check afterwards that the neighbouring helpers still exist — that
is the specific thing attempt 1 broke.

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
