# Tool Reliability Rules

Rules layered onto the proxy that catch and correct mis-parameterized, hallucinated, or stuck tool-call patterns before the client ever executes them. The motivation: when the model emits a malformed `tool_call` (e.g., missing a required `startLine`), the downstream tool returns a validation error, the model retries (often with the same mistake), tokens get burned, and the user sees confusing failures. These rules turn that into a fast self-correction loop *inside the proxy*.

## Status

| # | Rule | Phase | Status |
|---|------|-------|--------|
| 1 | `schema_validator` | post-flight (response) | **Implemented** |
| 2 | `hallucinated_tool` | post-flight (response) | **Implemented** |
| 3 | `tool_failure_breaker` | pre-flight (request) | **Implemented** |
| 4 | `tool_args_autofix` | post-flight (response) | **Implemented** |
| 5 | `system_preamble` | pre-flight (request) | Deferred — wording is opinionated |
| 6 | `argument_trimming` | post-flight (response) | Deferred — can mask real client bugs |

## How they intercept

Two phases, two mechanisms:

- **Pre-flight rules** look at the incoming request body before forwarding upstream. They block the request (return 400 with an OpenAI-shaped error) or warn (log + forward). Existing rules: `loop_detector`, `model_router`. New: `tool_failure_breaker`.
- **Post-flight rules** look at the response coming back from upstream. If a tool call is invalid, the proxy *replaces* the response with a synthetic one whose `message.content` describes the error and whose `finish_reason` is `stop`. The client sees a normal completion; the model, on the next turn, sees its own previous mistake in conversation history and adapts.

### Streaming

Post-flight rules need the full response before validating. For streaming requests **that include `tools[]`** AND have at least one post-flight rule enabled, the proxy buffers the upstream stream, validates the assembled response, then either:

- **replays the buffered chunks verbatim** if the response is valid, or
- **emits a synthetic SSE stream** with the corrective assistant content if it's not.

Latency cost: tool-call responses appear all at once instead of chunk-by-chunk. Most agent clients (Cursor, Copilot, Continue, etc.) already buffer tool-calls before acting, so user-visible UX is unchanged. Pure text generation streams on requests *without* tools pass through normally.

## Rule 1: `schema_validator`

Parses every tool call the model emits. Validates `function.arguments` against the `function.parameters` JSON Schema declared in the original request's `tools[]`. Catches:

- arguments that aren't valid JSON
- missing required fields
- type mismatches (string/integer/number/boolean/object/array)

When a violation is found, the entire response's `tool_calls` are stripped and replaced with assistant `content`:

> `[AI Proxy] Your tool call to read_file was rejected: missing required field 'startLine'. Please retry with all required fields populated.`

`finish_reason` becomes `stop`. The original (bad) tool call is preserved in `gate_details.intercepted_calls` for audit.

### Config

```json
{
  "schema_validator": {
    "enabled": true,
    "action": "block",
    "strict_types": true,
    "reject_unknown_fields": false
  }
}
```

- `strict_types` — when false, type mismatches are warned but not rejected.
- `reject_unknown_fields` — when true, also reject calls with fields not declared in `properties`.

## Rule 2: `hallucinated_tool`

Rejects tool calls that name a function not present in the request's `tools[]` array. The model invented the tool. Uses the same intercept mechanism — replaces the bad call with assistant content.

### Config

```json
{
  "hallucinated_tool": {
    "enabled": true,
    "action": "block"
  }
}
```

## Rule 3: `tool_failure_breaker`

Pre-flight. Walks the incoming request's `messages[]` history and counts recent tool-result errors per tool name (using the same heuristics as the auditor's "Tool X fails N% of the time" detector). When the latest `max_errors` tool results for the same tool are all errors, blocks the request with:

> `[AI Proxy] Tool 'read_file' has failed 3 consecutive times. The model should try a different approach.`

Pairs with `loop_detector`: that catches *identical retries*; this catches *different retries that still fail*.

### Config

```json
{
  "tool_failure_breaker": {
    "enabled": true,
    "action": "block",
    "max_errors": 3,
    "window": 5
  }
}
```

- `max_errors` — block when this many of the most recent tool results from the same tool are errors.
- `window` — how far back to look in `messages[]` for tool results.

## Rule 4: `tool_args_autofix`

Post-flight, runs **before** `schema_validator`. For each tool_call the model emits, looks at the `parameters` schema's `required` list. For any required field missing from the actual arguments, fills it in from configured defaults.

This is the "stop telling me to retry, just do it" rule — for cases like Copilot's `read_file` where the model keeps forgetting `startLine` and the proxy already knows a sensible default.

Default precedence: per-tool config → wildcard (`"*"`) config → the schema's own `default` property if declared.

The fixed response is what the client sees; the original (broken) tool_call is preserved in the audit row's `gate_details.fixes`. The audit verdict is `rewrite` with rule `tool_args_autofix` (set `action: "silent"` to suppress the audit row).

If autofix patches a call and the result still fails validation (e.g., a wrong type was already there), `schema_validator` takes over with its `intercept` correction.

### Config

```json
{
  "tool_args_autofix": {
    "enabled": true,
    "action": "audit",
    "defaults": {
      "*": { "startLine": 1 },
      "read_file": { "startLine": 1, "endLine": 1000 }
    }
  }
}
```

- `action`: `audit` (default — record each fix in the audit log) or `silent` (apply silently).
- `defaults["*"]`: applied to every tool that has the field as required.
- `defaults["<tool_name>"]`: per-tool overrides (take precedence over `*`).

## Audit visibility

All three rules write a row to the existing audit columns (`gate_verdict`, `gate_rule`, `gate_reason`, `gate_details`). Pre-flight blocks use verdict `block`. Post-flight intercepts use verdict `intercept` (new). The Audit tab shows them alongside `block`, `warn`, and `rewrite` events.

## Tradeoffs and open questions

- **Streaming latency**: tool-call responses on streams are no longer progressively delivered. If a user has a non-tool-using streaming workload mixed in, they pay no cost; tool workloads pay the buffering cost.
- **Schema compliance varies**: not all `tools[]` definitions include thorough JSON Schemas. When `parameters` is empty or unparseable, `schema_validator` no-ops (better than false positives).
- **Stripped-vs-rewritten**: the current design replaces the entire `tool_calls` array with assistant text. An alternative is to keep `tool_calls` but inject a `tool` role result message into the next request — that requires per-conversation state and is deferred.
- **Tool naming**: `hallucinated_tool` is exact-string match. If the model emits `read-file` vs declared `read_file`, that's a hallucination. Some clients normalize; the proxy doesn't (yet).

## Future work

- Rule 4 (`system_preamble`): inject a tool-usage reliability reminder into the system message when tools are present.
- Rule 5 (`argument_trimming`): drop fields not in the schema rather than rejecting (off by default).
- Streaming-aware *partial* validation: validate a tool call as soon as its arguments are complete, instead of buffering the whole response.
- Per-tool retry policy: configurable `max_attempts` before blocking, with exponential context (each retry adds more guidance to the system message).
