#!/usr/bin/env python3
"""Streaming throughput benchmark for the AI Proxy.

Sends N streaming chat-completion requests against a chosen model and reports
TTFT (time-to-first-token) and decoding rate (tokens-per-second) percentiles.
Each request streams via SSE so we can measure the real prefill-vs-decode split.

Usage:
    python3 bench.py --base http://localhost:11444 \
                     --model llama3.1:8b \
                     --runs 5 --max-tokens 256

Designed to be self-contained — uses only stdlib (urllib + json + time).
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from typing import List, Optional, Tuple


PROMPT = (
    "Write a clear, well-commented Python implementation of binary search over a sorted "
    "list of integers. Include type hints, a docstring, and a small test block at the "
    "bottom that exercises edge cases (empty list, single element, target missing, target "
    "at start, target at end). Be thorough."
)

# Realistic long-context filler: a chunk of code-ish content. Repeated to reach the target
# size. Using actual code (not lorem ipsum) so the model's prefill does representative work.
LONG_CONTEXT_CHUNK = '''
# Module: data_pipeline/transforms.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Callable, TypeVar, Generic, Optional
import logging
import time

logger = logging.getLogger(__name__)
T = TypeVar("T")
U = TypeVar("U")


@dataclass
class TransformResult(Generic[T]):
    """Outcome of running a transform: the value plus diagnostics."""
    value: Optional[T] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Pipeline(Generic[T, U]):
    """Composes a sequence of stages into a single callable. Each stage takes a T and
    returns a U; the pipeline propagates results through the stages, short-circuiting on
    None or on the first error encountered. Diagnostics are collected at every stage."""

    def __init__(self, stages: list[Callable[[T], U]]):
        self.stages = stages
        self._call_count = 0
        self._error_count = 0

    def run(self, item: T) -> TransformResult[U]:
        result = TransformResult(value=item)
        start = time.perf_counter()
        for i, stage in enumerate(self.stages):
            if result.value is None or result.error:
                break
            try:
                result.value = stage(result.value)
            except Exception as e:
                result.error = f"stage {i} ({stage.__name__}): {e}"
                self._error_count += 1
                logger.exception("Pipeline stage %d failed", i)
                break
        result.duration_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1
        return result

    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "stages": len(self.stages),
        }


def filter_none(items: Iterable[Optional[T]]) -> Iterator[T]:
    """Yield items that aren't None. Saves a one-line list comprehension everywhere."""
    return (x for x in items if x is not None)


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Group items into lists of `size`. Final batch may be smaller. Useful for batch APIs
    that have request-size limits (Anthropic 100, OpenAI 2048, etc.)."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
'''


def _build_long_prompt(target_tokens: int, base: str) -> str:
    """Pad `base` with repeated LONG_CONTEXT_CHUNK until reaching target_tokens (~chars/3.5).
    Frames the request so the model has something coherent to do at the end."""
    target_chars = int(target_tokens * 3.5)
    head = (
        "Below is a code module. Read it carefully. After the module, you'll be given a task.\n\n"
        "<CODE>\n"
    )
    tail = (
        "\n</CODE>\n\n"
        "Task: " + base + " You may reference the patterns shown in the code above."
    )
    body = ""
    while len(head) + len(body) + len(tail) < target_chars:
        body += LONG_CONTEXT_CHUNK
    return head + body + tail


def stream_one(base: str, model: str, max_tokens: int, prompt: str, seq: int
               ) -> Tuple[Optional[float], Optional[float], int, Optional[str]]:
    """Send one streaming /v1/chat/completions request. Returns
    (ttft_ms, total_ms, completion_tokens, error)."""
    url = base.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("x-client-name", "ai-proxy-bench")
    req.add_header("x-priority", "high")  # don't queue behind other traffic
    t0 = time.perf_counter()
    ttft_ms = None
    completion_tokens = 0
    upstream_completion = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                if not raw_line.strip():
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # Capture TTFT on the first chunk that has actual content (not just role).
                for ch in (j.get("choices") or []):
                    delta = ch.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        # Heuristic token count by whitespace-split (close enough for percentile work).
                        completion_tokens += len(re.findall(r"\S+", content))
                # Final usage block when stream_options.include_usage is set.
                u = j.get("usage")
                if isinstance(u, dict) and u.get("completion_tokens"):
                    upstream_completion = u["completion_tokens"]
    except Exception as e:
        return (ttft_ms, (time.perf_counter() - t0) * 1000, 0, f"{type(e).__name__}: {e}")
    total_ms = (time.perf_counter() - t0) * 1000
    if upstream_completion is not None:
        completion_tokens = upstream_completion
    return (ttft_ms, total_ms, completion_tokens, None)


def pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def fmt_tps(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:11444",
                    help="proxy base URL (default: http://localhost:11444)")
    ap.add_argument("--model", required=True, help="model id, e.g. qwen3-coder-next:latest")
    ap.add_argument("--runs", type=int, default=5, help="number of requests (default 5)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="cap completion length so the bench finishes (default 256)")
    ap.add_argument("--prompt", default=PROMPT, help="prompt text (used as-is unless --prompt-tokens is set)")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad prompt with realistic code filler to ~this many tokens (e.g. 8000)")
    args = ap.parse_args()
    if args.prompt_tokens > 0:
        args.prompt = _build_long_prompt(args.prompt_tokens, args.prompt)

    print(f"Benchmark: {args.runs} streaming requests to {args.base} model={args.model}")
    print(f"Prompt length: {len(args.prompt)} chars · max_tokens={args.max_tokens}")
    print("─" * 78)

    rows = []
    for i in range(args.runs):
        ttft, total, ct, err = stream_one(
            args.base, args.model, args.max_tokens, args.prompt, i
        )
        decode_ms = (total - ttft) if (ttft is not None and total is not None) else None
        decode_tps = (ct / (decode_ms / 1000.0)) if (decode_ms and decode_ms > 0 and ct > 0) else None
        rows.append({
            "ttft_ms": ttft, "total_ms": total, "completion_tokens": ct,
            "decode_tps": decode_tps, "error": err,
        })
        if err:
            print(f"  #{i+1:>2}  ERROR: {err}")
        else:
            print(
                f"  #{i+1:>2}  TTFT={fmt_ms(ttft):>7}  "
                f"total={fmt_ms(total):>7}  "
                f"completion={ct:>4} tok  "
                f"decode={fmt_tps(decode_tps):>5} tok/s"
            )

    print("─" * 78)
    successes = [r for r in rows if not r["error"]]
    if not successes:
        print("All runs failed.")
        sys.exit(1)
    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    decodes = [r["decode_tps"] for r in successes if r["decode_tps"] is not None]
    print(f"\nResults ({len(successes)}/{args.runs} successful):")
    if ttfts:
        print(f"  TTFT          min={fmt_ms(min(ttfts))}  "
              f"p50={fmt_ms(pct(ttfts, 50))}  "
              f"p90={fmt_ms(pct(ttfts, 90))}  "
              f"max={fmt_ms(max(ttfts))}")
    if decodes:
        print(f"  Decode rate   min={fmt_tps(min(decodes))} tok/s  "
              f"p50={fmt_tps(pct(decodes, 50))}  "
              f"p90={fmt_tps(pct(decodes, 90))}  "
              f"max={fmt_tps(max(decodes))}")
        avg = statistics.mean(decodes)
        print(f"                mean={fmt_tps(avg)} tok/s")


if __name__ == "__main__":
    main()
