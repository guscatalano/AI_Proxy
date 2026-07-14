"""Tests for the performance guards: body-size cap, analytics offloading, caching,
and system-history downsampling. See the perf investigation that motivated these."""
import inspect
import time

from ai_proxy import proxy as P


def test_truncate_none_and_small_unchanged():
    assert P._truncate_for_store(None) is None
    assert P._truncate_for_store("") == ""
    assert P._truncate_for_store("hello") == "hello"


def test_truncate_over_cap_marks(monkeypatch):
    monkeypatch.setattr(P, "MAX_STORED_BODY", 100)
    out = P._truncate_for_store("x" * 500)
    assert out.startswith("x" * 100)
    assert len(out) < 500
    assert "truncated by ai-proxy" in out


def test_truncate_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(P, "MAX_STORED_BODY", 0)
    big = "x" * 1000
    assert P._truncate_for_store(big) == big


def test_analytics_handlers_are_sync():
    # These must be plain `def` so Starlette runs them in a threadpool; an `async def`
    # here would run their blocking SQLite queries on the event loop and stall proxying.
    for name in ("stats", "suggestions", "system_history", "audit",
                 "list_conversations", "get_conversation"):
        fn = getattr(P, name)
        assert not inspect.iscoroutinefunction(fn), f"{name} must be sync (threadpool)"


def test_analytics_cache_roundtrip(monkeypatch):
    monkeypatch.setattr(P, "ANALYTICS_CACHE_TTL_S", 30)
    P._ANALYTICS_CACHE.clear()
    assert P._analytics_cache_get("k") is None
    P._analytics_cache_put("k", {"v": 1})
    assert P._analytics_cache_get("k") == {"v": 1}


def test_analytics_cache_respects_ttl(monkeypatch):
    monkeypatch.setattr(P, "ANALYTICS_CACHE_TTL_S", 0)  # disabled
    P._ANALYTICS_CACHE.clear()
    P._analytics_cache_put("k", 1)
    assert P._analytics_cache_get("k") is None


def test_cap_num_ctx_top_level_and_nested():
    # top-level (OpenAI-compat /v1/*)
    b = {"model": "m", "num_ctx": 262144}
    assert P._cap_num_ctx(b, 32768) == 32768
    assert b["num_ctx"] == 32768
    # nested under options (Ollama-native /api/*)
    b2 = {"model": "m", "options": {"num_ctx": 262144, "temperature": 1}}
    assert P._cap_num_ctx(b2, 32768) == 32768
    assert b2["options"]["num_ctx"] == 32768
    assert b2["options"]["temperature"] == 1  # untouched


def test_cap_num_ctx_leaves_small_values_and_absent():
    assert P._cap_num_ctx({"num_ctx": 8192}, 32768) is None      # under the cap → unchanged
    assert P._cap_num_ctx({"model": "m"}, 32768) is None          # no num_ctx → nothing
    assert P._cap_num_ctx({"num_ctx": 262144}, 0) is None         # cap disabled (0)


def test_model_parallelism_ollama_serializes_qwen3():
    # Ollama refuses to batch the Qwen3 family — flagged serial by arch/family or name.
    assert P._model_parallelism("ollama", "qwen3moe")["parallel"] is False
    assert P._model_parallelism("ollama", "qwen3next")["parallel"] is False
    assert P._model_parallelism("ollama", None, "qwen3.6:27b")["parallel"] is False
    # Non-qwen3 architectures batch fine under Ollama.
    assert P._model_parallelism("ollama", "llama")["parallel"] is True
    assert P._model_parallelism("ollama", "gemma3")["parallel"] is True
    assert P._model_parallelism("ollama", "qwen2")["parallel"] is True   # only qwen3* is serial
    # Unknown arch/name → can't tell.
    assert P._model_parallelism("ollama", None, None)["parallel"] is None


def test_model_parallelism_lmstudio_always_parallel():
    # llama.cpp continuous batching parallelizes every architecture, including qwen3next.
    assert P._model_parallelism("lmstudio", "qwen3next")["parallel"] is True
    assert P._model_parallelism("lmstudio", "gpt-oss")["parallel"] is True


def test_cache_verdict_token_based_ollama():
    # Ollama reports evaluated-only prompt tokens: evaluated << estimated ⇒ hit.
    assert P._cache_verdict(2000, 100000, upstream="ollama")[1] == "hit"
    assert P._cache_verdict(95000, 100000, upstream="ollama")[1] == "miss"
    assert P._cache_verdict(80000, 100000, upstream="ollama")[1] == "partial"
    assert P._cache_verdict(None, 100000, upstream="ollama") == (None, None)  # in flight
    assert P._cache_verdict(2000, 0, upstream="ollama") == (None, None)       # no estimate
    pct, verdict = P._cache_verdict(1000, 100000, upstream="ollama")
    assert verdict == "hit" and pct == 99.0


def test_cache_verdict_lmstudio_token_method_is_blind():
    # LM Studio reports the FULL prompt_tokens regardless of reuse, so the token method must NOT
    # fire (would falsely say 'miss'/'hit'). Without a timing signal → unknown.
    assert P._cache_verdict(2000, 100000, upstream="lmstudio") == (None, None)
    assert P._cache_verdict(100000, 100000, upstream="lmstudio") == (None, None)


def test_cache_verdict_timing_detects_reuse_across_upstreams():
    # Implausibly fast prefill (2500 tokens in 30ms ≈ 83k tok/s) ⇒ prefill skipped ⇒ hit,
    # for both LM Studio (only signal available) and Ollama.
    assert P._cache_verdict(2500, 2500, ttft_ms=30, prompt_tokens=2500, upstream="lmstudio")[1] == "hit"
    assert P._cache_verdict(2500, 2500, ttft_ms=30, prompt_tokens=2500, upstream="ollama")[1] == "hit"
    # Normal prefill speed (2500 tokens in 2500ms = 1000 tok/s) is NOT flagged by timing.
    assert P._cache_verdict(2500, 2500, ttft_ms=2500, prompt_tokens=2500, upstream="lmstudio") == (None, None)
    # Tiny prompt: timing too noisy to judge even if fast.
    assert P._cache_verdict(50, 50, ttft_ms=1, prompt_tokens=50, upstream="lmstudio") == (None, None)


def test_iter_and_strip_request_images():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}]}]}
    imgs = list(P._iter_request_images(body))
    assert len(imgs) == 2
    assert imgs[0] == (0, "image/png", "data", "QUJDRA==")
    assert imgs[1][2] == "url" and imgs[1][3] == "https://example.com/x.jpg"
    # Stripping replaces the inline base64 but leaves the external URL and structure intact.
    assert P._strip_image_data(body) == 1
    url0 = body["messages"][0]["content"][1]["image_url"]["url"]
    assert "base64,QUJDRA==" not in url0 and "see Images" in url0
    assert body["messages"][0]["content"][2]["image_url"]["url"] == "https://example.com/x.jpg"


def test_load_images_data():
    assert P._load_images_data(None) == []
    assert P._load_images_data("not json") == []
    assert P._load_images_data('[{"media_type":"image/png","data":"QUJD"}]') == [
        {"media_type": "image/png", "data": "QUJD"}]


def test_body_has_images():
    assert P._body_has_images({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}) is True
    assert P._body_has_images({"messages": [{"role": "user", "content": [
        {"type": "image", "source": {}}]}]}) is True   # Anthropic shape
    assert P._body_has_images({"messages": [{"role": "user", "content": "just text"}]}) is False
    assert P._body_has_images({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "no images here"}]}]}) is False


def test_router_images_bypass_coder_rule(monkeypatch):
    # Vision requests keep the multimodal model (rule 1, no rewrite); text routes to the coder.
    cfg = {"model_router": {"enabled": True, "aliases": {}, "rules": [
        {"if": {"from_model_prefix": "qwen", "has_images": True}, "then": "qwen3.6:27b"},
        {"if": {"from_model_prefix": "qwen"}, "then": "qwen/qwen3-coder-next", "upstream": "lmstudio"},
    ]}}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    img = {"model": "qwen3.6:27b", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    assert P.evaluate_router(img, {}) is None            # rule 1 matches, target==original → no rewrite → stays on Ollama
    assert img["model"] == "qwen3.6:27b"
    txt = {"model": "qwen3.6:27b", "messages": [{"role": "user", "content": "write code"}]}
    out = P.evaluate_router(txt, {})
    assert out and out["to"] == "qwen/qwen3-coder-next" and out["upstream"] == "lmstudio"


def test_evaluate_router_upstream_override(monkeypatch):
    cfg = {"model_router": {"enabled": True, "aliases": {},
           "rules": [{"if": {"from_model_prefix": "qwen"}, "then": "qwen/qwen3-coder-next",
                      "upstream": "lmstudio"}]}}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    body = {"model": "qwen3.6:27b"}
    out = P.evaluate_router(body, {})
    assert out["to"] == "qwen/qwen3-coder-next"
    assert out["upstream"] == "lmstudio"
    assert body["model"] == "qwen/qwen3-coder-next"  # body mutated in place
    # A non-matching model is left alone.
    assert P.evaluate_router({"model": "gemma3:27b"}, {}) is None


def test_system_history_downsampled(client):
    conn = P.db()
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO system_metrics (ts, cpu_pct) VALUES (?, ?)",
        [(now - i, 1.0) for i in range(1000)],
    )
    conn.commit()
    conn.close()
    r = client.get("/__proxy/api/system/history?minutes=60")
    assert r.status_code == 200
    body = r.json()
    assert body["total_samples"] >= 1000       # all rows in the window were counted
    assert len(body["samples"]) <= 800         # but the payload is downsampled
