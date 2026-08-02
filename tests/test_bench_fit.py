"""A model larger than the machine is not offered.

qwen3:235b-a22b is 132.4 GB of weights on a 121 GB box. It cannot be resident, so every second
spent selecting it, queueing it and waiting for it to fail is wasted — and in a sweep it fails
somewhere in the middle of a multi-hour run.
"""
import asyncio

from ai_proxy import proxy as P

GB = 1024


def test_a_model_larger_than_the_machine_does_not_fit():
    assert P._bench_fits(132.4 * GB, 121 * GB) is False


def test_models_that_fit_are_not_refused():
    # Real sizes off this box. The largest of these must stay selectable.
    for gb in (62.8, 60.9, 48.2, 37.2, 17.3, 2.3, 0.5):
        assert P._bench_fits(gb * GB, 121 * GB) is True, f"{gb} GB wrongly refused"


def test_the_check_allows_for_more_than_the_weights():
    """Weights are the floor: a runtime also needs a KV cache, activations and its own
    footprint, and the OS needs room left to not fall over."""
    assert P._bench_fits(115 * GB, 121 * GB) is False


def test_an_unknown_size_is_not_a_refusal():
    """vLLM checkpoints live inside a container and cannot be sized from here. Hiding a model
    because it could not be measured is indistinguishable from it not existing."""
    assert P._bench_fits(None, 121 * GB) is None
    assert P._bench_fits(0, 121 * GB) is None
    assert P._bench_fits(40 * GB, None) is None


def test_annotation_marks_only_what_cannot_fit(monkeypatch):
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": 121 * GB})
    index = {
        "ollama:huge": {"model": "huge", "upstream": "ollama", "size_mb": 132.4 * GB},
        "ollama:big": {"model": "big", "upstream": "ollama", "size_mb": 60.9 * GB},
        "vllm:unknown": {"model": "unknown", "upstream": "vllm"},
    }
    P._bench_annotate_fit(index)
    assert index["ollama:huge"]["fits"] is False
    assert "132.4 GB" in index["ollama:huge"]["fit_detail"]
    assert "121 GB" in index["ollama:huge"]["fit_detail"]
    assert index["ollama:big"]["fits"] is True
    assert "fits" not in index["vllm:unknown"], "an unmeasurable model must not be marked"


def test_capacity_is_what_counts_not_free_memory(monkeypatch):
    """The bench evicts everything before it measures, so what matters is what the box can
    hold — not what happens to be resident while someone reads the picker. With llama.cpp up,
    free memory here is ~31 GB and every large model would look impossible."""
    monkeypatch.setattr(P, "_mem_snapshot",
                        lambda: {"total_mb": 121 * GB, "avail_mb": 31 * GB})
    index = {"ollama:big": {"model": "big", "upstream": "ollama", "size_mb": 60.9 * GB}}
    P._bench_annotate_fit(index)
    assert index["ollama:big"]["fits"] is True


def test_preflight_refuses_a_model_that_cannot_fit(client):
    """Defence in depth: the picker disables it, and the endpoint refuses it, so an API caller
    cannot queue a four-hour sweep around a cell that can never run."""
    meta = {"model": "huge", "upstream": "ollama", "loaded": False, "fits": False,
            "fit_detail": "132.4 GB of weights needs about 152 GB resident; "
                          "this machine has 121 GB"}
    why = P._bench_preflight("huge", meta, "ollama", {"ollama:huge": meta})
    assert why and "does not fit" in why
    assert "121 GB" in why, "should say what it was measured against"


def test_a_fitting_model_still_passes_preflight(client):
    meta = {"model": "ok", "upstream": "ollama", "loaded": True, "fits": True}
    assert P._bench_preflight("ok", meta, "ollama", {"ollama:ok": meta}) is None


def test_the_size_is_read_from_the_shape_the_snapshot_actually_uses(client, monkeypatch):
    """_ollama_snapshot flattens /api/tags into size_mb and parameter_size. Reading `size` and
    `details` instead returned None for every model, so nothing was ever blocked and the check
    looked like it worked."""
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": 121 * GB})
    monkeypatch.setattr(P, "system_now", lambda: {
        "ollama": {"tags": [{"name": "qwen3:235b-a22b", "size_mb": 135_580,
                             "parameter_size": "235.1B"},
                            {"name": "qwen3:4b", "size_mb": 2_355,
                             "parameter_size": "4.0B"}]}})
    idx = asyncio.run(P._bench_model_index())
    assert idx["ollama:qwen3:235b-a22b"]["size_mb"] == 135_580, "size never reached the index"
    assert idx["ollama:qwen3:235b-a22b"]["fits"] is False
    assert idx["ollama:qwen3:4b"]["fits"] is True


# ---- capability, as distinct from capacity -------------------------------------------------

def test_a_model_that_cannot_complete_is_refused(client, monkeypatch):
    """An embedding model cannot produce a measurement under any configuration, so it is
    refused the way a model that does not fit is."""
    async def caps(client_, name):
        return {"nomic-embed": ["embedding"], "qwen3:4b": ["completion", "tools"]}.get(name)

    monkeypatch.setattr(P, "_ollama_capabilities", caps)
    index = {"ollama:nomic-embed": {"model": "nomic-embed", "upstream": "ollama"},
             "ollama:qwen3:4b": {"model": "qwen3:4b", "upstream": "ollama"}}
    asyncio.run(P._bench_annotate_caps(index, None))
    assert index["ollama:nomic-embed"]["benchable"] is False
    assert "embedding" in index["ollama:nomic-embed"]["bench_detail"]
    assert "benchable" not in index["ollama:qwen3:4b"]


def test_a_vision_model_is_flagged_but_not_refused(client, monkeypatch):
    """It completes text, so timing it is a perfectly good speed benchmark. Only its score on a
    coding suite is meaningless — and that is a fact about the suite, not the model."""
    async def caps(client_, name):
        return ["tools", "thinking", "completion", "vision"]

    monkeypatch.setattr(P, "_ollama_capabilities", caps)
    index = {"ollama:minicpm-v4.5": {"model": "minicpm-v4.5", "upstream": "ollama"}}
    asyncio.run(P._bench_annotate_caps(index, None))
    assert index["ollama:minicpm-v4.5"]["vision"] is True
    assert index["ollama:minicpm-v4.5"].get("benchable") is not False, \
        "a vision model can still be measured for speed"


def test_silence_from_ollama_is_not_a_refusal(client, monkeypatch):
    async def caps(client_, name):
        return None

    monkeypatch.setattr(P, "_ollama_capabilities", caps)
    index = {"ollama:mystery": {"model": "mystery", "upstream": "ollama"}}
    asyncio.run(P._bench_annotate_caps(index, None))
    assert "benchable" not in index["ollama:mystery"]


def test_preflight_refuses_a_model_that_cannot_complete(client):
    meta = {"model": "nomic-embed", "upstream": "ollama", "loaded": True,
            "benchable": False, "bench_detail": "cannot answer a chat request — "
                                                "Ollama reports embedding"}
    why = P._bench_preflight("nomic-embed", meta, "ollama", {"ollama:nomic-embed": meta})
    assert why and "cannot be benchmarked" in why
