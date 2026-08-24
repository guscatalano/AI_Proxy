"""Price the KV cache for models that do not spell out every field.

qwen3.6:35b-a3b crash-looped Ollama once a minute while the fit guard waved it through. Two
independent faults, either of which was enough:

  * head_count_kv is absent on that model, and the estimator returned None rather than reading
    the absence as "no grouped-query attention", so nothing could be priced at all. Its real KV
    at Ollama's 262,144 x 4 slots is about 340 GB on a 122 GB box.
  * the size lookup keyed on the normalised name, so qwen3.6:35b-a3b (23 GB) was priced at
    qwen3.6:27b's 16 GB — the wrong model's weight, and small enough to look like it fit.
"""
from ai_proxy import proxy as P


def _info(**over):
    base = {"general.architecture": "arch", "arch.block_count": 40,
            "arch.attention.head_count": 16, "arch.attention.key_length": 256,
            "arch.attention.value_length": 256}
    base.update({("arch." + k if not k.startswith("general") else k): v
                 for k, v in over.items()})
    return base


def test_a_model_without_head_count_kv_is_still_priced():
    """Absent means MHA — every head carries its own KV — not unknowable."""
    kv = P._bench_ollama_kv_mb(_info(), 1000)
    assert kv and kv > 0


def test_absent_head_count_kv_is_read_as_full_multi_head():
    """It must equal what an explicit head_count_kv of the same value produces."""
    implied = P._bench_ollama_kv_mb(_info(), 1000)
    explicit = P._bench_ollama_kv_mb(_info(**{"attention.head_count_kv": 16}), 1000)
    assert implied == explicit


def test_declared_grouped_query_attention_still_wins():
    """A model that says 8 KV heads must not be priced as though it had 16."""
    gqa = P._bench_ollama_kv_mb(_info(**{"attention.head_count_kv": 8}), 1000)
    mha = P._bench_ollama_kv_mb(_info(), 1000)
    assert gqa < mha


def test_the_real_model_prices_far_beyond_this_box():
    """The number that matters: ~340 GB at Ollama's 262,144 x 4 slots, on 122 GB of memory."""
    kv_gb = P._bench_ollama_kv_mb(_info(), 262144 * 4) / 1024
    assert kv_gb > 200, "an estimate this low would have let it through again"


def test_a_model_missing_block_count_is_still_unpriceable():
    """Guessing the layer count would be inventing the answer, not being conservative."""
    info = _info()
    del info["arch.block_count"]
    assert P._bench_ollama_kv_mb(info, 1000) is None


def test_tags_of_one_family_are_not_conflated():
    """qwen3.6:35b-a3b and qwen3.6:27b normalise alike; pricing one as the other is what let a
    23 GB model be judged as 16 GB."""
    assert P._norm_model_id("qwen3.6:35b-a3b") == P._norm_model_id("qwen3.6:27b")
