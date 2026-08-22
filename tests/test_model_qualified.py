"""The same model served by two backends must be selectable, and tellable apart.

qwen3-coder-next exists as a 51 GB GGUF in Ollama and as NVFP4 in vLLM — different builds with
different speeds. /v1/models deduped by name alone, so only one appeared, attributed to whichever
backend was polled first, and there was no way to ask for the other.

Names are qualified ONLY when ambiguous. Suffixing everything would break every client config
that names a model today.
"""
from ai_proxy import proxy as P


# --- splitting a qualified name ----------------------------------------------------------------


def test_a_backend_suffix_is_split_off():
    assert P._split_qualified_model("qwen3-coder-next/vllm") == ("qwen3-coder-next", "vllm")
    assert P._split_qualified_model("qwen3-coder-next/ollama") == ("qwen3-coder-next", "ollama")


def test_huggingface_ids_are_left_alone():
    """Model ids contain slashes routinely; splitting those would break them."""
    for name in ("hf.co/unsloth/Qwen-AgentWorld-35B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                 "saricles/Qwen3-Coder-Next-NVFP4-GB10"):
        assert P._split_qualified_model(name) == (name, None), name


def test_an_unknown_suffix_is_not_a_backend():
    assert P._split_qualified_model("some-model/v2") == ("some-model/v2", None)


def test_a_plain_name_is_unchanged():
    assert P._split_qualified_model("gemma4") == ("gemma4", None)


def test_every_provider_name_works_as_a_suffix():
    for up in P.PROVIDERS:
        assert P._split_qualified_model("m/" + up) == ("m", up), up


# --- the catalogue -------------------------------------------------------------------------------


def test_the_listing_qualifies_only_ambiguous_names(client):
    """A name served by one backend stays bare; served by two, both appear qualified."""
    data = client.get("/v1/models").json().get("data") or []
    ids = [e["id"] for e in data]
    assert len(ids) == len(set(ids)), "ids must be unique or a picker cannot address them"
    for e in data:
        if "/" in e["id"] and e.get("upstream"):
            # A qualified entry carries the parts it was built from, so a client does not
            # have to re-parse the string to know what it selected.
            assert e["id"] == "%s/%s" % (e["served_model"], e["upstream"])
            assert e["upstream"] in P.PROVIDERS
