"""Weight size for vLLM-served models.

Ollama reports size in its tag list, LM Studio in its API, llama.cpp gets its GGUF stat'd —
vLLM reported nothing, so every vLLM row in a report rendered a blank size and the memory
chart could not describe the backend where memory is tightest. vLLM knows only what --model
named, so the size has to come from the Hugging Face cache it downloaded into.
"""
import os

from ai_proxy import proxy


def _repo(tmp_path, slug, files, rev="abc123"):
    snap = tmp_path / "hub" / ("models--" + slug.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True)
    for name, size in files.items():
        f = snap / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\0" * size)
    return snap


def test_a_cached_repo_is_sized_from_its_weights(tmp_path, monkeypatch):
    _repo(tmp_path, "nvidia/Some-Model-NVFP4",
          {"model-00001-of-00002.safetensors": 4000, "model-00002-of-00002.safetensors": 6000})
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes("nvidia/Some-Model-NVFP4") == 10000


def test_configs_and_tokenizers_do_not_count_as_weights(tmp_path, monkeypatch):
    """A repo carries plenty that is never resident; summing the directory would report a
    number nobody is holding in memory."""
    _repo(tmp_path, "org/m", {"model.safetensors": 5000, "config.json": 900,
                              "tokenizer.json": 4000, "README.md": 800})
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes("org/m") == 5000


def test_only_the_newest_revision_counts(tmp_path, monkeypatch):
    """An updated repo leaves the old snapshot behind; adding both doubles the model."""
    old = _repo(tmp_path, "org/m", {"model.safetensors": 3000}, rev="old")
    _repo(tmp_path, "org/m", {"model.safetensors": 7000}, rev="new")
    os.utime(old, (10**9, 10**9))          # age the old snapshot, don't touch the new one
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes("org/m") == 7000


def test_a_local_checkpoint_directory_is_sized_directly(tmp_path, monkeypatch):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\0" * 1234)
    (d / "config.json").write_bytes(b"\0" * 50)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes(str(d)) == 1234


def test_an_uncached_repo_reports_zero_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes("org/never-downloaded") == 0


def test_gguf_weights_are_recognised_too(tmp_path, monkeypatch):
    _repo(tmp_path, "org/g", {"m-Q4_K_M.gguf": 2500})
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert proxy._hf_cache_bytes("org/g") == 2500
