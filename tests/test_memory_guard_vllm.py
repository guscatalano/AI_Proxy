"""Memory accounting for a model that is already resident.

Both defects here share one assumption: that a benchmarked model loads during the run. It
does for Ollama. It does not for vLLM, which allocated its pool at container boot — so the
fit check demanded free memory for weights already loaded ("only 22 GB free; 23 GB wanted",
stamped on every vLLM cell while the 20 GB in question was serving requests), and the
resident figure, inferred from memory lost while loading, reported 712 MB for a 20 GB model.
"""
import pytest

from ai_proxy import proxy


FIT = proxy._BENCH_FIT_OVERHEAD


def want_for(meta):
    """The guard's target, as computed in the run path."""
    want = 0 if meta.get("loaded") else (meta.get("size_mb") or 0)
    return want * FIT if want else 0


def test_an_already_loaded_model_needs_no_room_made():
    assert want_for({"size_mb": 20563, "loaded": True}) == 0


def test_a_model_that_must_load_still_reserves_room():
    assert want_for({"size_mb": 20563, "loaded": False}) == pytest.approx(20563 * FIT)


def test_unknown_size_asks_for_nothing_rather_than_guessing():
    assert want_for({"loaded": False}) == 0


@pytest.mark.parametrize("cell,mb", [
    ("12.34GiB / 121GiB", 12636),
    ("512MiB / 121GiB", 512),
    ("1.5GB / 100GB", 1536),
    ("900KiB / 8GiB", 1),
    ("20.08GiB / 118.9GiB", 20562),      # the real Nemotron container
])
def test_docker_mem_usage_is_parsed_to_mb(cell, mb):
    assert proxy._docker_mem_mb(cell) == mb


@pytest.mark.parametrize("junk", ["", "   ", "N/A", "-- / --", None])
def test_unparseable_usage_yields_none_not_an_exception(junk):
    assert proxy._docker_mem_mb(junk) is None


def test_a_20gb_container_no_longer_reads_as_712mb():
    """The specific wrong number, kept as a regression: the inferred fallback measured the
    delta across a load that never happened."""
    assert proxy._docker_mem_mb("20.08GiB / 118.9GiB") > 20000
