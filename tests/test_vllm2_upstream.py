"""A second vLLM slot, so two models can be resident and a rule can choose between them.

One vLLM process serves one model. Keeping qwen3-coder-next and nemotron both in memory needs
two processes on two ports, and the router can only send traffic to an upstream that exists in
PROVIDERS — which is what builds _UPSTREAM_BASES.

It was control="none" while the docker path resolved "the container publishing VLLM_URL" —
managing this slot would have acted on the other one's container. The helpers take the slot's
URL now, so it is a full twin; see test_vllm_slots.py for the per-slot resolution itself.
"""
from ai_proxy import proxy as P


def test_the_second_slot_exists():
    assert "vllm2" in P.PROVIDERS


def test_it_points_at_a_different_port_than_the_first():
    assert P.PROVIDERS["vllm2"].base_url != P.PROVIDERS["vllm"].base_url
    assert P.PROVIDERS["vllm2"].base_url == P.VLLM2_URL


def test_the_router_can_target_it():
    """_UPSTREAM_BASES is built from PROVIDERS; a rule naming an upstream absent from it is
    silently ignored and the request goes to the path default instead."""
    bases = {n: p.base_url for n, p in P.PROVIDERS.items()}
    assert bases.get("vllm2") == P.VLLM2_URL


def test_it_is_proxy_managed_like_the_first():
    """Start/stop, auto-load and the residency snapshot all key off control; leaving this at
    "none" is what made the second slot invisible to every one of them."""
    assert P.PROVIDERS["vllm2"].control == "docker"


def test_the_first_slot_is_unchanged():
    assert P.PROVIDERS["vllm"].base_url == P.VLLM_URL
    assert P.PROVIDERS["vllm"].control == "docker"
