"""What can be co-resident, answered for a machine reader.

Deciding which models fit together previously meant trying it: a sweep loaded two models, parked
them for two hours under keep_alive, and the third was refused for memory its own predecessors
were holding. The arithmetic that refuses a request is the same one exposed here, so a plan built
from this endpoint agrees with what the proxy will actually allow instead of being a second
opinion that drifts from it.
"""
from ai_proxy import proxy as P


def test_it_reports_the_memory_budget(client):
    d = client.get("/__proxy/api/capacity").json()
    mem = d["memory"]
    for k in ("total_gb", "available_gb", "reserved_gb", "usable_gb"):
        assert k in mem, k
    assert mem["usable_gb"] <= mem["available_gb"], "reserve must be held back"


def test_it_states_its_formula_and_assumptions(client):
    """A planner cannot check its own arithmetic against a number it cannot reproduce."""
    a = client.get("/__proxy/api/capacity").json()["assumptions"]
    assert "formula" in a and "weights_gb" in a["formula"]
    assert a["parallel_slots"] >= 1
    assert "over-estimate" in a["note"] or "over-estimated" in a["note"]


def test_each_model_carries_the_numbers_a_planner_needs(client):
    for e in client.get("/__proxy/api/capacity").json()["models"]:
        for k in ("id", "weights_gb", "kv_gb_per_1k_tokens", "max_context_alone",
                  "kv_measurable", "resident"):
            assert k in e, k


def test_an_unsizable_model_says_so_rather_than_guessing(client):
    """Some architectures cannot be priced by this build; reporting a number would be a lie."""
    for e in client.get("/__proxy/api/capacity").json()["models"]:
        if not e["kv_measurable"]:
            assert e["kv_gb_per_1k_tokens"] is None


def test_a_named_combination_is_judged_directly(client):
    """The interesting failure is a set whose members would each fit alone."""
    d = client.get("/__proxy/api/capacity?models=a-model,b-model&context=32768").json()
    c = d["combination"]
    assert c["models"] == ["a-model", "b-model"]
    assert c["context"] == 32768
    assert c["verdict"] in ("fits", "does not fit", "cannot judge")


def test_context_scales_the_requirement(client):
    """KV is the term that grows, and it is what actually decides co-residency."""
    small = client.get("/__proxy/api/capacity?models=x&context=4096").json()["combination"]
    large = client.get("/__proxy/api/capacity?models=x&context=262144").json()["combination"]
    assert large["required_gb"] >= small["required_gb"]
