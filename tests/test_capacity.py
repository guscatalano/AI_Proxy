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


# --- "alone" means alone on the box ----------------------------------------------------------
#
# Every entry reported max_context_alone=0 while vLLM held 80 of the box's 121 GB, because the
# number was measured against free memory rather than total. That is the moment a planner asks
# the question, and the answer it needs is what fits once the box is cleared — otherwise a model
# that cannot be co-resident at any size is indistinguishable from one that is merely waiting.


class _FakeOllama:
    """Two models with known weights, so the arithmetic has something to bite on."""
    TAGS = {"big:latest": 22 * 1024 * 1024 * 1024, "small:latest": 4 * 1024 * 1024 * 1024}
    INFO = {"general.architecture": "arch", "arch.block_count": 40,
            "arch.attention.head_count": 16, "arch.attention.key_length": 128,
            "arch.attention.value_length": 128}

    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def get(self, url, **k):
        payload = ({"models": [{"name": n, "size": s} for n, s in _FakeOllama.TAGS.items()]}
                   if url.endswith("/api/tags") else {"models": []})

        class R:
            status_code = 200
            @staticmethod
            def json(): return payload
        return R()

    async def post(self, url, json=None, **k):
        class R:
            status_code = 200
            @staticmethod
            def json(): return {"model_info": _FakeOllama.INFO}
        return R()


def _busy_box(monkeypatch, total_gb=121, avail_gb=10):
    monkeypatch.setattr(P.httpx, "AsyncClient", _FakeOllama)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": total_gb * 1024,
                                                     "avail_mb": avail_gb * 1024,
                                                     "used_mb": (total_gb - avail_gb) * 1024})


def test_a_busy_box_still_reports_what_would_fit_alone(client, monkeypatch):
    _busy_box(monkeypatch)
    entries = {e["id"]: e for e in client.get("/__proxy/api/capacity").json()["models"]}
    big = entries["big:latest"]
    assert big["max_context_now"] == 0, "nothing fits beside 111 GB of someone else's memory"
    assert big["max_context_alone"] > 0, "but the box could hold it if it were cleared"


def test_the_two_answers_are_both_reported_for_a_combination(client, monkeypatch):
    _busy_box(monkeypatch)
    c = client.get("/__proxy/api/capacity?models=big:latest&context=32768").json()["combination"]
    assert c["fits"] is False and c["fits_on_empty_box"] is True
    assert c["usable_gb_on_empty_box"] > c["usable_gb"]
    assert c["verdict"] == "does not fit now, but would on an idle box"


def test_a_model_too_large_for_the_machine_is_not_merely_busy(client, monkeypatch):
    """The distinction that matters: no amount of waiting or evicting makes this one loadable."""
    _busy_box(monkeypatch, total_gb=121, avail_gb=100)
    c = client.get("/__proxy/api/capacity?models=big:latest&context=262144").json()["combination"]
    assert c["fits"] is False and c["fits_on_empty_box"] is False
    assert c["verdict"] == "does not fit"


def test_an_idle_box_gives_the_same_answer_to_both_questions(client, monkeypatch):
    _busy_box(monkeypatch, total_gb=121, avail_gb=121)
    e = [x for x in client.get("/__proxy/api/capacity").json()["models"]
         if x["id"] == "small:latest"][0]
    assert e["max_context_alone"] == e["max_context_now"]
