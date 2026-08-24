"""A partial write to the rules API must be visible and undoable.

POST replaces the stored document rather than merging it, because the dashboard posts the whole
config and deleting a section has to be possible. From a shell that is a trap: posting one
section deleted the model_router rule that sent qwen3-coder-next to vLLM, and the main client's
next request 503'd against Ollama, which does not have room for that model.
"""
from ai_proxy import proxy as P


def _post(client, doc):
    return client.post("/__proxy/api/rules", json=doc)


def test_a_write_that_drops_sections_says_so(client):
    _post(client, {"model_router": {"enabled": True}, "ollama_options": {"enabled": True}})
    r = _post(client, {"ollama_options": {"enabled": True}})
    body = r.json()
    assert body["ok"] is True
    assert body["dropped_keys"] == ["model_router"]
    assert "model_router" in body["warning"]


def test_a_full_write_is_not_flagged(client):
    _post(client, {"model_router": {"enabled": True}})
    body = _post(client, {"model_router": {"enabled": False}}).json()
    assert "dropped_keys" not in body and "warning" not in body


def test_the_previous_document_can_be_read_back(client):
    _post(client, {"model_router": {"enabled": True, "rules": [{"if": {"from_model": "m"},
                                                                "upstream": "vllm"}]}})
    _post(client, {"ollama_options": {"enabled": True}})
    versions = client.get("/__proxy/api/rules/history").json()["versions"]
    assert versions, "the overwritten document must still be recoverable"
    newest = versions[0]
    assert newest["keys"] == ["model_router"]
    assert newest["config"]["model_router"]["rules"][0]["upstream"] == "vllm"


def test_history_is_newest_first_and_bounded(client):
    for i in range(P._RULES_HISTORY_KEEP + 4):
        _post(client, {"model_router": {"enabled": True, "aliases": {"n": str(i)}}})
    versions = client.get("/__proxy/api/rules/history").json()["versions"]
    assert len(versions) == P._RULES_HISTORY_KEEP
    seen = [v["config"]["model_router"]["aliases"]["n"] for v in versions]
    assert seen == sorted(seen, key=int, reverse=True), "newest first"


def test_restoring_a_version_puts_the_rule_back(client):
    """The whole point: the qwen3-coder-next rule comes back without being retyped."""
    good = {"model_router": {"enabled": True,
                             "rules": [{"if": {"from_model": "qwen3-coder-next"},
                                        "upstream": "vllm"}]}}
    _post(client, good)
    _post(client, {"ollama_options": {"enabled": True}})     # the destructive partial write
    lost = client.get("/__proxy/api/rules/history").json()["versions"][0]["config"]
    _post(client, lost)
    live = client.get("/__proxy/api/rules").json()["config"]["model_router"]
    assert live["rules"][0]["upstream"] == "vllm"
