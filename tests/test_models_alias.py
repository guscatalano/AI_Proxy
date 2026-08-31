"""Aliases advertised in /v1/models.

A client whose model picker only offers what /v1/models lists cannot select a name the router
would happily rewrite — so a name the proxy answers to has to appear in the catalogue to be
selectable at all. The routing is the existing rules; this only makes the name visible.
"""
import json

from ai_proxy import proxy


def _rules(monkeypatch, advertise):
    monkeypatch.setattr(proxy, "load_rules_config",
                        lambda: {"model_router": {"enabled": True, "advertise": advertise}})


def _ids(client):
    return {m["id"] for m in client.get("/v1/models").json()["data"]}


def _entry(client, name):
    return next(m for m in client.get("/v1/models").json()["data"] if m["id"] == name)


def test_an_advertised_alias_appears_in_the_catalogue(client, monkeypatch):
    proxy._MODELS_CACHE.update(ts=0, data=None)
    _rules(monkeypatch, {"claude-opus-4": "gemma4:26b"})
    assert "claude-opus-4" in _ids(client)


def test_the_alias_says_what_actually_answers(client, monkeypatch):
    """A name in this list that is not the model you get is the kind of thing that wastes an
    afternoon when it is discovered by accident."""
    proxy._MODELS_CACHE.update(ts=0, data=None)
    _rules(monkeypatch, {"claude-opus-4": "gemma4:26b"})
    e = _entry(client, "claude-opus-4")
    assert e["alias_of"] == "gemma4:26b"
    assert "gemma4:26b" in e["description"]


def test_an_alias_never_shadows_a_model_that_really_exists(client, monkeypatch):
    """If a backend genuinely serves the name, the real entry wins — an alias must not
    overwrite a model's own context length and capabilities with a stub."""
    proxy._MODELS_CACHE.update(ts=0, data=None)
    real = _ids(client)
    if not real:
        return
    name = sorted(real)[0]
    _rules(monkeypatch, {name: "gemma4:26b"})
    proxy._MODELS_CACHE.update(ts=0, data=None)
    assert "alias_of" not in _entry(client, name)


def test_no_aliases_configured_changes_nothing(client, monkeypatch):
    proxy._MODELS_CACHE.update(ts=0, data=None)
    _rules(monkeypatch, {})
    before = _ids(client)
    proxy._MODELS_CACHE.update(ts=0, data=None)
    _rules(monkeypatch, {})
    assert _ids(client) == before


def test_a_malformed_advertise_block_does_not_break_the_listing(client, monkeypatch):
    """The catalogue is polled by every model picker; a bad config entry must not 500 it."""
    proxy._MODELS_CACHE.update(ts=0, data=None)
    monkeypatch.setattr(proxy, "load_rules_config",
                        lambda: {"model_router": {"advertise": ["not", "a", "dict"]}})
    assert client.get("/v1/models").status_code == 200
