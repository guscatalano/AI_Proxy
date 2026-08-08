"""Credentials are fingerprinted, never stored.

Callers here share one IP and often one client label, so the key is the only thing that
reliably separates them — which is a reason to record its IDENTITY, not its value. The
request log is copied, archived and shared; a bearer token in it is a bearer token in all of
those. These tests pin both halves: the value never lands in the database, and the
fingerprint that replaces it is stable enough to group a caller's traffic.
"""
import json

from ai_proxy import proxy as P


class _StubRequest:
    """Just enough Request for the logging path. Going through the real endpoint would
    reach for an upstream that is not running in CI and wait out its timeout — the write
    path is what these tests are about, not the proxying."""

    def __init__(self, headers):
        self.headers = headers
        self.method = "POST"
        self.client = type("C", (), {"host": "10.1.2.3"})()


def test_a_credential_is_never_written_verbatim(client):
    key = "sk-live-9f3aQ2xLb7ZzKKn41vv"
    body = {"model": "qwen3:4b", "messages": [{"role": "user", "content": "hi"}],
            "x-marker": "keytest"}
    req = _StubRequest({"authorization": f"Bearer {key}", "x-client-name": "keytest",
                        "content-type": "application/json"})
    P._save_pending("r_credtest", req, "v1/chat/completions",
                    "http://localhost:11434/v1/chat/completions",
                    json.dumps(body), body, "qwen3:4b", False, upstream="ollama")
    conn = P.db()
    row = conn.execute("SELECT request_headers, api_key_fp, client_app FROM requests "
                       "WHERE id='r_credtest'").fetchone()
    conn.execute("DELETE FROM requests WHERE id='r_credtest'")
    conn.commit()
    conn.close()
    assert row is not None, "the request was not logged at all"
    assert row["client_app"] == "keytest", "x-client-name still identifies the caller"
    blob = row["request_headers"] or ""
    assert key not in blob, "the api key was stored verbatim"
    assert "[redacted]" in blob
    assert row["api_key_fp"], "identification was dropped along with the secret"
    assert row["api_key_fp"].endswith("41vv"), "the last four are kept to eyeball a match"
    assert key not in row["api_key_fp"]


def test_the_fingerprint_is_stable_and_distinguishes_callers():
    a = {"authorization": "Bearer sk-live-aaaaaaaaaaaaaaaa1111"}
    b = {"authorization": "Bearer sk-live-bbbbbbbbbbbbbbbb2222"}
    assert P._credential_fingerprint(a) == P._credential_fingerprint(dict(a))
    assert P._credential_fingerprint(a) != P._credential_fingerprint(b)
    # The scheme prefix must not change the identity of the key behind it.
    assert (P._credential_fingerprint({"authorization": "Bearer sk-live-aaaaaaaaaaaaaaaa1111"})
            == P._credential_fingerprint({"x-api-key": "sk-live-aaaaaaaaaaaaaaaa1111"}))


def test_placeholder_credentials_are_not_fingerprinted():
    """Local backends accept anything, so every caller sends the same non-secret. A column
    full of one shared value looks like identification and is not."""
    for placeholder in ("Bearer no-key-required", "Bearer not-required", "Bearer none",
                        "Bearer dummy", "Bearer NO_KEY_NEEDED"):
        assert P._credential_fingerprint({"authorization": placeholder}) is None, placeholder
    assert P._credential_fingerprint({"authorization": "Bearer short"}) is None
    assert P._credential_fingerprint({}) is None
    assert P._credential_fingerprint(None) is None


def test_every_credential_header_shape_is_covered():
    for hdr in ("authorization", "X-Api-Key", "api-key", "proxy-authorization",
                "cookie", "x-auth-token"):
        h = {hdr: "sk-live-aaaaaaaaaaaaaaaa1111", "user-agent": "curl/8"}
        assert P._credential_fingerprint(h), hdr
        out = P._redact_credential_headers(h)
        assert out[hdr] == "[redacted]", hdr
        assert out["user-agent"] == "curl/8", "ordinary headers must survive intact"


def test_hindsight_is_identified_by_what_it_asks_for(client):
    """It calls through the stock AsyncOpenAI SDK with no x-client-name, so it sat in the
    generic 'openai-sdk' bucket while being the second-busiest thing on the box."""
    consolidate = {"messages": [
        {"role": "system", "content": "You are a memory consolidation system. Synthesize "
                                      "new facts into observations."},
        {"role": "user", "content": "Return valid json only."}]}
    extract = {"messages": [
        {"role": "system", "content": "Extract SIGNIFICANT facts from text. Be SELECTIVE."}]}
    title = {"messages": [
        {"role": "system", "content": "Generate a short, descriptive title (3-7 words)"}]}
    ua = {"user-agent": "AsyncOpenAI/Python 2.24.0", "x-stainless-lang": "python"}
    assert P._detect_client_app(ua, consolidate) == "hindsight-consolidate"
    assert P._detect_client_app(ua, extract) == "hindsight-extract"
    assert P._detect_client_app(ua, title) == "title-generator"
    # Anything else on the same SDK keeps its generic label rather than being guessed at.
    assert P._detect_client_app(ua, {"messages": [{"role": "user", "content": "hi"}]}) \
        in ("openai-sdk", "openai-python")
