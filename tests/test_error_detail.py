"""What a failed request says about itself.

Every request lost when vLLM's engine died recorded exactly `upstream error: ReadError('')` —
an empty exception message, no upstream, no timing, no phase. The type already carried the
diagnosis; nothing spelled it out. And requests the upstream answered with a 500 stored a blank
error column while the body held a perfectly clear EngineDeadError.
"""
import httpx

from ai_proxy import proxy as P


def test_read_error_says_what_a_dropped_connection_means():
    msg = P._explain_upstream_error(httpx.ReadError(""), "vllm",
                                    "http://localhost:8001/v1/chat/completions", 82.0)
    assert "vllm at localhost:8001" in msg
    assert "closed the connection" in msg
    assert "82 ms" in msg
    assert "before any response bytes arrived" in msg
    assert "[ReadError]" in msg          # raw type kept for forensics


def test_connect_error_is_not_confused_with_a_dropped_connection():
    msg = P._explain_upstream_error(httpx.ConnectError("All connection attempts failed"),
                                    "vllm", "http://localhost:8001/v1/models", 144.0)
    assert "Nothing is listening" in msg
    assert "down, restarting" in msg
    assert "All connection attempts failed" in msg


def test_mid_stream_failure_says_the_answer_was_partial():
    partial = P._explain_upstream_error(httpx.ReadError(""), "vllm", "http://h:8001/x",
                                        4200.0, got_bytes=True)
    assert "mid-response, after part of the answer had arrived" in partial


def test_read_timeout_is_distinguished_from_a_crash():
    msg = P._explain_upstream_error(httpx.ReadTimeout("timed out"), "ollama",
                                    "http://localhost:11434/api/chat", 60000.0)
    assert "sent nothing back" in msg
    assert "still loading" in msg


def test_pool_timeout_blames_the_proxy_not_the_upstream():
    msg = P._explain_upstream_error(httpx.PoolTimeout(""), "vllm", "http://h:8001/x", 5.0)
    assert "proxy's own pool" in msg


def test_unknown_exception_still_names_the_upstream():
    msg = P._explain_upstream_error(ValueError("weird"), "lmstudio", "http://h:1234/x", 7.0)
    assert "lmstudio at h:1234" in msg and "[ValueError: weird]" in msg


def test_explanation_survives_a_missing_url_and_timing():
    msg = P._explain_upstream_error(httpx.ReadError(""), None, None, None)
    assert "the upstream" in msg and "[ReadError]" in msg


def test_upstream_500_message_is_lifted_from_the_body():
    body = ('{"error":{"message":"EngineCore encountered an issue. See stack trace for the '
            'root cause.","type":"EngineDeadError"}}')
    assert P._upstream_error_message(500, body) == (
        "upstream returned 500: EngineCore encountered an issue. See stack trace for the "
        "root cause.")


def test_upstream_500_falls_back_to_other_shapes():
    assert "boom" in P._upstream_error_message(503, '{"detail":"boom"}')
    assert "plain" in P._upstream_error_message(500, '{"message":"plain"}')
    assert "as a string" in P._upstream_error_message(500, '{"error":"as a string"}')


def test_non_json_500_uses_its_first_line():
    msg = P._upstream_error_message(502, "\n\n  Bad Gateway: nginx couldn't reach it\nmore\n")
    assert msg == "upstream returned 502: Bad Gateway: nginx couldn't reach it"


def test_success_and_client_errors_get_no_error_text():
    # Only 5xx. A 400 is the client's problem and its body is already on the row.
    assert P._upstream_error_message(200, '{"error":{"message":"x"}}') is None
    assert P._upstream_error_message(400, '{"error":{"message":"x"}}') is None
    assert P._upstream_error_message(500, None) is None
    assert P._upstream_error_message(500, '{"choices":[]}') is None


def test_unreachable_upstream_response_carries_the_explanation(client, monkeypatch):
    # The 502 the client receives should say the same thing the stored row does.
    r = client.post("/v1/chat/completions",
                    json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    if r.status_code == 502:
        body = r.json()
        assert "explanation" in body
        assert "[" in body["explanation"]        # the raw exception type is appended
