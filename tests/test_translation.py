"""Unit tests for the Anthropic <-> OpenAI request/response translation.

These pure functions are the trickiest part of the proxy; they carry the promise that
you can point an Anthropic client (Claude Code) at an OpenAI-compatible backend.
"""
from ai_proxy.proxy import _anthropic_to_openai_request, _openai_to_anthropic_response


def test_system_and_user_message():
    out = _anthropic_to_openai_request(
        {
            "model": "some-model",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    assert out["model"] == "some-model"
    assert out["max_tokens"] == 100
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert out["messages"][1] == {"role": "user", "content": "Hi"}


def test_system_list_blocks_are_joined():
    out = _anthropic_to_openai_request(
        {
            "system": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    assert out["messages"][0] == {"role": "system", "content": "line1\nline2"}


def test_stop_sequences_and_stream_options():
    out = _anthropic_to_openai_request(
        {
            "stream": True,
            "stop_sequences": ["STOP"],
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    assert out["stop"] == ["STOP"]
    assert out["stream"] is True
    # Streaming requests must ask upstream for usage so token counts survive translation.
    assert out["stream_options"] == {"include_usage": True}


def test_assistant_tool_use_becomes_tool_calls():
    out = _anthropic_to_openai_request(
        {
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "get_weather",
                            "input": {"city": "SF"},
                        },
                    ],
                },
            ]
        }
    )
    assistant = [m for m in out["messages"] if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["id"] == "toolu_abc"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert '"city"' in assistant["tool_calls"][0]["function"]["arguments"]


def test_tool_result_becomes_tool_role_message():
    out = _anthropic_to_openai_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "sunny",
                        }
                    ],
                }
            ]
        }
    )
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "toolu_abc"
    assert tool_msgs[0]["content"] == "sunny"


def test_does_not_mutate_input():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    snapshot = dict(body)
    _anthropic_to_openai_request(body)
    assert body == snapshot


def test_openai_response_text_and_usage():
    anth = _openai_to_anthropic_response(
        {
            "id": "chatcmpl-123",
            "model": "llama3",
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
    )
    assert anth["type"] == "message"
    assert anth["role"] == "assistant"
    assert anth["content"] == [{"type": "text", "text": "hello"}]
    assert anth["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert anth["id"].startswith("msg_")
    assert anth["model"] == "llama3"


def test_openai_response_tool_call_becomes_tool_use():
    anth = _openai_to_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    tool_uses = [b for b in anth["content"] if b["type"] == "tool_use"]
    assert tool_uses[0]["name"] == "get_weather"
    assert tool_uses[0]["input"] == {"city": "SF"}
    assert tool_uses[0]["id"] == "call_1"


def test_openai_response_malformed_tool_arguments_are_preserved():
    anth = _openai_to_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c", "function": {"name": "f", "arguments": "{not json"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    tool_use = [b for b in anth["content"] if b["type"] == "tool_use"][0]
    assert tool_use["input"] == {"_raw_arguments": "{not json"}
