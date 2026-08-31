"""Identify an app by the toolbox it hands the model, when it names itself no other way.

hermes drives the OpenAI Python SDK and stopped setting x-client-name, so its traffic arrived as
a generic "openai-sdk" alongside every other script and stopped being tellable apart — which
also silently broke every per-client measurement built on top of it.

The system prompt is no help here: hermes subagents run under their own personas ("# UX
Reviewer"), not a fixed banner. The tools are the stable signal.
"""
from ai_proxy import proxy as P


def _tools(*names):
    return {"tools": [{"type": "function", "function": {"name": n}} for n in names]}


UA = {"user-agent": "OpenAI/Python 2.24.0"}


def test_it_is_recognised_by_its_toolbox():
    assert P._detect_client_app(UA, _tools(
        "skill_manage", "skills_list", "session_search", "read_file")) == "hermes"


def test_an_explicit_header_still_wins():
    """hermes-safety names itself and must keep that label rather than being flattened."""
    assert P._detect_client_app(
        {**UA, "x-client-name": "hermes-safety"},
        _tools("skill_manage", "skills_list", "session_search")) == "hermes-safety"


def test_common_tool_names_do_not_match():
    """read_file and terminal would match half the agents in existence."""
    assert P._detect_client_app(UA, _tools("read_file", "terminal", "write_file")) == "openai-sdk"


def test_two_markers_are_not_enough():
    """A client sharing a name or two must not be mislabelled."""
    assert P._detect_client_app(UA, _tools("skill_manage", "session_search")) != "hermes"


def test_three_markers_are_enough():
    assert P._detect_client_app(UA, _tools(
        "skill_manage", "session_search", "delegate_task")) == "hermes"


def test_the_system_prompt_is_a_second_route():
    assert P._detect_client_app({}, {"messages": [
        {"role": "system", "content": "You are Hermes Agent, created by Nous Research."}]}) == "hermes"


def test_a_request_with_no_tools_is_unaffected():
    assert P._detect_client_app(UA, {"messages": []}) == "openai-sdk"


def test_claude_code_still_wins_on_its_own_marker():
    """Its billing-header fingerprint runs first and must not be displaced."""
    body = {"system": "cc_entrypoint=cli", **_tools("skill_manage", "skills_list", "session_search")}
    assert P._detect_client_app({}, body) == "claude-code"


def test_the_anthropic_tool_shape_is_read_too():
    body = {"tools": [{"name": n} for n in ("skill_manage", "skills_list", "session_search")]}
    assert P._detect_client_app(UA, body) == "hermes"
