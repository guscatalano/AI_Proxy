"""Does a tool call survive the trip? A property of the backend, not of the model.

Every other suite here measures what a model knows. This one measures whether the transport
delivers what the model decided, because on 2026-08-17 it did not: Ollama's /v1 endpoint drops
tool calls from *streamed* replies, sending `{"delta":{"content":""},"finish_reason":"stop"}`
with the call missing. Roughly 2.5% of claude-code turns arrived empty and read as the agent
giving up.

The benchmark suite could not see it, and the reason is worth stating plainly: single-turn tasks
stream but offer no tools, and agent episodes offer tools but run stream:False. Tools-over-a-
stream was the one combination nothing exercised, so every model scored well on agent-v2 through
the entire period the bug was live. These tasks are marked `stream_tools`, which keeps them on
the streaming single-turn path *with* tools attached.

Grading is deliberately blunt: a task passes if a tool call arrives. The prompts are trivial on
purpose — any model that can call a tool at all should score 100%, so anything less is the
transport losing calls rather than the model failing to make them. A model genuinely incapable of
tool calling scores 0% everywhere, which is a different and equally visible shape.

Detection is statistical. At a ~2.5% loss rate, 12 tasks x runs=1 will usually see nothing;
runs=10 gives 120 attempts and about a 95% chance of catching at least one. Run it with runs
high, or read a clean result as "no gross breakage" rather than "no loss".
"""

# One tool, one obvious reason to call it. Anything cleverer risks measuring whether the model
# chose well, which is agent-v2's job and would blur what a failure here means.
_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute file path"}},
            "required": ["path"],
        },
    },
}

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the codebase for a string.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

_MULTI = [_READ_TOOL, _SEARCH_TOOL, _RUN_TOOL]


def _task(tid, prompt, tools, note):
    return {
        "id": tid,
        "category": "transport",
        # `lang` gates grading. A value missing from _bench_lang_available is SKIPPED rather
        # than failed — silently, and two whole suites once vanished from a full run that way —
        # so "toolstream" is registered there alongside the other in-process grading modes.
        "lang": "toolstream",
        # `entry` and `cases` are required shape, not grading inputs: the suites listing
        # endpoint reads both for every task and a task lacking them raises KeyError there,
        # taking the whole listing down rather than just this suite.
        "entry": "tool_call",
        "cases": [{"expect": "a tool call delivered over the stream"}],
        "stream_tools": True,
        "tools": tools,
        "prompt": prompt,
        "note": note,
    }


TRANSPORT_TASKS = [
    _task("ts_read_simple",
          "Read the file /etc/hosts and tell me what is in it. Use the read_file tool.",
          [_READ_TOOL], "single tool, explicit instruction"),
    _task("ts_read_implicit",
          "What are the contents of /var/log/syslog?",
          [_READ_TOOL], "single tool, no explicit instruction to call it"),
    _task("ts_search_simple",
          "Find every place the string TODO appears in this codebase. Use the search tool.",
          [_SEARCH_TOOL], "a different tool, in case one name is special"),
    _task("ts_run_simple",
          "Show me the current directory listing. Use the run_command tool.",
          [_RUN_TOOL], "command-shaped arguments"),
    _task("ts_choose_read",
          "I need to see what is inside /etc/passwd. Pick the right tool and use it.",
          _MULTI, "three tools offered, one obviously right"),
    _task("ts_choose_search",
          "Where in the code is the function parse_config defined? Pick the right tool.",
          _MULTI, "three tools offered, search is the fit"),
    _task("ts_choose_run",
          "Tell me how much disk space is free on this machine. Pick the right tool.",
          _MULTI, "three tools offered, command is the fit"),
    _task("ts_long_args",
          "Search for this exact phrase: "
          "'the quick brown fox jumps over the lazy dog and keeps on running for a while'. "
          "Use the search tool.",
          [_SEARCH_TOOL], "long argument string spans several deltas"),
    _task("ts_path_args",
          "Read /home/user/projects/deeply/nested/directory/structure/config.settings.json "
          "and summarise it. Use read_file.",
          [_READ_TOOL], "argument full of punctuation the parser must not mangle"),
    _task("ts_after_prose",
          "Explain in one sentence what a hosts file is, then read /etc/hosts using the tool.",
          [_READ_TOOL], "prose BEFORE the call — the call arrives late in the stream"),
    _task("ts_quoted_args",
          "Search for the string \"key=\\\"value\\\"\" in the codebase using the search tool.",
          [_SEARCH_TOOL], "nested quotes, which broke argument JSON before"),
    _task("ts_unicode_args",
          "Read the file /tmp/notes—café—2026.txt with read_file.",
          [_READ_TOOL], "non-ASCII in arguments"),
]
