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

HOW MANY ATTEMPTS, AND AT WHAT SIZE. A clean run never proves the rate is zero; it puts a
ceiling on it. With no failures in n attempts the 95% upper bound is 3/n — the rule of three:

    n =   120  ->  rate < 2.5%          n =  600  ->  rate < 0.5%
    n = 1,200  ->  rate < 0.25%         n = 3000  ->  rate < 0.1%

12 tasks x runs=10 is 120 attempts, which bounds the rate at 2.5% — the very rate being tested
for, so the first clean run of this suite established nothing at all. Pick n from the bound you
want: `runs=50` (600 attempts) is the first setting that says something the incident did not
already say.

SIZE MATTERS MORE THAN COUNT. The loss was measured on claude-code, whose prompts start around
70k tokens, and 3,000 attempts at 2k tokens would still say nothing about that. Use the
`prompt_tokens` axis to ask these same tasks from inside a real-sized context — `prompt_tokens:
70000` puts them where the failure actually lived. It costs what it costs: ~35s per attempt at
70k on this box, so 600 attempts is roughly six hours, and that is the honest price of a 0.5%
bound under realistic conditions.

STILL NOT COVERED: the Anthropic bridge. claude-code's traffic is translated on the way through,
and the bench posts OpenAI-shape requests to /v1/chat/completions, so the bridged path this bug
was measured on is not exercised here at any size or count.

AND THE CHEAPER ANSWER: production already measures this continuously and under perfect
conditions. Every recovery lands as a `tool_stream` gate rule and every undelivered turn as
`empty_output`, across thousands of real requests a week. This suite is the pre-deployment gate;
those counters are the evidence.
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


def _task(tid, prompt, tools, note, desc):
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
        # desc (what it does) and note (why it is in the suite) are both required: the guide
        # endpoint asserts on each, and a task carrying only one fails the whole listing.
        "desc": desc,
    }


TRANSPORT_TASKS = [
    _task("ts_read_simple",
          "Read the file /etc/hosts and tell me what is in it. Use the read_file tool.",
          [_READ_TOOL], "single tool, explicit instruction", "Call read_file when told to, one tool offered"),
    _task("ts_read_implicit",
          "What are the contents of /var/log/syslog?",
          [_READ_TOOL], "single tool, no explicit instruction to call it", "Call read_file without being told to name it"),
    _task("ts_search_simple",
          "Find every place the string TODO appears in this codebase. Use the search tool.",
          [_SEARCH_TOOL], "a different tool, in case one name is special", "Call a differently-named tool"),
    _task("ts_run_simple",
          "Show me the current directory listing. Use the run_command tool.",
          [_RUN_TOOL], "command-shaped arguments", "Call run_command with a shell argument"),
    _task("ts_choose_read",
          "I need to see what is inside /etc/passwd. Pick the right tool and use it.",
          _MULTI, "three tools offered, one obviously right", "Pick read_file from three offered tools"),
    _task("ts_choose_search",
          "Where in the code is the function parse_config defined? Pick the right tool.",
          _MULTI, "three tools offered, search is the fit", "Pick search from three offered tools"),
    _task("ts_choose_run",
          "Tell me how much disk space is free on this machine. Pick the right tool.",
          _MULTI, "three tools offered, command is the fit", "Pick run_command from three offered tools"),
    _task("ts_long_args",
          "Search for this exact phrase: "
          "'the quick brown fox jumps over the lazy dog and keeps on running for a while'. "
          "Use the search tool.",
          [_SEARCH_TOOL], "long argument string spans several deltas", "Deliver a call whose arguments span several deltas"),
    _task("ts_path_args",
          "Read /home/user/projects/deeply/nested/directory/structure/config.settings.json "
          "and summarise it. Use read_file.",
          [_READ_TOOL], "argument full of punctuation the parser must not mangle", "Deliver a call with a punctuation-heavy path argument"),
    _task("ts_after_prose",
          "Explain in one sentence what a hosts file is, then read /etc/hosts using the tool.",
          [_READ_TOOL], "prose BEFORE the call — the call arrives late in the stream", "Deliver a call that arrives late, after prose"),
    _task("ts_quoted_args",
          "Search for the string \"key=\\\"value\\\"\" in the codebase using the search tool.",
          [_SEARCH_TOOL], "nested quotes, which broke argument JSON before", "Deliver a call with nested quotes in its arguments"),
    _task("ts_unicode_args",
          "Read the file /tmp/notes—café—2026.txt with read_file.",
          [_READ_TOOL], "non-ASCII in arguments", "Deliver a call with non-ASCII arguments"),
]
