"""Surface detection: which execution context an agentic client was driven from.

One client_app can front several surfaces — hermes arrives as a Discord bot, a scheduled
cron job, and an interactive CLI session with identical IP/UA/x-client-name. The context
leaks into the request anyway (guild blocks, "no user present" cron prompts, local exec
toolsets); these tests pin the fingerprints against real shapes observed in live traffic.
"""
import json
import time

from ai_proxy import proxy as P


def _tools(*names):
    return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]


def _body(sys_text, tools=None):
    b = {"messages": [{"role": "system", "content": sys_text},
                      {"role": "user", "content": "hi"}]}
    if tools:
        b["tools"] = tools
    return b


def test_cron_prompt_wins():
    b = _body("You are Hermes Agent. You are running as a scheduled cron job. "
              "There is no user present — you cannot ask questions.",
              _tools("browser_click", "discord", "execute_code"))
    # cron outranks the discord tools: the discord PLUGIN being loaded doesn't change
    # that nobody is driving.
    assert P._detect_surface({}, b) == "cron"


def test_guild_block_means_discord():
    b = _body("You are Hermes Agent.\n## Current Session Context\n- Guild: `1523167625293729912`\n"
              "- Channel: general")
    assert P._detect_surface({}, b) == "discord"


def test_discord_tools_alone_mean_discord():
    """DMs carry the discord tools without a guild block."""
    b = _body("You are Hermes Agent.", _tools("browser_click", "discord", "discord_admin"))
    assert P._detect_surface({}, b) == "discord"


def test_local_exec_toolset_means_cli():
    b = _body("You are Hermes Agent.",
              _tools("execute_code", "read_file", "patch", "session_search"))
    assert P._detect_surface({}, b) == "cli"


def test_header_override_beats_fingerprints():
    b = _body("## Current Session Context\n- Guild: `123`")
    assert P._detect_surface({"x-client-surface": "Discord DM"}, b) == "discord-dm"


def test_plain_traffic_stays_unlabeled():
    """A chat completion with no agentic markers must NOT get a guessed label."""
    assert P._detect_surface({}, _body("You are a helpful assistant.")) is None
    assert P._detect_surface({}, {"messages": [{"role": "user", "content": "hi"}]}) is None
    assert P._detect_surface(None, None) is None


def test_anthropic_style_system_field_is_read():
    b = {"system": [{"type": "text", "text": "You are running as a scheduled cron job."}],
         "messages": [{"role": "user", "content": "go"}]}
    assert P._detect_surface({}, b) == "cron"


def test_surface_is_stored_and_flows_to_conversations(client):
    """End to end through the DB: an ingested row carries surface, and the conversations
    listing aggregates it — the browsing views read from there."""
    conn = P.db()
    conn.execute(
        "INSERT INTO requests (id, ts, method, path, upstream_url, request_headers, model, "
        "client_ip, conversation_id, turn_index, client_app, surface) VALUES "
        "('r_sf1', ?, 'POST', '/v1/chat/completions', 'http://x/v1/chat/completions', '{}', "
        "'gemma4', '127.0.0.1', 'conv_sf', 1, 'hermes', 'discord')", (time.time(),))
    conn.commit()
    conn.close()
    d = client.get("/__proxy/api/conversations").json()
    row = next(i for i in d["items"] if i["conversation_id"] == "conv_sf")
    assert row["surfaces"] == "discord"
    turns = client.get("/__proxy/api/conversations/conv_sf").json()["turns"]
    assert turns[0]["surface"] == "discord"
    conn = P.db()
    conn.execute("DELETE FROM requests WHERE id='r_sf1'")
    conn.commit()
    conn.close()
