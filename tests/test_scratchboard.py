"""The shared scratchboard.

Everything else this dashboard shows is captured traffic, so it passes through the PII gate
and a viewer on another subnet sees placeholders. These notes are written BY the viewers, and
a board each subnet sees a different version of is not a shared board — so this endpoint does
not redact, and that is the feature rather than an oversight.
"""
import ai_proxy
from ai_proxy import proxy


def _clear():
    conn = proxy.db()
    conn.execute("DELETE FROM scratchboard")
    conn.commit()
    conn.close()


def test_a_note_posted_comes_back(client):
    _clear()
    r = client.post("/__proxy/api/scratchboard",
                    json={"text": "vLLM is down while benchmarks run", "author": "gus"})
    assert r.status_code == 200 and r.json()["ok"]
    items = client.get("/__proxy/api/scratchboard").json()["items"]
    assert len(items) == 1
    assert items[0]["text"] == "vLLM is down while benchmarks run"
    assert items[0]["author"] == "gus"


def test_notes_are_newest_first(client):
    _clear()
    for t in ("first", "second", "third"):
        client.post("/__proxy/api/scratchboard", json={"text": t})
    texts = [i["text"] for i in client.get("/__proxy/api/scratchboard").json()["items"]]
    assert texts[0] == "third", "a board reads newest-first or it reads as a log"


def test_pinned_notes_sit_above_newer_ones(client):
    """Pinning is how you say 'this one is not scratch'."""
    _clear()
    client.post("/__proxy/api/scratchboard", json={"text": "standing context", "pinned": True})
    client.post("/__proxy/api/scratchboard", json={"text": "later note"})
    items = client.get("/__proxy/api/scratchboard").json()["items"]
    assert items[0]["text"] == "standing context"
    assert items[0]["pinned"] == 1


def test_an_empty_note_is_refused(client):
    _clear()
    assert client.post("/__proxy/api/scratchboard", json={"text": "   "}).status_code == 400
    assert client.get("/__proxy/api/scratchboard").json()["items"] == []


def test_an_oversized_note_is_refused_with_its_size(client):
    _clear()
    r = client.post("/__proxy/api/scratchboard",
                    json={"text": "x" * (proxy._SCRATCH_MAX_CHARS + 1)})
    assert r.status_code == 413
    assert str(proxy._SCRATCH_MAX_CHARS) in r.json()["error"]


def test_a_note_can_be_deleted(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "temporary"}).json()["id"]
    assert client.delete(f"/__proxy/api/scratchboard/{nid}").json()["removed"] == 1
    assert client.get("/__proxy/api/scratchboard").json()["items"] == []


def test_pinning_is_reversible(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    client.post(f"/__proxy/api/scratchboard/{nid}/pin", json={"pinned": True})
    assert client.get("/__proxy/api/scratchboard").json()["items"][0]["pinned"] == 1
    client.post(f"/__proxy/api/scratchboard/{nid}/pin", json={"pinned": False})
    assert client.get("/__proxy/api/scratchboard").json()["items"][0]["pinned"] == 0


def test_the_board_is_not_redacted_by_subnet(client):
    """The whole point. Request bodies are hidden from cross-subnet viewers; notes are not."""
    _clear()
    client.post("/__proxy/api/scratchboard", json={"text": "visible everywhere"})
    body = client.get("/__proxy/api/scratchboard").json()
    assert body["items"][0]["text"] == "visible everywhere"
    assert "_pii_redacted" not in body["items"][0]
    assert proxy.REDACT_PLACEHOLDER not in str(body)


def test_mcp_exposes_a_read_tool_without_write_permission():
    names = [t["name"] for t in proxy.MCP_TOOLS]
    assert "read_scratchboard" in names
    read = next(t for t in proxy.MCP_TOOLS if t["name"] == "read_scratchboard")
    assert not read.get("_write"), "reading the board must not require MCP_ALLOW_WRITE"
    write = next(t for t in proxy.MCP_TOOLS if t["name"] == "write_scratchboard")
    assert write.get("_write"), "posting must stay behind the write gate"


# --- replies -----------------------------------------------------------------------------


def test_a_reply_nests_under_its_note(client):
    _clear()
    root = client.post("/__proxy/api/scratchboard", json={"text": "run this command"}).json()["id"]
    client.post("/__proxy/api/scratchboard", json={"text": "here is the output", "parent_id": root})
    items = client.get("/__proxy/api/scratchboard").json()["items"]
    assert len(items) == 1, "a reply must not appear as its own note"
    assert [r["text"] for r in items[0]["replies"]] == ["here is the output"]


def test_replies_read_forwards_while_the_board_reads_backwards(client):
    _clear()
    root = client.post("/__proxy/api/scratchboard", json={"text": "q"}).json()["id"]
    for t in ("first answer", "second answer"):
        client.post("/__proxy/api/scratchboard", json={"text": t, "parent_id": root})
    reps = client.get("/__proxy/api/scratchboard").json()["items"][0]["replies"]
    assert [r["text"] for r in reps] == ["first answer", "second answer"]


def test_replying_to_a_reply_stays_on_the_same_thread(client):
    """One level only — otherwise a board becomes a tree nobody can render."""
    _clear()
    root = client.post("/__proxy/api/scratchboard", json={"text": "root"}).json()["id"]
    rep = client.post("/__proxy/api/scratchboard",
                      json={"text": "reply", "parent_id": root}).json()["id"]
    client.post("/__proxy/api/scratchboard", json={"text": "reply to reply", "parent_id": rep})
    items = client.get("/__proxy/api/scratchboard").json()["items"]
    assert len(items) == 1
    assert len(items[0]["replies"]) == 2, "the second reply should join the thread, not nest"


def test_replying_to_a_missing_note_is_refused(client):
    _clear()
    r = client.post("/__proxy/api/scratchboard", json={"text": "x", "parent_id": "nope"})
    assert r.status_code == 404


def test_deleting_a_note_takes_its_replies(client):
    _clear()
    root = client.post("/__proxy/api/scratchboard", json={"text": "root"}).json()["id"]
    client.post("/__proxy/api/scratchboard", json={"text": "answer", "parent_id": root})
    assert client.delete(f"/__proxy/api/scratchboard/{root}").json()["removed"] == 2
    assert client.get("/__proxy/api/scratchboard").json()["items"] == []


def test_a_reply_cannot_be_pinned_to_the_top(client):
    """Pinning is about standing context; a reply's place is under its note."""
    _clear()
    root = client.post("/__proxy/api/scratchboard", json={"text": "root"}).json()["id"]
    client.post("/__proxy/api/scratchboard",
                json={"text": "reply", "parent_id": root, "pinned": True})
    reps = client.get("/__proxy/api/scratchboard").json()["items"][0]["replies"]
    assert reps[0]["pinned"] == 0


def test_mcp_write_tool_accepts_reply_to():
    tool = next(t for t in proxy.MCP_TOOLS if t["name"] == "write_scratchboard")
    assert "reply_to" in tool["inputSchema"]["properties"]


# --- attachments -------------------------------------------------------------------------

import base64 as _b64


def _attach(client, note_id, name, blob, mime="text/plain"):
    return client.post(f"/__proxy/api/scratchboard/{note_id}/files",
                       json={"name": name, "mime": mime,
                             "data_b64": _b64.b64encode(blob).decode()})


def test_a_file_attaches_and_downloads_byte_for_byte(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "log attached"}).json()["id"]
    blob = b"exit code 126\nZugriff verweigert\x00\xff binary too"
    fid = _attach(client, nid, "hermes.log", blob).json()["id"]
    got = client.get(f"/__proxy/api/scratchboard/files/{fid}")
    assert got.status_code == 200
    assert got.content == blob, "a log that changes in transit is worse than no log"


def test_the_board_lists_attachments_without_shipping_the_bytes(client):
    """The page polls constantly; sending file contents in the listing would be megabytes."""
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    _attach(client, nid, "big.bin", b"x" * 5000)
    note = client.get("/__proxy/api/scratchboard").json()["items"][0]
    assert note["files"][0]["name"] == "big.bin"
    assert note["files"][0]["size"] == 5000
    assert "data" not in note["files"][0]


def test_an_oversized_file_is_refused_with_both_numbers(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    r = _attach(client, nid, "huge.bin", b"x" * (proxy._SCRATCH_MAX_FILE_BYTES + 1))
    assert r.status_code == 413
    assert "limit" in r.json()["error"]


def test_attaching_to_a_missing_note_is_refused(client):
    _clear()
    assert _attach(client, "nope", "a.txt", b"hi").status_code == 404


def test_garbage_base64_is_refused_rather_than_stored(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    r = client.post(f"/__proxy/api/scratchboard/{nid}/files",
                    json={"name": "a.txt", "data_b64": "!!!not base64!!!"})
    assert r.status_code == 400


def test_files_are_served_as_downloads_never_rendered(client):
    """Uploads are served from the dashboard's own origin — an HTML file rendered inline
    would be script execution against this page."""
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    fid = _attach(client, nid, "evil.html", b"<script>alert(1)</script>", "text/html").json()["id"]
    got = client.get(f"/__proxy/api/scratchboard/files/{fid}")
    assert got.headers["content-type"].startswith("application/octet-stream")
    assert got.headers["content-disposition"].startswith("attachment")
    assert got.headers["x-content-type-options"] == "nosniff"


def test_a_hostile_filename_cannot_escape_the_header(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    fid = _attach(client, nid, '../../etc/pa"sswd\r\nX-Evil: 1', b"x").json()["id"]
    cd = client.get(f"/__proxy/api/scratchboard/files/{fid}").headers["content-disposition"]
    assert '"' not in cd.split("filename=")[1][1:-1]
    assert "\r" not in cd and "\n" not in cd and "/" not in cd


def test_deleting_a_note_takes_its_files(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    fid = _attach(client, nid, "a.txt", b"hi").json()["id"]
    client.delete(f"/__proxy/api/scratchboard/{nid}")
    assert client.get(f"/__proxy/api/scratchboard/files/{fid}").status_code == 404


def test_the_per_note_file_cap_is_enforced(client):
    _clear()
    nid = client.post("/__proxy/api/scratchboard", json={"text": "n"}).json()["id"]
    for i in range(proxy._SCRATCH_MAX_FILES_PER_NOTE):
        assert _attach(client, nid, f"f{i}.txt", b"x").status_code == 200
    assert _attach(client, nid, "one-too-many.txt", b"x").status_code == 409
