"""Every tab must survive a page refresh.

setView() writes the active view to storage unconditionally, but the restore path filters
through VALID_VIEWS — so a tab left out of that set works when clicked and silently falls back
to Requests on the next reload. Work shipped that way. This is a static check against the one
file the dashboard lives in, because the bug is a mismatch between two lists in it.
"""
import re
from pathlib import Path

import ai_proxy

HTML = (Path(ai_proxy.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

# Entered from a conversation rather than the nav rail, so it is not a restorable view.
NOT_RESTORABLE = {"artifacts"}


def _valid_views() -> set:
    m = re.search(r"const VALID_VIEWS = new Set\(\[(.*?)\]\)", HTML, re.S)
    assert m, "VALID_VIEWS is gone or was renamed"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def _setview_views() -> set:
    """The views setView() knows how to show — the loop that toggles .show on each container."""
    m = re.search(r"for \(const c of \[([^\]]*)\]\) \{\s*\$\('main'\)\.classList\.toggle\(c",
                  HTML)
    assert m, "the setView container loop moved; update this test with it"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_every_view_setview_can_show_can_also_be_restored():
    missing = _setview_views() - _valid_views() - NOT_RESTORABLE
    assert not missing, (f"{sorted(missing)} can be opened but not restored — refreshing the "
                         f"page drops you back on Requests")


def test_work_specifically_is_restorable():
    assert "work" in _valid_views()


def test_the_default_view_is_still_listed():
    assert "requests" in _valid_views()


# --- request detail: start and end -----------------------------------------------------------


def test_the_detail_view_shows_a_timing_row():
    """A duration alone does not say WHICH eleven minutes the GPU was busy, which is the
    question being asked whenever two requests overlap."""
    assert 'class="label">Timing</span> ${timingLine(d)}' in HTML


def test_the_end_time_is_derived_rather_than_read_from_a_column():
    """There is no finish timestamp on the row — only ts and duration_ms — so anything that
    reads d.end_ts would render undefined."""
    m = re.search(r"function timingLine\(d\) \{(.*?)\n\}", HTML, re.S)
    assert m, "timingLine is gone or was renamed"
    body = m.group(1)
    assert "d.duration_ms / 1000" in body, "the end must be computed from the duration"
    assert "end_ts" not in body, "requests has no end_ts column"


def test_an_in_flight_request_says_so_instead_of_inventing_an_end():
    m = re.search(r"function timingLine\(d\) \{(.*?)\n\}", HTML, re.S)
    assert "in flight" in m.group(1), \
        "a request with no duration yet has not ended; showing an end time would be a lie"


# --- the assistant's reply, in plain text -----------------------------------------------------


def test_the_reply_has_a_readable_section_like_the_user_message_does():
    """"Latest User Message" existed; its counterpart did not. A non-streaming reply was
    readable only as raw JSON."""
    assert "Assistant Reply (" in HTML
    assert "Latest User Message (" in HTML, "the section this one mirrors is gone"


def test_reasoning_is_captured_from_every_response_shape():
    """A 700k request returned empty content with the whole answer in `reasoning`, which read
    as "the model returned nothing" until the raw JSON was opened by hand."""
    assert HTML.count("out.reasoning +=") >= 4, \
        "OpenAI message, OpenAI delta, Ollama native and Anthropic thinking blocks"
    m = re.search(r"const reasoningOf = \(m\) =>(.*?);\n", HTML, re.S)
    assert m, "the reasoning extractor is gone"
    for field in ("reasoning", "reasoning_content", "thinking"):
        assert field in m.group(1), f"{field} is not read"


def test_an_empty_reply_explains_itself_instead_of_rendering_nothing():
    """"No reply" and "a reply you cannot see" look identical; the difference is usually the
    token cap eating an unterminated tool call."""
    assert "(empty) — " in HTML
    assert "max_tokens before anything was emitted" in HTML


def test_reasoning_alone_keeps_the_analysis_section_alive():
    """It used to bail out when content_chars was 0, which is exactly the case where the
    reasoning field is the only place the answer exists."""
    m = re.search(r"if \(!req && \(!resp\.tool_calls\.length.*?\) return '';", HTML, re.S)
    assert m and "resp.reasoning" in m.group(0), \
        "the early return still discards a reasoning-only response"
