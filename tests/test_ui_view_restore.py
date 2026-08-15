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
