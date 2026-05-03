"""Automated screenshot capture for the AI Proxy UI.

Drives a headless Chromium via Playwright through every tab and stores PNGs to disk.
Use it to refresh the README screenshots before publishing or after UI changes.

Setup (one time):
    pip install playwright
    playwright install chromium

Run:
    python scripts/screenshots.py
    python scripts/screenshots.py --url http://192.168.6.183:11444/__proxy/ --out docs/screenshots
    python scripts/screenshots.py --width 1800 --height 1100

The script:
  - Visits each tab (Requests, Conversations, Stats, System, Setup, Audit) and saves a full-page PNG.
  - Picks the most recent request and screenshots its detail panel.
  - Picks the most recent conversation and screenshots its turn-by-turn view.
  - If any request has shadow runs, opens the compare view and screenshots it.

It expects the proxy to have at least a few real requests in its DB; for a clean demo, run a
short Claude Code session through the proxy first so the views aren't empty.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.stderr.write(
        "playwright not installed.\n"
        "  pip install playwright\n"
        "  playwright install chromium\n"
    )
    sys.exit(1)


TABS = [
    ("requests", "Requests", "tab-req"),
    ("convs", "Conversations", "tab-convs"),
    ("system", "System", "tab-system"),
    ("setup", "Setup", "tab-setup"),
    ("stats", "Stats (with traffic flow)", "tab-stats"),
    ("audit", "Audit (rules + suggestions)", "tab-audit"),
]


async def click_tab(page, tab_button_id: str, view_id: str) -> None:
    await page.click(f"#{tab_button_id}")
    # The view is shown by toggling .show on its container; wait for that to flip.
    try:
        await page.wait_for_selector(f"#{view_id}.show", timeout=4000)
    except PWTimeout:
        pass
    await page.wait_for_timeout(700)  # let auto-refresh paint at least once


async def capture(page, out_dir: Path, name: str) -> None:
    out = out_dir / f"{name}.png"
    await page.screenshot(path=str(out), full_page=True)
    print(f"  ✓ {out}")


async def first_request_id(page) -> str | None:
    return await page.evaluate(
        """async () => {
            const r = await fetch('/__proxy/api/requests?limit=20');
            if (!r.ok) return null;
            const d = await r.json();
            const items = (d.items || []).filter(it => it.status);
            return items.length ? items[0].id : null;
        }"""
    )


async def first_conversation_id(page) -> str | None:
    return await page.evaluate(
        """async () => {
            const r = await fetch('/__proxy/api/conversations?limit=10');
            if (!r.ok) return null;
            const d = await r.json();
            return (d.items && d.items.length) ? d.items[0].conversation_id : null;
        }"""
    )


async def first_request_with_shadow(page) -> tuple[str | None, str | None]:
    pair = await page.evaluate(
        """async () => {
            const r = await fetch('/__proxy/api/requests?limit=50');
            if (!r.ok) return null;
            const d = await r.json();
            for (const it of (d.items || [])) {
                const detail = await fetch('/__proxy/api/requests/' + it.id);
                if (!detail.ok) continue;
                const dd = await detail.json();
                if (dd.shadows && dd.shadows.length) {
                    const s = dd.shadows.find(s => s.response_body || s.stream_chunks);
                    if (s) return { primary: it.id, shadow: s.id };
                }
            }
            return null;
        }"""
    )
    if pair:
        return pair.get("primary"), pair.get("shadow")
    return None, None


async def run(url: str, out: Path, width: int, height: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": width, "height": height},
            color_scheme="dark",
            device_scale_factor=2,  # crisp PNGs
        )
        page = await ctx.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector("#tab-req", timeout=8000)
        await page.wait_for_timeout(500)

        # 1. Each tab.
        for slug, label, btn in TABS:
            print(f"[{label}]")
            await click_tab(page, btn, slug)
            await capture(page, out, f"tab-{slug}")

        # 2. Request detail (latest request with a status).
        print("[Request detail]")
        await click_tab(page, "tab-req", "requests")
        rid = await first_request_id(page)
        if rid:
            await page.evaluate(f"selectItem('{rid}')")
            await page.wait_for_timeout(800)
            await capture(page, out, "request-detail")
        else:
            print("  - no completed requests available")

        # 3. Conversation detail.
        print("[Conversation detail]")
        await click_tab(page, "tab-convs", "convs")
        cid = await first_conversation_id(page)
        if cid:
            await page.evaluate(f"loadConversationDetail('{cid}')")
            await page.wait_for_timeout(800)
            await capture(page, out, "conversation-detail")
        else:
            print("  - no conversations available")

        # 4. Shadow comparison page.
        print("[Compare view]")
        await click_tab(page, "tab-req", "requests")
        primary_id, shadow_id = await first_request_with_shadow(page)
        if primary_id and shadow_id:
            await page.evaluate(f"openCompare('{primary_id}', '{shadow_id}')")
            await page.wait_for_timeout(900)
            await capture(page, out, "compare-view")
        else:
            print("  - no request has a completed shadow run yet")

        await browser.close()
    print(f"\nDone. Screenshots in: {out.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.168.6.183:11444/__proxy/",
                   help="Base URL of the proxy UI (must end with /__proxy/)")
    p.add_argument("--out", default="docs/screenshots",
                   help="Output directory for PNGs")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(run(args.url, Path(args.out), args.width, args.height))


if __name__ == "__main__":
    main()
