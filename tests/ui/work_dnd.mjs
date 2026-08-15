// Drag-and-drop on the Work scratchboard, exercised the way a person does it: synthesize a
// DataTransfer carrying real files and dispatch dragover/drop at the element. There is no way
// to assert this from Python — the whole feature lives in the browser's drag events.
//
//   UI_URL="http://127.0.0.1:8000/__proxy/" node tests/ui/work_dnd.mjs
//
// It posts to whatever board UI_URL points at and deletes what it posted on the way out. Run it
// against a scratch instance, not production.
import { chromium } from 'playwright';

const URL = process.env.UI_URL || 'http://127.0.0.1:8000/__proxy/';
const fails = [];
const check = (ok, what) => { if (!ok) fails.push(what); console.log(`${ok ? 'ok  ' : 'FAIL'}  ${what}`); };

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('pageerror', (e) => fails.push('pageerror: ' + e.message));
// Every alert() in the scratchboard path is a failure the user would have to read, so surface
// it here rather than letting Playwright auto-dismiss it into silence.
const alerts = [];
page.on('dialog', async (d) => { alerts.push(d.message()); await d.dismiss(); });

await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
await page.evaluate(() => setView('work'));
await page.waitForSelector('.sb-compose', { timeout: 15000 });

// Fires a real drop carrying `files` ([{name, type, body}]) at `selector`.
const drop = (selector, files) => page.evaluate(({ selector, files }) => {
  const dt = new DataTransfer();
  for (const f of files) dt.items.add(new File([f.body], f.name, { type: f.type }));
  const el = document.querySelector(selector);
  for (const type of ['dragover', 'drop']) {
    el.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
  }
}, { selector, files });

const marker = 'dnd-smoke-' + Math.random().toString(36).slice(2, 8);

// --- the compose box queues, and shows what it queued ------------------------------------
await drop('.sb-compose', [{ name: 'a.txt', type: 'text/plain', body: 'alpha' }]);
await page.waitForTimeout(150);
check((await page.locator('#sb-pending .sb-file.pend').count()) === 1, 'one drop queues one chip');

// A second drop must add to the queue, not replace it — the bug a read-only FileList would give.
await drop('.sb-compose', [{ name: 'b.txt', type: 'text/plain', body: 'beta' }]);
await page.waitForTimeout(150);
check((await page.locator('#sb-pending .sb-file.pend').count()) === 2, 'a second drop adds rather than replaces');
check((await page.locator('#sb-attach-btn').textContent()).includes('2'), 'the attach button counts the queue');

// Removing one leaves the other.
await page.locator('#sb-pending .sb-file.pend a').first().click();
await page.waitForTimeout(100);
check((await page.locator('#sb-pending .sb-file.pend').count()) === 1, 'a queued file can be removed');
check((await page.locator('#sb-pending').textContent()).includes('b.txt'), 'the right one was removed');

// --- posting carries the queue up ---------------------------------------------------------
await page.fill('#sb-text', marker + ' posted with a dropped file');
await page.click('.sb-compose button:text("Post")');
await page.waitForTimeout(1500);

const note = page.locator('.sb-note', { hasText: marker }).first();
check(await note.count() > 0, 'the note posted');
check((await note.locator('.sb-file').allTextContents()).some((t) => t.includes('b.txt')),
      'the dropped file went up with the note');
check((await page.locator('#sb-pending .sb-file.pend').count()) === 0, 'the queue clears after posting');

// --- dropping on an existing note attaches to that note -----------------------------------
const noteId = await note.evaluate((el) => {
  el.id = 'dnd-target';   // the note carries no id attribute; give it one to aim the drop
  return el.id;
});
await drop('#' + noteId, [{ name: 'c.txt', type: 'text/plain', body: 'gamma' }]);
await page.waitForTimeout(1800);

const after = page.locator('.sb-note', { hasText: marker }).first();
const names = await after.locator('.sb-file').allTextContents();
check(names.some((t) => t.includes('c.txt')), 'a file dropped on a note attaches to it');
check(names.some((t) => t.includes('b.txt')), 'and does not displace what was already there');

// --- the page must not navigate away on a stray drop --------------------------------------
const before = page.url();
await drop('body', [{ name: 'stray.txt', type: 'text/plain', body: 'x' }]);
await page.waitForTimeout(400);
check(page.url() === before, 'a drop on the page background does not navigate away');

// --- cleanup -------------------------------------------------------------------------------
await page.evaluate(async (m) => {
  const r = await fetch('/__proxy/api/scratchboard');
  for (const n of (await r.json()).notes || []) {
    if ((n.text || '').includes(m)) {
      await fetch('/__proxy/api/scratchboard/' + encodeURIComponent(n.id), { method: 'DELETE' });
    }
  }
}, marker);

check(alerts.length === 0, 'no error dialogs' + (alerts.length ? ': ' + alerts.join(' | ') : ''));

await browser.close();
if (fails.length) { console.error('\n' + fails.length + ' failed'); process.exit(1); }
console.log('\nall passed');
