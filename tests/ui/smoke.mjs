// Headless-browser smoke test: loads the proxy UI, waits for it to populate from the
// API, and asserts it actually rendered and is interactive — not just that the server
// returned 200. Fails (exit 1) on any uncaught JS error, failed asset, or missing
// expected content. Run against a live instance:  UI_URL=http://127.0.0.1:8000/__proxy/ node smoke.mjs
import { chromium } from 'playwright';

const URL = process.env.UI_URL || 'http://127.0.0.1:8000/__proxy/';
const errors = [];
const benign = (s) => /favicon/i.test(s);

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push('console: ' + m.text()); });
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('requestfailed', (r) => { if (!benign(r.url())) errors.push('reqfail: ' + r.url() + ' ' + ((r.failure() || {}).errorText || '')); });

await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
// The requests list is populated by an API call after load; wait for a real row.
await page.waitForSelector('#rows tr', { timeout: 20000 });

const info = await page.evaluate(() => ({
  title: document.title,
  tabCount: document.querySelectorAll('.tab').length,
  rowCount: document.querySelectorAll('#rows tr').length,
  hasClientFilter: !!document.getElementById('rq-client'),
  clientOptions: document.getElementById('rq-client')
    ? [...document.getElementById('rq-client').options].map((o) => o.textContent.trim())
    : [],
  hasPagerJump: !!document.getElementById('rq-jump'),
}));

// Exercise a tab switch to confirm the app's JS runs without throwing.
try {
  await page.evaluate(() => setView('stats'));
  await page.waitForTimeout(600);
} catch (e) {
  errors.push('stats-tab: ' + e.message);
}

await browser.close();

const fail = [];
if (info.title !== 'AI Proxy') fail.push(`title is ${JSON.stringify(info.title)}`);
if (info.tabCount < 5) fail.push(`only ${info.tabCount} tabs rendered`);
if (info.rowCount < 1) fail.push('request table rendered no rows');
if (!info.hasClientFilter) fail.push('client filter dropdown missing');
if (!info.hasPagerJump) fail.push('pager jump input missing');
if (info.clientOptions.length < 2) fail.push(`client dropdown not populated: ${JSON.stringify(info.clientOptions)}`);
fail.push(...errors);

console.log(JSON.stringify(info, null, 2));
if (fail.length) {
  console.error('UI SMOKE FAILED:\n- ' + fail.join('\n- '));
  process.exit(1);
}
console.log('UI SMOKE PASSED');
