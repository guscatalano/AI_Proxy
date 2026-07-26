// Rasterizes logo.svg into the PNG assets referenced by AppxManifest.xml (plus master
// icons for the README). Uses the Playwright/Chromium already installed under tests/ui.
//
//   node packaging/windows/msix/make-assets.mjs
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';

const here = dirname(fileURLToPath(import.meta.url));
// Reuse the Playwright/Chromium already installed for the UI smoke test.
const require = createRequire(join(here, '..', '..', '..', 'tests', 'ui', 'package.json'));
const { chromium } = require('playwright');
const svg = readFileSync(join(here, 'logo.svg'), 'utf8');

const targets = [
  ['assets/Square44x44Logo.png', 44],
  ['assets/Square150x150Logo.png', 150],
  ['assets/Square310x310Logo.png', 310],
  ['assets/StoreLogo.png', 50],
  ['assets/icon-256.png', 256],
  ['assets/icon-512.png', 512],
];

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
for (const [rel, size] of targets) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(
    `<style>*{margin:0;padding:0}svg{display:block;width:${size}px;height:${size}px}</style>${svg}`
  );
  const el = await page.$('svg');
  await el.screenshot({ path: join(here, rel) });
  console.log(`wrote ${rel} (${size}x${size})`);
}
await browser.close();
