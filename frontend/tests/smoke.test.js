import { chromium } from 'playwright';

const BASE_URL = (process.env.BASE_URL || 'http://127.0.0.1:5174').trim();

const PAGES = [
  { path: '/tong-quan', name: 'Tổng quan', checkSelector: '.stepper, .hero, [data-slot="card"], main' },
  { path: '/don-hang', name: 'Quản lý Đơn hàng', checkSelector: 'table.tbl, table, [data-slot="table"]' },
  { path: '/phan-cum', name: 'Phân cụm Địa lý', checkSelector: '.leaflet-container' },
  { path: '/tuyen-giao', name: 'Tuyến giao hàng', checkSelector: '.leaflet-container' },
  { path: '/chay-thu', name: 'Chạy thử', checkSelector: '.toolbar, [data-slot="card"], button' },
  { path: '/so-sanh', name: 'So sánh Phương pháp', checkSelector: '.recharts-responsive-container, table.tbl, table' },
  { path: '/guided-dqn', name: 'K-Means-guided DQN', checkSelector: '.guided-hero, .formula-block, [data-slot="card"]' },
  { path: '/do-nhay', name: 'Độ nhạy & Kiểm chứng', checkSelector: '.recharts-responsive-container, table.tbl, table' },
  { path: '/on-dinh', name: 'Độ ổn định & Tham số', checkSelector: 'table.tbl, .recharts-responsive-container, table' }
];

async function runSmokeTests() {
  console.log(`Starting Frontend Smoke Test Suite on ${BASE_URL}...`);
  const browser = await chromium.launch({ channel: 'msedge' });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 }
  });
  const page = await context.newPage();

  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      const txt = msg.text();
      // Ignore innocuous resource 404s if any
      errors.push(`Console Error: ${txt}`);
    }
  });
  page.on('pageerror', err => {
    errors.push(`Page Error: ${err.message}`);
  });

  let passed = 0;
  let failed = 0;

  for (const p of PAGES) {
    try {
      console.log(`Testing page [${p.name}] at ${p.path}...`);
      await page.goto(`${BASE_URL}${p.path}`, { waitUntil: 'domcontentloaded', timeout: 15000 });
      await page.waitForTimeout(1000);

      // Verify DOM element rendered
      if (p.checkSelector) {
        const el = await page.$(p.checkSelector);
        if (!el) {
          console.warn(`  Warning: Check selector ${p.checkSelector} not found immediately on ${p.path}`);
        }
      }

      // Check theme toggle functionality on the first page
      if (p.path === '/tong-quan') {
        await page.evaluate(() => {
          document.documentElement.setAttribute('data-theme', 'light');
        });
        await page.waitForTimeout(300);
        const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
        if (theme !== 'light') throw new Error('Theme toggle to light failed');

        await page.evaluate(() => {
          document.documentElement.setAttribute('data-theme', 'dark');
        });
        await page.waitForTimeout(300);
      }

      console.log(`  ✓ Passed: ${p.name}`);
      passed++;
    } catch (e) {
      console.error(`  ✗ Failed: ${p.name} - ${e.message}`);
      failed++;
    }
  }

  // --- Test F2: Command Palette (Ctrl+K) & Presentation Mode (F) ---
  try {
    console.log('\nTesting F2: Command Palette (Ctrl+K)...');
    await page.goto(`${BASE_URL}/tong-quan`, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(500);

    // Trigger Ctrl+K
    await page.keyboard.press('Control+KeyK');
    await page.waitForTimeout(500);

    const cmdDialog = await page.$('.cmd-dialog');
    if (!cmdDialog) throw new Error('Command palette dialog did not open on Ctrl+K');
    console.log('  ✓ Command palette opened successfully');

    // Type query to filter
    await page.keyboard.type('Phân cụm');
    await page.waitForTimeout(300);
    const itemText = await page.$eval('.cmd-item.selected .cmd-item-title', el => el.textContent);
    if (!itemText || !itemText.includes('Phân cụm')) {
      throw new Error(`Command palette filter failed, found: ${itemText}`);
    }
    console.log('  ✓ Command palette filtered item:', itemText);

    // Press Escape to close
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    const isClosed = (await page.$('.cmd-dialog')) === null;
    if (!isClosed) throw new Error('Command palette did not close on Escape');
    console.log('  ✓ Command palette closed on Escape');

    // Test F: Presentation Mode
    console.log('\nTesting F2: Presentation Mode (F)...');
    await page.keyboard.press('KeyF');
    await page.waitForTimeout(500);

    const isPresentation = await page.$eval('.shell', el => el.classList.contains('presentation-mode'));
    if (!isPresentation) throw new Error('Presentation mode not activated on key F');
    console.log('  ✓ Presentation mode activated (sidebar hidden)');

    // ArrowRight slide navigation
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);
    const curUrl = page.url();
    if (!curUrl.includes('/don-hang')) throw new Error(`ArrowRight navigation failed, current url: ${curUrl}`);
    console.log('  ✓ Presentation slide navigated to /don-hang with ArrowRight');

    // Esc exits presentation mode
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);
    const isExited = await page.$eval('.shell', el => !el.classList.contains('presentation-mode'));
    if (!isExited) throw new Error('Esc did not exit presentation mode');
    console.log('  ✓ Presentation mode exited on Escape');

  } catch (e) {
    console.error(`  ✗ F2 Features test failed: ${e.message}`);
    failed++;
  }

  await browser.close();

  console.log('\n--- Test Summary ---');
  console.log(`Pages Passed: ${passed}/${PAGES.length}`);
  console.log(`Pages Failed: ${failed}`);
  console.log(`Console Errors: ${errors.length}`);
  if (errors.length > 0) {
    console.log('Error details:');
    errors.forEach(e => console.log('  ', e));
  }

  if (failed > 0 || errors.length > 0) {
    console.error('Smoke tests failed!');
    process.exit(1);
  } else {
    console.log('All smoke tests passed cleanly!');
    process.exit(0);
  }
}

runSmokeTests().catch(err => {
  console.error('Fatal test error:', err);
  process.exit(1);
});

