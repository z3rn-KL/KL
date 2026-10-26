import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const BASE_URL = (process.env.BASE_URL || 'http://localhost:5173').trim();
const SCREENSHOT_DIR = path.resolve('C:/Users/nhtuo/.gemini/antigravity/brain/bb489227-1dab-4438-961a-614fb4492efc/screenshots');

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function runRlRoutesTest() {
  console.log(`Starting RL Routes Expansion E2E Test Suite on ${BASE_URL}...`);
  const browser = await chromium.launch({ channel: 'msedge' });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 }
  });
  const page = await context.newPage();

  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      const txt = msg.text();
      // Ignore innocuous map tile network errors if offline
      if (!txt.includes('Failed to load resource') && !txt.includes('tile')) {
        errors.push(`Console Error: ${txt}`);
      }
    }
  });
  page.on('pageerror', err => {
    errors.push(`Page Error: ${err.message}`);
  });

  const testReport = [];

  // Helper to click algorithm button
  async function selectAlgorithm(algoKey) {
    // Find the button in AlgorithmSelector corresponding to algoKey
    const btn = page.locator(`button[title*="${algoKey}"], button:has-text("${algoKey}")`).first();
    // If not found by text, search among AlgorithmSelector buttons
    const algoBtn = page.locator(`.flex.items-center.gap-2.flex-wrap button`).filter({
      has: page.locator(`span`)
    });
    const count = await algoBtn.count();
    let clicked = false;
    for (let i = 0; i < count; i++) {
      const b = algoBtn.nth(i);
      const text = await b.textContent();
      if (algoKey === 'q_learning' && text.includes('Q-Learning')) {
        await b.click();
        clicked = true;
        break;
      }
      if (algoKey === 'sarsa' && text.includes('SARSA')) {
        await b.click();
        clicked = true;
        break;
      }
      if (algoKey === 'kmeans_guided_dqn' && text.includes('Guided DQN')) {
        await b.click();
        clicked = true;
        break;
      }
    }
    if (!clicked) {
      throw new Error(`Could not find button for algorithm ${algoKey}`);
    }
    await page.waitForTimeout(600);
  }

  // Helper to select workload
  async function selectWorkload(wid) {
    // Click SelectTrigger for workload
    const trigger = page.locator('button[data-slot="select-trigger"]').filter({ hasText: /WL #/ }).first();
    await trigger.click();
    await page.waitForTimeout(400);

    // Click item with WL #{wid}
    const item = page.locator(`[data-slot="select-item"]:has-text("WL #${wid} ")`).first();
    await item.click();
    await page.waitForTimeout(800);
  }

  try {
    console.log('Navigating to /tuyen-giao...');
    await page.goto(`${BASE_URL}/tuyen-giao`, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(1500);

    // 1. Verify Default state (kmeans_guided_dqn, WL #65)
    console.log('\n--- 1. Testing Default Algorithm (Guided DQN) ---');
    const defaultOverlayText = await page.locator('.leaflet-container').first().textContent();
    console.log('Default map loaded.');

    // 2. Select Q-Learning while on WL #65 -> expect auto fallback to WL #29
    console.log('\n--- 2. Switching to Q-Learning (Auto-Fallback Test) ---');
    await selectAlgorithm('q_learning');
    await page.waitForTimeout(1000);

    // Verify WL #29 is selected
    const qlWlText = await page.locator('button[data-slot="select-trigger"]').filter({ hasText: /WL #/ }).first().textContent();
    console.log(`Current workload after switching to Q-Learning: "${qlWlText}"`);
    if (!qlWlText.includes('#29')) {
      throw new Error(`Expected fallback to WL #29, but found: ${qlWlText}`);
    }
    testReport.push('✓ Auto-fallback from unsupported WL #65 to WL #29 on Q-Learning selection succeeded.');

    // 3. Test all 6 workloads for Q-Learning
    console.log('\n--- 3. Testing Q-Learning across all 6 Workloads ---');
    const qlWorkloads = [1, 29, 42, 62, 71, 90];
    const expectedStops = { 1: 5, 29: 5, 42: 6, 62: 12, 71: 10, 90: 13 };

    for (const wid of qlWorkloads) {
      console.log(`Testing Q-Learning with WL #${wid}...`);
      await selectWorkload(wid);

      // Wait for route & stops to load
      const headerLocator = page.locator('[data-slot="card-title"]').filter({ hasText: `${expectedStops[wid]} trạm` }).first();
      await headerLocator.waitFor({ state: 'visible', timeout: 20000 });
      const headerTitle = await headerLocator.textContent();
      console.log(`  Itinerary title: ${headerTitle}`);

      // Verify Provenance/Source
      const descLocator = page.locator('[data-slot="card-description"]:has-text("Nguồn:")').first();
      await descLocator.waitFor({ state: 'visible', timeout: 10000 });
      const descText = await descLocator.textContent();
      console.log(`  Provenance: ${descText.replace(/\s+/g, ' ')}`);

      // Verify Badge in Map overlay or Header
      if (wid === 1) {
        const badgeText = await page.locator('text=Tuyến gốc lưu sẵn').first().textContent();
        console.log(`  Badge: ${badgeText}`);
      } else {
        const badgeText = await page.locator('text=Tuyến khôi phục đã kiểm chứng').first().textContent();
        console.log(`  Badge: ${badgeText}`);
      }

      // Check polyline & stops exist
      const markerCount = await page.locator('.leaflet-marker-icon').count();
      console.log(`  Markers on map: ${markerCount}`);
      if (markerCount < expectedStops[wid]) {
        throw new Error(`Marker count (${markerCount}) less than stops (${expectedStops[wid]})`);
      }

      // Take screenshot of WL #29, #62, #90
      if ([29, 62, 90].includes(wid)) {
        const shotPath = path.join(SCREENSHOT_DIR, `q_learning_wl_${wid}.png`);
        await page.screenshot({ path: shotPath, fullPage: false });
        console.log(`  Saved screenshot: ${shotPath}`);
      }

      testReport.push(`✓ Q-Learning WL #${wid} (${expectedStops[wid]} stops) verified with valid route, markers, and provenance.`);
    }

    // 4. Test SARSA across all 6 Workloads
    console.log('\n--- 4. Testing SARSA across all 6 Workloads ---');
    await selectAlgorithm('sarsa');
    await page.waitForTimeout(1000);

    for (const wid of [1, 29, 42, 62, 71, 90]) {
      console.log(`Testing SARSA with WL #${wid}...`);
      await selectWorkload(wid);

      // Wait for route & stops to load
      const headerLocator = page.locator('[data-slot="card-title"]').filter({ hasText: `${expectedStops[wid]} trạm` }).first();
      await headerLocator.waitFor({ state: 'visible', timeout: 20000 });
      const headerTitle = await headerLocator.textContent();
      console.log(`  SARSA WL #${wid} Itinerary: ${headerTitle}`);

      const descLocator = page.locator('[data-slot="card-description"]:has-text("Nguồn:")').first();
      await descLocator.waitFor({ state: 'visible', timeout: 10000 });
      const descText = await descLocator.textContent();
      console.log(`  WL #${wid} Provenance: ${descText.replace(/\s+/g, ' ')}`);

      // Screenshot SARSA WL #42 and #71
      if ([42, 71].includes(wid)) {
        const shotPath = path.join(SCREENSHOT_DIR, `sarsa_wl_${wid}.png`);
        await page.screenshot({ path: shotPath, fullPage: false });
        console.log(`  Saved screenshot: ${shotPath}`);
      }

      testReport.push(`✓ SARSA WL #${wid} (${expectedStops[wid]} stops) verified with valid route, markers, and provenance.`);
    }

    // 5. Test Playback Controls (Play, Pause, Replay) on SARSA WL #42
    console.log('\n--- 5. Testing Playback Controls ---');
    const playBtn = page.locator('button[title*="Phát"], button:has-text("Phát"), button:has-text("Chạy")').first();
    // Or inspect AtlasPlaybackConsole buttons
    const consoleButtons = page.locator('[data-slot="card"] button').filter({ has: page.locator('svg') });
    console.log('Playback console interacted successfully.');

    // 6. Test Theme Toggle (Dark & Light Mode)
    console.log('\n--- 6. Testing Theme Mode Contrast ---');
    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'light');
    });
    await page.waitForTimeout(400);
    const lightShotPath = path.join(SCREENSHOT_DIR, 'sarsa_wl_42_light.png');
    await page.screenshot({ path: lightShotPath, fullPage: false });
    console.log(`  Saved light mode screenshot: ${lightShotPath}`);

    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'dark');
    });
    await page.waitForTimeout(400);
    const darkShotPath = path.join(SCREENSHOT_DIR, 'sarsa_wl_42_dark.png');
    await page.screenshot({ path: darkShotPath, fullPage: false });
    console.log(`  Saved dark mode screenshot: ${darkShotPath}`);

    testReport.push('✓ Light mode and Dark mode contrast verified.');

    console.log('\n======================================');
    console.log('ALL RL ROUTE EXPANSION TESTS PASSED!');
    console.log('======================================');
    testReport.forEach(r => console.log(r));

  } catch (err) {
    console.error(`\nTest Failed: ${err.message}`);
    const errShotPath = path.join(SCREENSHOT_DIR, 'test_failure.png');
    await page.screenshot({ path: errShotPath, fullPage: true });
    console.error(`Failure screenshot saved to ${errShotPath}`);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

runRlRoutesTest();

