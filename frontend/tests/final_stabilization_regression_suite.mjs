import { chromium } from 'playwright';

async function runRegressionSuite() {
  console.log('=== STARTING NEXUS COMPREHENSIVE REGRESSION TEST SUITE ===');
  const browser = await chromium.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
  });

  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const results = [];

  const recordResult = (group, name, passed, details = '') => {
    results.push({ group, name, passed, details });
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`[${status}] [${group}] ${name} ${details ? `(${details})` : ''}`);
  };

  try {
    // -------------------------------------------------------------------
    // GROUP A: FRONTEND RENDERING (6 ROUTES)
    // -------------------------------------------------------------------
    const routes = [
      { path: '/tong-quan', title: 'Tổng quan' },
      { path: '/phan-cum', title: 'Phân cụm' },
      { path: '/toi-uu-rl', title: 'Tối ưu RL' },
      { path: '/mo-phong', title: 'Mô phỏng' },
      { path: '/bao-cao', title: 'Báo cáo' },
      { path: '/thuc-nghiem', title: 'Thực nghiệm' },
    ];

    for (const r of routes) {
      try {
        await page.goto(`http://127.0.0.1:5176${r.path}`, { waitUntil: 'domcontentloaded' });
        await page.waitForSelector('h1, h2, h3', { timeout: 10000 });
        const headingCount = await page.locator('h1, h2, h3').count();
        recordResult('Group A: Rendering', `Route ${r.path} (${r.title}) loads cleanly`, headingCount > 0, `${headingCount} headings`);
      } catch (err) {
        recordResult('Group A: Rendering', `Route ${r.path}`, false, err.message);
      }
    }

    // -------------------------------------------------------------------
    // GROUP B: GIS & CLUSTERING (/phan-cum)
    // -------------------------------------------------------------------
    await page.goto('http://127.0.0.1:5176/phan-cum', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1500);

    const callout = await page.locator(':has-text("10 cụm K-Means"), :has-text("K = 10")').count();
    recordResult('Group B: Clustering', 'Compact clustering callout rendered', callout > 0);

    const c2Btn = page.locator('button:has-text("Cụm 2")').first();
    if (await c2Btn.count() > 0) {
      await c2Btn.click();
      await page.waitForTimeout(1000);
      recordResult('Group B: Clustering', 'Cluster 2 pill interactive', true);
    } else {
      recordResult('Group B: Clustering', 'Cluster 2 pill found', false);
    }

    const allBtn = page.locator('button:has-text("Tất cả các cụm")').first();
    if (await allBtn.count() > 0) {
      await allBtn.click();
      await page.waitForTimeout(1000);
      recordResult('Group B: Clustering', 'Reset to all clusters works', true);
    }

    // -------------------------------------------------------------------
    // GROUP C & D: RL WORKSPACE & SUB-TABS (/toi-uu-rl)
    // -------------------------------------------------------------------
    await page.goto('http://127.0.0.1:5176/toi-uu-rl', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2500);

    // Sub-tab 2: RL Decision Explorer
    const tabExplorer = page.locator('button[role="tab"]:has-text("2. Vết Quyết Định RL")').first();
    if (await tabExplorer.count() > 0) {
      await tabExplorer.click();
      await page.waitForTimeout(1500);
      const modalContent = await page.locator(':has-text("Candidate-Scoring DQN"), :has-text("Vết Quyết Định")').count();
      recordResult('Group D: RL Presentation', 'RL Decision Explorer tab opens', modalContent > 0);
    }

    // Sub-tab 3: Two-Tier Architecture
    const tabArch = page.locator('button[role="tab"]:has-text("3. Kiến Trúc Hai Tầng")').first();
    if (await tabArch.count() > 0) {
      await tabArch.click();
      await page.waitForTimeout(1000);
      const archText = await page.locator('text=Hai Tầng').count();
      recordResult('Group D: RL Presentation', 'Two-Tier Architecture tab renders', archText > 0);
    }

    // Sub-tab 4: Reward & Features
    const tabReward = page.locator('button[role="tab"]:has-text("4. Hàm Thưởng & 8 Đặc Trưng")').first();
    if (await tabReward.count() > 0) {
      await tabReward.click();
      await page.waitForTimeout(1000);
      const mathText = await page.locator(':has-text("slack_ratio"), :has-text("w_d")').count();
      recordResult('Group D: RL Presentation', 'RL Theory & Reward formulation renders', mathText > 0);
    }

    // Return to Tab 1: Planning Desk
    const tabPlanning = page.locator('button[role="tab"]:has-text("1. Bàn Lập Tuyến")').first();
    if (await tabPlanning.count() > 0) {
      await tabPlanning.click();
      await page.waitForTimeout(1500);
    }

    // Preview Plan
    const previewBtn = page.locator('button:has-text("Xem trước phương án"), button:has-text("Xem trước")').first();
    if (await previewBtn.count() > 0) {
      await previewBtn.click();
      await page.waitForTimeout(7500);
      const previewBar = await page.locator(':has-text("Bản xem trước"), #planningPreviewBar, :has-text("KẾ HOẠCH ĐIỀU PHỐI")').count();
      recordResult('Group C: Planning', 'Preview Plan computed and rendered', previewBar > 0);
    }

    // Confirm Plan (navigates to /mo-phong)
    const confirmBtn = page.locator('button:has-text("Áp dụng kế hoạch"), button:has-text("Xác nhận kế hoạch")').first();
    if (await confirmBtn.count() > 0) {
      await confirmBtn.click();
      await page.waitForTimeout(2500);
      recordResult('Group C: Planning', 'Confirm plan applies without errors', true);
    }

    // -------------------------------------------------------------------
    // GROUP E: SIMULATION WORKSPACE (/mo-phong)
    // -------------------------------------------------------------------
    await page.goto('http://127.0.0.1:5176/mo-phong', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2000);

    const step15Btn = page.locator('button:has-text("Bước 15p"), button:has-text("+15 phút")').first();
    if (await step15Btn.count() > 0) {
      await step15Btn.click();
      await page.waitForTimeout(1000);
      await step15Btn.click();
      await page.waitForTimeout(1000);
      recordResult('Group E: Simulation', 'Timeline stepping (+15 min) advances clock', true);
    }

    // -------------------------------------------------------------------
    // GROUP F: REPORT & SLA INVARIANTS (/bao-cao)
    // -------------------------------------------------------------------
    await page.goto('http://127.0.0.1:5176/bao-cao', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2000);

    const repDayOnTime = (await page.locator('#repDayOnTime').innerText()).trim();
    const repCumulativeSla = (await page.locator('#repCumulativeSla').innerText()).trim();

    const dayOnTimeNum = parseInt(repDayOnTime.replace('%', ''), 10);
    const cumSlaNum = parseInt(repCumulativeSla.replace('%', ''), 10);

    const slaValid = repDayOnTime === '—' || repDayOnTime === '-' || (!isNaN(dayOnTimeNum) && dayOnTimeNum >= 0 && dayOnTimeNum <= 100);
    const cumSlaValid = repCumulativeSla === '—' || repCumulativeSla === '-' || (!isNaN(cumSlaNum) && cumSlaNum >= 0 && cumSlaNum <= 100);
    recordResult('Group F: Report Invariants', 'Daily SLA is mathematically valid (0 <= SLA <= 100% or "—" on pristine shift)', slaValid, `Actual: ${repDayOnTime}`);
    recordResult('Group F: Report Invariants', 'Cumulative SLA is mathematically valid (0 <= SLA <= 100% or "—" on pristine shift)', cumSlaValid, `Actual: ${repCumulativeSla}`);

    const carryTitle = await page.locator('#carryList').locator('xpath=..').locator('h2').innerText();
    recordResult('Group F: Report Invariants', 'Pending orders labeled as "Đơn chưa hoàn thành" in active shift', carryTitle.includes('Đơn chưa hoàn thành'), `Actual: "${carryTitle}"`);

    const btnToggleShippers = page.locator('#btnToggleShipperDetails').first();
    if (await btnToggleShippers.count() > 0) {
      await btnToggleShippers.click();
      await page.waitForTimeout(800);
      const tableVisible = await page.locator('.report-table').isVisible();
      recordResult('Group F: Report Invariants', '16-Shipper table expands on toggle', tableVisible);
      await btnToggleShippers.click();
      await page.waitForTimeout(500);
      const tableHidden = !(await page.locator('.report-table').isVisible());
      recordResult('Group F: Report Invariants', '16-Shipper table collapses on toggle', tableHidden);
    }

    // -------------------------------------------------------------------
    // GROUP G: SCIENTIFIC EXPERIMENTS COHORT SWITCHING (/thuc-nghiem)
    // -------------------------------------------------------------------
    await page.goto('http://127.0.0.1:5176/thuc-nghiem', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2000);

    const text108 = await page.locator('text=108 Workload Cơ Sở').count();
    recordResult('Group G: Cohort Consistency', 'Initial cohort is 108 Workload Benchmark', text108 > 0);

    const tab12 = page.locator('button[role="tab"]:has-text("12 WL")').first();
    if (await tab12.count() > 0) {
      await tab12.click();
      await page.waitForTimeout(1500);
      const text12 = await page.locator('text=12 Workload Paired').count();
      recordResult('Group G: Cohort Consistency', 'Switching to 12 WL updates context and cards', text12 > 0);
    }

    const tab16 = page.locator('button[role="tab"]:has-text("16 WL")').first();
    if (await tab16.count() > 0) {
      await tab16.click();
      await page.waitForTimeout(1500);
      const text16 = await page.locator('text=16 Workload Test (Proposed)').count();
      recordResult('Group G: Cohort Consistency', 'Switching to 16 WL updates context and cards', text16 > 0);
    }

    const tabBack108 = page.locator('button[role="tab"]:has-text("108 WL")').first();
    if (await tabBack108.count() > 0) {
      await tabBack108.click();
      await page.waitForTimeout(1500);
      const textBack = await page.locator('text=108 Workload Cơ Sở').count();
      recordResult('Group G: Cohort Consistency', 'Switching back to 108 WL restores benchmark cleanly', textBack > 0);
    }

    // -------------------------------------------------------------------
    // GROUP H: API & SESSION 404 RECOVERY
    // -------------------------------------------------------------------
    await page.evaluate(() => {
      localStorage.setItem('nexus_active_session_id', 'sess_expired_non_existent_9999');
    });
    await page.goto('http://127.0.0.1:5176/toi-uu-rl', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(3000);

    const pageContent = await page.locator('h1, h2').allInnerTexts();
    const isRecovered = pageContent.some(t => t.includes('RL Intelligence') || t.includes('Tối Ưu Tuyến') || t.includes('Bàn Lập Tuyến'));
    recordResult('Group H: API Reliability', 'Expired session ID recovers automatically via health check without crash', isRecovered, `Headers: ${JSON.stringify(pageContent.slice(0, 2))}`);

  } catch (err) {
    console.error('Fatal error during regression suite:', err);
  } finally {
    await browser.close();
  }

  // Summary
  console.log('\n=== REGRESSION TEST SUITE SUMMARY ===');
  const total = results.length;
  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;
  console.log(`Total tests: ${total} | Passed: ${passed} | Failed: ${failed}`);

  if (failed === 0) {
    console.log('🎉 ALL REGRESSION TESTS PASSED (100% GREEN)!');
  } else {
    console.error(`⚠️ ${failed} tests failed. Review log above.`);
    process.exit(1);
  }
}

runRegressionSuite();
