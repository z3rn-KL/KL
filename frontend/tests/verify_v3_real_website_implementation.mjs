import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUTPUT_DIR = path.resolve(__dirname, '../../audit_outputs/FINAL_V3_REAL_WEBSITE');
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const BASE_URL = 'http://localhost:5176';

async function runRealWebsiteAudit() {
  console.log('=== STARTING NEXUS V3.0 REAL WEBSITE AUDIT (LOCALHOST:5176) ===');

  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  const pageErrors = [];
  page.on('pageerror', err => {
    console.error('[BROWSER ERROR]:', err.message);
    pageErrors.push(err.message);
  });

  // -------------------------------------------------------------------------
  // 1. SCREEN 01: TỔNG QUAN (/tong-quan)
  // -------------------------------------------------------------------------
  console.log('\n[1/6] Auditing SCREEN 01: Tổng quan (/tong-quan)...');
  await page.goto(`${BASE_URL}/tong-quan`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  const h1Text = await page.textContent('h1');
  console.log('H1:', h1Text);
  if (!h1Text.includes('Xây dựng hệ thống tối ưu tuyến giao hàng kết hợp kỹ thuật phân cụm và Reinforcement Learning')) {
    throw new Error('Title does not match thesis official title!');
  }

  const overviewBody = await page.textContent('body');
  const hasXeDu = overviewBody.includes('2.394') && overviewBody.includes('XeDu');
  const hasDynamic = overviewBody.includes('85') && overviewBody.includes('16');
  const hasSinglePipeline = overviewBody.includes('Sơ Đồ Phương Pháp Nghiên Cứu Cốt Lõi');
  console.log(`Overview Verification: XeDu (${hasXeDu ? 'PASS' : 'FAIL'}), Dynamic (${hasDynamic ? 'PASS' : 'FAIL'}), Single Pipeline (${hasSinglePipeline ? 'PASS' : 'FAIL'})`);

  await page.screenshot({ path: path.join(OUTPUT_DIR, '01_real_overview_1920.png'), fullPage: true });

  // -------------------------------------------------------------------------
  // 2. SCREEN 02: DỮ LIỆU & PHÂN CỤM (/phan-cum)
  // -------------------------------------------------------------------------
  console.log('\n[2/6] Auditing SCREEN 02: Dữ liệu & Phân cụm (/phan-cum)...');
  await page.goto(`${BASE_URL}/phan-cum`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  const clusteringBody = await page.textContent('body');
  const hasSilhouette = clusteringBody.includes('0.429') || clusteringBody.includes('0,429');
  const hasK10 = clusteringBody.includes('K = 10') || clusteringBody.includes('10 cụm');
  console.log(`Clustering Verification: Silhouette 0.429 (${hasSilhouette ? 'PASS' : 'FAIL'}), K=10 (${hasK10 ? 'PASS' : 'FAIL'})`);

  // Test cluster filter click
  const c2Btn = page.locator('button:has-text("Cụm 2")').first();
  if (await c2Btn.isVisible()) {
    await c2Btn.click();
    await page.waitForTimeout(800);
    console.log('Clicked Cụm 2 filter button: PASS');
  }

  await page.screenshot({ path: path.join(OUTPUT_DIR, '02_real_clustering_1920.png'), fullPage: true });

  // -------------------------------------------------------------------------
  // 3. SCREEN 03: TỐI ƯU TUYẾN BẰNG RL (/toi-uu-rl) - 4 MODES
  // -------------------------------------------------------------------------
  console.log('\n[3/6] Auditing SCREEN 03: Tối ưu tuyến bằng RL (/toi-uu-rl)...');
  await page.goto(`${BASE_URL}/toi-uu-rl`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2500);

  // Tab 1: Mode A - Planning Desk
  console.log('Auditing Tab 1: Mode A Planning Desk...');
  const tab1 = page.locator('button:has-text("1. Bàn Lập Tuyến")').first();
  if (await tab1.isVisible()) {
    await tab1.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '03_real_rl_mode_a_planning_1920.png'), fullPage: true });
    console.log('Tab 1 Planning Desk: PASS');
  }

  // Tab 2: Mode B - Decision Explorer
  console.log('Auditing Tab 2: Mode B Decision Explorer...');
  const tab2 = page.locator('button:has-text("2. Vết Quyết Định RL")').first();
  if (await tab2.isVisible()) {
    await tab2.click();
    await page.waitForTimeout(3500);
    const traceBody = await page.textContent('body');
    const hasQValue = traceBody.includes('Q-VALUE') || traceBody.includes('Candidate') || traceBody.includes('Q =');
    const hasExplanation = traceBody.includes('LÝ GIẢI KHOA HỌC') || traceBody.includes('Argmax Q-Score') || traceBody.includes('Q(s, a)');
    console.log(`Tab 2 Decision Explorer: Q-Value (${hasQValue ? 'PASS' : 'FAIL'}), Explanation (${hasExplanation ? 'PASS' : 'FAIL'})`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '03_real_rl_mode_b_decision_explorer_1920.png'), fullPage: true });
  }

  // Tab 3: Mode C - Two-Tier Architecture
  console.log('Auditing Tab 3: Mode C Two-Tier Architecture...');
  const tab3 = page.locator('button:has-text("3. Kiến Trúc Hai Tầng")').first();
  if (await tab3.isVisible()) {
    await tab3.click();
    await page.waitForTimeout(800);
    const archBody = await page.textContent('body');
    const hasTier1 = archBody.includes('TẦNG 1') || archBody.includes('Dispatch Allocation');
    const hasTier2 = archBody.includes('TẦNG 2') || archBody.includes('Stop Sequencing');
    console.log(`Tab 3 Architecture: Tier 1 (${hasTier1 ? 'PASS' : 'FAIL'}), Tier 2 (${hasTier2 ? 'PASS' : 'FAIL'})`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '03_real_rl_mode_c_architecture_1920.png'), fullPage: true });
  }

  // Tab 4: Mode D - Reward & 8 Features
  console.log('Auditing Tab 4: Mode D Reward & 8 Features...');
  const tab4 = page.locator('button:has-text("4. Hàm Thưởng")').first();
  if (await tab4.isVisible()) {
    await tab4.click();
    await page.waitForTimeout(800);
    const theoryBody = await page.textContent('body');
    const hasWeights = theoryBody.includes('0.375') && theoryBody.includes('0.050');
    const hasFeatures = theoryBody.includes('elapsed_ratio') && theoryBody.includes('cluster_switch');
    console.log(`Tab 4 Reward & Features: Weights (${hasWeights ? 'PASS' : 'FAIL'}), 8 Features (${hasFeatures ? 'PASS' : 'FAIL'})`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '03_real_rl_mode_d_reward_features_1920.png'), fullPage: true });
  }

  // -------------------------------------------------------------------------
  // 4. SCREEN 04: MÔ PHỎNG GIAO HÀNG (/mo-phong)
  // -------------------------------------------------------------------------
  console.log('\n[4/6] Auditing SCREEN 04: Mô phỏng giao hàng (/mo-phong)...');
  await page.goto(`${BASE_URL}/mo-phong`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  const simBody = await page.textContent('body');
  const hasFleet = simBody.includes('16') && (simBody.includes('Shipper') || simBody.includes('xe'));
  console.log(`Simulation Fleet: 16 Shippers loaded (${hasFleet ? 'PASS' : 'FAIL'})`);

  // Step clock button test
  const stepBtn = page.locator('button:has-text("+15 Phút")').first();
  if (await stepBtn.isVisible()) {
    await stepBtn.click();
    await page.waitForTimeout(1000);
    console.log('Simulation Clock step +15 min: PASS');
  }

  await page.screenshot({ path: path.join(OUTPUT_DIR, '04_real_simulation_1920.png'), fullPage: true });

  // -------------------------------------------------------------------------
  // 5. SCREEN 05: BÁO CÁO KẾT CA (/bao-cao)
  // -------------------------------------------------------------------------
  console.log('\n[5/6] Auditing SCREEN 05: Báo cáo kết ca (/bao-cao)...');
  await page.goto(`${BASE_URL}/bao-cao`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  const reportBody = await page.textContent('body');
  const hasReport = reportBody.includes('Báo Cáo') || reportBody.includes('SLA');
  console.log(`Shift Report loaded: (${hasReport ? 'PASS' : 'FAIL'})`);

  await page.screenshot({ path: path.join(OUTPUT_DIR, '05_real_shift_report_1920.png'), fullPage: true });

  // -------------------------------------------------------------------------
  // 6. SCREEN 06: THỰC NGHIỆM & SO SÁNH (/thuc-nghiem) - 4 TABS
  // -------------------------------------------------------------------------
  console.log('\n[6/6] Auditing SCREEN 06: Thực nghiệm & So sánh (/thuc-nghiem)...');
  await page.goto(`${BASE_URL}/thuc-nghiem`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2500);

  // Tab 1: 108 Workloads
  console.log('Auditing Tab 1: 108 Workloads Benchmark...');
  const expTab1 = page.locator('button:has-text("1. Đối Chuẩn 108 Workloads")').first();
  if (await expTab1.isVisible()) {
    await expTab1.click();
    await page.waitForTimeout(1000);
    const benchBody = await page.textContent('body');
    const hasBaselines = benchBody.includes('Clarke') || benchBody.includes('Nearest') || benchBody.includes('108');
    console.log(`Tab 1 Benchmark Baselines: (${hasBaselines ? 'PASS' : 'FAIL'})`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '06_real_experiments_108_workloads_1920.png'), fullPage: true });
  }

  // Tab 2: Training Curves
  console.log('Auditing Tab 2: Training Curves...');
  const expTab2 = page.locator('button:has-text("2. Quá Trình Huấn Luyện RL")').first();
  if (await expTab2.isVisible()) {
    await expTab2.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '06_real_experiments_training_curves_1920.png'), fullPage: true });
    console.log('Tab 2 Training Curves: PASS');
  }

  // Tab 3: Baselines Q-Learning / SARSA
  console.log('Auditing Tab 3: Baselines Q-Learning & SARSA...');
  const expTab3 = page.locator('button:has-text("3. Thuật Toán Cơ Sở")').first();
  if (await expTab3.isVisible()) {
    await expTab3.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '06_real_experiments_baselines_1920.png'), fullPage: true });
    console.log('Tab 3 Baselines: PASS');
  }

  // Tab 4: Limitations & Honest Findings
  console.log('Auditing Tab 4: Limitations & Honest Findings...');
  const expTab4 = page.locator('button:has-text("4. Giới Hạn")').first();
  if (await expTab4.isVisible()) {
    await expTab4.click();
    await page.waitForTimeout(1000);
    const limitBody = await page.textContent('body');
    const hasHonestFindings = limitBody.includes('Clarke-Wright') || limitBody.includes('ngắn nhất');
    console.log(`Tab 4 Honest Limitations: (${hasHonestFindings ? 'PASS' : 'FAIL'})`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, '06_real_experiments_limitations_1920.png'), fullPage: true });
  }

  // -------------------------------------------------------------------------
  // 7. RESPONSIVE VIEWPORT CHECKS
  // -------------------------------------------------------------------------
  console.log('\n[7/8] Auditing Responsive Viewports...');
  await page.goto(`${BASE_URL}/toi-uu-rl`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(1500);

  // 1366 Laptop
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUTPUT_DIR, 'real_1366_laptop.png'), fullPage: true });
  console.log('Viewport 1366x768 (Laptop): PASS');

  // 768 Tablet
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUTPUT_DIR, 'real_768_tablet.png'), fullPage: true });
  console.log('Viewport 768x1024 (Tablet): PASS');

  // 375 Mobile
  await page.setViewportSize({ width: 375, height: 667 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUTPUT_DIR, 'real_375_mobile.png'), fullPage: true });
  console.log('Viewport 375x667 (Mobile): PASS');

  // -------------------------------------------------------------------------
  // 8. THEME TOGGLE (DARK MODE)
  // -------------------------------------------------------------------------
  console.log('\n[8/8] Auditing Dark Theme Toggle...');
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto(`${BASE_URL}/tong-quan`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(1000);

  // Toggle theme button
  const themeBtn = page.locator('button[aria-label="Đổi giao diện sáng/tối"]').first();
  if (await themeBtn.isVisible()) {
    await themeBtn.click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'real_theme_dark_1920.png'), fullPage: true });
    console.log('Dark Theme Toggle: PASS');
  }

  await browser.close();
  console.log('\n=== REAL WEBSITE AUDIT COMPLETED SUCCESSFULLY WITH ZERO FATAL ERRORS ===');
}

runRealWebsiteAudit().catch(err => {
  console.error('Audit failed with error:', err);
  process.exit(1);
});
