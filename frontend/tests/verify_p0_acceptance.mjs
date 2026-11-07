import { chromium } from '../node_modules/playwright/index.mjs';

async function verify() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  });

  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  let currentRoute = 'init';
  const errors = [];
  page.on('pageerror', err => errors.push({ route: currentRoute, msg: err.message, stack: err.stack }));
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('404')) {
      errors.push({ route: currentRoute, msg: msg.text(), loc: msg.location() });
    }
  });

  console.log('=== 1. VERIFY ROOT & SCREEN 01 (TỔNG QUAN) ===');
  currentRoute = '/';
  await page.goto('http://127.0.0.1:5176');
  await page.waitForTimeout(2000);
  const s1Hero = await page.locator('h1').innerText();
  const s1Map = await page.locator('.leaflet-container').count();
  console.log(`Screen 01: Hero='${s1Hero}', MapCount=${s1Map}`);

  console.log('=== 2. VERIFY SCREEN 02 (PHÂN CỤM) ===');
  currentRoute = '/phan-cum';
  await page.locator('header nav a:has-text("Phân cụm")').click();
  await page.waitForTimeout(2000);
  const s2Title = await page.locator('h1').innerText();
  const s2TableRows = await page.locator('table tbody tr').count();
  const s2Map = await page.locator('.leaflet-container').count();
  console.log(`Screen 02: Title='${s2Title}', TableRows=${s2TableRows}, MapCount=${s2Map}`);

  console.log('=== 3. VERIFY SCREEN 03 (TỐI ƯU RL) ===');
  currentRoute = '/toi-uu-rl';
  await page.locator('header nav a:has-text("Tối ưu RL")').click();
  await page.waitForTimeout(2500);
  const s3Title = await page.locator('h1').innerText();
  const s3Map = await page.locator('.leaflet-container').count();
  console.log(`Screen 03: Title='${s3Title}', MapCount=${s3Map}`);

  console.log('=== 4. VERIFY SCREEN 04 (MÔ PHỎNG) ===');
  currentRoute = '/mo-phong';
  await page.locator('header nav a:has-text("Mô phỏng")').click();
  await page.waitForTimeout(2500);
  const s4Title = await page.locator('h1').innerText();
  const s4StepBtn = await page.locator('#stepBtn').count();
  console.log(`Screen 04: Title='${s4Title}', StepBtnCount=${s4StepBtn}`);

  console.log('=== 5. VERIFY SCREEN 05 (BÁO CÁO) ===');
  currentRoute = '/bao-cao';
  await page.locator('header nav a:has-text("Báo cáo")').click();
  await page.waitForTimeout(2500);
  const s5Title = await page.locator('h1').innerText();
  console.log(`Screen 05: Title='${s5Title}'`);

  console.log('=== 6. VERIFY SCREEN 06 (THỰC NGHIỆM) ===');
  currentRoute = '/thuc-nghiem';
  await page.locator('header nav a:has-text("Thực nghiệm")').click();
  await page.waitForTimeout(2500);
  const s6Title = await page.locator('h1').innerText();
  console.log(`Screen 06: Title='${s6Title}'`);

  console.log('=== 7. HARD RELOAD ON EACH ROUTE ===');
  const routes = ['/tong-quan', '/phan-cum', '/toi-uu-rl', '/mo-phong', '/bao-cao', '/thuc-nghiem'];
  for (const r of routes) {
    currentRoute = 'reload:' + r;
    await page.goto('http://127.0.0.1:5176' + r);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1500);
    const mainTextLen = await page.locator('main').innerText().then(t => t.trim().length);
    console.log(`Hard reload ${r}: mainTextLength = ${mainTextLen} (${mainTextLen > 500 ? 'PASS' : 'FAIL'})`);
  }

  console.log('=== 8. CONSOLE RUNTIME ERRORS CHECK ===');
  console.log('Console Errors recorded:', JSON.stringify(errors, null, 2));

  await browser.close();
}

verify().catch(console.error);
