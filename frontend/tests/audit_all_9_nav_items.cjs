const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const OUT_DIR = path.resolve(__dirname, '../../audit/nav_active_audit');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

const BASE_URL = (process.env.BASE_URL || 'http://localhost:5173').trim();

const NAV_ITEMS = [
  { id: 'overview', to: '/tong-quan', title: 'Tổng quan' },
  { id: 'orders', to: '/don-hang', title: 'Đơn hàng' },
  { id: 'clusters', to: '/phan-cum', title: 'Phân cụm' },
  { id: 'routes', to: '/tuyen-giao', title: 'Tuyến giao' },
  { id: 'run-lab', to: '/chay-thu', title: 'Chạy thử' },
  { id: 'guided-dqn', to: '/guided-dqn', title: 'Guided DQN' },
  { id: 'compare', to: '/so-sanh', title: 'So sánh' },
  { id: 'sensitivity', to: '/do-nhay', title: 'Độ nhạy' },
  { id: 'stability', to: '/on-dinh', title: 'Độ ổn định' },
];

function parseRgb(rgbStr) {
  const match = rgbStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return [0, 0, 0];
  return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
}

function getLuminance(r, g, b) {
  const a = [r, g, b].map(v => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });
  return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
}

function calculateContrast(rgb1Str, rgb2Str) {
  const [r1, g1, b1] = parseRgb(rgb1Str);
  const [r2, g2, b2] = parseRgb(rgb2Str);
  const l1 = getLuminance(r1, g1, b1);
  const l2 = getLuminance(r2, g2, b2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

async function audit() {
  console.log('=== STARTING 9-ITEM TOP NAVIGATION ACTIVE STATE AUDIT ===');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  let totalAssertions = 0;
  let passedAssertions = 0;

  const testMatrix = [
    { label: 'desktop_light', width: 1366, height: 768, theme: 'light' },
    { label: 'desktop_dark', width: 1366, height: 768, theme: 'dark' },
    { label: 'mobile_light', width: 390, height: 844, theme: 'light' },
    { label: 'mobile_dark', width: 390, height: 844, theme: 'dark' }
  ];

  for (const config of testMatrix) {
    console.log(`\n--- Testing Environment: ${config.label} (${config.theme.toUpperCase()} MODE) ---`);
    const context = await browser.newContext({
      viewport: { width: config.width, height: config.height },
      colorScheme: config.theme
    });
    const page = await context.newPage();

    await page.addInitScript((th) => {
      localStorage.setItem('kltn11-theme', th);
    }, config.theme);

    await page.goto(`${BASE_URL}/tong-quan`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    for (const item of NAV_ITEMS) {
      // Find nav link by href
      const navSelector = `nav a[href="${item.to}"]`;
      await page.waitForSelector(navSelector);
      
      // Click nav item
      await page.click(navSelector);
      await page.waitForTimeout(300);

      // Check URL
      const currentUrl = page.url();
      const urlMatches = currentUrl.includes(item.to);
      totalAssertions++;
      if (urlMatches) passedAssertions++;

      // Measure styles of active item
      const metrics = await page.evaluate((sel) => {
        const a = document.querySelector(sel);
        const span = a ? a.querySelector('span') : null;
        const icon = a ? a.querySelector('svg') : null;
        if (!a) return null;

        const sA = window.getComputedStyle(a);
        const sSpan = span ? window.getComputedStyle(span) : null;
        const sIcon = icon ? window.getComputedStyle(icon) : null;
        const rectA = a.getBoundingClientRect();
        const rectSpan = span ? span.getBoundingClientRect() : null;

        return {
          isActiveClass: a.classList.contains('active'),
          text: a.innerText.trim(),
          spanText: span ? span.innerText.trim() : null,
          color: sA.color,
          bgColor: sA.backgroundColor,
          spanColor: sSpan ? sSpan.color : null,
          iconColor: sIcon ? sIcon.color : null,
          opacity: sA.opacity,
          visibility: sA.visibility,
          width: rectA.width,
          height: rectA.height,
          spanWidth: rectSpan ? rectSpan.width : 0,
          spanHeight: rectSpan ? rectSpan.height : 0
        };
      }, navSelector);

      if (!metrics) {
        throw new Error(`Failed to read metrics for ${item.title}`);
      }

      const contrastRatio = calculateContrast(metrics.spanColor, metrics.bgColor);
      const isContrastPass = contrastRatio >= 4.5;
      const isTextVisible = metrics.spanWidth > 10 && metrics.spanHeight > 10 && metrics.spanText === item.title;

      totalAssertions += 3;
      if (metrics.isActiveClass) passedAssertions++;
      if (isContrastPass) passedAssertions++;
      if (isTextVisible) passedAssertions++;

      const shotFile = `active_${config.label}_${item.id}.png`;
      await page.screenshot({ path: path.join(OUT_DIR, shotFile) });

      // Test reload persistence for first 2 items
      let reloadPreserved = true;
      if (item.id === 'orders' || item.id === 'routes') {
        await page.reload({ waitUntil: 'networkidle' });
        await page.waitForTimeout(300);
        const recheck = await page.evaluate((sel) => {
          const a = document.querySelector(sel);
          return a ? a.classList.contains('active') : false;
        }, navSelector);
        totalAssertions++;
        if (recheck) passedAssertions++;
        reloadPreserved = recheck;
      }

      const record = {
        item: item.title,
        route: item.to,
        env: config.label,
        theme: config.theme,
        textColor: metrics.spanColor,
        bgColor: metrics.bgColor,
        contrastRatio: contrastRatio.toFixed(2) + ':1',
        isContrastPass,
        isTextVisible,
        textWidth: metrics.spanWidth.toFixed(1) + 'px',
        textHeight: metrics.spanHeight.toFixed(1) + 'px',
        reloadPreserved,
        screenshot: shotFile
      };

      results.push(record);
      console.log(`  [${isContrastPass && isTextVisible ? 'PASS' : 'FAIL'}] ${item.title} (${item.to}): Text="${metrics.spanText}", Color=${metrics.spanColor}, Bg=${metrics.bgColor}, Contrast=${record.contrastRatio} (WCAG AA: ${isContrastPass})`);
    }

    // Test Keyboard navigation (Tab through nav)
    console.log(`  Testing Keyboard Navigation in ${config.label}...`);
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    const focusedTag = await page.evaluate(() => document.activeElement ? document.activeElement.tagName : null);
    console.log(`  Focused element after tabs: ${focusedTag}`);

    await context.close();
  }

  await browser.close();

  fs.writeFileSync(path.join(OUT_DIR, 'audit_report.json'), JSON.stringify(results, null, 2), 'utf8');
  console.log(`\n=== AUDIT COMPLETE ===`);
  console.log(`Total Assertions: ${totalAssertions}`);
  console.log(`Passed Assertions: ${passedAssertions}`);
  console.log(`Pass Rate: ${((passedAssertions / totalAssertions) * 100).toFixed(1)}%`);
  console.log(`Summary written to ${path.join(OUT_DIR, 'audit_report.json')}`);

  if (passedAssertions !== totalAssertions) {
    process.exit(1);
  }
}

audit().catch(err => {
  console.error(err);
  process.exit(1);
});

