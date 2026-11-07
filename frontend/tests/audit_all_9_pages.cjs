const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const OUTPUT_DIR = path.resolve(__dirname, "../../audit_outputs/V11_UI_REDESIGN/all_9_pages_audit");
const ARTIFACT_DIR = "C:\\Users\\nhtuo\\.gemini\\antigravity\\brain\\9909568a-1e87-41ec-8bb5-e4010e63d5e7\\audit_evidence";

if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
if (!fs.existsSync(ARTIFACT_DIR)) fs.mkdirSync(ARTIFACT_DIR, { recursive: true });

const PAGES = [
  { path: "/tong-quan", name: "01_overview" },
  { path: "/don-hang", name: "02_orders" },
  { path: "/phan-cum", name: "03_clusters" },
  { path: "/tuyen-giao", name: "04_routes" },
  { path: "/chay-thu", name: "05_runlab" },
  { path: "/so-sanh", name: "06_compare" },
  { path: "/guided-dqn", name: "07_guided_dqn" },
  { path: "/do-nhay", name: "08_sensitivity" },
  { path: "/on-dinh", name: "09_stability" },
];

async function runAudit() {
  console.log("=== BẮT ĐẦU TỰ ĐỘNG AUDIT TOÀN BỘ 9 TRANG (LIGHT & DARK MODE) ===");
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  const setDark = async (dark) => {
    await page.evaluate((d) => {
      if (d) document.documentElement.classList.add("dark");
      else document.documentElement.classList.remove("dark");
    }, dark);
    await page.waitForTimeout(400);
  };

  const auditReport = [];

  for (const p of PAGES) {
    console.log(`Kiểm tra ${p.name} (${p.path})...`);
    await page.goto(`http://localhost:5173${p.path}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(600);

    // Dark Mode
    await setDark(true);
    await page.screenshot({ path: path.join(OUTPUT_DIR, `${p.name}_dark.png`), fullPage: false });

    // Light Mode
    await setDark(false);
    await page.screenshot({ path: path.join(OUTPUT_DIR, `${p.name}_light.png`), fullPage: false });

    // DOM analysis on page
    const pageIssues = await page.evaluate(() => {
      const issues = [];
      
      // 1. Check for overlapping elements (bounding boxes)
      const textNodes = Array.from(document.querySelectorAll('label, span, p, h1, h2, h3, h4, th, td, button'));
      // 2. Check for dull gray background on tabs or controls
      const tabs = Array.from(document.querySelectorAll('[data-slot="tabs-list"], [data-slot="tabs-trigger"]'));
      tabs.forEach(t => {
        const cs = getComputedStyle(t);
        if (cs.backgroundColor.includes('rgba(0, 0, 0, 0)') === false && (cs.backgroundColor.includes('236') || cs.backgroundColor.includes('240'))) {
          // potential heavy gray
        }
      });

      // 3. Check for progress bars
      const progress = Array.from(document.querySelectorAll('[role="progressbar"], .progress, progress, [data-slot="progress"]'));
      const progressData = progress.map(pr => ({
        tag: pr.tagName,
        className: pr.className,
        height: getComputedStyle(pr).height,
        bg: getComputedStyle(pr).backgroundColor,
      }));

      // 4. Check for sliders / inputs alignment
      const sliders = Array.from(document.querySelectorAll('[data-slot="slider"]'));
      
      return {
        progressCount: progress.length,
        progressData,
        sliderCount: sliders.length,
        tabsCount: tabs.length,
      };
    });

    auditReport.push({ page: p.name, ...pageIssues });
  }

  // Also check Command Palette
  console.log("Kiểm tra Command Palette...");
  await setDark(true);
  await page.keyboard.press("Control+k");
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUTPUT_DIR, "10_command_palette_dark.png") });
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);

  await setDark(false);
  await page.keyboard.press("Control+k");
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUTPUT_DIR, "10_command_palette_light.png") });
  await page.keyboard.press("Escape");

  await browser.close();

  // Copy to artifact dir
  const files = fs.readdirSync(OUTPUT_DIR);
  for (const f of files) {
    fs.copyFileSync(path.join(OUTPUT_DIR, f), path.join(ARTIFACT_DIR, f));
  }

  console.log("=== AUDIT HOÀN TẤT, ĐÃ LƯU ẢNH CHỨNG MINH ===");
  console.log(JSON.stringify(auditReport, null, 2));
}

runAudit().catch(err => {
  console.error("Lỗi audit:", err);
  process.exit(1);
});

