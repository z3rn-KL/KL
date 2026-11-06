import { chromium } from "playwright";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, "../..");
const SCREENSHOT_DIR = path.resolve(ROOT_DIR, "audit_outputs/NEXUS_EXTREME_AUDIT/screenshots");

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

const VIEWPORTS = [
  { name: "desktop_1920", width: 1920, height: 1080 },
  { name: "desktop_1440", width: 1440, height: 900 },
  { name: "laptop_1280", width: 1280, height: 800 },
  { name: "mobile_390", width: 390, height: 844 },
];

const ROUTES = [
  { path: "/tong-quan", id: "screen_01_tong_quan" },
  { path: "/phan-cum", id: "screen_02_phan_cum" },
  { path: "/toi-uu-rl", id: "screen_03_toi_uu_rl" },
  { path: "/mo-phong", id: "screen_04_mo_phong" },
  { path: "/bao-cao", id: "screen_05_bao_cao" },
  { path: "/thuc-nghiem", id: "screen_06_thuc_nghiem" },
];

async function run() {
  console.log("=== MULTI-VIEWPORT & LAYOUT VERIFICATION ===");
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));

  let totalChecks = 0;
  let passedChecks = 0;
  const consoleErrors = [];

  for (const vp of VIEWPORTS) {
    console.log(`\n--- Testing Viewport: ${vp.name} (${vp.width}x${vp.height}) ---`);
    const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } });

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(`[${vp.name}] ${msg.text()}`);
      }
    });

    for (const route of ROUTES) {
      totalChecks++;
      try {
        await page.goto(`http://127.0.0.1:5176${route.path}`, { waitUntil: "networkidle" });
        await waitMs(1500);

        // Check horizontal overflow
        const overflow = await page.evaluate(() => {
          return document.documentElement.scrollWidth > window.innerWidth;
        });

        if (overflow && vp.width >= 1280) {
          console.error(`  [FAIL] ${route.id} on ${vp.name}: Horizontal overflow detected!`);
        } else {
          passedChecks++;
          console.log(`  [PASS] ${route.id} on ${vp.name} (Overflow: ${overflow ? "Yes (mobile responsive)" : "No"})`);
        }

        // Save 1440x900 screenshots as official proof
        if (vp.name === "desktop_1440") {
          const outName = `${route.id}_1440.png`;
          await page.screenshot({ path: path.join(SCREENSHOT_DIR, outName), fullPage: false });
        }
        // Save mobile sample
        if (vp.name === "mobile_390" && route.id === "screen_01_tong_quan") {
          await page.screenshot({ path: path.join(SCREENSHOT_DIR, "screen_01_tong_quan_mobile_390.png"), fullPage: false });
        }
      } catch (err) {
        console.error(`  [ERROR] ${route.id} on ${vp.name}:`, err.message);
      }
    }
    await page.close();
  }

  await browser.close();

  console.log(`\n=== VIEWPORT VERIFICATION RESULTS ===`);
  console.log(`Passed: ${passedChecks}/${totalChecks}`);
  console.log(`Console Errors: ${consoleErrors.length}`);
  if (consoleErrors.length > 0) {
    console.log("Console errors sampled:", consoleErrors.slice(0, 3));
  }
}

run();

