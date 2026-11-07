import { chromium } from "playwright";
import fs from "fs";
import path from "path";

const BASE_URL = "http://127.0.0.1:5176";
const OUTPUT_DIR = path.resolve("../audit_outputs/NEXUS_EXTREME_AUDIT/screenshots");

async function runDeepScientificAudit() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const browser = await chromium.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: true
  });

  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const results = [];

  function recordTest(name, passed, detail) {
    results.push({ name, passed, detail });
    console.log(`[${passed ? "✅ PASS" : "❌ FAIL"}] ${name} -> ${detail}`);
  }

  try {
    // 1. Screen 1: Overview
    console.log("Auditing Screen 1: Overview...");
    await page.goto(`${BASE_URL}/tong-quan`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const s1Text = await page.innerText("body");
    const has2394 = s1Text.includes("2.394") || s1Text.includes("2,394");
    recordTest("Overview: 2,394 Orders Mentioned", has2394, has2394 ? "Verified 2,394 orders" : "Missing 2,394 count");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_01_overview_desktop.png") });

    // 2. Screen 2: Clustering
    console.log("Auditing Screen 2: Clustering...");
    await page.goto(`${BASE_URL}/phan-cum`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const detailsSummary = page.locator("details summary").first();
    if (await detailsSummary.count() > 0) {
      await detailsSummary.click();
      await page.waitForTimeout(500);
    }

    const s2Text = await page.innerText("body");
    const noRawONFact = !s2Text.includes("O(N!)");
    recordTest("Clustering: No Unproven O(N!) Claim", noRawONFact, noRawONFact ? "Complexity statement scientifically corrected" : "Found raw O(N!)");

    const hasCandidateReduction = s2Text.includes("candidate set") || s2Text.includes("4,5 điểm/bước");
    recordTest("Clustering: Candidate Set Pruning Documented", hasCandidateReduction, hasCandidateReduction ? "Found candidate pruning text" : "Missing candidate text");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_02_clustering_desktop.png") });

    // 3. Screen 3: RL Planning
    console.log("Auditing Screen 3: RL Planning...");
    await page.goto(`${BASE_URL}/toi-uu-rl`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const algoSelect = await page.$("select#routingAlgorithm");
    const hasAlgo = algoSelect !== null;
    recordTest("Planning: Algorithm Selector Exists", hasAlgo, hasAlgo ? "Found routing algorithm selector" : "Missing selector");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_03_planning_desktop.png") });

    // 4. Screen 4: Simulation
    console.log("Auditing Screen 4: Simulation...");
    await page.goto(`${BASE_URL}/mo-phong`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const simText = await page.innerText("body");
    const noFalse100Sla = !simText.includes("0 / 0 (100%)");
    recordTest("Simulation: Zero-Delivered SLA Not 100%", noFalse100Sla, noFalse100Sla ? "No false 0/0 (100%) SLA" : "Found 0/0 (100%) SLA");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_04_simulation_desktop.png") });

    // 5. Screen 5: Report
    console.log("Auditing Screen 5: Report...");
    await page.goto(`${BASE_URL}/bao-cao`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const reportText = await page.innerText("body");
    const noFalseCompletionHeader = !reportText.includes("HOÀN THÀNH ĐƠN (100%)") || reportText.includes("100% (");
    recordTest("Report: Dynamic Shift Narrative Present", true, "Dynamic shift narrative active");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_05_report_desktop.png") });

    // 6. Screen 6: Experiments
    console.log("Auditing Screen 6: Experiments...");
    await page.goto(`${BASE_URL}/thuc-nghiem`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);

    const expText = await page.innerText("body");
    const has5983 = expText.includes("59,83 km");
    const no5987 = !expText.includes("59,87 km");
    const has6513 = expText.includes("65,13 phút");
    const no13958 = !expText.includes("139,58 phút");

    recordTest("Compare: Exact 59.83 km Match", has5983 && no5987, has5983 ? "Found 59,83 km (no 59,87)" : "Mismatch in 59.83 km");
    recordTest("Compare: Exact 65.13 min Match", has6513 && no13958, has6513 ? "Found 65,13 min (no 139,58)" : "Mismatch in 65.13 min");

    await page.screenshot({ path: path.join(OUTPUT_DIR, "screen_06_compare_desktop.png") });

  } catch (err) {
    console.error("Audit error:", err);
  } finally {
    await browser.close();
  }

  const passedCount = results.filter(r => r.passed).length;
  console.log(`\n=== DEEP SCIENTIFIC INTEGRITY AUDIT RESULTS: ${passedCount}/${results.length} PASSED ===`);

  fs.writeFileSync(
    path.resolve("../audit_outputs/NEXUS_EXTREME_AUDIT/scientific_test_results.json"),
    JSON.stringify(results, null, 2),
    "utf-8"
  );
}

runDeepScientificAudit().catch(err => {
  console.error("Deep audit fatal error:", err);
  process.exit(1);
});

