import { chromium } from "playwright";

const BASE_URL = "http://127.0.0.1:5176";
const API_URL = "http://127.0.0.1:8011";

async function runThreeDayLifecycleTest() {
  console.log("==================================================================");
  console.log("🚀 STARTING THREE-DAY LIFECYCLE & SCIENTIFIC INTEGRITY AUDIT");
  console.log("==================================================================");

  const results = [];
  function logResult(step, name, passed, detail = "") {
    results.push({ step, name, passed, detail });
    console.log(`[${passed ? "✅ PASS" : "❌ FAIL"}] [${step}] ${name} ${detail ? `-> ${detail}` : ""}`);
  }

  // Helper fetch
  async function apiPost(endpoint, body = {}) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API ${endpoint} failed (${res.status}): ${txt}`);
    }
    return res.json();
  }

  async function apiGet(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API ${endpoint} failed (${res.status}): ${txt}`);
    }
    return res.json();
  }

  const browser = await chromium.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: true,
  });

  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  try {
    // -------------------------------------------------------------------
    // STEP 1: INITIALIZE SESSION (Day 1)
    // -------------------------------------------------------------------
    console.log("\n--- STEP 1: Initializing New Dispatch Session ---");
    const sessionInit = await apiPost("/api/dispatch/sessions", {
      scenario_id: "default_3day_scenario",
      routing_algorithm: "proposed_rl",
      total_days: 3,
      seed: 42,
    });

    const sessionId = sessionInit.session_id;
    logResult("Day 1 Init", "Session created successfully", !!sessionId, `ID: ${sessionId}`);

    const sessDay1 = await apiGet(`/api/dispatch/sessions/${sessionId}`);
    logResult("Day 1 Init", "Day is 1", sessDay1.current_day === 1, `current_day: ${sessDay1.current_day}`);
    logResult("Day 1 Init", "Day 1 has 85 available orders", sessDay1.orders.length === 85, `Order count: ${sessDay1.orders.length}`);

    // -------------------------------------------------------------------
    // STEP 2: SCREEN 03 PLANNING (Day 1)
    // -------------------------------------------------------------------
    console.log("\n--- STEP 2: Screen 03 (Planning Desk) Day 1 ---");
    await page.goto(`${BASE_URL}/toi-uu-rl?session_id=${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("h1, h2", { timeout: 10000 });
    await page.waitForTimeout(1500);

    const pageText = await page.innerText("body");
    const hasPlanningDesk = pageText.toLowerCase().includes("bàn lập tuyến");
    logResult("Screen 03", "Planning Desk header verified", hasPlanningDesk);

    // Compute preview
    console.log("Triggering route preview computation...");
    const previewRes = await apiPost(`/api/dispatch/sessions/${sessionId}/preview`, {
      algorithm: "proposed_rl",
      strategy: "cluster_aware_rl",
    });
    logResult("Screen 03", "Preview computation API", previewRes.success, `Preview ID: ${previewRes.preview_id}`);

    // Confirm plan
    console.log("Confirming plan...");
    const applyRes = await apiPost(`/api/dispatch/sessions/${sessionId}/confirm-preview`, {
      preview_id: previewRes.preview_id,
    });
    logResult("Screen 03", "Confirm Plan applied", applyRes.success, `Dispatched: ${applyRes.dispatched_orders}`);

    // -------------------------------------------------------------------
    // STEP 3: SCREEN 04 SIMULATION (Day 1)
    // -------------------------------------------------------------------
    console.log("\n--- STEP 3: Screen 04 (Simulation) Day 1 ---");
    await page.goto(`${BASE_URL}/mo-phong?session_id=${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("h1", { timeout: 12000 });
    await page.waitForTimeout(1500);

    const simBody = (await page.innerText("body")).toLowerCase();
    // Check 6 standardized KPIs (case-insensitive)
    const kpiNames = [
      "ngày mô phỏng",
      "thời gian đã trôi qua",
      "đơn đã giao",
      "đơn đang giao",
      "shipper đang hoạt động",
      "quãng đường đã di chuyển",
    ];
    for (const kpi of kpiNames) {
      logResult("Screen 04 KPIs", `KPI card: "${kpi}" present`, simBody.includes(kpi));
    }

    // Step simulation clock by 120 mins
    for (let i = 0; i < 4; i++) {
      await apiPost(`/api/dispatch/sessions/${sessionId}/clock/step`, { delta_minutes: 30 });
    }
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForTimeout(1500);

    const sessAfterStep = await apiGet(`/api/dispatch/sessions/${sessionId}`);
    logResult("Screen 04 Clock", "Clock stepped forward", sessAfterStep.simulation_clock > 0, `Clock: ${sessAfterStep.simulation_clock}m`);

    // End Day 1
    console.log("Ending Day 1 shift...");
    const endDay1Res = await apiPost(`/api/dispatch/sessions/${sessionId}/end-day`);
    logResult("Day 1 End", "Day 1 ended successfully", endDay1Res.success, `Carried over: ${endDay1Res.carried_over_orders_count}`);

    // -------------------------------------------------------------------
    // STEP 4: SCREEN 05 REPORT (Day 1)
    // -------------------------------------------------------------------
    console.log("\n--- STEP 4: Screen 05 (Shift Report) Day 1 ---");
    await page.goto(`${BASE_URL}/bao-cao?session_id=${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("h1", { timeout: 12000 });
    await page.waitForTimeout(1500);

    const sessEndD1 = await apiGet(`/api/dispatch/sessions/${sessionId}`);
    const d1Metric = sessEndD1.daily_metrics?.find(m => m.day === 1);
    logResult("Screen 05", "Day 1 Metric recorded in session", !!d1Metric, `Delivered: ${d1Metric?.delivered || 0}`);

    const reportPageText = await page.innerText("body");
    const hasReportDay1 = reportPageText.toLowerCase().includes("báo cáo ca");
    logResult("Screen 05", "Shift Report page rendered", hasReportDay1);

    // -------------------------------------------------------------------
    // STEP 5: ADVANCE TO DAY 2 & LIFECYCLE CHECK
    // -------------------------------------------------------------------
    console.log("\n--- STEP 5: Advancing to Day 2 ---");
    const nextDay2Res = await apiPost(`/api/dispatch/sessions/${sessionId}/next-day`);
    logResult("Day 2 Advance", "Advanced to Day 2", nextDay2Res.day === 2, `Day: ${nextDay2Res.day}`);
    logResult("Day 2 Invariants", "New incoming orders = 85", nextDay2Res.new_incoming_count === 85, `Incoming: ${nextDay2Res.new_incoming_count}`);
    logResult("Day 2 Invariants", "Carried over count matches Day 1 carryout", nextDay2Res.carried_over_count === endDay1Res.carried_over_orders_count, `Carried in: ${nextDay2Res.carried_over_count}`);

    // Preview and apply routes for Day 2
    const prev2 = await apiPost(`/api/dispatch/sessions/${sessionId}/preview`, {
      algorithm: "proposed_rl",
      strategy: "cluster_aware_rl",
    });
    await apiPost(`/api/dispatch/sessions/${sessionId}/confirm-preview`, { preview_id: prev2.preview_id });

    // Step simulation on Day 2
    for (let i = 0; i < 4; i++) {
      await apiPost(`/api/dispatch/sessions/${sessionId}/clock/step`, { delta_minutes: 30 });
    }
    const endDay2Res = await apiPost(`/api/dispatch/sessions/${sessionId}/end-day`);
    logResult("Day 2 End", "Day 2 ended successfully", endDay2Res.success, `Carried over: ${endDay2Res.carried_over_orders_count}`);

    // -------------------------------------------------------------------
    // STEP 6: ADVANCE TO DAY 3 & MULTIDAY FINAL REPORT
    // -------------------------------------------------------------------
    console.log("\n--- STEP 6: Advancing to Day 3 ---");
    const nextDay3Res = await apiPost(`/api/dispatch/sessions/${sessionId}/next-day`);
    logResult("Day 3 Advance", "Advanced to Day 3", nextDay3Res.day === 3, `Day: ${nextDay3Res.day}`);

    const prev3 = await apiPost(`/api/dispatch/sessions/${sessionId}/preview`, {
      algorithm: "proposed_rl",
      strategy: "cluster_aware_rl",
    });
    await apiPost(`/api/dispatch/sessions/${sessionId}/confirm-preview`, { preview_id: prev3.preview_id });

    for (let i = 0; i < 4; i++) {
      await apiPost(`/api/dispatch/sessions/${sessionId}/clock/step`, { delta_minutes: 30 });
    }
    const endDay3Res = await apiPost(`/api/dispatch/sessions/${sessionId}/end-day`);
    logResult("Day 3 End", "Day 3 ended successfully", endDay3Res.success);

    // Final Report verification on Screen 05
    await page.goto(`${BASE_URL}/bao-cao?session_id=${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("h1", { timeout: 12000 });
    await page.waitForTimeout(1500);

    const sessEndD3 = await apiGet(`/api/dispatch/sessions/${sessionId}`);
    logResult("Final Report", "3-Day metrics recorded", sessEndD3.daily_metrics?.length === 3, `Metrics count: ${sessEndD3.daily_metrics?.length}`);

    // Invariant: Total orders released across 3 days must equal 255 (85 * 3)
    const totalOrderCount = sessEndD3.orders.length;
    logResult("Scientific Invariant", "Total orders released across 3 days = 255", totalOrderCount === 255, `Actual: ${totalOrderCount}`);

    // Invariant: Check duplicate order deliveries
    const deliveredOrders = sessEndD3.orders.filter(o => o.status === "DELIVERED");
    const deliveredIds = deliveredOrders.map(o => o.order_id);
    const uniqueDeliveredIds = new Set(deliveredIds);
    logResult("Scientific Invariant", "No duplicate order deliveries", deliveredIds.length === uniqueDeliveredIds.size, `Unique delivered: ${uniqueDeliveredIds.size} / ${deliveredIds.length}`);

    // -------------------------------------------------------------------
    // STEP 7: NEW SESSION RESET & PREVIEW VALIDATION
    // -------------------------------------------------------------------
    console.log("\n--- STEP 7: New Session Reset & Clean Slate ---");
    const resetSession = await apiPost("/api/dispatch/sessions", {
      scenario_id: "default_3day_scenario",
      routing_algorithm: "proposed_rl",
      total_days: 3,
      seed: 99,
    });
    const newSid = resetSession.session_id;
    logResult("Session Reset", "New independent session created", !!newSid && newSid !== sessionId, `New ID: ${newSid}`);

    const resetSess = await apiGet(`/api/dispatch/sessions/${newSid}`);
    logResult("Session Reset", "Reset session starts at Day 1", resetSess.current_day === 1);
    logResult("Session Reset", "Reset session has exactly 85 pristine orders", resetSess.orders.length === 85);

    const resetPreview = await apiPost(`/api/dispatch/sessions/${newSid}/preview`, {
      algorithm: "proposed_rl",
      strategy: "cluster_aware_rl",
    });
    logResult("Session Reset", "Preview validates on reset session", resetPreview.success && !!resetPreview.preview_id);

  } catch (err) {
    console.error("❌ Fatal error in 3-day lifecycle test:", err);
    logResult("Fatal Exception", err.message, false);
  } finally {
    await browser.close();
  }

  console.log("\n==================================================================");
  console.log("📊 THREE-DAY LIFECYCLE AUDIT SUMMARY");
  console.log("==================================================================");
  const total = results.length;
  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  console.log(`Total assertions: ${total} | Passed: ${passed} | Failed: ${failed}`);

  if (failed === 0) {
    console.log("🏆 100% GREEN! THREE-DAY LIFECYCLE & SCIENTIFIC INVARIANTS VERIFIED!");
    process.exit(0);
  } else {
    console.error(`⚠️ ${failed} assertions failed! Review log above.`);
    process.exit(1);
  }
}

runThreeDayLifecycleTest();
