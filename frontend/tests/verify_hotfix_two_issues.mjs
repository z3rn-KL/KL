import { chromium } from "playwright";
import path from "path";
import fs from "fs";

const OUTPUT_DIR = path.resolve("audit_outputs/NEXUS_HOTFIX");
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function run() {
  console.log("=== BẮT ĐẦU KIỂM THỬ XÁC MINH HAI LỖI HOTFIX ===");
  const browser = await chromium.launch({ channel: "msedge", headless: true });

  // ---------------------------------------------------------------------------
  // 1. TỔNG QUAN Ở 1440x900 VÀ 1920x1080
  // ---------------------------------------------------------------------------
  console.log("\n[1] Kiểm thử Tổng quan (/tong-quan)...");
  
  // 1440x900
  const ctx1440 = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const p1440 = await ctx1440.newPage();
  await p1440.goto("http://127.0.0.1:5176/tong-quan", { waitUntil: "networkidle" });
  await p1440.waitForSelector(".leaflet-container", { timeout: 15000 });
  await wait(3500); // Wait for all 2,394 points and tile layers

  await p1440.screenshot({ path: path.join(OUTPUT_DIR, "OVERVIEW_1440x900.png") });
  console.log("  -> Đã chụp: OVERVIEW_1440x900.png");

  // Check DOM layout structure
  const overviewMeasurements = await p1440.evaluate(() => {
    const hero = document.querySelector(".rounded-xl.border-teal-500\\/30");
    const map = document.querySelector(".leaflet-container");
    const sidebar = document.querySelector(".grid.grid-cols-1.lg\\:grid-cols-12 > div:last-child");
    const allButtons = Array.from(document.querySelectorAll("button"));
    const ctas = allButtons.filter(b => b.textContent.includes("Dữ liệu") || b.textContent.includes("Bàn Lập Tuyến"));
    return {
      heroHeight: hero ? Math.round(hero.getBoundingClientRect().height) : 0,
      mapHeight: map ? Math.round(map.getBoundingClientRect().height) : 0,
      sidebarHeight: sidebar ? Math.round(sidebar.getBoundingClientRect().height) : 0,
      ctaCount: ctas.length,
      windowScrollHeight: document.documentElement.scrollHeight,
      windowClientHeight: document.documentElement.clientHeight,
    };
  });
  console.log("  -> Đo đạc bố cục Tổng quan 1440x900:", overviewMeasurements);
  await ctx1440.close();

  // 1920x1080
  const ctx1920 = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const p1920 = await ctx1920.newPage();
  await p1920.goto("http://127.0.0.1:5176/tong-quan", { waitUntil: "networkidle" });
  await p1920.waitForSelector(".leaflet-container", { timeout: 15000 });
  await wait(3500);
  await p1920.screenshot({ path: path.join(OUTPUT_DIR, "OVERVIEW_1920x1080.png") });
  console.log("  -> Đã chụp: OVERVIEW_1920x1080.png");
  await ctx1920.close();

  // ---------------------------------------------------------------------------
  // 2. SIMULATION CONTROL DOCK TRÊN /mo-phong
  // ---------------------------------------------------------------------------
  console.log("\n[2] Kiểm thử Simulation Control Dock (/mo-phong)...");
  const ctxSim = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const pSim = await ctxSim.newPage();
  await pSim.goto("http://127.0.0.1:5176/mo-phong", { waitUntil: "networkidle" });
  await wait(2500);

  const stepBtn = pSim.locator("#stepBtn");
  const simbar = pSim.locator(".simbar");

  // A. Normal State
  const normalClip = await simbar.boundingBox();
  if (normalClip) {
    await pSim.screenshot({ path: path.join(OUTPUT_DIR, "SIMDOCK_01_NORMAL.png"), clip: normalClip });
  }

  const normalStyles = await stepBtn.evaluate((el) => {
    const s = window.getComputedStyle(el);
    return {
      color: s.color,
      backgroundColor: s.backgroundColor,
      borderColor: s.borderColor,
      height: s.height,
      opacity: s.opacity,
    };
  });
  console.log("  -> Computed style [NORMAL]:", normalStyles);

  // B. Hover State
  await stepBtn.hover();
  await wait(400);
  if (normalClip) {
    // Add 45px top to capture tooltip above
    const hoverClip = {
      x: normalClip.x,
      y: Math.max(0, normalClip.y - 45),
      width: normalClip.width,
      height: normalClip.height + 45,
    };
    await pSim.screenshot({ path: path.join(OUTPUT_DIR, "SIMDOCK_02_HOVER.png"), clip: hoverClip });
  }

  const hoverStyles = await stepBtn.evaluate((el) => {
    const s = window.getComputedStyle(el);
    return {
      color: s.color,
      backgroundColor: s.backgroundColor,
      borderColor: s.borderColor,
      height: s.height,
    };
  });
  console.log("  -> Computed style [HOVER]:", hoverStyles);

  // C. Focus State
  await stepBtn.focus();
  await wait(300);
  if (normalClip) {
    await pSim.screenshot({ path: path.join(OUTPUT_DIR, "SIMDOCK_03_FOCUS.png"), clip: normalClip });
  }
  const focusStyles = await stepBtn.evaluate((el) => {
    const s = window.getComputedStyle(el);
    return {
      color: s.color,
      backgroundColor: s.backgroundColor,
      outline: s.outline,
    };
  });
  console.log("  -> Computed style [FOCUS]:", focusStyles);

  // D. Thao tác Click +15 phút
  const clockBefore = await pSim.locator("#simulationTime").innerText();
  console.log(`  -> Đồng hồ trước khi click: ${clockBefore}`);
  await stepBtn.click();
  await wait(1500);
  const clockAfter = await pSim.locator("#simulationTime").innerText();
  console.log(`  -> Đồng hồ sau khi click (+15p): ${clockAfter}`);

  // E. Dark Mode check
  await pSim.evaluate(() => {
    document.documentElement.classList.add("dark");
    document.documentElement.setAttribute("data-theme", "dark");
  });
  await wait(400);
  await stepBtn.hover();
  await wait(300);
  if (normalClip) {
    const hoverClip = {
      x: normalClip.x,
      y: Math.max(0, normalClip.y - 45),
      width: normalClip.width,
      height: normalClip.height + 45,
    };
    await pSim.screenshot({ path: path.join(OUTPUT_DIR, "SIMDOCK_04_HOVER_DARK.png"), clip: hoverClip });
  }

  // F. Disabled State check (End shift)
  console.log("  -> Chốt ca để kiểm tra trạng thái Disabled của #stepBtn...");
  const endDayBtn = pSim.locator("#endDayBtn");
  if (await endDayBtn.isVisible()) {
    await endDayBtn.click();
    await wait(2500);
    // Return to /mo-phong to inspect ended state
    await pSim.goto("http://127.0.0.1:5176/mo-phong", { waitUntil: "networkidle" });
    await wait(1500);
    const disabledStep = pSim.locator("#stepBtn");
    const isDisabled = await disabledStep.isDisabled();
    console.log(`  -> #stepBtn isDisabled: ${isDisabled}`);
    const disabledClip = await pSim.locator(".simbar").boundingBox();
    if (disabledClip) {
      await pSim.screenshot({ path: path.join(OUTPUT_DIR, "SIMDOCK_05_DISABLED.png"), clip: disabledClip });
    }
  }

  await ctxSim.close();
  await browser.close();

  console.log("\n=== TẤT CẢ KIỂM THỬ XÁC MINH HOÀN TẤT ===");
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
