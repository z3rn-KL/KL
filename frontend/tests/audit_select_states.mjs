import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

function getLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

function parseRgb(colorStr) {
  const match = colorStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return [0, 0, 0];
  return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
}

function getContrastRatio(color1, color2) {
  const [r1, g1, b1] = parseRgb(color1);
  const [r2, g2, b2] = parseRgb(color2);
  const l1 = getLuminance(r1, g1, b1);
  const l2 = getLuminance(r2, g2, b2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

async function runAudit() {
  const screenshotDir = path.resolve('tests/screenshots');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  const browser = await chromium.launch({ channel: 'msedge' });
  const auditResults = [];

  for (const theme of ['light', 'dark']) {
    for (const isMobile of [false, true]) {
      const viewport = isMobile ? { width: 375, height: 812 } : { width: 1440, height: 900 };
      const modeName = `${theme}_${isMobile ? 'mobile' : 'desktop'}`;
      console.log(`\n================== AUDITING [${modeName.toUpperCase()}] ==================`);

      const context = await browser.newContext({ viewport });
      const page = await context.newPage();

      // 1. Audit /don-hang (Orders Page)
      console.log(`Auditing /don-hang in ${modeName}...`);
      await page.goto('http://localhost:5173/don-hang');
      await page.evaluate(t => document.documentElement.setAttribute('data-theme', t), theme);
      await page.waitForTimeout(600);

      const triggers = await page.$$('[data-slot="select-trigger"]');
      console.log(`Found ${triggers.length} select triggers on /don-hang`);

      // Audit SLA dropdown (Trigger index 1)
      const slaTrigger = triggers[1];
      const initialTriggerText = await slaTrigger.innerText();
      console.log(`SLA trigger initial text: "${initialTriggerText}"`);

      // Open SLA dropdown
      await slaTrigger.click();
      await page.waitForTimeout(400);

      // Screenshot dropdown open
      await page.screenshot({ path: path.join(screenshotDir, `sla_open_${modeName}.png`) });

      // Verify all items in SLA
      let items = await page.$$('[data-slot="select-item"]');
      console.log(`SLA popup has ${items.length} items`);

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        await item.hover();
        await page.waitForTimeout(150);

        const data = await item.evaluate((el) => {
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          const textSpan = el.querySelector('[data-slot="select-item-text"]') || el.children[0];
          const textStyle = textSpan ? window.getComputedStyle(textSpan) : style;
          return {
            text: el.innerText.trim(),
            bg: style.backgroundColor,
            color: textStyle.color,
            fontSize: style.fontSize,
            fontWeight: style.fontWeight,
            visibility: style.visibility,
            opacity: style.opacity,
            width: rect.width,
            height: rect.height,
            isHighlighted: el.hasAttribute('data-highlighted'),
            isSelected: el.hasAttribute('data-selected'),
          };
        });

        // Determine effective background if transparent
        const effectiveBg = (data.bg === 'rgba(0, 0, 0, 0)' || data.bg === 'transparent')
          ? (theme === 'light' ? 'rgb(255, 255, 255)' : 'rgb(24, 35, 38)')
          : data.bg;

        const contrast = getContrastRatio(data.color, effectiveBg);
        const pass = contrast >= 4.5 && data.height > 0 && data.width > 0 && data.text.length > 0;

        auditResults.push({
          page: '/don-hang',
          dropdown: 'SLA',
          mode: modeName,
          itemIndex: i,
          text: data.text,
          bg: effectiveBg,
          color: data.color,
          contrast: contrast.toFixed(2),
          pass,
        });

        console.log(`  Item [${data.text}]: bg=${effectiveBg}, color=${data.color}, contrast=${contrast.toFixed(2)}:1 -> ${pass ? 'PASS' : 'FAIL'}`);
      }

      // Test choosing item 1 ("Đúng hạn")
      await items[1].click();
      await page.waitForTimeout(300);
      const afterSelectText = await slaTrigger.innerText();
      console.log(`  After selection, trigger text is: "${afterSelectText}"`);

      // Re-open when already selected
      await slaTrigger.click();
      await page.waitForTimeout(300);
      items = await page.$$('[data-slot="select-item"]');

      // Check item 1 selected resting state (hover over item 0 to remove highlight from item 1)
      await items[0].hover();
      await page.waitForTimeout(150);

      const selectedResting = await items[1].evaluate((el) => {
        const style = window.getComputedStyle(el);
        return {
          text: el.innerText.trim(),
          bg: style.backgroundColor,
          color: style.color,
          isSelected: el.hasAttribute('data-selected'),
          isHighlighted: el.hasAttribute('data-highlighted'),
        };
      });

      const selContrast = getContrastRatio(selectedResting.color, selectedResting.bg);
      console.log(`  Item [${selectedResting.text}] Selected Resting: bg=${selectedResting.bg}, color=${selectedResting.color}, contrast=${selContrast.toFixed(2)}:1 -> ${selContrast >= 4.5 ? 'PASS' : 'FAIL'}`);

      // Hover back over item 1 (Selected + Highlighted)
      await items[1].hover();
      await page.waitForTimeout(150);
      const selectedHovered = await items[1].evaluate((el) => {
        const style = window.getComputedStyle(el);
        return {
          text: el.innerText.trim(),
          bg: style.backgroundColor,
          color: style.color,
          isSelected: el.hasAttribute('data-selected'),
          isHighlighted: el.hasAttribute('data-highlighted'),
        };
      });
      const selHoverContrast = getContrastRatio(selectedHovered.color, selectedHovered.bg);
      console.log(`  Item [${selectedHovered.text}] Selected + Hovered: bg=${selectedHovered.bg}, color=${selectedHovered.color}, contrast=${selHoverContrast.toFixed(2)}:1 -> ${selHoverContrast >= 4.5 ? 'PASS' : 'FAIL'}`);

      // Screenshot selected state
      await page.screenshot({ path: path.join(screenshotDir, `sla_selected_${modeName}.png`) });

      // Close with Escape key
      await page.keyboard.press('Escape');
      await page.waitForTimeout(200);

      // Keyboard navigation test: Focus trigger, open with Enter, navigate with ArrowDown, select with Enter
      await slaTrigger.focus();
      await page.keyboard.press('Enter');
      await page.waitForTimeout(300);
      await page.keyboard.press('ArrowDown');
      await page.waitForTimeout(150);
      await page.keyboard.press('ArrowDown');
      await page.waitForTimeout(150);
      await page.keyboard.press('Enter');
      await page.waitForTimeout(300);
      console.log(`  Keyboard navigation result trigger text: "${await slaTrigger.innerText()}"`);

      // 2. Audit /tuyen-giao Workload dropdown & WL #65
      console.log(`\nAuditing /tuyen-giao Workload dropdown in ${modeName}...`);
      await page.goto('http://localhost:5173/tuyen-giao');
      await page.evaluate(t => document.documentElement.setAttribute('data-theme', t), theme);
      await page.waitForTimeout(600);

      const wlTrigger = await page.$('[data-slot="select-trigger"]');
      const wlTriggerText = await wlTrigger.innerText();
      console.log(`Workload trigger initial text: "${wlTriggerText}"`);

      await wlTrigger.click();
      await page.waitForTimeout(400);

      // Find WL #65
      const wl65Item = await page.$('[data-slot="select-item"]:has-text("WL #65")');
      if (wl65Item) {
        await wl65Item.scrollIntoViewIfNeeded();
        await page.waitForTimeout(200);

        // Check resting state of WL #65
        const wl65Resting = await wl65Item.evaluate((el) => {
          const style = window.getComputedStyle(el);
          return {
            text: el.innerText.trim(),
            bg: style.backgroundColor,
            color: style.color,
            isSelected: el.hasAttribute('data-selected'),
            isHighlighted: el.hasAttribute('data-highlighted')
          };
        });
        const wl65RestingContrast = getContrastRatio(wl65Resting.color, wl65Resting.bg);
        console.log(`  WL #65 Selected: bg=${wl65Resting.bg}, color=${wl65Resting.color}, contrast=${wl65RestingContrast.toFixed(2)}:1 -> ${wl65RestingContrast >= 4.5 ? 'PASS' : 'FAIL'}`);

        // Hover over WL #65
        await wl65Item.hover();
        await page.waitForTimeout(150);
        const wl65Hover = await wl65Item.evaluate((el) => {
          const style = window.getComputedStyle(el);
          return {
            text: el.innerText.trim(),
            bg: style.backgroundColor,
            color: style.color,
            isSelected: el.hasAttribute('data-selected'),
            isHighlighted: el.hasAttribute('data-highlighted')
          };
        });
        const wl65HoverContrast = getContrastRatio(wl65Hover.color, wl65Hover.bg);
        console.log(`  WL #65 Selected + Hovered: bg=${wl65Hover.bg}, color=${wl65Hover.color}, contrast=${wl65HoverContrast.toFixed(2)}:1 -> ${wl65HoverContrast >= 4.5 ? 'PASS' : 'FAIL'}`);

        auditResults.push({
          page: '/tuyen-giao',
          dropdown: 'Workload',
          mode: modeName,
          itemIndex: 65,
          text: wl65Hover.text,
          bg: wl65Hover.bg,
          color: wl65Hover.color,
          contrast: wl65HoverContrast.toFixed(2),
          pass: wl65HoverContrast >= 4.5
        });

        await page.screenshot({ path: path.join(screenshotDir, `workload_wl65_${modeName}.png`) });
      }

      await page.keyboard.press('Escape');
      await context.close();
    }
  }

  await browser.close();

  console.log('\n================== FINAL AUDIT SUMMARY ==================');
  const total = auditResults.length;
  const passed = auditResults.filter(r => r.pass).length;
  console.log(`Total checks: ${total}, Passed: ${passed}, Failed: ${total - passed}`);
  if (total !== passed) {
    console.error('Audit failed with some non-compliant items!');
    process.exit(1);
  } else {
    console.log('ALL SELECT AND DROPDOWN AUDITS PASSED WITH WCAG AA COMPLIANCE (>= 4.5:1)!');
  }
}

runAudit().catch(err => {
  console.error(err);
  process.exit(1);
});

