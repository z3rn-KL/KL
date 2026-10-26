import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { Marker, Polyline, useMap } from "react-leaflet";
import { validLL } from "../lib/geo";

type S = { playing: boolean; speed: number; progress: number; run: number };

/** External store for smooth 60fps animation without triggering full page re-renders. */
export interface AnimBus {
  s: S;
  subs: Set<() => void>;
  set: (p: Partial<S>) => void;
}

export function createBus(): AnimBus {
  const bus: AnimBus = {
    s: { playing: true, speed: 1, progress: 0, run: 0 },
    subs: new Set(),
    set(p) {
      bus.s = { ...bus.s, ...p };
      bus.subs.forEach((f) => f());
    },
  };
  return bus;
}

export function useBus(bus: AnimBus): S {
  const [s, setS] = useState(bus.s);
  useEffect(() => {
    const f = () => setS(bus.s);
    bus.subs.add(f);
    f();
    return () => {
      bus.subs.delete(f);
    };
  }, [bus]);
  return s;
}

const vehIcon = L.divIcon({
  className: "",
  html: '<div class="veh"><span style="border-color: #00f0ff; box-shadow: 0 0 12px #00f0ff;"></span></div>',
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

const BASE_SECONDS = 14;
type Pt = [number, number];

function cumulative(pts: Pt[]) {
  if (!pts.length || !Number.isFinite(pts[0][0]) || !Number.isFinite(pts[0][1])) {
    return new Array(pts.length).fill(0);
  }
  const k = Math.cos((pts[0][0] * Math.PI) / 180);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    const dx = (pts[i][1] - pts[i - 1][1]) * k;
    const dy = pts[i][0] - pts[i - 1][0];
    const d = Math.hypot(dx, dy);
    cum.push(cum[i - 1] + (Number.isFinite(d) ? d : 0));
  }
  return cum;
}

/** Remaps low-contrast dark blues into luminous neon cyan to prevent blending with dark tiles */
export function getContrastingRouteColor(c: string): string {
  const lower = c.toLowerCase();
  if (
    lower === "#2563eb" ||
    lower === "#1d4ed8" ||
    lower === "#1e40af" ||
    lower === "#60a5fa" ||
    lower === "#3b82f6" ||
    lower === "#1e3a8a"
  ) {
    return "#00f0ff"; // High-luminance electric cyan
  }
  return c;
}

export function RouteAnimator({
  positions,
  color,
  resetKey,
  bus,
}: {
  positions: Pt[];
  color: string;
  resetKey: string;
  bus: AnimBus;
}) {
  const s = useBus(bus);
  const map = useMap();
  const activeColor = useMemo(() => getContrastingRouteColor(color), [color]);

  const cum = useMemo(
    () => (positions.length > 1 ? cumulative(positions) : [0]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [resetKey],
  );

  useEffect(() => {
    bus.set({ progress: 0, playing: true, run: bus.s.run + 1 });
  }, [resetKey, bus]);

  const loopRef = useRef(0);
  useEffect(() => {
    if (!s.playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const p = bus.s.progress + (dt * bus.s.speed) / BASE_SECONDS;
      if (p >= 1) {
        bus.set({ progress: 1, playing: false });
        return;
      }
      bus.set({ progress: p });
      loopRef.current = requestAnimationFrame(tick);
    };
    loopRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(loopRef.current);
  }, [s.playing, s.run, bus]);

  // Sequential stop markers appearance according to animation progress
  useEffect(() => {
    const container = map.getContainer();
    const pins = Array.from(container.querySelectorAll<HTMLElement>(".stop-pin"));
    if (!pins.length) return;

    // Sort by stop number
    const sorted = [...pins].sort((a, b) => {
      const na = Number(a.getAttribute("data-stop") || "0");
      const nb = Number(b.getAttribute("data-stop") || "0");
      return na - nb;
    });

    const nStops = sorted.length;
    sorted.forEach((pin) => {
      const num = Number(pin.getAttribute("data-stop") || "0");
      if (!num) return;
      const threshold = num === 1 ? 0 : (num - 0.75) / nStops;
      const isReached = s.progress >= threshold || s.progress >= 0.98;

      if (isReached) {
        if (!pin.classList.contains("stop-active-seen")) {
          pin.classList.add("stop-active-seen");
          pin.classList.add("stop-just-reached");
        }
        pin.style.opacity = "1";
        pin.style.transform = "scale(1)";
        pin.style.filter = "none";
      } else {
        pin.classList.remove("stop-active-seen");
        pin.classList.remove("stop-just-reached");
        pin.style.opacity = "0.22";
        pin.style.transform = "scale(0.75)";
        pin.style.filter = "grayscale(0.6)";
      }
    });
  }, [s.progress, map]);

  if (positions.length < 2 || cum.length !== positions.length) return null;
  // Extra safety: bail out if any position is invalid
  const safePositions = positions.filter(validLL);
  if (safePositions.length < 2) return null;

  const total = cum[cum.length - 1] || 1;
  const target = s.progress * total;
  let lo = 0;
  let hi = cum.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  const i = Math.max(1, Math.min(lo, positions.length - 1));
  const seg = cum[i] - cum[i - 1] || 1;
  const t = Math.min(1, Math.max(0, (target - cum[i - 1]) / seg));
  let cur: Pt = [
    positions[i - 1][0] + (positions[i][0] - positions[i - 1][0]) * t,
    positions[i - 1][1] + (positions[i][1] - positions[i - 1][1]) * t,
  ];
  // Fallback: if interpolation produced NaN, use last known good position
  if (!validLL(cur)) cur = positions[i - 1] ?? positions[0];
  if (!validLL(cur)) return null;
  const drawn = positions.slice(0, i).concat([cur]);

  return (
    <>
      {/* Dynamic Keyframes for flowing dash stream */}
      <style>{`
        @keyframes flowDashStream {
          from {
            stroke-dashoffset: 52px;
          }
          to {
            stroke-dashoffset: 0px;
          }
        }
        .flow-dash-line {
          animation: flowDashStream 1.1s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .flow-dash-line {
            animation: none !important;
          }
        }
        .no-effects .flow-dash-line {
          animation: none !important;
        }
      `}</style>

      {/* 1. Dark contrast casing border beneath */}
      <Polyline
        positions={drawn}
        pathOptions={{
          color: "#050811",
          weight: 12,
          opacity: 0.85,
          lineCap: "round",
          lineJoin: "round",
        }}
        interactive={false}
      />

      {/* 2. Outer luminous glow halo */}
      <Polyline
        positions={drawn}
        pathOptions={{
          color: activeColor,
          weight: 9,
          opacity: 0.45,
          lineCap: "round",
          lineJoin: "round",
        }}
        interactive={false}
      />

      {/* 3. Solid vibrant path */}
      <Polyline
        positions={drawn}
        pathOptions={{
          color: activeColor,
          weight: 5.5,
          opacity: 1.0,
          lineCap: "round",
          lineJoin: "round",
        }}
        interactive={false}
      />

      {/* 4. Bright white core beam */}
      <Polyline
        positions={drawn}
        pathOptions={{
          color: "#ffffff",
          weight: 2,
          opacity: 0.85,
          lineCap: "round",
          lineJoin: "round",
        }}
        interactive={false}
      />

      {/* 5. Animated directional dash flow overlay */}
      <Polyline
        positions={drawn}
        pathOptions={{
          className: "flow-dash-line",
          color: "#ffffff",
          weight: 2.2,
          opacity: 0.95,
          dashArray: "10 16",
          lineCap: "round",
        }}
        interactive={false}
      />

      {/* Vehicle marker */}
      <Marker position={cur} icon={vehIcon} interactive={false} zIndexOffset={1000} />
    </>
  );
}

export function AnimControls({ bus }: { bus: AnimBus }) {
  const s = useBus(bus);

  // Keyboard shortcut: Spacebar to play/pause
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        if (bus.s.progress >= 1) {
          bus.set({ progress: 0, playing: true, run: bus.s.run + 1 });
        } else {
          bus.set({ playing: !bus.s.playing });
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bus]);

  const toggle = () => {
    if (s.progress >= 1) bus.set({ progress: 0, playing: true, run: s.run + 1 });
    else bus.set({ playing: !s.playing });
  };

  const ico = (d: string) => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d={d} />
    </svg>
  );

  return (
    <div className="map-hud-bottom">
      <button
        className="hud-btn"
        onClick={toggle}
        title={s.playing ? "Tạm dừng (Phím Space)" : "Phát tuyến (Phím Space)"}
        aria-label={s.playing ? "Tạm dừng" : "Phát"}
      >
        {s.playing ? ico("M6 5h4v14H6zM14 5h4v14h-4z") : ico("M8 5v14l11-7z")}
      </button>
      <button
        className="hud-btn"
        title="Phát lại từ đầu"
        aria-label="Phát lại"
        onClick={() => bus.set({ progress: 0, playing: true, run: s.run + 1 })}
      >
        {ico("M12 5V1L7 6l5 5V7a6 6 0 11-6 6H4a8 8 0 108-8z")}
      </button>
      <input
        type="range"
        min={0}
        max={1000}
        value={Math.round(s.progress * 1000)}
        onChange={(e) => bus.set({ progress: Number(e.target.value) / 1000, playing: false })}
        aria-label="Tiến độ chạy tuyến"
        style={{ flex: 1 }}
      />
      <div className="seg hud-seg" style={{ padding: 2 }}>
        {[0.5, 1, 2, 4].map((v) => (
          <button
            key={v}
            className={s.speed === v ? "on" : ""}
            onClick={() => bus.set({ speed: v })}
            style={{ padding: "4px 8px", fontSize: "11.5px" }}
          >
            {v}x
          </button>
        ))}
      </div>
      <span className="hud-pct">{Math.round(s.progress * 100)}%</span>
    </div>
  );
}
