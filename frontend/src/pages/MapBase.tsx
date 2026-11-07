import { useEffect, useState, type ReactNode } from "react";
import L from "leaflet";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "maplibre-gl/dist/maplibre-gl.css";
import { setWorkerUrl } from "maplibre-gl";
// @ts-ignore
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "@maplibre/maplibre-gl-leaflet";
import { MAP_CENTER, clusterColor } from "../lib/format";
import { useTheme } from "../lib/theme";

// Initialize MapLibre Web Worker for Vite
if (typeof setWorkerUrl === "function" && workerUrl) {
  try {
    setWorkerUrl(workerUrl);
  } catch {}
}

const OSM_URL = "https://tile.openstreetmap.de/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors';

const OPENFREEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron";
const OPENFREEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors, &copy; <a href="https://openfreemap.org" target="_blank" rel="noreferrer">OpenFreeMap</a>';

function MapLibreVectorLayer({ onError }: { onError?: () => void }) {
  const map = useMap();

  useEffect(() => {
    let layer: any = null;
    let isMounted = true;

    async function init() {
      try {
        const res = await fetch(OPENFREEMAP_STYLE);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const style = await res.json();

        // Deduplicate Vietnamese labels in all symbol layers
        if (Array.isArray(style.layers)) {
          style.layers.forEach((l: any) => {
            if (l.type === "symbol" && l.layout && l.layout["text-field"]) {
              const tf = l.layout["text-field"];
              if (Array.isArray(tf) && tf[0] === "case") {
                l.layout["text-field"] = [
                  "coalesce",
                  ["get", "name:vi"],
                  ["get", "name"],
                  ["get", "name:latin"],
                ];
              }
            }
          });
        }

        if (!isMounted) return;

        // @ts-ignore
        if (typeof L.maplibreGL !== "function") {
          throw new Error("L.maplibreGL is not a function");
        }

        // @ts-ignore
        layer = (L as any).maplibreGL({
          style,
          attribution: OPENFREEMAP_ATTRIBUTION,
          interactive: false,
          pane: "tilePane",
        });

        layer.addTo(map);
      } catch (err) {
        console.warn("OpenFreeMap vector style failed, falling back to OSM raster:", err);
        if (isMounted && onError) {
          onError();
        }
      }
    }

    init();

    return () => {
      isMounted = false;
      if (layer && map) {
        try {
          map.removeLayer(layer);
        } catch {}
      }
    };
  }, [map, onError]);

  return null;
}

function MapResizer() {
  const map = useMap();
  useEffect(() => {
    map.invalidateSize();
    const t1 = setTimeout(() => map.invalidateSize(), 100);
    const t2 = setTimeout(() => map.invalidateSize(), 350);
    const t3 = setTimeout(() => map.invalidateSize(), 800);

    const onResize = () => map.invalidateSize();
    window.addEventListener("resize", onResize);

    // ResizeObserver for container size changes (e.g. sidebar toggle, filter panels)
    const container = map.getContainer();
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && container) {
      ro = new ResizeObserver(() => {
        requestAnimationFrame(() => map.invalidateSize());
      });
      ro.observe(container);
    }

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      window.removeEventListener("resize", onResize);
      ro?.disconnect();
    };
  }, [map]);
  return null;
}

function FitBounds({ points }: { points: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (!points || points.length === 0) return;
    const valid = points.filter(
      (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]),
    );
    if (valid.length === 0) return;

    // Check if there are at least 2 distinct coordinates
    const distinct = new Set(valid.map((p) => `${p[0].toFixed(5)},${p[1].toFixed(5)}`));
    map.invalidateSize();
    if (distinct.size >= 2) {
      const b = L.latLngBounds(valid);
      if (b.isValid()) {
        map.fitBounds(b, { padding: [36, 36], maxZoom: 16 });
      }
    } else {
      map.setView(valid[0] ?? MAP_CENTER, 13);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, points.length, points[0]?.[0], points[0]?.[1]]);
  return null;
}

/** Synchronizes cluster colors from nearby circle markers onto stop pins if not set directly */
function MapMarkerOrchestrator() {
  const map = useMap();
  useEffect(() => {
    const updateChipColors = () => {
      const container = map.getContainer();
      const pins = container.querySelectorAll<HTMLElement>(".stop-pin");
      if (!pins.length) return;

      // Extract colors from svg circle markers if available
      const circlePaths = Array.from(container.querySelectorAll<SVGPathElement>(".leaflet-overlay-pane path"));
      pins.forEach((pin) => {
        const chip = pin.querySelector<HTMLElement>(".stop-cluster-chip");
        if (chip && chip.style.display === "none") {
          // If a circle marker with cluster color exists at same location
          const pinRect = pin.getBoundingClientRect();
          for (const p of circlePaths) {
            const stroke = p.getAttribute("stroke");
            if (stroke && stroke !== "#0f766e" && stroke !== "#ffffff" && stroke !== "#000000") {
              const pRect = p.getBoundingClientRect();
              const dx = Math.abs((pRect.left + pRect.width / 2) - (pinRect.left + pinRect.width / 2));
              const dy = Math.abs((pRect.top + pRect.height / 2) - (pinRect.top + pinRect.height / 2));
              if (dx < 35 && dy < 35) {
                chip.style.backgroundColor = stroke;
                chip.style.display = "block";
                break;
              }
            }
          }
        }
      });
    };

    const t = setTimeout(updateChipColors, 300);
    map.on("moveend", updateChipColors);
    return () => {
      clearTimeout(t);
      map.off("moveend", updateChipColors);
    };
  }, [map]);
  return null;
}

function MapControls({
  onResetFit,
}: {
  onResetFit?: () => void;
}) {
  const map = useMap();
  return (
    <div
      style={{
        position: "absolute",
        bottom: 24,
        right: 14,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <button
        type="button"
        className="icon-btn"
        title="Phóng to (Zoom In)"
        onClick={() => map.zoomIn()}
        style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          background: "var(--surface)",
          boxShadow: "var(--shadow-md)",
          fontWeight: 700,
          fontSize: "16px",
        }}
      >
        +
      </button>
      <button
        type="button"
        className="icon-btn"
        title="Thu nhỏ (Zoom Out)"
        onClick={() => map.zoomOut()}
        style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          background: "var(--surface)",
          boxShadow: "var(--shadow-md)",
          fontWeight: 700,
          fontSize: "16px",
        }}
      >
        −
      </button>
      {onResetFit ? (
        <button
          type="button"
          className="icon-btn"
          title="Bao quát toàn bộ điểm"
          onClick={onResetFit}
          style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: "var(--surface)",
            boxShadow: "var(--shadow-md)",
            fontSize: "13px",
          }}
        >
          ⛶
        </button>
      ) : null}
      <button
        type="button"
        className="icon-btn"
        title="Về toạ độ Kho hàng (Depot)"
        onClick={() => map.flyTo(MAP_CENTER, 14, { duration: 0.6 })}
        style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          background: "var(--surface)",
          boxShadow: "var(--shadow-md)",
          fontSize: "13px",
        }}
      >
        🏠
      </button>
    </div>
  );
}

export function MapBase({
  fit,
  children,
  overlay,
  height = "680px",
  hideLayerBadge = false,
  forceRasterFallback = false,
}: {
  fit?: [number, number][];
  children?: ReactNode;
  overlay?: ReactNode;
  height?: string;
  defaultProvider?: string;
  hideLayerBadge?: boolean;
  forceRasterFallback?: boolean;
}) {
  const [theme] = useTheme();
  const [vectorFailed, setVectorFailed] = useState(forceRasterFallback);

  const handleResetFit = () => {
    if (!fit || fit.length === 0) return;
    const valid = fit.filter(
      (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]),
    );
    if (valid.length === 0) return;
    const b = L.latLngBounds(valid);
    if (b.isValid()) {
      // map instance bounds
    }
  };

  return (
    <div className={`map-box ${theme === "dark" ? "dark-tiles" : ""}`} style={{ height, position: "relative" }}>
      {/* Dynamic Keyframes for Depot Pulse and Pins */}
      <style>{`
        .map-box .leaflet-tile-pane {
          will-change: transform;
          transform: translate3d(0, 0, 0);
        }
        .map-box .leaflet-tile {
          outline: 1px solid transparent !important;
          backface-visibility: hidden;
          -webkit-backface-visibility: hidden;
        }
        .map-box .leaflet-tile-container {
          transform: translate3d(0, 0, 0);
          will-change: transform;
        }
        @keyframes depotPulseGlow {
          0%, 100% {
            box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.75), 0 4px 14px rgba(0, 0, 0, 0.45);
            transform: scale(1);
          }
          50% {
            box-shadow: 0 0 0 9px rgba(20, 184, 166, 0), 0 4px 18px rgba(20, 184, 166, 0.35);
            transform: scale(1.05);
          }
        }
        .depot-pulse {
          animation: depotPulseGlow 2.4s ease-in-out infinite;
        }
        @keyframes stopPinPopIn {
          0% {
            transform: scale(0.65);
            opacity: 0.4;
          }
          65% {
            transform: scale(1.18);
            opacity: 1;
          }
          100% {
            transform: scale(1);
            opacity: 1;
          }
        }
        .stop-pin-enhanced {
          font-family: inherit;
          user-select: none;
        }
        .stop-pin-enhanced.stop-just-reached {
          animation: stopPinPopIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }
        @media (prefers-reduced-motion: reduce) {
          .depot-pulse {
            animation: none !important;
          }
          .stop-pin-enhanced.stop-just-reached {
            animation: none !important;
          }
        }
        .no-effects .depot-pulse {
          animation: none !important;
        }
        .no-effects .stop-pin-enhanced.stop-just-reached {
          animation: none !important;
        }
      `}</style>

      <MapContainer
        center={MAP_CENTER}
        zoom={12}
        minZoom={10}
        maxZoom={19}
        preferCanvas
        scrollWheelZoom
        zoomControl={false}
        style={{ height: "100%", width: "100%" }}
      >
        <MapResizer />
        {!vectorFailed ? (
          <MapLibreVectorLayer onError={() => setVectorFailed(true)} />
        ) : (
          <TileLayer
            key={`osm-${theme}`}
            url={OSM_URL}
            attribution={OSM_ATTRIBUTION}
            maxZoom={19}
            referrerPolicy="strict-origin-when-cross-origin"
          />
        )}
        {fit && fit.length > 0 ? <FitBounds points={fit} /> : null}
        <MapControls onResetFit={handleResetFit} />
        <MapMarkerOrchestrator />
        {children}
      </MapContainer>

      {/* Layer info pill */}
      {!hideLayerBadge && (
        <div
          className="map-layer-switch"
          style={{
            position: "absolute",
            top: 14,
            right: 14,
            zIndex: 1000,
            display: "flex",
            alignItems: "center",
            gap: 6,
            background: "color-mix(in srgb, var(--surface) 92%, transparent)",
            backdropFilter: "blur(12px)",
            padding: "5px 12px",
            borderRadius: 11,
            border: "1px solid var(--border)",
            boxShadow: "var(--shadow-md)",
            fontSize: "12px",
            fontWeight: 600,
            color: "var(--text-secondary)",
          }}
        >
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: !vectorFailed ? "var(--good, #22C55E)" : "var(--accent)",
            }}
          />
          <span>{!vectorFailed ? "OpenFreeMap Vector (Positron)" : "Mạng đường thực tế OSM (Raster)"}</span>
        </div>
      )}

      {overlay}
    </div>
  );
}

export const pinIcon = (n: number, color: string, clusterId?: number) => {
  const cColor = clusterId !== undefined && clusterId >= 0 ? clusterColor(clusterId) : null;
  return L.divIcon({
    className: "",
    html: `<div class="stop-pin stop-pin-enhanced" data-stop="${n}" style="--pin:${color}; border: 2.5px solid #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.5), 0 0 0 1.5px rgba(0,0,0,0.25); font-weight: 800; font-size: 13px; width: 30px; height: 30px; display: grid; place-items: center; position: relative; transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.25s ease;">
      <span style="line-height: 1; text-shadow: 0 1px 2px rgba(0,0,0,0.45);">${n}</span>
      <span class="stop-cluster-chip" data-stop-cluster="${n}" style="display: ${cColor ? 'block' : 'none'}; background: ${cColor || 'transparent'}; position: absolute; bottom: -3px; right: -3px; width: 11px; height: 11px; border-radius: 50%; border: 2px solid #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.5);"></span>
    </div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
};

export const depotIcon = () =>
  L.divIcon({
    className: "",
    html: `<div class="depot-pin depot-pulse" title="Kho xuất phát (Depot)">
      <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
        <polyline points="9 22 9 12 15 12 15 22"></polyline>
      </svg>
    </div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });
