import { useEffect, useState, useRef, type ReactNode } from "react";
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

/* ─── Geographic Color Palettes ─── */
function applyGeographicTheme(rawStyle: any, theme: "light" | "dark"): any {
  const s = JSON.parse(JSON.stringify(rawStyle));
  const isDark = theme === "dark";

  // Pearl Geographic (Light) / Midnight Geographic (Dark)
  const C = {
    background:      isDark ? "#182231" : "#F4F5F0",
    residential:     isDark ? "#202D3D" : "#F0F1ED",
    commercial:      isDark ? "#313042" : "#F3EDE8",
    park:            isDark ? "#29463F" : "#D9EBD9",
    wood:            isDark ? "#23483B" : "#C7DFC9",
    water:           isDark ? "#1C5366" : "#C5E7F3",
    waterway:        isDark ? "#1C5366" : "#A2D8EB",
    building:        isDark ? "#1F2C3C" : "#E9EBE5",
    buildingOutline: isDark ? "#141D29" : "#DCDED6",
    motorway:        isDark ? "#806B71" : "#E8B69C",
    motorwayCasing:  isDark ? "#141E2B" : "#D9DCE0",
    majorRoad:       isDark ? "#A58F83" : "#F1D4A8",
    secondaryRoad:   isDark ? "#526474" : "#FFFFFF",
    minorRoad:       isDark ? "#344458" : "#FFFFFF",
    roadCasing:      isDark ? "#141E2B" : "#D9DCE0",
    railway:         isDark ? "#3E4F63" : "#D6D9DE",
    railwayDash:     isDark ? "#253347" : "#FAFAFA",
    pier:            isDark ? "#1E2C3E" : "#F2F3F0",
    boundary:        isDark ? "#4A5C72" : "#B0B5BC",
    labelMajor:      isDark ? "#DCE6EF" : "#37465B",
    labelMinor:      isDark ? "#9FAFC2" : "#708197",
    waterLabel:      isDark ? "#5BA8C8" : "#495E91",
    waterLabelHalo:  isDark ? "rgba(24, 34, 49, 0.7)" : "rgba(255,255,255,0.7)",
    labelHalo:       isDark ? "rgba(24, 34, 49, 0.85)" : "rgba(244, 245, 240, 0.85)",
    labelHaloMinor:  isDark ? "rgba(24, 34, 49, 0.8)" : "rgba(244, 245, 240, 0.8)",
  };

  const newLayers: any[] = [];

  s.layers.forEach((l: any) => {
    const id: string = l.id;
    const p = l.paint || {};

    // ── Geographic fills ──
    if (id === "background") {
      p["background-color"] = C.background;
    } else if (id === "park") {
      p["fill-color"] = C.park;
    } else if (id === "water") {
      p["fill-color"] = C.water;
    } else if (id === "landcover_wood") {
      p["fill-color"] = C.wood;
      p["fill-opacity"] = 0.85;
    } else if (id === "landuse_residential") {
      p["fill-color"] = C.residential;
      p["fill-opacity"] = 0.75;
    } else if (id === "waterway") {
      p["line-color"] = C.waterway;
    } else if (id === "building") {
      p["fill-color"] = C.building;
      p["fill-outline-color"] = C.buildingOutline;

    // ── Motorway / Highway casings ──
    } else if (
      id === "tunnel_motorway_casing" ||
      id === "highway_motorway_casing" ||
      id === "highway_motorway_bridge_casing"
    ) {
      p["line-color"] = C.motorwayCasing;
    } else if (
      id === "tunnel_motorway_inner" ||
      id === "highway_motorway_inner" ||
      id === "highway_motorway_bridge_inner" ||
      id === "highway_motorway_subtle"
    ) {
      p["line-color"] = C.motorway;

    // ── Major roads (trunk / primary / secondary / tertiary) ──
    } else if (id === "highway_major_casing") {
      p["line-color"] = C.roadCasing;
    } else if (id === "highway_major_inner" || id === "highway_major_subtle") {
      p["line-color"] = [
        "match", ["get", "class"],
        ["motorway"], C.motorway,
        ["trunk", "primary"], C.majorRoad,
        ["secondary", "tertiary"], C.secondaryRoad,
        C.secondaryRoad,
      ];

    // ── Minor roads & paths ──
    } else if (
      id === "highway_minor" || id === "highway_path" ||
      id === "aeroway-taxiway" || id === "aeroway-runway"
    ) {
      p["line-color"] = C.minorRoad;
    } else if (id === "aeroway-area" || id === "road_area_pier") {
      p["fill-color"] = C.pier;
    } else if (id === "road_pier") {
      p["line-color"] = C.pier;

    // ── Railway ──
    } else if (id.startsWith("railway") && id.includes("dashline")) {
      p["line-color"] = C.railwayDash;
    } else if (id.startsWith("railway")) {
      p["line-color"] = C.railway;

    // ── Boundaries ──
    } else if (id.startsWith("boundary")) {
      p["line-color"] = C.boundary;

    // ── Symbol / Label layers ──
    } else if (l.type === "symbol") {
      // Deduplicate Vietnamese labels
      if (l.layout && l.layout["text-field"]) {
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

      // Water labels
      if (id.startsWith("water_name") || id === "waterway_line_label") {
        p["text-color"] = C.waterLabel;
        p["text-halo-color"] = C.waterLabelHalo;
        p["text-halo-width"] = 1.5;
      } else {
        // Place labels: major vs minor
        const isMajor = [
          "label_city", "label_city_capital", "label_town", "label_state",
          "label_country_1", "label_country_2", "label_country_3",
        ].includes(id);

        p["text-color"] = isMajor ? C.labelMajor : C.labelMinor;
        p["text-halo-color"] = isMajor ? C.labelHalo : C.labelHaloMinor;
        p["text-halo-width"] = isMajor ? 1.5 : 1.2;
        if (p["text-halo-blur"] !== undefined) p["text-halo-blur"] = 0.5;
      }
    }

    l.paint = p;
    newLayers.push(l);

    // Inject commercial/retail/industrial landuse after residential
    if (id === "landuse_residential") {
      newLayers.push({
        id: "landuse_commercial",
        type: "fill",
        source: l.source,
        "source-layer": l["source-layer"],
        filter: ["match", ["get", "class"], ["commercial", "retail", "industrial"], true, false],
        paint: { "fill-color": C.commercial, "fill-opacity": 0.75 },
      });
    }
  });

  s.layers = newLayers;
  return s;
}

// Cache the fetched base style so theme switches don't re-fetch
let _baseStyleCache: any = null;
async function fetchBaseStyle(): Promise<any> {
  if (_baseStyleCache) return _baseStyleCache;
  const res = await fetch(OPENFREEMAP_STYLE);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  _baseStyleCache = await res.json();
  return _baseStyleCache;
}

function MapLibreVectorLayer({ onError, theme }: { onError?: () => void; theme: "light" | "dark" }) {
  const map = useMap();
  const layerRef = useRef<any>(null);
  const glMapRef = useRef<any>(null);
  const baseStyleRef = useRef<any>(null);

  // Stable callback
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  // Initial mount: create the layer
  useEffect(() => {
    let isMounted = true;

    async function init() {
      try {
        const baseStyle = await fetchBaseStyle();
        if (!isMounted) return;

        baseStyleRef.current = baseStyle;

        // @ts-ignore
        if (typeof L.maplibreGL !== "function") {
          throw new Error("L.maplibreGL is not a function");
        }

        const themedStyle = applyGeographicTheme(baseStyle, theme);

        // @ts-ignore
        const layer = (L as any).maplibreGL({
          style: themedStyle,
          attribution: OPENFREEMAP_ATTRIBUTION,
          interactive: false,
          pane: "tilePane",
        });

        layer.addTo(map);
        layerRef.current = layer;

        // Wait for MapLibre map to initialize, then store reference
        const checkGl = () => {
          if (!isMounted) return;
          try {
            const glMap = layer.getMaplibreMap();
            if (glMap) {
              glMapRef.current = glMap;
            }
          } catch {}
        };
        tCheck1 = setTimeout(checkGl, 200);
        tCheck2 = setTimeout(checkGl, 800);
      } catch (err) {
        console.warn("OpenFreeMap vector style failed, falling back to OSM raster:", err);
        if (isMounted && onErrorRef.current) {
          onErrorRef.current();
        }
      }
    }

    let tCheck1: ReturnType<typeof setTimeout> | undefined;
    let tCheck2: ReturnType<typeof setTimeout> | undefined;

    init();

    return () => {
      isMounted = false;
      if (tCheck1) clearTimeout(tCheck1);
      if (tCheck2) clearTimeout(tCheck2);
      if (layerRef.current && map && isMapAlive(map)) {
        try {
          map.removeLayer(layerRef.current);
        } catch {}
      }
      layerRef.current = null;
      glMapRef.current = null;
    };
    // Only run on mount/unmount — theme changes handled by separate effect
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map]);

  // Theme change: update style in-place via MapLibre's setStyle
  useEffect(() => {
    const glMap = glMapRef.current;
    const baseStyle = baseStyleRef.current;
    if (!glMap || !baseStyle) return;

    try {
      const themedStyle = applyGeographicTheme(baseStyle, theme);
      glMap.setStyle(themedStyle);
    } catch (err) {
      console.warn("Failed to switch geographic theme:", err);
    }
  }, [theme]);

  return null;
}

/** Checks if a Leaflet Map instance is still active, loaded, and attached to the DOM */
function isMapAlive(map: L.Map | null | undefined): boolean {
  if (!map) return false;
  try {
    const container = map.getContainer();
    if (!container || !container.parentElement) return false;
    // Leaflet removes _mapPane on map.remove()
    if (!(map as any)._mapPane) return false;
    return true;
  } catch {
    return false;
  }
}

/** Safely calls map.invalidateSize() only when the map instance and DOM pane are alive */
function safeInvalidateMap(map: L.Map | null | undefined, options?: boolean | L.ZoomPanOptions) {
  if (!isMapAlive(map)) return;
  try {
    map!.invalidateSize(options as any);
  } catch {
    // Gracefully ignore during unmount / StrictMode cleanup
  }
}

function MapResizer() {
  const map = useMap();
  useEffect(() => {
    let isMounted = true;
    let rafId: number | null = null;

    const safeInvalidate = () => {
      if (!isMounted) return;
      safeInvalidateMap(map);
    };

    safeInvalidate();
    const t1 = setTimeout(safeInvalidate, 100);
    const t2 = setTimeout(safeInvalidate, 350);
    const t3 = setTimeout(safeInvalidate, 800);

    const onResize = () => safeInvalidate();
    window.addEventListener("resize", onResize);

    // ResizeObserver for container size changes (e.g. sidebar toggle, filter panels)
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      try {
        const container = map.getContainer();
        if (container) {
          ro = new ResizeObserver(() => {
            if (!isMounted) return;
            if (rafId !== null) {
              cancelAnimationFrame(rafId);
            }
            rafId = requestAnimationFrame(() => {
              rafId = null;
              safeInvalidate();
            });
          });
          ro.observe(container);
        }
      } catch {}
    }

    return () => {
      isMounted = false;
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      window.removeEventListener("resize", onResize);
      if (ro) {
        ro.disconnect();
        ro = null;
      }
    };
  }, [map]);
  return null;
}

function FitBounds({ points }: { points: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (!points || points.length === 0) return;
    if (!isMapAlive(map)) return;
    const valid = points.filter(
      (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]),
    );
    if (valid.length === 0) return;

    // Check if there are at least 2 distinct coordinates
    const distinct = new Set(valid.map((p) => `${p[0].toFixed(5)},${p[1].toFixed(5)}`));
    safeInvalidateMap(map);
    try {
      if (distinct.size >= 2) {
        const b = L.latLngBounds(valid);
        if (b.isValid() && isMapAlive(map)) {
          map.fitBounds(b, { padding: [36, 36], maxZoom: 16 });
        }
      } else if (isMapAlive(map)) {
        map.setView(valid[0] ?? MAP_CENTER, 13);
      }
    } catch {
      // Ignored during unmount / StrictMode transitions
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, points.length, points[0]?.[0], points[0]?.[1]]);
  return null;
}

/** Synchronizes cluster colors from nearby circle markers onto stop pins if not set directly */
function MapMarkerOrchestrator() {
  const map = useMap();
  useEffect(() => {
    let isMounted = true;
    const updateChipColors = () => {
      if (!isMounted || !isMapAlive(map)) return;
      let container: HTMLElement | null = null;
      try {
        container = map.getContainer();
      } catch {
        return;
      }
      if (!container) return;
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
      isMounted = false;
      clearTimeout(t);
      try {
        map.off("moveend", updateChipColors);
      } catch {}
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
        onClick={() => {
          if (isMapAlive(map)) map.zoomIn();
        }}
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
        onClick={() => {
          if (isMapAlive(map)) map.zoomOut();
        }}
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
        onClick={() => {
          if (isMapAlive(map)) map.flyTo(MAP_CENTER, 14, { duration: 0.6 });
        }}
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
    <div className={`map-box ${theme === "dark" ? "dark-tiles" : ""} ${vectorFailed ? "raster-fallback" : ""}`} style={{ height, position: "relative", isolation: "isolate" }}>
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
          <MapLibreVectorLayer onError={() => setVectorFailed(true)} theme={theme as "light" | "dark"} />
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
