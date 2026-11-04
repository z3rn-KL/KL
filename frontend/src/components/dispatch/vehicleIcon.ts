import L from "leaflet";

/**
 * Creates a delivery motorcycle marker with directional heading,
 * rear cargo container, and stationary shipper identifier badge.
 * Designed to fit Mineral Light / Deep Jade enterprise aesthetics.
 */
export function createVehicleMarkerIcon(
  shipperId: string,
  heading: number = 0,
  status: string = "IDLE",
  isSelected: boolean = false,
  isDimmed: boolean = false
) {
  const num = shipperId.replace("shipper_", "");
  const size = isSelected ? 34 : (isDimmed ? 22 : 28);
  const isDelivering = status === "DELIVERING";
  const isPickup = status === "PICKING_UP";

  // Palette: Deep Jade & Emerald with Cyan/Amber accents; dimmed for background fleet
  const primaryBg = isSelected ? "#0f766e" : (isDimmed ? "#475569" : "#0d9488");
  const cargoBg = isPickup ? "#0284c7" : isDelivering ? "#f59e0b" : (isDimmed ? "#64748b" : "#14b8a6");
  const ringColor = isSelected ? "#38bdf8" : (isDimmed ? "#94a3b8" : "#ffffff");
  const shadow = isSelected
    ? "0 3px 10px rgba(15,118,110,0.45)"
    : (isDimmed ? "none" : "0 2px 6px rgba(0,0,0,0.25)");
  const markerOpacity = isDimmed ? 0.45 : 1.0;

  const html = `
    <div style="position: relative; width: ${size}px; height: ${size}px; display: flex; align-items: center; justify-content: center; cursor: pointer; opacity: ${markerOpacity};">
      <!-- Rotatable Vehicle Body (Heading Angle) -->
      <div id="veh-rot-${shipperId}" style="width: 100%; height: 100%; transform: rotate(${heading}deg); transform-origin: center center; transition: transform 0.12s linear; display: flex; align-items: center; justify-content: center;">
        <svg width="${size}" height="${size}" viewBox="0 0 32 32" fill="none" style="filter: drop-shadow(${shadow});">
          <!-- Outer circular base -->
          <circle cx="16" cy="16" r="14" fill="${primaryBg}" stroke="${ringColor}" stroke-width="${isSelected ? '2.2' : '1.5'}" />

          <!-- Forward Heading Arrow Pointer -->
          <polygon points="16,3 19.5,9 12.5,9" fill="${ringColor}" />

          <!-- Motorbike Body & Windshield -->
          <ellipse cx="16" cy="15" rx="3" ry="5.5" fill="#ffffff" opacity="0.95" />

          <!-- Rear Cargo Box -->
          <rect x="12" y="18" width="8" height="6" rx="1.2" fill="${cargoBg}" stroke="#ffffff" stroke-width="0.8" />

          <!-- Handlebars -->
          <line x1="11" y1="12" x2="21" y2="12" stroke="#ffffff" stroke-width="1.4" stroke-linecap="round" />
        </svg>
      </div>

      <!-- Static Shipper Number Badge (Stationary, does not rotate) -->
      <div style="position: absolute; top: -4px; right: -4px; background: ${isSelected ? '#0f172a' : '#1e293b'}; color: #ffffff; border: 1.2px solid #ffffff; border-radius: 9999px; min-width: 16px; height: 16px; padding: 0 3px; display: flex; align-items: center; justify-content: center; font-family: monospace; font-size: 8.5px; font-weight: 800; box-shadow: 0 1px 3px rgba(0,0,0,0.4); z-index: 10;">
        ${num}
      </div>
    </div>
  `;

  return L.divIcon({
    html,
    className: `animated-vehicle-marker-${shipperId}`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}
