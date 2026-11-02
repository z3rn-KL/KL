import React, { useState, useMemo, useEffect } from "react";
import {
  Play,
  Pause,
  FastForward,
  Calendar,
  MapPin,
  Check,
  Navigation,
  Compass,
  Package,
  Crosshair,
  Truck,
  Building2,
  Clock,
  BrainCircuit,
  Eye,
  Users,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  DollarSign,
  TrendingDown,
} from "lucide-react";
import L from "leaflet";
import { Marker, Polyline, Popup, useMap } from "react-leaflet";
import { MapBase, depotIcon } from "@/components/MapBase";
import {
  DispatchSessionState,
  ShipperData,
  DispatchOrderItem,
} from "@/lib/api";
import {
  useShipperVehicleAnimation,
  ShipperPose,
} from "./useShipperVehicleAnimation";
import { createVehicleMarkerIcon } from "./vehicleIcon";
import { RlDecisionExplorerModal } from "./RlDecisionExplorerModal";

// Custom Order Stop Marker Pin
function createOrderPinIcon(
  order: DispatchOrderItem | undefined,
  stopSequence?: number | string,
  isDelivered?: boolean,
  isDepotPickup?: boolean,
  isNext?: boolean
) {
  if (isDepotPickup) {
    const html = `
      <div style="background-color: #0284c7; border: 2.2px solid #ffffff; width: 26px; height: 26px; border-radius: 50%; box-shadow: 0 3px 8px rgba(2,132,199,0.5); display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 800;">
        📦
      </div>
    `;
    return L.divIcon({
      html,
      className: "custom-order-pin depot-pickup-pin",
      iconSize: [26, 26],
      iconAnchor: [13, 13],
    });
  }

  const bg = isDelivered ? "#10b981" : isNext ? "#f59e0b" : "#64748b";
  const size = isNext ? 26 : 22;
  const shadow = isNext ? "0 3px 10px rgba(245,158,11,0.5)" : "0 2px 5px rgba(0,0,0,0.25)";

  const html = `
    <div style="background-color: ${bg}; border: 2px solid white; width: ${size}px; height: ${size}px; border-radius: 50%; box-shadow: ${shadow}; display: flex; align-items: center; justify-content: center; color: white; font-size: ${isNext ? "11px" : "9.5px"}; font-weight: 800; transition: transform 0.2s ease;">
      ${isDelivered ? "✓" : stopSequence !== undefined ? stopSequence : ""}
    </div>
  `;

  return L.divIcon({
    html,
    className: "custom-order-pin",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

// Staging Base Marker Icon
function createBaseMarkerIcon(baseName: string) {
  const html = `
    <div style="background: #1e293b; border: 1.8px solid #38bdf8; border-radius: 6px; padding: 2px 6px; color: #38bdf8; font-size: 10px; font-weight: 700; display: flex; align-items: center; gap: 3px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); white-space: nowrap;">
      <span>🏠</span>
      <span>${baseName}</span>
    </div>
  `;
  return L.divIcon({
    html,
    className: "staging-base-pin",
    iconSize: [80, 24],
    iconAnchor: [40, 12],
  });
}

// Map Auto Fit Bounds
function MapAutoFit({ fitPoints }: { fitPoints: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (fitPoints && fitPoints.length > 1) {
      try {
        const bounds = L.latLngBounds(fitPoints);
        map.fitBounds(bounds, { padding: [45, 45], maxZoom: 15 });
      } catch {}
    }
  }, [fitPoints, map]);
  return null;
}

// Map Camera Follower for Active Vehicle
function MapAutoFollower({
  center,
  enabled,
}: {
  center: [number, number];
  enabled: boolean;
}) {
  const map = useMap();
  useEffect(() => {
    if (enabled && center && center[0] && center[1]) {
      map.panTo(center, { animate: true, duration: 0.35 });
    }
  }, [center, enabled, map]);
  return null;
}

export interface LiveOperationsProps {
  session: DispatchSessionState;
  selectedShipperId: string;
  onSelectShipper: (id: string) => void;
  onStepClock: (deltaMinutes?: number) => Promise<void>;
  isPlaying: boolean;
  onTogglePlay: () => void;
  simSpeed: number;
  onChangeSimSpeed: (speed: number) => void;
  onEndDay: () => Promise<void>;
  onSwitchToPlanning: () => void;
  onSwitchToReport: () => void;
  actionLoading: boolean;
  actionError: string | null;
  actionSuccess: string | null;
}

export function LiveOperations({
  session,
  selectedShipperId,
  onSelectShipper,
  onStepClock,
  isPlaying,
  onTogglePlay,
  simSpeed,
  onChangeSimSpeed,
  onEndDay,
  onSwitchToPlanning,
  onSwitchToReport,
  actionLoading,
  actionError,
  actionSuccess,
}: LiveOperationsProps) {
  // Tracking View Mode: "FOCUS" (Selected shipper only) vs "FLEET" (All 16 shippers)
  const [viewMode, setViewMode] = useState<"FOCUS" | "FLEET">("FOCUS");
  const [isSlaPanelOpen, setIsSlaPanelOpen] = useState<boolean>(true);
  const [isRlModalOpen, setIsRlModalOpen] = useState<boolean>(false);

  // Vehicle animation hook
  const {
    presentationClock,
    displayClock,
    poses,
    selectedShipperPose,
    autoFollow,
    setAutoFollow,
  } = useShipperVehicleAnimation({
    shippers: session.shippers,
    simulationClock: session.simulation_clock || 0,
    isPlaying,
    simSpeed,
    depot: session.depot,
    selectedShipperId,
  });

  // Selected shipper data
  const selectedShipper: ShipperData | undefined =
    session.shippers.find((s) => s.shipper_id === selectedShipperId) ||
    session.shippers[0];

  const pad = (n: number | string) => String(n).padStart(2, "0");

  // Selected shipper orders
  const shipperOrders = useMemo(() => {
    const idSet = new Set(selectedShipper?.assigned_order_ids || []);
    return session.orders.filter((o) => idSet.has(o.order_id));
  }, [selectedShipper, session.orders]);

  const shipperDelivered = useMemo(() => {
    return shipperOrders.filter((o) => o.delivery_status === "DELIVERED");
  }, [shipperOrders]);

  const shipperWeight = useMemo(() => {
    return shipperOrders.reduce((sum, o) => sum + (o.weight_kg || 0), 0);
  }, [shipperOrders]);

  const deliveryProgress = shipperOrders.length
    ? Math.round((shipperDelivered.length / shipperOrders.length) * 100)
    : 0;

  // Staging bases map
  const stagingBases = useMemo(() => {
    if (session.staging_bases && session.staging_bases.length > 0) {
      return session.staging_bases.map((b) => ({
        base_id: b.base_id,
        name: b.name,
        lat: b.lat,
        lng: b.lng,
      }));
    }
    const seen = new Set<string>();
    const bases: Array<{ base_id: string; name: string; lat: number; lng: number }> = [];
    session.shippers.forEach((s) => {
      if (s.base_id && s.base_lat && s.base_lng && !seen.has(s.base_id)) {
        seen.add(s.base_id);
        bases.push({
          base_id: s.base_id,
          name: s.base_name || s.base_id,
          lat: s.base_lat,
          lng: s.base_lng,
        });
      }
    });
    return bases;
  }, [session.staging_bases, session.shippers]);

  // Relevant Staging Base for Selected Shipper
  const selectedShipperBase = useMemo(() => {
    if (
      selectedShipper?.shift_start_mode === "START_AT_BASE" &&
      selectedShipper.base_lat &&
      selectedShipper.base_lng
    ) {
      return {
        base_id: selectedShipper.base_id || "base_current",
        name: selectedShipper.base_name || "Trạm cơ sở",
        lat: selectedShipper.base_lat,
        lng: selectedShipper.base_lng,
      };
    }
    return null;
  }, [selectedShipper]);

  // Compute Full Route Segments for Selected Shipper
  const routeSegments = useMemo(() => {
    if (!selectedShipper) return { completed: [], active: [], remaining: [] };

    // Build ordered list of coordinates: Origin -> Stops
    const isModeB = selectedShipper.shift_start_mode === "START_AT_BASE";
    const origin: [number, number] = isModeB && selectedShipper.base_lat && selectedShipper.base_lng
      ? [selectedShipper.base_lat, selectedShipper.base_lng]
      : [session.depot.lat, session.depot.lng];

    const stops = selectedShipper.timeline_stops || [];
    if (stops.length === 0) return { completed: [], active: [], remaining: [] };

    const waypoints: [number, number][] = [origin, ...stops.map((st) => [st.lat, st.lng] as [number, number])];

    const currentIdx = selectedShipperPose?.currentStopIndex ?? 0;
    const completedIdx = Math.max(0, Math.min(currentIdx, stops.length));

    // Completed: from origin through stops already delivered
    const completedLegs: [number, number][] = waypoints.slice(0, completedIdx + 1);

    // Active leg: from previous stop to current moving vehicle position
    const activeLegs: [number, number][] = [];
    if (selectedShipperPose && completedIdx < stops.length) {
      const prevPt = waypoints[completedIdx];
      const currPt: [number, number] = [selectedShipperPose.lat, selectedShipperPose.lng];
      const nextPt = waypoints[completedIdx + 1];
      activeLegs.push(prevPt, currPt);
      if (nextPt) activeLegs.push(nextPt);
    }

    // Remaining: from current stop to the remaining stops
    const remainingLegs: [number, number][] = waypoints.slice(completedIdx);

    return {
      completed: completedLegs.length > 1 ? completedLegs : [],
      active: activeLegs.length > 1 ? activeLegs : [],
      remaining: remainingLegs.length > 1 ? remainingLegs : [],
    };
  }, [selectedShipper, selectedShipperPose, session.depot]);

  // Fit bounds: include depot, shipper bases, current position, and stops
  const fitPoints = useMemo(() => {
    const points: [number, number][] = [[session.depot.lat, session.depot.lng]];
    if (selectedShipper) {
      if (selectedShipper.base_lat && selectedShipper.base_lng) {
        points.push([selectedShipper.base_lat, selectedShipper.base_lng]);
      }
      points.push([selectedShipper.current_lat, selectedShipper.current_lng]);
      if (selectedShipper.timeline_stops) {
        selectedShipper.timeline_stops.forEach((st) => points.push([st.lat, st.lng]));
      }
    }
    return points;
  }, [session.depot, selectedShipper]);

  // Operational SLA & Simulated Cost Calculations
  const operationalMetrics = useMemo(() => {
    const delivered = session.orders.filter((o) => o.delivery_status === "DELIVERED");
    const onTime = delivered.filter((o) => o.sla_status !== "OVERDUE");
    const late = delivered.filter((o) => o.sla_status === "OVERDUE");
    const latenessMins = delivered.reduce((acc, o) => acc + (o.sla_status === "OVERDUE" ? Math.max(0, (o.delivered_at_minute || 0) - (o.deadline_minutes || 0)) : 0), 0);
    const totalDistKm = session.fleet_summary?.total_distance_km || 0;
    const shiftMins = session.shift_minutes || 600;
    const clockMins = session.simulation_clock || 0;
    const overtimeMins = Math.max(0, clockMins - shiftMins);

    // Simulated operational cost models
    const fuelCostVnd = Math.round(totalDistKm * 3200);
    const slaPenaltyVnd = Math.round(late.length * 45000 + latenessMins * 1200);
    const overtimePenaltyVnd = Math.round(overtimeMins * 1500 * (session.fleet_summary?.active_shippers || 1));
    const totalCostVnd = fuelCostVnd + slaPenaltyVnd + overtimePenaltyVnd;

    const slaRate = delivered.length ? Math.round((onTime.length / delivered.length) * 100) : 100;

    return {
      deliveredCount: delivered.length,
      onTimeCount: onTime.length,
      lateCount: late.length,
      latenessMins: Math.round(latenessMins),
      slaRate,
      overtimeMins: Math.round(overtimeMins),
      fuelCostVnd,
      slaPenaltyVnd,
      overtimePenaltyVnd,
      totalCostVnd,
      isOvertime: overtimeMins > 0,
    };
  }, [session.orders, session.fleet_summary, session.shift_minutes, session.simulation_clock]);

  // Activity events from session log
  const activityLogs = useMemo(() => {
    const logs = [];
    const sNum = selectedShipper?.shipper_id.replace("shipper_", "");
    const pose = selectedShipperPose;

    if (pose) {
      logs.push(`${displayClock}: Shipper #${sNum} — ${pose.statusText}`);
      if (pose.nextStopName) {
        logs.push(`${displayClock}: Đang hướng về ${pose.nextStopName}`);
      }
    }

    if (selectedShipper?.shift_start_mode === "START_AT_BASE") {
      logs.push(
        `08:00: Shipper #${sNum} xuất phát từ Trạm ${selectedShipper.base_name || "khu vực"} → Kho trung tâm (DEPOT_PICKUP)`
      );
    } else {
      logs.push(
        `08:00: ${session.shippers.length} Shipper xuất phát từ ${session.depot.name}.`
      );
    }

    return logs;
  }, [displayClock, selectedShipper, selectedShipperPose, session.depot.name, session.shippers.length]);

  // Overtime and Shift Status Evaluation
  const shiftMins = session.shift_minutes || 600;
  const maxOvertimeMins = session.max_overtime_minutes || 48;
  const clockMins = session.simulation_clock || 0;
  const isOvertime = clockMins > shiftMins;
  const isMaxOvertimeReached = clockMins >= shiftMins + maxOvertimeMins;
  const allDelivered = session.order_summary.assigned === 0 && session.order_summary.delivered > 0;

  const currentPose = selectedShipperPose;

  return (
    <div id="view-live" className="view">
      {/* Topline Metrics */}
      <div className="topline">
        <div className="compact-metrics">
          <div className="metric">
            <b id="liveShippers">{session.fleet_summary?.active_shippers || session.shippers.length}</b>
            <span>Shipper</span>
          </div>
          <div className="metric">
            <strong id="liveDone">{session.order_summary.delivered}</strong>
            <span>Đã giao</span>
          </div>
          <div className="metric">
            <b id="liveToDo">
              {session.order_summary.total - session.order_summary.delivered}
            </b>
            <span>Chưa hoàn tất</span>
          </div>
          <div className="metric">
            <span
              style={{
                fontFamily: "monospace",
                fontWeight: 800,
                color: isOvertime ? "#f59e0b" : "#0d9488",
                fontSize: "15px",
              }}
            >
              {displayClock}
            </span>
            <span>
              {isOvertime ? `Tăng ca (+${Math.round(clockMins - shiftMins)}p)` : "Đồng hồ thực địa"}
            </span>
          </div>
        </div>

        {/* View Mode Toggle: Focus vs Fleet */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div
            style={{
              display: "inline-flex",
              borderRadius: "8px",
              border: "1px solid rgba(0,0,0,0.15)",
              background: "rgba(0,0,0,0.04)",
              padding: "2px",
            }}
          >
            <button
              type="button"
              className={`btn sm ${viewMode === "FOCUS" ? "primary" : ""}`}
              style={{
                fontSize: "11px",
                padding: "4px 8px",
                height: "26px",
                borderRadius: "6px",
                fontWeight: 600,
              }}
              onClick={() => setViewMode("FOCUS")}
              title="Chế độ tập trung: Chỉ hiển thị xe và tuyến của shipper được chọn, ẩn các xe khác để chống rối bản đồ"
            >
              <Eye className="size-3" style={{ marginRight: "4px" }} />
              Theo dõi shipper
            </button>
            <button
              type="button"
              className={`btn sm ${viewMode === "FLEET" ? "primary" : ""}`}
              style={{
                fontSize: "11px",
                padding: "4px 8px",
                height: "26px",
                borderRadius: "6px",
                fontWeight: 600,
              }}
              onClick={() => setViewMode("FLEET")}
              title="Chế độ toàn đội: Hiển thị toàn bộ 16 xe trên bản đồ GIS"
            >
              <Users className="size-3" style={{ marginRight: "4px" }} />
              Toàn đội (16 xe)
            </button>
          </div>

          <button
            type="button"
            className="btn sm"
            style={{
              fontSize: "11px",
              height: "28px",
              display: "flex",
              alignItems: "center",
              gap: "4px",
              borderColor: "#0d9488",
              color: "#0f766e",
            }}
            onClick={() => setIsRlModalOpen(true)}
            title="Mở bảng phân tích vết quyết định RL (Q-scores & State features)"
          >
            <BrainCircuit className="size-3.5" />
            <span>Khám phá RL</span>
          </button>
        </div>
      </div>

      {/* Live Operations Layout */}
      <div className="live-layout">
        {/* GIS Map Frame */}
        <div className="mapframe" id="map">
          <MapBase height="100%" fit={fitPoints}>
            {!autoFollow && <MapAutoFit fitPoints={fitPoints} />}
            <MapAutoFollower
              center={currentPose ? [currentPose.lat, currentPose.lng] : [session.depot.lat, session.depot.lng]}
              enabled={autoFollow}
            />

            {/* Central Depot Marker */}
            <Marker position={[session.depot.lat, session.depot.lng]} icon={depotIcon()}>
              <Popup>
                <div style={{ fontSize: "12px", padding: "4px" }}>
                  <strong>{session.depot.name}</strong>
                  <div style={{ color: "#71818a" }}>Kho trung tâm điều phối NEXUS</div>
                  <div style={{ fontSize: "11px", color: "#0d9488", marginTop: "3px" }}>
                    Tọa độ: {session.depot.lat.toFixed(4)}, {session.depot.lng.toFixed(4)}
                  </div>
                </div>
              </Popup>
            </Marker>

            {/* Staging Bases Markers: Render all in FLEET mode, or only selected shipper's base in FOCUS mode */}
            {viewMode === "FLEET"
              ? stagingBases.map((b) => (
                  <Marker
                    key={b.base_id}
                    position={[b.lat, b.lng]}
                    icon={createBaseMarkerIcon(b.name)}
                  >
                    <Popup>
                      <div style={{ fontSize: "12px", padding: "4px" }}>
                        <strong>🏠 {b.name}</strong>
                        <div style={{ color: "#0284c7" }}>Trạm xuất phát cơ sở (Base)</div>
                      </div>
                    </Popup>
                  </Marker>
                ))
              : selectedShipperBase && (
                  <Marker
                    key={selectedShipperBase.base_id}
                    position={[selectedShipperBase.lat, selectedShipperBase.lng]}
                    icon={createBaseMarkerIcon(selectedShipperBase.name)}
                  >
                    <Popup>
                      <div style={{ fontSize: "12px", padding: "4px" }}>
                        <strong>🏠 {selectedShipperBase.name}</strong>
                        <div style={{ color: "#0284c7" }}>
                          Trạm xuất phát của Shipper #{selectedShipper?.shipper_id.replace("shipper_", "")}
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                )}

            {/* Complete Route Visualization Layers for Selected Shipper */}
            {/* 1. Completed Legs (Solid Green) */}
            {routeSegments.completed.length > 1 && (
              <Polyline
                positions={routeSegments.completed}
                pathOptions={{
                  color: "#10b981",
                  weight: 4,
                  opacity: 0.8,
                }}
              />
            )}

            {/* 2. Active Leg (Vibrant Cyan / Amber to Current Vehicle) */}
            {routeSegments.active.length > 1 && (
              <Polyline
                positions={routeSegments.active}
                pathOptions={{
                  color: "#06b6d4",
                  weight: 5,
                  opacity: 0.95,
                }}
              />
            )}

            {/* 3. Remaining Legs (Dashed Teal) */}
            {routeSegments.remaining.length > 1 && (
              <Polyline
                positions={routeSegments.remaining}
                pathOptions={{
                  color: "#0d9488",
                  weight: 3.5,
                  opacity: 0.75,
                  dashArray: "6, 8",
                }}
              />
            )}

            {/* Timeline Stops for Selected Shipper */}
            {selectedShipper?.timeline_stops?.map((stop, idx) => {
              const isDepotPickup = (stop as any).is_depot_pickup || stop.order_id === "DEPOT_PICKUP";
              const ord = !isDepotPickup
                ? session.orders.find((o) => o.order_id === stop.order_id)
                : undefined;
              const isDelivered = ord?.delivery_status === "DELIVERED";
              const isNext = currentPose?.currentStopIndex === idx;

              return (
                <Marker
                  key={stop.order_id + "_" + idx}
                  position={[stop.lat, stop.lng]}
                  icon={createOrderPinIcon(ord, idx + 1, isDelivered, isDepotPickup, isNext)}
                >
                  <Popup>
                    <div style={{ fontSize: "12px", padding: "4px" }}>
                      {isDepotPickup ? (
                        <>
                          <strong style={{ color: "#0284c7" }}>📦 Lấy hàng tại Depot</strong>
                          <div>Thời gian dự kiến: {stop.arrival_minute}m - {stop.departure_minute}m</div>
                          <div style={{ color: "#64748b" }}>Nhận đơn hàng trước khi đi giao</div>
                        </>
                      ) : (
                        <>
                          <strong>Điểm dừng #{idx + 1}</strong>
                          <div style={{ color: "#0f766e", fontWeight: 700 }}>
                            {ord?.order_code || stop.order_id}
                          </div>
                          <div>Khối lượng: {ord?.weight_kg?.toFixed(1) || 0} kg</div>
                          <div>Trạng thái: {isDelivered ? "Đã giao thành công" : "Đang chờ giao"}</div>
                          <div>Dự kiến đến: phút {stop.arrival_minute} (ca làm việc)</div>
                        </>
                      )}
                    </div>
                  </Popup>
                </Marker>
              );
            })}

            {/* Shippers Animated Position Markers: Filtered by viewMode */}
            {session.shippers
              .filter((s) => viewMode === "FLEET" || s.shipper_id === selectedShipperId)
              .map((s) => {
                const isSelected = s.shipper_id === selectedShipperId;
                const pose: ShipperPose | undefined = poses[s.shipper_id];
                const lat = pose ? pose.lat : s.current_lat;
                const lng = pose ? pose.lng : s.current_lng;
                const heading = pose ? pose.heading : 0;
                const status = pose ? pose.status : s.status;

                return (
                  <Marker
                    key={s.shipper_id}
                    position={[lat, lng]}
                    icon={createVehicleMarkerIcon(s.shipper_id, heading, status, isSelected)}
                    eventHandlers={{
                      click: () => onSelectShipper(s.shipper_id),
                    }}
                  >
                    <Popup>
                      <div style={{ fontSize: "12px", padding: "4px" }}>
                        <strong>Shipper #{s.shipper_id.replace("shipper_", "")}</strong>
                        <div style={{ color: "#0f766e", fontWeight: 600 }}>
                          {pose?.statusText || s.status}
                        </div>
                        <div style={{ fontSize: "11px", color: "#64748b" }}>
                          Hướng di chuyển: {Math.round(heading)}° · Tải: {s.current_load_kg.toFixed(1)} / {s.capacity_kg} kg
                        </div>
                        <div style={{ fontSize: "11px", color: "#64748b" }}>
                          Số đơn nhận: {s.assigned_order_ids.length}
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                );
              })}
          </MapBase>

          {/* Map Overlay Badges */}
          <div className="map-legend">
            <Navigation className="icons smico" style={{ color: "#0d9488" }} />
            <strong>GIS Tracking</strong>
            <span style={{ color: "#91a0a6" }}>/</span>
            <span id="mapMode">
              {viewMode === "FOCUS" ? "Theo dõi tập trung" : "Toàn đội (16 xe)"} · Shipper #{selectedShipper?.shipper_id.replace("shipper_", "")}
            </span>
          </div>

          <div className="map-tile-label">
            OpenFreeMap Positron · Mạng đường TP.HCM · Tọa độ thời gian thực
          </div>

          {/* Camera Follow Toggle */}
          <div
            style={{
              position: "absolute",
              top: "14px",
              right: "14px",
              zIndex: 1000,
            }}
          >
            <button
              type="button"
              className={`btn sm ${autoFollow ? "primary" : ""}`}
              onClick={() => setAutoFollow(!autoFollow)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "5px",
                background: autoFollow ? "#0d9488" : "rgba(255, 255, 255, 0.9)",
                color: autoFollow ? "#ffffff" : "#1e293b",
                border: "1px solid rgba(0,0,0,0.15)",
                borderRadius: "6px",
                padding: "6px 10px",
                fontSize: "11px",
                fontWeight: 600,
                boxShadow: "0 2px 6px rgba(0,0,0,0.15)",
                cursor: "pointer",
              }}
              title="Tự động bám theo vị trí xe di chuyển trên bản đồ"
            >
              <Crosshair size={13} />
              <span>{autoFollow ? "Bám theo xe: BẬT" : "Bám theo xe"}</span>
            </button>
          </div>

          {/* Collapsible Floating SLA & Cost Breakdown Panel */}
          <div
            style={{
              position: "absolute",
              bottom: "78px",
              left: "14px",
              zIndex: 1000,
              maxWidth: "280px",
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(4px)",
              border: "1px solid rgba(0,0,0,0.15)",
              borderRadius: "10px",
              padding: "10px 12px",
              boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
              fontSize: "11px",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                cursor: "pointer",
                fontWeight: 700,
                color: "#0f172a",
              }}
              onClick={() => setIsSlaPanelOpen(!isSlaPanelOpen)}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                <DollarSign size={13} style={{ color: "#0d9488" }} />
                <span>Chi phí & Vi phạm SLA</span>
              </div>
              {isSlaPanelOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </div>

            {isSlaPanelOpen && (
              <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "5px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#64748b" }}>Đơn đúng hạn:</span>
                  <strong style={{ color: "#10b981" }}>
                    {operationalMetrics.onTimeCount} / {operationalMetrics.deliveredCount} ({operationalMetrics.slaRate}%)
                  </strong>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#64748b" }}>Đơn trễ hạn:</span>
                  <strong style={{ color: operationalMetrics.lateCount > 0 ? "#f43f5e" : "#64748b" }}>
                    {operationalMetrics.lateCount} đơn ({operationalMetrics.latenessMins}p)
                  </strong>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#64748b" }}>Phạt vi phạm SLA:</span>
                  <span style={{ fontWeight: 600, fontFamily: "monospace" }}>
                    {operationalMetrics.slaPenaltyVnd.toLocaleString("vi-VN")} đ
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#64748b" }}>Phạt tăng ca:</span>
                  <span style={{ fontWeight: 600, fontFamily: "monospace", color: operationalMetrics.isOvertime ? "#d97706" : undefined }}>
                    {operationalMetrics.overtimePenaltyVnd.toLocaleString("vi-VN")} đ
                  </span>
                </div>

                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    borderTop: "1px solid rgba(0,0,0,0.1)",
                    paddingTop: "5px",
                    marginTop: "2px",
                    fontWeight: 700,
                  }}
                >
                  <span>Tổng chi phí vận hành:</span>
                  <span style={{ color: "#0d9488", fontFamily: "monospace" }}>
                    {operationalMetrics.totalCostVnd.toLocaleString("vi-VN")} đ
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Tracking Aside */}
        <aside className="box tracking">
          <div className="tracking-top">
            <div className="tracking-row">
              <div>
                <span className="captioncaps">FLEET TRACKING · THỜI GIAN THỰC</span>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" }}>
                  <h2 id="trackingShipper" style={{ margin: 0 }}>
                    Shipper #{selectedShipper?.shipper_id.replace("shipper_", "")}
                  </h2>
                  <span
                    className="chip"
                    style={{
                      background:
                        selectedShipper?.shift_start_mode === "START_AT_BASE"
                          ? "#e0f2fe"
                          : "#f0fdf4",
                      color:
                        selectedShipper?.shift_start_mode === "START_AT_BASE"
                          ? "#0369a1"
                          : "#15803d",
                      border: "1px solid currentColor",
                      fontSize: "10px",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      fontWeight: 700,
                    }}
                  >
                    {selectedShipper?.shift_start_mode === "START_AT_BASE"
                      ? `Mode B · ${selectedShipper.base_name || "Trạm cơ sở"}`
                      : "Mode A · Kho trung tâm"}
                  </span>
                </div>
              </div>

              <select
                id="trackingPick"
                className="select"
                aria-label="Chọn shipper"
                value={selectedShipperId}
                onChange={(e) => onSelectShipper(e.target.value)}
              >
                {session.shippers.map((s) => {
                  const sNum = s.shipper_id.replace("shipper_", "");
                  const isB = s.shift_start_mode === "START_AT_BASE";
                  return (
                    <option key={s.shipper_id} value={s.shipper_id}>
                      Shipper #{pad(sNum)} {isB ? `(Trạm ${s.base_name || ""})` : "(Depot)"} · {s.assigned_order_ids.length} đơn
                    </option>
                  );
                })}
              </select>
            </div>

            {/* Live Motion Status Card */}
            <div
              style={{
                marginTop: "12px",
                padding: "8px 12px",
                background: "rgba(13, 148, 136, 0.08)",
                border: "1px solid rgba(13, 148, 136, 0.2)",
                borderRadius: "8px",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: "11px", fontWeight: 700, color: "#0f766e" }}>
                  TRẠNG THÁI CHUYỂN ĐỘNG
                </span>
                <span
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                    fontSize: "11px",
                    fontWeight: 600,
                    color: "#475569",
                  }}
                >
                  <Compass size={13} style={{ transform: `rotate(${currentPose?.heading || 0}deg)` }} />
                  {Math.round(currentPose?.heading || 0)}°
                </span>
              </div>
              <div style={{ fontSize: "13px", fontWeight: 700, color: "#0f172a" }}>
                {currentPose?.statusText || "Chờ lệnh xuất phát"}
              </div>
              {currentPose?.nextStopName && (
                <div style={{ fontSize: "11px", color: "#64748b" }}>
                  Mục tiêu tiếp theo: <strong>{currentPose.nextStopName}</strong>
                </div>
              )}
            </div>

            <div className="track-stats" style={{ marginTop: "12px" }}>
              <div>
                <b id="trackOrders">{shipperOrders.length}</b>
                <span>Đơn nhận</span>
              </div>
              <div>
                <b id="trackDelivered">{shipperDelivered.length}</b>
                <span>Đã giao</span>
              </div>
              <div>
                <b id="trackCapacity">{shipperWeight.toFixed(1)} kg</b>
                <span>Tải trọng</span>
              </div>
            </div>

            <div className="progress" style={{ marginTop: "13px" }}>
              <span id="trackProgress" style={{ width: `${deliveryProgress}%` }} />
            </div>
          </div>

          {/* Stops List */}
          <div className="list-head">
            <h3>Tuyến đang theo dõi</h3>
            <span className="chip" id="routeStops">
              {selectedShipper?.timeline_stops?.length || 0} chặng dừng
            </span>
          </div>

          <div className="route-list" id="routeList">
            {selectedShipper?.timeline_stops && selectedShipper.timeline_stops.length > 0 ? (
              selectedShipper.timeline_stops.map((stop, i) => {
                const isDepotPickup = (stop as any).is_depot_pickup || stop.order_id === "DEPOT_PICKUP";
                const ord = !isDepotPickup
                  ? session.orders.find((o) => o.order_id === stop.order_id)
                  : undefined;
                const isDelivered = ord?.delivery_status === "DELIVERED";
                const isCurrent = currentPose?.currentStopIndex === i;

                return (
                  <div
                    className={`stop ${isCurrent ? "active-stop" : ""}`}
                    key={stop.order_id + "_" + i}
                    style={{
                      borderLeft: isCurrent
                        ? "3px solid #0d9488"
                        : isDepotPickup
                        ? "3px solid #0284c7"
                        : undefined,
                      background: isCurrent ? "rgba(13,148,136,0.06)" : undefined,
                    }}
                  >
                    <div
                      className="stop-num"
                      style={{
                        background: isDepotPickup
                          ? "#0284c7"
                          : isDelivered
                          ? "#10b981"
                          : undefined,
                        color: isDepotPickup || isDelivered ? "#ffffff" : undefined,
                      }}
                    >
                      {isDepotPickup ? "📦" : isDelivered ? <Check className="icons smico" /> : i + 1}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <strong>
                          {isDepotPickup ? "Lấy hàng tại Kho trung tâm" : (ord?.order_code || stop.order_id)}
                        </strong>
                        <span style={{ fontSize: "10px", color: "#64748b" }}>
                          {stop.arrival_minute}m - {stop.departure_minute}m
                        </span>
                      </div>
                      <span>
                        {isDepotPickup
                          ? "DEPOT_PICKUP · Nhận kiện hàng đã phân công"
                          : `${ord?.package_name || "Kiện hàng"} · ${
                              isDelivered ? "Đã giao thành công" : "Chờ giao"
                            }`}
                      </span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="nodata">
                Chưa có tuyến cho shipper này.
                <br />
                Phân công đơn từ Bàn lập kế hoạch.
              </div>
            )}
          </div>

          {/* Recent Activity */}
          <div className="activity">
            <div className="tracking-row">
              <h3>Nhật ký hành trình</h3>
              <span className="captioncaps">THỜI GIAN THỰC</span>
            </div>
            <ul id="activityList">
              {activityLogs.map((log: string, i: number) => (
                <li key={i}>{log}</li>
              ))}
            </ul>
          </div>
        </aside>
      </div>

      {/* Floating 3-Column Simulation Bar */}
      <div className="simbar" style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", alignItems: "center", gap: "16px" }}>
        {/* COLUMN 1: LEFT — SIMULATION STATUS */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div>
            <div className="simtime" id="simulationTime" style={{ fontFamily: "monospace", fontWeight: 800 }}>
              {displayClock}
            </div>
            <div className="sub" id="simulationSub">
              Ngày {pad(session.current_day)} · Ca 08:00–18:00 (10h)
            </div>
          </div>

          <div>
            <span
              className="chip"
              style={{
                background: isPlaying
                  ? "#dcfce7"
                  : isOvertime
                  ? "#fef3c7"
                  : isMaxOvertimeReached
                  ? "#fee2e2"
                  : "#f1f5f9",
                color: isPlaying
                  ? "#15803d"
                  : isOvertime
                  ? "#b45309"
                  : isMaxOvertimeReached
                  ? "#b91c1c"
                  : "#475569",
                border: "1px solid currentColor",
                fontSize: "10px",
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: "6px",
              }}
            >
              {isPlaying
                ? `Đang chạy (${simSpeed}×)`
                : isMaxOvertimeReached
                ? "Hết ca làm việc"
                : isOvertime
                ? `TĂNG CA (+${Math.round(clockMins - shiftMins)}p)`
                : allDelivered
                ? "Hoàn tất ca giao"
                : clockMins > 0
                ? "Đã tạm dừng"
                : "Sẵn sàng"}
            </span>
          </div>
        </div>

        {/* COLUMN 2: CENTER — PLAYBACK CONTROLS */}
        <div className="sim-controls" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {/* Play/Pause: NEVER disabled when isPlaying is true! */}
          <button
            type="button"
            className="btn primary"
            id="playBtn"
            onClick={onTogglePlay}
            disabled={!isPlaying && (actionLoading || session.is_day_ended || (session.order_summary.assigned === 0 && session.order_summary.delivered === 0))}
            style={{
              minWidth: "100px",
              background: isPlaying ? "#475569" : "#0d9488",
            }}
            title={
              isPlaying
                ? "Tạm dừng mô phỏng thực địa"
                : session.is_day_ended
                ? "Ca làm việc đã kết thúc"
                : (session.order_summary.assigned === 0 && session.order_summary.delivered === 0)
                ? "Vui lòng phân công đơn tại Bàn Lập Kế Hoạch trước khi chạy"
                : "Bắt đầu mô phỏng hành trình"
            }
          >
            {isPlaying ? <Pause className="icons" /> : <Play className="icons" />}
            <span>{isPlaying ? "Tạm dừng" : clockMins > 0 ? "Tiếp tục" : "Bắt đầu"}</span>
          </button>

          {/* Step +15m */}
          <button
            type="button"
            className="btn"
            id="stepBtn"
            onClick={() => onStepClock(15)}
            disabled={actionLoading || session.is_day_ended}
            title="Tiến đồng hồ thêm 15 phút"
          >
            <FastForward className="icons" />
            <span>+15 phút</span>
          </button>

          {/* Speed Selectors */}
          <div className="speedset" id="speedChoices">
            {[1, 5, 10, 20].map((s) => (
              <button
                key={s}
                type="button"
                className={`speed ${simSpeed === s ? "active" : ""}`}
                onClick={() => onChangeSimSpeed(s)}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        {/* COLUMN 3: RIGHT — ACTIONS */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "8px" }}>
          <button
            type="button"
            className="btn"
            style={{ fontSize: "11px", height: "34px", borderColor: "#0d9488", color: "#0f766e" }}
            onClick={() => setIsRlModalOpen(true)}
            title="Mở bảng phân tích vết quyết định RL"
          >
            <BrainCircuit size={14} style={{ marginRight: "4px" }} />
            <span>Vết quyết định RL</span>
          </button>

          <button
            type="button"
            className="btn"
            id="endDayBtn"
            style={{
              borderColor: "#b4615c",
              color: "#f9cbc6",
              background: "#503735",
              height: "34px",
              fontSize: "11px",
            }}
            onClick={onEndDay}
            disabled={actionLoading}
            title="Chốt ca hiện tại — các đơn chưa hoàn thành sẽ được chuyển tiếp sang ngày tiếp theo (Carryover Out)"
          >
            <Calendar className="icons" />
            <span>Chốt ca</span>
          </button>
        </div>
      </div>

      {/* RL Decision Explorer Modal */}
      <RlDecisionExplorerModal
        isOpen={isRlModalOpen}
        onClose={() => setIsRlModalOpen(false)}
        sessionId={session.session_id}
        selectedShipperId={selectedShipperId}
      />
    </div>
  );
}
