import React, { useState, useMemo, useEffect } from "react";
import {
  Package,
  Truck,
  Search,
  Check,
  RefreshCw,
  Play,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Eye,
  CheckCheck,
  X,
  Layers,
  Sparkles,
  Compass,
  MapPin,
  Clock,
  Weight,
  Cpu,
  ChevronDown,
  ChevronUp,
  Info,
  Users,
  Sliders,
  ShieldCheck,
  Warehouse,
  Zap,
  RotateCcw,
  FileText,
} from "lucide-react";
import L from "leaflet";
import { Marker, Polyline, Popup, CircleMarker, useMap } from "react-leaflet";
import { MapBase, depotIcon } from "@/components/MapBase";
import { RlDecisionExplorerModal } from "./RlDecisionExplorerModal";
import { DepotServiceAreaModal } from "./DepotServiceAreaModal";
import { NewScenarioModal } from "./NewScenarioModal";
import { BrainCircuit } from "lucide-react";
import { clusterColor, MAP_CENTER } from "@/lib/format";
import {
  DispatchSessionState,
  ShipperData,
  DispatchOrderItem,
  NearbyOrderItem,
  PlanPreviewResponse,
  StagingBase,
} from "@/lib/api";

interface PlanningDeskProps {
  session: DispatchSessionState;
  selectedShipperId: string;
  onSelectShipper: (id: string) => void;
  nearbyOrders: NearbyOrderItem[];
  onAssign: (shipperId: string, orderIds: string[]) => Promise<void>;
  onUnassign: (orderIds: string[]) => Promise<void>;
  onAutoDispatch: (strategy?: string, algorithm?: string) => Promise<void>;
  onPreviewPlan?: (strategy: string, algorithm: string) => Promise<PlanPreviewResponse>;
  onConfirmPreview?: (previewId: string) => Promise<void>;
  onCancelPreview?: () => Promise<void>;
  onSwitchToLive: () => void;
  onSwitchToReport?: () => void;
  onCreateNewSession?: (totalDays: number) => Promise<void>;
  onNextDay?: () => Promise<void>;
  onSelectAlgorithm: (algo: string) => Promise<void>;
  actionLoading: boolean;
  actionError: string | null;
  actionSuccess: string | null;
  onReloadSession?: () => Promise<void>;
  previewPlan?: PlanPreviewResponse | null;
  onSetPreviewPlan?: (plan: PlanPreviewResponse | null) => void;
}

const PAGE_SIZE = 12;

function createShipperIcon(shipperId: string, isSelected: boolean) {
  const num = shipperId.replace("shipper_", "");
  const bg = isSelected ? "#0f766e" : "#0d9488";
  const size = isSelected ? 34 : 26;
  const border = isSelected ? "3px solid #2dd4bf" : "2px solid #ffffff";
  const shadow = isSelected ? "0 4px 12px rgba(15,118,110,0.5)" : "0 2px 6px rgba(0,0,0,0.3)";
  return L.divIcon({
    html: `<div style="background-color: ${bg}; border: ${border}; width: ${size}px; height: ${size}px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: ${isSelected ? "11px" : "10px"}; box-shadow: ${shadow};"><span>${num}</span></div>`,
    className: "desk-shipper-marker",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function createStopPinIcon(seq: number, isPreview = false) {
  const bg = isPreview ? "#06b6d4" : "#f97316";
  const size = 20;
  return L.divIcon({
    html: `<div style="background-color: ${bg}; border: 2px solid white; width: ${size}px; height: ${size}px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 9px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">${seq}</div>`,
    className: "desk-stop-pin",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function createDepotFleetIcon(count: number) {
  return L.divIcon({
    html: `<div style="position: relative; display: flex; align-items: center; justify-content: center;">
      <div style="background: #1e293b; border: 2.5px solid #38bdf8; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.4); color: white;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>
      </div>
      <div style="position: absolute; top: -6px; right: -8px; background: #0284c7; color: white; border: 1.5px solid white; border-radius: 9999px; padding: 1px 5px; font-size: 9px; font-weight: 800; line-height: 1;">
        ${count}
      </div>
    </div>`,
    className: "depot-fleet-marker",
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });
}

function MapController({ focusPoint }: { focusPoint: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (focusPoint) {
      map.flyTo(focusPoint, Math.max(map.getZoom(), 14), { duration: 0.8 });
    }
  }, [focusPoint, map]);
  return null;
}

function createStagingBaseIcon(baseId: string, name: string) {
  const shortCode = baseId.replace("base_", "").toUpperCase();
  return L.divIcon({
    html: `<div style="background-color: #7c3aed; border: 2px solid #ffffff; width: 28px; height: 28px; border-radius: 6px; transform: rotate(45deg); display: flex; align-items: center; justify-content: center; box-shadow: 0 3px 8px rgba(124,58,237,0.45);"><span style="transform: rotate(-45deg); color: white; font-weight: 800; font-size: 10px;">${shortCode}</span></div>`,
    className: "staging-base-marker",
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

const formatStrategyName = (s: string) => {
  switch (s) {
    case "cluster_aware_rl": return "Ưu tiên cụm + chi phí";
    case "location_aware_nearest": return "Shipper gần nhất (Nearest Shipper)";
    case "location_aware_min_cost": return "Tối ưu cự ly chèn (Min Cost)";
    case "depot_zone_capacity_aware": return "Kho & khu vực (Depot-Aware)";
    case "cluster_round_robin": return "Phân cụm Round-Robin";
    default: return s;
  }
};

const formatAlgorithmName = (a: string) => {
  switch (a) {
    case "proposed_rl": return "Proposed DQN (RL Candidate Scoring)";
    case "nearest_neighbor": return "Nearest Neighbor (Tham lam cự ly)";
    case "nearest_neighbor_fallback": return "Nearest Neighbor (Dự phòng RL)";
    case "clarke_wright": return "Clarke–Wright (Tiết kiệm)";
    default: return a;
  }
};

export function PlanningDesk({
  session,
  selectedShipperId,
  onSelectShipper,
  nearbyOrders,
  onAssign,
  onUnassign,
  onAutoDispatch,
  onPreviewPlan,
  onConfirmPreview,
  onCancelPreview,
  onSwitchToLive,
  onSwitchToReport,
  onCreateNewSession,
  onNextDay,
  onSelectAlgorithm,
  actionLoading,
  actionError,
  actionSuccess,
  onReloadSession,
  previewPlan: propPreviewPlan,
  onSetPreviewPlan,
}: PlanningDeskProps) {
  // Strategy & Algorithm Controls
  const [selectedStrategy, setSelectedStrategy] = useState<string>("cluster_aware_rl");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>(session.routing_algorithm || "proposed_rl");

  useEffect(() => {
    if (session.routing_algorithm) {
      setSelectedAlgorithm(session.routing_algorithm);
    }
  }, [session.routing_algorithm]);
  
  // Plan Preview State (Lifted & Synchronized)
  const [internalPreviewPlan, setInternalPreviewPlan] = useState<PlanPreviewResponse | null>(null);
  const previewPlan = propPreviewPlan !== undefined ? propPreviewPlan : internalPreviewPlan;
  const setPreviewPlan = (plan: PlanPreviewResponse | null) => {
    if (onSetPreviewPlan) {
      onSetPreviewPlan(plan);
    }
    setInternalPreviewPlan(plan);
  };
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [confirmLoading, setConfirmLoading] = useState<boolean>(false);
  const [rlExplorerOpen, setRlExplorerOpen] = useState<boolean>(false);
  const [depotModalOpen, setDepotModalOpen] = useState<boolean>(false);
  const [newScenarioModalOpen, setNewScenarioModalOpen] = useState<boolean>(false);

  // Table search & filters
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [clusterFilter, setClusterFilter] = useState<string>("all");
  const [page, setPage] = useState<number>(0);
  const [selectedOrderIds, setSelectedOrderIds] = useState<string[]>([]);

  // GIS Map interactive state
  const [mapFocusPoint, setMapFocusPoint] = useState<[number, number] | null>(null);
  const [activeLayers, setActiveLayers] = useState({
    orders: true,
    shippers: true,
    clusters: true,
    routes: true,
    depot: true,
    bases: true,
  });
  const [legendOpen, setLegendOpen] = useState<boolean>(false);
  const [fleetFilter, setFleetFilter] = useState<"all" | "assigned" | "idle">("all");

  // Selected shipper
  const selectedShipper: ShipperData | undefined = session.shippers.find(
    (s) => s.shipper_id === selectedShipperId
  ) || session.shippers[0];

  // Shippers currently at depot
  const shippersAtDepot = useMemo(() => {
    return session.shippers.filter(
      (s) =>
        s.location_source === "DEPOT" ||
        (Math.abs(s.current_lat - session.depot.lat) < 0.001 &&
          Math.abs(s.current_lng - session.depot.lng) < 0.001)
    );
  }, [session.shippers, session.depot]);

  // Shippers currently in transit or away from depot
  const shippersOnField = useMemo(() => {
    return session.shippers.filter(
      (s) => !shippersAtDepot.some((depotS) => depotS.shipper_id === s.shipper_id)
    );
  }, [session.shippers, shippersAtDepot]);

  // Dispatchable orders = unassigned or carried-over orders up to current day
  const dispatchableOrdersCount = useMemo(() => {
    return session.orders.filter(
      (o) => o.release_day <= session.current_day && (o.delivery_status === "UNASSIGNED" || o.delivery_status === "CARRIED_OVER")
    ).length;
  }, [session.orders, session.current_day]);

  // Detailed 3-State Lifecycle: State A, State B, State C
  const isShiftEnded = session.is_day_ended;
  const isTotalDaysReached = session.current_day >= session.total_days;
  const isScenarioCompleted = isTotalDaysReached && isShiftEnded;
  const isDayEndedHasNextDay = isShiftEnded && !isTotalDaysReached;

  const handleNextDayInDesk = async () => {
    if (onNextDay) {
      await onNextDay();
    } else if (onSwitchToReport) {
      onSwitchToReport();
    }
  };

  // Plan summary metrics
  const isPlanCompleted = dispatchableOrdersCount === 0 && session.order_summary.assigned > 0;
  const isButtonsDisabled = previewLoading || actionLoading || dispatchableOrdersCount === 0 || session.is_day_ended;

  const totalShippersCount = session.shippers.length;
  const totalFleetCapacityKg = useMemo(() => {
    return session.shippers.reduce((sum, s) => sum + (s.capacity_kg || 100), 0);
  }, [session.shippers]);
  // "Đơn tồn đầu ngày" means an older, still-undelivered order, not
  // an order CREATED today that will be carried OUT at shift end.
  const carriedOverOrdersCount = useMemo(() => {
    return session.orders.filter((o) => o.release_day < session.current_day && o.delivery_status !== "DELIVERED").length;
  }, [session.orders, session.current_day]);
  const newOrdersCount = useMemo(() =>
    session.orders.filter((o) => o.release_day === session.current_day).length,
  [session.orders, session.current_day]);

  const activeShippersCount = useMemo(() => {
    return session.shippers.filter((s) => s.assigned_order_ids.length > 0).length;
  }, [session.shippers]);

  const totalAssignedWeight = useMemo(() => {
    return session.shippers.reduce((sum, s) => sum + s.current_load_kg, 0);
  }, [session.shippers]);

  const overweightShippers = useMemo(() => {
    return session.shippers.filter((s) => s.current_load_kg > s.capacity_kg);
  }, [session.shippers]);

  // Orders eligible for today's planning
  const todayEligibleOrders = useMemo(() => {
    return session.orders.filter((o) =>
      o.release_day === session.current_day ||
      (o.release_day < session.current_day && o.delivery_status !== "DELIVERED")
    );
  }, [session.orders, session.current_day]);

  // Cluster counts for dynamic GIS legend
  const clusterLegendItems = useMemo(() => {
    const map = new Map<number, number>();
    todayEligibleOrders.forEach((o) => {
      map.set(o.cluster_id, (map.get(o.cluster_id) || 0) + 1);
    });
    return Array.from(map.entries())
      .map(([clusterId, count]) => ({ clusterId, count }))
      .sort((a, b) => a.clusterId - b.clusterId);
  }, [todayEligibleOrders]);

  // Filtered shippers for fleet panel
  const filteredShippers = useMemo(() => {
    return session.shippers.filter((s) => {
      if (fleetFilter === "assigned") return s.assigned_order_ids.length > 0;
      if (fleetFilter === "idle") return s.assigned_order_ids.length === 0;
      return true;
    });
  }, [session.shippers, fleetFilter]);

  // Filtered orders for left panel
  const filteredOrders = useMemo(() => {
    return todayEligibleOrders.filter((o) => {
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        const matchCode = (o.order_code || o.order_id).toLowerCase().includes(term);
        const matchPkg = (o.package_name || "").toLowerCase().includes(term);
        if (!matchCode && !matchPkg) return false;
      }
      if (statusFilter === "unassigned") {
        if (o.delivery_status !== "UNASSIGNED" && o.delivery_status !== "CARRIED_OVER") return false;
      } else if (statusFilter === "assigned") {
        if (o.delivery_status !== "ASSIGNED" && o.delivery_status !== "IN_TRANSIT") return false;
      } else if (statusFilter === "delivered") {
        if (o.delivery_status !== "DELIVERED") return false;
      } else if (statusFilter === "carry") {
        if (o.carryover_count <= 0) return false;
      }
      if (clusterFilter !== "all") {
        if (o.cluster_id !== parseInt(clusterFilter, 10)) return false;
      }
      return true;
    });
  }, [todayEligibleOrders, searchTerm, statusFilter, clusterFilter]);

  // Paginated orders
  const totalPages = Math.max(1, Math.ceil(filteredOrders.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages - 1);
  const pageOrders = useMemo(() => {
    const start = currentPage * PAGE_SIZE;
    return filteredOrders.slice(start, start + PAGE_SIZE);
  }, [filteredOrders, currentPage]);

  const isPageAllSelected =
    pageOrders.length > 0 && pageOrders.every((o) => selectedOrderIds.includes(o.order_id));

  const toggleSelectPage = () => {
    if (isPageAllSelected) {
      const pageIds = new Set(pageOrders.map((o) => o.order_id));
      setSelectedOrderIds((prev) => prev.filter((id) => !pageIds.has(id)));
    } else {
      const newSelected = new Set(selectedOrderIds);
      pageOrders.forEach((o) => newSelected.add(o.order_id));
      setSelectedOrderIds(Array.from(newSelected));
    }
  };

  const toggleSelectOrder = (id: string) => {
    setSelectedOrderIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const focusOrderOnMap = (order: DispatchOrderItem) => {
    setMapFocusPoint([order.latitude, order.longitude]);
    if (!selectedOrderIds.includes(order.order_id)) {
      setSelectedOrderIds((prev) => [...prev, order.order_id]);
    }
  };

  // Selected orders metrics
  const selectedOrders = useMemo(() => {
    const idSet = new Set(selectedOrderIds);
    return session.orders.filter((o) => idSet.has(o.order_id));
  }, [selectedOrderIds, session.orders]);

  const selectedWeight = useMemo(() => {
    return selectedOrders.reduce((sum, o) => sum + (o.weight_kg || 0), 0);
  }, [selectedOrders]);

  const shipperCurrentLoad = selectedShipper ? selectedShipper.current_load_kg : 0;
  const shipperMaxCap = selectedShipper ? selectedShipper.capacity_kg : 100;
  const projectedLoad = shipperCurrentLoad + selectedWeight;
  const isOverweight = projectedLoad > shipperMaxCap;

  const canAssign =
    selectedOrders.length > 0 &&
    !isOverweight &&
    selectedOrders.every(
      (o) => o.delivery_status === "UNASSIGNED" || o.delivery_status === "CARRIED_OVER"
    );

  // Preview shipper data
  const previewShipper = useMemo(() => {
    if (!previewPlan || !selectedShipperId) return null;
    return previewPlan.shipper_previews.find((s) => s.shipper_id === selectedShipperId) || null;
  }, [previewPlan, selectedShipperId]);

  // Trigger Plan Preview
  const handleTriggerPreview = async () => {
    if (!onPreviewPlan) return;
    try {
      setPreviewLoading(true);
      const res = await onPreviewPlan(selectedStrategy, selectedAlgorithm);
      setPreviewPlan(res);
    } catch (err) {
      console.error("Preview failed:", err);
    } finally {
      setPreviewLoading(false);
    }
  };

  // Confirm Active Preview
  const handleConfirmActivePreview = async () => {
    if (!previewPlan || !onConfirmPreview) return;
    try {
      setConfirmLoading(true);
      await onConfirmPreview(previewPlan.preview_id);
      setPreviewPlan(null);
      setSelectedOrderIds([]);
    } catch (err) {
      console.error("Confirm preview failed:", err);
    } finally {
      setConfirmLoading(false);
    }
  };

  // Cancel Active Preview
  const handleCancelActivePreview = async () => {
    if (onCancelPreview) {
      await onCancelPreview();
    }
    setPreviewPlan(null);
  };

  // Route positions for selected shipper
  const confirmedRoutePositions = useMemo(() => {
    if (!selectedShipper || !selectedShipper.route_geometry) return [];
    return selectedShipper.route_geometry.map(([lng, lat]) => [lat, lng] as [number, number]);
  }, [selectedShipper]);

  const previewRoutePositions = useMemo(() => {
    if (!previewShipper || !previewShipper.route_geometry) return [];
    return previewShipper.route_geometry.map(([lng, lat]) => [lat, lng] as [number, number]);
  }, [previewShipper]);

  return (
    <div className="flex flex-col gap-2.5 sm:gap-3 flex-1 min-h-0 w-full max-w-full">
      {/* Scenario Completed Banner */}
      {isScenarioCompleted && (
        <div id="scenarioCompletedBanner" className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-4 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-emerald-600 text-white shrink-0">
              <CheckCheck className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-foreground">
                  Kịch bản mô phỏng {session.total_days} ngày đã hoàn thành toàn bộ chu kỳ điều phối
                </h3>
                <span className="rounded-full bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 px-2 py-0.2 text-[10px] font-bold">
                  Hoàn tất chu kỳ
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Toàn bộ {session.orders.length} đơn hàng đã hoàn tất qua {session.total_days} ngày mô phỏng. Lịch sử ca, vết quyết định RL và cấu hình kho vẫn được lưu trữ nguyên vẹn để đối sánh và thuyết minh.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {onSwitchToReport && (
              <button
                type="button"
                id="btnViewFullReportFromCompleted"
                onClick={onSwitchToReport}
                className="flex h-8.5 items-center gap-1.5 rounded-lg bg-emerald-600 px-3.5 text-xs font-bold text-white shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
              >
                <FileText className="size-3.5" />
                <span>Xem Báo cáo Tổng kết ca</span>
              </button>
            )}
            <button
              type="button"
              id="btnOpenNewScenarioModalDesk"
              onClick={() => setNewScenarioModalOpen(true)}
              className="flex h-8.5 items-center gap-1.5 rounded-lg border border-border bg-card px-3 text-xs font-semibold text-foreground hover:bg-muted transition-all cursor-pointer shadow-2xs"
            >
              <RotateCcw className="size-3.5 text-teal-600" />
              <span>Khởi tạo kịch bản mới</span>
            </button>
          </div>
        </div>
      )}
      {/* ============================================================== */}
      {/* 1. THREE-ZONE LOGICAL CONTROL TOWER & UNIFIED ACTION PANEL     */}
      {/* ============================================================== */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 gap-3 rounded-2xl border border-border/80 bg-card p-3 sm:p-3.5 shadow-2xs max-w-full min-w-0">
        
        {/* ============================================================ */}
        {/* KHU VỰC A: THÔNG TIN & CẤU HÌNH (4/12 COLS)                 */}
        {/* ============================================================ */}
        <div className="md:col-span-1 xl:col-span-4 flex flex-col justify-between gap-2.5 p-3 rounded-xl border border-border/60 bg-muted/20">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex size-7 items-center justify-center rounded-lg bg-teal-600/15 text-teal-600 dark:text-teal-400 border border-teal-500/20 shrink-0">
                <Compass className="size-4" />
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="font-extrabold text-sm sm:text-base tracking-tight text-foreground m-0">
                  Bàn lập tuyến
                </h2>
                <span className="rounded-md bg-teal-600/15 border border-teal-500/30 px-2 py-0.5 font-mono text-xs font-bold text-teal-700 dark:text-teal-300">
                  Ngày {session.current_day}/{session.total_days}
                </span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-1 mb-0 leading-relaxed">
              Phân bổ <strong className="text-foreground font-mono">{session.orders.length} đơn</strong> cho <strong className="text-foreground font-mono">{totalShippersCount} shipper</strong> • Candidate-Scoring DQN với 8 đặc trưng trạng thái.
            </p>
          </div>

          <div className="space-y-1.5 pt-1 border-t border-border/50">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <select
                id="assignmentStrategy"
                className="h-8.5 rounded-lg border border-border/80 bg-background px-2 text-xs font-semibold text-foreground focus:outline-hidden focus:ring-1 focus:ring-teal-500 cursor-pointer shadow-2xs w-full truncate"
                value={selectedStrategy}
                onChange={(e) => setSelectedStrategy(e.target.value)}
                disabled={actionLoading || previewLoading || isShiftEnded}
                title="Chiến lược gán đơn hàng vào shipper"
              >
                <option value="depot_zone_capacity_aware">Kho, khu vực &amp; tải trọng (Depot-Aware)</option>
                <option value="cluster_aware_rl">Ưu tiên cụm + chi phí (Cluster-Aware Min Cost)</option>
                <option value="location_aware_min_cost">Tối ưu cự ly chèn (Min Incremental Detour)</option>
                <option value="location_aware_nearest">Shipper gần nhất (Nearest Shipper)</option>
                <option value="cluster_round_robin">Phân cụm Round-Robin (K-Means)</option>
              </select>

              <select
                id="routingAlgorithm"
                className="h-8.5 rounded-lg border border-border/80 bg-background px-2 text-xs font-semibold text-foreground focus:outline-hidden focus:ring-1 focus:ring-teal-500 cursor-pointer shadow-2xs w-full truncate"
                value={selectedAlgorithm}
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
                disabled={actionLoading || previewLoading || isShiftEnded}
                title="Thuật toán sắp xếp thứ tự giao hàng"
              >
                <option value="proposed_rl">Proposed DQN (Candidate-Scoring DQN)</option>
                <option value="nearest_neighbor">Nearest Neighbor (Tham lam cự ly)</option>
                <option value="clarke_wright">Clarke–Wright (Heuristic tiết kiệm)</option>
              </select>
            </div>
          </div>
        </div>

        {/* ============================================================ */}
        {/* KHU VỰC B: TIẾN TRÌNH & KPI (4/12 COLS - 2 DISTINCT TIERS)    */}
        {/* ============================================================ */}
        <div className="md:col-span-1 xl:col-span-4 flex flex-col justify-between gap-2.5 p-3 rounded-xl border border-border/60 bg-muted/20">
          {/* Tầng 1: Pipeline 5 bước */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-muted-foreground font-semibold">
              <span className="uppercase tracking-wider text-[10px]">Tiến trình điều phối</span>
              <span className="font-mono text-[10px]">5 Bước</span>
            </div>
            <div className="flex items-center gap-1 bg-card/80 p-1 rounded-lg border border-border/60 text-xs overflow-x-auto max-w-full no-scrollbar">
              <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 transition-colors border text-[11px] ${!previewPlan && !isPlanCompleted ? "bg-teal-600 text-white font-bold border-teal-600 shadow-2xs" : "bg-card text-foreground/80 border-border font-medium"}`}>
                <span className="flex size-3.5 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">1</span>
                <span>Cấu hình</span>
              </span>
              <span className="text-muted-foreground/40 font-bold text-[10px]">→</span>
              <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 transition-colors border text-[11px] ${previewPlan ? "bg-cyan-600 text-white font-bold border-cyan-600 shadow-2xs" : "bg-card text-foreground/80 border-border font-medium"}`}>
                <span className="flex size-3.5 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">2</span>
                <span>Xem trước</span>
              </span>
              <span className="text-muted-foreground/40 font-bold text-[10px]">→</span>
              <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 transition-colors border text-[11px] ${previewPlan ? "bg-cyan-500/20 text-cyan-800 dark:text-cyan-200 font-bold border-cyan-500/40" : "bg-card text-foreground/80 border-border font-medium"}`}>
                <span className="flex size-3.5 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">3</span>
                <span>So sánh</span>
              </span>
              <span className="text-muted-foreground/40 font-bold text-[10px]">→</span>
              <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 transition-colors border text-[11px] ${isPlanCompleted ? "bg-emerald-600 text-white font-bold border-emerald-600 shadow-2xs" : "bg-card text-foreground/80 border-border font-medium"}`}>
                <span className="flex size-3.5 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">4</span>
                <span>Xác nhận</span>
              </span>
              <span className="text-muted-foreground/40 font-bold text-[10px]">→</span>
              <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-medium bg-card text-foreground/80 border border-border text-[11px]">
                <span className="flex size-3.5 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">5</span>
                <span>Thực địa</span>
              </span>
            </div>
          </div>

          {/* Tầng 2: Nhóm KPI chips */}
          <div className="flex flex-wrap items-center gap-1.5 text-xs pt-1.5 border-t border-border/50">
            <span className="inline-flex items-center gap-1 rounded-md bg-teal-500/10 px-2 py-0.5 font-medium text-teal-700 dark:text-teal-300 border border-teal-500/20">
              <span>Đơn mới:</span>
              <strong className="font-mono">{newOrdersCount}</strong>
            </span>
            {carriedOverOrdersCount > 0 && (
              <span className="inline-flex items-center gap-1 rounded-md bg-rose-500/10 px-2 py-0.5 font-medium text-rose-700 dark:text-rose-400 border border-rose-500/20">
                <span>Tồn đầu ngày:</span>
                <strong className="font-mono">{carriedOverOrdersCount}</strong>
              </span>
            )}
            <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-0.5 font-medium text-amber-700 dark:text-amber-400 border border-amber-500/20">
              <span>Chưa gán:</span>
              <strong className="font-mono">{dispatchableOrdersCount}</strong>
            </span>
            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 font-medium text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
              <span>{isShiftEnded ? "Đang gán cuối ca:" : "Đã gán:"}</span>
              <strong className="font-mono">{session.order_summary.assigned}</strong>
            </span>
            <span className="inline-flex items-center gap-1 rounded-md bg-sky-500/10 px-2 py-0.5 font-medium text-sky-700 dark:text-sky-400 border border-sky-500/20">
              <span>Đội xe:</span>
              <strong className="font-mono">{activeShippersCount}/{totalShippersCount}</strong>
            </span>
            {!isShiftEnded && (
              <span className="inline-flex items-center gap-1 rounded-md bg-card px-2 py-0.5 font-medium text-foreground border border-border">
                <span className="text-muted-foreground">Tải hiện tại:</span>
                <strong className="font-mono">{totalAssignedWeight.toFixed(1)}/{totalFleetCapacityKg} kg</strong>
              </span>
            )}
          </div>
        </div>

        {/* ============================================================ */}
        {/* KHU VỰC C: THAO TÁC ĐIỀU PHỐI (4/12 COLS - ACTION CONTROL)   */}
        {/* ============================================================ */}
        <div className="md:col-span-2 xl:col-span-4 flex flex-col justify-between gap-2 p-3 rounded-xl border border-border/80 bg-background/60 dark:bg-card/90 shadow-2xs">
          {/* Action Header */}
          <div className="flex items-center justify-between gap-2 border-b border-border/60 pb-1.5">
            <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
              <Sliders className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
              <span className="uppercase tracking-wider text-xs font-bold text-muted-foreground">Thao tác điều phối</span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                id="btnHeaderNewScenario"
                onClick={() => setNewScenarioModalOpen(true)}
                className="inline-flex items-center gap-1 text-xs font-semibold text-muted-foreground hover:text-foreground cursor-pointer px-2 py-0.5 rounded border border-border/60 hover:bg-muted"
                title="Khởi tạo kịch bản mới với quy mô 3, 5, 7 hoặc 14 ngày"
              >
                <RotateCcw className="size-3 text-teal-600" />
                <span>Kịch bản mới</span>
              </button>

              {isScenarioCompleted ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-xs font-bold text-emerald-700 dark:text-emerald-300">
                  <CheckCheck className="size-3.5" />
                  <span>Hoàn tất {session.total_days}/{session.total_days} ngày</span>
                </span>
              ) : isDayEndedHasNextDay ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-xs font-bold text-amber-700 dark:text-amber-400">
                  <Clock className="size-3.5" />
                  <span>Đã chốt Ngày {session.current_day}</span>
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-teal-500/15 border border-teal-500/30 px-2 py-0.5 text-xs font-bold text-teal-700 dark:text-teal-300">
                  <Sparkles className="size-3.5" />
                  <span>Đang lập tuyến</span>
                </span>
              )}
            </div>
          </div>

          {/* Hàng 1: Nút phụ trợ (Kho và điểm trực | Vết quyết định RL) - Đồng chiều cao h-8.5 */}
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              id="btnOpenDepotConfig"
              onClick={() => setDepotModalOpen(true)}
              className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg border border-border bg-card px-2 text-xs font-semibold text-foreground hover:bg-muted hover:border-teal-500/40 transition-colors cursor-pointer shadow-2xs"
              title="Cấu hình vị trí kho trung tâm và khu vực phục vụ của shipper"
            >
              <Warehouse className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
              <span className="truncate">Kho và điểm trực</span>
            </button>

            <button
              type="button"
              id="btnOpenRlExplorer"
              onClick={() => setRlExplorerOpen(true)}
              className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/10 px-2 text-xs font-semibold text-teal-700 dark:text-teal-300 hover:bg-teal-500/20 transition-colors cursor-pointer shadow-2xs"
              title="Mở bộ giải thích vết quyết định mạng nơ-ron Candidate-Scoring DQN"
            >
              <BrainCircuit className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
              <span className="truncate">Vết quyết định RL</span>
            </button>
          </div>

          {/* Hàng 2: Nút thực thi chính (Đồng chiều cao h-8.5, primary rõ hơn secondary) */}
          {isScenarioCompleted ? (
            <div className="space-y-1">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  id="btnViewReportCompletedInDesk"
                  onClick={onSwitchToReport}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg border border-border bg-card px-2 text-xs font-semibold text-foreground hover:bg-muted transition-colors cursor-pointer shadow-2xs"
                  title="Xem báo cáo kiểm toán tổng kết toàn bộ chu kỳ"
                >
                  <FileText className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
                  <span className="truncate">Báo cáo tổng kết</span>
                </button>

                <button
                  type="button"
                  id="btnNewScenarioDesk"
                  onClick={() => setNewScenarioModalOpen(true)}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg bg-emerald-600 px-2 text-xs font-bold text-white shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
                  title="Khởi tạo phiên mô phỏng mới với quy mô 3, 5, 7 hoặc 14 ngày"
                >
                  <RotateCcw className="size-3.5 shrink-0" />
                  <span className="truncate">Tạo phiên mới ↺</span>
                </button>
              </div>
              <p className="text-[10px] text-emerald-700 dark:text-emerald-300 leading-snug m-0">
                ✓ Phiên đã hoàn tất {session.total_days}/{session.total_days} ngày. Vui lòng bấm &quot;Tạo phiên mới&quot; để tiếp tục.
              </p>
            </div>
          ) : isDayEndedHasNextDay ? (
            <div className="space-y-1">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  id="btnViewReportInDesk"
                  onClick={onSwitchToReport}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg border border-border bg-card px-2 text-xs font-semibold text-foreground hover:bg-muted transition-colors cursor-pointer shadow-2xs"
                  title={`Xem báo cáo ca Ngày ${session.current_day}`}
                >
                  <FileText className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
                  <span className="truncate">Báo cáo Ngày {session.current_day}</span>
                </button>

                <button
                  type="button"
                  id="btnStartNextDayInDesk"
                  onClick={handleNextDayInDesk}
                  disabled={actionLoading}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg bg-teal-600 px-2 text-xs font-bold text-white shadow-xs hover:bg-teal-700 transition-all cursor-pointer disabled:opacity-50"
                  title={`Chốt ca và chuyển sang Ngày ${session.current_day + 1}`}
                >
                  <span className="truncate">Bắt đầu Ngày {session.current_day + 1}</span>
                  <ArrowRight className="size-3.5 shrink-0" />
                </button>
              </div>
              <p className="text-[10px] text-amber-700 dark:text-amber-400 leading-snug m-0">
                ℹ️ Ca Ngày {session.current_day} đã chốt. Bấm &quot;Bắt đầu Ngày {session.current_day + 1}&quot; để nạp đơn mới và đơn tồn.
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  id="btnPreviewPlan"
                  onClick={handleTriggerPreview}
                  disabled={isButtonsDisabled}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-2 text-xs font-bold text-cyan-800 dark:text-cyan-200 hover:bg-cyan-500/25 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed shadow-2xs"
                  title={
                    dispatchableOrdersCount === 0
                      ? "Toàn bộ đơn hôm nay đã được phân bổ. Sẵn sàng vào Mô phỏng."
                      : "Xem trước kết quả phân công và đối sánh KPI trước khi áp dụng"
                  }
                >
                  <Eye className={`size-3.5 shrink-0 ${previewLoading ? "animate-spin" : ""}`} />
                  <span className="truncate">{previewLoading ? "Đang tính…" : (previewPlan ? "Cập nhật xem trước" : "Xem trước phương án")}</span>
                </button>

                <button
                  type="button"
                  id="btnDirectDispatch"
                  onClick={() => onAutoDispatch(selectedStrategy, selectedAlgorithm)}
                  disabled={isButtonsDisabled}
                  className="flex h-8.5 items-center justify-center gap-1.5 rounded-lg bg-teal-600 px-2 text-xs font-bold text-white shadow-xs hover:bg-teal-700 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                  title={
                    dispatchableOrdersCount === 0
                      ? "Không còn đơn chưa gán. Bấm 'Bắt đầu mô phỏng' để chạy tuyến."
                      : "Tự động phân bổ và áp dụng trực tiếp kế hoạch điều phối vào đội xe"
                  }
                >
                  <RefreshCw className={`size-3.5 shrink-0 ${actionLoading ? "animate-spin" : ""}`} />
                  <span className="truncate">{actionLoading ? "Đang gán…" : "Xác nhận kế hoạch"}</span>
                </button>
              </div>

              <p className="text-[10px] text-muted-foreground leading-snug m-0">
                {dispatchableOrdersCount > 0 ? (
                  <span>Sẵn sàng phân bổ <b>{dispatchableOrdersCount} đơn</b> cho {totalShippersCount} shipper.</span>
                ) : (
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">✓ 100% đơn đã phân bổ. Sẵn sàng bắt đầu mô phỏng thực địa.</span>
                )}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. MAP-FIRST THREE-COLUMN WORKSPACE (~60-65% Map Canvas)       */}
      {/* ============================================================== */}
      <div className="flex flex-col lg:flex-row gap-3 w-full max-w-full min-w-0">
        {/* ──────────────────────────────────────────────────────────── */}
        {/* LEFT COLUMN: ORDERS WORKLIST (Fixed 300px on desktop)        */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="w-full lg:w-[300px] xl:w-[320px] shrink-0 order-2 lg:order-1 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs h-[480px] sm:h-[520px] lg:h-[500px] xl:h-[520px]">
          {/* Worklist Header */}
          <div className="p-2.5 border-b border-border/70 bg-muted/20 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Package className="size-4 text-teal-600 dark:text-teal-400" />
                <span className="font-bold text-xs uppercase tracking-wider text-foreground">
                  Danh sách đơn hàng
                </span>
              </div>
              <span className="rounded-full bg-muted px-2 py-0.5 font-mono text-xs font-semibold text-muted-foreground">
                {filteredOrders.length} đơn
              </span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-2.5 top-2 size-3.5 text-muted-foreground" />
              <input
                id="searchOrderDesk"
                type="text"
                placeholder="Tìm mã đơn, tên kiện…"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setPage(0);
                }}
                className="w-full h-7.5 rounded-lg border border-border/80 bg-background pl-8 pr-2 text-xs placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-teal-500"
              />
            </div>

            {/* Quick Filters */}
            <div className="grid grid-cols-2 gap-1.5">
              <select
                id="filterStatusDesk"
                className="h-7.5 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground cursor-pointer"
                value={statusFilter}
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setPage(0);
                }}
              >
                <option value="all">Tất cả trạng thái</option>
                <option value="unassigned">Chưa gán</option>
                <option value="assigned">Đã gán</option>
                <option value="delivered">Đã giao</option>
                <option value="carry">Tồn đọng</option>
              </select>

              <select
                id="filterClusterDesk"
                className="h-7.5 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground cursor-pointer"
                value={clusterFilter}
                onChange={(e) => {
                  setClusterFilter(e.target.value);
                  setPage(0);
                }}
              >
                <option value="all">Tất cả cụm (0-9)</option>
                {Array.from({ length: 10 }, (_, i) => (
                  <option key={i} value={String(i)}>
                    Cụm {i}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Orders Scrollable List */}
          <div className="flex-1 min-h-0 overflow-y-auto divide-y divide-border/50 p-1">
            {pageOrders.length > 0 ? (
              pageOrders.map((o) => {
                const isSelected = selectedOrderIds.includes(o.order_id);
                const isCandidate = previewPlan?.order_previews.some(
                  (po) => po.order_id === o.order_id
                );
                const candidateInfo = previewPlan?.order_previews.find(
                  (po) => po.order_id === o.order_id
                );

                return (
                  <div
                    key={o.order_id}
                    onClick={() => focusOrderOnMap(o)}
                    className={`group flex items-start gap-2 p-2 rounded-lg transition-colors cursor-pointer text-xs ${
                      isSelected
                        ? "bg-teal-500/10 border-l-3 border-teal-600"
                        : "hover:bg-muted/50"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={(e) => {
                        e.stopPropagation();
                        toggleSelectOrder(o.order_id);
                      }}
                      className="mt-0.5 size-3.5 rounded border-border text-teal-600 cursor-pointer"
                    />

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono font-bold text-foreground truncate">
                          {o.order_code || o.order_id}
                        </span>
                        <span
                          className="shrink-0 text-xs font-bold px-1.5 py-0.5 rounded-full text-white"
                          style={{ backgroundColor: clusterColor(o.cluster_id) }}
                        >
                          Cụm {o.cluster_id}
                        </span>
                      </div>

                      <div className="text-xs text-muted-foreground truncate mt-0.5">
                        {o.package_name || "Kiện hàng tiêu chuẩn"}
                      </div>

                      <div className="flex items-center gap-2 mt-1 text-xs">
                        <span className="font-semibold text-foreground">
                          {o.weight_kg.toFixed(1)} kg
                        </span>
                        <span>•</span>
                        {o.delivery_status === "ASSIGNED" ? (
                          <span className="text-teal-600 dark:text-teal-400 font-semibold">
                            Shipper #{o.assigned_shipper_id?.replace("shipper_", "")}
                          </span>
                        ) : isCandidate && candidateInfo ? (
                          <span className="text-cyan-600 dark:text-cyan-400 font-semibold">
                            Dự kiến → #{candidateInfo.candidate_shipper_id?.replace("shipper_", "")}
                          </span>
                        ) : (
                          <span className="text-amber-500 font-semibold">Chưa gán</span>
                        )}
                      </div>

                      {candidateInfo?.dispatch_reason && (
                        <div className="text-xs text-muted-foreground mt-1 bg-muted/40 px-2 py-0.5 rounded flex items-center justify-between">
                          <span className="truncate">Lý do: {candidateInfo.dispatch_reason}</span>
                          {candidateInfo.dispatch_distance_km !== undefined && candidateInfo.dispatch_distance_km !== null && (
                            <span className="font-mono font-bold shrink-0 ml-1">+{candidateInfo.dispatch_distance_km.toFixed(1)}km</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="p-6 text-center text-xs text-muted-foreground">
                Không tìm thấy đơn hàng phù hợp.
              </div>
            )}
          </div>

          {/* Worklist Footer: Pagination & Select All */}
          <div className="p-2 border-t border-border/70 bg-muted/20 flex items-center justify-between text-xs">
            <button
              type="button"
              onClick={toggleSelectPage}
              className="text-xs font-semibold text-teal-600 dark:text-teal-400 hover:underline cursor-pointer"
            >
              {isPageAllSelected ? "Bỏ chọn trang" : "Chọn trang này"}
            </button>

            <div className="flex items-center gap-1.5">
              <span className="text-xs font-mono text-muted-foreground">
                {currentPage + 1}/{totalPages}
              </span>
              <button
                type="button"
                disabled={currentPage <= 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="h-7 px-2 rounded border border-border bg-background text-xs font-medium disabled:opacity-40 cursor-pointer"
              >
                ←
              </button>
              <button
                type="button"
                disabled={currentPage + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="h-7 px-2 rounded border border-border bg-background text-xs font-medium disabled:opacity-40 cursor-pointer"
              >
                →
              </button>
            </div>
          </div>
        </div>

        {/* ──────────────────────────────────────────────────────────── */}
        {/* CENTER COLUMN: GIS MAP & PREVIEW DELTA BAR (~60-65% width)   */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="flex-1 min-w-0 order-1 lg:order-2 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs relative h-[480px] sm:h-[520px] lg:h-[500px] xl:h-[520px]">
          {/* Floating Map Layer Toggles */}
          <div className="absolute top-3 right-3 z-30 flex items-center gap-1 bg-card/90 p-1 rounded-lg border border-border/80 shadow-md backdrop-blur-md text-xs max-w-[calc(100%-24px)] overflow-x-auto no-scrollbar">
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, orders: !prev.orders }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.orders ? "bg-teal-600 text-white" : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Đơn ({todayEligibleOrders.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, routes: !prev.routes }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.routes ? "bg-teal-600 text-white" : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Tuyến
            </button>
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, bases: !prev.bases }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.bases ? "bg-purple-600 text-white" : "text-muted-foreground hover:bg-muted"
              }`}
              title="Bật/tắt hiển thị 6 trạm trực ca tại TP.HCM"
            >
              Trạm ca
            </button>
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, depot: !prev.depot }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.depot ? "bg-blue-600 text-white" : "text-muted-foreground hover:bg-muted"
              }`}
              title="Bật/tắt hiển thị Kho trung tâm"
            >
              Kho chính
            </button>
            <button
              type="button"
              onClick={() => setMapFocusPoint([session.depot.lat, session.depot.lng])}
              className="px-2 py-0.5 rounded text-muted-foreground hover:bg-muted font-medium transition-colors cursor-pointer"
              title="Về Depot trung tâm"
            >
              Depot
            </button>
          </div>

          {/* Interactive Leaflet Map */}
          <div className="flex-1 w-full relative min-h-0">
            <MapBase height="100%">
              <MapController focusPoint={mapFocusPoint} />

              {/* Central Depot Marker */}
              {activeLayers.depot && (
                <Marker position={[session.depot.lat, session.depot.lng]} icon={depotIcon()}>
                  <Popup>
                    <div className="text-xs space-y-1">
                      <strong className="text-teal-600 dark:text-teal-400 block font-bold">KHO TRUNG TÂM (CENTRAL HUB)</strong>
                      <div>Tên: <strong>{session.depot.name}</strong></div>
                      <div>Tọa độ: [{session.depot.lat}, {session.depot.lng}]</div>
                      <div>Loại: <span className="font-mono text-xs">{session.depot.depot_type || "CENTRAL_HUB"}</span></div>
                      <div className="text-xs text-muted-foreground">
                        Điểm tập kết hàng hóa, nhận kiện và kết thúc ca cho toàn đội giao vận TP.HCM.
                      </div>
                    </div>
                  </Popup>
                </Marker>
              )}

              {/* 6 Logistics Staging Bases on HCMC Network */}
              {activeLayers.bases && session.staging_bases?.map((base: StagingBase) => (
                <Marker
                  key={base.base_id}
                  position={[base.lat, base.lng]}
                  icon={createStagingBaseIcon(base.base_id, base.name)}
                >
                  <Popup>
                    <div className="text-xs space-y-1">
                      <strong className="text-purple-600 dark:text-purple-400 block font-bold">
                        {base.name}
                      </strong>
                      <div>Quận/Khu vực: <strong>{base.district}</strong></div>
                      <div>Tọa độ: [{base.lat}, {base.lng}]</div>
                      <div>Cụm ưu tiên: {base.primary_clusters.map((ci: number) => `Cụm ${ci}`).join(", ")}</div>
                      <div className="text-xs text-muted-foreground">
                        Trạm điều phối trực ca vệ tinh trong mạng lưới logistics HCMC
                      </div>
                    </div>
                  </Popup>
                </Marker>
              ))}

              {/* Depot Fleet Hub Marker */}
              {shippersAtDepot.length > 0 && (
                <Marker
                  position={[session.depot.lat, session.depot.lng]}
                  icon={createDepotFleetIcon(shippersAtDepot.length)}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 max-w-[200px]">
                      <strong className="block text-teal-600 dark:text-teal-400 font-bold">
                        {shippersAtDepot.length} Shipper đang tại Depot
                      </strong>
                      <div className="max-h-36 overflow-y-auto divide-y divide-border text-xs">
                        {shippersAtDepot.map((s) => (
                          <div
                            key={s.shipper_id}
                            onClick={() => onSelectShipper(s.shipper_id)}
                            className="py-1 flex items-center justify-between cursor-pointer hover:text-teal-600"
                          >
                            <span>Shipper #{s.shipper_id.replace("shipper_", "")}</span>
                            <span className="font-mono text-muted-foreground">{s.assigned_order_ids.length} đơn</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </Popup>
                </Marker>
              )}

              {/* Shippers currently on field */}
              {activeLayers.shippers &&
                shippersOnField.map((s) => (
                  <Marker
                    key={s.shipper_id}
                    position={[s.current_lat, s.current_lng]}
                    icon={createShipperIcon(s.shipper_id, s.shipper_id === selectedShipperId)}
                    eventHandlers={{
                      click: () => onSelectShipper(s.shipper_id),
                    }}
                  >
                    <Popup>
                      <div className="text-xs space-y-1">
                        <strong>Shipper #{s.shipper_id.replace("shipper_", "")}</strong>
                        <div>Tải trọng: {s.current_load_kg.toFixed(1)} / {s.capacity_kg} kg</div>
                        <div>Đơn đã gán: {s.assigned_order_ids.length}</div>
                      </div>
                    </Popup>
                  </Marker>
                ))}

              {/* Orders Markers */}
              {activeLayers.orders &&
                todayEligibleOrders.map((o) => {
                  const isSelected = selectedOrderIds.includes(o.order_id);
                  const color = clusterColor(o.cluster_id);
                  const isAssigned = o.delivery_status === "ASSIGNED";

                  return (
                    <CircleMarker
                      key={o.order_id}
                      center={[o.latitude, o.longitude]}
                      radius={isSelected ? 8 : 5}
                      pathOptions={{
                        color: isSelected ? "#eab308" : isAssigned ? "#0ea5e9" : color,
                        fillColor: color,
                        fillOpacity: isSelected ? 0.9 : 0.75,
                        weight: isSelected ? 3 : 1.5,
                      }}
                      eventHandlers={{
                        click: () => {
                          toggleSelectOrder(o.order_id);
                        },
                      }}
                    >
                      <Popup>
                        <div className="text-xs space-y-1">
                          <strong className="block">{o.order_code || o.order_id}</strong>
                          <div>{o.package_name || "Kiện hàng"}</div>
                          <div>Khối lượng: {o.weight_kg.toFixed(1)} kg</div>
                          <div>Cụm K-Means: #{o.cluster_id}</div>
                          <div>Trạng thái: <strong>{o.delivery_status}</strong></div>
                          {o.assigned_shipper_id && (
                            <div className="text-teal-600 dark:text-teal-400 font-semibold">
                              Gán cho: {o.assigned_shipper_id}
                            </div>
                          )}
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}

              {/* Confirmed Route Polyline */}
              {activeLayers.routes && confirmedRoutePositions.length > 1 && (
                <Polyline
                  positions={confirmedRoutePositions}
                  pathOptions={{
                    color: "#0f766e",
                    weight: 4,
                    opacity: 0.85,
                  }}
                />
              )}

              {/* Candidate Preview Route Polyline (Dashed Cyan) */}
              {activeLayers.routes && previewRoutePositions.length > 1 && (
                <Polyline
                  positions={previewRoutePositions}
                  pathOptions={{
                    color: "#06b6d4",
                    weight: 3.5,
                    dashArray: "6, 6",
                    opacity: 0.9,
                  }}
                />
              )}

              {/* Route sequence stop pins */}
              {activeLayers.routes &&
                selectedShipper &&
                selectedShipper.timeline_stops &&
                selectedShipper.timeline_stops.map((stop, idx) => (
                  <Marker
                    key={`stop_${idx}`}
                    position={[stop.lat, stop.lng]}
                    icon={createStopPinIcon(idx + 1, false)}
                  >
                    <Popup>
                      <div className="text-xs space-y-1">
                        <strong>Điểm dừng #{idx + 1}</strong>
                        <div>{stop.package_name || stop.order_code || stop.order_id}</div>
                        <div>Dự kiến đến: phút {stop.arrival_minute.toFixed(0)}</div>
                        <div>Cự ly chặng: {stop.leg_distance_km.toFixed(1)} km</div>
                      </div>
                    </Popup>
                  </Marker>
                ))}
            </MapBase>

            {/* Floating Collapsible GIS Dynamic Legend */}
            <div className="absolute bottom-4 left-4 z-30 max-w-[260px] bg-card/95 border border-border/80 rounded-xl shadow-lg backdrop-blur-md overflow-hidden text-xs">
              <div
                onClick={() => setLegendOpen((prev) => !prev)}
                className="flex items-center justify-between px-3 py-1.5 bg-muted/40 hover:bg-muted/60 cursor-pointer border-b border-border/60 select-none"
              >
                <div className="flex items-center gap-1.5 font-bold text-xs text-foreground">
                  <Layers className="size-3.5 text-teal-600 dark:text-teal-400" />
                  <span>Chú giải Cụm &amp; Tuyến GIS</span>
                </div>
                <button
                  type="button"
                  className="text-muted-foreground p-0.5 hover:text-foreground cursor-pointer"
                  aria-label={legendOpen ? "Thu gọn chú giải" : "Mở rộng chú giải"}
                >
                  {legendOpen ? <ChevronDown className="size-3.5" /> : <ChevronUp className="size-3.5" />}
                </button>
              </div>

              {legendOpen && (
                <div className="p-2.5 space-y-2.5 max-h-52 overflow-y-auto text-xs">
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-1">
                      Phân bố Đơn theo Cụm K-Means ({clusterLegendItems.length} Cụm)
                    </span>
                    <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                      {clusterLegendItems.map((c) => (
                        <div key={c.clusterId} className="flex items-center gap-1.5 py-0.5">
                          <span
                            className="size-2.5 rounded-full shrink-0 shadow-2xs"
                            style={{ backgroundColor: clusterColor(c.clusterId) }}
                          />
                          <span className="font-mono text-xs font-medium text-foreground truncate">
                            Cụm {c.clusterId}: <strong>{c.count}</strong> đơn
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="pt-2 border-t border-border/60 space-y-1.5">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-1 bg-[#0f766e] rounded-full inline-block shrink-0" />
                      <span className="text-xs font-medium text-foreground">Tuyến chính thức đã gán</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-0.5 border-b-2 border-dashed border-[#06b6d4] inline-block shrink-0" />
                      <span className="text-xs font-medium text-foreground">Tuyến xem trước (Preview)</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ──────────────────────────────────────────────────────────── */}
          {/* FLOATING PREVIEW COMPARISON BAR (Active when preview is ready) */}
          {/* ──────────────────────────────────────────────────────────── */}
          {previewPlan && (
            <div className="border-t border-border/80 bg-card/98 p-3 shadow-lg z-20 animate-in slide-in-from-bottom-2">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="size-2.5 rounded-full bg-cyan-500 animate-pulse" />
                    <div>
                      <span className="font-bold text-foreground block">
                        Bản xem trước phương án (Preview)
                      </span>
                      <div className="flex flex-wrap items-center gap-1.5 text-xs">
                        <span className="text-muted-foreground">
                          {formatStrategyName(previewPlan.strategy_used)} • {formatAlgorithmName(previewPlan.algorithm_used)}
                        </span>
                        {previewPlan.is_fallback && (
                          <span
                            className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 text-xs font-semibold"
                            title={previewPlan.fallback_reason || "Đã kích hoạt dự phòng sang Nearest Neighbor"}
                          >
                            ⚠️ Dự phòng: Nearest Neighbor
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
                    <div className="bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                      <span className="text-muted-foreground mr-1">Cự ly:</span>
                      <strong className="text-cyan-600 dark:text-cyan-400">{previewPlan.candidate_kpi.total_distance_km} km</strong>
                      {previewPlan.deltas && (
                        <span className={`ml-1 text-xs font-bold ${previewPlan.deltas.delta_distance_km <= 0 ? 'text-emerald-600' : 'text-amber-600'}`}>
                          ({previewPlan.deltas.delta_distance_km >= 0 ? "+" : ""}{previewPlan.deltas.delta_distance_km.toFixed(1)} km)
                        </span>
                      )}
                    </div>

                    <div className="bg-muted px-2 py-0.5 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Đã gán:</span>
                      <strong>{previewPlan.candidate_kpi.dispatched_orders} đơn</strong>
                      {previewPlan.deltas && (
                        <span className="text-teal-600 ml-1 text-xs font-bold">
                          (+{previewPlan.deltas.delta_orders_dispatched})
                        </span>
                      )}
                    </div>

                    <div className="bg-muted px-2 py-0.5 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Đội xe:</span>
                      <strong>{previewPlan.candidate_kpi.active_shippers}/{totalShippersCount}</strong>
                    </div>

                    <div className="bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                      <span className="text-muted-foreground mr-1">SLA:</span>
                      <strong className="text-emerald-600 dark:text-emerald-400">
                        {(previewPlan.candidate_kpi.sla_on_time_rate * 100).toFixed(1)}%
                      </strong>
                      {previewPlan.deltas && previewPlan.deltas.delta_sla_rate !== undefined && (
                        <span className={`ml-1 text-xs font-bold ${(previewPlan.deltas.delta_sla_rate ?? 0) >= 0 ? "text-emerald-600" : "text-amber-600"}`}>
                          ({(previewPlan.deltas.delta_sla_rate ?? 0) >= 0 ? "+" : ""}{(((previewPlan.deltas.delta_sla_rate ?? 0) * 100)).toFixed(1)}%)
                        </span>
                      )}
                    </div>

                    <div className="bg-muted px-2 py-0.5 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Chi phí:</span>
                      <strong>{previewPlan.candidate_kpi.estimated_cost.toLocaleString()} đ</strong>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    id="btnCancelPreview"
                    onClick={handleCancelActivePreview}
                    className="flex h-8.5 items-center gap-1.5 rounded-lg border border-border px-3 text-xs font-semibold text-muted-foreground hover:bg-muted cursor-pointer transition-colors"
                  >
                    <X className="size-3.5" />
                    <span>Hủy</span>
                  </button>

                  <button
                    type="button"
                    id="btnConfirmPreview"
                    onClick={handleConfirmActivePreview}
                    disabled={confirmLoading}
                    className="flex h-8.5 items-center gap-1.5 rounded-lg bg-emerald-600 px-3.5 text-xs font-bold text-white shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
                  >
                    <CheckCheck className={`size-3.5 ${confirmLoading ? "animate-spin" : ""}`} />
                    <span>{confirmLoading ? "Đang áp dụng…" : "Xác nhận áp dụng kế hoạch"}</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ──────────────────────────────────────────────────────────── */}
        {/* RIGHT COLUMN: FLEET & ASSIGNMENT PANEL (Fixed 290px)         */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="w-full lg:w-[290px] xl:w-[310px] shrink-0 order-3 lg:order-3 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs h-[480px] sm:h-[520px] lg:h-[500px] xl:h-[520px]">
          {/* Selected Shipper Hero Card */}
          <div className="p-2.5 border-b border-border/70 bg-muted/20 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex size-7 items-center justify-center rounded-md bg-teal-600 text-white font-mono text-xs font-bold">
                  {selectedShipper ? selectedShipper.shipper_id.replace("shipper_", "") : "01"}
                </div>
                <div>
                  <strong className="text-xs text-foreground block font-bold">
                    {selectedShipper ? selectedShipper.name : "Shipper #01"}
                  </strong>
                  <div className="flex flex-wrap items-center gap-1.5 mt-0.5">
                    <span className="text-[11px] text-muted-foreground">
                      Cụm #{selectedShipper?.home_cluster_id ?? 0}
                    </span>
                    <span className="text-[11px] text-muted-foreground">•</span>
                    {selectedShipper?.shift_start_mode === "START_AT_BASE" ? (
                      <span className="rounded bg-purple-500/10 px-1.5 py-0.2 text-[10px] font-semibold text-purple-600 dark:text-purple-400 border border-purple-500/20">
                        {selectedShipper.base_name || "Trạm vệ tinh"}
                      </span>
                    ) : (
                      <span className="rounded bg-sky-500/10 px-1.5 py-0.2 text-[10px] font-semibold text-sky-600 dark:text-sky-400 border border-sky-500/20">
                        Kho chính
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <span className="text-xs font-mono font-bold text-foreground">
                {shipperCurrentLoad.toFixed(1)} / {shipperMaxCap} kg
              </span>
            </div>

            {/* Capacity Bar */}
            <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full transition-all ${
                  isOverweight ? "bg-rose-500" : "bg-teal-600"
                }`}
                style={{
                  width: `${Math.min(100, (projectedLoad / shipperMaxCap) * 100)}%`,
                }}
              />
            </div>

            {/* Selected Orders Assignment Action */}
            {selectedOrders.length > 0 && (
              <div className="space-y-1 pt-1">
                <button
                  id="btnAssignToShipper"
                  type="button"
                  onClick={() => { onAssign(selectedShipperId, selectedOrderIds); setSelectedOrderIds([]); }}
                  disabled={!canAssign || actionLoading}
                  className={`w-full h-8 flex items-center justify-center gap-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    canAssign
                      ? "bg-teal-600 text-white hover:bg-teal-700 shadow-xs"
                      : "bg-muted text-muted-foreground opacity-60 cursor-not-allowed"
                  }`}
                >
                  <Check className="size-3.5" />
                  <span>
                    Gán {selectedOrders.length} đơn ({selectedWeight.toFixed(1)} kg)
                  </span>
                </button>

                {isOverweight && (
                  <p className="text-[11px] text-rose-500 font-medium">
                    Vượt tải trọng {shipperMaxCap} kg (+{(projectedLoad - shipperMaxCap).toFixed(1)} kg).
                  </p>
                )}
              </div>
            )}

            {/* Switch to Live simulation */}
            {selectedShipper && selectedShipper.assigned_order_ids.length > 0 && selectedOrders.length === 0 && (
              <button
                id="btnSwitchShipperToLive"
                type="button"
                onClick={onSwitchToLive}
                className="w-full h-7.5 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition-all cursor-pointer shadow-xs mt-0.5"
              >
                <Play className="size-3" />
                <span>Theo dõi ({selectedShipper.assigned_order_ids.length} đơn) →</span>
              </button>
            )}
          </div>

          {/* Fleet Filter Header */}
          <div className="px-2.5 py-1.5 border-b border-border/60 bg-muted/30 flex items-center justify-between text-xs">
            <span className="font-bold uppercase tracking-wider text-xs text-muted-foreground">
              Đội xe ({totalShippersCount})
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setFleetFilter("all")}
                className={`px-2 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "all" ? "bg-teal-600 text-white" : "text-muted-foreground hover:bg-muted"}`}
              >
                Tất cả ({totalShippersCount})
              </button>
              <button
                type="button"
                onClick={() => setFleetFilter("assigned")}
                className={`px-2 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "assigned" ? "bg-teal-600 text-white" : "text-muted-foreground hover:bg-muted"}`}
              >
                Đã gán ({activeShippersCount})
              </button>
              <button
                type="button"
                onClick={() => setFleetFilter("idle")}
                className={`px-2 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "idle" ? "bg-teal-600 text-white" : "text-muted-foreground hover:bg-muted"}`}
              >
                Trống ({totalShippersCount - activeShippersCount})
              </button>
            </div>
          </div>

          {/* Full Fleet List */}
          <div className="flex-1 min-h-0 overflow-y-auto divide-y divide-border/50 p-1">
            {filteredShippers.map((s) => {
              const isSelected = s.shipper_id === selectedShipperId;
              const sNum = s.shipper_id.replace("shipper_", "");
              const isIdle = s.assigned_order_ids.length === 0;
              const loadPct = Math.min(100, (s.current_load_kg / s.capacity_kg) * 100);

              return (
                <div
                  key={s.shipper_id}
                  onClick={() => onSelectShipper(s.shipper_id)}
                  className={`group flex items-center justify-between p-2 rounded-lg transition-colors cursor-pointer text-xs ${
                    isSelected ? "bg-teal-500/10 border-l-3 border-teal-600" : "hover:bg-muted/50"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="flex size-5.5 items-center justify-center rounded-full bg-muted font-mono text-xs font-bold text-foreground">
                      {sNum}
                    </span>
                    <div>
                      <strong className="text-foreground block text-xs leading-tight font-bold">
                        Shipper #{sNum}
                      </strong>
                      <span className="text-xs text-muted-foreground">
                        {isIdle ? "Sẵn sàng" : `${s.assigned_order_ids.length} đơn`} · Cụm #{s.home_cluster_id}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    {isIdle ? (
                      <span className="inline-block px-1.5 py-0.5 rounded bg-muted/80 text-xs font-mono text-muted-foreground">
                        0/100 kg
                      </span>
                    ) : (
                      <>
                        <span className="font-mono text-xs font-semibold text-foreground">
                          {s.current_load_kg.toFixed(0)} kg
                        </span>
                        <div className="w-10 h-1 rounded-full bg-muted mt-0.5 overflow-hidden ml-auto">
                          <div
                            className="h-full bg-teal-600"
                            style={{ width: `${loadPct}%` }}
                          />
                        </div>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Plan Completion Ready Trigger */}
          {(isPlanCompleted || activeShippersCount > 0) && (
            <div className="p-2.5 border-t border-border/80 bg-emerald-500/10 space-y-1.5">
              <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold text-xs">
                <CheckCircle2 className="size-3.5" />
                <span>
                  {isPlanCompleted
                    ? "100% Đơn hàng đã phân công"
                    : `${session.order_summary.assigned} đơn đã gán (${activeShippersCount}/${totalShippersCount} shipper)`}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground">
                Đội xe đã sẵn sàng thực hiện tuyến giao trong ngày.
              </p>
              <button
                id="btnStartLiveDispatch"
                type="button"
                onClick={onSwitchToLive}
                className="w-full h-7.5 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
              >
                <Play className="size-3" />
                <span>Bắt đầu mô phỏng thực địa →</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Explainable RL Decision Explorer Modal */}
      <RlDecisionExplorerModal
        isOpen={rlExplorerOpen}
        onClose={() => setRlExplorerOpen(false)}
        sessionId={session.session_id}
        selectedShipperId={selectedShipperId}
      />

      <DepotServiceAreaModal
        isOpen={depotModalOpen}
        onClose={() => setDepotModalOpen(false)}
        session={session}
        onSessionUpdated={async () => {
          if (onReloadSession) await onReloadSession();
        }}
        onFocusCoordinates={(coords: [number, number]) => setMapFocusPoint(coords)}
      />
      <NewScenarioModal
        isOpen={newScenarioModalOpen}
        onClose={() => setNewScenarioModalOpen(false)}
        onConfirm={async (days) => {
          if (onCreateNewSession) {
            await onCreateNewSession(days);
          }
        }}
        loading={actionLoading}
      />
    </div>
  );
}
