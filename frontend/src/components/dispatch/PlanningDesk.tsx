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
} from "lucide-react";
import L from "leaflet";
import { Marker, Polyline, Popup, CircleMarker, useMap } from "react-leaflet";
import { MapBase, depotIcon } from "@/components/MapBase";
import { RlDecisionExplorerModal } from "./RlDecisionExplorerModal";
import { DepotServiceAreaModal } from "./DepotServiceAreaModal";
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
  const [legendOpen, setLegendOpen] = useState<boolean>(true);
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

  // Plan summary metrics
  const isPlanCompleted = dispatchableOrdersCount === 0 && session.order_summary.assigned > 0;

  const isButtonsDisabled = previewLoading || actionLoading || dispatchableOrdersCount === 0 || session.is_day_ended;

  const previewButtonTooltip = useMemo(() => {
    if (session.is_day_ended) return `Ca làm việc Ngày ${session.current_day} đã kết thúc. Vui lòng chuyển ngày tiếp theo để tiếp tục phân công.`;
    if (dispatchableOrdersCount === 0) return `Toàn bộ ${session.orders.length} đơn hàng đã được phân công thành công vào đội xe. Không còn đơn chờ phân công trong ca.`;
    if (previewPlan) return "Đang xem trước phương án. Bấm để tính toán cập nhật lại.";
    return "Xem trước kết quả phân công và đối sánh KPI trước khi lưu chính thức";
  }, [session.is_day_ended, session.current_day, dispatchableOrdersCount, session.orders.length, previewPlan]);

  const directDispatchTooltip = useMemo(() => {
    if (session.is_day_ended) return `Ca làm việc Ngày ${session.current_day} đã kết thúc. Vui lòng chuyển ngày tiếp theo.`;
    if (dispatchableOrdersCount === 0) return `Toàn bộ ${session.orders.length} đơn hàng đã được phân công. Không còn đơn chờ phân công.`;
    return "Tự động phân bổ và áp dụng trực tiếp kế hoạch điều phối vào đội xe";
  }, [session.is_day_ended, session.current_day, dispatchableOrdersCount, session.orders.length]);
  const totalShippersCount = session.shippers.length;
  const totalFleetCapacityKg = useMemo(() => {
    return session.shippers.reduce((sum, s) => sum + (s.capacity_kg || 100), 0);
  }, [session.shippers]);
  const carriedOverOrdersCount = useMemo(() => {
    return session.orders.filter((o) => o.delivery_status === "CARRIED_OVER" || (o.carryover_count && o.carryover_count > 0)).length;
  }, [session.orders]);
  const newOrdersCount = session.orders.length - carriedOverOrdersCount;

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
    return session.orders.filter((o) => o.release_day <= session.current_day);
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
    <div className="flex flex-col gap-3 min-h-[calc(100vh-140px)] w-full">
      {/* ============================================================== */}
      {/* 1. TOP LOGISTICS COMMAND BAR & WORKFLOW PIPELINE */}
      {/* ============================================================== */}
      <div className="flex flex-col gap-3 rounded-2xl border border-border bg-card p-3.5 shadow-xs">
        {/* ROW 1: Context Header & RL Provenance */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-3">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20 shrink-0 shadow-2xs">
              <Compass className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-extrabold text-base sm:text-lg tracking-tight text-foreground">
                  Bàn Lập Kế Hoạch Ca Giao
                </h2>
                <span className="rounded-md bg-primary/15 border border-primary/30 px-2 py-0.5 font-mono text-xs font-bold text-primary">
                  Ngày {session.current_day}/{session.total_days}
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Tối ưu phân công {session.orders.length} đơn cho {totalShippersCount} shipper theo vị trí thực địa & DQN candidate scoring.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 font-mono text-xs font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-500/15 px-3 py-1 rounded-full border border-emerald-500/30 shadow-2xs">
              <Cpu className="size-3.5" />
              <span>RL: D_moderate_sla_best.pt</span>
            </span>
          </div>
        </div>

        {/* ROW 2: Workflow Progression Stepper (High-Contrast 5-Step Pipeline) */}
        <div className="flex flex-wrap items-center justify-between gap-2 bg-muted/40 p-2 rounded-xl border border-border/60">
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground mr-1 flex items-center gap-1">
              <Zap className="size-3.5 text-primary" />
              <span>Quy trình:</span>
            </span>
            <span className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs transition-colors border ${!previewPlan && !isPlanCompleted ? "bg-primary text-primary-foreground font-bold shadow-xs border-primary" : "bg-card text-foreground/80 border-border font-medium"}`}>
              <span className="flex size-4 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">1</span>
              <span>Cấu hình ca</span>
            </span>
            <span className="text-muted-foreground/50 font-bold">→</span>
            <span className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs transition-colors border ${previewPlan ? "bg-cyan-600 text-white font-bold shadow-xs border-cyan-600" : "bg-card text-foreground/80 border-border font-medium"}`}>
              <span className="flex size-4 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">2</span>
              <span>Xem trước phương án</span>
            </span>
            <span className="text-muted-foreground/50 font-bold">→</span>
            <span className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs transition-colors border ${previewPlan ? "bg-cyan-500/20 text-cyan-800 dark:text-cyan-200 font-bold border-cyan-500/40" : "bg-card text-foreground/80 border-border font-medium"}`}>
              <span className="flex size-4 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">3</span>
              <span>Đối sánh KPI & Delta</span>
            </span>
            <span className="text-muted-foreground/50 font-bold">→</span>
            <span className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs transition-colors border ${isPlanCompleted ? "bg-emerald-600 text-white font-bold shadow-xs border-emerald-600" : "bg-card text-foreground/80 border-border font-medium"}`}>
              <span className="flex size-4 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">4</span>
              <span>Xác nhận kế hoạch</span>
            </span>
            <span className="text-muted-foreground/50 font-bold">→</span>
            <span className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium bg-card text-foreground/80 border border-border">
              <span className="flex size-4 items-center justify-center rounded-full bg-black/15 text-[10px] font-bold">5</span>
              <span>Vận hành thực địa</span>
            </span>
          </div>

          <div className="text-xs text-muted-foreground hidden xl:block font-medium">
            <span>Mô phỏng: <strong>25 km/h</strong> • Khung ca: <strong>10 giờ</strong></span>
          </div>
        </div>

        {/* ROW 3: Strategy & Algorithm Selectors + Action Toolbar */}
        <div className="flex flex-wrap items-end justify-between gap-3 pt-1">
          {/* Left: Configuration Selectors */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Strategy Dropdown */}
            <div className="flex flex-col">
              <label htmlFor="assignmentStrategy" className="text-xs font-bold uppercase tracking-wider text-muted-foreground px-0.5 pb-1">
                Chiến lược phân công
              </label>
              <select
                id="assignmentStrategy"
                className="h-9.5 w-full sm:min-w-[280px] sm:max-w-[340px] rounded-lg border border-border bg-background px-3 text-sm font-semibold text-foreground focus:outline-hidden focus:ring-2 focus:ring-primary/40 cursor-pointer shadow-xs"
                value={selectedStrategy}
                onChange={(e) => setSelectedStrategy(e.target.value)}
                disabled={actionLoading || previewLoading}
              >
                <option value="depot_zone_capacity_aware">Kho, khu vực & tải trọng (Depot-Aware)</option>
                <option value="cluster_aware_rl">Ưu tiên cụm + chi phí (Cluster-Aware Min Cost)</option>
                <option value="location_aware_min_cost">Tối ưu cự ly chèn (Min Incremental Detour)</option>
                <option value="location_aware_nearest">Shipper gần nhất (Nearest Shipper)</option>
                <option value="cluster_round_robin">Phân cụm Round-Robin (K-Means)</option>
              </select>
            </div>

            {/* Algorithm Dropdown */}
            <div className="flex flex-col">
              <label htmlFor="routingAlgorithm" className="text-xs font-bold uppercase tracking-wider text-muted-foreground px-0.5 pb-1">
                Thuật toán định tuyến
              </label>
              <select
                id="routingAlgorithm"
                className="h-9.5 w-full sm:min-w-[260px] sm:max-w-[320px] rounded-lg border border-border bg-background px-3 text-sm font-semibold text-foreground focus:outline-hidden focus:ring-2 focus:ring-primary/40 cursor-pointer shadow-xs"
                value={selectedAlgorithm}
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
                disabled={actionLoading || previewLoading}
              >
                <option value="proposed_rl">Proposed DQN (Candidate-Scoring DQN)</option>
                <option value="nearest_neighbor">Nearest Neighbor (Tham lam cự ly)</option>
                <option value="clarke_wright">Clarke–Wright (Heuristic tiết kiệm)</option>
              </select>
            </div>
          </div>

          {/* Right: Action Buttons Toolbar */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              id="btnOpenDepotConfig"
              onClick={() => setDepotModalOpen(true)}
              className="flex h-9.5 items-center gap-2 rounded-lg border border-border bg-card px-3.5 text-xs font-bold text-foreground hover:bg-muted transition-all cursor-pointer shadow-2xs"
              title="Cấu hình vị trí kho trung tâm và khu vực phục vụ của shipper"
            >
              <Warehouse className="size-4 text-primary" />
              <span>Kho & Điểm trực ca</span>
            </button>

            <button
              type="button"
              id="btnOpenRlExplorer"
              onClick={() => setRlExplorerOpen(true)}
              className="flex h-9.5 items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3.5 text-xs font-bold text-primary hover:bg-primary/20 transition-all cursor-pointer shadow-2xs"
              title="Mở bộ giải thích vết quyết định mạng nơ-ron Candidate-Scoring DQN"
            >
              <BrainCircuit className="size-4 text-primary" />
              <span>RL Decision Explorer</span>
            </button>

            <button
              type="button"
              id="btnPreviewPlan"
              onClick={handleTriggerPreview}
              disabled={isButtonsDisabled}
              className="flex h-9.5 items-center gap-2 rounded-lg border border-cyan-500/50 bg-cyan-500/15 px-4 text-xs font-bold text-cyan-800 dark:text-cyan-200 hover:bg-cyan-500/25 transition-all cursor-pointer disabled:opacity-50 shadow-xs"
              title={previewButtonTooltip}
            >
              <Eye className={`size-4 ${previewLoading ? "animate-spin" : ""}`} />
              <span>{previewLoading ? "Đang tính toán…" : (previewPlan ? "Cập nhật xem trước" : "Xem trước phương án")}</span>
            </button>

            <button
              type="button"
              id="btnDirectDispatch"
              onClick={() => onAutoDispatch(selectedStrategy, selectedAlgorithm)}
              disabled={isButtonsDisabled}
              className="flex h-9.5 items-center gap-2 rounded-lg bg-primary px-4 text-xs font-bold text-primary-foreground shadow-xs hover:bg-primary/90 transition-all cursor-pointer disabled:opacity-50"
              title={directDispatchTooltip}
            >
              <RefreshCw className={`size-4 ${actionLoading ? "animate-spin" : ""}`} />
              <span>Áp dụng ngay</span>
            </button>
          </div>
        </div>

        {/* ROW 4: Operational State Ribbon & KPI Badges (Including Total Load) */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-2.5 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-primary/10 px-2.5 py-1 font-medium text-primary border border-primary/20 shadow-2xs">
              <span>Đơn mới:</span>
              <strong className="font-mono font-bold">{newOrdersCount}</strong>
            </span>
            {carriedOverOrdersCount > 0 && (
              <span className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500/10 px-2.5 py-1 font-medium text-rose-700 dark:text-rose-400 border border-rose-500/20 shadow-2xs">
                <span>Tồn đầu ngày:</span>
                <strong className="font-mono font-bold">{carriedOverOrdersCount}</strong>
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-amber-500/10 px-2.5 py-1 font-medium text-amber-700 dark:text-amber-400 border border-amber-500/20 shadow-2xs">
              <span>Chưa gán:</span>
              <strong className="font-mono font-bold">{dispatchableOrdersCount}</strong>
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500/10 px-2.5 py-1 font-medium text-emerald-700 dark:text-emerald-400 border border-emerald-500/20 shadow-2xs">
              <span>Đã gán:</span>
              <strong className="font-mono font-bold">{session.order_summary.assigned}</strong>
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500/10 px-2.5 py-1 font-medium text-sky-700 dark:text-sky-400 border border-sky-500/20 shadow-2xs">
              <span>Đội xe trực:</span>
              <strong className="font-mono font-bold">{activeShippersCount}/{totalShippersCount}</strong>
            </span>
            {/* High-Contrast Clear Total Load Badge */}
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-card px-2.5 py-1 font-medium text-foreground border border-border shadow-2xs">
              <span className="text-muted-foreground font-medium">Tổng tải:</span>
              <strong className="font-mono text-foreground font-bold">{totalAssignedWeight.toFixed(1)} / {totalFleetCapacityKg.toLocaleString()} kg</strong>
            </span>
          </div>

          <div className="text-xs text-muted-foreground hidden md:block">
            {previewPlan ? (
              <span className="text-cyan-700 dark:text-cyan-300 font-bold flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-cyan-500 animate-pulse" />
                <span>Đang hiển thị bản xem trước. Hãy kiểm tra tuyến trên bản đồ và đối sánh KPI bên dưới.</span>
              </span>
            ) : (
              <span className="text-muted-foreground">Chọn chiến lược rồi bấm <strong>"Xem trước phương án"</strong> hoặc <strong>"Áp dụng ngay"</strong>.</span>
            )}
          </div>
        </div>
      </div>

      {/* 2. THREE-COLUMN MAP-FIRST WORKSPACE */}
      {/* ============================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 flex-1 min-h-[640px]">
        {/* ──────────────────────────────────────────────────────────── */}
        {/* LEFT COLUMN: ORDERS WORKLIST (~330px, 3/12 cols) */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="lg:col-span-4 xl:col-span-3 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs">
          {/* Worklist Header */}
          <div className="p-3 border-b border-border/80 bg-muted/20 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Package className="size-4 text-primary" />
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
                className="w-full h-7.5 rounded-lg border border-border/80 bg-background pl-8 pr-2 text-xs placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-primary"
              />
            </div>

            {/* Quick Filters: Status & Cluster */}
            <div className="grid grid-cols-2 gap-1.5">
              <select
                id="filterStatusDesk"
                className="h-8 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground cursor-pointer"
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
                className="h-8 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground cursor-pointer"
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
          <div className="flex-1 overflow-y-auto divide-y divide-border/60 p-1 max-h-[520px]">
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
                        ? "bg-primary/10 border-l-3 border-primary"
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
                      className="mt-0.5 size-3.5 rounded border-border text-primary cursor-pointer"
                    />

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono font-bold text-foreground truncate">
                          {o.order_code || o.order_id}
                        </span>
                        <span
                          className="shrink-0 text-xs font-bold px-2 py-0.5 rounded-full text-white"
                          style={{ backgroundColor: clusterColor(o.cluster_id) }}
                        >
                          Cụm {o.cluster_id}
                        </span>
                      </div>

                      <div className="text-xs text-muted-foreground truncate mt-0.5">
                        {o.package_name || "Kiện hàng tiêu chuẩn"}
                      </div>

                      <div className="flex items-center gap-2 mt-1 text-xs">
                        <span className="font-medium text-foreground">
                          {o.weight_kg.toFixed(1)} kg
                        </span>
                        <span>•</span>
                        {o.delivery_status === "ASSIGNED" ? (
                          <span className="text-primary font-semibold">
                            Shipper #{o.assigned_shipper_id?.replace("shipper_", "")}
                          </span>
                        ) : isCandidate && candidateInfo ? (
                          <span className="text-cyan-500 font-semibold">
                            Dự kiến → Shipper #{candidateInfo.candidate_shipper_id?.replace("shipper_", "")}
                          </span>
                        ) : (
                          <span className="text-amber-500 font-medium">Chưa phân công</span>
                        )}
                      </div>

                      {candidateInfo?.dispatch_reason && (
                        <div className="text-xs text-muted-foreground mt-1 bg-muted/50 px-2 py-1 rounded flex items-center justify-between">
                          <span className="truncate">Lý do: {candidateInfo.dispatch_reason}</span>
                          {candidateInfo.dispatch_distance_km !== undefined && candidateInfo.dispatch_distance_km !== null && (
                            <span className="font-mono text-xs font-bold shrink-0 ml-1">+{candidateInfo.dispatch_distance_km.toFixed(1)}km</span>
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
          <div className="p-2.5 border-t border-border/80 bg-muted/20 flex items-center justify-between text-xs">
            <button
              type="button"
              onClick={toggleSelectPage}
              className="text-xs font-semibold text-primary hover:underline cursor-pointer"
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
                className="h-7.5 px-2.5 rounded-md border border-border bg-background text-xs font-medium disabled:opacity-40 cursor-pointer"
              >
                ←
              </button>
              <button
                type="button"
                disabled={currentPage + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="h-7.5 px-2.5 rounded-md border border-border bg-background text-xs font-medium disabled:opacity-40 cursor-pointer"
              >
                →
              </button>
            </div>
          </div>
        </div>

        {/* ──────────────────────────────────────────────────────────── */}
        {/* CENTER COLUMN: GIS MAP & PREVIEW DELTA BAR (~6/12 cols) */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="lg:col-span-5 xl:col-span-6 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs relative">
          {/* Floating Map Layer Toggles */}
          <div className="absolute top-3 right-3 z-30 flex items-center gap-1 bg-card/90 p-1 rounded-lg border border-border/80 shadow-md backdrop-blur-md text-xs">
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, orders: !prev.orders }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.orders ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Đơn ({todayEligibleOrders.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveLayers((prev) => ({ ...prev, routes: !prev.routes }))}
              className={`px-2 py-0.5 rounded font-medium transition-colors cursor-pointer ${
                activeLayers.routes ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
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
          <div className="flex-1 w-full min-h-[460px] relative">
            <MapBase height="100%">
              <MapController focusPoint={mapFocusPoint} />

              {/* Central Depot Marker */}
              {activeLayers.depot && (
                <Marker position={[session.depot.lat, session.depot.lng]} icon={depotIcon()}>
                  <Popup>
                    <div className="text-xs space-y-1">
                      <strong className="text-primary block font-bold">KHO TRUNG TÂM (CENTRAL HUB)</strong>
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

              {/* Depot Fleet Hub Marker: resolves visual overlap of 16 shippers at start */}
              {shippersAtDepot.length > 0 && (
                <Marker
                  position={[session.depot.lat, session.depot.lng]}
                  icon={createDepotFleetIcon(shippersAtDepot.length)}
                >
                  <Popup>
                    <div className="text-xs space-y-1.5 max-w-[200px]">
                      <strong className="block text-primary">
                        {shippersAtDepot.length} Shipper đang tại Depot
                      </strong>
                      <div className="max-h-36 overflow-y-auto divide-y divide-border text-xs">
                        {shippersAtDepot.map((s) => (
                          <div
                            key={s.shipper_id}
                            onClick={() => onSelectShipper(s.shipper_id)}
                            className="py-1 flex items-center justify-between cursor-pointer hover:text-primary"
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

              {/* Shippers currently in transit away from depot */}
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
                            <div className="text-primary font-semibold">
                              Gán cho: {o.assigned_shipper_id}
                            </div>
                          )}
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}

              {/* Confirmed Route Polyline for Selected Shipper */}
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

              {/* Candidate Preview Route Polyline (Dashed Cyan/Emerald) */}
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
                className="flex items-center justify-between px-3 py-2 bg-muted/40 hover:bg-muted/60 cursor-pointer border-b border-border/60 select-none"
              >
                <div className="flex items-center gap-1.5 font-bold text-xs text-foreground">
                  <Layers className="size-3.5 text-primary" />
                  <span>Chú giải Cụm & Tuyến GIS</span>
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
                <div className="p-2.5 space-y-2.5 max-h-56 overflow-y-auto text-xs">
                  {/* Cluster distribution */}
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

                  {/* Route styles */}
                  <div className="pt-2 border-t border-border/60">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-1">
                      Ký hiệu Tuyến đường
                    </span>
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-1 bg-[#0f766e] rounded-full inline-block shrink-0" />
                        <span className="text-xs font-medium text-foreground">Tuyến chính thức đã gán</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-0.5 border-b-2 border-dashed border-[#06b6d4] inline-block shrink-0" />
                        <span className="text-xs font-medium text-foreground">Tuyến dự kiến xem trước (Preview)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="size-3.5 rounded-full bg-[#1e293b] border border-[#38bdf8] flex items-center justify-center text-xs font-bold text-white shrink-0">
                          D
                        </div>
                        <span className="text-xs font-medium text-foreground">Depot trung tâm (Kho tập kết)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="size-4 rounded-full bg-[#f97316] text-xs font-bold text-white flex items-center justify-center shrink-0">
                          1
                        </div>
                        <span className="text-xs font-medium text-foreground">Thứ tự điểm dừng trên lộ trình</span>
                      </div>
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
                {/* Preview Metadata & Deltas */}
                <div className="flex flex-wrap items-center gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="size-2.5 rounded-full bg-cyan-500 animate-pulse" />
                    <div>
                      <span className="font-bold text-foreground block">
                        Bản xem trước phương án (Preview)
                      </span>
                      <div className="flex flex-wrap items-center gap-1.5 text-xs">
                        <span className="text-xs text-muted-foreground">
                          Chiến lược: <strong className="text-foreground font-mono font-semibold">{formatStrategyName(previewPlan.strategy_used)}</strong>
                          {" · "}
                          Thuật toán: <strong className="text-foreground font-mono font-semibold">{formatAlgorithmName(previewPlan.algorithm_used)}</strong>
                        </span>
                        {previewPlan.is_fallback && (
                          <span
                            className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 text-[11px] font-semibold"
                            title={previewPlan.fallback_reason || "Đã kích hoạt dự phòng sang Nearest Neighbor"}
                          >
                            ⚠️ Dự phòng: Nearest Neighbor
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5 font-mono text-xs">
                    <div className="bg-cyan-500/10 px-2 py-1 rounded border border-cyan-500/20">
                      <span className="text-muted-foreground mr-1">Cự ly:</span>
                      <strong className="text-cyan-600 dark:text-cyan-400">{previewPlan.candidate_kpi.total_distance_km} km</strong>
                      {previewPlan.deltas && (
                        <span className={`ml-1 text-xs font-bold ${previewPlan.deltas.delta_distance_km <= 0 ? 'text-emerald-600' : 'text-amber-600'}`}>
                          ({previewPlan.deltas.delta_distance_km >= 0 ? "+" : ""}{previewPlan.deltas.delta_distance_km.toFixed(1)} km)
                        </span>
                      )}
                    </div>

                    <div className="bg-muted px-2 py-1 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Đã gán:</span>
                      <strong>{previewPlan.candidate_kpi.dispatched_orders} đơn</strong>
                      {previewPlan.deltas && (
                        <span className="text-primary ml-1 text-xs font-bold">
                          (+{previewPlan.deltas.delta_orders_dispatched})
                        </span>
                      )}
                    </div>

                    <div className="bg-muted px-2 py-1 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Đội xe:</span>
                      <strong>{previewPlan.candidate_kpi.active_shippers}/{totalShippersCount}</strong>
                      {previewPlan.deltas && (
                        <span className="text-muted-foreground ml-1 text-xs">
                          (+{previewPlan.deltas.delta_active_shippers})
                        </span>
                      )}
                    </div>

                    <div className="bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20">
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

                    <div className="bg-muted px-2 py-1 rounded border border-border/60">
                      <span className="text-muted-foreground mr-1">Chi phí:</span>
                      <strong>{previewPlan.candidate_kpi.estimated_cost.toLocaleString()} đ</strong>
                      {previewPlan.deltas && (
                        <span className="text-muted-foreground ml-1 text-xs">
                          ({previewPlan.deltas.delta_cost >= 0 ? "+" : ""}{previewPlan.deltas.delta_cost.toLocaleString()} đ)
                        </span>
                      )}
                    </div>

                    {!previewPlan.deltas && (
                      <span className="bg-amber-500/10 text-amber-700 dark:text-amber-400 px-2 py-1 rounded border border-amber-500/20 text-xs font-medium">
                        Chưa đủ dữ liệu đối chứng (Baseline trống)
                      </span>
                    )}
                  </div>
                </div>

                {/* Confirm & Cancel Actions */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    id="btnCancelPreview"
                    onClick={handleCancelActivePreview}
                    className="flex h-9 items-center gap-2 rounded-lg border border-border px-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted cursor-pointer transition-colors"
                  >
                    <X className="size-3.5" />
                    <span>Hủy</span>
                  </button>

                  <button
                    type="button"
                    id="btnConfirmPreview"
                    onClick={handleConfirmActivePreview}
                    disabled={confirmLoading}
                    className="flex h-9 items-center gap-2 rounded-lg bg-emerald-600 px-4 text-xs font-bold text-white shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
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
        {/* RIGHT COLUMN: FLEET & ASSIGNMENT PANEL (~320px, 3/12 cols) */}
        {/* ──────────────────────────────────────────────────────────── */}
        <div className="lg:col-span-3 xl:col-span-3 flex flex-col rounded-xl border border-border/80 bg-card overflow-hidden shadow-xs">
          {/* Selected Shipper Hero Card */}
          <div className="p-3 border-b border-border/80 bg-muted/20 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground font-mono text-xs font-bold">
                  {selectedShipper ? selectedShipper.shipper_id.replace("shipper_", "") : "01"}
                </div>
                <div>
                  <strong className="text-xs text-foreground block">
                    {selectedShipper ? selectedShipper.name : "Shipper #01"}
                  </strong>
                  <div className="flex flex-wrap items-center gap-1.5 mt-0.5">
                    <span className="text-xs text-muted-foreground">
                      Cụm #{selectedShipper?.home_cluster_id ?? 0}
                    </span>
                    <span className="text-xs text-muted-foreground">•</span>
                    {selectedShipper?.shift_start_mode === "START_AT_BASE" ? (
                      <span className="rounded bg-purple-500/10 px-1.5 py-0.2 text-xs font-semibold text-purple-600 dark:text-purple-400 border border-purple-500/20">
                        Trạm: {selectedShipper.base_name || "Vệ tinh"}
                      </span>
                    ) : (
                      <span className="rounded bg-sky-500/10 px-1.5 py-0.2 text-xs font-semibold text-sky-600 dark:text-sky-400 border border-sky-500/20">
                        Xuất phát: Kho chính
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
            <div className="w-full h-2 rounded-full bg-muted overflow-hidden">
              <div
                className={`h-full transition-all ${
                  isOverweight ? "bg-rose-500" : "bg-primary"
                }`}
                style={{
                  width: `${Math.min(100, (projectedLoad / shipperMaxCap) * 100)}%`,
                }}
              />
            </div>

            {/* Mode B Logistics Notice */}
            {selectedShipper?.shift_start_mode === "START_AT_BASE" && (
              <div className="rounded-lg bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800/50 p-2.5 text-xs text-purple-700 dark:text-purple-300 space-y-0.5">
                <div className="font-semibold flex items-center gap-1">
                  <span>📍 Xuất phát từ trạm trực ca</span>
                </div>
                <div>Shipper khởi hành từ <strong>{selectedShipper.base_name}</strong>, di chuyển đến Kho trung tâm nhận kiện trước khi giao hàng.</div>
              </div>
            )}

            {/* Empty state guidance when shipper has 0 assigned orders */}
            {selectedShipper && selectedShipper.assigned_order_ids.length === 0 && selectedOrders.length === 0 && (
              <div className="rounded-lg bg-background/80 border border-border/60 p-2 text-xs text-muted-foreground space-y-1">
                <div className="flex items-center gap-1.5 font-semibold text-foreground">
                  <Info className="size-3 text-primary" />
                  <span>Sẵn sàng nhận tải tại Depot</span>
                </div>
                <p className="text-xs leading-relaxed">
                  Chưa có đơn gán cho Shipper này. Chọn đơn từ danh sách bên trái hoặc bấm <strong>"Xem trước phương án"</strong> để hệ thống tự động tối ưu hóa.
                </p>
              </div>
            )}

            {/* Selected Orders Assignment Action */}
            {selectedOrders.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <button
                  id="btnAssignToShipper"
                  type="button"
                  onClick={() => { onAssign(selectedShipperId, selectedOrderIds); setSelectedOrderIds([]); }}
                  disabled={!canAssign || actionLoading}
                  className={`w-full h-8 flex items-center justify-center gap-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    canAssign
                      ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-xs"
                      : "bg-muted text-muted-foreground opacity-60 cursor-not-allowed"
                  }`}
                >
                  <Check className="size-3.5" />
                  <span>
                    Gán {selectedOrders.length} đơn ({selectedWeight.toFixed(1)} kg)
                  </span>
                </button>

                {isOverweight && (
                  <p className="text-xs text-rose-500 font-medium">
                    Vượt tải trọng {shipperMaxCap} kg (+{(projectedLoad - shipperMaxCap).toFixed(1)} kg).
                  </p>
                )}
              </div>
            )}

            {/* Quick Switch to Live Operations for this Shipper */}
            {selectedShipper && selectedShipper.assigned_order_ids.length > 0 && selectedOrders.length === 0 && (
              <button
                id="btnSwitchShipperToLive"
                type="button"
                onClick={onSwitchToLive}
                className="w-full h-8 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition-all cursor-pointer shadow-xs mt-1"
              >
                <Play className="size-3.5" />
                <span>Theo dõi giao hàng shipper này ({selectedShipper.assigned_order_ids.length} đơn) →</span>
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
                className={`px-1.5 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "all" ? "bg-primary text-primary-foreground font-semibold" : "text-muted-foreground hover:bg-muted"}`}
              >
                Tất cả ({totalShippersCount})
              </button>
              <button
                type="button"
                onClick={() => setFleetFilter("assigned")}
                className={`px-1.5 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "assigned" ? "bg-primary text-primary-foreground font-semibold" : "text-muted-foreground hover:bg-muted"}`}
              >
                Đã gán ({activeShippersCount})
              </button>
              <button
                type="button"
                onClick={() => setFleetFilter("idle")}
                className={`px-1.5 py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${fleetFilter === "idle" ? "bg-primary text-primary-foreground font-semibold" : "text-muted-foreground hover:bg-muted"}`}
              >
                Trống ({totalShippersCount - activeShippersCount})
              </button>
            </div>
          </div>

          {/* Full Fleet List of 16 Shippers (Filterable) */}
          <div className="flex-1 overflow-y-auto divide-y divide-border/60 p-1 max-h-[440px]">
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
                    isSelected ? "bg-primary/10 border-l-3 border-primary" : "hover:bg-muted/50"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="flex size-6 items-center justify-center rounded-full bg-muted font-mono text-xs font-bold text-foreground">
                      {sNum}
                    </span>
                    <div>
                      <strong className="text-foreground block text-xs leading-tight font-bold">
                        Shipper #{sNum}
                      </strong>
                      <span className="text-xs text-muted-foreground">
                        {isIdle ? "Sẵn sàng nhận ca" : `${s.assigned_order_ids.length} đơn`} · Cụm #{s.home_cluster_id}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    {isIdle ? (
                      <span className="inline-block px-2 py-0.5 rounded bg-muted/80 text-xs font-mono font-medium text-muted-foreground">
                        0/100 kg
                      </span>
                    ) : (
                      <>
                        <span className="font-mono text-xs font-semibold text-foreground">
                          {s.current_load_kg.toFixed(0)} kg
                        </span>
                        <div className="w-12 h-1 rounded-full bg-muted mt-1 overflow-hidden ml-auto">
                          <div
                            className="h-full bg-primary"
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
            <div className="p-3 border-t border-border/80 bg-emerald-500/10 space-y-2">
              <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold text-xs">
                <CheckCircle2 className="size-4" />
                <span>
                  {isPlanCompleted
                    ? "100% Đơn hàng đã phân công"
                    : `${session.order_summary.assigned} đơn đã gán (${activeShippersCount}/${totalShippersCount} shipper)`}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Đội xe đã sẵn sàng thực hiện tuyến giao. Bạn có thể chuyển sang phân hệ Theo dõi giao hàng để bắt đầu ca.
              </p>
              <button
                id="btnStartLiveDispatch"
                type="button"
                onClick={onSwitchToLive}
                className="w-full h-8 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold shadow-xs hover:bg-emerald-700 transition-all cursor-pointer"
              >
                <Play className="size-3.5" />
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
    </div>
  );
}
