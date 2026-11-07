import React, { createContext, useContext, useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import {
  api,
  DispatchSessionState,
  NearbyOrderItem,
  PlanPreviewResponse,
} from "@/lib/api";

export interface DispatchContextType {
  session: DispatchSessionState | null;
  loading: boolean;
  connectionError: string | null;
  sessionNotFoundError: string | null;
  selectedShipperId: string;
  setSelectedShipperId: (id: string) => void;
  nearbyOrders: NearbyOrderItem[];
  activePreviewPlan: PlanPreviewResponse | null;
  setActivePreviewPlan: (plan: PlanPreviewResponse | null) => void;
  isPlaying: boolean;
  simSpeed: number;
  setSimSpeed: (speed: number) => void;
  actionLoading: boolean;
  actionError: string | null;
  actionSuccess: string | null;
  setActionError: (err: string | null) => void;
  setActionSuccess: (msg: string | null) => void;
  initSession: (forceNew?: boolean) => Promise<void>;
  createNewSession: (totalDays?: number) => Promise<void>;
  handleAssign: (shipperId: string, orderIds: string[]) => Promise<void>;
  handleUnassign: (orderIds: string[]) => Promise<void>;
  handleAutoDispatch: (strategy?: string, algorithm?: string) => Promise<void>;
  handlePreviewDispatch: (strategy: string, algorithm: string) => Promise<PlanPreviewResponse>;
  handleConfirmPreview: (previewId: string) => Promise<void>;
  handleCancelPreview: () => Promise<void>;
  handleSelectAlgorithm: (algo: string) => Promise<void>;
  handleStepClock: (deltaMinutes?: number) => Promise<void>;
  handleTogglePlay: () => void;
  handleEndDay: () => Promise<void>;
  handleNextDay: () => Promise<void>;
  refreshNearby: (shipperId: string, sessId?: string) => Promise<void>;
}

const DispatchContext = createContext<DispatchContextType | null>(null);

export function DispatchProvider({ children }: { children: React.ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();

  // Session & Global State
  const [session, setSession] = useState<DispatchSessionState | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [sessionNotFoundError, setSessionNotFoundError] = useState<string | null>(null);

  // Operational Context
  const [selectedShipperId, setSelectedShipperId] = useState<string>("shipper_01");
  const [nearbyOrders, setNearbyOrders] = useState<NearbyOrderItem[]>([]);
  const [activePreviewPlan, setActivePreviewPlan] = useState<PlanPreviewResponse | null>(null);

  // Simulation State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [simSpeed, setSimSpeed] = useState<number>(5);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const playTimerRef = useRef<any>(null);

  // Auto-dismiss notification toasts
  useEffect(() => {
    if (actionSuccess || actionError) {
      const timer = setTimeout(() => {
        setActionSuccess(null);
        setActionError(null);
      }, 4500);
      return () => clearTimeout(timer);
    }
  }, [actionSuccess, actionError]);

  // Initialize or reload dispatch session
  const initSession = async (forceNew = false) => {
    try {
      setLoading(true);
      setConnectionError(null);
      setSessionNotFoundError(null);
      setActionError(null);

      const querySessId = searchParams.get("session_id");
      const storedSessId = typeof window !== "undefined" ? localStorage.getItem("nexus_active_session_id") : null;
      const targetSessId = !forceNew ? (querySessId || storedSessId) : null;

      let sess: DispatchSessionState | null = null;
      if (targetSessId) {
        try {
          sess = await api.getDispatchSession(targetSessId);
        } catch (err: any) {
          console.warn("Phiên làm việc không tồn tại hoặc đã hết hạn (404):", targetSessId);
          if (typeof window !== "undefined") {
            localStorage.removeItem("nexus_active_session_id");
          }
          // Ping health to verify whether backend is alive or truly unreachable
          try {
            const health = await api.health();
            if (health && (health.status === "ready" || health.status === "healthy" || health.ready_for_frontend)) {
              console.info("Backend hoạt động bình thường. Tự động khởi tạo phiên làm việc mới...");
              // Clear stale session ID so it falls through to fresh creation
              sess = null;
            } else {
              setSessionNotFoundError(targetSessId);
              setLoading(false);
              return;
            }
          } catch (healthErr) {
            setConnectionError("Không thể kết nối đến Backend tại 127.0.0.1:8011. Vui lòng đảm bảo FastAPI đang chạy.");
            setLoading(false);
            return;
          }
        }
      }

      if (!sess) {
        sess = await api.createDispatchSession({
          scenario_id: "simulated_stress_day1_to_3",
          routing_algorithm: "proposed_rl",
          priority_policy: "sla_aware",
          total_days: 3,
          seed: 42,
        });
      }

      setSession(sess);
      if (typeof window !== "undefined") {
        localStorage.setItem("nexus_active_session_id", sess.session_id);
      }
      // If the URL already had an explicit session_id that differs (e.g. stale 404 recovered), update it cleanly
      const currentParam = searchParams.get("session_id");
      if (currentParam && currentParam !== sess.session_id && typeof window !== "undefined" && window.location.pathname !== "/") {
        setSearchParams((prev) => {
          const next = new URLSearchParams(prev);
          next.set("session_id", sess!.session_id);
          return next;
        }, { replace: true });
      }

      if (sess.shippers && sess.shippers.length > 0) {
        setSelectedShipperId(sess.shippers[0].shipper_id);
      }
    } catch (err: any) {
      setConnectionError(
        err.message ||
          "Không thể kết nối đến Backend tại 127.0.0.1:8011. Vui lòng đảm bảo FastAPI đang chạy."
      );
    } finally {
      setLoading(false);
    }
  };

  // Create a brand new scenario with explicit total_days
  const createNewSession = async (totalDays: number = 3) => {
    try {
      setLoading(true);
      setActionLoading(true);
      setActionError(null);
      const sess = await api.createDispatchSession({
        scenario_id: "simulated_stress_day1_to_3",
        routing_algorithm: "proposed_rl",
        priority_policy: "sla_aware",
        total_days: totalDays,
        seed: 42,
      });
      setSession(sess);
      setActivePreviewPlan(null);
      if (typeof window !== "undefined") {
        localStorage.setItem("nexus_active_session_id", sess.session_id);
      }
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("session_id", sess.session_id);
        return next;
      }, { replace: true });
      if (sess.shippers && sess.shippers.length > 0) {
        setSelectedShipperId(sess.shippers[0].shipper_id);
      }
      setActionSuccess(`Đã tạo kịch bản mới: ${totalDays} ngày (${totalDays * 85} đơn) - ID: ${sess.session_id}`);
    } catch (err: any) {
      setActionError(err.message || "Không thể khởi tạo phiên làm việc mới.");
    } finally {
      setLoading(false);
      setActionLoading(false);
    }
  };

  useEffect(() => {
    initSession();
  }, []);

  // Refresh nearby orders for selected shipper
  const refreshNearby = async (shipperId: string, sessId?: string) => {
    const sId = sessId || session?.session_id;
    if (!sId || !shipperId) return;
    try {
      const orders = await api.getNearbyOrders(sId, shipperId, 30);
      setNearbyOrders(orders);
    } catch (err) {
      console.warn("Could not refresh nearby orders", err);
    }
  };

  useEffect(() => {
    if (session && selectedShipperId) {
      refreshNearby(selectedShipperId);
    }
  }, [selectedShipperId, session?.session_id]);

  // Handle Simulation Loop
  useEffect(() => {
    if (isPlaying) {
      const intervalMs = Math.max(250, 1500 / simSpeed);
      playTimerRef.current = setInterval(async () => {
        try {
          if (!session) return;
          await api.stepSimulationClock(session.session_id, 15);
          const updated = await api.getDispatchSession(session.session_id);
          setSession(updated);

          // Stop if day ended
          if (updated.is_day_ended) {
            setIsPlaying(false);
            return;
          }

          // Stop if max overtime reached
          const maxClock = (updated.shift_minutes || 600) + (updated.max_overtime_minutes || 48);
          if (updated.simulation_clock >= maxClock || !updated.is_running) {
            setIsPlaying(false);
            return;
          }

          // Stop if all assigned orders delivered and shippers idle
          if (
            updated.order_summary.assigned === 0 &&
            updated.order_summary.delivered > 0 &&
            updated.fleet_summary.active_shippers === 0
          ) {
            setIsPlaying(false);
            return;
          }
        } catch (err: any) {
          setIsPlaying(false);
          setActionError(err.message || "Lỗi mô phỏng vòng lặp.");
        }
      }, intervalMs);
    } else {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    }
    return () => {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [isPlaying, simSpeed, session?.session_id]);

  const handleTogglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    } else {
      if (session?.is_day_ended) {
        setActionError("Ca làm việc hôm nay đã kết thúc. Vui lòng chuyển sang Ngày tiếp theo hoặc xem Báo cáo kết ca.");
        return;
      }
      setIsPlaying(true);
    }
  };

  // Action Handlers
  const handleAssign = async (shipperId: string, orderIds: string[]) => {
    if (!session || actionLoading || orderIds.length === 0) return;
    try {
      setActionLoading(true);
      setActionError(null);
      setActionSuccess(null);
      const res = await api.assignOrders(session.session_id, shipperId, orderIds);
      if (res.success) {
        const updated = await api.getDispatchSession(session.session_id);
        setSession(updated);
        await refreshNearby(shipperId, session.session_id);
        setActionSuccess(`Đã phân công thành công ${orderIds.length} đơn cho ${shipperId}.`);
      } else {
        setActionError("Phân công đơn hàng thất bại.");
      }
    } catch (err: any) {
      setActionError(err.message || "Không thể phân công đơn hàng.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleUnassign = async (orderIds: string[]) => {
    if (!session || actionLoading || orderIds.length === 0) return;
    try {
      setActionLoading(true);
      setActionError(null);
      setActionSuccess(null);
      const res = await api.unassignOrders(session.session_id, orderIds);
      if (res.success) {
        const updated = await api.getDispatchSession(session.session_id);
        setSession(updated);
        await refreshNearby(selectedShipperId, session.session_id);
        setActionSuccess(`Đã thu hồi ${orderIds.length} đơn hàng về trạng thái chưa gán.`);
      } else {
        setActionError("Hủy phân công thất bại.");
      }
    } catch (err: any) {
      setActionError(err.message || "Lỗi khi thu hồi đơn hàng.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleAutoDispatch = async (strategy?: string, algorithm?: string) => {
    if (!session || actionLoading) return;
    try {
      setActionLoading(true);
      setActionError(null);
      setActionSuccess(null);
      const strat = (strategy || "cluster_aware_rl") as any;
      const algo = (algorithm || session.routing_algorithm) as any;
      const res = await api.autoDispatch(session.session_id, algo, strat);
      if (res.success) {
        const updated = await api.getDispatchSession(session.session_id);
        setSession(updated);
        await refreshNearby(selectedShipperId, session.session_id);
        setActionSuccess(`Tự động phân công hoàn tất! ${res.dispatched_orders} đơn đã được gán theo ${strat} + ${algo}.`);
      } else {
        setActionError("Tự động phân công thất bại.");
      }
    } catch (err: any) {
      setActionError(err.message || "Lỗi chạy thuật toán tự động phân công.");
    } finally {
      setActionLoading(false);
    }
  };

  const handlePreviewDispatch = async (strategy: string, algorithm: string) => {
    if (!session) throw new Error("Chưa có phiên làm việc.");
    try {
      setActionError(null);
      const plan = await api.previewDispatch(session.session_id, strategy as any, algorithm as any);
      setActivePreviewPlan(plan);
      return plan;
    } catch (err: any) {
      setActionError(err.message || "Không thể tải trước phương án điều phối.");
      throw err;
    }
  };

  const handleConfirmPreview = async (previewId: string) => {
    if (!session || actionLoading) return;
    try {
      setActionLoading(true);
      setActionError(null);
      setActionSuccess(null);
      const res = await api.confirmPreview(session.session_id, previewId);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      setActivePreviewPlan(null);
      await refreshNearby(selectedShipperId, session.session_id);
      setActionSuccess(
        `Đã áp dụng chính thức kế hoạch: Phân công thành công ${res.dispatched_orders} đơn hàng!`
      );
    } catch (err: any) {
      setActionError(err.message || "Không thể áp dụng bản xem trước kế hoạch.");
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  const handleCancelPreview = async () => {
    if (!session) return;
    try {
      await api.cancelPreview(session.session_id);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      setActivePreviewPlan(null);
      setActionSuccess("Đã hủy bản xem trước. Kế hoạch hiện tại được giữ nguyên.");
    } catch (err: any) {
      console.warn("Could not cancel preview", err);
      setActivePreviewPlan(null);
    }
  };

  const handleSelectAlgorithm = async (algo: string) => {
    if (!session || actionLoading) return;
    try {
      setActionLoading(true);
      setActionError(null);
      setActionSuccess(null);
      await api.autoDispatch(session.session_id, algo as any);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      setActionSuccess(`Đã cập nhật và áp dụng thuật toán: ${algo.toUpperCase()}`);
    } catch (err: any) {
      setActionError(err.message || "Không thể cập nhật thuật toán định tuyến.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleStepClock = async (deltaMinutes = 15) => {
    if (!session || actionLoading) return;
    try {
      setActionLoading(true);
      setActionError(null);
      await api.stepSimulationClock(session.session_id, deltaMinutes);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      if (updated.is_day_ended) {
        setIsPlaying(false);
      }
    } catch (err: any) {
      setActionError(err.message || "Không thể tua đồng hồ mô phỏng.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleEndDay = async () => {
    if (!session || actionLoading) return;
    try {
      setIsPlaying(false);
      setActionLoading(true);
      setActionError(null);
      await api.endDay(session.session_id);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      setActionSuccess(`Đã chốt ca kết thúc Ngày ${session.current_day}.`);
    } catch (err: any) {
      setActionError(err.message || "Không thể chốt ca kết thúc ngày.");
    } finally {
      setActionLoading(false);
    }
  };

  const handleNextDay = async () => {
    if (!session || actionLoading) return;
    try {
      setIsPlaying(false);
      setActionLoading(true);
      setActionError(null);
      await api.nextDay(session.session_id);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      await refreshNearby(selectedShipperId, session.session_id);
      if (updated.current_day > updated.total_days || (updated.current_day === updated.total_days && updated.is_day_ended)) {
        setActionSuccess("Toàn bộ kịch bản mô phỏng đã hoàn thành!");
      } else {
        setActionSuccess(`Đã bước sang Ngày ${updated.current_day}! Đơn tồn đã được chuyển ca bảo toàn SLA.`);
      }
    } catch (err: any) {
      setActionError(err.message || "Không thể chuyển sang ngày tiếp theo.");
    } finally {
      setActionLoading(false);
    }
  };

  const value: DispatchContextType = {
    session,
    loading,
    connectionError,
    sessionNotFoundError,
    selectedShipperId,
    setSelectedShipperId,
    nearbyOrders,
    activePreviewPlan,
    setActivePreviewPlan,
    isPlaying,
    simSpeed,
    setSimSpeed,
    actionLoading,
    actionError,
    actionSuccess,
    setActionError,
    setActionSuccess,
    initSession,
    createNewSession,
    handleAssign,
    handleUnassign,
    handleAutoDispatch,
    handlePreviewDispatch,
    handleConfirmPreview,
    handleCancelPreview,
    handleSelectAlgorithm,
    handleStepClock,
    handleTogglePlay,
    handleEndDay,
    handleNextDay,
    refreshNearby,
  };

  return (
    <DispatchContext.Provider value={value}>
      {children}
    </DispatchContext.Provider>
  );
}

export function useDispatchContext() {
  const context = useContext(DispatchContext);
  if (!context) {
    throw new Error("useDispatchContext must be used within a DispatchProvider");
  }
  return context;
}
