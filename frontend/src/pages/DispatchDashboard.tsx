import React, { useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Calendar,
  RotateCcw,
  RefreshCw,
  ServerOff,
  AlertTriangle,
  Check,
  X,
} from "lucide-react";
import {
  api,
  DispatchSessionState,
  NearbyOrderItem,
  PlanPreviewResponse,
} from "@/lib/api";
import { PlanningDesk } from "@/components/dispatch/PlanningDesk";
import { LiveOperations } from "@/components/dispatch/LiveOperations";
import { DayReport } from "@/components/dispatch/DayReport";

export type DispatchWorkspace = "planning" | "live" | "report";

export default function DispatchDashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlWorkspace = searchParams.get("workspace") as DispatchWorkspace;
  const initialWs: DispatchWorkspace =
    urlWorkspace === "live" || urlWorkspace === "report" ? urlWorkspace : "planning";

  // Session & Global State
  const [session, setSession] = useState<DispatchSessionState | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [sessionNotFoundError, setSessionNotFoundError] = useState<string | null>(null);
  const [activeWorkspace, setActiveWorkspace] = useState<DispatchWorkspace>(initialWs);

  const switchWorkspace = (ws: DispatchWorkspace, sid?: string) => {
    setActiveWorkspace(ws);
    const activeId = sid || session?.session_id || searchParams.get("session_id") || localStorage.getItem("nexus_active_session_id");
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("workspace", ws);
      if (activeId) {
        next.set("session_id", activeId);
      }
      return next;
    });
  };

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

  // Auto-dismiss notification toasts so they never linger during active operations
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
          // Rule 6: Do not auto-create a new session when old session does not exist without explicit user action
          console.warn("Phiên làm việc không tồn tại hoặc đã hết hạn:", targetSessId);
          if (typeof window !== "undefined") {
            localStorage.removeItem("nexus_active_session_id");
          }
          setSessionNotFoundError(targetSessId);
          setLoading(false);
          return;
        }
      }

      if (!sess) {
        // Only auto-create when user navigated fresh to /dieu-phoi without any session requested
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
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("session_id", sess!.session_id);
        if (!next.has("workspace")) {
          next.set("workspace", initialWs);
        }
        return next;
      }, { replace: true });

      if (sess.shippers && sess.shippers.length > 0) {
        setSelectedShipperId(sess.shippers[0].shipper_id);
      }
    } catch (err: any) {
      setConnectionError(
        err.message ||
          "Không thể kết nối đến Backend tại 127.0.0.1:8010. Vui lòng đảm bảo FastAPI đang chạy."
      );
    } finally {
      setLoading(false);
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

  // Handle Simulation Loop with precise auto-pause and overtime checks
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
            switchWorkspace("report");
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
        setActionError("Ca làm việc hôm nay đã kết thúc. Vui lòng chuyển sang Ngày tiếp theo hoặc xem Báo cáo ca.");
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
        setActionSuccess(`Đã hủy phân công ${orderIds.length} đơn.`);
      } else {
        setActionError("Hủy phân công thất bại.");
      }
    } catch (err: any) {
      setActionError(err.message || "Không thể hủy phân công đơn.");
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
        `Đã áp dụng chính thức kế hoạch (${previewId}): Phân công thành công ${res.dispatched_orders} đơn hàng!`
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
      // Run auto dispatch with the new algorithm
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
        switchWorkspace("report");
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
      switchWorkspace("report");
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
      const res = await api.nextDay(session.session_id);
      const updated = await api.getDispatchSession(session.session_id);
      setSession(updated);
      await refreshNearby(selectedShipperId, session.session_id);
      switchWorkspace("planning");
      setActionSuccess(
        `Bắt đầu Ngày ${res.day}: Nhận ${res.carried_over_count} đơn tồn kho + ${res.new_incoming_count} đơn mới!`
      );
    } catch (err: any) {
      setActionError((err as any).message || "Không thể chuyển sang ngày tiếp theo.");
    } finally {
      setActionLoading(false);
    }
  };

  const pad = (n: number) => String(n).padStart(2, "0");

  // Loading state
  if (loading && !session) {
    return (
      <div className="wrap" style={{ minHeight: "60vh", display: "grid", placeItems: "center" }}>
        <div style={{ textAlign: "center" }}>
          <RefreshCw className="icons animate-spin" style={{ width: "32px", height: "32px", color: "var(--accent)", margin: "0 auto 12px" }} />
          <div style={{ fontWeight: 700, fontSize: "15px" }}>
            Đang khởi tạo phiên làm việc NEXUS Dispatch Studio...
          </div>
          <div className="subline" style={{ marginTop: "4px" }}>
            Kết nối Backend tại 127.0.0.1:8010
          </div>
        </div>
      </div>
    );
  }

  // Session Not Found state (Rule 6 compliant)
  if (sessionNotFoundError && !session) {
    return (
      <div className="wrap" style={{ minHeight: "60vh", display: "grid", placeItems: "center", padding: "40px 20px" }}>
        <div className="box" style={{ maxWidth: "520px", padding: "32px", textAlign: "center" }}>
          <div style={{ width: "52px", height: "52px", borderRadius: "50%", background: "rgba(245, 158, 11, 0.15)", color: "#f59e0b", display: "grid", placeItems: "center", margin: "0 auto 16px", border: "1px solid rgba(245, 158, 11, 0.3)" }}>
            <AlertTriangle className="icons" style={{ width: "26px", height: "26px" }} />
          </div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "8px" }}>Không tìm thấy phiên điều phối</h2>
          <p className="subline" style={{ margin: "10px 0 20px", fontSize: "0.875rem", lineHeight: "1.5" }}>
            Phiên làm việc <code style={{ padding: "2px 6px", borderRadius: "4px", background: "var(--muted)", fontWeight: 600 }}>{sessionNotFoundError}</code> không tồn tại trên hệ thống hoặc đã hết hạn (có thể do Backend đã khởi động lại).
          </p>
          <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
            <button
              type="button"
              id="btnCreateNewSessionExplicit"
              onClick={() => {
                setSessionNotFoundError(null);
                initSession(true);
              }}
              className="btn primary"
              style={{ fontWeight: 600, padding: "8px 20px" }}
            >
              Khởi tạo phiên điều phối mới
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Connection Error state
  if (connectionError && !session) {
    return (
      <div className="wrap" style={{ minHeight: "60vh", display: "grid", placeItems: "center", padding: "40px 20px" }}>
        <div className="box" style={{ maxWidth: "460px", padding: "32px", textAlign: "center" }}>
          <div style={{ width: "48px", height: "48px", borderRadius: "50%", background: "#fdebed", color: "var(--red)", display: "grid", placeItems: "center", margin: "0 auto 16px" }}>
            <ServerOff className="icons" style={{ width: "24px", height: "24px" }} />
          </div>
          <h2>Không thể kết nối Backend Dispatch API</h2>
          <p className="subline" style={{ margin: "10px 0 20px" }}>{connectionError}</p>
          <button onClick={() => initSession()} className="btn primary block">
            Thử kết nối lại
          </button>
        </div>
      </div>
    );
  }

  if (!session) return null;

  const titles: Record<DispatchWorkspace, { title: string; subtitle: string }> = {
    planning: {
      title: "Bàn lập kế hoạch",
      subtitle: "Phân công đơn hàng, kiểm tra năng lực đội xe và chuẩn bị tuyến giao.",
    },
    live: {
      title: "Theo dõi giao hàng",
      subtitle: "Quan sát tiến độ, tuyến giao và hoạt động của đội xe trên bản đồ.",
    },
    report: {
      title: "Tổng kết ngày giao",
      subtitle: "Kết quả giao hàng và đơn cần chuyển sang ca tiếp theo.",
    },
  };

  return (
    <section id="page-dispatch" className="page">
      <div className="wrap">
        {/* Page Intro matching Approved Reference */}
        <div className="page-intro">
          <div>
            <div className="eyebrow">
              DISPATCH / NGÀY <span>{pad(session.current_day)}</span>
            </div>
            <h1>{titles[activeWorkspace].title}</h1>
            <p className="subline">{titles[activeWorkspace].subtitle}</p>
          </div>

          <div className="hero-right">
            <div className="date-pill">
              <Calendar className="icons smico" /> Ngày <span>{pad(session.current_day)}</span> / {pad(session.total_days)} <span style={{ opacity: 0.7, fontSize: "12px", marginLeft: "4px" }}>• Haversine 25km/h</span>
            </div>

            <div className="view-tabs" role="tablist" aria-label="Chế độ điều phối">
              <button
                type="button"
                id="tabPlanning"
                className={`tabbtn ${activeWorkspace === "planning" ? "active" : ""}`}
                onClick={() => switchWorkspace("planning")}
              >
                Lập kế hoạch
              </button>
              <button
                type="button"
                id="tabLive"
                className={`tabbtn ${activeWorkspace === "live" ? "active" : ""}`}
                onClick={() => switchWorkspace("live")}
              >
                Theo dõi giao hàng
              </button>
              <button
                type="button"
                id="tabReport"
                className={`tabbtn ${activeWorkspace === "report" ? "active" : ""}`}
                onClick={() => switchWorkspace("report")}
              >
                Báo cáo ngày
              </button>
            </div>

            {/* Quick Session Reset */}
            <button
              type="button"
              onClick={() => initSession(true)}
              disabled={actionLoading}
              className="icon-button"
              title="Khởi tạo lại phiên mô phỏng mới"
              aria-label="Khởi tạo lại phiên mới"
            >
              <RotateCcw className="icons smico" />
            </button>
          </div>
        </div>

        {/* Global Toast for Action Success/Error — Positioned Top-Right to Never Block Controls */}
        {(actionSuccess || actionError) && (
          <div
            id="nexusGlobalToast"
            role="status"
            aria-live="polite"
            style={{
              position: "fixed",
              top: "76px",
              right: "24px",
              zIndex: 1000,
              display: "flex",
              alignItems: "center",
              gap: "10px",
              background: actionError ? "#7f1d1d" : "#064e3b",
              color: "#ffffff",
              padding: "10px 16px",
              borderRadius: "8px",
              boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3)",
              border: actionError ? "1px solid #b91c1c" : "1px solid #059669",
              fontSize: "13px",
              fontWeight: 600,
              maxWidth: "420px",
            }}
          >
            {actionError ? (
              <AlertTriangle className="size-4 shrink-0 text-red-300" />
            ) : (
              <Check className="size-4 shrink-0 text-emerald-300" />
            )}
            <span style={{ flex: 1, lineHeight: 1.4 }}>{actionError || actionSuccess}</span>
            <button
              type="button"
              id="btnCloseToast"
              onClick={() => {
                setActionSuccess(null);
                setActionError(null);
              }}
              style={{
                background: "transparent",
                border: "none",
                color: "rgba(255, 255, 255, 0.8)",
                cursor: "pointer",
                padding: "2px",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: "4px",
              }}
              title="Đóng thông báo"
              aria-label="Đóng thông báo"
            >
              <X className="size-4" />
            </button>
          </div>
        )}

        {/* ── Active Workspace Viewport ── */}
        <div className="view-content">
          {activeWorkspace === "planning" && (
            <PlanningDesk
              session={session}
              selectedShipperId={selectedShipperId}
              onSelectShipper={setSelectedShipperId}
              nearbyOrders={nearbyOrders}
              onAssign={handleAssign}
              onUnassign={handleUnassign}
              onAutoDispatch={handleAutoDispatch}
              onPreviewPlan={handlePreviewDispatch}
              onConfirmPreview={handleConfirmPreview}
              onCancelPreview={handleCancelPreview}
              previewPlan={activePreviewPlan}
              onSetPreviewPlan={setActivePreviewPlan}
              onSwitchToLive={() => switchWorkspace("live")}
              onSelectAlgorithm={handleSelectAlgorithm}
              actionLoading={actionLoading}
              actionError={actionError}
              actionSuccess={actionSuccess}
              onReloadSession={() => initSession(false)}
            />
          )}

          {activeWorkspace === "live" && (
            <LiveOperations
              session={session}
              selectedShipperId={selectedShipperId}
              onSelectShipper={setSelectedShipperId}
              onStepClock={handleStepClock}
              isPlaying={isPlaying}
              onTogglePlay={handleTogglePlay}
              simSpeed={simSpeed}
              onChangeSimSpeed={setSimSpeed}
              onEndDay={handleEndDay}
              onSwitchToPlanning={() => switchWorkspace("planning")}
              onSwitchToReport={() => switchWorkspace("report")}
              actionLoading={actionLoading}
              actionError={actionError}
              actionSuccess={actionSuccess}
            />
          )}

          {activeWorkspace === "report" && (
            <DayReport
              session={session}
              onNextDay={handleNextDay}
              onSwitchToPlanning={() => switchWorkspace("planning")}
              onSwitchToLive={() => switchWorkspace("live")}
              actionLoading={actionLoading}
              actionError={actionError}
              actionSuccess={actionSuccess}
            />
          )}
        </div>
      </div>
    </section>
  );
}
