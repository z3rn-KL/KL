import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Truck,
  Play,
  Pause,
  Clock,
  RotateCcw,
  FastForward,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ShieldAlert,
  Building2,
  Compass,
  FileText,
  Calendar,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LiveOperations } from "@/components/dispatch/LiveOperations";
import { activeShippersOnDay } from "@/lib/dispatchMetrics";
import { useDispatchContext } from "@/context/DispatchContext";

export default function SimulationOps() {
  const navigate = useNavigate();
  const {
    session,
    loading,
    connectionError,
    sessionNotFoundError,
    selectedShipperId,
    setSelectedShipperId,
    isPlaying,
    simSpeed,
    setSimSpeed,
    actionLoading,
    actionError,
    actionSuccess,
    handleTogglePlay,
    handleStepClock,
    handleEndDay,
    initSession,
  } = useDispatchContext();

  const onEndDayAndGoToReport = async () => {
    await handleEndDay();
    navigate("/bao-cao");
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3">
        <div className="size-8 rounded-full border-2 border-teal-600 border-t-transparent animate-spin" />
        <span className="text-xs text-muted-foreground font-mono">Đang nạp trạng thái mô phỏng 16 shipper…</span>
      </div>
    );
  }

  if (connectionError || !session) {
    return (
      <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-center space-y-3 max-w-xl mx-auto my-12">
        <AlertTriangle className="size-8 text-rose-500 mx-auto" />
        <h2 className="text-base font-bold text-foreground">Không thể kết nối đến Backend</h2>
        <p className="text-xs text-muted-foreground">{connectionError || "Vui lòng đảm bảo FastAPI đang chạy tại 127.0.0.1:8011."}</p>
        <Button onClick={() => initSession(true)} className="bg-teal-600 text-white hover:bg-teal-700 text-xs">
          Thử kết nối lại
        </Button>
      </div>
    );
  }

  const deliveredCount = session.order_summary?.delivered ?? 0;
  const assignedCount = session.order_summary?.assigned ?? 0;
  const totalCount = session.order_summary?.total ?? 0;
  // Backend's active_shippers counts non-IDLE, including SHIFT_ENDED and idle
  // staff after close. Show shipper participation based on actual day orders.
  const daySnapshot = session.daily_metrics.find((m) => m.day === session.current_day);
  const activeShippers = session.is_day_ended && daySnapshot?.shipper_performance
    ? daySnapshot.shipper_performance.filter((p) => p.assigned_count > 0 || p.completed_count > 0).length
    : activeShippersOnDay(session.orders, session.shippers, session.current_day);
  const totalDistance = session.fleet_summary?.total_distance_km ?? 0;

  return (
    <div className="space-y-2">
      {/* ============================================================== */}
      {/* 1. SCREEN HEADER & QUICK ACTIONS (Compact, no subtitle)        */}
      {/* ============================================================== */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-border/70 pb-2">
        <div className="flex items-center gap-2">
          <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground">
            Mô phỏng giao hàng
          </h1>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/toi-uu-rl")}
            className="text-xs h-8 px-3 cursor-pointer font-medium"
          >
            Về bàn lập tuyến
          </Button>
          <Button
            size="sm"
            onClick={onEndDayAndGoToReport}
            disabled={actionLoading}
            className="gap-1.5 text-xs h-8 px-3.5 bg-teal-600 text-white hover:bg-teal-700 font-semibold shadow-xs cursor-pointer"
          >
            <FileText className="size-3.5" />
            <span>Chốt ca &amp; xem báo cáo</span>
            <ArrowRight className="size-3.5" />
          </Button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. REAL-TIME FLEET & SIMULATION STRIP (Compact 1-row)          */}
      {/* ============================================================== */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Ngày mô phỏng</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-foreground mt-0.5 block">
            Ngày {session.current_day} / {session.total_days}
          </span>
        </div>

        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Thời gian đã trôi qua</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-teal-600 dark:text-teal-400 mt-0.5 block">
            {Math.floor(session.simulation_clock / 60)}h {Math.floor(session.simulation_clock % 60).toString().padStart(2, "0")}'
          </span>
        </div>

        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Đơn đã giao</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-emerald-600 dark:text-emerald-400 mt-0.5 block">
            {deliveredCount} / {totalCount}
          </span>
        </div>

        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Đơn đang giao</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-amber-500 mt-0.5 block">
            {assignedCount} đơn
          </span>
        </div>

        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Shipper đang hoạt động</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-foreground mt-0.5 block">
            {activeShippers} / {session.shippers.length} shipper
          </span>
        </div>

        <div className="p-2 sm:px-3 sm:py-2 rounded-lg border border-border bg-card shadow-2xs">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block truncate">Quãng đường đã di chuyển</span>
          <span className="text-sm sm:text-base font-extrabold font-mono text-foreground mt-0.5 block">
            {totalDistance} km
          </span>
        </div>
      </div>

      {/* Empty State Callout when no plan confirmed yet (Compact Single-line) */}
      {assignedCount === 0 && deliveredCount === 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 sm:px-3 flex items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle className="size-4 text-amber-600 shrink-0" />
            <span className="text-foreground text-xs font-medium truncate">
              Chưa có kế hoạch tuyến cho ca hiện tại. Hãy lập và xác nhận kế hoạch tại Bàn lập tuyến.
            </span>
          </div>
          <Button
            size="sm"
            onClick={() => navigate("/toi-uu-rl")}
            className="bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs h-7 px-3 shrink-0 gap-1.5 cursor-pointer"
          >
            <span>Sang Bàn lập tuyến</span>
            <ArrowRight className="size-3" />
          </Button>
        </div>
      )}

      {/* ============================================================== */}
      {/* 3. CORE SIMULATION COMPONENT (LIVE OPERATIONS)                 */}
      {/* ============================================================== */}
      <LiveOperations
        session={session}
        selectedShipperId={selectedShipperId}
        onSelectShipper={setSelectedShipperId}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlay}
        simSpeed={simSpeed}
        onChangeSimSpeed={setSimSpeed}
        onStepClock={handleStepClock}
        onEndDay={onEndDayAndGoToReport}
        onSwitchToPlanning={() => navigate("/toi-uu-rl")}
        onSwitchToReport={() => navigate("/bao-cao")}
        actionLoading={actionLoading}
        actionError={actionError}
        actionSuccess={actionSuccess}
      />
    </div>
  );
}
