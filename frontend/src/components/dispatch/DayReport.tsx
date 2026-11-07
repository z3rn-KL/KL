import React, { useState, useEffect, useMemo } from "react";
import { ArrowRight, Calendar, RotateCcw, CheckCircle2, Clock, ShieldAlert, Truck, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { DispatchSessionState } from "@/lib/api";
import { carryoverEnteringDay, workloadForDay, outstandingAtEndOfDay, activeShippersOnDay } from "@/lib/dispatchMetrics";
import { NewScenarioModal } from "./NewScenarioModal";

export interface DayReportProps {
  session: DispatchSessionState;
  onNextDay: () => Promise<void>;
  onSwitchToPlanning: () => void;
  onSwitchToLive: () => void;
  onCreateNewSession?: (totalDays: number) => Promise<void>;
  actionLoading: boolean;
  actionError: string | null;
  actionSuccess: string | null;
}

export function DayReport({
  session,
  onNextDay,
  onSwitchToPlanning,
  onSwitchToLive,
  onCreateNewSession,
  actionLoading,
  actionError,
  actionSuccess,
}: DayReportProps) {
  const pad = (n: number | string) => String(n).padStart(2, "0");

  const [viewDay, setViewDay] = useState<number>(session.current_day);
  const [showShipperDetails, setShowShipperDetails] = useState<boolean>(false);
  const [newScenarioModalOpen, setNewScenarioModalOpen] = useState<boolean>(false);

  useEffect(() => {
    setViewDay(session.current_day);
  }, [session.current_day]);

  // Historical snapshot for viewDay if already recorded in daily_metrics
  const dayReportSnapshot = useMemo(() => {
    if (!session.daily_metrics || session.daily_metrics.length === 0) return null;
    return session.daily_metrics.find((m) => m.day === viewDay) || null;
  }, [session.daily_metrics, viewDay]);

  const isHistoricalDay = dayReportSnapshot !== null;
  const isShiftEnded = isHistoricalDay || session.is_day_ended || session.simulation_clock >= 600;

  // 1. New Orders Released Today
  const newOrdersToday = session.orders.filter((o) => o.release_day === viewDay).length;

  // 2. Carryover IN is not all historical orders: those delivered BEFORE this day
  // cannot be carried in. Backend daily_metrics currently overcounts this field.
  const carriedIn = carryoverEnteringDay(session.orders, viewDay);

  // 3. Correct day-scoped workload, without re-counting prior completed orders.
  const totalToProcess = workloadForDay(session.orders, viewDay);

  // Exact predicate for orders delivered in this specific viewDay shift
  const isDeliveredInViewDay = (o: any) =>
    o.delivery_status === "DELIVERED" &&
    (o.delivered_day === viewDay || (o.delivered_day == null && viewDay === 1));

  // 4. Delivered Today
  const dayDelivered = isHistoricalDay
    ? (dayReportSnapshot.day_delivered ?? 0)
    : session.orders.filter(isDeliveredInViewDay).length;

  // 5. Daily SLA On-time Rate (Strict subset: OnTimeDelivered ⊆ TotalDelivered, 0% <= SLA <= 100%)
  const dayDeliveredOnTime = isHistoricalDay
    ? (dayReportSnapshot.day_delivered_on_time ?? Math.round((dayReportSnapshot.day_sla_rate ?? dayReportSnapshot.day_on_time_rate ?? 0) * dayDelivered))
    : session.orders.filter((o) => isDeliveredInViewDay(o) && o.sla_status === "ON_TIME").length;

  // Prefer exact numerator/denominator over a rounded API fraction.
  const dayOnTimeRate = dayDelivered > 0
    ? Math.round((dayDeliveredOnTime / dayDelivered) * 100)
    : 0;

  // 6. Remaining Orders Today
  const dayRemaining = isHistoricalDay
    ? (dayReportSnapshot.day_remaining ?? dayReportSnapshot.carried_over_out ?? Math.max(0, totalToProcess - dayDelivered))
    : Math.max(0, totalToProcess - dayDelivered);

  // 7. Carryover Out to Next Day
  const carriedOut = isHistoricalDay
    ? (dayReportSnapshot.day_carried_over_out ?? dayReportSnapshot.carried_over_out ?? dayRemaining)
    : dayRemaining;

  // 8. Cumulative Delivered
  const rawCumDelivered = (isHistoricalDay
    ? (dayReportSnapshot.cumulative_delivered ?? dayReportSnapshot.delivered)
    : session.orders.filter((o) => o.delivery_status === "DELIVERED").length) ?? 0;
  // Preserve the authoritative count. Math.max would silently hide inconsistent snapshots.
  const cumulativeDelivered = rawCumDelivered;

  // 9. Cumulative Total Released
  const rawCumTotal = isHistoricalDay
    ? (dayReportSnapshot.cumulative_total_released ?? dayReportSnapshot.total_orders ?? session.orders.filter((o) => o.release_day <= viewDay).length)
    : session.orders.filter((o) => o.release_day <= viewDay).length;
  // Carryover is included in today's workload, NOT added as new unique orders.
  const cumulativeTotalReleased = rawCumTotal;

  // 10. Cumulative SLA Rate (Zero denominator guard: null if no deliveries)
  const rawCumSla = isHistoricalDay
    ? (cumulativeDelivered > 0
        ? (dayReportSnapshot.delivered_on_time != null
            ? Math.round(dayReportSnapshot.delivered_on_time / cumulativeDelivered * 100)
            : Math.round((dayReportSnapshot.cumulative_sla_rate ?? dayReportSnapshot.on_time_rate ?? 0) * 100))
        : null)
    : (cumulativeDelivered > 0
        ? Math.round(
            (session.orders.filter((o) => o.delivery_status === "DELIVERED" && o.sla_status === "ON_TIME").length /
              cumulativeDelivered) *
              100
          )
        : null);
  // Mathematical Invariant: On Day 1, cumulative SLA rate must match Day 1 single-day rate
  const cumulativeSlaRate = (viewDay === 1 && dayDelivered > 0) ? dayOnTimeRate : rawCumSla;

  const dayCompletionPct = totalToProcess > 0 ? (dayDelivered / totalToProcess) * 100 : 0;
  const cumulativeCompletionPct = cumulativeTotalReleased > 0 ? (cumulativeDelivered / cumulativeTotalReleased) * 100 : 0;

  // Shift status
  const shiftStatus = isHistoricalDay
    ? dayReportSnapshot.shift_status
    : (session.simulation_clock >= 600 ? "FULL_SHIFT" : "IN_PROGRESS");

  const shiftStatusLabel = useMemo(() => {
    switch (shiftStatus) {
      case "OVERTIME":
        return "Tăng ca";
      case "FULL_SHIFT":
        return "Đủ ca";
      case "EARLY_CUT":
        return "Chốt sớm";
      default:
        return "Đang thực hiện";
    }
  }, [shiftStatus]);

  // Carried over orders list to display
  // Reconstruct historical carryover from immutable release/delivery day stamps.
  // A delivery on Day 3 was still outstanding at the end of Day 2.
  const carriedOrdersList = useMemo(() => {
    return outstandingAtEndOfDay(session.orders, viewDay);
  }, [session.orders, viewDay]);

  // Total distance
  const totalDistanceKm = useMemo(() => {
    if (isHistoricalDay) {
      return dayReportSnapshot.total_distance_km;
    }
    return session.shippers.reduce((sum, s) => sum + (s.distance_travelled_km || 0), 0);
  }, [session.shippers, isHistoricalDay, dayReportSnapshot]);

  // Number of shippers actively assigned or completed orders in viewDay
  const activeShippersCountInDay = useMemo(() => {
    if (isHistoricalDay && dayReportSnapshot?.shipper_performance) {
      return dayReportSnapshot.shipper_performance.filter((p) => (p.assigned_count > 0 || p.completed_count > 0)).length;
    }
    return activeShippersOnDay(session.orders, session.shippers, viewDay);
  }, [session.shippers, session.orders, isHistoricalDay, dayReportSnapshot, viewDay]);

  // Dynamic State-Aware Shift Narrative (Cases A, B, C, D handled rigorously)
  const shiftNarrative = useMemo(() => {
    const isCompleted = dayDelivered >= totalToProcess && totalToProcess > 0;
    const isZeroDelivered = dayDelivered === 0;
    const assignedCount = session.orders.filter((o) =>
      (o.release_day === viewDay ||
       (o.release_day < viewDay && (o.delivered_day == null || o.delivered_day >= viewDay))) &&
      o.assigned_shipper_id != null
    ).length;

    let title = "";
    let completionText = "";
    let fleetText = "";
    let slaText = "";

    if (isCompleted) {
      title = `HOÀN THÀNH TOÀN BỘ CA: 100% (${dayDelivered}/${totalToProcess} ĐƠN)`;
      completionText = `Toàn bộ ${dayDelivered}/${totalToProcess} đơn trong ca đã được giao thành công.`;
    } else if (isZeroDelivered) {
      title = `TRẠNG THÁI CA: ĐANG KHỞI TẠO / CHƯA GIAO ĐƠN (0/${totalToProcess} ĐƠN)`;
      completionText = assignedCount > 0
        ? `Đã phân bổ ${assignedCount}/${totalToProcess} đơn cho đội xe, đang chờ tiến trình mô phỏng thực thi (Tỷ lệ hoàn thành: 0%).`
        : `Chưa có đơn hàng nào được hoàn thành trong ca này (0/${totalToProcess} đơn, 0% hoàn thành).`;
    } else {
      title = `TIẾN ĐỘ CA: ĐANG THỰC HIỆN (${Math.round(dayCompletionPct)}% · ${dayDelivered}/${totalToProcess} ĐƠN)`;
      completionText = `Đã giao ${dayDelivered}/${totalToProcess} đơn (${Math.round(dayCompletionPct)}%). Còn lại ${dayRemaining} đơn đang vận chuyển hoặc chuyển tiếp.`;
    }

    fleetText = `Có ${activeShippersCountInDay}/${session.shippers.length} shipper có đơn được ghi nhận trong ngày; ${session.shippers.length - activeShippersCountInDay} shipper không có đơn được ghi nhận.`;

    slaText = dayDelivered > 0
      ? `Tỷ lệ đúng hạn SLA: ${dayOnTimeRate}% (${dayDeliveredOnTime}/${dayDelivered} đơn giao đúng hạn).`
      : `Tỷ lệ đúng hạn SLA: Chưa có dữ liệu (chỉ đánh giá khi có ít nhất 1 đơn hoàn thành).`;

    return { title, completionText, fleetText, slaText };
  }, [dayDelivered, totalToProcess, dayCompletionPct, dayRemaining, activeShippersCountInDay, session.shippers.length, dayOnTimeRate, dayDeliveredOnTime, session.orders, viewDay]);


  const operatingHours = isHistoricalDay
    ? ((dayReportSnapshot.simulation_clock_at_end ?? 600) / 60)
    : (session.simulation_clock / 60);

  return (
    <div id="view-report" className="view max-w-full overflow-hidden">
      {/* Day Selector Navigation Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 mb-4 rounded-xl border border-border bg-card">
        <div className="flex items-center gap-2">
          <Calendar className="size-4 text-primary" />
          <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
            Chọn Ngày Báo Cáo:
          </span>
          <div className="flex items-center gap-1.5 ml-2">
            {Array.from({ length: session.total_days }, (_, i) => i + 1).map((d) => {
              const isAvailable = d <= session.current_day;
              const isSelected = d === viewDay;
              const hasSnapshot = session.daily_metrics?.some((m) => m.day === d);
              return (
                <button
                  key={d}
                  type="button"
                  disabled={!isAvailable}
                  onClick={() => setViewDay(d)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    isSelected
                      ? "bg-primary text-primary-foreground shadow-xs"
                      : isAvailable
                      ? "bg-muted/60 border border-border text-foreground hover:bg-muted"
                      : "opacity-40 bg-muted/20 text-muted-foreground cursor-not-allowed"
                  }`}
                >
                  Ngày {pad(d)} {hasSnapshot ? "✓ Đã chốt ca" : (d === session.current_day ? "• Đang chạy" : "")}
                </button>
              );
            })}
          </div>
        </div>

        <div className="text-xs text-muted-foreground font-medium">
          {isHistoricalDay ? (
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold flex items-center gap-1">
              <CheckCircle2 className="size-3.5" /> Dữ liệu lịch sử đã chốt ca
            </span>
          ) : (
            <span className="text-amber-600 dark:text-amber-400 font-semibold flex items-center gap-1">
              <Clock className="size-3.5" /> Dữ liệu trực tiếp ca làm việc hiện hành
            </span>
          )}
        </div>
      </div>

      {/* Top Banner */}
      <div className="report-top">
        <div>
          <h2 id="reportTitle">Báo cáo ca Ngày {pad(viewDay)}</h2>
          <p>
            {isHistoricalDay
              ? `Dữ liệu ca làm việc Ngày ${viewDay} đã chốt kiểm toán.`
              : `Tiến độ vận hành thực tế ca làm việc Ngày ${viewDay} đang diễn ra.`}
          </p>

                    {/* Scenario Completion Strip when session has reached total_days */}
          {viewDay === session.current_day && session.current_day >= session.total_days && session.is_day_ended && (
            <div id="reportScenarioCompletedStrip" className="mt-4 p-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 space-y-3 max-w-xl">
              <div className="flex items-center gap-2 text-emerald-800 dark:text-emerald-300 font-bold text-sm">
                <CheckCircle2 className="size-5 text-emerald-600 shrink-0" />
                <span>KỊCH BẢN ĐÃ HOÀN THÀNH TOÀN BỘ CHU KỲ ({session.total_days} NGÀY)</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Hệ thống đã hoàn tất chu kỳ điều phối {session.total_days} ngày với {cumulativeDelivered}/{cumulativeTotalReleased} đơn đã giao ({cumulativeSlaRate !== null ? `${cumulativeSlaRate}% SLA` : "Chưa có dữ liệu SLA"}). Bạn có thể đối chiếu các mô hình tại trang Thực nghiệm &amp; So sánh hoặc khởi tạo kịch bản quy mô mở rộng (5, 7, 14 ngày).
              </p>
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <button
                  type="button"
                  id="btnNewScenarioFromReport"
                  onClick={() => setNewScenarioModalOpen(true)}
                  className="btn primary sm"
                >
                  <RotateCcw className="size-3.5 mr-1" />
                  <span>Khởi tạo kịch bản mới</span>
                </button>
                <a
                  href="/thuc-nghiem"
                  id="btnGoToExperimentsFromReport"
                  className="btn sm"
                  style={{ textDecoration: "none" }}
                >
                  <span>Xem Thực nghiệm &amp; So sánh →</span>
                </a>
              </div>
            </div>
          )}

          {/* State A vs State B Dynamic Action Strip */}
          {viewDay === session.current_day && session.current_day < session.total_days && (
            <div style={{ marginTop: "20px" }} className="space-y-2">
              {isHistoricalDay || session.simulation_clock >= 600 ? (
                /* State B: Shift Ended -> Ready for next day */
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    className="btn primary"
                    id="nextDayBtn"
                    onClick={onNextDay}
                    disabled={actionLoading}
                  >
                    <span>Bắt đầu Ngày {session.current_day + 1}</span>
                    <ArrowRight className="icons" />
                  </button>
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold flex items-center gap-1">
                    <CheckCircle2 className="size-3.5" /> Ca đã chốt kiểm toán — Sẵn sàng chuyển ngày mới
                  </span>
                </div>
              ) : (
                /* State A: Shift In Progress -> Guide back to simulation */
                <div className="p-3 rounded-xl border border-amber-500/40 bg-amber-500/10 space-y-2 text-xs max-w-lg">
                  <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400 font-bold">
                    <Clock className="size-4" />
                    <span>TRẠNG THÁI: CA LÀM VIỆC ĐANG DIỄN RA ({Math.floor(session.simulation_clock / 60)}h {Math.floor(session.simulation_clock % 60)}')</span>
                  </div>
                  <p className="text-muted-foreground leading-snug">
                    Chỉ số SLA hiện tại là số liệu sơ bộ. Để chốt ca chính thức và chuyển sang Ngày {session.current_day + 1}, hãy hoàn tất hành trình tại Mô phỏng.
                  </p>
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      type="button"
                      className="btn sm primary"
                      onClick={onSwitchToLive}
                    >
                      <span>Vào Mô phỏng để Chốt ca</span>
                    </button>
                    <button
                      type="button"
                      className="btn sm"
                      id="nextDayBtn"
                      disabled={true}
                      title="Phải chốt ca trước khi chuyển sang ngày tiếp theo"
                      style={{ opacity: 0.5, cursor: "not-allowed" }}
                    >
                      <span>Khóa Chuyển ngày</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div>
          <div className="report-big" id="completionPct">
            {Math.round(dayCompletionPct)}
            <small>% hoàn thành trong ngày</small>
          </div>
          <p id="reportSubtitle">
            {dayDelivered} / {totalToProcess} đơn trong ngày {viewDay} · Lũy kế toàn phiên: {cumulativeDelivered} / {cumulativeTotalReleased} đơn
          </p>
        </div>
      </div>

      {/* Grid: Operational Metrics & Carried Over List */}
      <div className="report-grid">
        {/* Left: Operational Metrics */}
        <div className="box report-inner">
          <h2>Hiệu quả giao hàng Ngày {pad(viewDay)}</h2>
          <p className="subline" style={{ marginTop: "5px" }}>
            Phân định minh bạch giữa chỉ số riêng trong ngày và lũy kế toàn phiên.
          </p>

          <div className="report-metrics">
            <div>
              <b id="repNewToday">{newOrdersToday}</b>
              <span>Đơn mới trong ngày</span>
            </div>
            <div>
              <b id="repCarriedIn">{carriedIn}</b>
              <span>Đơn tồn đầu ngày</span>
            </div>
            <div>
              <b id="repTotalProcess">{totalToProcess}</b>
              <span>Tổng cần xử lý</span>
            </div>
            <div>
              <b id="repDayDelivered">{dayDelivered}</b>
              <span>Đã giao trong ngày</span>
            </div>
            <div>
              <b id="repDayOnTime" style={{ color: "#0aa58b" }}>
                {dayDelivered > 0 ? `${dayOnTimeRate}%` : "—"}
              </b>
              <span>Đúng hạn trong ngày</span>
            </div>
            <div>
              <b id="repCarry">{carriedOut}</b>
              <span>{isShiftEnded ? "Tồn chuyển tiếp" : "Chưa hoàn thành"}</span>
            </div>
            <div>
              <b id="repDelivered">
                {cumulativeDelivered} / {cumulativeTotalReleased}
              </b>
              <span>Lũy kế đã giao</span>
            </div>
            <div>
              <b id="repCumulativeSla" style={{ color: "#0aa58b" }}>
                {cumulativeDelivered > 0 ? `${cumulativeSlaRate}%` : "—"}
              </b>
              <span>Lũy kế đúng hạn</span>
            </div>
            <div>
              <b id="repDistance">{totalDistanceKm > 0 ? `${totalDistanceKm.toFixed(1)} km` : "—"}</b>
              <span>Cự ly di chuyển</span>
            </div>
            <div>
              <b id="repHours">{operatingHours > 0 ? `${operatingHours.toFixed(1)}h` : "—"}</b>
              <span>Thời lượng ca ({shiftStatusLabel})</span>
            </div>
            <div>
              <b id="repFleetUtilization" style={{ color: activeShippersCountInDay < session.shippers.length ? "#0284c7" : "#0aa58b" }}>
                {activeShippersCountInDay} / {session.shippers.length}
              </b>
              <span>Shipper nhận đơn ({(session.shippers.length > 0 ? ((activeShippersCountInDay / session.shippers.length) * 100).toFixed(0) : "0")}%)</span>
            </div>
          </div>

          <div className="stack-bar" id="reportBar" style={{ marginTop: "16px" }}>
            <span style={{ width: `${dayCompletionPct}%`, background: "#0aa58b" }} />
            <span style={{ width: `${100 - dayCompletionPct}%`, background: "#e5eceb" }} />
          </div>

          <details className="mt-3 text-xs text-muted-foreground group">
            <summary className="cursor-pointer font-semibold text-teal-700 dark:text-teal-300 select-none">
              [Định nghĩa &amp; thông số kỹ thuật ▼]
            </summary>
            <div className="p-2.5 mt-1.5 rounded-lg border border-border/70 bg-muted/30 space-y-1.5 leading-relaxed">
              <p>• <b>Đơn mới trong ngày ({newOrdersToday}):</b> Khối lượng phát sinh theo kịch bản.</p>
              <p>• <b>Đơn tồn đầu ngày ({carriedIn}):</b> Đơn chuyển tiếp từ ca trước giữ nguyên hạn SLA.</p>
              <p>• <b>Lũy kế đã giao ({cumulativeDelivered}/{cumulativeTotalReleased}):</b> Tổng sản lượng từ Ngày 1 đến Ngày {viewDay}.</p>
              <p className="font-mono text-xs text-muted-foreground pt-1 border-t border-border/50">Mã phiên: {session.session_id} · Kịch bản: {session.scenario_id}</p>
            </div>
          </details>
        </div>

        {/* Right: Carried Over Orders */}
        <div className="box report-inner">
          <div className="tracking-row">
            <h2>{isShiftEnded ? "Đơn tồn chuyển ca" : "Đơn chưa hoàn thành"}</h2>
            <span className="chip orange" id="carryBadge">
              {isShiftEnded ? carriedOut : dayRemaining} đơn
            </span>
          </div>
          <p className="subline" style={{ marginTop: "6px" }}>
            {isShiftEnded ? "Bảo toàn mốc cam kết SLA và thứ tự ưu tiên khi chuyển ca." : "Đơn hàng trong ca hiện tại chưa được giao hoặc đang được xử lý."}
          </p>

          <div id="carryList" style={{ maxHeight: "280px", overflowY: "auto", marginTop: "10px" }}>
            {carriedOrdersList.length > 0 ? (
              carriedOrdersList.slice(0, 10).map((o) => (
                <div className="carryrow" key={o.order_id}>
                  <div>
                    <strong>{o.order_code || o.order_id}</strong>
                    <small>
                      {o.package_name || "Kiện hàng"} · Cụm {o.cluster_id} · {o.weight_kg.toFixed(1)} kg
                    </small>
                  </div>
                  <span className="chip orange">
                    {isShiftEnded ? `Chưa giao cuối Ngày ${viewDay}` : "Chưa hoàn thành"}
                  </span>
                </div>
              ))
            ) : (
              <div className="nodata">{carriedOut > 0 ? `Còn ${carriedOut} đơn chưa hoàn thành; danh sách hiện thời chưa đồng bộ.` : "Không có đơn chưa hoàn thành trong phạm vi báo cáo."}</div>
            )}
          </div>
        </div>
      </div>

      {/* 16-Shipper Fleet Performance Table */}
      <div className="box" style={{ padding: "20px", marginBottom: "20px" }}>
        <div className="flex flex-wrap items-center justify-between gap-3 cursor-pointer" onClick={() => setShowShipperDetails((prev) => !prev)}>
          <div>
            <div className="flex items-center gap-2">
              <h2 style={{ margin: 0 }}>Hiệu suất đội ngũ shipper — Ngày {pad(viewDay)}</h2>
              <span className="chip cyan">{session.shippers.length} shipper</span>
            </div>
            <span className="muted small" style={{ display: "block", marginTop: "4px" }}>
              {isHistoricalDay ? "Ảnh chụp lịch sử chính thức tại thời điểm chốt ca" : "Số liệu theo dõi trực tiếp ca hiện hành"}
            </span>
          </div>
          <button
            type="button"
            className="btn sm"
            id="btnToggleShipperDetails"
            onClick={(e) => {
              e.stopPropagation();
              setShowShipperDetails((prev) => !prev);
            }}
          >
            <span>{showShipperDetails ? "Thu gọn danh sách" : "Xem chi tiết 16 shipper"}</span>
            {showShipperDetails ? <ChevronUp className="size-4 ml-1" /> : <ChevronDown className="size-4 ml-1" />}
          </button>
        </div>
        {showShipperDetails && (
        <div style={{ marginTop: "14px" }}>
          <div className="rounded-lg border border-teal-500/30 bg-teal-500/5 p-3.5 mb-3 text-xs leading-relaxed text-muted-foreground">
            <div className="flex items-center gap-2 font-bold text-teal-800 dark:text-teal-300 text-xs mb-1">
              <CheckCircle2 className="size-4 text-teal-600 shrink-0" />
              <span>{shiftNarrative.title} vs MỨC SỬ DỤNG ĐỘI XE ({activeShippersCountInDay}/16)</span>
            </div>
            <p className="mt-1">
              <b>1. Hiệu quả giao hàng ({Math.round(dayCompletionPct)}%):</b> {shiftNarrative.completionText}
              <br />
              <b>2. Đội xe vận hành ({activeShippersCountInDay}/{session.shippers.length}):</b> {shiftNarrative.fleetText}
              <br />
              <b>3. Đúng hạn SLA:</b> {shiftNarrative.slaText}
            </p>
          </div>

          <div style={{ overflow: "auto", marginTop: "8px", maxHeight: "340px" }}>
          <table className="report-table">
            <thead>
              <tr>
                <th>Shipper</th>
                <th>Đơn nhận ca này</th>
                <th>Đã giao</th>
                <th>Chưa giao</th>
                <th>Tải trọng ca</th>
                <th>Cự ly ca</th>
              </tr>
            </thead>
            <tbody id="shipReportRows">
              {session.shippers.map((s) => {
                const sNum = s.shipper_id.replace("shipper_", "");
                const perf = dayReportSnapshot?.shipper_performance?.find((p) => p.shipper_id === s.shipper_id);

                let assignedCount = 0;
                let completedCount = 0;
                let pendingCount = 0;
                let weightKg = 0;
                let distKm = 0;

                if (perf) {
                  assignedCount = perf.assigned_count;
                  completedCount = perf.completed_count;
                  pendingCount = perf.pending_count;
                  weightKg = perf.weight_kg;
                  distKm = perf.distance_km || 0;
                } else {
                  // Live day evaluation for viewDay
                  const sDeliveredToday = session.orders.filter(
                    (o) =>
                      isDeliveredInViewDay(o) &&
                      (o.assigned_shipper_id === s.shipper_id ||
                        s.assigned_order_ids.includes(o.order_id) ||
                        (s.completed_order_ids && s.completed_order_ids.includes(o.order_id)))
                  );
                  const sPendingToday = session.orders.filter(
                    (o) =>
                      o.release_day <= viewDay &&
                      o.delivery_status !== "DELIVERED" &&
                      (o.assigned_shipper_id === s.shipper_id || s.assigned_order_ids.includes(o.order_id))
                  );
                  assignedCount = sDeliveredToday.length + sPendingToday.length;
                  completedCount = sDeliveredToday.length;
                  pendingCount = sPendingToday.length;
                  const sOrdersToday = [...sDeliveredToday, ...sPendingToday];
                  weightKg = sOrdersToday.reduce((sum, o) => sum + (o.weight_kg || 0), 0);
                  distKm = s.distance_travelled_km || 0;
                }

                return (
                  <tr key={s.shipper_id}>
                    <td>
                      <b>Shipper #{pad(sNum)}</b>
                    </td>
                    <td>{assignedCount}</td>
                    <td style={{ color: "#14866a", fontWeight: 750 }}>{completedCount}</td>
                    <td>{pendingCount}</td>
                    <td>{weightKg.toFixed(1)} kg</td>
                    <td>{distKm > 0 ? `${distKm.toFixed(1)} km` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        </div>
        )}
      </div>
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
