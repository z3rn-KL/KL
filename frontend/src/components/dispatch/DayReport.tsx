import React, { useState, useEffect, useMemo } from "react";
import { ArrowRight, Calendar, CheckCircle2, Clock, ShieldAlert, Truck, AlertTriangle } from "lucide-react";
import { DispatchSessionState } from "@/lib/api";

export interface DayReportProps {
  session: DispatchSessionState;
  onNextDay: () => Promise<void>;
  onSwitchToPlanning: () => void;
  onSwitchToLive: () => void;
  actionLoading: boolean;
  actionError: string | null;
  actionSuccess: string | null;
}

export function DayReport({
  session,
  onNextDay,
  onSwitchToPlanning,
  onSwitchToLive,
  actionLoading,
  actionError,
  actionSuccess,
}: DayReportProps) {
  const pad = (n: number | string) => String(n).padStart(2, "0");

  const [viewDay, setViewDay] = useState<number>(session.current_day);

  useEffect(() => {
    setViewDay(session.current_day);
  }, [session.current_day]);

  // Historical snapshot for viewDay if already recorded in daily_metrics
  const dayReportSnapshot = useMemo(() => {
    if (!session.daily_metrics || session.daily_metrics.length === 0) return null;
    return session.daily_metrics.find((m) => m.day === viewDay) || null;
  }, [session.daily_metrics, viewDay]);

  const isHistoricalDay = dayReportSnapshot !== null;

  // 1. New Orders Released Today
  const newOrdersToday = isHistoricalDay
    ? (dayReportSnapshot.new_orders_today ?? 85)
    : session.orders.filter((o) => o.release_day === viewDay).length;

  // 2. Carryover Orders In from previous day
  const carriedIn = isHistoricalDay
    ? (dayReportSnapshot.carried_over_in ?? 0)
    : session.orders.filter((o) => o.release_day < viewDay && o.delivery_status !== "DELIVERED").length;

  // 3. Total Eligible Orders to Process
  const totalToProcess = isHistoricalDay
    ? (dayReportSnapshot.total_orders_to_process ?? (newOrdersToday + carriedIn))
    : (newOrdersToday + carriedIn);

  // 4. Delivered Today
  const dayDelivered = isHistoricalDay
    ? (dayReportSnapshot.day_delivered ?? 0)
    : session.orders.filter((o) => o.delivered_day === viewDay || (o.delivered_day == null && o.delivery_status === "DELIVERED" && viewDay === 1)).length;

  // 5. Daily SLA On-time Rate
  const dayOnTimeRate = isHistoricalDay
    ? Math.round(((dayReportSnapshot.day_sla_rate ?? dayReportSnapshot.day_on_time_rate) || 0) * 100)
    : (dayDelivered > 0
        ? Math.round(
            (session.orders.filter((o) => (o.delivered_day === viewDay || (o.delivered_day == null && viewDay === 1)) && o.sla_status === "ON_TIME").length /
              dayDelivered) *
              100
          )
        : 0);

  // 6. Remaining Orders Today
  const dayRemaining = isHistoricalDay
    ? (dayReportSnapshot.day_remaining ?? dayReportSnapshot.carried_over_out ?? Math.max(0, totalToProcess - dayDelivered))
    : Math.max(0, totalToProcess - dayDelivered);

  // 7. Carryover Out to Next Day
  const carriedOut = isHistoricalDay
    ? (dayReportSnapshot.day_carried_over_out ?? dayReportSnapshot.carried_over_out ?? dayRemaining)
    : dayRemaining;

  // 8. Cumulative Delivered
  const cumulativeDelivered = (isHistoricalDay
    ? (dayReportSnapshot.cumulative_delivered ?? dayReportSnapshot.delivered)
    : session.orders.filter((o) => o.delivery_status === "DELIVERED").length) ?? 0;

  // 9. Cumulative Total Released
  const cumulativeTotalReleased = isHistoricalDay
    ? (dayReportSnapshot.cumulative_total_released ?? dayReportSnapshot.total_orders ?? session.orders.filter((o) => o.release_day <= viewDay).length)
    : session.orders.filter((o) => o.release_day <= viewDay).length;

  // 10. Cumulative SLA Rate
  const cumulativeSlaRate = isHistoricalDay
    ? Math.round(((dayReportSnapshot.cumulative_sla_rate ?? dayReportSnapshot.on_time_rate) || 0) * 100)
    : (cumulativeDelivered > 0
        ? Math.round(
            (session.orders.filter((o) => o.delivery_status === "DELIVERED" && o.sla_status === "ON_TIME").length /
              cumulativeDelivered) *
              100
          )
        : 100);

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
  const carriedOrdersList = useMemo(() => {
    return session.orders.filter((o) => o.release_day <= viewDay && o.delivery_status !== "DELIVERED");
  }, [session.orders, viewDay]);

  // Total distance
  const totalDistanceKm = useMemo(() => {
    if (isHistoricalDay && dayReportSnapshot.total_distance_km) {
      return dayReportSnapshot.total_distance_km;
    }
    return session.shippers.reduce((sum, s) => sum + (s.distance_travelled_km || 0), 0);
  }, [session.shippers, isHistoricalDay, dayReportSnapshot]);

  const operatingHours = isHistoricalDay
    ? ((dayReportSnapshot.simulation_clock_at_end || 600) / 60)
    : (session.simulation_clock / 60);

  return (
    <div id="view-report" className="view">
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
          <div className="eyebrow">BÁO CÁO KẾT CA · {session.scenario_id}</div>
          <h2 id="reportTitle">Kết quả Ngày {pad(viewDay)}</h2>
          <p>
            {isHistoricalDay
              ? `Tổng kết số liệu chính thức ca làm việc Ngày ${viewDay} đã hoàn tất kiểm toán.`
              : `Theo dõi tiến độ vận hành thực tế ca làm việc Ngày ${viewDay} đang diễn ra.`}
          </p>

          {viewDay === session.current_day && session.current_day < session.total_days && (
            <button
              type="button"
              className="btn primary"
              id="nextDayBtn"
              style={{ marginTop: "20px" }}
              onClick={onNextDay}
              disabled={actionLoading}
            >
              <span>Bắt đầu ngày tiếp theo</span>
              <ArrowRight className="icons" />
            </button>
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
              <span>Tồn chuyển tiếp</span>
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
          </div>

          <div className="stack-bar" id="reportBar" style={{ marginTop: "16px" }}>
            <span style={{ width: `${dayCompletionPct}%`, background: "#0aa58b" }} />
            <span style={{ width: `${100 - dayCompletionPct}%`, background: "#e5eceb" }} />
          </div>

          <div className="small muted" style={{ marginTop: "10px", lineHeight: "1.4" }}>
            <b>Minh bạch khoa học:</b> Chỉ số &quot;Đơn mới trong ngày&quot; ({newOrdersToday}) phản ánh khối lượng phát sinh theo kịch bản. &quot;Đơn tồn đầu ngày&quot; ({carriedIn}) là các đơn chuyển tiếp từ ca trước giữ nguyên hạn giao SLA. Chỉ số &quot;Lũy kế đã giao&quot; ({cumulativeDelivered}/{cumulativeTotalReleased}) phản ánh tổng sản lượng hoàn thành từ ngày 1 đến ngày {viewDay}.
          </div>
        </div>

        {/* Right: Carried Over Orders */}
        <div className="box report-inner">
          <div className="tracking-row">
            <h2>Đơn tồn chuyển ca</h2>
            <span className="chip orange" id="carryBadge">
              {carriedOrdersList.length} đơn
            </span>
          </div>
          <p className="subline" style={{ marginTop: "6px" }}>
            Bảo toàn mốc cam kết SLA và thứ tự ưu tiên khi chuyển ca.
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
                    {o.carryover_count > 0 ? `Tồn ${o.carryover_count} ngày` : "Tồn chuyển ca"}
                  </span>
                </div>
              ))
            ) : (
              <div className="nodata">Không còn đơn tồn đọng. Tất cả đơn đã giao hoàn tất!</div>
            )}
          </div>
        </div>
      </div>

      {/* 16-Shipper Fleet Performance Table */}
      <div className="box" style={{ padding: "23px", marginBottom: "20px" }}>
        <div className="tracking-row">
          <h2>Hiệu suất đội ngũ shipper — Ngày {pad(viewDay)}</h2>
          <span className="muted small">
            {isHistoricalDay ? "Ảnh chụp lịch sử chính thức tại thời điểm chốt ca" : "Số liệu theo dõi trực tiếp ca hiện hành"}
          </span>
        </div>

        <div style={{ overflow: "auto", marginTop: "12px" }}>
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
                      (o.delivered_day === viewDay || (o.delivered_day == null && viewDay === 1 && o.delivery_status === "DELIVERED")) &&
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
    </div>
  );
}
