import React, { useMemo } from "react";
import { ArrowRight } from "lucide-react";
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

  const total = session.order_summary.total;
  const delivered = session.order_summary.delivered;
  const pending = total - delivered;
  const completionPct = total > 0 ? (delivered / total) * 100 : 0;

  // Carried over orders (not delivered)
  const carriedOrders = useMemo(() => {
    return session.orders.filter((o) => o.delivery_status !== "DELIVERED");
  }, [session.orders]);

  // Operational metrics
  const deliveredOrders = useMemo(() => {
    return session.orders.filter((o) => o.delivery_status === "DELIVERED");
  }, [session.orders]);

  const onTimeCount = useMemo(() => {
    return deliveredOrders.filter((o) => o.sla_status === "ON_TIME").length;
  }, [deliveredOrders]);

  const onTimePct = delivered > 0 ? Math.round((onTimeCount / delivered) * 100) : 0;

  // Single-day vs Cumulative report details
  const latestReport = useMemo(() => {
    if (!session.daily_metrics || session.daily_metrics.length === 0) return null;
    return session.daily_metrics[session.daily_metrics.length - 1];
  }, [session.daily_metrics]);

  const dayDelivered = latestReport ? (latestReport.day_delivered ?? latestReport.delivered) : delivered;
  const dayOnTimeRate = latestReport ? Math.round(((latestReport.day_on_time_rate ?? latestReport.on_time_rate) || 0) * 100) : onTimePct;
  const dayOverdue = latestReport ? (latestReport.day_delivered_overdue ?? latestReport.delivered_overdue) : (delivered - onTimeCount);
  const shiftStatus = latestReport?.shift_status || (session.simulation_clock >= 600 ? "FULL_SHIFT" : "EARLY_CUT");
  const shiftClockMin = latestReport?.simulation_clock_at_end ?? session.simulation_clock;

  // Operating stats
  const totalDistanceKm = useMemo(() => {
    return session.shippers.reduce((sum, s) => sum + (s.distance_travelled_km || 0), 0);
  }, [session.shippers]);

  const operatingHours = (session.simulation_clock || 0) / 60;

  return (
    <div id="view-report" className="view">
      {/* Dark Navy Banner Top */}
      <div className="report-top">
        <div>
          <div className="eyebrow">BÁO CÁO KẾT CA · {session.scenario_id}</div>
          <h2 id="reportTitle">Kết quả ngày {pad(session.current_day)}</h2>
          <p>Tổng kết vận hành và danh sách cần xử lý vào ngày tiếp theo.</p>
          <button
            type="button"
            className="btn primary"
            id="nextDayBtn"
            style={{ marginTop: "20px" }}
            onClick={onNextDay}
            disabled={actionLoading}
          >
            <span>
              {session.current_day >= session.total_days
                ? "Hoàn tất kịch bản 3 ngày"
                : "Bắt đầu ngày tiếp theo"}
            </span>
            <ArrowRight className="icons" />
          </button>
        </div>

        <div>
          <div className="report-big" id="completionPct">
            {Math.round(completionPct)}
            <small>% hoàn thành</small>
          </div>
          <p id="reportSubtitle">
            {delivered} / {total} đơn đã giao trong phiên mô phỏng.
          </p>
        </div>
      </div>

      {/* Grid: Metrics & Carried Over List */}
      <div className="report-grid">
        {/* Left: Operational Metrics */}
        <div className="box report-inner">
          <h2>Hiệu quả giao hàng</h2>
          <p className="subline" style={{ marginTop: "5px" }}>
            Chỉ số vận hành tính từ phiên điều phối thực tế.
          </p>

          <div className="report-metrics">
            <div>
              <b id="repDayDelivered">{dayDelivered}</b>
              <span>Đã giao trong ngày</span>
            </div>
            <div>
              <b id="repDayOnTime" style={{ color: "#0aa58b" }}>{dayDelivered > 0 ? `${dayOnTimeRate}%` : "—"}</b>
              <span>Đúng hạn trong ngày</span>
            </div>
            <div>
              <b id="repDayOverdue" style={{ color: dayOverdue > 0 ? "#e11d48" : "#64748b" }}>{dayOverdue}</b>
              <span>Trễ hạn trong ngày</span>
            </div>
            <div>
              <b id="repTotal">{total}</b>
              <span>Tổng đơn toàn phiên</span>
            </div>
            <div>
              <b id="repDelivered">{delivered}</b>
              <span>Đã giao lũy kế</span>
            </div>
            <div>
              <b id="repOnTime">{delivered > 0 ? `${onTimePct}%` : "—"}</b>
              <span>Đúng hạn lũy kế</span>
            </div>
            <div>
              <b id="repCarry">{pending}</b>
              <span>Tồn đọng chuyển ngày</span>
            </div>
            <div>
              <b id="repDistance">{totalDistanceKm > 0 ? `${totalDistanceKm.toFixed(1)} km` : "—"}</b>
              <span>Cự ly di chuyển</span>
            </div>
            <div>
              <b id="repHours">{operatingHours > 0 ? `${operatingHours.toFixed(1)}h` : "—"}</b>
              <span>Thời lượng ca ({shiftStatus})</span>
            </div>
          </div>

          <div className="stack-bar" id="reportBar">
            <span style={{ width: `${completionPct}%`, background: "#0aa58b" }} />
            <span style={{ width: `${100 - completionPct}%`, background: "#e5eceb" }} />
          </div>

          <div className="small muted" style={{ marginTop: "8px", lineHeight: "1.4" }}>
            <b>Minh bạch chỉ số:</b> Chỉ số &quot;Trong ngày&quot; phản ánh riêng hiệu quả giao trong ca của ngày {session.current_day}. Chỉ số &quot;Lũy kế&quot; phản ánh tỷ lệ trên toàn bộ đơn đã giao tính từ ngày 1 đến hiện tại. Đơn không có cam kết deadline được tính bảo toàn đúng hạn (ON_TIME).
          </div>
        </div>

        {/* Right: Carried Over Orders */}
        <div className="box report-inner">
          <div className="tracking-row">
            <h2>Đơn chuyển ngày</h2>
            <span className="chip orange" id="carryBadge">
              {carriedOrders.length} đơn
            </span>
          </div>
          <p className="subline" style={{ marginTop: "6px" }}>
            Giữ nguyên mã đơn và mốc hạn giao trong phiên điều phối.
          </p>

          <div id="carryList" style={{ maxHeight: "250px", overflowY: "auto", marginTop: "10px" }}>
            {carriedOrders.length > 0 ? (
              carriedOrders.slice(0, 10).map((o) => (
                <div className="carryrow" key={o.order_id}>
                  <div>
                    <strong>{o.order_code || o.order_id}</strong>
                    <small>
                      {o.package_name || "Kiện hàng"} · Cụm {o.cluster_id} · {o.weight_kg.toFixed(1)} kg
                    </small>
                  </div>
                  <span className="chip orange">
                    {o.carryover_count > 0 ? `Tồn cũ (${o.carryover_count}d)` : "Chuyển ngày"}
                  </span>
                </div>
              ))
            ) : (
              <div className="nodata">Không còn đơn tồn đọng. Tất cả đơn đã hoàn tất!</div>
            )}
          </div>
        </div>
      </div>

      {/* 16-Shipper Fleet Performance Table */}
      <div className="box" style={{ padding: "23px", marginBottom: "20px" }}>
        <div className="tracking-row">
          <h2>Hiệu suất đội xe</h2>
          <span className="muted small">16 shipper · tổng hợp theo phiên điều phối</span>
        </div>

        <div style={{ overflow: "auto", marginTop: "12px" }}>
          <table className="report-table">
            <thead>
              <tr>
                <th>Shipper</th>
                <th>Đơn nhận</th>
                <th>Hoàn tất</th>
                <th>Chưa giao</th>
                <th>Tải trọng ban đầu</th>
              </tr>
            </thead>
            <tbody id="shipReportRows">
              {session.shippers.map((s) => {
                const sNum = s.shipper_id.replace("shipper_", "");
                const sOrders = session.orders.filter((o) =>
                  s.assigned_order_ids.includes(o.order_id)
                );
                const sDelivered = sOrders.filter((o) => o.delivery_status === "DELIVERED");
                const sPending = sOrders.length - sDelivered.length;
                const sWeight = sOrders.reduce((sum, o) => sum + (o.weight_kg || 0), 0);

                return (
                  <tr key={s.shipper_id}>
                    <td>
                      <b>Shipper #{pad(sNum)}</b>
                    </td>
                    <td>{sOrders.length}</td>
                    <td style={{ color: "#14866a", fontWeight: 750 }}>{sDelivered.length}</td>
                    <td>{sPending}</td>
                    <td>{sWeight.toFixed(1)} kg</td>
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
