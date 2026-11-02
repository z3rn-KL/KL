import React, { useState, useEffect } from "react";
import { api, RLDecisionTrace, RLRouteMetrics } from "../../lib/api";
import {
  Scale,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingDown,
  Award,
  Sparkles,
  Zap,
  Info,
  Clock,
  Layers,
} from "lucide-react";

export const RLBaselineExplorer: React.FC = () => {
  const [shipperId, setShipperId] = useState<string>("shipper_08");
  const [day, setDay] = useState<number>(1);
  const [data, setData] = useState<RLDecisionTrace | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    api
      .rlDecisionTrace({ shipperId, day, seed: 42 })
      .then((res) => {
        if (isMounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Lỗi tải dữ liệu so sánh");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [shipperId, day]);

  const featuredShippers = [
    { id: "shipper_08", label: "Shipper #08 (7 đơn)" },
    { id: "shipper_11", label: "Shipper #11 (5 đơn)" },
    { id: "shipper_12", label: "Shipper #12 (5 đơn)" },
    { id: "shipper_03", label: "Shipper #03 (6 đơn)" },
  ];

  const getMetric = (algo: "proposed_rl" | "nearest_neighbor" | "clarke_wright"): RLRouteMetrics | null => {
    return data?.metrics?.[algo] ?? null;
  };

  const rlM = getMetric("proposed_rl");
  const nnM = getMetric("nearest_neighbor");
  const cwM = getMetric("clarke_wright");

  return (
    <div className="card" style={{ marginBottom: 24, padding: "24px 28px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: 16,
          marginBottom: 20,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 34,
                height: 34,
                borderRadius: 8,
                background: "rgba(139, 92, 246, 0.12)",
                color: "#8b5cf6",
              }}
            >
              <Scale size={20} />
            </span>
            <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>
              RL vs Baseline Explorer — So Sánh Thực Nghiệm Công Bằng
            </h3>
            <span
              style={{
                fontSize: 11,
                padding: "3px 8px",
                borderRadius: 999,
                fontWeight: 600,
                background: "rgba(16, 185, 129, 0.12)",
                color: "#10b981",
              }}
            >
              Cùng Điều Kiện Đầu Vào 100%
            </span>
          </div>
          <p style={{ margin: "6px 0 0 0", fontSize: 13, color: "var(--muted)" }}>
            Đối chiếu trực diện giữa Proposed RL, Nearest Neighbor và Clarke-Wright trên cùng tập đơn hàng, depot, vận tốc và thời gian phục vụ.
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {featuredShippers.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`btn small ${shipperId === s.id ? "primary" : "outline"}`}
              onClick={() => setShipperId(s.id)}
              style={{ fontSize: 12, padding: "5px 10px" }}
            >
              {s.label}
            </button>
          ))}
          <select
            value={shipperId}
            onChange={(e) => setShipperId(e.target.value)}
            className="input small"
            style={{ fontSize: 12, padding: "4px 8px" }}
          >
            {Array.from({ length: 16 }, (_, i) => {
              const id = `shipper_${String(i + 1).padStart(2, "0")}`;
              return (
                <option key={id} value={id}>
                  {id}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Identical Conditions Protocol Banner */}
      <div
        style={{
          background: "var(--surface)",
          padding: "12px 16px",
          borderRadius: 8,
          border: "1px solid var(--border)",
          marginBottom: 20,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: 12,
          fontSize: 12,
        }}
      >
        <div>
          <span style={{ color: "var(--muted)" }}>Depot Trung Tâm:</span>
          <strong style={{ display: "block", color: "var(--text)" }}>10.7744, 106.6575 (Quận 10)</strong>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Khoảng cách & Vận tốc:</span>
          <strong style={{ display: "block", color: "var(--text)" }}>Haversine • 25.0 km/h</strong>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Thời gian giao mỗi điểm:</span>
          <strong style={{ display: "block", color: "var(--text)" }}>10.0 phút / đơn</strong>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Ca làm việc (Shift Limit):</span>
          <strong style={{ display: "block", color: "var(--text)" }}>600.0 phút (10 giờ)</strong>
        </div>
      </div>

      {loading && (
        <div style={{ padding: "50px 0", textAlign: "center", color: "var(--muted)" }}>
          Đang tính toán ma trận và sinh tuyến cho 3 thuật toán...
        </div>
      )}

      {error && (
        <div style={{ padding: 16, borderRadius: 8, background: "rgba(239, 68, 68, 0.1)", color: "#ef4444" }}>
          {error}
        </div>
      )}

      {!loading && !error && data && rlM && nnM && cwM && (
        <div>
          {/* Comparison Cards Grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 16,
              marginBottom: 20,
            }}
          >
            {/* Proposed RL Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "16px 18px",
                borderRadius: 8,
                border: "2px solid #10b981",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: -10,
                  right: 14,
                  background: "#10b981",
                  color: "#ffffff",
                  fontSize: 10,
                  fontWeight: 800,
                  padding: "2px 8px",
                  borderRadius: 999,
                }}
              >
                PROPOSED RL
              </div>
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 700, color: "#10b981" }}>
                Proposed Candidate-Scoring DQN
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 14 }}>{rlM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)" }}>{rlM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: "#10b981", fontWeight: 700 }}>{rlM.on_time_rate}%</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Số đơn quá hạn:</span>
                  <strong style={{ color: rlM.late_count > 0 ? "#ef4444" : "#10b981" }}>
                    {rlM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>

            {/* Nearest Neighbor Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "16px 18px",
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 700, color: "var(--text)" }}>
                Nearest Neighbor (Greedy Baseline)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 14 }}>{nnM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)" }}>{nnM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: "#10b981", fontWeight: 700 }}>{nnM.on_time_rate}%</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Số đơn quá hạn:</span>
                  <strong style={{ color: nnM.late_count > 0 ? "#ef4444" : "#10b981" }}>
                    {nnM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>

            {/* Clarke-Wright Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "16px 18px",
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <h4 style={{ margin: "0 0 12px 0", fontSize: 14, fontWeight: 700, color: "var(--text)" }}>
                Clarke-Wright Savings Heuristic
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 14 }}>{cwM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)" }}>{cwM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: cwM.on_time_rate === 100 ? "#10b981" : "#ef4444", fontWeight: 700 }}>
                    {cwM.on_time_rate}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--muted)" }}>Số đơn quá hạn:</span>
                  <strong style={{ color: cwM.late_count > 0 ? "#ef4444" : "#10b981" }}>
                    {cwM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>
          </div>

          {/* Sequence Comparison Visualizer */}
          <div
            style={{
              background: "var(--panel-bg)",
              padding: "16px 20px",
              borderRadius: 8,
              border: "1px solid var(--border)",
              marginBottom: 20,
            }}
          >
            <h4 style={{ margin: "0 0 14px 0", fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
              So Sánh Thứ Tự Thăm Điểm Dọc Tuyến (Stop Sequencing)
            </h4>

            {/* Proposed RL Sequence */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <strong style={{ fontSize: 12, color: "#10b981", minWidth: 160 }}>
                  Proposed RL Route:
                </strong>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                  {data.routes.proposed_rl.map((oid, idx) => {
                    const ord = data.orders_metadata.find((o) => o.order_id === oid);
                    const isDivFromNN = data.routes.nearest_neighbor[idx] !== oid;
                    return (
                      <React.Fragment key={oid}>
                        <ArrowRight size={12} color="var(--muted)" />
                        <span
                          className="badge small"
                          style={{
                            background: isDivFromNN ? "rgba(245, 158, 11, 0.25)" : "var(--surface)",
                            border: isDivFromNN ? "1px solid #f59e0b" : "1px solid var(--border)",
                            color: isDivFromNN ? "#f59e0b" : "var(--text)",
                            fontWeight: isDivFromNN ? 700 : 500,
                          }}
                        >
                          #{ord?.order_code ?? oid.slice(-4)}
                          {isDivFromNN && " *"}
                        </span>
                      </React.Fragment>
                    );
                  })}
                  <ArrowRight size={12} color="var(--muted)" />
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                </div>
              </div>
            </div>

            {/* Nearest Neighbor Sequence */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <strong style={{ fontSize: 12, color: "var(--text)", minWidth: 160 }}>
                  Nearest Neighbor Route:
                </strong>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                  {data.routes.nearest_neighbor.map((oid) => {
                    const ord = data.orders_metadata.find((o) => o.order_id === oid);
                    return (
                      <React.Fragment key={oid}>
                        <ArrowRight size={12} color="var(--muted)" />
                        <span className="badge small" style={{ background: "var(--surface)", color: "var(--text)" }}>
                          #{ord?.order_code ?? oid.slice(-4)}
                        </span>
                      </React.Fragment>
                    );
                  })}
                  <ArrowRight size={12} color="var(--muted)" />
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                </div>
              </div>
            </div>

            {/* Clarke-Wright Sequence */}
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <strong style={{ fontSize: 12, color: "var(--text)", minWidth: 160 }}>
                  Clarke-Wright Route:
                </strong>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                  {data.routes.clarke_wright.map((oid) => {
                    const ord = data.orders_metadata.find((o) => o.order_id === oid);
                    return (
                      <React.Fragment key={oid}>
                        <ArrowRight size={12} color="var(--muted)" />
                        <span className="badge small" style={{ background: "var(--surface)", color: "var(--text)" }}>
                          #{ord?.order_code ?? oid.slice(-4)}
                        </span>
                      </React.Fragment>
                    );
                  })}
                  <ArrowRight size={12} color="var(--muted)" />
                  <span className="badge small" style={{ background: "#2563eb", color: "#fff" }}>
                    Depot
                  </span>
                </div>
              </div>
            </div>

            {data.divergence.diverges_from_nn && (
              <div style={{ marginTop: 12, fontSize: 11, color: "#f59e0b" }}>
                * Đánh dấu điểm phân kỳ thứ tự giữa Proposed RL và Nearest Neighbor (bắt đầu tại bước #{data.divergence.first_divergence_step! + 1}).
              </div>
            )}
          </div>

          {/* Scientific Fairness Disclosure */}
          <div
            style={{
              padding: "16px 20px",
              borderRadius: 8,
              background: "rgba(139, 92, 246, 0.05)",
              border: "1px solid rgba(139, 92, 246, 0.2)",
              fontSize: 12,
              lineHeight: 1.6,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 6,
                fontWeight: 700,
                color: "#8b5cf6",
              }}
            >
              <Sparkles size={16} />
              Đánh Giá Tính Công Bằng Khoa Học & Phân Biệt Các Kịch Bản:
            </div>
            <p style={{ margin: "0 0 6px 0", color: "var(--text)" }}>
              • <strong>Phân biệt rạch ròi hai kịch bản Clarke-Wright trong Ablation Study:</strong>
              <br />
              - <em>Nhóm E2 (K-Means + Clarke-Wright):</em> Đạt cự ly ngắn nhất <strong>339.39 km</strong> (Ngày 1) và <strong>1.016.90 km</strong> (tích lũy 3 ngày) với <strong>100.00% SLA đúng hạn (0 đơn trễ)</strong>. Khi các đơn hàng đã được gom cụm chặt chẽ theo khu vực địa bàn, giải thuật tiết kiệm Clarke-Wright phát huy tối đa ưu thế rút ngắn cự ly mà không vi phạm hạn giao.
              <br />
              - <em>Nhóm E1 (Non-cluster + Clarke-Wright):</em> Khi không phân cụm, cự ly đạt 283.34 km nhưng xuất hiện <strong>1 đơn trễ hạn</strong> trên Ngày 1 (98.82% SLA, trễ 3.14 phút) và <strong>tích lũy 3 đơn trễ sau 3 ngày</strong> (tổng cự ly <strong>829.49 km</strong>) do ghép tour xuyên tâm toàn thành phố mà không xét thứ tự hạn giao SLA.
            </p>
            <p style={{ margin: "0 0 6px 0", color: "var(--text)" }}>
              • <strong>So sánh khách quan Proposed RL vs Baselines:</strong>
              <br />
              - <em>Trên mô phỏng 85 đơn/ngày (có phân cụm):</em> Clarke-Wright đạt 339.39 km &lt; Nearest Neighbor 352.70 km &lt; Proposed RL 356.30 km (tất cả đều đạt 100% SLA).
              <br />
              - <em>Trên tập kiểm thử Held-Out 16 Workloads (131 đơn trên OSM via Dijkstra):</em> Clarke-Wright đạt 55.58 km (100% SLA) &lt; Single DQN 57.11 km (100% SLA) &lt; NN 59.58 km (99.24% SLA) &lt; Proposed RL 61.90 km (99.24% SLA).
              <br />
              - <em>Trên 108 Workloads gốc (813 đơn trên OSM):</em> Baseline Clarke-Wright đạt 58.05 km (99.14% SLA, 7 trễ), Single DQN đạt 59.83 km (100% SLA, 0 trễ), Q-Learning đạt 62.26 km (100% SLA, 0 trễ), NN đạt 62.68 km (99.14% SLA, 7 trễ). Proposed RL được phân tách độc lập Train (76) / Val (16) / Test (16) để bảo toàn tính tổng quát hóa khoa học, không đánh giá chồng chéo lên tập huấn luyện nhằm tránh thiên vị dữ liệu (Data Leakage).
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

