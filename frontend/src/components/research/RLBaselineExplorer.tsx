import React, { useState, useEffect } from "react";
import { api, RLDecisionTrace, RLRouteMetrics } from "../../lib/api";
import {
  Scale,
  ArrowRight,
  Sparkles,
  Info,
  CheckCircle2,
  Clock,
  Compass,
  Layers,
  MapPin,
  Calendar,
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
    { id: "shipper_11", label: "Shipper #11 (7 đơn)" },
    { id: "shipper_12", label: "Shipper #12 (6 đơn)" },
    { id: "shipper_03", label: "Shipper #03 (7 đơn)" },
  ];

  const getMetric = (algo: "proposed_rl" | "nearest_neighbor" | "clarke_wright"): RLRouteMetrics | null => {
    return data?.metrics?.[algo] ?? null;
  };

  const rlM = getMetric("proposed_rl");
  const nnM = getMetric("nearest_neighbor");
  const cwM = getMetric("clarke_wright");

  return (
    <div className="card" style={{ marginBottom: 28, padding: "24px 28px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: 16,
          marginBottom: 22,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 38,
                height: 38,
                borderRadius: 10,
                background: "rgba(139, 92, 246, 0.15)",
                color: "#8b5cf6",
              }}
            >
              <Scale size={22} />
            </span>
            <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "var(--text)" }}>
              RL vs Baseline Explorer — So Sánh Thực Nghiệm Công Bằng
            </h3>
            <span
              style={{
                fontSize: 13,
                padding: "3px 10px",
                borderRadius: 999,
                fontWeight: 600,
                background: "rgba(16, 185, 129, 0.15)",
                color: "#10b981",
              }}
            >
              Điều Kiện Đầu Vào Đồng Nhất (Live Comparison)
            </span>
          </div>
          <p style={{ margin: "8px 0 0 0", fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.55 }}>
            Đối chiếu trực diện giữa Proposed RL, Nearest Neighbor và Clarke-Wright trên cùng tập đơn hàng, depot, vận tốc và thời gian phục vụ.
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          {featuredShippers.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`btn small ${shipperId === s.id ? "primary" : "outline"}`}
              onClick={() => setShipperId(s.id)}
              style={{ fontSize: 13, padding: "6px 12px", fontWeight: 600 }}
            >
              {s.label}
            </button>
          ))}
          <select
            value={shipperId}
            onChange={(e) => setShipperId(e.target.value)}
            className="input small"
            style={{ fontSize: 13, padding: "6px 10px", fontWeight: 500, color: "var(--text)", background: "var(--surface)" }}
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
          padding: "16px 20px",
          borderRadius: 10,
          border: "1px solid var(--border)",
          marginBottom: 24,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 16,
        }}
      >
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <MapPin size={18} color="#3b82f6" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <span style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>Depot Trung Tâm:</span>
            <strong style={{ display: "block", color: "var(--text)", fontSize: 14, marginTop: 2 }}>
              10.7744, 106.6575 (Quận 10)
            </strong>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <Compass size={18} color="#10b981" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <span style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>Khoảng cách & Vận tốc:</span>
            <strong style={{ display: "block", color: "var(--text)", fontSize: 14, marginTop: 2 }}>
              Haversine • 25.0 km/h
            </strong>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <Clock size={18} color="#f59e0b" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <span style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>Thời gian phục vụ / đơn:</span>
            <strong style={{ display: "block", color: "var(--text)", fontSize: 14, marginTop: 2 }}>
              10.0 phút / điểm giao
            </strong>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <Calendar size={18} color="#8b5cf6" style={{ marginTop: 2, flexShrink: 0 }} />
          <div>
            <span style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>Giới hạn ca làm việc:</span>
            <strong style={{ display: "block", color: "var(--text)", fontSize: 14, marginTop: 2 }}>
              600.0 phút (10.0 giờ)
            </strong>
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ padding: "50px 0", textAlign: "center", color: "var(--text-secondary)", fontSize: 14 }}>
          Đang tính toán ma trận và sinh tuyến cho 3 thuật toán...
        </div>
      )}

      {error && (
        <div style={{ padding: "16px 20px", borderRadius: 8, background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.25)", color: "#ef4444", fontSize: 14, marginBottom: 20 }}>
          {error}
        </div>
      )}

      {!loading && !error && data && rlM && nnM && cwM && (
        <div>
          {/* Comparison Cards Grid - Objective Presentation */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(290px, 1fr))",
              gap: 18,
              marginBottom: 24,
            }}
          >
            {/* Proposed RL Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "18px 20px",
                borderRadius: 10,
                border: "1.5px solid rgba(16, 185, 129, 0.6)",
                position: "relative",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  Proposed Candidate-Scoring DQN
                </h4>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: "rgba(16, 185, 129, 0.15)",
                    color: "#10b981",
                  }}
                >
                  RL AGENT
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 19, fontWeight: 700 }}>{rlM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)", fontSize: 15 }}>{rlM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: rlM.on_time_rate === 100 ? "#10b981" : "#ef4444", fontSize: 15, fontWeight: 700 }}>
                    {rlM.on_time_rate}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Số đơn quá hạn:</span>
                  <strong style={{ color: rlM.late_count > 0 ? "#ef4444" : "#10b981", fontSize: 14 }}>
                    {rlM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>

            {/* Nearest Neighbor Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "18px 20px",
                borderRadius: 10,
                border: "1.5px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  Nearest Neighbor (Greedy Baseline)
                </h4>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: "rgba(100, 116, 139, 0.15)",
                    color: "var(--text-secondary)",
                  }}
                >
                  HEURISTIC
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 19, fontWeight: 700 }}>{nnM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)", fontSize: 15 }}>{nnM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: nnM.on_time_rate === 100 ? "#10b981" : "#ef4444", fontSize: 15, fontWeight: 700 }}>
                    {nnM.on_time_rate}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Số đơn quá hạn:</span>
                  <strong style={{ color: nnM.late_count > 0 ? "#ef4444" : "#10b981", fontSize: 14 }}>
                    {nnM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>

            {/* Clarke-Wright Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "18px 20px",
                borderRadius: 10,
                border: "1.5px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  Clarke-Wright Savings Heuristic
                </h4>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: "rgba(100, 116, 139, 0.15)",
                    color: "var(--text-secondary)",
                  }}
                >
                  SAVINGS
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Quãng đường tour:</span>
                  <strong style={{ color: "var(--text)", fontSize: 19, fontWeight: 700 }}>{cwM.distance_km} km</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Thời gian hoàn thành:</span>
                  <strong style={{ color: "var(--text)", fontSize: 15 }}>{cwM.total_time_minutes} phút</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Tỷ lệ đúng hạn SLA:</span>
                  <strong style={{ color: cwM.on_time_rate === 100 ? "#10b981" : "#ef4444", fontSize: 15, fontWeight: 700 }}>
                    {cwM.on_time_rate}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Số đơn quá hạn:</span>
                  <strong style={{ color: cwM.late_count > 0 ? "#ef4444" : "#10b981", fontSize: 14 }}>
                    {cwM.late_count} đơn
                  </strong>
                </div>
              </div>
            </div>
          </div>

          {/* Sequence Comparison Visualizer — PRIORITY REDESIGN */}
          <div
            style={{
              background: "var(--panel-bg)",
              padding: "20px 22px",
              borderRadius: 10,
              border: "1px solid var(--border)",
              marginBottom: 24,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
              <h4 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "var(--text)" }}>
                So Sánh Thứ Tự Thăm Điểm Dọc Tuyến (Stop Sequencing Lanes)
              </h4>
              <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                Thứ tự các điểm giao thực tế xuất phát từ Depot và trở về Depot
              </span>
            </div>

            {/* Helper to render individual route lane with non-breaking waypoint groups and horizontal scroll */}
            {(() => {
              const renderRouteLane = (
                laneTitle: string,
                metricText: string,
                titleColor: string,
                rightNote: React.ReactNode,
                routeStops: string[],
                compareStops?: string[]
              ) => {
                // Filter out any duplicate depot string if backend provided it
                const cleanStops = (routeStops || []).filter(
                  (oid) => !String(oid).toLowerCase().includes("depot")
                );

                return (
                  <div
                    style={{
                      background: "var(--surface)",
                      padding: "14px 16px",
                      borderRadius: 8,
                      border: "1px solid var(--border)",
                      marginBottom: 12,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        flexWrap: "wrap",
                        gap: 8,
                        marginBottom: 10,
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                        <strong style={{ fontSize: 14.5, color: titleColor, fontWeight: 700 }}>
                          {laneTitle}
                        </strong>
                        <span style={{ fontSize: 12.5, fontWeight: 500, color: "var(--text-secondary)" }}>
                          {metricText}
                        </span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                        {rightNote}
                      </div>
                    </div>

                    <div
                      className="custom-scrollbar"
                      style={{
                        display: "flex",
                        alignItems: "center",
                        overflowX: "auto",
                        overflowY: "hidden",
                        paddingBottom: 8,
                        paddingTop: 4,
                        scrollbarWidth: "thin",
                        WebkitOverflowScrolling: "touch",
                        maxWidth: "100%",
                      }}
                    >
                      {/* Start Depot */}
                      <div style={{ display: "inline-flex", alignItems: "center", flexShrink: 0 }}>
                        <span
                          style={{
                            background: "#1d4ed8",
                            color: "#ffffff",
                            fontSize: 13,
                            fontWeight: 700,
                            padding: "4px 10px",
                            borderRadius: 6,
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 5,
                            whiteSpace: "nowrap",
                          }}
                        >
                          <span>🏢</span>
                          <span>Depot (Xuất phát)</span>
                        </span>
                      </div>

                      {/* Waypoints */}
                      {cleanStops.map((oid, idx) => {
                        const ord = data.orders_metadata.find((o) => o.order_id === oid);
                        const isDivergent = compareStops && compareStops[idx] !== oid;

                        return (
                          <div
                            key={`${oid}-${idx}`}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              flexShrink: 0,
                            }}
                          >
                            <ArrowRight
                              size={14}
                              style={{
                                color: "var(--text-secondary)",
                                margin: "0 6px",
                                flexShrink: 0,
                                opacity: 0.85,
                              }}
                            />
                            <span
                              title={`Đơn hàng: ${ord?.order_code ?? oid}`}
                              style={{
                                background: isDivergent ? "rgba(245, 158, 11, 0.18)" : "var(--panel-bg)",
                                border: isDivergent ? "1.5px solid #d97706" : "1px solid var(--border)",
                                color: isDivergent ? "#d97706" : "var(--text)",
                                fontWeight: isDivergent ? 700 : 600,
                                fontSize: 13,
                                padding: "4px 10px",
                                borderRadius: 6,
                                whiteSpace: "nowrap",
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 4,
                              }}
                            >
                              <span>#{ord?.order_code ?? oid.slice(-4)}</span>
                              {isDivergent && <span style={{ fontSize: 11, color: "#d97706" }}>*</span>}
                            </span>
                          </div>
                        );
                      })}

                      {/* End Depot */}
                      <div style={{ display: "inline-flex", alignItems: "center", flexShrink: 0 }}>
                        <ArrowRight
                          size={14}
                          style={{
                            color: "var(--text-secondary)",
                            margin: "0 6px",
                            flexShrink: 0,
                            opacity: 0.85,
                          }}
                        />
                        <span
                          style={{
                            background: "#1e3a8a",
                            color: "#ffffff",
                            fontSize: 13,
                            fontWeight: 700,
                            padding: "4px 10px",
                            borderRadius: 6,
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 5,
                            whiteSpace: "nowrap",
                          }}
                        >
                          <span>🏢</span>
                          <span>Depot (Kết thúc)</span>
                        </span>
                      </div>
                    </div>
                  </div>
                );
              };

              return (
                <>
                  {/* Lane 1: Proposed RL Sequence */}
                  {renderRouteLane(
                    "Proposed RL Route",
                    `(${rlM.distance_km} km • ${rlM.on_time_rate}% SLA)`,
                    "#10b981",
                    data.divergence.diverges_from_nn ? (
                      <span style={{ fontSize: 12.5, color: "#d97706", fontWeight: 600 }}>
                        * Phân kỳ khỏi NN tại bước #{data.divergence.first_divergence_step! + 1}
                      </span>
                    ) : (
                      <span style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                        Mô hình tối ưu hóa đa mục tiêu
                      </span>
                    ),
                    data.routes.proposed_rl,
                    data.routes.nearest_neighbor
                  )}

                  {/* Lane 2: Nearest Neighbor Sequence */}
                  {renderRouteLane(
                    "Nearest Neighbor Route",
                    `(${nnM.distance_km} km • ${nnM.on_time_rate}% SLA)`,
                    "var(--text)",
                    <span style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                      Baseline tham lam theo cự ly
                    </span>,
                    data.routes.nearest_neighbor
                  )}

                  {/* Lane 3: Clarke-Wright Sequence */}
                  {renderRouteLane(
                    "Clarke-Wright Route",
                    `(${cwM.distance_km} km • ${cwM.on_time_rate}% SLA)`,
                    "var(--text)",
                    <span style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                      Heuristic tiết kiệm cự ly (Savings)
                    </span>,
                    data.routes.clarke_wright
                  )}
                </>
              );
            })()}
          </div>

          {/* Scientific Fairness & Structured Interpretation — SECTION 5.5 */}
          <div
            style={{
              padding: "20px 24px",
              borderRadius: 10,
              background: "var(--surface)",
              border: "1px solid var(--border)",
              fontSize: 14,
              lineHeight: 1.6,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 14,
                fontWeight: 700,
                color: "#8b5cf6",
                fontSize: 15,
              }}
            >
              <Sparkles size={18} />
              Đánh Giá Tính Công Bằng Khoa Học & Phân Biệt Các Kịch Bản Thực Nghiệm:
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Point 1: Identical protocol */}
              <div>
                <strong style={{ color: "var(--text)", display: "block", marginBottom: 3 }}>
                  1. Đối sánh thực thi trực tiếp trên cùng điều kiện đầu vào (Live Comparison):
                </strong>
                <span style={{ color: "var(--text-secondary)" }}>
                  Cả ba giải thuật được tính toán trực tiếp trên cùng tập đơn hàng của shipper trong ca làm việc, xuất phát cùng vị trí kho Tân Bình, cùng vận tốc chuẩn 25.0 km/h, cùng thời gian phục vụ 10 phút/điểm và thời lượng ca làm việc 10 giờ.
                </span>
              </div>

              {/* Point 2: Visual observation */}
              <div>
                <strong style={{ color: "var(--text)", display: "block", marginBottom: 3 }}>
                  2. Kết quả quan sát khách quan trên ca làm việc hiện hành:
                </strong>
                <span style={{ color: "var(--text-secondary)" }}>
                  Trên phân bổ đơn hàng đã gom cụm (Ngày 1), Clarke-Wright đạt cự ly tối ưu <strong>{cwM.distance_km} km</strong> &le; Nearest Neighbor <strong>{nnM.distance_km} km</strong> &le; Proposed RL <strong>{rlM.distance_km} km</strong> (cả 3 đều đạt 100% SLA đúng hạn). Khi các đơn hàng đã nằm trong cùng khu vực địa bàn, thuật toán tiết kiệm Clarke-Wright phát huy tối đa ưu thế rút ngắn cự ly mà không lo vi phạm hạn giao.
                </span>
              </div>

              {/* Point 3: E1 vs E2 Ablation distinction */}
              <div>
                <strong style={{ color: "var(--text)", display: "block", marginBottom: 3 }}>
                  3. Đối chiếu với kết quả Historical Benchmark đã cố định (Ablation Study E1 vs E2):
                </strong>
                <ul style={{ margin: "4px 0 0 18px", padding: 0, color: "var(--text-secondary)" }}>
                  <li style={{ marginBottom: 4 }}>
                    <em style={{ color: "var(--text)", fontWeight: 600 }}>Nhóm E2 (K-Means + Clarke-Wright):</em> Đạt cự ly ngắn nhất <strong>339.39 km</strong> (Ngày 1) và <strong>1.016.90 km</strong> (tích lũy 3 ngày) với <strong>100.00% SLA đúng hạn (0 đơn trễ)</strong>.
                  </li>
                  <li>
                    <em style={{ color: "var(--text)", fontWeight: 600 }}>Nhóm E1 (Non-cluster + Clarke-Wright):</em> Khi không phân cụm, cự ly đạt 283.34 km nhưng xuất hiện <strong>1 đơn trễ hạn</strong> trên Ngày 1 (98.82% SLA, trễ 3.14 phút) và <strong>tích lũy 3 đơn trễ sau 3 ngày</strong> (tổng cự ly 829.49 km) do ghép tour xuyên tâm toàn thành phố mà không xét thứ tự hạn giao SLA.
                  </li>
                </ul>
              </div>

              {/* Point 4: Scope & Limits */}
              <div>
                <strong style={{ color: "var(--text)", display: "block", marginBottom: 3 }}>
                  4. Phạm vi và giới hạn kết luận khoa học (Tập kiểm thử độc lập 16 Held-Out Workloads):
                </strong>
                <span style={{ color: "var(--text-secondary)" }}>
                  Trên tập kiểm thử Held-Out 16 Workloads độc lập (131 đơn trên mạng đường OpenStreetMap via Dijkstra), Clarke-Wright đạt 55.58 km (100% SLA), Single DQN đạt 57.11 km (100% SLA), NN đạt 59.58 km (99.24% SLA), Proposed RL đạt 61.90 km (99.24% SLA). Proposed RL được phân tách độc lập Train (76) / Val (16) / Test (16) để bảo toàn tính tổng quát hóa khoa học, không đánh giá chồng chéo lên tập huấn luyện nhằm tránh thiên vị dữ liệu (Data Leakage).
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
