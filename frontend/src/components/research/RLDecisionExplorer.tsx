import React, { useState, useEffect, useRef } from "react";
import { api, DecisionTraceStep, RlDecisionTraceResponse, CandidateEvaluation } from "../../lib/api";

interface Props {
  sessionId?: string;
  shipperId?: string;
}

export const RlDecisionExplorer: React.FC<Props> = ({ sessionId, shipperId }) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [traceData, setTraceData] = useState<RlDecisionTraceResponse | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  const fetchTrace = async () => {
    setLoading(true);
    setError(null);
    try {
      let targetSession = sessionId;
      if (!targetSession) {
        const sRes = await api.createDispatchSession({
          scenario_id: "simulated_stress_day1_to_3",
          routing_algorithm: "proposed_rl",
          priority_policy: "sla_aware",
          total_days: 3,
          seed: 42,
        });
        targetSession = sRes.session_id;
      }
      const res = await api.getSessionRlDecisionTrace(targetSession, { shipper_id: shipperId });
      if (res.success && res.decision_trace.length > 0) {
        setTraceData(res);
        setCurrentStepIndex(0);
      } else {
        setError(res.message || "Không thể tải trace quyết định của RL Agent.");
      }
    } catch (err: any) {
      setError(err?.message || "Lỗi kết nối API giải thích RL.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrace();
  }, [sessionId, shipperId]);

  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = window.setInterval(() => {
        setCurrentStepIndex((prev) => {
          if (!traceData || prev >= traceData.decision_trace.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1500);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, traceData]);

  if (loading) {
    return (
      <div style={{ padding: "2.5rem", textAlign: "center", color: "var(--text-secondary)" }}>
        <div style={{ display: "inline-block", width: 28, height: 28, border: "3px solid var(--border)", borderTopColor: "#0f766e", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
        <p style={{ marginTop: "0.75rem", fontSize: 14 }}>Đang nạp trace suy luận từ mô hình PyTorch Proposed RL...</p>
      </div>
    );
  }

  if (error || !traceData || traceData.decision_trace.length === 0) {
    return (
      <div style={{ padding: "1.5rem", background: "rgba(239, 68, 68, 0.08)", border: "1px solid rgba(239, 68, 68, 0.25)", borderRadius: 10, color: "#ef4444", fontSize: 14 }}>
        <strong>Không thể mở RL Decision Explorer:</strong> {error || "Chưa có dữ liệu bước đi."}
        <button onClick={fetchTrace} className="btn small primary" style={{ marginLeft: "1rem", fontSize: 13 }}>Thử lại</button>
      </div>
    );
  }

  const steps = traceData.decision_trace;
  const currentStep: DecisionTraceStep = steps[currentStepIndex];

  return (
    <div className="card" style={{ background: "var(--panel-bg)", borderRadius: 12, border: "1px solid var(--border)", padding: "22px 26px", marginBottom: 28 }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid var(--border)", paddingBottom: "1rem", marginBottom: "1.25rem", flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            <span style={{ background: "rgba(15, 118, 110, 0.15)", color: "#0f766e", padding: "3px 10px", borderRadius: 6, fontSize: "0.8rem", fontWeight: 700 }}>
              EXPLAINABLE RL
            </span>
            <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700, color: "var(--text)" }}>
              RL Decision Explorer (Truy Vết Quyết Định Từng Bước)
            </h3>
          </div>
          <p style={{ margin: "0.35rem 0 0", fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
            Quan sát trực tiếp cách mạng nơ-ron <code>CandidateQNetwork</code> chấm điểm Q-Score và chọn hành động tối ưu cho shipper.
          </p>
        </div>
        <div style={{ textAlign: "right", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
          <div>Mô hình: <strong style={{ color: "var(--text)" }}>{traceData.model_type}</strong></div>
          <div style={{ marginTop: 2 }}>Checkpoint: <span style={{ fontFamily: "monospace", color: "#0f766e", fontWeight: 600 }}>{traceData.checkpoint.split("/").pop()}</span></div>
        </div>
      </div>

      {/* Control Toolbar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "var(--surface)", padding: "10px 16px", borderRadius: 10, border: "1px solid var(--border)", marginBottom: "1.5rem", flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
          <button
            onClick={() => setCurrentStepIndex(0)}
            disabled={currentStepIndex === 0}
            className="btn small outline"
            style={{ fontSize: 13, padding: "5px 10px" }}
          >
            ⏮ Đầu
          </button>
          <button
            onClick={() => setCurrentStepIndex((p) => Math.max(0, p - 1))}
            disabled={currentStepIndex === 0}
            className="btn small outline"
            style={{ fontSize: 13, padding: "5px 10px" }}
          >
            ◀ Lùi 1 Bước
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ padding: "6px 16px", borderRadius: 6, border: "none", background: isPlaying ? "#ea580c" : "#0f766e", color: "#fff", fontWeight: 700, cursor: "pointer", fontSize: 13 }}
          >
            {isPlaying ? "⏸ Tạm Dừng" : "▶ Tự Động Chạy"}
          </button>
          <button
            onClick={() => setCurrentStepIndex((p) => Math.min(steps.length - 1, p + 1))}
            disabled={currentStepIndex >= steps.length - 1}
            className="btn small outline"
            style={{ fontSize: 13, padding: "5px 10px" }}
          >
            Tiếp 1 Bước ▶
          </button>
          <button
            onClick={() => setCurrentStepIndex(steps.length - 1)}
            disabled={currentStepIndex >= steps.length - 1}
            className="btn small outline"
            style={{ fontSize: 13, padding: "5px 10px" }}
          >
            Cuối ⏭
          </button>
        </div>
        <div style={{ fontWeight: 600, color: "var(--text)", fontSize: 14 }}>
          Bước <span style={{ color: "#0f766e", fontSize: "1.2rem", fontWeight: 800 }}>{currentStepIndex + 1}</span> / {steps.length}
        </div>
      </div>

      {/* Main Grid: Step Decision Details & Candidates */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "1.5rem" }}>
        {/* Left: Step Context & Candidate Evaluations */}
        <div>
          <div style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.3)", padding: "14px 16px", borderRadius: 10, marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.85rem", color: "#059669", fontWeight: 700 }}>QUYẾT ĐỊNH TẠI BƯỚC #{currentStep.step_index}:</div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
              <span style={{ background: "rgba(16, 185, 129, 0.15)", color: "var(--text)", padding: "4px 10px", borderRadius: 6, fontWeight: 700, fontSize: 13 }}>
                {currentStep.current_node === "DEPOT" ? "🏢 Kho Tân Bình" : `📦 ${currentStep.current_node.slice(0, 8)}...`}
              </span>
              <span style={{ color: "#10b981", fontSize: "1.2rem", fontWeight: 900 }}>➔</span>
              <span style={{ background: "#0f766e", color: "#fff", padding: "4px 10px", borderRadius: 6, fontWeight: 700, fontSize: 13 }}>
                ⭐ Chọn: {currentStep.chosen_delivery_id.slice(0, 8)}...
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", marginTop: "0.75rem", fontSize: "0.85rem" }}>
              <div>Q-Score Chọn: <strong style={{ color: "var(--text)", fontSize: 14 }}>{currentStep.chosen_q_score.toFixed(4)}</strong></div>
              <div>Phần Thưởng Step ($r_t$): <strong style={{ color: "var(--text)", fontSize: 14 }}>{currentStep.step_reward.toFixed(4)}</strong></div>
              <div>Số Ứng Viên Xét: <strong style={{ color: "var(--text)", fontSize: 14 }}>{currentStep.candidates_evaluated.length}</strong></div>
            </div>
          </div>

          <h4 style={{ margin: "0 0 0.6rem", fontSize: "1rem", fontWeight: 700, color: "var(--text)" }}>
            Bảng Chấm Điểm Ứng Viên (Candidate Scoring):
          </h4>
          <div style={{ maxHeight: 360, overflowX: "auto", overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10, scrollbarWidth: "thin" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
              <thead style={{ background: "var(--surface)", position: "sticky", top: 0, borderBottom: "1.5px solid var(--border)", zIndex: 1 }}>
                <tr>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 140, whiteSpace: "nowrap" }}>Ứng Viên</th>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 80, whiteSpace: "nowrap" }}>Cụm</th>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 140, whiteSpace: "nowrap" }}>Q-Score Mạng Nơ-ron</th>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 95, whiteSpace: "nowrap" }}>Khoảng Cách</th>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 80, whiteSpace: "nowrap" }}>Gấp Rút</th>
                  <th style={{ padding: "8px 10px", color: "var(--text)", fontWeight: 700, minWidth: 120, whiteSpace: "nowrap" }}>Trạng Thái</th>
                </tr>
              </thead>
              <tbody>
                {currentStep.candidates_evaluated.map((cand, cIdx) => {
                  const isSelected = cand.delivery_id === currentStep.chosen_delivery_id;
                  return (
                    <tr
                      key={cand.delivery_id}
                      style={{
                        background: isSelected ? "rgba(16, 185, 129, 0.12)" : cIdx % 2 === 0 ? "var(--panel-bg)" : "var(--surface)",
                        borderBottom: "1px solid var(--border)",
                        fontWeight: isSelected ? 700 : 400,
                      }}
                    >
                      <td style={{ padding: "8px 10px", minWidth: 140, whiteSpace: "nowrap" }}>
                        <code style={{ fontSize: 13, color: "var(--text)" }}>{cand.delivery_id.slice(0, 8)}...</code>
                        {cand.order_code && <span style={{ color: "var(--text-secondary)", marginLeft: 6, fontWeight: 500 }}>({cand.order_code})</span>}
                      </td>
                      <td style={{ padding: "8px 10px", color: "var(--text)", minWidth: 80, whiteSpace: "nowrap" }}>Cụm #{cand.cluster_id ?? "-"}</td>
                      <td style={{ padding: "8px 10px", color: isSelected ? "#0f766e" : "var(--text)", minWidth: 140, whiteSpace: "nowrap" }}>
                        <strong style={{ fontSize: 14 }}>{cand.q_score.toFixed(4)}</strong>
                      </td>
                      <td style={{ padding: "8px 10px", color: "var(--text-secondary)", minWidth: 95, whiteSpace: "nowrap" }}>{cand.feature_dict.norm_distance ?? "-"}</td>
                      <td style={{ padding: "8px 10px", color: "var(--text-secondary)", minWidth: 80, whiteSpace: "nowrap" }}>{cand.feature_dict.deadline_tightness ?? "-"}</td>
                      <td style={{ padding: "8px 10px", minWidth: 120, whiteSpace: "nowrap" }}>
                        {isSelected ? (
                          <span
                            style={{
                              background: "#0f766e",
                              color: "#fff",
                              padding: "3px 10px",
                              borderRadius: 4,
                              fontSize: "0.75rem",
                              fontWeight: 700,
                              whiteSpace: "nowrap",
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                              flexShrink: 0,
                              letterSpacing: "0.02em",
                            }}
                          >
                            ĐƯỢC CHỌN
                          </span>
                        ) : (
                          <span style={{ color: "var(--text-secondary)", fontWeight: 500, whiteSpace: "nowrap" }}>Hạng #{cIdx + 1}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Cumulative Route Metrics & 8 Invariant State Features */}
        <div>
          <div style={{ background: "var(--surface)", padding: "16px 18px", borderRadius: 10, border: "1px solid var(--border)", marginBottom: "1.25rem" }}>
            <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text)", marginBottom: "0.75rem" }}>LŨY KẾ TIẾN TRÌNH TUYẾN:</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              <div style={{ background: "var(--panel-bg)", padding: "10px 14px", borderRadius: 8, border: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem", fontWeight: 500 }}>Quãng đường:</span>
                <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text)", marginTop: 2 }}>{currentStep.cumulative_distance_km} km</div>
              </div>
              <div style={{ background: "var(--panel-bg)", padding: "10px 14px", borderRadius: 8, border: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem", fontWeight: 500 }}>Thời gian di chuyển:</span>
                <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text)", marginTop: 2 }}>{currentStep.cumulative_time_minutes} min</div>
              </div>
              <div style={{ background: "var(--panel-bg)", padding: "10px 14px", borderRadius: 8, border: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem", fontWeight: 500 }}>Số lần đổi cụm:</span>
                <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text)", marginTop: 2 }}>{currentStep.cluster_switches} lần</div>
              </div>
              <div style={{ background: "var(--panel-bg)", padding: "10px 14px", borderRadius: 8, border: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem", fontWeight: 500 }}>Tổng phút trễ:</span>
                <div style={{ fontSize: "1.2rem", fontWeight: 700, color: currentStep.lateness_minutes > 0 ? "#ef4444" : "#10b981", marginTop: 2 }}>
                  {currentStep.lateness_minutes} min
                </div>
              </div>
            </div>
          </div>

          <div style={{ background: "var(--panel-bg)", padding: "16px 18px", borderRadius: 10, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text)", marginBottom: "0.75rem" }}>
              8 ĐẶC TRƯNG TRẠNG THÁI (INVARIANT FEATURES):
            </div>
            {(() => {
              const selectedCand = currentStep.candidates_evaluated.find((c) => c.delivery_id === currentStep.chosen_delivery_id);
              if (!selectedCand) return null;
              const f = selectedCand.feature_dict;
              const featList = [
                { label: "1. Khoảng cách (d_norm)", val: f.norm_distance },
                { label: "2. Thời gian chạy (t_norm)", val: f.norm_travel_time },
                { label: "3. Gấp rút deadline", val: f.deadline_tightness },
                { label: "4. Thời gian chùng (slack)", val: f.slack_ratio },
                { label: "5. Góc rẽ lệch (detour)", val: f.detour_penalty },
                { label: "6. Trùng cụm (cluster)", val: f.cluster_alignment },
                { label: "7. Hướng về Depot", val: f.return_depot_impact },
                { label: "8. Mật độ lân cận", val: f.candidate_density },
              ];
              return (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem", fontSize: "0.85rem" }}>
                  {featList.map((item, idx) => (
                    <div key={idx} style={{ background: "var(--surface)", padding: "8px 12px", borderRadius: 6, border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>{item.label}:</span>
                      <strong style={{ color: "var(--text)", fontSize: 13.5 }}>{item.val !== undefined && item.val !== null ? item.val : "-"}</strong>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
};

export const RLDecisionExplorer = RlDecisionExplorer;
