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
      // If no session passed, we can create a temporary session or use an active one
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
      <div style={{ padding: "2rem", textAlign: "center", color: "#64748b" }}>
        <div style={{ display: "inline-block", width: 24, height: 24, border: "3px solid #cbd5e1", borderTopColor: "#0f766e", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
        <p style={{ marginTop: "0.5rem" }}>Đang nạp trace suy luận từ mô hình PyTorch Proposed RL...</p>
      </div>
    );
  }

  if (error || !traceData || traceData.decision_trace.length === 0) {
    return (
      <div style={{ padding: "1.5rem", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, color: "#991b1b" }}>
        <strong>Không thể mở RL Decision Explorer:</strong> {error || "Chưa có dữ liệu bước đi."}
        <button onClick={fetchTrace} style={{ marginLeft: "1rem", padding: "0.25rem 0.75rem", borderRadius: 4, background: "#dc2626", color: "#fff", border: "none", cursor: "pointer" }}>Thử lại</button>
      </div>
    );
  }

  const steps = traceData.decision_trace;
  const currentStep: DecisionTraceStep = steps[currentStepIndex];

  return (
    <div style={{ background: "#ffffff", borderRadius: 12, border: "1px solid #e2e8f0", padding: "1.5rem", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid #f1f5f9", paddingBottom: "1rem", marginBottom: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ background: "#ccfbf1", color: "#0f766e", padding: "0.2rem 0.5rem", borderRadius: 4, fontSize: "0.75rem", fontWeight: 700 }}>EXPLAINABLE RL</span>
            <h3 style={{ margin: 0, fontSize: "1.15rem", color: "#0f172a" }}>RL Decision Explorer (Truy Vết Quyết Định Từng Bước)</h3>
          </div>
          <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "#64748b" }}>
            Quan sát trực tiếp cách mạng nơ-ron <code>CandidateQNetwork</code> chấm điểm Q-Score và chọn hành động tối ưu cho shipper.
          </p>
        </div>
        <div style={{ textAlign: "right", fontSize: "0.8rem", color: "#64748b" }}>
          <div>Mô hình: <strong>{traceData.model_type}</strong></div>
          <div>Checkpoint: <span style={{ fontFamily: "monospace", color: "#0f766e" }}>{traceData.checkpoint.split("/").pop()}</span></div>
        </div>
      </div>

      {/* Control Toolbar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#f8fafc", padding: "0.75rem 1rem", borderRadius: 8, marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <button
            onClick={() => setCurrentStepIndex(0)}
            disabled={currentStepIndex === 0}
            style={{ padding: "0.4rem 0.8rem", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: currentStepIndex === 0 ? "not-allowed" : "pointer" }}
          >
            ⏮ Đầu
          </button>
          <button
            onClick={() => setCurrentStepIndex((p) => Math.max(0, p - 1))}
            disabled={currentStepIndex === 0}
            style={{ padding: "0.4rem 0.8rem", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: currentStepIndex === 0 ? "not-allowed" : "pointer" }}
          >
            ◀ Lùi 1 Bước
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ padding: "0.4rem 1.2rem", borderRadius: 6, border: "none", background: isPlaying ? "#ea580c" : "#0f766e", color: "#fff", fontWeight: 600, cursor: "pointer" }}
          >
            {isPlaying ? "⏸ Tạm Dừng" : "▶ Tự Động Chạy"}
          </button>
          <button
            onClick={() => setCurrentStepIndex((p) => Math.min(steps.length - 1, p + 1))}
            disabled={currentStepIndex >= steps.length - 1}
            style={{ padding: "0.4rem 0.8rem", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: currentStepIndex >= steps.length - 1 ? "not-allowed" : "pointer" }}
          >
            Tiếp 1 Bước ▶
          </button>
          <button
            onClick={() => setCurrentStepIndex(steps.length - 1)}
            disabled={currentStepIndex >= steps.length - 1}
            style={{ padding: "0.4rem 0.8rem", borderRadius: 6, border: "1px solid #cbd5e1", background: "#fff", cursor: currentStepIndex >= steps.length - 1 ? "not-allowed" : "pointer" }}
          >
            Cuối ⏭
          </button>
        </div>
        <div style={{ fontWeight: 600, color: "#334155" }}>
          Bước <span style={{ color: "#0f766e", fontSize: "1.1rem" }}>{currentStepIndex + 1}</span> / {steps.length}
        </div>
      </div>

      {/* Main Grid: Step Decision Details & Candidates */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "1.5rem" }}>
        {/* Left: Step Context & Candidate Evaluations */}
        <div>
          <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", padding: "1rem", borderRadius: 8, marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.85rem", color: "#166534", fontWeight: 600 }}>QUYẾT ĐỊNH TẠI BƯỚC #{currentStep.step_index}:</div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
              <span style={{ background: "#dcfce7", color: "#15803d", padding: "0.25rem 0.6rem", borderRadius: 4, fontWeight: 700 }}>
                {currentStep.current_node === "DEPOT" ? "🏢 Kho Tân Bình" : `📦 ${currentStep.current_node.slice(0, 8)}...`}
              </span>
              <span style={{ color: "#16a34a", fontSize: "1.2rem", fontWeight: 900 }}>➔</span>
              <span style={{ background: "#0f766e", color: "#fff", padding: "0.25rem 0.6rem", borderRadius: 4, fontWeight: 700 }}>
                ⭐ Chọn: {currentStep.chosen_delivery_id.slice(0, 8)}...
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem", marginTop: "0.75rem", fontSize: "0.85rem" }}>
              <div>Q-Score Chọn: <strong>{currentStep.chosen_q_score.toFixed(4)}</strong></div>
              <div>Phần Thưởng Step ($r_t$): <strong>{currentStep.step_reward.toFixed(4)}</strong></div>
              <div>Số Ứng Viên Xét: <strong>{currentStep.candidates_evaluated.length}</strong></div>
            </div>
          </div>

          <h4 style={{ margin: "0 0 0.5rem", fontSize: "0.95rem", color: "#334155" }}>
            Bảng Chấm Điểm Ứng Viên (Candidate Scoring):
          </h4>
          <div style={{ maxHeight: 340, overflowY: "auto", border: "1px solid #e2e8f0", borderRadius: 8 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem", textAlign: "left" }}>
              <thead style={{ background: "#f8fafc", position: "sticky", top: 0, borderBottom: "1px solid #cbd5e1" }}>
                <tr>
                  <th style={{ padding: "0.5rem" }}>Ứng Viên</th>
                  <th style={{ padding: "0.5rem" }}>Cụm</th>
                  <th style={{ padding: "0.5rem" }}>Q-Score Mạng Nơ-ron</th>
                  <th style={{ padding: "0.5rem" }}>Khoảng Cách</th>
                  <th style={{ padding: "0.5rem" }}>Gấp Rút</th>
                  <th style={{ padding: "0.5rem" }}>Trạng Thái</th>
                </tr>
              </thead>
              <tbody>
                {currentStep.candidates_evaluated.map((cand, cIdx) => {
                  const isSelected = cand.delivery_id === currentStep.chosen_delivery_id;
                  return (
                    <tr
                      key={cand.delivery_id}
                      style={{
                        background: isSelected ? "#f0fdf4" : cIdx % 2 === 0 ? "#ffffff" : "#fcfcfc",
                        borderBottom: "1px solid #f1f5f9",
                        fontWeight: isSelected ? 700 : 400,
                      }}
                    >
                      <td style={{ padding: "0.5rem" }}>
                        <code>{cand.delivery_id.slice(0, 8)}...</code>
                        {cand.order_code && <span style={{ color: "#64748b", marginLeft: 4 }}>({cand.order_code})</span>}
                      </td>
                      <td style={{ padding: "0.5rem" }}>Cụm #{cand.cluster_id ?? "-"}</td>
                      <td style={{ padding: "0.5rem", color: isSelected ? "#0f766e" : "#334155" }}>
                        <strong>{cand.q_score.toFixed(4)}</strong>
                      </td>
                      <td style={{ padding: "0.5rem" }}>{cand.feature_dict.norm_distance ?? "-"}</td>
                      <td style={{ padding: "0.5rem" }}>{cand.feature_dict.deadline_tightness ?? "-"}</td>
                      <td style={{ padding: "0.5rem" }}>
                        {isSelected ? (
                          <span style={{ background: "#0f766e", color: "#fff", padding: "0.15rem 0.4rem", borderRadius: 4, fontSize: "0.7rem" }}>
                            ĐƯỢC CHỌN
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>Hạng #{cIdx + 1}</span>
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
          <div style={{ background: "#f8fafc", padding: "1rem", borderRadius: 8, border: "1px solid #e2e8f0", marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#334155", marginBottom: "0.5rem" }}>LŨY KẾ TIẾN TRÌNH TUYẾN:</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.85rem" }}>
              <div style={{ background: "#fff", padding: "0.5rem", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <span style={{ color: "#64748b", fontSize: "0.75rem" }}>Quãng đường:</span>
                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#0f172a" }}>{currentStep.cumulative_distance_km} km</div>
              </div>
              <div style={{ background: "#fff", padding: "0.5rem", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <span style={{ color: "#64748b", fontSize: "0.75rem" }}>Thời gian di chuyển:</span>
                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#0f172a" }}>{currentStep.cumulative_time_minutes} min</div>
              </div>
              <div style={{ background: "#fff", padding: "0.5rem", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <span style={{ color: "#64748b", fontSize: "0.75rem" }}>Số lần đổi cụm:</span>
                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#0f172a" }}>{currentStep.cluster_switches} lần</div>
              </div>
              <div style={{ background: "#fff", padding: "0.5rem", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                <span style={{ color: "#64748b", fontSize: "0.75rem" }}>Tổng phút trễ:</span>
                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: currentStep.lateness_minutes > 0 ? "#dc2626" : "#16a34a" }}>
                  {currentStep.lateness_minutes} min
                </div>
              </div>
            </div>
          </div>

          <div style={{ background: "#fff", padding: "1rem", borderRadius: 8, border: "1px solid #e2e8f0" }}>
            <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#334155", marginBottom: "0.5rem" }}>
              8 ĐẶC TRƯNG TRẠNG THÁI (INVARIANT FEATURES):
            </div>
            {(() => {
              const selectedCand = currentStep.candidates_evaluated.find((c) => c.delivery_id === currentStep.chosen_delivery_id);
              if (!selectedCand) return null;
              const f = selectedCand.feature_dict;
              return (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem", fontSize: "0.75rem" }}>
                  <div>1. Khoảng cách: <strong>{f.norm_distance}</strong></div>
                  <div>2. Thời gian chạy: <strong>{f.norm_travel_time}</strong></div>
                  <div>3. Gấp rút deadline: <strong>{f.deadline_tightness}</strong></div>
                  <div>4. Thời gian chùng: <strong>{f.slack_ratio}</strong></div>
                  <div>5. Góc rẽ lệch (Detour): <strong>{f.detour_penalty}</strong></div>
                  <div>6. Trùng cụm: <strong>{f.cluster_alignment}</strong></div>
                  <div>7. Hướng về Depot: <strong>{f.return_depot_impact}</strong></div>
                  <div>8. Mật độ lân cận: <strong>{f.candidate_density}</strong></div>
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
