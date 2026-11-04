import React, { useEffect, useState } from "react";
import {
  api,
  TrainingHistoryResponse,
  ProposedRLDatapoint,
  BaselineRLDatapoint,
} from "../../lib/api";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceDot,
} from "recharts";
import {
  Cpu,
  TrendingUp,
  ShieldCheck,
  Sparkles,
  AlertCircle,
  Database,
  Layers,
  CheckCircle2,
  Sliders,
  RefreshCw,
  FileCode,
} from "lucide-react";

type AlgorithmType = "proposed_rl" | "dqn" | "sarsa" | "q_learning";

export const RLTrainingExplorer: React.FC = () => {
  const [activeAlgo, setActiveAlgo] = useState<AlgorithmType>("proposed_rl");
  const [data, setData] = useState<TrainingHistoryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sampleRate, setSampleRate] = useState<number>(50);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    api
      .trainingHistory(activeAlgo, sampleRate)
      .then((res) => {
        if (isMounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Lỗi tải lịch sử huấn luyện");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [activeAlgo, sampleRate]);

  const isProposed = activeAlgo === "proposed_rl";
  const proposedPoints = isProposed
    ? (data?.datapoints as ProposedRLDatapoint[]) || []
    : [];
  const baselinePoints = !isProposed
    ? (data?.datapoints as BaselineRLDatapoint[]) || []
    : [];

  const bestCheckpointEpoch = 30;

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
                background: "rgba(59, 130, 246, 0.15)",
                color: "#3b82f6",
              }}
            >
              <Cpu size={22} />
            </span>
            <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "var(--text)" }}>
              RL Training & Learning Explorer
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
              Dữ Liệu Huấn Luyện Thực Nghiệm (Logged Experiments)
            </span>
          </div>
          <p style={{ margin: "8px 0 0 0", fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.55 }}>
            Trực quan hóa đường cong học tập thực tế, độ hội tụ, hàm mục tiêu và xác minh tính ổn định của mô hình.
          </p>
        </div>

        {/* Algorithm Tabs */}
        <div
          style={{
            display: "flex",
            gap: 8,
            background: "var(--surface)",
            padding: 5,
            borderRadius: 10,
            border: "1px solid var(--border)",
            flexWrap: "wrap",
          }}
        >
          <button
            type="button"
            className={`btn small ${activeAlgo === "proposed_rl" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("proposed_rl")}
            style={{ fontSize: 13, padding: "7px 14px", fontWeight: 600 }}
          >
            Proposed RL (Multi-Workload)
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "dqn" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("dqn")}
            style={{ fontSize: 13, padding: "7px 14px", fontWeight: 600 }}
          >
            Vanilla DQN
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "sarsa" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("sarsa")}
            style={{ fontSize: 13, padding: "7px 14px", fontWeight: 600 }}
          >
            SARSA
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "q_learning" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("q_learning")}
            style={{ fontSize: 13, padding: "7px 14px", fontWeight: 600 }}
          >
            Q-Learning
          </button>
        </div>
      </div>

      {/* Provenance Banner — SECTION 4.2 SUMMARY CARDS */}
      {data && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: 16,
            marginBottom: 24,
          }}
        >
          <div
            style={{
              background: "var(--surface)",
              padding: "16px 18px",
              borderRadius: 10,
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500 }}>
              <Database size={16} color="#3b82f6" />
              <span>Dataset ID / Phân loại kịch bản:</span>
            </div>
            <strong style={{ color: "var(--text)", fontSize: 15, marginTop: 6, display: "block", lineHeight: 1.4 }}>
              {data.dataset_id}
            </strong>
          </div>

          <div
            style={{
              background: "var(--surface)",
              padding: "16px 18px",
              borderRadius: 10,
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500 }}>
              <FileCode size={16} color="#10b981" />
              <span>Tệp dữ liệu thực nghiệm nguồn:</span>
            </div>
            <div style={{ marginTop: 6 }}>
              <code
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  background: "var(--panel-bg)",
                  color: "var(--text)",
                  padding: "4px 8px",
                  borderRadius: 6,
                  border: "1px solid var(--border)",
                  wordBreak: "break-all",
                  display: "inline-block",
                }}
              >
                {data.source_file}
              </code>
            </div>
          </div>

          <div
            style={{
              background: "var(--surface)",
              padding: "16px 18px",
              borderRadius: 10,
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500 }}>
              <TrendingUp size={16} color="#f59e0b" />
              <span>Quy mô dữ liệu huấn luyện:</span>
            </div>
            <strong style={{ color: "var(--text)", fontSize: 15, marginTop: 6, display: "block" }}>
              {isProposed
                ? `${data.total_epochs} epochs (~${data.total_episodes?.toLocaleString()} episodes đa kịch bản)`
                : `${data.total_episodes?.toLocaleString()} episodes (Ngân sách chuẩn hóa • Bước lấy mẫu: ${data.sample_rate})`}
            </strong>
          </div>
        </div>
      )}

      {/* Loading & Error States */}
      {loading && (
        <div
          style={{
            padding: "60px 0",
            textAlign: "center",
            color: "var(--text-secondary)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
            fontSize: 14,
          }}
        >
          <RefreshCw className="animate-spin" size={20} />
          Đang truy xuất lịch sử huấn luyện từ hệ thống tệp kết quả...
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "16px 20px",
            borderRadius: 8,
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.25)",
            color: "#ef4444",
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 20,
            fontSize: 14,
          }}
        >
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Charts & Visuals — SECTION 4.3 */}
      {!loading && !error && data && (
        <div>
          {/* Main Charts Grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(460px, 1fr))",
              gap: 22,
              marginBottom: 24,
            }}
          >
            {/* Chart 1 */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "20px 22px",
                borderRadius: 10,
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                  flexWrap: "wrap",
                  gap: 8,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  {isProposed
                    ? "Mean Training Reward & Objective vs Epoch"
                    : "Episode Reward & Moving Average (MA-50)"}
                </h4>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>
                  {isProposed ? "40 Epochs Multi-Workload" : "5,000 Episodes"}
                </span>
              </div>
              <div style={{ height: 280, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={(isProposed ? proposedPoints : baselinePoints) as any[]}
                    margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis
                      dataKey={isProposed ? "epoch" : "episode"}
                      stroke="var(--text-secondary)"
                      tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                      tickFormatter={(val) => (isProposed ? `E${val}` : `${val}`)}
                    />
                    <YAxis
                      stroke="var(--text-secondary)"
                      tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                      width={48}
                      tickFormatter={(val) => val.toFixed(1)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--panel-bg)",
                        borderColor: "var(--border)",
                        color: "var(--text)",
                        fontSize: 13,
                        borderRadius: 8,
                        boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8, color: "var(--text)" }} />
                    {isProposed ? (
                      <>
                        <Line
                          type="monotone"
                          dataKey="mean_training_reward"
                          name="Training Reward"
                          stroke="#3b82f6"
                          strokeWidth={2.5}
                          dot={{ r: 2 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="mean_training_objective"
                          name="Training Objective J"
                          stroke="#ef4444"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          dot={false}
                        />
                      </>
                    ) : (
                      <>
                        <Line
                          type="monotone"
                          dataKey="total_reward"
                          name="Raw Episode Reward"
                          stroke="#94a3b8"
                          strokeWidth={1}
                          dot={false}
                          opacity={0.4}
                        />
                        <Line
                          type="monotone"
                          dataKey="reward_ma50"
                          name="Reward MA-50"
                          stroke="#3b82f6"
                          strokeWidth={2.5}
                          dot={false}
                        />
                      </>
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ fontSize: 13.5, color: "var(--text-secondary)", marginTop: 10, textAlign: "center", lineHeight: 1.5 }}>
                {isProposed
                  ? "Phần thưởng trung bình qua 40 epochs biến thiên tương thích với batch độ khó kịch bản."
                  : "Đường MA-50 (xanh) lọc nhiễu dao động do epsilon exploration giai đoạn đầu."}
              </div>
            </div>

            {/* Chart 2 */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "20px 22px",
                borderRadius: 10,
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 14,
                  flexWrap: "wrap",
                  gap: 8,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  {isProposed
                    ? "Validation SLA (%) & Mean Distance (km) vs Epoch"
                    : "Quãng đường di chuyển (km) & Hội tụ"}
                </h4>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>
                  {isProposed ? "Đánh giá độc lập mỗi 5 epochs" : "Đơn vị: km"}
                </span>
              </div>
              <div style={{ height: 280, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  {isProposed ? (
                    <LineChart
                      data={proposedPoints.filter((p) => p.validation_ran)}
                      margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="epoch"
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                        tickFormatter={(val) => `Epoch ${val}`}
                      />
                      <YAxis
                        yAxisId="left"
                        domain={[60, 100]}
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                        width={45}
                        tickFormatter={(v) => `${v}%`}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        domain={[50, 80]}
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                        width={50}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--panel-bg)",
                          borderColor: "var(--border)",
                          color: "var(--text)",
                          fontSize: 13,
                          borderRadius: 8,
                          boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8, color: "var(--text)" }} />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="validation_weighted_sla_pct"
                        name="Validation SLA (%)"
                        stroke="#10b981"
                        strokeWidth={2.5}
                        dot={{ r: 4 }}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="validation_mean_distance_km"
                        name="Val Distance (km)"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                      <ReferenceDot
                        yAxisId="left"
                        x={bestCheckpointEpoch}
                        y={98.55}
                        r={7}
                        fill="#10b981"
                        stroke="#ffffff"
                        label={{
                          value: "Best Checkpoint (Epoch 30)",
                          position: "top",
                          fill: "#10b981",
                          fontSize: 13,
                          fontWeight: 700,
                        }}
                      />
                    </LineChart>
                  ) : (
                    <LineChart
                      data={baselinePoints}
                      margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="episode"
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                      />
                      <YAxis
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                        width={48}
                        domain={[50, 80]}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--panel-bg)",
                          borderColor: "var(--border)",
                          color: "var(--text)",
                          fontSize: 13,
                          borderRadius: 8,
                          boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8, color: "var(--text)" }} />
                      <Line
                        type="monotone"
                        dataKey="total_distance_km"
                        name="Raw Distance (km)"
                        stroke="#94a3b8"
                        strokeWidth={1}
                        dot={false}
                        opacity={0.4}
                      />
                      <Line
                        type="monotone"
                        dataKey="distance_ma50"
                        name="Distance MA-50 (km)"
                        stroke="#f59e0b"
                        strokeWidth={2.5}
                        dot={false}
                      />
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </div>
              <div style={{ fontSize: 13.5, color: "var(--text-secondary)", marginTop: 10, textAlign: "center", lineHeight: 1.5 }}>
                {isProposed
                  ? "Điểm chấm xanh là Checkpoint tối ưu toàn cục tại Epoch 30 (Val SLA 98.55%, 2 đơn trễ) được lưu trong tệp best.pt."
                  : "Khoảng cách hội tụ dần về mức 57.3 km sau 2.000 episodes khi chính sách tối ưu hóa tour giao."}
              </div>
            </div>
          </div>

          {/* Hyperparameters & Training vs Inference Grids — SECTIONS 4.4 & 4.5 */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
              gap: 20,
              marginBottom: 20,
            }}
          >
            {/* Hyperparameters Card (Section 4.4: 2-column clean table) */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "20px 22px",
                borderRadius: 10,
                border: "1px solid var(--border)",
              }}
            >
              <h4
                style={{
                  margin: "0 0 16px 0",
                  fontSize: 15,
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: "var(--text)",
                }}
              >
                <Sliders size={18} color="#3b82f6" />
                Siêu tham số thực nghiệm (Hyperparameters)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {Object.entries(data.hyperparameters).map(([key, val]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "baseline",
                      padding: "8px 0",
                      borderBottom: "1px solid var(--border)",
                      gap: 12,
                    }}
                  >
                    <span style={{ color: "var(--text-secondary)", fontSize: 13.5, textTransform: "capitalize", fontWeight: 500 }}>
                      {key.replace(/_/g, " ")}:
                    </span>
                    <strong style={{ color: "var(--text)", fontSize: 14, maxWidth: "65%", textAlign: "right", wordBreak: "break-word" }}>
                      {String(val)}
                    </strong>
                  </div>
                ))}
              </div>
            </div>

            {/* Training vs Inference Card (Section 4.5: 3 distinct subsections) */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "20px 22px",
                borderRadius: 10,
                border: "1px solid var(--border)",
              }}
            >
              <h4
                style={{
                  margin: "0 0 16px 0",
                  fontSize: 15,
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: "var(--text)",
                }}
              >
                <ShieldCheck size={18} color="#10b981" />
                Khác biệt giữa Huấn luyện và Suy luận (Inference)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ padding: "12px 14px", background: "var(--surface)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <span style={{ color: "#2563eb", fontWeight: 700, display: "block", marginBottom: 4, fontSize: 14 }}>
                    Chế độ Huấn luyện (Training Mode):
                  </span>
                  <span style={{ color: "var(--text)", fontSize: 13.5, lineHeight: 1.55 }}>
                    {data.training_vs_inference.training_mode}
                  </span>
                </div>
                <div style={{ padding: "12px 14px", background: "var(--surface)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <span style={{ color: "#059669", fontWeight: 700, display: "block", marginBottom: 4, fontSize: 14 }}>
                    Chế độ Suy luận (Inference Mode):
                  </span>
                  <span style={{ color: "var(--text)", fontSize: 13.5, lineHeight: 1.55 }}>
                    {data.training_vs_inference.inference_mode}
                  </span>
                </div>
                {data.training_vs_inference.action_space && (
                  <div style={{ padding: "12px 14px", background: "var(--surface)", borderRadius: 8, border: "1px solid var(--border)" }}>
                    <span style={{ color: "#7c3aed", fontWeight: 700, display: "block", marginBottom: 4, fontSize: 14 }}>
                      Không gian hành động (Action Space):
                    </span>
                    <span style={{ color: "var(--text)", fontSize: 13.5, lineHeight: 1.55 }}>
                      {data.training_vs_inference.action_space}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Scientific Interpretation Disclosure — SECTION 4.6 */}
          <div
            style={{
              padding: "18px 22px",
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
                marginBottom: 8,
                fontWeight: 700,
                color: "#2563eb",
                fontSize: 15,
              }}
            >
              <Sparkles size={18} />
              Minh bạch khoa học & Diễn giải độ ổn định của giải thuật:
            </div>
            <p style={{ margin: 0, color: "var(--text)", lineHeight: 1.65 }}>
              {data.scientific_interpretation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
