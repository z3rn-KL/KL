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
                background: "rgba(59, 130, 246, 0.12)",
                color: "#3b82f6",
              }}
            >
              <Cpu size={20} />
            </span>
            <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>
              RL Training & Learning Explorer
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
              Thực nghiệm gốc 100%
            </span>
          </div>
          <p style={{ margin: "6px 0 0 0", fontSize: 13, color: "var(--muted)" }}>
            Trực quan hóa đường cong học tập thực tế, độ hội tụ, hàm mục tiêu và xác minh tính ổn định của mô hình.
          </p>
        </div>

        {/* Algorithm Tabs */}
        <div
          style={{
            display: "flex",
            gap: 6,
            background: "var(--panel-bg)",
            padding: 4,
            borderRadius: 8,
            border: "1px solid var(--border)",
          }}
        >
          <button
            type="button"
            className={`btn small ${activeAlgo === "proposed_rl" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("proposed_rl")}
            style={{ fontSize: 12, padding: "6px 12px" }}
          >
            Proposed RL (Multi-Workload)
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "dqn" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("dqn")}
            style={{ fontSize: 12, padding: "6px 12px" }}
          >
            Vanilla DQN
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "sarsa" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("sarsa")}
            style={{ fontSize: 12, padding: "6px 12px" }}
          >
            SARSA
          </button>
          <button
            type="button"
            className={`btn small ${activeAlgo === "q_learning" ? "primary" : "outline"}`}
            onClick={() => setActiveAlgo("q_learning")}
            style={{ fontSize: 12, padding: "6px 12px" }}
          >
            Q-Learning
          </button>
        </div>
      </div>

      {/* Provenance Banner */}
      {data && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
            background: "var(--surface)",
            padding: "12px 16px",
            borderRadius: 8,
            border: "1px solid var(--border)",
            marginBottom: 20,
            fontSize: 12,
          }}
        >
          <div>
            <span style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: 5 }}>
              <Database size={13} /> Dataset ID / Phạm vi:
            </span>
            <strong style={{ color: "var(--text)", marginTop: 2, display: "block" }}>
              {data.dataset_id}
            </strong>
          </div>
          <div>
            <span style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: 5 }}>
              <Layers size={13} /> Dữ liệu nguồn:
            </span>
            <code style={{ fontSize: 11, background: "rgba(0,0,0,0.15)", padding: "1px 4px", borderRadius: 4 }}>
              {data.source_file}
            </code>
          </div>
          <div>
            <span style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: 5 }}>
              <TrendingUp size={13} /> Quy mô huấn luyện:
            </span>
            <strong style={{ color: "var(--text)", marginTop: 2, display: "block" }}>
              {isProposed
                ? `${data.total_epochs} epochs (~${data.total_episodes?.toLocaleString()} episodes)`
                : `${data.total_episodes?.toLocaleString()} episodes (Sample rate: ${data.sample_rate})`}
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
            color: "var(--muted)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
          }}
        >
          <RefreshCw className="animate-spin" size={20} />
          Đang truy xuất lịch sử huấn luyện từ hệ thống tệp kết quả...
        </div>
      )}

      {error && (
        <div
          style={{
            padding: 16,
            borderRadius: 8,
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.2)",
            color: "#ef4444",
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 20,
          }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Charts & Visuals */}
      {!loading && !error && data && (
        <div>
          {/* Main Charts Grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(460px, 1fr))",
              gap: 20,
              marginBottom: 20,
            }}
          >
            {/* Chart 1 */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
                  {isProposed
                    ? "Mean Training Reward & Objective vs Epoch"
                    : "Episode Reward & Moving Average (MA-50)"}
                </h4>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  {isProposed ? "40 Epochs Multi-Workload" : "5,000 Episodes"}
                </span>
              </div>
              <div style={{ height: 260, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={(isProposed ? proposedPoints : baselinePoints) as any[]}
                    margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color)" vertical={false} />
                    <XAxis
                      dataKey={isProposed ? "epoch" : "episode"}
                      tick={{ fontSize: 11, fill: "var(--muted)" }}
                      tickFormatter={(val) => (isProposed ? `E${val}` : `${val}`)}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "var(--muted)" }}
                      width={45}
                      tickFormatter={(val) => val.toFixed(1)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--surface)",
                        borderColor: "var(--border)",
                        fontSize: 12,
                        borderRadius: 6,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                    {isProposed ? (
                      <>
                        <Line
                          type="monotone"
                          dataKey="mean_training_reward"
                          name="Training Reward"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={{ r: 2 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="mean_training_objective"
                          name="Training Objective J"
                          stroke="#ef4444"
                          strokeWidth={1.5}
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
                          opacity={0.5}
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
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6, textAlign: "center" }}>
                {isProposed
                  ? "Phần thưởng trung bình qua 40 epochs biến thiên tương thích với batch độ khó kịch bản."
                  : "Đường MA-50 (xanh) lọc nhiễu dao động do epsilon exploration giai đoạn đầu."}
              </div>
            </div>

            {/* Chart 2 */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
              >
                <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
                  {isProposed
                    ? "Validation SLA (%) & Mean Distance (km) vs Epoch"
                    : "Quãng đường di chuyển (km) & Hội tụ"}
                </h4>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  {isProposed ? "Đánh giá độc lập mỗi 5 epochs" : "Đơn vị: km"}
                </span>
              </div>
              <div style={{ height: 260, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  {isProposed ? (
                    <LineChart
                      data={proposedPoints.filter((p) => p.validation_ran)}
                      margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color)" vertical={false} />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 11, fill: "var(--muted)" }}
                        tickFormatter={(val) => `Epoch ${val}`}
                      />
                      <YAxis
                        yAxisId="left"
                        domain={[60, 100]}
                        tick={{ fontSize: 11, fill: "var(--muted)" }}
                        width={40}
                        tickFormatter={(v) => `${v}%`}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        domain={[50, 80]}
                        tick={{ fontSize: 11, fill: "var(--muted)" }}
                        width={45}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--surface)",
                          borderColor: "var(--border)",
                          fontSize: 12,
                          borderRadius: 6,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
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
                        r={6}
                        fill="#10b981"
                        stroke="#ffffff"
                        label={{
                          value: "Best Checkpoint (Epoch 30)",
                          position: "top",
                          fill: "#10b981",
                          fontSize: 10,
                          fontWeight: 700,
                        }}
                      />
                    </LineChart>
                  ) : (
                    <LineChart
                      data={baselinePoints}
                      margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color)" vertical={false} />
                      <XAxis dataKey="episode" tick={{ fontSize: 11, fill: "var(--muted)" }} />
                      <YAxis
                        tick={{ fontSize: 11, fill: "var(--muted)" }}
                        width={45}
                        domain={[50, 80]}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--surface)",
                          borderColor: "var(--border)",
                          fontSize: 12,
                          borderRadius: 6,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                      <Line
                        type="monotone"
                        dataKey="total_distance_km"
                        name="Raw Distance (km)"
                        stroke="#cbd5e1"
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
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6, textAlign: "center" }}>
                {isProposed
                  ? "Điểm chấm xanh là Checkpoint tối ưu toàn cục tại Epoch 30 (Val SLA 98.55%, 2 đơn trễ) được lưu trong tệp best.pt."
                  : "Khoảng cách hội tụ dần về mức 57.3 km sau 2.000 episodes khi chính sách tối ưu hóa tour giao."}
              </div>
            </div>
          </div>

          {/* Hyperparameters & Training vs Inference Grids */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
              gap: 16,
              marginBottom: 16,
            }}
          >
            {/* Hyperparameters Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <h4
                style={{
                  margin: "0 0 12px 0",
                  fontSize: 13,
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: "var(--text)",
                }}
              >
                <Sliders size={16} color="#3b82f6" />
                Siêu tham số thực nghiệm (Hyperparameters)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                {Object.entries(data.hyperparameters).map(([key, val]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      padding: "4px 0",
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    <span style={{ color: "var(--muted)", textTransform: "capitalize" }}>
                      {key.replace(/_/g, " ")}:
                    </span>
                    <strong style={{ color: "var(--text)", maxWidth: "60%", textAlign: "right" }}>
                      {String(val)}
                    </strong>
                  </div>
                ))}
              </div>
            </div>

            {/* Training vs Inference Card */}
            <div
              style={{
                background: "var(--panel-bg)",
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--border)",
              }}
            >
              <h4
                style={{
                  margin: "0 0 12px 0",
                  fontSize: 13,
                  fontWeight: 700,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: "var(--text)",
                }}
              >
                <ShieldCheck size={16} color="#10b981" />
                Khác biệt giữa Huấn luyện và Suy luận (Inference)
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 12 }}>
                <div style={{ padding: "8px 12px", background: "var(--surface)", borderRadius: 6 }}>
                  <span style={{ color: "#3b82f6", fontWeight: 700, display: "block", marginBottom: 3 }}>
                    Chế độ Huấn luyện (Training):
                  </span>
                  <span style={{ color: "var(--text)" }}>
                    {data.training_vs_inference.training_mode}
                  </span>
                </div>
                <div style={{ padding: "8px 12px", background: "var(--surface)", borderRadius: 6 }}>
                  <span style={{ color: "#10b981", fontWeight: 700, display: "block", marginBottom: 3 }}>
                    Chế độ Suy luận (Inference):
                  </span>
                  <span style={{ color: "var(--text)" }}>
                    {data.training_vs_inference.inference_mode}
                  </span>
                </div>
                {data.training_vs_inference.action_space && (
                  <div style={{ padding: "8px 12px", background: "var(--surface)", borderRadius: 6 }}>
                    <span style={{ color: "#8b5cf6", fontWeight: 700, display: "block", marginBottom: 3 }}>
                      Không gian hành động:
                    </span>
                    <span style={{ color: "var(--text)" }}>
                      {data.training_vs_inference.action_space}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Scientific Interpretation Disclosure */}
          <div
            style={{
              padding: "14px 18px",
              borderRadius: 8,
              background: "rgba(59, 130, 246, 0.06)",
              border: "1px solid rgba(59, 130, 246, 0.2)",
              fontSize: 13,
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
                color: "#3b82f6",
              }}
            >
              <Sparkles size={16} />
              Minh bạch khoa học & Diễn giải độ ổn định:
            </div>
            <p style={{ margin: 0, color: "var(--text)" }}>
              {data.scientific_interpretation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
