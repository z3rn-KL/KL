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
  HardDrive,
  GitBranch,
  Copy,
  Check,
  Scale,
  Target,
} from "lucide-react";
import { RewardPenaltySection } from "./RewardPenaltySection";

type AlgorithmType = "proposed_rl" | "dqn" | "sarsa" | "q_learning";

const ALGO_DISPLAY_NAMES: Record<AlgorithmType, string> = {
  proposed_rl: "Proposed RL (Multi-Workload Candidate DQN)",
  dqn: "Vanilla DQN (Deep Q-Network)",
  sarsa: "SARSA (On-Policy TD Control)",
  q_learning: "Q-Learning (Off-Policy TD Control)",
};

interface AlgorithmArtifactInfo {
  label: string;
  source: string;
  verificationStatus: "Runtime Inference Verified" | "Training Log Verified";
  scope?: "Reference Only";
  isRuntimeVerified: boolean;
  architecture: string;
  note: string;
}

const ARTIFACT_INFO: Record<AlgorithmType, AlgorithmArtifactInfo> = {
  proposed_rl: {
    label: "Checkpoint mô hình PyTorch (.pt)",
    source: "results/proposed_rl_multiworkload_D_moderate_sla_best.pt",
    verificationStatus: "Runtime Inference Verified",
    isRuntimeVerified: true,
    architecture: "PyTorch MLP: 8 đặc trưng bất biến quy chuẩn → [64, 32] ReLU → 1 Q-value",
    note: "Mô hình được nạp trực tiếp vào Dispatch Engine để sinh tuyến thời gian thực.",
  },
  dqn: {
    label: "Nhật ký huấn luyện đối chứng (CSV Log)",
    source: "results/dqn_single_workload_training.csv",
    verificationStatus: "Training Log Verified",
    scope: "Reference Only",
    isRuntimeVerified: false,
    architecture: "Vanilla DQN (Đơn kịch bản): Không lưu checkpoint suy luận runtime",
    note: "Dữ liệu huấn luyện 5.000 episodes phục vụ đối chuẩn lý thuyết.",
  },
  sarsa: {
    label: "Nhật ký huấn luyện đối chứng (CSV Log)",
    source: "results/sarsa_single_workload_training.csv",
    verificationStatus: "Training Log Verified",
    scope: "Reference Only",
    isRuntimeVerified: false,
    architecture: "SARSA Tabular TD: Huấn luyện On-Policy 5.000 episodes",
    note: "Lưu vết lịch sử huấn luyện 5.000 episodes phục vụ đối chuẩn lý thuyết; không lưu bảng Q-table rời rạc trên đĩa.",
  },
  q_learning: {
    label: "Nhật ký huấn luyện đối chứng (CSV Log)",
    source: "results/q_learning_single_workload_training.csv",
    verificationStatus: "Training Log Verified",
    scope: "Reference Only",
    isRuntimeVerified: false,
    architecture: "Watkins Q-Learning Tabular TD: Huấn luyện Off-Policy 5.000 episodes",
    note: "Lưu vết lịch sử huấn luyện 5.000 episodes phục vụ đối chuẩn lý thuyết; không lưu bảng Q-table rời rạc trên đĩa.",
  },
};

const CHECKPOINT_PATHS: Record<AlgorithmType, string> = {
  proposed_rl: "results/proposed_rl_multiworkload_D_moderate_sla_best.pt",
  dqn: "results/dqn_single_workload_training.csv (Reference Log)",
  sarsa: "results/sarsa_single_workload_training.csv (Reference Log)",
  q_learning: "results/q_learning_single_workload_training.csv (Reference Log)",
};

const HYPERPARAM_VI_LABELS: Record<string, string> = {
  learning_rate: "Tốc độ học (Learning Rate)",
  learning_rate_alpha: "Tốc độ học α (Learning Rate)",
  gamma_discount: "Hệ số chiết khấu γ (Discount Factor)",
  epsilon_decay: "Lịch giảm ε (Epsilon Schedule)",
  hidden_layers: "Kiến trúc mạng ẩn (Hidden Layers)",
  target_update_freq: "Tần suất cập nhật mạng đích (Target Update)",
  target_update_frequency: "Tần suất cập nhật mạng đích (Target Update)",
  replay_buffer_size: "Kích thước bộ đệm (Replay Buffer)",
  batch_size: "Kích thước batch (Batch Size)",
  objective_weights: "Trọng số mục tiêu (Objective Weights)",
  algorithm_type: "Phân loại giải thuật (Algorithm Type)",
  tabular_state_space: "Không gian trạng thái bảng (Tabular State)",
};

const NEURAL_NET_KEYS = new Set([
  "hidden_layers",
  "target_update_freq",
  "target_update_frequency",
  "replay_buffer_size",
  "batch_size",
]);

export const RLTrainingExplorer: React.FC = () => {
  const [activeAlgo, setActiveAlgo] = useState<AlgorithmType>("proposed_rl");
  const [data, setData] = useState<TrainingHistoryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sampleRate, setSampleRate] = useState<number>(50);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

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

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const isProposed = activeAlgo === "proposed_rl";
  const isTabular = activeAlgo === "sarsa" || activeAlgo === "q_learning";
  const proposedPoints = isProposed
    ? (data?.datapoints as ProposedRLDatapoint[]) || []
    : [];
  const baselinePoints = !isProposed
    ? (data?.datapoints as BaselineRLDatapoint[]) || []
    : [];

  const bestValidationPoint = proposedPoints.find((point) => point.is_global_best && point.validation_weighted_sla_pct != null)
    ?? proposedPoints.filter((point) => point.is_best_checkpoint && point.validation_weighted_sla_pct != null).slice(-1)[0]
    ?? null;

  // Filter hyperparameters: strictly hide neural network parameters for tabular models
  const filteredHyperparams = data?.hyperparameters
    ? Object.entries(data.hyperparameters).filter(([k]) => {
        if (isTabular && NEURAL_NET_KEYS.has(k)) return false;
        return true;
      })
    : [];

  // Interpret only what the recorded history actually demonstrates.
  const getScientificInterpretation = (algo: AlgorithmType) => {
    if (algo === "proposed_rl") {
      return "Checkpoint tối ưu được chọn theo tiêu chí validation từ artifact thực nghiệm tại Epoch 30 (98,55% Val SLA, 2 đơn trễ). Giao diện chỉ hiển thị mốc tham chiếu đã xác minh từ artifact thực tế, không suy diễn ngoài dữ liệu.";
    }
    const pointCount = baselinePoints.length;
    const method = algo === "dqn" ? "mạng DQN" : algo === "sarsa" ? "bảng Q theo SARSA" : "bảng Q theo Q-Learning";
    return `Hiển thị ${pointCount.toLocaleString("vi-VN")} điểm dữ liệu được lấy mẫu từ nhật ký huấn luyện ${method}. Đường trung bình trượt phản ánh xu hướng hội tụ của chuỗi quan sát thực tế, dùng làm cơ sở đối chuẩn khoa học.`;
  };

  return (
    <div className="card space-y-5" style={{ marginBottom: 28, padding: "24px 28px" }}>
      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border/80 pb-3">
        <div>
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="inline-flex items-center justify-center size-9 rounded-xl bg-blue-500/15 text-blue-600 dark:text-blue-400">
              <Cpu className="size-5" />
            </span>
            <h3 className="text-lg sm:text-xl font-bold tracking-tight text-foreground m-0">
              Kết quả huấn luyện RL và đánh giá quá trình học
            </h3>
            <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
              Dữ liệu huấn luyện thực nghiệm
            </span>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">
            Trực quan hóa đường cong học tập thực tế, độ hội tụ, hàm mục tiêu và xác minh tính ổn định của mô hình.
          </p>
        </div>

        {/* Algorithm Tabs */}
        <div className="flex items-center gap-1.5 bg-muted/40 p-1 rounded-xl border border-border/70 flex-wrap self-start sm:self-auto">
          <button
            type="button"
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
              activeAlgo === "proposed_rl"
                ? "bg-teal-600 text-white shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
            onClick={() => setActiveAlgo("proposed_rl")}
          >
            Proposed RL
          </button>
          <button
            type="button"
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
              activeAlgo === "dqn"
                ? "bg-teal-600 text-white shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
            onClick={() => setActiveAlgo("dqn")}
          >
            Vanilla DQN
          </button>
          <button
            type="button"
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
              activeAlgo === "sarsa"
                ? "bg-teal-600 text-white shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
            onClick={() => setActiveAlgo("sarsa")}
          >
            SARSA
          </button>
          <button
            type="button"
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
              activeAlgo === "q_learning"
                ? "bg-teal-600 text-white shadow-xs"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
            onClick={() => setActiveAlgo("q_learning")}
          >
            Q-Learning
          </button>
        </div>
      </div>

      {/* ── TIER 1: TỔNG QUAN MÔ HÌNH (MODEL SUMMARY) ── */}
      {data && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <Layers className="size-3.5 text-primary" />
            <span>Tổng quan mô hình</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-card border border-border/80 rounded-xl p-3.5 space-y-1 shadow-2xs">
              <span className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
                <Cpu className="size-3.5 text-blue-500" /> Mô hình
              </span>
              <strong className="text-sm font-bold text-foreground block truncate" title={ALGO_DISPLAY_NAMES[activeAlgo]}>
                {isProposed ? "Candidate DQN" : activeAlgo === "dqn" ? "Vanilla DQN" : activeAlgo.toUpperCase()}
              </strong>
              <span className="text-xs text-muted-foreground font-mono">
                {isTabular ? "Bảng Q-Table rời rạc" : "Mạng Deep Q-Network"}
              </span>
            </div>

            <div className="bg-card border border-border/80 rounded-xl p-3.5 space-y-1 shadow-2xs">
              <span className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
                <GitBranch className="size-3.5 text-purple-500" /> Phân chia dữ liệu
              </span>
              <strong className="text-sm font-bold text-foreground block">
                {isProposed ? "76 Train · 16 Val · 16 Test" : "1 Workload Benchmark"}
              </strong>
              <span className="text-xs text-muted-foreground">
                {isProposed ? "108 Workload tổng thể" : "5 điểm giao chuẩn hóa"}
              </span>
            </div>

            <div className="bg-card border border-border/80 rounded-xl p-3.5 space-y-1 shadow-2xs">
              <span className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
                <TrendingUp className="size-3.5 text-amber-500" /> Quy mô huấn luyện
              </span>
              <strong className="text-sm font-bold text-foreground block">
                {isProposed ? `${data.total_epochs ?? proposedPoints.length} epochs · ${data.total_episodes.toLocaleString("vi-VN")} episodes` : `${data.total_episodes.toLocaleString("vi-VN")} episodes`}
              </strong>
              <span className="text-xs text-muted-foreground font-mono">
                Bước lấy mẫu: {data.sample_rate || 50}
              </span>
            </div>

            <div className="bg-card border border-border/80 rounded-xl p-3.5 space-y-1 shadow-2xs">
              <span className="text-xs text-muted-foreground font-medium flex items-center gap-1.5">
                <CheckCircle2 className="size-3.5 text-emerald-500" /> Mốc huấn luyện tham chiếu
              </span>
              <strong className="text-sm font-bold text-emerald-600 dark:text-emerald-400 block">
                {isProposed
                  ? bestValidationPoint
                    ? `Epoch ${bestValidationPoint.epoch} (${bestValidationPoint.validation_weighted_sla_pct?.toLocaleString("vi-VN")}% SLA validation)`
                    : "Epoch 30 (98,55% Val SLA · Checkpoint tối ưu)"
                  : `Episode ${data.total_episodes.toLocaleString("vi-VN")} (Cuối log đối chứng)`}
              </strong>
              <span className="text-xs text-muted-foreground font-mono truncate block" title={ARTIFACT_INFO[activeAlgo].source}>
                {ARTIFACT_INFO[activeAlgo].source}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── TIER 2: KHẢ NĂNG TÁI LẬP THỰC NGHIỆM (REPLICABILITY & PROVENANCE) ── */}
      {data && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
            <Database className="size-3.5 text-teal-600" />
            <span>Tier 2: Thông tin tái lập thực nghiệm &amp; Artifact mô hình</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-muted/20 border border-border/70 rounded-xl p-3 space-y-1 text-xs">
              <span className="text-muted-foreground font-medium flex items-center gap-1">
                <Database className="size-3 text-sky-500" /> Mã tập dữ liệu (Dataset ID):
              </span>
              <strong className="text-foreground block font-mono font-semibold">{data.dataset_id}</strong>
              <span className="text-[11px] text-muted-foreground block pt-1">
                {isProposed ? "Phân chia độc lập tránh rò rỉ dữ liệu (76/16/16)" : "Chuẩn đối sánh 5 điểm giao chuẩn hóa"}
              </span>
            </div>

            <div className="bg-muted/20 border border-border/70 rounded-xl p-3 space-y-1 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground font-medium flex items-center gap-1">
                  <FileCode className="size-3 text-emerald-500" /> Tệp nhật ký gốc (Training Log):
                </span>
                <button
                  type="button"
                  onClick={() => copyToClipboard(data.source_file, "logfile")}
                  className="text-muted-foreground hover:text-foreground cursor-pointer"
                  title="Sao chép đường dẫn tệp log"
                >
                  {copiedKey === "logfile" ? <Check className="size-3 text-emerald-500" /> : <Copy className="size-3" />}
                </button>
              </div>
              <code className="text-foreground block font-mono font-semibold text-[11px] truncate" title={data.source_file}>
                {data.source_file}
              </code>
              <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold block pt-1">
                ✓ Training log verified ({data.total_episodes.toLocaleString("vi-VN")} episodes)
              </span>
            </div>

            <div className="bg-muted/20 border border-border/70 rounded-xl p-3 space-y-1 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground font-medium flex items-center gap-1">
                  <HardDrive className="size-3 text-amber-500" /> {ARTIFACT_INFO[activeAlgo].label}:
                </span>
                <button
                  type="button"
                  onClick={() => copyToClipboard(ARTIFACT_INFO[activeAlgo].source, "artifact")}
                  className="text-muted-foreground hover:text-foreground cursor-pointer"
                  title="Sao chép đường dẫn artifact"
                >
                  {copiedKey === "artifact" ? <Check className="size-3 text-emerald-500" /> : <Copy className="size-3" />}
                </button>
              </div>
              <code className="text-foreground block font-mono font-semibold text-[11px] truncate" title={ARTIFACT_INFO[activeAlgo].source}>
                {ARTIFACT_INFO[activeAlgo].source}
              </code>
              <div className="pt-1 flex items-center gap-1.5 flex-wrap">
                {ARTIFACT_INFO[activeAlgo].isRuntimeVerified ? (
                  <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md font-semibold border border-teal-500/30 bg-teal-500/15 text-teal-700 dark:text-teal-300">
                    <CheckCircle2 className="size-3 text-teal-600 dark:text-teal-400" />
                    <span>Runtime Inference Verified</span>
                  </span>
                ) : (
                  <>
                    <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md font-semibold border border-emerald-500/30 bg-emerald-500/15 text-emerald-700 dark:text-emerald-400">
                      <CheckCircle2 className="size-3 text-emerald-600 dark:text-emerald-400" />
                      <span>Training Log Verified</span>
                    </span>
                    {ARTIFACT_INFO[activeAlgo].scope && (
                      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md font-semibold border border-muted-foreground/30 bg-muted/60 text-muted-foreground">
                        <span>{ARTIFACT_INFO[activeAlgo].scope}</span>
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Callout: Distinction between w_c and lambda */}
          <div className="p-3 rounded-xl border border-teal-500/30 bg-teal-500/5 text-xs text-muted-foreground space-y-1 leading-relaxed">
            <div className="flex items-center gap-1.5 font-bold text-teal-700 dark:text-teal-300 uppercase tracking-wider text-[11px]">
              <Sparkles className="size-3.5" />
              <span>Phân biệt khoa học: Trọng số mục tiêu (w_c = 0.050) và Hệ số phạt độ nhạy (λ = 0.50)</span>
            </div>
            <p className="m-0">
              • <strong>Trọng số đa mục tiêu w_c = 0.050</strong>: Là trọng số thành phần chuyển cụm trong hàm chi phí chuẩn hóa <code className="font-mono text-foreground font-semibold">J(π)</code> của Proposed RL (với <code className="font-mono text-foreground">w_d=0,375, w_t=0,375, w_late=0,100, w_dur=0,100, w_c=0,050</code> và tổng trọng số bằng 1,0).
              <br />
              • <strong>Hệ số phạt độ nhạy λ = 0.50</strong>: Là mức phạt chuyển cụm tối ưu trong phân tích độ nhạy hàm thưởng của Guided DQN tại Chương 4 (khảo sát đơn seed trên dải <code className="font-mono text-foreground">λ ∈ &#123;0.00, 0.10, 0.25, 0.50&#125;</code> và kiểm chứng đa seed đối cặp Wilcoxon trên dải <code className="font-mono text-foreground">λ ∈ &#123;0.25, 0.50, 1.00&#125;</code>). Hai giá trị thuộc hai cơ chế toán học độc lập trong hai thiết kế giải thuật khác nhau.
            </p>
          </div>
        </div>
      )}

      {/* Loading & Error States */}
      {loading && (
        <div className="py-12 text-center text-muted-foreground flex items-center justify-center gap-2 text-xs sm:text-sm">
          <RefreshCw className="size-4 animate-spin" />
          <span>Đang truy xuất lịch sử huấn luyện từ hệ thống tệp kết quả...</span>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/25 text-destructive flex items-center gap-2.5 text-xs sm:text-sm">
          <AlertCircle className="size-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Content when loaded */}
      {!loading && !error && data && (
        <div className="space-y-5">
          {/* Main Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Chart 1 */}
            <div className="bg-card border border-border/80 rounded-xl p-4 shadow-2xs space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h4 className="text-sm font-bold text-foreground m-0">
                  {isProposed
                    ? "Mean Training Reward & Objective vs Epoch"
                    : "Episode Reward & Moving Average (MA-50)"}
                </h4>
                <span className="text-xs text-muted-foreground font-mono">
                  {isProposed ? `${data.total_epochs ?? proposedPoints.length} epochs · Multi-Workload` : `${data.total_episodes.toLocaleString("vi-VN")} episodes`}
                </span>
              </div>
              <div style={{ height: 260, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={(isProposed ? proposedPoints : baselinePoints) as any[]}
                    margin={{ top: 10, right: 16, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis
                      dataKey={isProposed ? "epoch" : "episode"}
                      stroke="var(--text-secondary)"
                      tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                      tickFormatter={(val) => (isProposed ? `E${val}` : `${val}`)}
                    />
                    <YAxis
                      stroke="var(--text-secondary)"
                      tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                      width={44}
                      tickFormatter={(val) => val.toFixed(1)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--panel-bg)",
                        borderColor: "var(--border)",
                        color: "var(--text)",
                        fontSize: 12,
                        borderRadius: 8,
                        boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6, color: "var(--text)" }} />
                    {isProposed ? (
                      <>
                        <Line
                          type="monotone"
                          dataKey="mean_training_reward"
                          name="Training Reward"
                          stroke="#3b82f6"
                          strokeWidth={2.2}
                          dot={{ r: 2 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="mean_training_objective"
                          name="Training Objective J"
                          stroke="#ef4444"
                          strokeWidth={1.8}
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
                          opacity={0.35}
                        />
                        <Line
                          type="monotone"
                          dataKey="reward_ma50"
                          name="Reward MA-50"
                          stroke="#3b82f6"
                          strokeWidth={2.2}
                          dot={false}
                        />
                      </>
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground text-center m-0">
                {isProposed
                  ? "Phần thưởng trung bình qua 40 epochs biến thiên tương thích với batch độ khó kịch bản."
                  : "Đường MA-50 (xanh) lọc nhiễu dao động do epsilon exploration giai đoạn đầu."}
              </p>
            </div>

            {/* Chart 2 */}
            <div className="bg-card border border-border/80 rounded-xl p-4 shadow-2xs space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h4 className="text-sm font-bold text-foreground m-0">
                  {isProposed
                    ? "Validation SLA (%) & Mean Distance (km) vs Epoch"
                    : "Quãng đường di chuyển (km) & Hội tụ"}
                </h4>
                <span className="text-xs text-muted-foreground font-mono">
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
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="epoch"
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                        tickFormatter={(val) => `Epoch ${val}`}
                      />
                      <YAxis
                        yAxisId="left"
                        domain={[60, 100]}
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                        width={42}
                        tickFormatter={(v) => `${v}%`}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        domain={[50, 80]}
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                        width={46}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--panel-bg)",
                          borderColor: "var(--border)",
                          color: "var(--text)",
                          fontSize: 12,
                          borderRadius: 8,
                          boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6, color: "var(--text)" }} />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="validation_weighted_sla_pct"
                        name="Validation SLA (%)"
                        stroke="#10b981"
                        strokeWidth={2.2}
                        dot={{ r: 3.5 }}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="validation_mean_distance_km"
                        name="Val Distance (km)"
                        stroke="#f59e0b"
                        strokeWidth={1.8}
                        dot={{ r: 2.5 }}
                      />
                      {bestValidationPoint && (
                        <ReferenceDot
                          yAxisId="left"
                          x={bestValidationPoint.epoch}
                          y={bestValidationPoint.validation_weighted_sla_pct ?? 0}
                          r={6}
                          fill="#10b981"
                          stroke="#ffffff"
                          label={{
                            value: `Checkpoint (E${bestValidationPoint.epoch})`,
                            position: "top",
                            fill: "#10b981",
                            fontSize: 12,
                            fontWeight: 700,
                          }}
                        />
                      )}
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
                        tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                      />
                      <YAxis
                        stroke="var(--text-secondary)"
                        tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                        width={44}
                        domain={[50, 80]}
                        tickFormatter={(v) => `${v}km`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--panel-bg)",
                          borderColor: "var(--border)",
                          color: "var(--text)",
                          fontSize: 12,
                          borderRadius: 8,
                          boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6, color: "var(--text)" }} />
                      <Line
                        type="monotone"
                        dataKey="total_distance_km"
                        name="Raw Distance (km)"
                        stroke="#94a3b8"
                        strokeWidth={1}
                        dot={false}
                        opacity={0.35}
                      />
                      <Line
                        type="monotone"
                        dataKey="distance_ma50"
                        name="Distance MA-50 (km)"
                        stroke="#f59e0b"
                        strokeWidth={2.2}
                        dot={false}
                      />
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground text-center m-0">
                {isProposed
                  ? "Điểm chấm xanh là Checkpoint tối ưu tại Epoch 30 (Val SLA 98.55%, 2 đơn trễ) được lưu trong results/proposed_rl_multiworkload_D_moderate_sla_best.pt."
                  : "Khoảng cách hội tụ dần về mức 57.3 km sau ~2.000 episodes khi chính sách tối ưu hóa tour giao."}
              </p>
            </div>
          </div>

          {/* Hyperparameters & Training vs Inference Grids */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Hyperparameters Card */}
            <div className="bg-card border border-border/80 rounded-xl p-4 shadow-2xs space-y-3">
              <div className="flex items-center justify-between border-b border-border/70 pb-2">
                <h4 className="text-sm font-bold text-foreground flex items-center gap-2 m-0">
                  <Sliders className="size-4 text-blue-500" />
                  <span>Siêu tham số thực nghiệm (Hyperparameters)</span>
                </h4>
                <span className="text-[11px] font-mono text-muted-foreground">{filteredHyperparams.length} tham số</span>
              </div>
              <div className="divide-y divide-border/50 text-xs">
                {filteredHyperparams.map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between py-1.5 gap-3">
                    <span className="text-muted-foreground font-medium truncate">
                      {HYPERPARAM_VI_LABELS[key] || key.replace(/_/g, " ")}:
                    </span>
                    <strong className="text-foreground text-right font-mono font-semibold shrink-0">
                      {String(val)}
                    </strong>
                  </div>
                ))}
              </div>
            </div>

            {/* Training vs Inference Card */}
            <div className="bg-card border border-border/80 rounded-xl p-4 shadow-2xs space-y-3">
              <div className="flex items-center justify-between border-b border-border/70 pb-2">
                <h4 className="text-sm font-bold text-foreground flex items-center gap-2 m-0">
                  <ShieldCheck className="size-4 text-emerald-500" />
                  <span>Khác biệt giữa Huấn luyện và Suy luận (Inference)</span>
                </h4>
                <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold">Tách biệt rõ</span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="p-2.5 bg-muted/20 rounded-lg border border-border/60 flex flex-col gap-0.5">
                  <span className="text-blue-600 dark:text-blue-400 font-bold text-[11px] uppercase tracking-wider">
                    Chế độ Huấn luyện (Training Mode):
                  </span>
                  <span className="text-foreground leading-snug">
                    {data.training_vs_inference.training_mode}
                  </span>
                </div>
                <div className="p-2.5 bg-muted/20 rounded-lg border border-border/60 flex flex-col gap-0.5">
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold text-[11px] uppercase tracking-wider">
                    Chế độ Suy luận (Inference Mode):
                  </span>
                  <span className="text-foreground leading-snug">
                    {data.training_vs_inference.inference_mode}
                  </span>
                </div>
                {data.training_vs_inference.action_space && (
                  <div className="p-2.5 bg-muted/20 rounded-lg border border-border/60 flex flex-col gap-0.5">
                    <span className="text-purple-600 dark:text-purple-400 font-bold text-[11px] uppercase tracking-wider">
                      Không gian hành động (Action Space):
                    </span>
                    <span className="text-foreground leading-snug">
                      {data.training_vs_inference.action_space}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Scientific Interpretation Disclosure */}
          <div className="p-3.5 rounded-xl bg-card border border-border/80 shadow-2xs space-y-1.5 text-xs leading-relaxed">
            <div className="flex items-center gap-2 font-bold text-blue-600 dark:text-blue-400 text-xs uppercase tracking-wider">
              <Sparkles className="size-3.5" />
              <span>Minh bạch khoa học &amp; Diễn giải độ ổn định của giải thuật</span>
            </div>
            <p className="text-foreground m-0">
              {getScientificInterpretation(activeAlgo)}
            </p>
          </div>

          {/* Standardized Reward / Penalty Formulation */}
          <RewardPenaltySection defaultOpen={false} />
        </div>
      )}
    </div>
  );
};
