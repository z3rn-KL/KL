import React, { useState, useEffect, useRef, useMemo } from "react";
import { createPortal } from "react-dom";
import {
  BrainCircuit,
  Play,
  Pause,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  X,
  Target,
  Clock,
  Navigation,
  ShieldAlert,
  Layers,
  Sparkles,
  Info,
  CheckCircle2,
  TrendingUp,
  User,
  Copy,
  Check,
} from "lucide-react";
import {
  api,
  type RlDecisionTraceResponse,
  type DecisionTraceStep,
  type CandidateEvaluation,
} from "@/lib/api";
import { formattedRlFeatures } from "@/lib/rlFeatureLabels";

interface RlDecisionExplorerModalProps {
  isOpen?: boolean;
  isInline?: boolean;
  onClose?: () => void;
  sessionId: string;
  selectedShipperId?: string;
  onSelectOrder?: (orderId: string) => void;
}

export function RlDecisionExplorerModal({
  isOpen = true,
  isInline = false,
  onClose,
  sessionId,
  selectedShipperId,
  onSelectOrder,
}: RlDecisionExplorerModalProps) {
  const [targetShipperId, setTargetShipperId] = useState<string>(selectedShipperId || "shipper_01");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [traceData, setTraceData] = useState<RlDecisionTraceResponse | null>(null);
  const [activeStepIdx, setActiveStepIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const playTimerRef = useRef<any>(null);

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  // Sync targetShipperId when selectedShipperId prop changes
  useEffect(() => {
    if (selectedShipperId) {
      setTargetShipperId(selectedShipperId);
    }
  }, [selectedShipperId]);

  // Load decision trace when modal opens or shipper changes
  useEffect(() => {
    if (!isOpen || !sessionId) return;
    let isCancelled = false;

    const fetchTrace = async () => {
      try {
        setLoading(true);
        setError(null);
        setIsPlaying(false);
        setActiveStepIdx(0);

        const data = await api.getSessionRlDecisionTrace(sessionId, {
          shipper_id: targetShipperId,
        });

        if (!isCancelled) {
          if (data && data.success && data.decision_trace && data.decision_trace.length > 0) {
            setTraceData(data);
            if (data.shipper_id && data.shipper_id !== targetShipperId) {
              setTargetShipperId(data.shipper_id);
            }
          } else {
            setTraceData(data || null);
            setError(data?.message || "Không có dữ liệu vết quyết định RL cho shipper này.");
          }
        }
      } catch (err: any) {
        if (!isCancelled) {
          setError(err.message || "Lỗi khi truy vấn vết quyết định RL từ backend.");
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    fetchTrace();

    return () => {
      isCancelled = true;
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isOpen, sessionId, targetShipperId]);

  // Handle Play/Pause Auto-Scrubber
  useEffect(() => {
    if (isPlaying && traceData && traceData.decision_trace.length > 0) {
      playTimerRef.current = setInterval(() => {
        setActiveStepIdx((prev) => {
          if (prev >= traceData.decision_trace.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1600);
    } else {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, traceData]);

  // Handle Keyboard (Esc to close, Left/Right for steps, Space for play/pause)
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose?.();
      } else if (e.key === "ArrowLeft") {
        setActiveStepIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === "ArrowRight") {
        if (traceData) {
          setActiveStepIdx((prev) => Math.min(traceData.decision_trace.length - 1, prev + 1));
        }
      } else if (e.key === " " && (e.target as HTMLElement).tagName !== "BUTTON" && (e.target as HTMLElement).tagName !== "SELECT") {
        e.preventDefault();
        setIsPlaying((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, traceData, onClose]);

  // Hook must be called unconditionally before any early return (Rules of Hooks)
  const cumulativeReward = useMemo(() => {
    if (!traceData?.decision_trace) return 0;
    return traceData.decision_trace
      .slice(0, activeStepIdx + 1)
      .reduce((sum, st) => sum + (st.step_reward || 0), 0);
  }, [traceData, activeStepIdx]);

  if (!isOpen && !isInline) return null;

  const currentStep: DecisionTraceStep | undefined =
    traceData?.decision_trace?.[activeStepIdx];
  const totalSteps = traceData?.decision_trace?.length || 0;
  // For a one-stop route the backend bypasses neural scoring and emits a
  // placeholder Q=1, zero feature vector and empty feature_dict. Do not
  // describe this special case as model inference.
  const isSingleStopShortcut = Boolean(currentStep &&
    currentStep.candidates_evaluated.length === 1 &&
    currentStep.candidates_evaluated[0].q_score === 1 &&
    Object.keys(currentStep.candidates_evaluated[0].feature_dict ?? {}).length === 0);

  // Find first active shipper with orders if available
  const firstActiveShipper = traceData?.active_shippers?.find((s) => s.order_count > 0);

  const content = (
    <div
      className={
        isInline
          ? "relative flex flex-col w-full bg-card border border-border/80 rounded-2xl shadow-xs overflow-hidden"
          : "relative flex flex-col w-full max-w-5xl max-h-[92vh] bg-card border border-border/80 rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200"
      }
      role={isInline ? "region" : "dialog"}
      aria-modal={isInline ? undefined : "true"}
      aria-label="RL Decision Explorer"
    >
      {/* 1. Header with Scientific Provenance & Shipper Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-border/80 bg-muted/30">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-teal-500/15 text-teal-600 dark:text-teal-400 border border-teal-500/20 shadow-2xs shrink-0">
            <BrainCircuit className="size-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-base font-bold text-foreground">
                Vết Quyết Định Mạng Nơ-ron (RL Decision Trace)
              </h2>
              <span className="rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                Candidate-Scoring DQN
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Giải thích từng bước lựa chọn điểm đến dựa trên 8 đặc trưng bất biến và hàm giá trị Q(s, a).
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Truncated Session ID */}
          <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-lg bg-background border border-border/80 text-xs font-mono">
            <span className="text-muted-foreground">Session:</span>
            <span className="font-semibold text-foreground">{sessionId ? `${sessionId.slice(0, 8)}...` : "–"}</span>
            <button
              type="button"
              onClick={() => copyToClipboard(sessionId, "sid")}
              className="text-muted-foreground hover:text-foreground cursor-pointer"
              title="Sao chép Session ID"
            >
              {copiedKey === "sid" ? <Check className="size-3 text-emerald-500" /> : <Copy className="size-3" />}
            </button>
          </div>

          {/* Shipper Selector Dropdown */}
          <div className="flex items-center gap-1.5 bg-background border border-border/80 rounded-lg px-2 py-1 text-xs shadow-2xs">
            <User className="size-3.5 text-muted-foreground" />
            <select
              value={targetShipperId}
              onChange={(e) => setTargetShipperId(e.target.value)}
              className="bg-transparent text-xs font-semibold text-foreground focus:outline-none cursor-pointer"
              aria-label="Chọn shipper phân tích RL"
            >
              {(traceData?.active_shippers || Array.from({ length: 16 }, (_, i) => ({
                shipper_id: `shipper_${String(i + 1).padStart(2, "0")}`,
                name: `Shipper #${String(i + 1).padStart(2, "0")}`,
                order_count: 0,
              }))).map((s) => (
                <option key={s.shipper_id} value={s.shipper_id}>
                  {s.name} {s.order_count > 0 ? `(${s.order_count} đơn)` : "(0 đơn)"}
                </option>
              ))}
            </select>
          </div>

          {onClose && (
            <button
              type="button"
              id="btnCloseRlModal"
              onClick={onClose}
              className="rounded-lg p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
              aria-label="Đóng cửa sổ"
            >
              <X className="size-4" />
            </button>
          )}
        </div>
      </div>

      {/* Mô tả hàm mục tiêu từ chính trọng số backend, không cố định trên UI. */}
      <div className="px-5 py-3 bg-muted/15 border-b border-border/60 space-y-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <Target className="size-4 text-primary" /> Hàm mục tiêu điều phối
        </div>
        <div className="rounded-lg border border-border/70 bg-background px-3 py-2 font-mono text-sm leading-7 overflow-x-auto" role="math" aria-label="Hàm mục tiêu điều phối gồm năm thành phần">
          <div><span className="font-semibold">J(π)</span> = w<sub>d</sub> · d + w<sub>t</sub> · t</div>
          <div className="pl-9">+ w<sub>late</sub> · I<sub>late</sub> + w<sub>dur</sub> · Δt<sub>late</sub></div>
          <div className="pl-9">+ w<sub>c</sub> · c<sub>switch</sub></div>
        </div>
        {traceData?.objective_weights ? (
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {([
              ["Cự ly", traceData.objective_weights.distance],
              ["Thời gian", traceData.objective_weights.travel_time],
              ["Phạt giao trễ", traceData.objective_weights.late_delivery],
              ["Thời lượng trễ", traceData.objective_weights.lateness],
              ["Đổi cụm", traceData.objective_weights.cluster_switch],
            ] as const).map(([label, weight]) => (
              <span key={label} className="rounded-md border border-border bg-card px-2 py-1">
                {label}: <strong className="font-mono text-foreground">{weight.toFixed(3)}</strong>
              </span>
            ))}
          </div>
        ) : <p className="text-xs text-muted-foreground">Trọng số được tải từ vết quyết định của Backend.</p>}
        <p className="text-xs text-muted-foreground">Các giá trị khoảng cách, thời gian và trễ hạn bên dưới là số liệu thô. Không cộng trực tiếp chúng thành J(π) nếu chưa áp dụng cùng phép chuẩn hóa của mô hình.</p>
      </div>

      {/* 3. Main Body */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-16 space-y-3">
            <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            <p className="text-xs text-muted-foreground font-medium">
              Đang nạp mô hình PyTorch và giải mã vết quyết định RL từ checkpoint...
            </p>
          </div>
        ) : error || !currentStep ? (
          /* Neutral / Informational Empty State (Issue 20 requirement) */
          <div className="rounded-2xl border border-border/70 bg-card/60 p-8 text-center space-y-3">
            <div className="flex size-12 items-center justify-center rounded-2xl bg-muted/60 text-muted-foreground mx-auto">
              <BrainCircuit className="size-6 text-primary" />
            </div>
            <h3 className="text-base font-bold text-foreground">Chưa có dữ liệu quyết định RL</h3>
            <p className="text-xs text-muted-foreground max-w-md mx-auto leading-relaxed">
              {error ||
                "Shipper hiện chưa được phân công đơn hoặc phiên hiện tại chưa chạy tính toán kế hoạch RL. Vui lòng bấm tạo kế hoạch tại Bàn lập tuyến hoặc chọn shipper khác."}
            </p>
            <div className="flex items-center justify-center gap-2 pt-2">
              {firstActiveShipper && firstActiveShipper.shipper_id !== targetShipperId && (
                <button
                  type="button"
                  className="px-3.5 py-1.5 rounded-lg bg-teal-600 text-white font-semibold text-xs hover:bg-teal-700 transition-colors cursor-pointer shadow-xs"
                  onClick={() => setTargetShipperId(firstActiveShipper.shipper_id)}
                >
                  Chuyển sang {firstActiveShipper.name} ({firstActiveShipper.order_count} đơn)
                </button>
              )}
              {onClose && (
                <button
                  type="button"
                  className="px-3.5 py-1.5 rounded-lg border border-border bg-background text-foreground font-semibold text-xs hover:bg-muted transition-colors cursor-pointer shadow-2xs"
                  onClick={onClose}
                >
                  Quay lại bàn lập tuyến
                </button>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Step Navigation Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-muted/30 border border-border/80">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-foreground">
                  Bước {activeStepIdx + 1} / {totalSteps}
                </span>
                <span className="text-xs text-muted-foreground font-mono">
                  ({currentStep.current_node} → {currentStep.chosen_delivery_id})
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setActiveStepIdx(0)}
                  disabled={activeStepIdx === 0}
                  className="p-1.5 rounded-lg border border-border/60 bg-background text-muted-foreground hover:text-foreground disabled:opacity-40 cursor-pointer"
                  title="Quay lại điểm xuất phát (Depot)"
                >
                  <RotateCcw className="size-3.5" />
                </button>

                <button
                  type="button"
                  onClick={() => setActiveStepIdx((prev) => Math.max(0, prev - 1))}
                  disabled={activeStepIdx === 0}
                  className="p-1.5 rounded-lg border border-border/60 bg-background text-muted-foreground hover:text-foreground disabled:opacity-40 cursor-pointer"
                  title="Bước trước (Phím ←)"
                >
                  <ChevronLeft className="size-3.5" />
                </button>

                <button
                  type="button"
                  onClick={() => setIsPlaying((prev) => !prev)}
                  className="px-3 py-1.5 rounded-lg bg-teal-600 text-white font-medium text-xs flex items-center gap-1.5 shadow-2xs hover:bg-teal-700 transition-colors cursor-pointer"
                  title="Phát / Dừng tự động bước (Phím Space)"
                >
                  {isPlaying ? <Pause className="size-3" /> : <Play className="size-3" />}
                  <span>{isPlaying ? "Tạm dừng" : "Tự động phát"}</span>
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setActiveStepIdx((prev) =>
                      Math.min((traceData?.decision_trace?.length || 1) - 1, prev + 1)
                    )
                  }
                  disabled={activeStepIdx >= totalSteps - 1}
                  className="p-1.5 rounded-lg border border-border/60 bg-background text-muted-foreground hover:text-foreground disabled:opacity-40 cursor-pointer"
                  title="Bước tiếp theo (Phím →)"
                >
                  <ChevronRight className="size-3.5" />
                </button>
              </div>
            </div>

            {/* Grid: Left Step Summary & Right Candidate Scoring Table */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Left Column: Metrics B, C, D Distinct Categorization */}
              <div className="space-y-3">
                {/* Category B: Environment Reward / Penalty */}
                <div className="p-3.5 rounded-xl border border-border/80 bg-card space-y-2.5 shadow-2xs">
                  <div className="flex items-center justify-between border-b border-border/60 pb-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                      <TrendingUp className="size-3.5" />
                      Phần thưởng bước (r_t)
                    </span>
                    <span className="text-xs text-muted-foreground font-mono">r_t = -ΔJ</span>
                  </div>

                  <div className="space-y-1.5 text-xs font-mono">
                    <div className="flex items-center justify-between p-1.5 rounded bg-emerald-500/10 border border-emerald-500/20 font-bold text-emerald-600 dark:text-emerald-400">
                      <span>Thưởng bước (r_t):</span>
                      <span>{currentStep.step_reward.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center justify-between p-1.5 rounded bg-muted/40 font-bold text-foreground">
                      <span>Thưởng tích lũy (Σ r_k):</span>
                      <span>{cumulativeReward.toFixed(4)}</span>
                    </div>

                    {currentStep.reward_breakdown ? (
                      <div className="p-2 rounded-lg bg-muted/30 border border-border/60 space-y-1 text-[11px] text-muted-foreground mt-1.5 font-mono">
                        <div className="font-semibold text-foreground text-[10px] uppercase tracking-wider mb-1">
                          Phân rã thành phần phạt (Step Cost Breakdown):
                        </div>
                        <div className="flex justify-between">
                          <span>• Cự ly (w_d·D̄):</span>
                          <span className="text-foreground">-{currentStep.reward_breakdown.distance_cost.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>• Thời gian (w_t·T̄):</span>
                          <span className="text-foreground">-{currentStep.reward_breakdown.travel_time_cost.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>• Vi phạm hạn (w_m·M̄):</span>
                          <span className={currentStep.reward_breakdown.late_delivery_cost > 0 ? "text-rose-500 font-bold" : "text-foreground"}>
                            -{currentStep.reward_breakdown.late_delivery_cost.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>• Số phút trễ (w_l·L̄):</span>
                          <span className={currentStep.reward_breakdown.lateness_cost > 0 ? "text-rose-500 font-bold" : "text-foreground"}>
                            -{currentStep.reward_breakdown.lateness_cost.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>• Đổi cụm (w_c·S̄):</span>
                          <span className={currentStep.reward_breakdown.cluster_switch_cost > 0 ? "text-amber-500 font-bold" : "text-foreground"}>
                            -{currentStep.reward_breakdown.cluster_switch_cost.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between border-t border-border/60 pt-1 font-bold text-foreground">
                          <span>Tổng số gia ΔJ_t:</span>
                          <span>+{currentStep.reward_breakdown.total_step_cost.toFixed(4)}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="p-1.5 rounded bg-muted/20 border border-border/50 text-[11px] text-muted-foreground font-sans">
                        Công thức hàm thưởng: <code className="font-mono text-foreground">r_t = -[w_d·D̄ + w_t·T̄ + w_m·M̄ + w_l·L̄ + w_c·S̄]</code>.
                      </div>
                    )}

                    <div className="text-xs text-muted-foreground font-sans leading-normal pt-1">
                      Mô hình ước lượng phần thưởng kỳ vọng <code className="font-mono text-foreground font-semibold">Q(s, a) = E[r_t + γ·max Q(s', a')]</code>.
                    </div>
                  </div>
                </div>

                {/* Category C: Objective Function Contributions */}
                <div className="p-3.5 rounded-xl border border-border/80 bg-card space-y-2 shadow-2xs">
                  <div className="flex items-center justify-between border-b border-border/60 pb-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-primary flex items-center gap-1.5">
                      <Target className="size-3.5" />
                      Thành phần hàm mục tiêu
                    </span>
                    <span className="text-xs text-muted-foreground font-mono">Đại lượng thô · chưa nhân trọng số</span>
                  </div>

                  <div className="space-y-1.5 text-xs font-mono">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Cự ly tích lũy:</span>
                      <strong>{currentStep.cumulative_distance_km.toFixed(2)} km</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Thời gian tích lũy:</span>
                      <strong>{currentStep.cumulative_time_minutes.toFixed(1)} phút</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Chỉ dấu giao trễ:</span>
                      <strong className={currentStep.lateness_minutes > 0 ? "text-rose-500" : "text-emerald-500"}>
                        {currentStep.lateness_minutes > 0 ? "1 (Vi phạm)" : "0 (Đúng hạn)"}
                      </strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Thời lượng trễ:</span>
                      <strong>{currentStep.lateness_minutes.toFixed(1)} phút</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Số lần chuyển cụm:</span>
                      <strong>{currentStep.cluster_switches} lần</strong>
                    </div>
                  </div>
                </div>

                {/* Category D: Operational Costs & Real SLA */}
                <div className="p-3.5 rounded-xl border border-border/80 bg-card space-y-2 shadow-2xs">
                  <div className="flex items-center justify-between border-b border-border/60 pb-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                      <Clock className="size-3.5" />
                      Trạng thái hạn giao
                    </span>
                    <span className="text-xs text-muted-foreground">Theo vết lập tuyến</span>
                  </div>

                  <div className="space-y-1.5 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Điểm dừng chọn:</span>
                      <strong className="text-primary font-mono">{currentStep.chosen_delivery_id}</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Cam kết SLA:</span>
                      {currentStep.lateness_minutes > 0 ? (
                        <span className="text-rose-600 dark:text-rose-400 font-bold flex items-center gap-1">
                          <ShieldAlert className="size-3" /> Trễ {currentStep.lateness_minutes.toFixed(1)}p
                        </span>
                      ) : (
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1">
                          <CheckCircle2 className="size-3" /> Đạt đúng hạn
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      Chi phí vận hành đầy đủ được đối soát tại Mô phỏng và Báo cáo ca; không suy ra tổng chi phí chỉ từ quãng đường.
                    </p>
                  </div>
                </div>
              </div>

              {/* Right Card: Evaluated Candidates Table */}
              <div className="md:col-span-2 rounded-xl border border-border/80 bg-card overflow-hidden flex flex-col">
                <div className="p-3 border-b border-border/80 bg-muted/20 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="size-4 text-teal-600" />
                    <h4 className="text-xs font-bold text-foreground m-0">
                      Đánh Giá Điểm Q(s, a) Các Ứng Viên (Candidate Ranking)
                    </h4>
                  </div>
                  <span className="text-xs text-muted-foreground font-medium">
                    {isSingleStopShortcut ? "Chỉ có một điểm dừng, không cần xếp hạng Q" : "Ứng viên có Q lớn nhất được ưu tiên"}
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto max-h-[380px] divide-y divide-border/60">
                  {currentStep.candidates_evaluated.map((cand, idx) => {
                    const isChosen = cand.delivery_id === currentStep.chosen_delivery_id;
                    const maxQ = Math.max(...currentStep.candidates_evaluated.map((c) => c.q_score));
                    const minQ = Math.min(...currentStep.candidates_evaluated.map((c) => c.q_score));
                    const qRange = Math.max(0.0001, maxQ - minQ);
                    const normBar = Math.max(10, Math.min(100, ((cand.q_score - minQ) / qRange) * 100));

                    return (
                      <div
                        key={cand.delivery_id}
                        className={`p-2.5 transition-colors text-xs ${
                          isChosen
                            ? "bg-teal-500/10 border-l-4 border-teal-600 dark:border-teal-400"
                            : "hover:bg-muted/40"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <div className="flex items-center gap-2">
                            <span
                              className={`size-5 rounded-full flex items-center justify-center font-mono text-xs font-bold ${
                                isChosen
                                  ? "bg-teal-600 text-white"
                                  : "bg-muted text-muted-foreground"
                              }`}
                            >
                              {idx + 1}
                            </span>
                            <div>
                              <strong className="text-foreground block text-xs">
                                {cand.delivery_id}
                              </strong>
                              <span className="text-xs text-muted-foreground">
                                {cand.package_name || "Kiện hàng"} · Cụm #{cand.cluster_id ?? 0}
                              </span>
                            </div>
                          </div>

                          <div className="text-right">
                            <span className="font-mono font-bold text-xs text-foreground block">
                              {isSingleStopShortcut ? "Một ứng viên duy nhất" : `Q = ${cand.q_score.toFixed(4)}`}
                            </span>
                            {isChosen && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded bg-teal-600 text-white text-[10px] font-bold uppercase tracking-wider whitespace-nowrap shrink-0">
                                Đã chọn bởi Agent
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Relative Q Bar: meaningful only for actual model rankings. */}
                        {!isSingleStopShortcut && (
                          <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className={`h-full transition-all ${isChosen ? "bg-teal-600" : "bg-muted-foreground/30"}`}
                              style={{ width: `${normBar}%` }}
                            />
                          </div>
                        )}

                        {/* Original 8-dimensional encoder vector: names must match
                            CandidateEncoder.FEATURE_NAMES, not the mismatched
                            feature_dict labels emitted by the API's trace solver. */}
                        {!isSingleStopShortcut && cand.features?.length === 8 ? (
                          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-xs text-muted-foreground bg-muted/30 p-2 rounded-lg" aria-label="Tám đặc trưng dùng để tính Q-value">
                            {formattedRlFeatures(cand.features).map((feature) => (
                              <div key={feature.key} className="min-w-0" title={feature.key}>
                                <span className="block leading-snug">{feature.label}</span>
                                <strong className="text-foreground font-mono">{feature.value}</strong>
                              </div>
                            ))}
                          </div>
                        ) : isSingleStopShortcut ? (
                          <p className="mt-2 text-xs text-muted-foreground">Tuyến chỉ có một điểm giao: bộ giải trả về phương án trực tiếp, không tạo vector đặc trưng hay Q-value dự đoán.</p>
                        ) : (
                          <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">Không có vector tám đặc trưng hợp lệ từ vết quyết định.</p>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Scientific Decision Rationale Box */}
                <div className="p-3 border-t border-border/80 bg-muted/10">
                  <div className="p-3 rounded-xl border border-teal-500/30 bg-teal-500/5 text-xs space-y-1">
                    <div className="flex items-center gap-1.5 font-bold text-teal-700 dark:text-teal-300 uppercase tracking-wider text-[11px]">
                      <Sparkles className="size-3.5" />
                      <span>Lý giải khoa học quyết định RL (Argmax Q-Score)</span>
                    </div>
                    <p className="text-muted-foreground leading-relaxed m-0">
                      {isSingleStopShortcut ? (
                        <>Chỉ có một điểm giao <strong className="text-foreground font-mono">{currentStep.chosen_delivery_id}</strong> nên bộ giải đưa ra tuyến trực tiếp, không có bước so sánh Q-value của mạng nơ-ron.</>
                      ) : (
                        <>Tại bước này, mô hình ưu tiên <strong className="text-foreground font-mono">{currentStep.chosen_delivery_id}</strong> do có giá trị <strong className="text-teal-700 dark:text-teal-300 font-mono">Q(s, a) lớn nhất</strong> trong tập ứng viên khả dụng. Đây là đánh giá của chính sách hiện tại, không phải bằng chứng tối ưu toàn cục. Thời lượng trễ ghi nhận: {currentStep.lateness_minutes.toFixed(1)} phút; số lần chuyển cụm lũy kế: {currentStep.cluster_switches}.</>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* 4. Footer */}
      <div className="px-5 py-3 border-t border-border/80 bg-muted/20 flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Info className="size-3.5 text-primary" />
          <span>Phím tắt: [← / →] Đổi bước · [Space] Phát/Dừng · [Esc] Đóng</span>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-teal-600 text-white font-semibold text-xs shadow-xs hover:bg-teal-700 transition-all cursor-pointer"
          >
            Hoàn tất xem xét
          </button>
        )}
      </div>
    </div>
  );

  if (isInline) {
    return content;
  }

  const modalOverlay = (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center p-3 sm:p-5 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200"
      style={{ isolation: "isolate" }}
    >
      {content}
    </div>
  );

  return typeof document !== "undefined"
    ? createPortal(modalOverlay, document.body)
    : modalOverlay;
}
