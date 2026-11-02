import React, { useState, useEffect, useRef } from "react";
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
} from "lucide-react";
import {
  api,
  type RlDecisionTraceResponse,
  type DecisionTraceStep,
  type CandidateEvaluation,
} from "@/lib/api";

interface RlDecisionExplorerModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  selectedShipperId?: string;
  onSelectOrder?: (orderId: string) => void;
}

export function RlDecisionExplorerModal({
  isOpen,
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
  const playTimerRef = useRef<any>(null);

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
        onClose();
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

  if (!isOpen) return null;

  const currentStep: DecisionTraceStep | undefined =
    traceData?.decision_trace?.[activeStepIdx];
  const totalSteps = traceData?.decision_trace?.length || 0;

  // Find first active shipper with orders if available
  const firstActiveShipper = traceData?.active_shippers?.find((s) => s.order_count > 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div
        className="relative flex flex-col w-full max-w-5xl max-h-[92vh] bg-card border border-border/80 rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
        aria-label="RL Decision Explorer"
      >
        {/* 1. Header with Scientific Provenance & Shipper Selector */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-border/80 bg-muted/30">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20 shadow-2xs shrink-0">
              <BrainCircuit className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-foreground">
                  RL Decision Explorer · Vết Quyết Định Mạng Nơ-ron
                </h2>
                <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-mono font-bold text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  Candidate-Scoring DQN (PyTorch)
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Giải thích từng bước lựa chọn điểm đến dựa trên 8 vector đặc trưng và hàm giá trị Q(s, a) từ mô hình đề xuất.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
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

            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
              aria-label="Đóng cửa sổ"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>

        {/* 2. Objective Weights Banner */}
        <div className="px-5 py-2 bg-muted/15 border-b border-border/60 flex flex-wrap items-center justify-between gap-2 text-[11px]">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Target className="size-3.5 text-primary" />
            <span className="font-semibold text-foreground">Trọng số Đa mục tiêu Locked:</span>
          </div>
          <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
            <span className="bg-background px-2 py-0.5 rounded border border-border/60">
              Cự ly (w_d): <strong className="text-foreground">0.375</strong>
            </span>
            <span className="bg-background px-2 py-0.5 rounded border border-border/60">
              Thời gian (w_t): <strong className="text-foreground">0.375</strong>
            </span>
            <span className="bg-background px-2 py-0.5 rounded border border-border/60">
              Chỉ dấu trễ (w_late): <strong className="text-foreground">0.100</strong>
            </span>
            <span className="bg-background px-2 py-0.5 rounded border border-border/60">
              Độ trễ SLA (w_dur): <strong className="text-foreground">0.100</strong>
            </span>
            <span className="bg-background px-2 py-0.5 rounded border border-border/60">
              Đổi cụm (w_c): <strong className="text-foreground">0.050</strong>
            </span>
          </div>
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
          ) : error ? (
            <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-6 text-center space-y-3">
              <ShieldAlert className="size-8 text-destructive mx-auto" />
              <h3 className="text-sm font-semibold text-foreground">Không thể nạp vết quyết định cho Shipper này</h3>
              <p className="text-xs text-muted-foreground max-w-md mx-auto">{error}</p>
              {firstActiveShipper && firstActiveShipper.shipper_id !== targetShipperId && (
                <button
                  type="button"
                  className="btn primary sm mx-auto mt-2"
                  onClick={() => setTargetShipperId(firstActiveShipper.shipper_id)}
                >
                  Chuyển sang {firstActiveShipper.name} ({firstActiveShipper.order_count} đơn)
                </button>
              )}
            </div>
          ) : currentStep ? (
            <>
              {/* Step Navigation Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-muted/30 border border-border/80">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-foreground">
                    Bước {activeStepIdx + 1} / {totalSteps}
                  </span>
                  <span className="text-[11px] text-muted-foreground font-mono">
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
                    className="px-3 py-1.5 rounded-lg bg-primary text-primary-foreground font-medium text-xs flex items-center gap-1.5 shadow-2xs hover:bg-primary/90 transition-colors cursor-pointer"
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
                {/* Left Card: Step Evaluation & Reward Delta */}
                <div className="space-y-3">
                  <div className="p-4 rounded-xl border border-border/80 bg-card space-y-3">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground block">
                      ĐÁNH GIÁ HÀNH ĐỘNG RL
                    </span>

                    <div className="space-y-2 text-xs">
                      <div className="flex items-center justify-between p-2 rounded-lg bg-primary/5 border border-primary/20">
                        <span className="font-medium text-foreground">Điểm chọn:</span>
                        <strong className="text-primary font-mono text-sm">
                          {currentStep.chosen_delivery_id}
                        </strong>
                      </div>

                      <div className="flex items-center justify-between p-1.5 rounded bg-muted/40">
                        <span className="text-muted-foreground">Thời điểm đến:</span>
                        <span className="font-mono font-medium">
                          {currentStep.cumulative_time_minutes.toFixed(1)} phút
                        </span>
                      </div>

                      <div className="flex items-center justify-between p-1.5 rounded bg-muted/40">
                        <span className="text-muted-foreground">Trạng thái SLA:</span>
                        {currentStep.lateness_minutes > 0 ? (
                          <span className="text-rose-600 dark:text-rose-400 font-bold flex items-center gap-1">
                            <ShieldAlert className="size-3" />
                            Trễ {currentStep.lateness_minutes.toFixed(1)}p
                          </span>
                        ) : (
                          <span className="text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1">
                            <CheckCircle2 className="size-3" />
                            Đúng hạn SLA
                          </span>
                        )}
                      </div>

                      <div className="flex items-center justify-between p-1.5 rounded bg-muted/40">
                        <span className="text-muted-foreground">Tổng cự ly:</span>
                        <strong>{currentStep.cumulative_distance_km.toFixed(2)} km</strong>
                      </div>

                      <div className="flex items-center justify-between p-1.5 rounded bg-muted/40">
                        <span className="text-muted-foreground">Số ứng viên so sánh:</span>
                        <span className="font-mono font-bold">
                          {currentStep.candidates_evaluated.length} đơn
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Objective & Step Reward Delta */}
                  <div className="p-4 rounded-xl border border-border/80 bg-card space-y-2.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground block">
                      HÀM MỤC TIÊU & PHẦN THƯỞNG
                    </span>

                    <div className="space-y-1.5 text-xs font-mono">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Thời gian lũy kế:</span>
                        <span>{currentStep.cumulative_time_minutes.toFixed(1)} phút</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Số lần đổi cụm:</span>
                        <span>{currentStep.cluster_switches} lần</span>
                      </div>
                      <div className="flex items-center justify-between font-bold pt-1 border-t border-border/60">
                        <span className="text-primary">Độ lệch mục tiêu (ΔJ):</span>
                        <span className="text-primary">{(-currentStep.step_reward).toFixed(3)}</span>
                      </div>
                      <div className="flex items-center justify-between font-bold text-emerald-600 dark:text-emerald-400">
                        <span>Phần thưởng bước (r_t = -ΔJ):</span>
                        <span>{currentStep.step_reward.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Card: Evaluated Candidates Table */}
                <div className="md:col-span-2 rounded-xl border border-border/80 bg-card overflow-hidden flex flex-col">
                  <div className="p-3 border-b border-border/80 bg-muted/20 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Sparkles className="size-4 text-primary" />
                      <h4 className="text-xs font-bold text-foreground">
                        Bảng Điểm Q(s, a) Các Ứng Viên Khả Dĩ
                      </h4>
                    </div>
                    <span className="text-[11px] text-muted-foreground font-medium">
                      Sắp xếp theo Điểm Q(s, a) giảm dần
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
                              ? "bg-primary/10 border-l-4 border-primary"
                              : "hover:bg-muted/40"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1.5">
                            <div className="flex items-center gap-2">
                              <span
                                className={`size-5 rounded-full flex items-center justify-center font-mono text-[10px] font-bold ${
                                  isChosen
                                    ? "bg-primary text-primary-foreground"
                                    : "bg-muted text-muted-foreground"
                                }`}
                              >
                                {idx + 1}
                              </span>
                              <div>
                                <strong className="text-foreground block text-xs">
                                  {cand.delivery_id}
                                </strong>
                                <span className="text-[10px] text-muted-foreground">
                                  {cand.package_name || "Kiện hàng"} · Cụm #{cand.cluster_id ?? 0}
                                </span>
                              </div>
                            </div>

                            <div className="text-right">
                              <span className="font-mono font-bold text-xs text-foreground block">
                                Q = {cand.q_score.toFixed(4)}
                              </span>
                              {isChosen && (
                                <span className="inline-block px-1.5 py-0.2 rounded bg-primary text-primary-foreground text-[9px] font-bold uppercase">
                                  Đã chọn bởi Agent
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Relative Q Bar */}
                          <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className={`h-full transition-all ${isChosen ? "bg-primary" : "bg-muted-foreground/40"}`}
                              style={{ width: `${normBar}%` }}
                            />
                          </div>

                          {/* Feature Dict Details */}
                          {cand.feature_dict && (
                            <div className="mt-2 grid grid-cols-4 gap-1 text-[9px] font-mono text-muted-foreground bg-muted/20 p-1.5 rounded">
                              <div>Cự ly: {cand.feature_dict.norm_distance?.toFixed(2) ?? "-"}</div>
                              <div>Thời gian: {cand.feature_dict.norm_travel_time?.toFixed(2) ?? "-"}</div>
                              <div>Hạn SLA: {cand.feature_dict.deadline_tightness?.toFixed(2) ?? "-"}</div>
                              <div>Cụm: {cand.feature_dict.cluster_alignment?.toFixed(2) ?? "-"}</div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </>
          ) : null}
        </div>

        {/* 4. Footer */}
        <div className="px-5 py-3 border-t border-border/80 bg-muted/20 flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 text-muted-foreground text-[11px]">
            <Info className="size-3.5 text-primary" />
            <span>Phím tắt: [← / →] Đổi bước · [Space] Phát/Dừng · [Esc] Đóng</span>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-primary text-primary-foreground font-semibold text-xs shadow-xs hover:bg-primary/90 transition-all cursor-pointer"
          >
            Hoàn tất xem xét
          </button>
        </div>
      </div>
    </div>
  );
}
