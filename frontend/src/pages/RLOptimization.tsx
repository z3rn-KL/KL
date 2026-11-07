import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BrainCircuit,
  Network,
  Truck,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
  Info,
  Layers,
  MapPin,
  Clock,
  Compass,
  Eye,
  CheckCircle2,
  AlertTriangle,
  PlayCircle,
  FileCode,
  Lock,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { PlanningDesk } from "@/components/dispatch/PlanningDesk";
import { RlDecisionExplorerModal } from "@/components/dispatch/RlDecisionExplorerModal";
import { useDispatchContext } from "@/context/DispatchContext";

export default function RLOptimization() {
  const navigate = useNavigate();
  const {
    session,
    loading,
    connectionError,
    sessionNotFoundError,
    selectedShipperId,
    setSelectedShipperId,
    nearbyOrders,
    activePreviewPlan,
    setActivePreviewPlan,
    actionLoading,
    actionError,
    actionSuccess,
    initSession,
    createNewSession,
    handleNextDay,
    handleAssign,
    handleUnassign,
    handleAutoDispatch,
    handlePreviewDispatch,
    handleConfirmPreview,
    handleCancelPreview,
    handleSelectAlgorithm,
  } = useDispatchContext();

  const [activeTab, setActiveTab] = useState<"planning" | "architecture" | "rl_theory" | "decision_explorer">("planning");
  const [isDecisionExplorerOpen, setIsDecisionExplorerOpen] = useState(false);

  const onConfirmAndGoToSimulation = async (previewId: string) => {
    await handleConfirmPreview(previewId);
    navigate("/mo-phong");
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3">
        <div className="size-8 rounded-full border-2 border-teal-600 border-t-transparent animate-spin" />
        <span className="text-xs text-muted-foreground font-mono">Đang nạp dữ liệu điều phối và mô hình Proposed RL…</span>
      </div>
    );
  }

  if (connectionError || !session) {
    return (
      <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-center space-y-3 max-w-xl mx-auto my-12">
        <AlertTriangle className="size-8 text-rose-500 mx-auto" />
        <h2 className="text-base font-bold text-foreground">Không thể kết nối đến Backend</h2>
        <p className="text-xs text-muted-foreground">{connectionError || "Vui lòng đảm bảo FastAPI đang chạy tại 127.0.0.1:8011."}</p>
        <Button onClick={() => initSession(true)} className="bg-teal-600 text-white hover:bg-teal-700 text-xs">
          Thử kết nối lại
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-2.5 sm:space-y-3">
      {/* ============================================================== */}
      {/* 1. SCREEN HEADER (Clean & Compact)                             */}
      {/* ============================================================== */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 border-b border-border/70 pb-2">
        <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground">
          Tối Ưu Tuyến Bằng RL
        </h1>

        {/* Quick action buttons */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveTab("decision_explorer")}
            className="gap-1.5 text-xs h-8 border-teal-500/30 text-teal-700 dark:text-teal-300 hover:bg-teal-500/10 cursor-pointer"
          >
            <Eye className="size-3.5" />
            <span>Vết quyết định RL</span>
          </Button>
          <Button
            size="sm"
            onClick={() => navigate("/mo-phong")}
            className="gap-1.5 text-xs h-8 bg-teal-600 text-white hover:bg-teal-700 font-semibold shadow-xs cursor-pointer"
          >
            <PlayCircle className="size-3.5" />
            <span>Sang Mô phỏng</span>
            <ArrowRight className="size-3.5" />
          </Button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. SUB-TABS: PLANNING DESK vs DECISION EXPLORER vs ARCH vs THEORY */}
      {/* ============================================================== */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-3">
        <TabsList className="bg-muted/40 p-1 border border-border/50 rounded-xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-1 w-full max-w-full">
          <TabsTrigger value="planning" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white cursor-pointer">
            <Truck className="size-3.5" />
            <span>1. Bàn Lập Tuyến</span>
          </TabsTrigger>
          <TabsTrigger value="decision_explorer" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white cursor-pointer">
            <Eye className="size-3.5" />
            <span>2. Vết Quyết Định RL</span>
          </TabsTrigger>
          <TabsTrigger value="architecture" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white cursor-pointer">
            <Layers className="size-3.5" />
            <span>3. Kiến Trúc Hai Tầng</span>
          </TabsTrigger>
          <TabsTrigger value="rl_theory" className="gap-1.5 text-xs font-semibold data-[state=active]:bg-teal-600 data-[state=active]:text-white cursor-pointer">
            <BrainCircuit className="size-3.5" />
            <span>4. Hàm Thưởng &amp; 8 Đặc Trưng</span>
          </TabsTrigger>
        </TabsList>

        {/* ============================================================== */}
        {/* TAB 1: PLANNING DESK & ROUTE PREVIEW                           */}
        {/* ============================================================== */}
        <TabsContent value="planning" className="m-0 space-y-3">
          <PlanningDesk
            session={session}
            selectedShipperId={selectedShipperId}
            onSelectShipper={setSelectedShipperId}
            nearbyOrders={nearbyOrders}
            onAssign={handleAssign}
            onUnassign={handleUnassign}
            onAutoDispatch={handleAutoDispatch}
            onPreviewPlan={handlePreviewDispatch}
            onConfirmPreview={onConfirmAndGoToSimulation}
            onCancelPreview={handleCancelPreview}
            onSwitchToLive={() => navigate("/mo-phong")}
            onSelectAlgorithm={handleSelectAlgorithm}
            actionLoading={actionLoading}
            actionError={actionError}
            actionSuccess={actionSuccess}
            onReloadSession={initSession}
            onSwitchToReport={() => navigate("/bao-cao")}
            onCreateNewSession={createNewSession}
            onNextDay={handleNextDay}
            previewPlan={activePreviewPlan}
            onSetPreviewPlan={setActivePreviewPlan}
          />
        </TabsContent>

        {/* ============================================================== */}
        {/* TAB 2: DIRECT RL DECISION EXPLORER                            */}
        {/* ============================================================== */}
        <TabsContent value="decision_explorer" className="m-0 space-y-3">
          <RlDecisionExplorerModal
            isOpen={true}
            isInline={true}
            sessionId={session.session_id}
            selectedShipperId={selectedShipperId}
          />
        </TabsContent>

        {/* ============================================================== */}
        {/* TAB 3: ALGORITHM ARCHITECTURE FLOW                             */}
        {/* ============================================================== */}
        <TabsContent value="architecture" className="m-0 space-y-3">
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="border-b border-border/60 pb-3">
              <div className="flex items-center gap-2">
                <Layers className="size-4 text-teal-600" />
                <CardTitle className="text-sm font-bold uppercase tracking-wider text-foreground">
                  Kiến Trúc Tối Ưu Tuyến Hai Tầng (Two-Tier Hierarchical Framework)
                </CardTitle>
              </div>
              <CardDescription className="text-xs">
                Phân định rành mạch giữa Tầng Điều phối (Dispatch Allocation) và Tầng Định tuyến (Stop Sequencing RL)
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Tier 1: Dispatch Allocation */}
                <div className="p-4 rounded-xl border border-teal-500/30 bg-teal-500/5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold uppercase text-teal-600 dark:text-teal-400">
                      TẦNG 1: ĐIỀU PHỐI ĐƠN HÀNG (CLUSTER AFFINITY)
                    </span>
                    <Badge variant="outline" className="text-[10px] font-mono">Affinity Heuristic</Badge>
                  </div>
                  <h3 className="text-sm font-bold text-foreground">
                    Phân bổ đơn hàng cho 16 Shipper theo Cụm Không Gian
                  </h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Đơn hàng mới xuất hiện trong ngày được phân cụm K-Means K=10. Hệ thống gán các đơn có cùng hoặc gần nhãn cụm cho shipper phụ trách khu vực tương ứng, bảo đảm giới hạn tải trọng xe <strong className="text-foreground">≤ 100 kg</strong> và cân bằng tải giữa các shipper.
                  </p>
                  <div className="p-2.5 rounded-lg bg-background border border-border text-[11px] text-muted-foreground">
                    <strong className="text-foreground">Lưu ý khoa học:</strong> Cơ chế phân bổ đơn hàng cho shipper <span className="text-rose-500 font-semibold">KHÔNG đồng nhất</span> với Candidate-Scoring DQN. Tầng 1 chỉ quyết định tập đơn giao ban đầu, còn thứ tự ghé điểm do Tầng 2 quyết định.
                  </div>
                </div>

                {/* Tier 2: Candidate-Scoring RL */}
                <div className="p-4 rounded-xl border border-teal-500/30 bg-teal-500/5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold uppercase text-teal-600 dark:text-teal-400">
                      TẦNG 2: ĐỊNH TUYẾN THỨ TỰ GHÉ ĐIỂM (PROPOSED RL)
                    </span>
                    <Badge variant="outline" className="text-[10px] font-mono">Candidate-Scoring DQN</Badge>
                  </div>
                  <h3 className="text-sm font-bold text-foreground">
                    Mô hình Học Tăng Cường Candidate-Scoring DQN
                  </h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Tại mỗi điểm dừng hiện tại, mạng nơ-ron nhận vector 8 đặc trưng bất biến của từng điểm giao ứng viên còn lại, tính toán điểm giá trị hành động <strong className="text-foreground">Q(s, a)</strong> và chọn điểm có Q cao nhất theo chính sách tham lam:
                    <span className="block my-1 font-mono text-center text-teal-700 dark:text-teal-300 font-bold">
                      π(s) = argmax Q(s, a)
                    </span>
                  </p>
                  <div className="p-2.5 rounded-lg bg-background border border-border text-[11px] text-muted-foreground">
                    <strong className="text-foreground">Cơ chế Fallback minh bạch:</strong> Khi số đơn hàng còn lại ≤ 2 hoặc không khả dụng inference, hệ thống sử dụng thuật toán Heuristic Nearest Neighbor có kiểm soát và công bố rõ ràng.
                  </div>
                </div>
              </div>

              {/* End-to-end Sequence Flow Diagram */}
              <div className="rounded-xl border border-border p-3.5 bg-muted/20 space-y-2.5">
                <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                  Chuỗi quy trình dữ liệu từ Kho đến Điểm cuối
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                  <span className="px-2.5 py-1 rounded bg-card border border-border font-bold">Depot Tân Bình</span>
                  <ArrowRight className="size-3 text-muted-foreground" />
                  <span className="px-2.5 py-1 rounded bg-teal-600/10 border border-teal-500/40 text-teal-700 dark:text-teal-300 font-bold">K-Means K=10</span>
                  <ArrowRight className="size-3 text-muted-foreground" />
                  <span className="px-2.5 py-1 rounded bg-card border border-border font-bold">Tập đơn Shipper ≤ 100kg</span>
                  <ArrowRight className="size-3 text-muted-foreground" />
                  <span className="px-2.5 py-1 rounded bg-teal-600 text-white font-bold">Candidate-Scoring DQN</span>
                  <ArrowRight className="size-3 text-muted-foreground" />
                  <span className="px-2.5 py-1 rounded bg-card border border-border font-bold">Thứ tự Stop Sequence</span>
                  <ArrowRight className="size-3 text-muted-foreground" />
                  <span className="px-2.5 py-1 rounded bg-card border border-border font-bold">Quay về Depot</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ============================================================== */}
        {/* TAB 4: REINFORCEMENT LEARNING THEORY (GROUND TRUTH FORMULAS)   */}
        {/* ============================================================== */}
        <TabsContent value="rl_theory" className="m-0 space-y-3">
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="border-b border-border/60 pb-3">
              <div className="flex items-center gap-2">
                <BrainCircuit className="size-4 text-teal-600" />
                <CardTitle className="text-sm font-bold uppercase tracking-wider text-foreground">
                  Đặc Tả Khoa Học Học Tăng Cường (Reinforcement Learning Ground Truth)
                </CardTitle>
              </div>
              <CardDescription className="text-xs">
                Toàn bộ công thức thưởng/phạt và vector 8 đặc trưng được trích xuất trực tiếp từ mã nguồn nghiên cứu
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4 space-y-5">
              {/* Formula and Weights */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400">
                  1. Hàm Mục Tiêu J(π) và Tín Hiệu Phần Thưởng r_t
                </h3>
                <div className="p-3.5 rounded-xl bg-muted/40 border border-border/70 font-mono text-xs sm:text-sm space-y-1.5">
                  <div className="text-foreground font-bold">
                    Hàm phần thưởng bước (Step Reward):
                  </div>
                  <div className="text-teal-700 dark:text-teal-300">
                    r_t = - ΔJ_t = - (w_d · D̄_t + w_t · T̄_t + w_m · M̄_t + w_l · L̄_t + w_c · S̄_t)
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
                  <div className="p-2 rounded-lg border border-border bg-card">
                    <span className="text-[10px] text-muted-foreground block">Cự ly (w_d)</span>
                    <strong className="font-mono text-teal-600 dark:text-teal-400">0.375</strong>
                  </div>
                  <div className="p-2 rounded-lg border border-border bg-card">
                    <span className="text-[10px] text-muted-foreground block">Thời gian (w_t)</span>
                    <strong className="font-mono text-teal-600 dark:text-teal-400">0.375</strong>
                  </div>
                  <div className="p-2 rounded-lg border border-border bg-card">
                    <span className="text-[10px] text-muted-foreground block">Giao trễ (w_m)</span>
                    <strong className="font-mono text-foreground">0.100</strong>
                  </div>
                  <div className="p-2 rounded-lg border border-border bg-card">
                    <span className="text-[10px] text-muted-foreground block">Phút trễ (w_l)</span>
                    <strong className="font-mono text-foreground">0.100</strong>
                  </div>
                  <div className="p-2 rounded-lg border border-teal-500/40 bg-teal-500/10">
                    <span className="text-[10px] text-teal-700 dark:text-teal-300 block font-bold">Phạt đổi cụm (w_c)</span>
                    <strong className="font-mono text-teal-600 dark:text-teal-400">0.050</strong>
                  </div>
                </div>
              </div>

              {/* 8 Invariant Candidate Features */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400">
                  2. Vector 8 Đặc Trưng Bất Biến Đầu Vào (Candidate Encoder Invariant Features)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
                  {[
                    { name: "elapsed_ratio", desc: "Tỷ lệ thời gian ca đã trôi qua so với tổng thời lượng ca 600 phút." },
                    { name: "remaining_ratio", desc: "Tỷ lệ số đơn hàng còn lại cần phục vụ trong tuyến của shipper." },
                    { name: "distance_ratio", desc: "Khoảng cách từ điểm hiện tại đến ứng viên, chuẩn hóa theo cự ly tham chiếu (10km)." },
                    { name: "travel_time_ratio", desc: "Thời gian di chuyển ước tính đến ứng viên, chuẩn hóa theo 15 phút." },
                    { name: "slack_ratio", desc: "Khoảng đệm thời gian trước hạn chót SLA của đơn ứng viên." },
                    { name: "deadline_valid", desc: "Cờ nhị phân (0 hoặc 1) chỉ định đơn hàng có ràng buộc mốc giờ SLA hay không." },
                    { name: "lateness_ratio", desc: "Số phút trễ dự kiến nếu chọn ứng viên này, chuẩn hóa theo mốc 30 phút." },
                    { name: "cluster_switch", desc: "Cờ nhị phân (0 hoặc 1): Bằng 1 nếu ứng viên thuộc cụm khác với điểm dừng vừa ghé." },
                  ].map((feat, i) => (
                    <div key={feat.name} className="p-2.5 rounded-lg border border-border bg-card space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-bold text-teal-600 dark:text-teal-400 text-[11px]">{i + 1}. {feat.name}</span>
                      </div>
                      <p className="text-[11px] text-muted-foreground leading-snug">{feat.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Checkpoint Provenance */}
              <div className="p-3.5 rounded-xl border border-border/80 bg-muted/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div>
                  <span className="font-bold text-foreground block">Mô hình PyTorch Đã Huấn Luyện &amp; Khóa Trọng Số:</span>
                  <span className="font-mono text-muted-foreground">results/proposed_rl_multiworkload_D_moderate_sla_best.pt</span>
                </div>
                <Badge variant="outline" className="border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 font-mono text-[10px] w-fit">
                  LOCKED CHECKPOINT
                </Badge>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Decision Explorer Modal */}
      <RlDecisionExplorerModal
        isOpen={isDecisionExplorerOpen}
        onClose={() => setIsDecisionExplorerOpen(false)}
        sessionId={session.session_id}
        selectedShipperId={selectedShipperId}
      />
    </div>
  );
}
