import React from "react";
import { Scale, Layers, Clock, ShieldCheck, Cpu, Activity, Award } from "lucide-react";

interface RewardPenaltySectionProps {
  defaultOpen?: boolean;
  className?: string;
}

export const RewardPenaltySection: React.FC<RewardPenaltySectionProps> = ({
  defaultOpen = true,
  className = "",
}) => {
  return (
    <details
      open={defaultOpen}
      className={`rounded-xl border border-border/80 bg-card p-4 sm:p-5 shadow-2xs space-y-4 text-xs transition-colors ${className}`}
    >
      <summary className="font-bold text-sm text-foreground cursor-pointer select-none flex items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <Scale className="size-4 text-teal-600 dark:text-teal-400" />
          <span>Cơ chế thưởng / phạt trong học tăng cường (Reward &amp; Penalty Formulation)</span>
        </span>
        <span className="text-[11px] font-mono font-medium text-muted-foreground bg-muted/50 px-2 py-0.5 rounded border border-border/60">
          Source of Truth: Backend Verified
        </span>
      </summary>

      <div className="pt-3 border-t border-border/60 space-y-4">
        {/* ============================================================== */}
        {/* A. HÀNG TRÊN: 4 THẺ KHÁI NIỆM CỐT LÕI (2x2 Grid)               */}
        {/* ============================================================== */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
            <Layers className="size-3 text-teal-600 dark:text-teal-400" />
            <span>Phân biệt 4 khái niệm nền tảng trong mô hình tối ưu</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Thẻ 1: r_t */}
            <div className="p-3 rounded-xl bg-muted/20 border border-border/70 space-y-1 shadow-2xs">
              <strong className="text-emerald-600 dark:text-emerald-400 block font-semibold text-xs flex items-center gap-1.5">
                <Activity className="size-3.5" />
                <span>1. Phần thưởng huấn luyện bước (r_t)</span>
              </strong>
              <p className="text-muted-foreground m-0 leading-relaxed text-[11.5px]">
                Tín hiệu phản hồi theo từng bước quyết định gán đơn. Với Proposed RL,{" "}
                <code className="font-mono text-foreground font-semibold bg-muted/60 px-1 py-0.5 rounded text-[11px]">
                  r_t = -ΔJ_t ≤ 0
                </code>
                , phạt trực tiếp số gia chi phí để ép mạng học cách giảm thiểu tổn thất cục bộ.
              </p>
            </div>

            {/* Thẻ 2: J(π) */}
            <div className="p-3 rounded-xl bg-muted/20 border border-border/70 space-y-1 shadow-2xs">
              <strong className="text-blue-600 dark:text-blue-400 block font-semibold text-xs flex items-center gap-1.5">
                <Cpu className="size-3.5" />
                <span>2. Hàm mục tiêu toàn cục (J(π))</span>
              </strong>
              <p className="text-muted-foreground m-0 leading-relaxed text-[11.5px]">
                Tổng chi phí đa tiêu chí cần tối thiểu hóa trên toàn tuyến:{" "}
                <code className="font-mono text-foreground font-semibold bg-muted/60 px-1 py-0.5 rounded text-[11px]">
                  min_π J(π) = ∑ w_i · f̄_i
                </code>
                , tích hợp chuẩn hóa quãng đường, thời gian, số đơn trễ, phút trễ và chuyển cụm với{" "}
                <code className="font-mono text-foreground text-[11px]">∑ w_i = 1,0</code>.
              </p>
            </div>

            {/* Thẻ 3: Q(s, a) */}
            <div className="p-3 rounded-xl bg-muted/20 border border-border/70 space-y-1 shadow-2xs">
              <strong className="text-purple-600 dark:text-purple-400 block font-semibold text-xs flex items-center gap-1.5">
                <Award className="size-3.5" />
                <span>3. Giá trị kỳ vọng hành động (Q(s, a))</span>
              </strong>
              <p className="text-muted-foreground m-0 leading-relaxed text-[11.5px]">
                Điểm đánh giá chất lượng hành động do mạng DQN xấp xỉ:{" "}
                <code className="font-mono text-foreground font-semibold bg-muted/60 px-1 py-0.5 rounded text-[11px]">
                  Q(s, a) = E[r_t + γ·max_a' Q(s', a')]
                </code>
                . Khi suy luận, agent thực hiện chính sách tham lam{" "}
                <code className="font-mono text-foreground text-[11px]">argmax_a Q(s, a)</code> để chọn đơn tiếp theo.
              </p>
            </div>

            {/* Thẻ 4: Chi phí nghiệp vụ */}
            <div className="p-3 rounded-xl bg-muted/20 border border-border/70 space-y-1 shadow-2xs">
              <strong className="text-amber-600 dark:text-amber-400 block font-semibold text-xs flex items-center gap-1.5">
                <Clock className="size-3.5" />
                <span>4. Chi phí nghiệp vụ thực tế (VNĐ)</span>
              </strong>
              <p className="text-muted-foreground m-0 leading-relaxed text-[11.5px]">
                Chi phí logistics thực tế tại màn hình Mô phỏng và Báo cáo ca (xăng xe 4.000 đ/km, lương tài xế 40.000 đ/h, phạt trễ 1.000 đ/phút). Hạch toán độc lập với điểm thưởng RL.
              </p>
            </div>
          </div>
        </div>

        {/* ============================================================== */}
        {/* B. HÀNG DƯỚI: 3 KHỐI BACKEND-LEVEL EXPLANATION (3 Columns)       */}
        {/* ============================================================== */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
            <Scale className="size-3 text-teal-600 dark:text-teal-400" />
            <span>Đặc tả công thức và trọng số từ mã nguồn backend</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {/* Card 1: Proposed Candidate RL */}
            <div className="p-3 rounded-xl border border-teal-500/30 bg-card space-y-2.5 shadow-2xs">
              <div className="flex items-center justify-between border-b border-border/60 pb-1.5">
                <strong className="text-teal-700 dark:text-teal-300 font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                  <Activity className="size-3.5" />
                  Proposed Candidate RL
                </strong>
                <span className="text-[10px] font-mono text-muted-foreground">proposed_rl_solver.py</span>
              </div>

              {/* Formulas block */}
              <div className="bg-muted/30 p-2 rounded-lg border border-border/50 font-mono text-[11px] space-y-1 text-foreground">
                <div className="text-emerald-600 dark:text-emerald-400 font-semibold">
                  r_t = -ΔJ_t ≤ 0
                </div>
                <div className="text-muted-foreground text-[10.5px] leading-tight">
                  J(π) = 0.375·D̄ + 0.375·T̄
                  <br />
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.10·M̄ + 0.10·L̄ + 0.05·S̄
                </div>
              </div>

              {/* Max 5 short bullets */}
              <ul className="space-y-1 text-muted-foreground text-[11.5px] list-disc list-inside">
                <li><strong>Cự ly (w_d = 0,375):</strong> Chuẩn hóa trên cự ly tham chiếu 10,0 km.</li>
                <li><strong>Thời gian (w_t = 0,375):</strong> Chuẩn hóa trên thời gian tham chiếu 15,0 phút.</li>
                <li><strong>Giao trễ / Phút trễ (0,20):</strong> 0,10 phạt số đơn trễ + 0,10 phạt số phút trễ.</li>
                <li><strong>Đổi cụm (w_c = 0,050):</strong> Phạt mềm định hướng shipper giữ vùng K-Means.</li>
                <li><strong>Trở về kho:</strong> Chịu chi phí cự ly và thời gian về kho, không tính phạt trễ.</li>
              </ul>
            </div>

            {/* Card 2: Guided DQN & Tabular RL */}
            <div className="p-3 rounded-xl border border-border bg-card space-y-2.5 shadow-2xs">
              <div className="flex items-center justify-between border-b border-border/60 pb-1.5">
                <strong className="text-foreground font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                  <Layers className="size-3.5" />
                  Guided DQN &amp; Tabular RL
                </strong>
                <span className="text-[10px] font-mono text-muted-foreground">routing_environment.py</span>
              </div>

              {/* Formulas block */}
              <div className="bg-muted/30 p-2 rounded-lg border border-border/50 font-mono text-[11px] text-foreground">
                <div className="text-blue-600 dark:text-blue-400 font-semibold">
                  R = Thưởng - Phạt - λ·I(Đổi cụm)
                </div>
                <div className="text-muted-foreground text-[10.5px]">
                  Guided DQN áp dụng λ = 0,50
                </div>
              </div>

              {/* Max 5 short bullets */}
              <ul className="space-y-1 text-muted-foreground text-[11.5px] list-disc list-inside">
                <li><strong>Thưởng ưu tiên (+1,00):</strong> Tỷ lệ thuận theo điểm khẩn cấp của đơn hàng.</li>
                <li><strong>Thưởng hiệu quả (+1,00):</strong> Khuyến khích tối ưu hành trình cục bộ.</li>
                <li><strong>Phạt cự ly &amp; thời gian (-0,50):</strong> Tỷ lệ quãng đường (10 km) và thời gian (15 p).</li>
                <li><strong>Phạt vi phạm hạn (-4,00):</strong> Phạt nặng cố định khi đơn hàng giao trễ SLA.</li>
                <li><strong>Phạt phút trễ &amp; đổi cụm:</strong> Tuyến tính theo phút trễ (-1,00) và đổi cụm (-λ).</li>
              </ul>
            </div>

            {/* Card 3: Điều phối ca & Mô phỏng */}
            <div className="p-3 rounded-xl border border-border bg-card space-y-2.5 shadow-2xs">
              <div className="flex items-center justify-between border-b border-border/60 pb-1.5">
                <strong className="text-amber-600 dark:text-amber-400 font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                  <Clock className="size-3.5" />
                  Điều phối ca &amp; Mô phỏng
                </strong>
                <span className="text-[10px] font-mono text-muted-foreground">Chi phí nghiệp vụ</span>
              </div>

              {/* Formulas block */}
              <div className="bg-muted/30 p-2 rounded-lg border border-border/50 font-mono text-[11px] text-foreground">
                <div className="text-amber-600 dark:text-amber-400 font-semibold">
                  Chi phí thực tế = Cước - Vận hành
                </div>
                <div className="text-muted-foreground text-[10.5px]">
                  Hạch toán tài chính (VNĐ), không phải điểm RL
                </div>
              </div>

              {/* Max 5 short bullets */}
              <ul className="space-y-1 text-muted-foreground text-[11.5px] list-disc list-inside">
                <li><strong>Hoa hồng:</strong> 3% cước đơn đúng hạn; thưởng thêm đến 7% khi giao sớm ≥ 60p.</li>
                <li><strong>Xăng xe:</strong> 4.000 VNĐ/km theo cự ly thực tế trên mạng lưới đường bộ.</li>
                <li><strong>Nhân công:</strong> Lương cứng cơ bản 40.000 VNĐ/giờ (tương đương 667 đ/phút).</li>
                <li><strong>Phạt giao trễ:</strong> Khấu trừ 50% cước phí + phạt 1.000 VNĐ cho mỗi phút trễ hạn.</li>
                <li><strong>Phạt tăng ca:</strong> Áp dụng hệ số tăng ca lũy tiến khi thời gian ca vượt 600 phút.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* ============================================================== */}
        {/* C. FOOTER CỦA SECTION                                          */}
        {/* ============================================================== */}
        <div className="pt-2 border-t border-border/50 flex items-center justify-between text-[11px] text-muted-foreground flex-wrap gap-2">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="size-3.5 text-emerald-500 shrink-0" />
            <span>
              Tham số và công thức được trích xuất từ source backend / artifact thực tế; giao diện chỉ diễn giải, không thay đổi thuật toán.
            </span>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground/80 shrink-0">
            routing_environment.py · proposed_rl_solver.py · dispatch_engine.py
          </span>
        </div>
      </div>
    </details>
  );
};

