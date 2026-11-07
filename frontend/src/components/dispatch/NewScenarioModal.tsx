import React, { useState } from "react";
import { X, Calendar, Check, Layers, AlertCircle } from "lucide-react";

interface NewScenarioModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (totalDays: number) => Promise<void>;
  loading?: boolean;
}

const SCENARIO_OPTIONS = [
  {
    days: 3,
    label: "3 ngày (Kịch bản chuẩn khóa luận)",
    orders: 255,
    badge: "Chuẩn KLTN",
    description: "Kịch bản thực nghiệm mặc định của khóa luận: 85 đơn/ngày x 3 ngày = 255 đơn, bảo toàn tính tương thích với toàn bộ baseline so sánh.",
  },
  {
    days: 5,
    label: "5 ngày (Kịch bản mở rộng trung hạn)",
    orders: 425,
    badge: "Mở rộng",
    description: "Mô phỏng chuỗi 5 ca giao hàng liên tục (425 đơn), kiểm chứng cơ chế chuyển giao đơn tồn đọng (carryover) qua nhiều chu kỳ liên tiếp.",
  },
  {
    days: 7,
    label: "7 ngày (Mở rộng trọn vẹn 1 tuần)",
    orders: 595,
    badge: "1 tuần",
    description: "Mô phỏng trọn vẹn 1 tuần hoạt động logistics (595 đơn), chứng minh độ ổn định hội tụ và khả năng mở rộng thời gian của thuật toán.",
  },
  {
    days: 14,
    label: "14 ngày (Giới hạn tối đa API)",
    orders: 1190,
    badge: "Tối đa API",
    description: "Quy mô tối đa được hỗ trợ bởi API FastAPI (1.190 đơn). Tập dữ liệu gốc XeDu (2.394 đơn) hoàn toàn đáp ứng không trùng lặp.",
  },
];

export function NewScenarioModal({
  isOpen,
  onClose,
  onConfirm,
  loading = false,
}: NewScenarioModalProps) {
  const [selectedDays, setSelectedDays] = useState<number>(3);

  if (!isOpen) return null;

  const handleConfirm = async () => {
    await onConfirm(selectedDays);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-xl border border-border bg-card p-6 shadow-2xl relative text-foreground animate-in fade-in-50 zoom-in-95"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 text-muted-foreground hover:text-foreground cursor-pointer"
          aria-label="Đóng"
        >
          <X className="size-5" />
        </button>

        <div className="flex items-center gap-3 border-b border-border/70 pb-4 mb-4">
          <div className="flex size-10 items-center justify-center rounded-lg bg-teal-600/10 text-teal-600 border border-teal-500/20">
            <Calendar className="size-5" />
          </div>
          <div>
            <h2 className="text-base font-bold tracking-tight text-foreground">
              Khởi Tạo Kịch Bản Điều Phối Mới
            </h2>
            <p className="text-xs text-muted-foreground">
              Lựa chọn quy mô số ngày mô phỏng (Scalability Verification)
            </p>
          </div>
        </div>

        {/* Scientific Context Callout */}
        <div className="mb-4 rounded-lg border border-teal-500/30 bg-teal-500/5 p-3 text-xs text-muted-foreground leading-relaxed">
          <div className="flex items-center gap-1.5 font-bold text-teal-700 dark:text-teal-300 mb-1">
            <Layers className="size-3.5 shrink-0" />
            <span>Khả năng mở rộng quy mô (Dataset Scalability):</span>
          </div>
          <p>
            Tập dữ liệu XeDu K-Means chuẩn có <b>2.394 đơn hàng</b> thực tế tại TP.HCM. API tự động trích xuất các đơn không trùng lặp cho mỗi ngày mới (85 đơn/ngày), bảo toàn tọa độ, khối lượng và cụm địa lý.
          </p>
        </div>

        {/* Options List */}
        <div className="space-y-2.5 mb-5">
          {SCENARIO_OPTIONS.map((opt) => {
            const isSelected = selectedDays === opt.days;
            return (
              <div
                key={opt.days}
                onClick={() => setSelectedDays(opt.days)}
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? "border-teal-500 bg-teal-500/10 shadow-xs"
                    : "border-border/80 hover:bg-muted/50"
                }`}
              >
                <input
                  type="radio"
                  name="scenarioDays"
                  checked={isSelected}
                  onChange={() => setSelectedDays(opt.days)}
                  className="mt-1 size-4 text-teal-600 cursor-pointer"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold text-foreground">
                      {opt.label}
                    </span>
                    <span className="rounded-full bg-teal-600/15 border border-teal-500/30 px-2 py-0.2 text-[10px] font-bold text-teal-700 dark:text-teal-300 shrink-0">
                      {opt.badge}
                    </span>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-1 leading-snug">
                    {opt.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-end gap-2 pt-3 border-t border-border/70">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-border text-xs font-semibold text-muted-foreground hover:bg-muted cursor-pointer transition-colors"
          >
            Hủy
          </button>
          <button
            type="button"
            id="btnConfirmNewScenario"
            onClick={handleConfirm}
            disabled={loading}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-teal-600 text-xs font-bold text-white hover:bg-teal-700 transition-all cursor-pointer shadow-xs disabled:opacity-50"
          >
            <Check className="size-4" />
            <span>{loading ? "Đang khởi tạo…" : `Khởi tạo ${selectedDays} ngày`}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

