import React from "react";
import { useNavigate } from "react-router-dom";
import {
  FileText,
  Calendar,
  Truck,
  CheckCircle2,
  Clock,
  AlertTriangle,
  ArrowRight,
  RotateCcw,
  ShieldCheck,
  Package,
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
import { DayReport } from "@/components/dispatch/DayReport";
import { useDispatchContext } from "@/context/DispatchContext";

export default function ShiftReport() {
  const navigate = useNavigate();
  const {
    session,
    loading,
    connectionError,
    sessionNotFoundError,
    actionLoading,
    actionError,
    actionSuccess,
    handleNextDay,
    initSession,
  } = useDispatchContext();

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3">
        <div className="size-8 rounded-full border-2 border-teal-600 border-t-transparent animate-spin" />
        <span className="text-xs text-muted-foreground font-mono">Đang tổng hợp báo cáo kết ca và chỉ số SLA…</span>
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
    <div className="space-y-4">
      {/* ============================================================== */}
      {/* 1. SCREEN HEADER                                               */}
      {/* ============================================================== */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-border/80 pb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground">
              Báo cáo ca
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Đối soát số liệu theo ngày • Phân định rành mạch chỉ số trong ca và lũy kế • Quản lý đơn tồn chuyển ngày
          </p>
        </div>

        {/* Navigation Buttons */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/mo-phong")}
            className="text-xs h-8 cursor-pointer font-medium"
          >
            Quay lại mô phỏng
          </Button>
          <Button
            size="sm"
            onClick={() => navigate("/toi-uu-rl")}
            className="gap-1.5 text-xs h-8 bg-teal-600 text-white hover:bg-teal-700 font-semibold shadow-xs cursor-pointer"
          >
            <span>Lập tuyến ngày mới</span>
            <ArrowRight className="size-3.5" />
          </Button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. CORE DAY REPORT COMPONENT                                   */}
      {/* ============================================================== */}
      <DayReport
        session={session}
        onNextDay={handleNextDay}
        onSwitchToPlanning={() => navigate("/toi-uu-rl")}
        onSwitchToLive={() => navigate("/mo-phong")}
        actionLoading={actionLoading}
        actionError={actionError}
        actionSuccess={actionSuccess}
      />
    </div>
  );
}

