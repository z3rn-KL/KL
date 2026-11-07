import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CircleMarker, Polyline, Tooltip as LeafletTooltip } from "react-leaflet";
import { Button } from "@/components/ui/button";
import { MapBase } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { api } from "@/lib/api";
import { DEPOT, clusterColor } from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import {
  Network,
  CheckCircle2,
  ArrowRight,
  MapPin,
  Sparkles,
  BrainCircuit,
  Layers,
} from "lucide-react";

function AnimatedRouteOverlay({ routeGeom }: { routeGeom: [number, number][] }) {
  const [drawProgress, setDrawProgress] = useState(0.4);

  useEffect(() => {
    if (!routeGeom.length) return;
    let rafId = 0;
    const start = performance.now();
    const duration = 12000;
    const tick = (now: number) => {
      const elapsed = (now - start) % duration;
      setDrawProgress(Math.max(0.1, elapsed / duration));
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [routeGeom.length]);

  const drawnRoute = useMemo(() => {
    if (routeGeom.length < 2) return [];
    const count = Math.max(2, Math.floor(drawProgress * routeGeom.length));
    return routeGeom.slice(0, count);
  }, [routeGeom, drawProgress]);

  const carHead = drawnRoute.length ? drawnRoute[drawnRoute.length - 1] : null;

  if (drawnRoute.length <= 1) return null;

  return (
    <>
      <Polyline
        positions={drawnRoute}
        pathOptions={{
          color: "#0f766e",
          weight: 4,
          opacity: 0.4,
          lineCap: "round",
        }}
        interactive={false}
      />
      <Polyline
        positions={drawnRoute}
        pathOptions={{
          color: "#14b8a6",
          weight: 2.5,
          opacity: 0.95,
          dashArray: "6 6",
          lineCap: "round",
        }}
        interactive={false}
      />
      {carHead && (
        <CircleMarker
          center={carHead}
          radius={6}
          pathOptions={{
            color: "#ffffff",
            fillColor: "#0f766e",
            fillOpacity: 1,
            weight: 2,
          }}
          interactive={false}
        />
      )}
    </>
  );
}

export default function Overview() {
  const navigate = useNavigate();
  const [refreshKey] = useState(0);

  const stats = useAsync((s) => api.deliveryStats(s), [refreshKey]);
  const km = useAsync((s) => api.kmeansAll(s), [refreshKey]);
  const sampleRoute = useAsync(
    (s) => api.precomputedRoute("kmeans_guided_dqn", 65, true, s),
    [refreshKey]
  );

  const routeGeom = (sampleRoute.data?.evaluation?.route_geometry ?? []) as [number, number][];

  // Render all 2,394 delivery points on GIS canvas (colored by K-Means cluster)
  const points = km.data?.points ?? [];

  const totalOrders = stats.data?.deliveries ?? km.data?.meta.total_deliveries ?? 2394;

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-3">
      {/* ============================================================== */}
      {/* ROW 1 — RESEARCH HERO (Spans full workspace width)              */}
      {/* ============================================================== */}
      <div className="rounded-2xl border border-teal-500/30 bg-card p-3.5 sm:px-5 sm:py-4 shadow-xs space-y-1.5 shrink-0">
        <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground leading-snug">
          Xây dựng hệ thống tối ưu tuyến giao hàng kết hợp kỹ thuật phân cụm và Reinforcement Learning trong Logistics
        </h1>

        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed max-w-4xl">
          Hệ thống kết hợp <strong className="text-foreground font-semibold">K-Means</strong> để phân vùng không gian đơn hàng và <strong className="text-foreground font-semibold">Candidate-Scoring DQN</strong> để định tuyến đa mục tiêu theo thời gian thực. Nền tảng hỗ trợ mô phỏng di chuyển thực địa và đối chuẩn khoa học với các giải thuật kinh điển.
        </p>
      </div>

      {/* ============================================================== */}
      {/* ROW 2 — MAIN WORKSPACE (Grid: Map 60-65% left, Sidebar 35-40%)  */}
      {/* ============================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 flex-1 min-h-0 items-stretch">
        
        {/* LEFT COLUMN: LEAFLET GIS MAP CANVAS (~60-65%) */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col rounded-2xl border border-border/80 bg-card overflow-hidden shadow-xs relative min-h-[440px] lg:min-h-0 h-full">
          {/* Map Header Overlay */}
          <div className="px-4 py-2.5 border-b border-border/60 bg-muted/30 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <MapPin className="size-4 text-teal-600 dark:text-teal-400 shrink-0" />
              <span className="text-xs sm:text-sm font-bold text-foreground">
                Bản đồ phân cụm đơn hàng
              </span>
            </div>
            <span className="text-xs font-mono font-semibold text-muted-foreground shrink-0 hidden sm:inline">
              {points.length > 0 ? `${points.length.toLocaleString("vi-VN")} điểm giao hàng · 10 cụm K-Means` : "2.394 điểm giao hàng · 10 cụm K-Means"}
            </span>
          </div>

          {/* Leaflet Map */}
          <div className="flex-1 w-full relative min-h-0">
            <MapErrorBoundary>
              <MapBase height="100%">
                {/* Animated Route Line */}
                {routeGeom.length > 0 && <AnimatedRouteOverlay routeGeom={routeGeom} />}

                {/* All delivery points colored by K-Means cluster (Canvas rendered) */}
                {points.map((pt, i) => (
                  <CircleMarker
                    key={`pt-${pt.delivery_id || i}`}
                    center={[pt.latitude, pt.longitude]}
                    radius={3.5}
                    pathOptions={{
                      color: "#ffffff",
                      weight: 0.8,
                      fillColor: clusterColor(pt.cluster_id ?? 0),
                      fillOpacity: 0.85,
                    }}
                  >
                    <LeafletTooltip direction="top" offset={[0, -4]} opacity={0.95}>
                      <div className="text-xs font-sans">
                        <div className="font-bold">Đơn hàng: {pt.delivery_id}</div>
                        <div>Cụm K-Means: Cụm #{pt.cluster_id}</div>
                      </div>
                    </LeafletTooltip>
                  </CircleMarker>
                ))}

                {/* Depot Marker */}
                <CircleMarker
                  center={[DEPOT.lat, DEPOT.lng]}
                  radius={8}
                  pathOptions={{
                    color: "#ffffff",
                    weight: 2.5,
                    fillColor: "#0f766e",
                    fillOpacity: 1,
                  }}
                >
                  <LeafletTooltip direction="top" offset={[0, -8]}>
                    <span className="text-xs font-bold font-mono">Kho trung tâm (Depot Tân Bình)</span>
                  </LeafletTooltip>
                </CircleMarker>
              </MapBase>
            </MapErrorBoundary>

            {/* Subtle Loading Badge */}
            {km.loading && (
              <div className="absolute top-3 left-3 z-[1000] bg-card/90 backdrop-blur-md px-2.5 py-1 rounded-md border border-border/80 shadow-xs text-[11px] font-medium text-teal-700 dark:text-teal-300 flex items-center gap-1.5 animate-pulse">
                <span className="size-2 rounded-full bg-teal-500 animate-ping" />
                <span>Đang tải 2.394 điểm giao hàng TP.HCM…</span>
              </div>
            )}

            {/* Floating Glass Legend Overlay */}
            <div className="absolute bottom-3 left-3 z-[1000] bg-card/90 backdrop-blur-md px-3 py-2 rounded-xl border border-border/80 shadow-md text-xs space-y-1">
              <div className="font-bold text-[10px] uppercase tracking-wider text-muted-foreground">Chú giải bản đồ</div>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="size-2.5 rounded-full bg-teal-600 border border-white" />
                <span>Kho trung tâm (Depot)</span>
              </div>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="size-2 rounded-full bg-amber-500" />
                <span>Đơn hàng theo 10 cụm địa lý</span>
              </div>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="w-3.5 h-0.5 bg-teal-500 inline-block" />
                <span>Tuyến di chuyển minh họa</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: SIDEBAR WITH PIPELINE, DATASETS, CTAs (~35-40%) */}
        {/* Grouped cohesively into 4 distinct tiers with consistent vertical spacing and card architecture */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col justify-start gap-3">
          {/* Tầng 1: Quy trình điều phối (5 Steps) */}
          <div className="rounded-2xl border border-border/80 bg-card p-3.5 sm:p-4 shadow-2xs space-y-2.5 shrink-0">
            <div className="flex items-center justify-between text-xs font-bold text-foreground">
              <span className="flex items-center gap-1.5">
                <Sparkles className="size-3.5 text-teal-600 dark:text-teal-400" />
                <span>Quy trình điều phối</span>
              </span>
              <span className="text-[10px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-full border border-teal-500/20">
                5 BƯỚC
              </span>
            </div>

            <div className="grid grid-cols-5 gap-1.5 sm:gap-2 text-center">
              <div className="p-2 rounded-xl border border-teal-500/50 bg-teal-500/15 text-teal-700 dark:text-teal-300 shadow-2xs select-none">
                <span className="font-mono font-bold block text-xs">01</span>
                <span className="text-[11px] sm:text-xs font-bold truncate block mt-0.5">Tiếp nhận</span>
                <span className="text-[9px] font-medium text-teal-700/80 dark:text-teal-300/80 hidden sm:block">Hiện tại</span>
              </div>
              <div onClick={() => navigate("/phan-cum")} className="p-2 rounded-xl border border-border/70 bg-muted/30 hover:border-teal-500/40 hover:bg-teal-500/5 cursor-pointer transition-colors" title="Chuyển sang Bước 2: Dữ liệu & Phân cụm">
                <span className="font-mono font-bold text-muted-foreground block text-xs">02</span>
                <span className="text-[11px] sm:text-xs font-semibold text-foreground truncate block mt-0.5">Phân cụm</span>
                <span className="text-[9px] text-muted-foreground hidden sm:block">Tiếp theo</span>
              </div>
              <div onClick={() => navigate("/toi-uu-rl")} className="p-2 rounded-xl border border-border/70 bg-muted/30 hover:border-teal-500/40 hover:bg-teal-500/5 cursor-pointer transition-colors" title="Chuyển sang Bước 3: Phân bổ">
                <span className="font-mono font-bold text-muted-foreground block text-xs">03</span>
                <span className="text-[11px] sm:text-xs font-semibold text-foreground truncate block mt-0.5">Phân bổ</span>
                <span className="text-[9px] text-muted-foreground hidden sm:block">Tiếp theo</span>
              </div>
              <div onClick={() => navigate("/toi-uu-rl")} className="p-2 rounded-xl border border-border/70 bg-muted/30 hover:border-teal-500/40 hover:bg-teal-500/5 cursor-pointer transition-colors" title="Chuyển sang Bước 4: Định tuyến RL">
                <span className="font-mono font-bold text-muted-foreground block text-xs">04</span>
                <span className="text-[11px] sm:text-xs font-semibold text-foreground truncate block mt-0.5">Định tuyến</span>
                <span className="text-[9px] text-muted-foreground hidden sm:block">Tiếp theo</span>
              </div>
              <div onClick={() => navigate("/mo-phong")} className="p-2 rounded-xl border border-border/70 bg-muted/30 hover:border-teal-500/40 hover:bg-teal-500/5 cursor-pointer transition-colors" title="Chuyển sang Bước 5: Mô phỏng">
                <span className="font-mono font-bold text-muted-foreground block text-xs">05</span>
                <span className="text-[11px] sm:text-xs font-semibold text-foreground truncate block mt-0.5">Mô phỏng</span>
                <span className="text-[9px] text-muted-foreground hidden sm:block">Tiếp theo</span>
              </div>
            </div>
          </div>

          {/* Tầng 2: Dataset Card */}
          <div className="rounded-2xl border border-border/80 bg-card p-3.5 sm:p-4 shadow-2xs space-y-1.5 shrink-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold uppercase text-teal-700 dark:text-teal-300 tracking-wider">TẬP XEDU TP.HCM</span>
              <span className="size-2 rounded-full bg-teal-500" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black font-mono text-foreground tracking-tight">{totalOrders.toLocaleString("vi-VN")}</span>
              <span className="text-xs text-muted-foreground font-semibold">đơn hàng · 10 cụm địa lý</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed m-0">
              Phân hoạch 10 cụm địa lý (Silhouette 0,429), giảm không gian tìm kiếm định tuyến.
            </p>
          </div>

          {/* Tầng 3: Kịch bản mô phỏng Card */}
          <div className="rounded-2xl border border-border/80 bg-card p-3.5 sm:p-4 shadow-2xs space-y-1.5 shrink-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold uppercase text-sky-700 dark:text-sky-300 tracking-wider">KỊCH BẢN MÔ PHỎNG</span>
              <span className="size-2 rounded-full bg-sky-500" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black font-mono text-foreground tracking-tight">85</span>
              <span className="text-xs text-muted-foreground font-semibold">đơn mới/ngày · 16 shipper</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed m-0">
              Mô phỏng 3 ngày ca trực, cơ chế chuyển tiếp đơn chưa hoàn thành sang ngày sau.
            </p>
          </div>

          {/* Tầng 4: Navigation Actions (Bố cục 1:1, đồng chiều cao h-10) */}
          <div className="grid grid-cols-2 gap-3 shrink-0">
            <Button
              onClick={() => navigate("/phan-cum")}
              variant="outline"
              size="sm"
              className="w-full text-xs sm:text-sm h-10 font-semibold rounded-xl border-border/80 gap-2 cursor-pointer shadow-2xs hover:bg-muted"
            >
              <Network className="size-4 text-teal-600 dark:text-teal-400 shrink-0" />
              <span className="truncate">Dữ liệu và phân cụm</span>
            </Button>
            <Button
              onClick={() => navigate("/toi-uu-rl")}
              size="sm"
              className="w-full text-xs sm:text-sm h-10 bg-teal-600 text-white hover:bg-teal-700 font-semibold rounded-xl shadow-xs gap-2 cursor-pointer"
            >
              <BrainCircuit className="size-4 shrink-0" />
              <span className="truncate">Bàn lập tuyến</span>
              <ArrowRight className="size-4 shrink-0" />
            </Button>
          </div>
        </div>
      </div>

      {/* ============================================================== */}
      {/* ROW 3 — FOOTER / SCIENTIFIC SCOPE RIBBON                        */}
      {/* ============================================================== */}
      <div className="rounded-xl border border-border/80 bg-card/70 backdrop-blur-xs p-2 sm:px-3.5 flex flex-wrap items-center justify-between gap-2 text-xs shrink-0 shadow-2xs">
        <div className="flex flex-wrap items-center gap-2 text-muted-foreground">
          <span className="font-bold text-foreground flex items-center gap-1.5">
            <Layers className="size-3.5 text-teal-600 dark:text-teal-400" />
            <span>Phạm vi đối chuẩn:</span>
          </span>
          <span className="font-mono text-xs bg-muted px-2 py-0.5 rounded border border-border/60 text-foreground">
            Mạng lưới TP.HCM (WGS84) · Kho Tân Bình &amp; 6 trạm trực vệ tinh
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-foreground bg-muted/60 px-2 py-0.5 rounded border border-border/60 flex items-center gap-1.5" title="PyTorch Checkpoint: D_moderate_sla_best.pt">
            <CheckCircle2 className="size-3.5 text-emerald-600" />
            <span className="font-semibold">Mô hình: Candidate DQN</span>
          </span>
        </div>
      </div>
    </div>
  );
}
