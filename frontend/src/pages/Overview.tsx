import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CircleMarker, Polyline, Tooltip as LeafletTooltip } from "react-leaflet";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Progress } from "@/components/ui/progress";
import {
  PageHeader,
  LoadingCard,
} from "@/components/shared/DomainComponents";
import { MapBase } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { AtlasMetricStrip, type AtlasMetricItem } from "@/components/atlas/AtlasMetricStrip";
import { api, type AlgorithmSummary } from "@/lib/api";
import {
  ALGO_COLOR,
  ALGO_LABEL,
  ALGO_ORDER,
  DEPOT,
  SERVICE_LABEL,
  clusterColor,
  num,
  pct,
} from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import {
  Package,
  Clock,
  Network,
  CheckCircle2,
  TrendingDown,
  RefreshCw,
  FileBarChart,
  PlayCircle,
  ArrowRight,
  ShieldAlert,
  ChevronRight,
  MapPin,
  Layers,
  Sparkles,
  Info,
} from "lucide-react";

const AXIS_TICK_STYLE = { fontSize: 11 };
const CHART_MARGIN = { top: 10, right: 10, bottom: 6, left: 0 };
const TOOLTIP_CURSOR = { fill: "rgba(0, 0, 0, 0.05)" };

function byAlgo(items: AlgorithmSummary[]) {
  const map = new Map(items.map((i) => [i.algorithm, i]));
  return ALGO_ORDER.map((a) => map.get(a)).filter((x): x is AlgorithmSummary => !!x);
}

function AnimatedRouteOverlay({ routeGeom }: { routeGeom: [number, number][] }) {
  const [drawProgress, setDrawProgress] = useState(0.3);

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
          color: "#176B5B",
          weight: 4,
          opacity: 0.35,
          lineCap: "round",
        }}
        interactive={false}
      />
      <Polyline
        positions={drawnRoute}
        pathOptions={{
          color: "#72C5AC",
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
            fillColor: "#176B5B",
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
  const [refreshKey, setRefreshKey] = useState(0);

  const ov = useAsync((s) => api.overview(s), [refreshKey]);
  const stats = useAsync((s) => api.deliveryStats(s), [refreshKey]);
  const km = useAsync((s) => api.kmeansAll(s), [refreshKey]);
  const val = useAsync((s) => api.finalValidation(s), [refreshKey]);
  const sampleRoute = useAsync(
    (s) => api.precomputedRoute("kmeans_guided_dqn", 65, true, s),
    [refreshKey]
  );

  const s = val.data?.summary;
  const distDelta =
    s?.ratio_of_means_distance_improvement_pct != null
      ? -Number(s.ratio_of_means_distance_improvement_pct)
      : 1.76;
  const timeDelta =
    s?.ratio_of_means_time_improvement_pct != null
      ? -Number(s.ratio_of_means_time_improvement_pct)
      : 1.87;
  const switchReduction =
    s?.relative_switch_rate_reduction_pct_where_available != null
      ? -Number(s.relative_switch_rate_reduction_pct_where_available)
      : -6.83;

  const routeGeom = (sampleRoute.data?.evaluation?.route_geometry ?? []) as [number, number][];

  const samplePoints = useMemo(() => {
    const pts = km.data?.points ?? [];
    if (pts.length <= 160) return pts;
    const step = Math.floor(pts.length / 150);
    return pts.filter((_, i) => i % step === 0);
  }, [km.data?.points]);

  const baseRows = useMemo(
    () => (ov.data ? byAlgo(ov.data.benchmark.algorithms) : []),
    [ov.data]
  );
  const nnDist = useMemo(() => {
    const nnBase = baseRows.find((r) => r.algorithm === "nearest_neighbor");
    return nnBase?.mean_distance_km ?? 62.68;
  }, [baseRows]);

  const chartDiverge = useMemo(() => {
    return baseRows.map((r) => {
      const diffPct = ((nnDist - r.mean_distance_km) / nnDist) * 100;
      return {
        name: ALGO_LABEL[r.algorithm as keyof typeof ALGO_LABEL] ?? r.algorithm,
        key: r.algorithm,
        diffPct: Number(diffPct.toFixed(2)),
        km: r.mean_distance_km,
      };
    });
  }, [baseRows, nnDist]);

  const totalOrders = stats.data?.deliveries ?? km.data?.meta.total_deliveries ?? 2394;
  const missingDeadline = stats.data?.missing_expected_delivery_time ?? 105;
  const withSlaOrders = totalOrders - missingDeadline;
  const uniqueShippers = stats.data?.unique_shippers ?? 16;
  const totalClusters = km.data?.meta.n_clusters ?? 10;

  // Measurement strip metrics (compact 4 items - cost proxy moved to methodology details)
  const measurementMetrics: AtlasMetricItem[] = [
    {
      id: "orders",
      label: "ĐƠN HÀNG XEDU",
      value: num(Number(totalOrders), 0),
      hint: `${withSlaOrders} đơn có SLA (95,6%)`,
      badge: "TP.HCM",
      icon: Package,
    },
    {
      id: "clusters",
      label: "PHÂN CỤM KHÔNG GIAN",
      value: `K = ${totalClusters}`,
      hint: "Silhouette 0,429 • Gom quận",
      badge: "K-Means",
      icon: Network,
    },
    {
      id: "sla",
      label: "TỶ LỆ ĐÚNG HẠN SLA",
      value: "100,0%",
      hint: "108 đơn / 12 WL kiểm chứng",
      badge: "Không vi phạm",
      intent: "positive",
      icon: CheckCircle2,
    },
    {
      id: "switch",
      label: "GIẢM ĐỔI CỤM ĐỊA BÀN",
      value: "-6,83%",
      hint: "Đánh đổi cự ly +1,76%",
      badge: "Guided DQN",
      intent: "positive",
      icon: TrendingDown,
    },
  ];

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      {/* 1. Header with Actions */}
      <PageHeader
        title="Tổng quan hệ thống"
        subtitle="Hệ thống nghiên cứu phân cụm địa lý K-Means & Học tăng cường Guided DQN tối ưu tuyến giao hàng tại TP.HCM."
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRefreshKey((k) => k + 1)}
              className="gap-1.5 text-xs h-8"
            >
              <RefreshCw className="size-3.5" />
              <span>Làm mới</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("/so-sanh")}
              className="gap-1.5 text-xs h-8"
            >
              <FileBarChart className="size-3.5" />
              <span>Đối chuẩn thuật toán</span>
            </Button>
            <Button
              size="sm"
              onClick={() => navigate("/chay-thu")}
              className="gap-1.5 text-xs h-8 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <PlayCircle className="size-3.5" />
              <span>Thử nghiệm tuyến</span>
            </Button>
          </div>
        }
      />

      {/* 2. Atlas High-Density Measurement Strip */}
      <AtlasMetricStrip metrics={measurementMetrics} />

      {/* 3. Main Body: Split 58 / 42 Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (58% ~ 7 cols): Immersive GIS Map Studio */}
        <div className="lg:col-span-7 space-y-4">
          <Card className="overflow-hidden border-border/80 shadow-xs relative">
            <CardHeader className="border-b border-border/60 p-4 sm:px-5 flex flex-row items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <CardTitle className="text-sm font-bold tracking-tight">
                    Bản đồ Giám sát Tuyến Thực địa & Phân bố Không gian
                  </CardTitle>
                  <Badge variant="outline" className="text-[10px] font-mono border-primary/30 text-primary">
                    OpenFreeMap Vector Positron
                  </Badge>
                </div>
                <CardDescription className="text-xs mt-0.5">
                  2.394 đơn XeDu TP.HCM theo 10 cụm địa lý • Lộ trình mô phỏng Guided DQN Workload #65
                </CardDescription>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate("/tuyen-giao")}
                className="gap-1 text-xs text-primary hover:text-primary/90 h-7"
              >
                <span>Mở Map Studio</span>
                <ArrowRight className="size-3" />
              </Button>
            </CardHeader>

            <div className="relative h-[480px] sm:h-[540px] w-full">
              <MapErrorBoundary>
                <MapBase height="100%">
                  {samplePoints.map((p, i) => (
                    <CircleMarker
                      key={i}
                      center={[p.latitude, p.longitude]}
                      radius={4}
                      pathOptions={{
                        color: clusterColor(p.cluster_id),
                        fillColor: clusterColor(p.cluster_id),
                        fillOpacity: 0.85,
                        weight: 1.5,
                      }}
                      eventHandlers={{
                        click: () => navigate(`/phan-cum?cluster=${p.cluster_id}`),
                      }}
                    >
                      <LeafletTooltip direction="top" offset={[0, -4]}>
                        <span className="text-xs font-bold">
                          Cụm #{p.cluster_id} — Bấm để lọc chi tiết
                        </span>
                      </LeafletTooltip>
                    </CircleMarker>
                  ))}

                  <CircleMarker
                    center={[DEPOT.lat, DEPOT.lng]}
                    radius={7}
                    pathOptions={{
                      color: "#ffffff",
                      fillColor: "#171717",
                      fillOpacity: 1,
                      weight: 2.5,
                    }}
                  >
                    <LeafletTooltip direction="top">
                      <span className="text-xs font-bold">Kho xuất phát Depot (Quận 1)</span>
                    </LeafletTooltip>
                  </CircleMarker>

                  <AnimatedRouteOverlay routeGeom={routeGeom} />
                </MapBase>
              </MapErrorBoundary>

              {/* Floating Map HUD */}
              <div className="absolute top-3 right-3 z-[400] max-w-[260px] rounded-lg border border-border/80 bg-card/90 p-3 shadow-md backdrop-blur-md">
                <div className="flex items-center justify-between text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                  <span>Mô phỏng đợt giao mẫu</span>
                  <Badge variant="secondary" className="text-[9px] font-mono px-1 py-0">
                    WL #65
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 rounded bg-muted/30 border border-border/50">
                    <span className="text-[10px] text-muted-foreground block">Cự ly tuyến</span>
                    <span className="font-mono font-bold text-foreground">
                      {num(sampleRoute.data?.evaluation?.total_distance_km ?? 73.34, 2)} km
                    </span>
                  </div>
                  <div className="p-2 rounded bg-muted/30 border border-border/50">
                    <span className="text-[10px] text-muted-foreground block">Thời gian</span>
                    <span className="font-mono font-bold text-foreground">
                      {num(sampleRoute.data?.evaluation?.total_travel_time_minutes ?? 184.2, 0)} ph
                    </span>
                  </div>
                </div>
                <div className="mt-2 text-[10px] text-muted-foreground flex items-center justify-between">
                  <span title="Chỉ số gom cụm đo lường trên tập 12 Validation Workloads">Gom cụm (12 WL): -6,83% đổi vùng</span>
                  <span className="text-primary font-semibold">100% SLA</span>
                </div>
              </div>
            </div>

            {/* Quick Cluster Selector Strip Below Map */}
            <div className="p-3 bg-muted/20 border-t border-border/60 flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar">
                <span className="text-[11px] font-semibold text-muted-foreground whitespace-nowrap">
                  10 Cụm địa lý:
                </span>
                {Array.from({ length: 10 }).map((_, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => navigate(`/phan-cum?cluster=${idx}`)}
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium border border-border/60 bg-card hover:border-primary transition-colors cursor-pointer"
                  >
                    <span
                      className="size-2 rounded-full"
                      style={{ backgroundColor: clusterColor(idx) }}
                    />
                    <span>#{idx}</span>
                  </button>
                ))}
              </div>
              <Button
                variant="link"
                size="sm"
                onClick={() => navigate("/phan-cum")}
                className="text-xs text-primary h-auto p-0 whitespace-nowrap hidden sm:inline"
              >
                Chi tiết cụm →
              </Button>
            </div>
          </Card>
        </div>

        {/* Right Column (42% ~ 5 cols): Divergence Ledger & Leaderboard */}
        <div className="lg:col-span-5 space-y-4">
          {/* Divergence Ledger vs Nearest Neighbor */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 sm:px-5 pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-bold tracking-tight">
                    Độ Lệch Tiết Kiệm Cự Ly so với Nearest Neighbor
                  </CardTitle>
                  <CardDescription className="text-xs mt-0.5">
                    108 Workload cơ sở: Dương (+) = tiết kiệm hơn NN; Âm (−) = dài hơn NN
                  </CardDescription>
                </div>
                <Badge variant="outline" className="font-mono text-[10px]">
                  108 WL
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-4 sm:px-5 pt-0">
              <div className="h-[220px] w-full min-w-0">
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <BarChart data={chartDiverge} margin={CHART_MARGIN}>
                    <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      tick={AXIS_TICK_STYLE}
                      interval={0}
                      stroke="var(--muted-foreground)"
                    />
                    <YAxis
                      tick={AXIS_TICK_STYLE}
                      unit="%"
                      stroke="var(--muted-foreground)"
                      tickFormatter={(v: any) => `${v > 0 ? "+" : ""}${num(v, 0)}%`}
                    />
                    <Tooltip
                      cursor={TOOLTIP_CURSOR}
                      formatter={(v: any, _, item: any) => [
                        `${v > 0 ? "+" : ""}${num(v, 2)}% (${num(item.payload.km, 2)} km)`,
                        "Chênh lệch so với NN",
                      ]}
                    />
                    <Bar dataKey="diffPct" radius={[4, 4, 0, 0]} maxBarSize={36}>
                      {chartDiverge.map((c) => (
                        <Cell
                          key={c.key}
                          fill={ALGO_COLOR[c.key as keyof typeof ALGO_COLOR] ?? "#176B5B"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground border-t border-border/40 pt-2 font-mono">
                <span>Baseline NN: {num(nnDist, 2)} km</span>
                <span className="text-primary font-semibold">Clarke-Wright tiết kiệm nhất</span>
              </div>
            </CardContent>
          </Card>

          {/* Algorithm Benchmark Leaderboard */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 sm:px-5 pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-bold tracking-tight">
                    Bảng Xếp Hạng Hiệu Năng Thuật Toán
                  </CardTitle>
                  <CardDescription className="text-xs mt-0.5">
                    Quãng đường và thời gian bình quân trên 108 đợt giao
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate("/so-sanh")}
                  className="text-xs text-primary h-7 p-0"
                >
                  Bảng đầy đủ →
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-4 sm:px-5 pt-0 space-y-2">
              {baseRows.length === 0 ? (
                <LoadingCard message="Đang nạp bảng xếp hạng..." />
              ) : (
                [...baseRows]
                  .sort((a, b) => a.mean_distance_km - b.mean_distance_km)
                  .map((r, i) => {
                    const maxDist = Math.max(...baseRows.map((x) => x.mean_distance_km));
                    const pctWidth = (r.mean_distance_km / maxDist) * 100;
                    return (
                      <div
                        key={r.algorithm}
                        className="flex flex-col gap-1 p-2 rounded-md border border-border/60 bg-card/70 hover:bg-muted/20 transition-colors"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="size-2 rounded-full"
                              style={{
                                backgroundColor:
                                  ALGO_COLOR[r.algorithm as keyof typeof ALGO_COLOR],
                              }}
                            />
                            <span className="font-semibold text-foreground">
                              {ALGO_LABEL[r.algorithm as keyof typeof ALGO_LABEL] ?? r.algorithm}
                            </span>
                            {i === 0 && (
                              <Badge
                                variant="secondary"
                                className="text-[9px] px-1 py-0 font-bold text-emerald-700 dark:text-emerald-400"
                              >
                                Ngắn nhất
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-3 font-mono text-[11px]">
                            <span className="font-bold text-foreground">
                              {num(r.mean_distance_km, 2)} km
                            </span>
                            <span className="text-muted-foreground">
                              {num(r.mean_travel_time_minutes, 1)} ph
                            </span>
                            <span className="text-emerald-700 dark:text-emerald-400 font-semibold">
                              {pct(r.weighted_on_time_rate, 1)}
                            </span>
                          </div>
                        </div>
                        <Progress value={pctWidth} className="h-1 bg-muted/60" />
                      </div>
                    );
                  })
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 4. Three Technical Research Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-border/80 shadow-xs flex flex-col justify-between">
          <CardHeader className="p-4 sm:p-5 pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-[10px] font-semibold uppercase">
                Đóng góp 1
              </Badge>
              <Badge variant="outline" className="font-mono text-[10px] border-primary/30 text-primary">
                λ = 0,50
              </Badge>
            </div>
            <CardTitle className="text-sm font-bold mt-2">1. Phạt Chuyển Vùng Không Gian</CardTitle>
            <CardDescription className="text-xs">
              Định hướng tác tử ưu tiên giao hết các đơn trong cùng cụm địa lý.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4 sm:p-5 pt-0 space-y-2">
            <div className="rounded-md bg-muted/30 p-2 font-mono text-xs font-semibold text-foreground border border-border/50">
              R&apos; = R − λ_c · 𝕀(c_t ≠ c_t+1)
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Cơ chế reward shaping bổ sung hệ số phạt λ khi shipper di chuyển giữa 2 cụm khác nhau, gom tuyến tự nhiên.
            </p>
          </CardContent>
          <CardFooter className="p-4 sm:p-5 pt-0 border-t border-border/40 mt-auto">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/guided-dqn")}
              className="w-full text-xs gap-1 justify-between px-0 text-primary hover:text-primary/90"
            >
              <span>Thực nghiệm Guided DQN</span>
              <ArrowRight className="size-3.5" />
            </Button>
          </CardFooter>
        </Card>

        <Card className="border-border/80 shadow-xs flex flex-col justify-between">
          <CardHeader className="p-4 sm:p-5 pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-[10px] font-semibold uppercase">
                Đánh đổi
              </Badge>
              <Badge variant="outline" className="font-mono text-[10px] text-emerald-700 dark:text-emerald-400">
                12 WL Kiểm chứng
              </Badge>
            </div>
            <CardTitle className="text-sm font-bold mt-2">2. Đánh Đổi Kỹ Thuật Hợp Lý</CardTitle>
            <CardDescription className="text-xs">
              Giữ nguyên 100% cam kết giao đúng hạn SLA (108 đơn).
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4 sm:p-5 pt-0 space-y-2">
            <div className="grid grid-cols-3 gap-1.5 text-center">
              <div className="p-2 rounded bg-muted/30 border border-border/40">
                <div className="text-[10px] text-muted-foreground font-semibold">ĐỔI CỤM</div>
                <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 font-mono">-6,83%</div>
              </div>
              <div className="p-2 rounded bg-muted/30 border border-border/40">
                <div className="text-[10px] text-muted-foreground font-semibold">CỰ LY</div>
                <div className="text-xs font-bold text-amber-700 dark:text-amber-400 font-mono">+1,76%</div>
              </div>
              <div className="p-2 rounded bg-muted/30 border border-border/40">
                <div className="text-[10px] text-muted-foreground font-semibold">SLA</div>
                <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 font-mono">100%</div>
              </div>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Mức tăng cự ly tối thiểu (+1,27 km trên tuyến 72,07 km) đổi lấy lợi ích gom cụm chặt chẽ và không trễ hạn.
            </p>
          </CardContent>
          <CardFooter className="p-4 sm:p-5 pt-0 border-t border-border/40 mt-auto">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/do-nhay")}
              className="w-full text-xs gap-1 justify-between px-0 text-primary hover:text-primary/90"
            >
              <span>Phân tích độ nhạy λ</span>
              <ArrowRight className="size-3.5" />
            </Button>
          </CardFooter>
        </Card>

        <Card className="border-border/80 shadow-xs flex flex-col justify-between">
          <CardHeader className="p-4 sm:p-5 pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-[10px] font-semibold uppercase">
                Đóng góp 2
              </Badge>
              <Badge variant="outline" className="font-mono text-[10px] text-blue-700 dark:text-blue-400">
                16 WL Held-out
              </Badge>
            </div>
            <CardTitle className="text-sm font-bold mt-2">3. Proposed RL Đa Mục Tiêu</CardTitle>
            <CardDescription className="text-xs">
              Mô hình đánh giá toàn diện trên tập dữ liệu kiểm thử độc lập.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4 sm:p-5 pt-0 space-y-2">
            <div className="rounded-md bg-muted/30 p-2 font-mono text-[11px] font-semibold text-foreground border border-border/50 truncate">
              J(π) = 0,375D̄ + 0,375T̄ + 0,1M̄ + 0,1L̄
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Đạt <b>4,00 lần đổi cụm</b> (thấp nhất trong các thuật toán) trên 16 Workload kiểm thử độc lập (131 đơn).
            </p>
          </CardContent>
          <CardFooter className="p-4 sm:p-5 pt-0 border-t border-border/40 mt-auto">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/so-sanh")}
              className="w-full text-xs gap-1 justify-between px-0 text-primary hover:text-primary/90"
            >
              <span>Đối chuẩn đa thuật toán</span>
              <ArrowRight className="size-3.5" />
            </Button>
          </CardFooter>
        </Card>
      </div>

      {/* 5. Collapsible Methodology Accordion */}
      <Card className="border-border/80 shadow-xs">
        <Accordion className="w-full">
          <AccordionItem value="methodology" className="border-0">
            <AccordionTrigger className="px-5 py-3.5 text-xs font-semibold hover:no-underline">
              <div className="flex items-center gap-2 text-muted-foreground">
                <ShieldAlert className="size-4 text-primary" />
                <span className="text-foreground">
                  Chi tiết Phương pháp luận & Tuyên bố về Điều kiện Thực nghiệm Khoa học
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-5 pb-4 pt-0 text-xs text-muted-foreground space-y-2 border-t border-border/40 mt-2">
              <p>
                • <b>Mô hình kịch bản chi phí mô phỏng:</b> Chi phí vận hành trong hệ thống được tính theo kịch bản mô phỏng giả định: <code>Khoảng cách × 4.000 đ/km + Thời gian × 666,67 đ/phút</code> (tương đương 40.000 đ/giờ), không phải chi phí thanh toán thực tế của doanh nghiệp.
              </p>
              <p>
                • <b>Độc lập giữa 4 tập dữ liệu:</b> 108 Workload (Benchmark cơ sở), 12 Workload (Kiểm chứng Guided DQN), 16 Workload (Held-out Test cho Proposed RL), và 1 Workload (Kiểm chứng chính xác 120 hoán vị). Tuyệt đối không đánh đồng hoặc xếp hạng chéo ngoài phạm vi thực nghiệm.
              </p>
              <p>
                • <b>Môi trường thực nghiệm GIS:</b> Bản đồ OpenFreeMap Vector Positron kết hợp đồ thị OpenStreetMap TP.HCM với ma trận cự ly và thời gian thực tế.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </Card>
    </div>
  );
}
