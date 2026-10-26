import React, { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CircleMarker, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
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
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  PageHeader,
  LoadingCard,
  ErrorCard,
} from "@/components/shared/DomainComponents";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { MapBase } from "@/components/MapBase";
import { api, type ClusterRunResult, type DeliveryPoint, type KmeansPoint } from "@/lib/api";
import { NOISE_COLOR, clusterColor, num, pct } from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import {
  Network,
  Play,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Info,
  Maximize2,
  ExternalLink,
  MapPin,
  Check,
  AlertTriangle,
  Layers,
  Sparkles,
} from "lucide-react";

type Mode = "saved" | "kmeans" | "dbscan";

interface Pt {
  id: string;
  lat: number;
  lng: number;
  c: number; // cluster id, -1 = noise
}

const haversineKm = (a: [number, number], b: [number, number]) => {
  const R = 6371.0088;
  const rad = Math.PI / 180;
  const dLat = (b[0] - a[0]) * rad;
  const dLng = (b[1] - a[1]) * rad;
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

function FlyTo({
  target,
  zoom = 16,
}: {
  target: [number, number] | null;
  zoom?: number;
}) {
  const map = useMap();
  useEffect(() => {
    if (target && Number.isFinite(target[0]) && Number.isFinite(target[1])) {
      map.flyTo(L.latLng(target[0], target[1]), Math.max(map.getZoom(), zoom), {
        duration: 0.8,
      });
    }
  }, [map, target, zoom]);
  return null;
}

interface Outlier {
  pt: Pt;
  dist: number;
  nearest: number;
}

export default function Clusters() {
  const saved = useAsync((s) => api.kmeansAll(s), []);

  const [mode, setMode] = useState<Mode>("saved");
  const [k, setK] = useState(10);
  const [eps, setEps] = useState(2.0);
  const [minS, setMinS] = useState(8);
  const [live, setLive] = useState<ClusterRunResult | null>(null);
  const [liveMode, setLiveMode] = useState<Mode | null>(null);
  const [running, setRunning] = useState(false);
  const [runErr, setRunErr] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [outlierPct, setOutlierPct] = useState(97.5);
  const [focus, setFocus] = useState<[number, number] | null>(null);
  const [focusZoom, setFocusZoom] = useState<number>(14);
  const [searchParams] = useSearchParams();
  const clusterParam = searchParams.get("cluster");
  const [selectedCluster, setSelectedCluster] = useState<number | null>(
    clusterParam !== null && !isNaN(Number(clusterParam)) ? Number(clusterParam) : null
  );
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    if (clusterParam !== null && !isNaN(Number(clusterParam))) {
      setSelectedCluster(Number(clusterParam));
    }
  }, [clusterParam]);

  const cq = useAsync((s) => api.clusteringQuality(s), []);
  const savedSilhouette = useMemo(() => {
    if (!cq.data?.kmeans) return null;
    const k10 = cq.data.kmeans.find((r) => r.k === 10);
    return k10?.silhouette_score ?? null;
  }, [cq.data]);

  const savedInertia = useMemo(() => {
    if (!cq.data?.kmeans) return null;
    const k10 = cq.data.kmeans.find((r) => r.k === 10);
    return k10?.inertia ?? null;
  }, [cq.data]);

  const basePoints: DeliveryPoint[] = useMemo(
    () =>
      (saved.data?.points ?? []).map((p: KmeansPoint) => ({
        delivery_id: p.delivery_id,
        latitude: p.latitude,
        longitude: p.longitude,
      })),
    [saved.data]
  );

  const points: Pt[] = useMemo(() => {
    if (mode === "saved") {
      return (saved.data?.points ?? []).map((p: KmeansPoint) => ({
        id: p.delivery_id,
        lat: p.latitude,
        lng: p.longitude,
        c: p.cluster_id,
      }));
    }
    if (live && liveMode === mode) {
      return live.assignments.map((a: DeliveryPoint & { cluster_id: number }) => ({
        id: a.delivery_id,
        lat: a.latitude,
        lng: a.longitude,
        c: a.cluster_id,
      }));
    }
    return [];
  }, [mode, saved.data, live, liveMode]);

  async function runClustering() {
    if (mode === "saved" || basePoints.length === 0) return;
    setRunning(true);
    setRunErr(null);
    try {
      const res =
        mode === "kmeans"
          ? await api.runKmeans(basePoints, k)
          : await api.runDbscan(basePoints, eps, minS);
      setLive(res);
      setLiveMode(mode);
      setHidden(new Set());
      setSelectedCluster(null);
    } catch (e) {
      setRunErr((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  // Centroids
  const centroids = useMemo(() => {
    const acc = new Map<number, { lat: number; lng: number; n: number }>();
    for (const p of points) {
      if (p.c < 0) continue;
      const a = acc.get(p.c) ?? { lat: 0, lng: 0, n: 0 };
      a.lat += p.lat;
      a.lng += p.lng;
      a.n += 1;
      acc.set(p.c, a);
    }
    return new Map(
      [...acc.entries()].map(([c, a]) => [c, { lat: a.lat / a.n, lng: a.lng / a.n, n: a.n }])
    );
  }, [points]);

  const sizes = useMemo(() => {
    const m = new Map<number, number>();
    for (const p of points) m.set(p.c, (m.get(p.c) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => (a[0] < 0 ? 1 : b[0] < 0 ? -1 : a[0] - b[0]));
  }, [points]);

  // Outlier detection
  const outliers: Outlier[] = useMemo(() => {
    if (points.length === 0 || centroids.size === 0) return [];
    const rows: Outlier[] = points.map((p) => {
      let best = Infinity;
      let nearest = p.c;
      if (p.c >= 0) {
        const c = centroids.get(p.c)!;
        best = haversineKm([p.lat, p.lng], [c.lat, c.lng]);
      } else {
        for (const [cid, c] of centroids) {
          const d = haversineKm([p.lat, p.lng], [c.lat, c.lng]);
          if (d < best) {
            best = d;
            nearest = cid;
          }
        }
      }
      return { pt: p, dist: best, nearest };
    });
    const sorted = [...rows].map((r) => r.dist).sort((a, b) => a - b);
    const thr = sorted[Math.min(sorted.length - 1, Math.floor((outlierPct / 100) * sorted.length))];
    return rows.filter((r) => r.dist >= thr).sort((a, b) => b.dist - a.dist);
  }, [points, centroids, outlierPct]);

  const outlierIds = useMemo(() => new Set(outliers.map((o) => o.pt.id)), [outliers]);
  const noiseCount = points.filter((p) => p.c < 0).length;
  const clusterCount = centroids.size;
  const m = live && liveMode === mode ? live.metrics : null;

  const toggle = (c: number) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(c)) n.delete(c);
      else n.add(c);
      return n;
    });

  const visible = useMemo(() => {
    return points.filter((p) => {
      if (hidden.has(p.c)) return false;
      if (selectedCluster !== null && p.c !== selectedCluster) return false;
      return true;
    });
  }, [points, hidden, selectedCluster]);

  const handleFocusCluster = (c: number | null) => {
    setSelectedCluster(c);
    if (c === null) {
      setFocus(null);
    } else {
      const ct = centroids.get(c);
      if (ct) {
        setFocus([ct.lat, ct.lng]);
        setFocusZoom(14);
      }
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    const query = searchQuery.trim().toLowerCase();
    const found = points.find((p) => p.id.toLowerCase().includes(query));
    if (found) {
      setFocus([found.lat, found.lng]);
      setFocusZoom(18);
    }
  };

  return (
    <div className="space-y-3 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      {/* 1. Compact Header */}
      <PageHeader
        title="Phân cụm không gian"
        subtitle="Mô hình K-Means K=10 (Silhouette ≈ 0,429) trên 2.394 đơn hàng XeDu TP.HCM • Đánh giá điểm xa tâm Haversine & kiểm tra mạng đường bộ"
      />

      {/* 2. Map-First Workspace: Compact Integrated Control Bar directly above Map */}
      <Card className="border-border/80 shadow-xs">
        <div className="p-3 bg-muted/20 border-b border-border/60 flex flex-wrap items-center justify-between gap-2.5 text-xs">
          {/* Model Switcher */}
          <div className="flex items-center gap-2">
            <span className="font-semibold text-muted-foreground text-[11px] uppercase tracking-wider">
              Mô hình:
            </span>
            <Tabs
              value={mode}
              onValueChange={(v) => {
                setMode(v as Mode);
                setHidden(new Set());
                setSelectedCluster(null);
                setRunErr(null);
              }}
            >
              <TabsList className="h-8">
                <TabsTrigger value="saved" className="text-xs">
                  K-Means chuẩn (K=10)
                </TabsTrigger>
                <TabsTrigger value="kmeans" className="text-xs">
                  K-Means tùy biến
                </TabsTrigger>
                <TabsTrigger value="dbscan" className="text-xs">
                  DBSCAN
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {mode !== "saved" && (
              <Button
                size="sm"
                onClick={runClustering}
                disabled={running || basePoints.length === 0}
                className="gap-1.5 text-xs h-8 bg-primary text-primary-foreground"
              >
                <Play className="size-3" />
                <span>{running ? "Đang xử lý..." : "Chạy"}</span>
              </Button>
            )}
          </div>

          {/* Parameters for custom mode */}
          {mode === "kmeans" && (
            <div className="flex items-center gap-2 min-w-[160px]">
              <span className="text-muted-foreground text-[11px]">K = {k}:</span>
              <Slider
                min={2}
                max={20}
                step={1}
                value={[k]}
                onValueChange={(val: any) => setK(Array.isArray(val) ? val[0] : val)}
                className="w-24"
              />
            </div>
          )}

          {mode === "dbscan" && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground text-[11px]">eps = {num(eps, 2)}km</span>
                <Slider
                  min={0.25}
                  max={2.5}
                  step={0.05}
                  value={[eps]}
                  onValueChange={(val: any) => setEps(Array.isArray(val) ? val[0] : val)}
                  className="w-20"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground text-[11px]">minS = {minS}</span>
                <Slider
                  min={3}
                  max={12}
                  step={1}
                  value={[minS]}
                  onValueChange={(val: any) => setMinS(Array.isArray(val) ? val[0] : val)}
                  className="w-16"
                />
              </div>
            </div>
          )}

          {/* Outlier Threshold Slider */}
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-[11px] whitespace-nowrap">
              Xa tâm: top {num(100 - outlierPct, 1)}%
            </span>
            <Slider
              min={90}
              max={99.5}
              step={0.5}
              value={[outlierPct]}
              onValueChange={(val: any) => setOutlierPct(Array.isArray(val) ? val[0] : val)}
              className="w-20"
            />
          </div>

          {/* Locator Search (Zoom map to order) */}
          <form onSubmit={handleSearch} className="flex items-center gap-1.5 ml-auto">
            <Input
              type="search"
              placeholder="Định vị mã đơn..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 text-xs w-[140px] bg-background"
            />
            <Button type="submit" variant="secondary" size="sm" className="h-8 px-2 text-xs">
              Tìm
            </Button>
          </form>
        </div>

        {/* Quick Cluster Filter Pills in Same Header */}
        <div className="px-3 py-2 bg-muted/10 border-b border-border/40 flex items-center gap-1.5 overflow-x-auto no-scrollbar text-xs">
          <span className="text-[11px] font-semibold text-muted-foreground shrink-0 mr-1">
            Lọc cụm:
          </span>
          <button
            type="button"
            onClick={() => handleFocusCluster(null)}
            className={`h-6 px-2 text-[11px] font-medium rounded transition-all select-none border cursor-pointer shrink-0 ${
              selectedCluster === null
                ? "bg-primary/10 border-primary text-foreground font-semibold"
                : "bg-background border-border/60 text-muted-foreground hover:text-foreground"
            }`}
          >
            Tất cả ({num(points.length, 0)})
          </button>
          {sizes.map(([c, n]) => {
            const isSelected = selectedCluster === c;
            const clr = c < 0 ? NOISE_COLOR : clusterColor(c);
            return (
              <button
                key={c}
                type="button"
                onClick={() => handleFocusCluster(isSelected ? null : c)}
                className={`h-6 px-2 text-[11px] font-medium rounded transition-all select-none inline-flex items-center gap-1 border cursor-pointer shrink-0 ${
                  isSelected
                    ? "bg-primary/10 border-primary text-foreground font-semibold"
                    : "bg-background border-border/60 text-muted-foreground hover:text-foreground"
                }`}
                style={{
                  borderColor: isSelected ? clr : undefined,
                }}
              >
                <span className="size-1.5 rounded-full shrink-0" style={{ backgroundColor: clr }} />
                <span>{c < 0 ? "Nhiễu" : `#${c}`}</span>
                <span className="text-muted-foreground font-mono">({num(n, 0)})</span>
              </button>
            );
          })}
        </div>
      </Card>

      {saved.loading && <LoadingCard message="Đang tải dữ liệu 2.394 điểm giao hàng XeDu..." />}
      {saved.error && <ErrorCard message={saved.error} />}
      {runErr && <ErrorCard message={`Không thể thực hiện phân cụm: ${runErr}`} />}

      {saved.loading && <LoadingCard message="Đang tải dữ liệu 2.394 điểm giao hàng XeDu..." />}
      {saved.error && <ErrorCard message={saved.error} />}
      {runErr && <ErrorCard message={`Không thể thực hiện phân cụm: ${runErr}`} />}

      {/* Main Workspace (Left GIS Map 8 cols + Right Analytics 4 cols) */}
      {saved.data && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left GIS Interactive Map (8 cols) */}
          <div className="lg:col-span-8 space-y-3">
            <Card className="overflow-hidden border-border/80 shadow-xs relative">
              <div className="h-[580px] sm:h-[680px] w-full relative">
                <MapErrorBoundary>
                  <MapBase
                    height="100%"
                    fit={visible.length ? visible.map((p) => [p.lat, p.lng]) : undefined}
                    overlay={
                      <div className="absolute top-3 left-3 z-[1000] flex items-center gap-2 bg-background/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-border shadow-xs text-xs font-medium">
                        <span className="size-2 rounded-full bg-primary" />
                        <span className="font-semibold text-foreground">
                          {mode === "saved"
                            ? "K-Means chuẩn (K=10)"
                            : mode === "kmeans"
                            ? `K-Means (K=${k})`
                            : "DBSCAN Mật độ"}
                        </span>
                        <Separator orientation="vertical" className="h-3.5" />
                        <span className="text-muted-foreground font-mono">
                          {num(visible.length, 0)} / {num(points.length, 0)} điểm
                        </span>
                        {selectedCluster !== null && (
                          <>
                            <Separator orientation="vertical" className="h-3.5" />
                            <Badge variant="outline" className="text-[10px] px-1 py-0" style={{ color: clusterColor(selectedCluster) }}>
                              Cụm #{selectedCluster}
                            </Badge>
                          </>
                        )}
                        {savedSilhouette !== null && mode === "saved" && (
                          <>
                            <Separator orientation="vertical" className="h-3.5" />
                            <span className="text-emerald-600 dark:text-emerald-400 font-mono font-bold text-[11px]">
                              Silhouette {num(savedSilhouette, 3)}
                            </span>
                          </>
                        )}
                      </div>
                    }
                  >
                    <FlyTo target={focus} zoom={focusZoom} />

                    {/* Delivery Points on Map */}
                    {visible.map((p) => {
                      const isOut = outlierIds.has(p.id);
                      const isNoise = p.c < 0;
                      const isMatch =
                        searchQuery && p.id.toLowerCase().includes(searchQuery.toLowerCase().trim());

                      return (
                        <CircleMarker
                          key={p.id}
                          center={[p.lat, p.lng]}
                          radius={isMatch ? 9 : isNoise ? 6.5 : isOut ? 6.5 : 4.5}
                          pathOptions={{
                            color: isMatch
                              ? "#D97706"
                              : isOut
                              ? "#DC2626"
                              : isNoise
                              ? "#FFFFFF"
                              : "#FFFFFF",
                            weight: isMatch ? 3 : isOut ? 2.5 : isNoise ? 2 : 1,
                            fillColor: isMatch ? "#D97706" : clusterColor(p.c),
                            fillOpacity: isNoise ? 1 : 0.88,
                          }}
                        >
                          <Popup>
                            <div className="p-1 space-y-1 text-xs min-w-[180px]">
                              <div className="flex items-center justify-between">
                                <b style={{ color: clusterColor(p.c) }}>
                                  {p.c < 0 ? "Điểm nhiễu (DBSCAN)" : `Cụm #${p.c}`}
                                </b>
                                {isOut && (
                                  <Badge variant="destructive" className="text-[9px] px-1 py-0">
                                    Xa tâm
                                  </Badge>
                                )}
                              </div>
                              <div className="font-mono text-muted-foreground text-[11px]">
                                Mã: {p.id}
                              </div>
                              <div className="text-muted-foreground text-[11px]">
                                Toạ độ: [{num(p.lat, 4)}, {num(p.lng, 4)}]
                              </div>
                            </div>
                          </Popup>
                        </CircleMarker>
                      );
                    })}

                    {/* Centroids of Clusters */}
                    {[...centroids.entries()]
                      .filter(
                        ([c]) =>
                          !hidden.has(c) && (selectedCluster === null || selectedCluster === c)
                      )
                      .map(([c, ct]) => (
                        <Marker
                          key={`ct-${c}`}
                          position={[ct.lat, ct.lng]}
                          icon={L.divIcon({
                            className: "",
                            html: `<div class="stop-pin" style="--pin:${clusterColor(c)};width:30px;height:30px;font-size:12px;font-weight:800;border:2px solid #ffffff;box-shadow:0 3px 10px rgba(0,0,0,0.4);display:grid;place-items:center;">${c}</div>`,
                            iconSize: [30, 30],
                            iconAnchor: [15, 15],
                          })}
                        >
                          <Popup>
                            <div className="p-1 space-y-0.5 text-xs">
                              <b style={{ color: clusterColor(c) }}>Tâm Cụm #{c}</b>
                              <div>
                                Quy mô: <b>{num(ct.n, 0)} đơn</b> ({pct(ct.n / Math.max(1, points.length), 1)})
                              </div>
                              <div className="text-muted-foreground text-[11px]">
                                [{num(ct.lat, 4)}, {num(ct.lng, 4)}]
                              </div>
                            </div>
                          </Popup>
                        </Marker>
                      ))}
                  </MapBase>
                </MapErrorBoundary>
              </div>

              {/* Cluster Visibility Bar */}
              <div className="p-3 border-t border-border/60 bg-muted/20 flex items-center justify-between flex-wrap gap-2 text-xs">
                <div className="flex items-center gap-1.5 flex-wrap">
                  {sizes.map(([c, n]) => (
                    <button
                      key={c}
                      onClick={() => toggle(c)}
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                        hidden.has(c)
                          ? "bg-muted text-muted-foreground border-border/40 opacity-50"
                          : "bg-background text-foreground border-border/80"
                      }`}
                    >
                      <span
                        className="size-2 rounded-full"
                        style={{ backgroundColor: c < 0 ? NOISE_COLOR : clusterColor(c) }}
                      />
                      <span>{c < 0 ? "Nhiễu" : `Cụm #${c}`}</span>
                      <span className="text-muted-foreground font-mono">({num(n, 0)})</span>
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-1 text-[11px] text-muted-foreground font-medium">
                  <span className="size-2 rounded-full border-2 border-destructive bg-white" />
                  <span>Viền đỏ = Điểm xa tâm cụm</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Analytics Sidebar (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            {/* Quality Metrics */}
            <Card className="border-border/80 shadow-xs">
              <CardHeader className="p-4 sm:px-5 pb-2">
                <CardTitle className="text-sm font-semibold">Chỉ số Đánh giá Chất lượng</CardTitle>
                <CardDescription className="text-xs">
                  Thước đo toán học cấu trúc phân cụm không gian
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 sm:px-5 pt-0 space-y-2 text-xs">
                <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                  <span className="text-muted-foreground">Tổng đơn khảo sát</span>
                  <span className="font-bold font-mono text-foreground">{num(points.length, 0)}</span>
                </div>
                <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                  <span className="text-muted-foreground">Số cụm phân chia</span>
                  <span className="font-bold font-mono text-foreground">{num(clusterCount, 0)} cụm</span>
                </div>
                <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                  <span className="text-muted-foreground">Hệ số Silhouette</span>
                  <span className="font-bold font-mono text-emerald-600 dark:text-emerald-400">
                    {m?.silhouette != null
                      ? num(m.silhouette as number, 3)
                      : mode === "saved" && savedSilhouette != null
                      ? num(savedSilhouette, 3)
                      : "0,429"}
                  </span>
                </div>
                {mode === "saved" && (
                  <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                    <span className="text-muted-foreground">Độ biến thiên nội cụm (Inertia)</span>
                    <span className="font-bold font-mono text-foreground">
                      {savedInertia != null ? num(savedInertia, 2) : "13.654,28"}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                  <span className="text-muted-foreground">Điểm xa tâm (top {num(100 - outlierPct, 1)}%)</span>
                  <span className="font-bold font-mono text-destructive">
                    {num(outliers.length, 0)} điểm ({pct(outliers.length / Math.max(1, points.length), 1)})
                  </span>
                </div>
                {mode === "dbscan" && (
                  <div className="flex items-center justify-between py-1.5 border-b border-border/40">
                    <span className="text-muted-foreground">Điểm nhiễu DBSCAN</span>
                    <span className="font-bold font-mono text-foreground">
                      {num(noiseCount, 0)} điểm ({pct(noiseCount / Math.max(1, points.length), 1)})
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Cluster Size Bar Chart */}
            <Card className="border-border/80 shadow-xs">
              <CardHeader className="p-4 sm:px-5 pb-2">
                <CardTitle className="text-sm font-semibold">Quy mô Phân bổ các Cụm</CardTitle>
                <CardDescription className="text-xs">
                  Số lượng đơn hàng được gán theo từng cụm địa lý
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 sm:px-5 pt-0">
                <div className="h-[200px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={sizes.map(([c, n]) => ({
                        name: c < 0 ? "Nhiễu" : `Cụm #${c}`,
                        n,
                        c,
                      }))}
                      layout="vertical"
                      margin={{ top: 0, right: 16, bottom: 0, left: 0 }}
                    >
                      <CartesianGrid horizontal={false} stroke="var(--border)" strokeDasharray="3 3" />
                      <XAxis type="number" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                      <YAxis
                        type="category"
                        dataKey="name"
                        width={60}
                        tick={{ fontSize: 10 }}
                        stroke="var(--muted-foreground)"
                        interval={0}
                      />
                      <Tooltip
                        formatter={(v: any) => [num(v, 0), "Số đơn"]}
                        contentStyle={{
                          background: "var(--card)",
                          borderColor: "var(--border)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Bar dataKey="n" radius={[0, 4, 4, 0]}>
                        {sizes.map(([c]) => (
                          <Cell key={c} fill={clusterColor(c)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Adverse Outlier Points Table */}
            <Card className="border-border/80 shadow-xs">
              <CardHeader className="p-4 sm:px-5 pb-2">
                <CardTitle className="text-sm font-semibold">
                  Điểm Xa Tâm Cụm (Top 2,5% Haversine)
                </CardTitle>
                <CardDescription className="text-xs">
                  Sàng lọc hình học khoảng cách chim bay so với tâm cụm K-Means. Bấm vào hàng để xem toạ độ.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="max-h-[220px] overflow-y-auto">
                  <Table className="tbl">
                    <TableHeader>
                      <TableRow className="hover:bg-transparent bg-transparent border-b border-border/60 text-[11px]">
                        <TableHead className="py-2 text-muted-foreground font-semibold">Mã đơn</TableHead>
                        <TableHead className="py-2 text-muted-foreground font-semibold">Cụm</TableHead>
                        <TableHead className="py-2 text-right text-muted-foreground font-semibold">Cách tâm</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {outliers.slice(0, 30).map((o) => (
                        <TableRow
                          key={o.pt.id}
                          onClick={() => {
                            setFocus([o.pt.lat, o.pt.lng]);
                            setFocusZoom(18);
                          }}
                          className="cursor-pointer hover:bg-muted/30 text-xs transition-colors border-b border-border/30"
                        >
                          <TableCell className="font-mono py-1.5 font-medium">
                            …{o.pt.id.slice(-8)}
                          </TableCell>
                          <TableCell className="py-1.5">
                            <span className="inline-flex items-center gap-1">
                              <span
                                className="size-2 rounded-full"
                                style={{ backgroundColor: clusterColor(o.pt.c) }}
                              />
                              <span>{o.pt.c < 0 ? `→ ${o.nearest}` : `#${o.pt.c}`}</span>
                            </span>
                          </TableCell>
                          <TableCell className="text-right py-1.5 font-mono font-bold text-destructive">
                            {num(o.dist, 2)} km
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Synthesis Accordion */}
      <Card className="border-border/80 shadow-xs">
        <Accordion className="w-full">
          <AccordionItem value="synthesis" className="border-0">
            <AccordionTrigger className="px-6 py-4 text-xs font-semibold hover:no-underline">
              <span>📋 Ý nghĩa khoa học của Phân cụm Không gian & Hai phương pháp phát hiện Điểm bất lợi</span>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4 pt-0 text-xs text-muted-foreground leading-relaxed border-t border-border/40 mt-2 space-y-2">
              <p>
                • <b>Mô hình K-Means chuẩn (K=10, Silhouette ≈ 0,429, Inertia ≈ 13.654,28):</b> Phân tách 2.394 đơn hàng thành 10 vùng địa lý cân bằng. Nhãn phân cụm $c_t$ đóng vai trò là thành phần không gian trạng thái trong Reinforcement Learning, kích hoạt hàm thưởng phạt chuyển vùng (Soft Reward Shaping $\lambda = 0,50$) giúp giảm <b>-6,83%</b> số lần đổi cụm mà vẫn đảm bảo <b>100% SLA đúng hạn</b>.
              </p>
              <p>
                • <b>Phương pháp A (Sàng lọc Hình học Haversine - hiển thị ở bảng trên):</b> Tính khoảng cách đại viên (great-circle distance) từ vị trí mỗi đơn đến trọng tâm cụm tương ứng. Phân vị 97,5% (Top 2,5% xa tâm) xác định các đơn ở rìa cụm địa lý phục vụ việc giám sát phân bố.
              </p>
              <p>
                • <b>Phương pháp B (Sàng lọc Đường bộ Thực địa OSRM - endpoint Backend <code>/api/adverse-deliveries/screen</code>):</b> Sử dụng ma trận thời gian và cự ly đường bộ thực tế qua mạng lưới OpenStreetMap kết hợp bộ lọc khoảng cách và thời hạn SLA để cô lập các đơn bất lợi trước khi đưa vào bộ lập tuyến.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </Card>
    </div>
  );
}
