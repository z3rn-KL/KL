import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CircleMarker, Marker, Polyline, Popup } from "react-leaflet";
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
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
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
import { MapBase, depotIcon, pinIcon } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { RouteTransitionInspector } from "@/components/shared/RouteTransitionInspector";
import {
  api,
  type DeliveryRecord,
  type KmeansPoint,
  type OptimizeResponse,
} from "@/lib/api";
import {
  ALGO_COLOR,
  ALGO_LABEL,
  DEPOT,
  SERVICE_LABEL,
  clusterColor,
  dateTime,
  num,
  pct,
  signedPct,
  vnd,
} from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import { useTheme } from "@/lib/theme";
import {
  Play,
  Loader2,
  RotateCcw,
  Sparkles,
  Layers,
  Search,
  Plus,
  Trash2,
  AlertCircle,
  Clock,
  Compass,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  ChevronRight,
} from "lucide-react";

type Source = "cluster" | "workload" | "manual";
type Live = "nearest_neighbor" | "clarke_wright";
const LIVE: Live[] = ["nearest_neighbor", "clarke_wright"];
const MAX = 30;

interface Outcome {
  res?: OptimizeResponse;
  error?: string;
  clientSeconds: number;
}

function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample<T>(items: T[], n: number, seed: number): T[] {
  const rnd = mulberry32(seed);
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

const latlng = (d: DeliveryRecord): [number, number] => [
  Number(d.receiverLat),
  Number(d.receiverLng),
];

export default function RunLab() {
  const [theme] = useTheme();
  const [searchParams] = useSearchParams();
  const initialSource = searchParams.get("source");
  const initialWid = searchParams.get("wid");
  const initialWidNum = initialWid ? Number(initialWid) : null;

  const casingColor = theme === "dark" ? "rgba(11, 19, 33, 0.95)" : "rgba(255, 255, 255, 0.95)";
  const [source, setSource] = useState<Source>(() => {
    return initialSource === "workload" ? "workload" : "cluster";
  });
  const [cluster, setCluster] = useState(0);
  const [count, setCount] = useState(8);
  const [seed, setSeed] = useState(1);
  const [wid, setWid] = useState<number>(() => {
    return initialWidNum && !isNaN(initialWidNum) && initialWidNum > 0 ? initialWidNum : 1;
  });
  const [manual, setManual] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [metric, setMetric] = useState<"distance" | "travel_time">("distance");
  const [ret, setRet] = useState(true);
  const [startLocal, setStartLocal] = useState("");

  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<Partial<Record<Live, Outcome>>>({});
  const [selectedInspectorAlgo, setSelectedInspectorAlgo] = useState<Live>("clarke_wright");
  const [showRoute, setShowRoute] = useState<Record<Live, boolean>>({
    nearest_neighbor: true,
    clarke_wright: true,
  });
  const abortRef = useRef<AbortController | null>(null);

  // Handle handoff navigation from RoutesPage or query parameters
  useEffect(() => {
    const srcParam = searchParams.get("source");
    const widParam = searchParams.get("wid");
    if (srcParam === "workload" && widParam) {
      const parsed = Number(widParam);
      if (!isNaN(parsed) && parsed > 0) {
        setSource("workload");
        setWid(parsed);
      }
    }
  }, [searchParams]);

  const km = useAsync((s) => api.kmeansAll(s), []);
  const wls = useAsync((s) => api.workloads(200, s), []);
  const found = useAsync(
    (s) =>
      source === "manual" && search.trim().length >= 2
        ? api.deliveries(0, 8, { q: search.trim() }, s)
        : Promise.resolve(null),
    [source, search]
  );

  const clusterPoints: KmeansPoint[] = useMemo(
    () => km.data?.points.filter((p) => p.cluster_id === cluster) ?? [],
    [km.data, cluster]
  );
  // Size of every cluster; clusterPoints above only holds the selected one.
  const clusterSizes = useMemo(() => {
    const sizes: Record<number, number> = {};
    for (const p of km.data?.points ?? []) sizes[p.cluster_id] = (sizes[p.cluster_id] ?? 0) + 1;
    return sizes;
  }, [km.data]);
  const workload = wls.data?.items.find((w) => w.workload_id === wid) ?? null;

  const ids: string[] = useMemo(() => {
    if (source === "cluster")
      return sample(clusterPoints, Math.min(count, MAX), seed * 7919 + cluster).map(
        (p) => p.delivery_id
      );
    if (source === "workload") return workload?.delivery_ids ?? [];
    return manual;
  }, [source, clusterPoints, count, seed, cluster, workload, manual]);

  const key = ids.join(",");
  const recs = useAsync<DeliveryRecord[]>(
    (s) =>
      ids.length ? Promise.all(ids.map((id) => api.delivery(id, s))) : Promise.resolve([]),
    [key]
  );

  useEffect(() => {
    abortRef.current?.abort();
    setOut({});
    setBusy(false);
  }, [key, metric, ret, startLocal]);

  async function run() {
    if (!ids.length) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setBusy(true);
    setOut({});
    const start_time = startLocal ? new Date(`${startLocal}:00+07:00`).toISOString() : undefined;
    const results = await Promise.all(
      LIVE.map(async (a): Promise<[Live, Outcome]> => {
        const t0 = performance.now();
        try {
          const res = await api.optimize(
            {
              algorithm: a,
              delivery_ids: ids,
              metric,
              return_to_depot: ret,
              start_time,
            },
            ctrl.signal
          );
          return [a, { res, clientSeconds: (performance.now() - t0) / 1000 }];
        } catch (e) {
          return [
            a,
            {
              error: (e as Error).message,
              clientSeconds: (performance.now() - t0) / 1000,
            },
          ];
        }
      })
    );
    if (ctrl.signal.aborted) return;
    setOut(Object.fromEntries(results));
    setBusy(false);
  }

  const ok = LIVE.flatMap((a) => {
    const o = out[a];
    return o?.res ? [[a, o.res, o.clientSeconds] as const] : [];
  });
  const nnEval = out.nearest_neighbor?.res?.evaluation;

  const costs = useAsync(
    async (s) => {
      const m: Partial<Record<Live, number>> = {};
      await Promise.all(
        ok.map(async ([a, r]) => {
          const e = r.evaluation;
          const c = await api.cost(
            {
              total_distance_km: e.total_distance_km,
              total_travel_time_minutes: e.total_travel_time_minutes,
              late_deliveries: e.late_deliveries,
              total_lateness_minutes: e.total_lateness_minutes,
            },
            s
          );
          m[a] = c.result.total_estimated_cost;
        })
      );
      return m;
    },
    [JSON.stringify(ok.map(([a, r]) => [a, r.evaluation.total_distance_km]))]
  );

  const byId = new Map((recs.data ?? []).map((d) => [d.id, d]));
  const allPts: [number, number][] = [
    [DEPOT.lat, DEPOT.lng],
    ...(recs.data ?? []).map(latlng).filter(([a, b]) => Number.isFinite(a + b)),
  ];

  const activeInspectorAlgo: Live = out[selectedInspectorAlgo]?.res
    ? selectedInspectorAlgo
    : out.clarke_wright?.res
    ? "clarke_wright"
    : "nearest_neighbor";
  const activeOutcome = out[activeInspectorAlgo];
  const activeEval = activeOutcome?.res?.evaluation;
  const activeDeliveryOrder = activeEval?.delivery_order ?? [];

  const inspectorStops = useMemo(() => {
    if (!activeDeliveryOrder.length) return [];
    return activeDeliveryOrder.map((id, idx) => {
      const d = byId.get(id);
      return {
        stopIndex: idx + 1,
        deliveryId: id,
        package_name: d?.package_name != null ? String(d.package_name) : undefined,
        clusterId: typeof d?.cluster_id === "number" ? d.cluster_id : null,
        lat: d ? Number(d.receiverLat) : 0,
        lng: d ? Number(d.receiverLng) : 0,
        expectedDeliveryTime: d?.expectedDeliveryTime != null ? String(d.expectedDeliveryTime) : null,
        isLate: typeof d?.isLate === "boolean" ? d.isLate : null,
        address: d?.receiverAddress != null ? String(d.receiverAddress) : null,
      };
    });
  }, [activeDeliveryOrder, byId]);

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Thử nghiệm định tuyến"
        subtitle="Thử nghiệm trực tiếp Nearest Neighbor & Clarke-Wright qua API trên đồ thị OpenStreetMap TP.HCM (DQN là kết quả tiền tính toán)"
      />

      {/* Workload Handoff Notice Banner if navigated with query params */}
      {source === "workload" && initialSource === "workload" && (
        <Alert className="border-primary/40 bg-primary/5 text-xs py-2 px-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Compass className="size-4 text-primary shrink-0" />
            <div>
              <span className="font-semibold text-foreground">
                Đã nhận Workload #{wid} ({workload?.number_of_deliveries ?? ids.length} đơn hàng)
              </span>
              <span className="text-muted-foreground ml-1.5 hidden sm:inline">
                từ Bản đồ tuyến. Nhấn &quot;Chạy tối ưu ngay&quot; để tính toán tuyến qua API.
              </span>
            </div>
          </div>
          <Badge variant="outline" className="font-mono text-[10px] text-primary border-primary/30 shrink-0">
            Handoff Active
          </Badge>
        </Alert>
      )}

      {/* Simulation Setup Card */}
      <Card className="border-border/80 shadow-xs">
        <CardHeader className="p-3.5 sm:px-5 pb-2.5 border-b border-border/60">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <CardTitle className="text-sm font-semibold">
                  Thiết lập tham số bài toán định tuyến
                </CardTitle>
                <Badge variant="outline" className="text-[10px] font-mono">
                  Live Solvers
                </Badge>
              </div>
              <CardDescription className="text-xs">
                Chọn tập đơn hàng đầu vào (tối đa {MAX} đơn) và chạy đối sánh trực tiếp Nearest Neighbor & Clarke-Wright.
              </CardDescription>
            </div>

            <Button
              size="sm"
              onClick={run}
              disabled={busy || ids.length === 0}
              className="gap-2 text-xs bg-primary text-primary-foreground hover:bg-primary/90 shrink-0 font-semibold shadow-xs h-8"
            >
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
              <span>{busy ? "Đang giải song song..." : "Chạy tối ưu ngay"}</span>
            </Button>
          </div>
        </CardHeader>

        <CardContent className="p-3.5 sm:px-5 space-y-4 pt-3">
          {/* Nhóm A & B: Nguồn đầu vào và Tham số thử nghiệm */}
          <div className="space-y-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                1. Nguồn dữ liệu đầu vào:
              </span>
              <Tabs value={source} onValueChange={(v) => setSource(v as Source)}>
                <TabsList className="h-7">
                  <TabsTrigger value="cluster" className="text-xs">
                    Cụm K-Means
                  </TabsTrigger>
                  <TabsTrigger value="workload" className="text-xs">
                    Workload Chuẩn
                  </TabsTrigger>
                  <TabsTrigger value="manual" className="text-xs">
                    Thủ công
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Input Parameters Content Box */}
            <div className="rounded-lg border border-border/70 bg-card p-3 shadow-2xs">
              {source === "cluster" && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 items-start">
                  {/* Cluster Select */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-foreground flex items-center justify-between">
                      <span>Cụm không gian K-Means</span>
                      <span className="text-[11px] font-mono text-muted-foreground">10 Cụm TP.HCM</span>
                    </label>
                    <Select value={String(cluster)} onValueChange={(v) => setCluster(Number(v))}>
                      <SelectTrigger className="h-8 text-xs bg-background">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 10 }, (_, i) => (
                          <SelectItem key={i} value={String(i)} className="text-xs">
                            <div className="flex items-center gap-2">
                              <span
                                className="size-2 rounded-full"
                                style={{ backgroundColor: clusterColor(i) }}
                              />
                              <span>Cụm #{i}{km.data ? ` (${clusterSizes[i] ?? 0} đơn)` : ""}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Sample count with slider */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-foreground">Số lượng đơn mẫu</span>
                      <Badge variant="secondary" className="font-mono text-xs font-bold px-1.5 py-0">
                        {count} đơn
                      </Badge>
                    </div>
                    <div className="h-8 flex items-center px-1">
                      <Slider
                        min={3}
                        max={Math.min(25, clusterPoints.length || 25)}
                        step={1}
                        value={[count]}
                        onValueChange={(val: any) => setCount(Array.isArray(val) ? val[0] : val)}
                      />
                    </div>
                  </div>

                  {/* Random Seed */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-foreground flex items-center justify-between">
                      <span>Random Seed</span>
                      <span className="text-[11px] text-muted-foreground">Tái lập kết quả</span>
                    </label>
                    <div className="flex gap-1.5">
                      <Input
                        type="number"
                        value={seed}
                        onChange={(e) => setSeed(Number(e.target.value))}
                        className="h-8 text-xs font-mono bg-background"
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSeed((s) => s + 1)}
                        className="h-8 px-2.5 text-xs font-medium shrink-0"
                      >
                        Đổi seed
                      </Button>
                    </div>
                  </div>
                </div>
              )}

              {source === "workload" && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 items-center">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-foreground flex items-center justify-between">
                      <span>Đợt giao chuẩn hóa (Workload)</span>
                      <span className="text-[11px] text-muted-foreground font-mono">108 Workload</span>
                    </label>
                    <Select value={String(wid)} onValueChange={(v) => setWid(Number(v))}>
                      <SelectTrigger className="h-8 text-xs bg-background">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="w-[280px] max-h-64 shadow-2xl">
                        {(wls.data?.items ?? []).map((w) => (
                          <SelectItem key={w.workload_id} value={String(w.workload_id)} className="text-xs">
                            Workload #{w.workload_id} ({w.number_of_deliveries} đơn hàng)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="rounded-md border border-border/60 bg-muted/30 p-2.5 text-xs space-y-0.5">
                    <div className="font-semibold text-foreground flex items-center justify-between">
                      <span>Thông tin đợt giao #{wid}:</span>
                      <Badge variant="outline" className="font-mono text-[10px]">
                        {workload?.number_of_deliveries ?? 0} điểm giao
                      </Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-normal">
                      Dữ liệu đợt giao thực tế từ tập XeDu nội thành TP.HCM kèm tọa độ GPS và SLA cam kết.
                    </p>
                  </div>
                </div>
              )}

              {source === "manual" && (
                <div className="space-y-3">
                  <div className="flex flex-col sm:flex-row gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
                      <Input
                        type="search"
                        placeholder="Nhập mã đơn, người nhận, địa chỉ để tìm và thêm đơn..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-8 h-8 text-xs bg-background"
                      />
                    </div>
                    {manual.length > 0 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setManual([])}
                        className="h-9 text-xs text-muted-foreground hover:text-destructive gap-1 px-2.5"
                      >
                        <Trash2 className="size-3.5" />
                        <span>Xóa tất cả ({manual.length})</span>
                      </Button>
                    )}
                  </div>

                  {/* Search results dropdown suggestions */}
                  {found.data?.items && found.data.items.length > 0 && (
                    <div className="rounded-md border border-border bg-popover p-2 text-xs shadow-md space-y-1 max-h-48 overflow-y-auto">
                      <div className="text-[10px] font-bold uppercase text-muted-foreground px-2 py-0.5">
                        Gợi ý đơn hàng ({found.data.items.length})
                      </div>
                      {found.data.items.map((d: DeliveryRecord) => {
                        const isAdded = manual.includes(d.id);
                        return (
                          <div
                            key={d.id}
                            className="flex items-center justify-between p-1.5 rounded hover:bg-muted/70 transition-colors"
                          >
                            <div className="min-w-0 flex-1 pr-2">
                              <span className="font-mono font-bold text-foreground mr-2">#{d.id}</span>
                              <span className="text-muted-foreground truncate">{d.recipient_name ?? d.address ?? "Đơn hàng nội thành"}</span>
                            </div>
                            <Button
                              variant={isAdded ? "secondary" : "outline"}
                              size="sm"
                              disabled={isAdded || manual.length >= MAX}
                              onClick={() => setManual((prev) => [...prev, d.id])}
                              className="h-6 text-[10px] px-2 gap-1 cursor-pointer"
                            >
                              <Plus className="size-2.5" />
                              <span>{isAdded ? "Đã thêm" : "Thêm vào"}</span>
                            </Button>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Added manual chips */}
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-[11px] font-semibold text-muted-foreground mr-1">
                      Đơn đã chọn ({manual.length}/{MAX}):
                    </span>
                    {manual.length === 0 ? (
                      <span className="text-xs text-muted-foreground italic">
                        Chưa chọn đơn nào. Hãy tìm kiếm ở trên để thêm đơn vào lộ trình.
                      </span>
                    ) : (
                      manual.map((id) => (
                        <Badge
                          key={id}
                          variant="secondary"
                          className="text-xs font-mono gap-1.5 py-0.5 px-2 bg-accent text-accent-foreground border border-border/80"
                        >
                          <span>#{id}</span>
                          <button
                            type="button"
                            onClick={() => setManual((prev) => prev.filter((m) => m !== id))}
                            className="hover:text-destructive cursor-pointer"
                          >
                            ✕
                          </button>
                        </Badge>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <Separator />

          {/* Nhóm C: Ràng buộc hành trình & Tiêu chí */}
          <div className="space-y-3">
            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              2. Tiêu chí tối ưu & Ràng buộc hành trình:
            </span>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 items-center">
              {/* Metric selection */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-foreground flex items-center justify-between h-5">
                  <span>Hàm mục tiêu Heuristic</span>
                  <span className="text-[11px] text-muted-foreground">Trọng số giải thuật</span>
                </label>
                <Select value={metric} onValueChange={(v) => setMetric(v as any)}>
                  <SelectTrigger className="h-9 text-xs bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="distance" className="text-xs">
                      Tối thiểu Quãng đường (Distance-first)
                    </SelectItem>
                    <SelectItem value="travel_time" className="text-xs">
                      Tối thiểu Thời gian (Time-first)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Return to depot toggle in unified card box */}
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-card p-3 shadow-2xs h-[60px]">
                <div className="space-y-0.5">
                  <label htmlFor="return-depot" className="text-xs font-semibold text-foreground cursor-pointer select-none">
                    Quay về kho Depot (Quận 1)
                  </label>
                  <p className="text-[11px] text-muted-foreground">
                    Tài xế kết thúc ca giao hàng và quay về điểm tập kết trung tâm.
                  </p>
                </div>
                <Switch checked={ret} onCheckedChange={setRet} id="return-depot" />
              </div>
            </div>
          </div>
        </CardContent>

        <CardFooter className="px-4 sm:px-6 py-2.5 bg-transparent border-t border-border/50 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Sẵn sàng giải song song trên <b>{ids.length}</b> đơn hàng</span>
          </div>
          <span className="font-mono text-[11px]">Đồ thị OSM TP.HCM • Ma trận Dijkstra</span>
        </CardFooter>
      </Card>

      {/* Head-to-Head Comparison Results if run */}
      {ok.length > 0 && (
        <Card className="border-border/80 shadow-xs">
          <CardHeader className="p-4 sm:px-6 pb-3 border-b border-border/60">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base font-semibold">
                  Kết quả Đối đầu Trực tiếp (Head-to-Head)
                </CardTitle>
                <CardDescription className="text-xs">
                  Giải trên đồ thị OpenStreetMap với {ids.length} điểm giao hàng
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-0">
            <Table className="tbl">
              <TableHeader>
                <TableRow className="hover:bg-transparent bg-muted/20 text-xs">
                  <TableHead>Thuật toán</TableHead>
                  <TableHead className="text-right">Quãng đường</TableHead>
                  <TableHead className="text-right">Thời gian</TableHead>
                  <TableHead className="text-right">SLA Đúng hạn</TableHead>
                  <TableHead className="text-right">Chi phí ước tính</TableHead>
                  <TableHead className="text-right">Thời gian phản hồi API</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ok.map(([algKey, res, sec]) => {
                  const evalRes = res.evaluation;
                  const isNN = algKey === "nearest_neighbor";
                  const kmDiff = !isNN && nnEval ? evalRes.total_distance_km - nnEval.total_distance_km : null;
                  const kmPct = !isNN && nnEval ? (kmDiff! / nnEval.total_distance_km) * 100 : null;

                  return (
                    <TableRow key={algKey} className="text-xs">
                      <TableCell className="font-semibold">
                        <div className="flex items-center gap-2">
                          <span
                            className="size-2 rounded-full"
                            style={{ backgroundColor: ALGO_COLOR[algKey] }}
                          />
                          <span>{ALGO_LABEL[algKey]}</span>
                          {algKey === "clarke_wright" && kmPct !== null && kmPct < 0 && (
                            <Badge variant="outline" className="text-[10px] text-emerald-600 dark:text-emerald-400 border-emerald-500/40">
                              Tiết kiệm {num(Math.abs(kmPct), 1)}%
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-mono font-bold">
                        {num(evalRes.total_distance_km, 2)} km
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {num(evalRes.total_travel_time_minutes, 1)} ph
                      </TableCell>
                      <TableCell className="text-right font-mono text-emerald-600 dark:text-emerald-400">
                        {pct(evalRes.on_time_rate, 0)}
                      </TableCell>
                      <TableCell className="text-right font-mono font-semibold">
                        {costs.data?.[algKey] ? vnd(costs.data[algKey]) : "–"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-muted-foreground text-[11px]">
                        {num(sec, 2)} s
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
          <CardFooter className="px-4 sm:px-6 py-2.5 border-t border-border/50 bg-muted/20 text-xs text-muted-foreground flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="size-3.5 text-primary shrink-0" />
              <span>
                <b>Mô hình ước tính chi phí kịch bản:</b> 4.000 đ/km, 40.000 đ/giờ (~666,67 đ/phút) qua endpoint <code>/api/evaluation/cost</code>. Đây là ước tính theo kịch bản mô phỏng nghiên cứu, không phải chi phí kế toán thực tế của XeDu.
              </span>
            </div>
            <Badge variant="outline" className="text-[10px] font-mono shrink-0">
              Scenario Estimate
            </Badge>
          </CardFooter>
        </Card>
      )}

      {/* Route Transition Inspector for Computed Route Sequence */}
      {ok.length > 0 && inspectorStops.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              Thanh tra chuỗi ghé thăm giải pháp:
            </span>
            <div className="flex items-center gap-2">
              {LIVE.map((a) => {
                if (!out[a]?.res) return null;
                const isSelected = activeInspectorAlgo === a;
                return (
                  <button
                    key={a}
                    type="button"
                    onClick={() => setSelectedInspectorAlgo(a)}
                    className={`h-7 px-2.5 text-xs font-medium rounded-md border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-primary text-primary-foreground border-primary font-semibold shadow-xs"
                        : "bg-card border-border/80 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {ALGO_LABEL[a]}
                  </button>
                );
              })}
            </div>
          </div>

          <RouteTransitionInspector
            stops={inspectorStops}
            algorithmLabel={ALGO_LABEL[activeInspectorAlgo]}
            workloadId={source === "workload" ? wid : undefined}
            totalDistanceKm={activeEval?.total_distance_km}
            totalTravelTimeMinutes={activeEval?.total_travel_time_minutes}
            onTimeRate={activeEval?.on_time_rate}
          />
        </div>
      )}

      {/* Map Viewport for Solutions */}
      <Card className="overflow-hidden border-border/80 shadow-xs">
        <CardHeader className="p-4 sm:px-6 border-b border-border/60">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold">
              Bản đồ Tuyến Giải pháp trên Mạng đường Thực tế
            </CardTitle>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <Switch
                  checked={showRoute.nearest_neighbor}
                  onCheckedChange={(v) => setShowRoute((p) => ({ ...p, nearest_neighbor: v }))}
                  id="show-nn"
                />
                <label htmlFor="show-nn" className="cursor-pointer font-medium text-muted-foreground">
                  Hiện Nearest Neighbor
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={showRoute.clarke_wright}
                  onCheckedChange={(v) => setShowRoute((p) => ({ ...p, clarke_wright: v }))}
                  id="show-cw"
                />
                <label htmlFor="show-cw" className="cursor-pointer font-medium text-muted-foreground">
                  Hiện Clarke–Wright
                </label>
              </div>
            </div>
          </div>
        </CardHeader>

        <div className="h-[520px] w-full relative">
          <MapErrorBoundary>
            <MapBase height="100%" fit={allPts.length > 1 ? allPts : undefined}>
              {/* Depot */}
              <Marker position={[DEPOT.lat, DEPOT.lng]} icon={depotIcon()}>
                <Popup>
                  <div className="p-1 text-xs">
                    <b>{DEPOT.name}</b>
                    <div className="text-muted-foreground">Điểm xuất phát tài xế</div>
                  </div>
                </Popup>
              </Marker>

              {/* Delivery markers */}
              {(recs.data ?? []).map((d, i) => {
                const lat = Number(d.receiverLat);
                const lng = Number(d.receiverLng);
                if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
                const cColor = typeof d.cluster_id === "number" ? clusterColor(d.cluster_id) : "#2563EB";

                return (
                  <CircleMarker
                    key={d.id}
                    center={[lat, lng]}
                    radius={7}
                    pathOptions={{
                      color: "#FFFFFF",
                      weight: 2,
                      fillColor: cColor,
                      fillOpacity: 1,
                    }}
                  >
                    <Popup>
                      <div className="p-1 text-xs space-y-0.5">
                        <b>{String(d.package_name ?? "Đơn hàng")}</b>
                        <div className="text-muted-foreground">{String(d.receiverAddress ?? "")}</div>
                        <div>Hạn SLA: {dateTime(d.expectedDeliveryTime)}</div>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}

              {/* Nearest Neighbor Route Polyline */}
              {showRoute.nearest_neighbor && out.nearest_neighbor?.res?.evaluation?.route_geometry && (
                <Polyline
                  positions={out.nearest_neighbor.res.evaluation.route_geometry}
                  pathOptions={{
                    color: ALGO_COLOR.nearest_neighbor,
                    weight: 3.5,
                    opacity: 0.85,
                  }}
                />
              )}

              {/* Clarke-Wright Route Polyline */}
              {showRoute.clarke_wright && out.clarke_wright?.res?.evaluation?.route_geometry && (
                <Polyline
                  positions={out.clarke_wright.res.evaluation.route_geometry}
                  pathOptions={{
                    color: ALGO_COLOR.clarke_wright,
                    weight: 3.5,
                    opacity: 0.9,
                    dashArray: "6 6",
                  }}
                />
              )}
            </MapBase>
          </MapErrorBoundary>
        </div>
      </Card>
    </div>
  );
}
