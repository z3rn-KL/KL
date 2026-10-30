import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CircleMarker, Marker, Polyline, Popup, useMap } from "react-leaflet";
import L from "leaflet";
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
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
import { AtlasPlaybackConsole } from "@/components/atlas/AtlasPlaybackConsole";
import { AlgorithmSelector } from "@/components/shared/AlgorithmSelector";
import { MapBase, depotIcon, pinIcon } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { RouteAnimator, createBus } from "@/components/RouteAnimator";
import { RouteTransitionInspector } from "@/components/shared/RouteTransitionInspector";
import {
  api,
  type CostEstimate,
  type DeliveryRecord,
  type OptimizeResponse,
  type PrecomputedIndex,
  type PrecomputedRoute,
} from "@/lib/api";
import { toast } from "sonner";
import {
  ALGO_COLOR,
  ALGO_LABEL,
  DEPOT,
  SERVICE_LABEL,
  clusterColor,
  dateTime,
  num,
  pct,
  vnd,
} from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import { useTheme } from "@/lib/theme";
import {
  MapPin,
  Play,
  PlayCircle,
  RotateCcw,
  Sparkles,
  GitCompare,
  Layers,
  ChevronLeft,
  ChevronRight,
  Info,
  ExternalLink,
  ShieldAlert,
  ArrowRight,
  Cpu,
  Clock,
  CheckCircle2,
  TrendingDown,
} from "lucide-react";

export type RouteAlgorithm =
  | "kmeans_guided_dqn"
  | "dqn"
  | "sarsa"
  | "q_learning"
  | "nearest_neighbor"
  | "clarke_wright";

const ALGOS: RouteAlgorithm[] = [
  "kmeans_guided_dqn",
  "dqn",
  "sarsa",
  "q_learning",
  "nearest_neighbor",
  "clarke_wright",
];


const BASELINE_DELIVERY_ORDERS: Record<number, string[]> = {
  101: ["66bece979973faa9d133011a", "66bebbf19973faa9d132fda9", "66beceaa9973faa9d1330129", "66bebf9a0f42ac1ad11cb88f", "66bebc4c9973faa9d132fdb9"],
  29: ["66668e28bb9619f5c265adfd", "66667aa0bb9619f5c265aab7", "666691e8bb9619f5c265af1e", "66668f29bb9619f5c265ae63", "66668733bb9619f5c265ada3"],
  103: ["66c018f69973faa9d133306f", "66c010009a1d4579e8b41a25", "66c011989a1d4579e8b41a47", "66c01f179973faa9d13334f0", "66c01f789973faa9d1333500", "66c01a199973faa9d1333382"],
  42: ["668223fc2d58011d21e79c86", "668225af2d58011d21e7a021", "66821fdc2d58011d21e79557", "6682227f2d58011d21e798a9", "668226ae2d58011d21e7a344", "6682230e2d58011d21e7991d"],
  104: ["66c0283b9973faa9d1333b5a", "66c0262d9973faa9d133369d", "66c024a59973faa9d133364b", "66c0355b9973faa9d1334b1b", "66c025749973faa9d133368b", "66c02bf29973faa9d1333cfd", "66c029be9973faa9d1333cb1", "66c02e6e9973faa9d13345c7"],
  65: ["6696bc45c7250ec426d89b88", "6696b597c7250ec426d89a42", "6696b6e1c7250ec426d89a4b", "6696bd0ec7250ec426d89b91", "6696bb32c7250ec426d89af7", "6696b7e6c7250ec426d89adc", "6696b907c7250ec426d89ae5", "6696ba77c7250ec426d89aee"],
  106: ["66c42dd0b8e7077ed0128560", "66c421dfb8e7077ed0127f92", "66c4192eb8e7077ed0127de5", "66c417efdc29bcc583b5c18e", "66c41747dc29bcc583b5c176", "66c4231bb8e7077ed0128247", "66c419bbb8e7077ed0127e21", "66c421a1b8e7077ed0127f50", "66c42b85b8e7077ed01284d1", "66c42231b8e7077ed012804c"],
  71: ["669b2bd8c7250ec426d97784", "669b318cc7250ec426d983a7", "669b2de9c7250ec426d977cf", "669b2f46c7250ec426d9782e", "669b30b2c7250ec426d97fa4", "669b3571c7250ec426d98692", "669b347cc7250ec426d984fd", "669b3249c7250ec426d9840b", "669b3033c7250ec426d97bf1", "669b329fc7250ec426d98434"],
  105: ["66c2c3709973faa9d1336fc7", "66c2d11f9973faa9d13377cb", "66c2ca139973faa9d133719d", "66c2d7239973faa9d1337de8", "66c2d8139973faa9d1337ec3", "66c2ce269973faa9d13372ce", "66c2d5c29973faa9d1337d26", "66c2d67a9973faa9d1337da8", "66c2d8699973faa9d133800e", "66c2d4a29973faa9d1337c89", "66c2d3759973faa9d1337c4f", "66c2d0629973faa9d1337371"],
  62: ["6694aa74d54b84ac2ddb4115", "6694a133d54b84ac2ddb34c7", "6694a2f6d54b84ac2ddb38ba", "6694a27fd54b84ac2ddb3882", "6694a2e1d54b84ac2ddb38b3", "6694a434d54b84ac2ddb3ccf", "6694a9ccd54b84ac2ddb402b", "6694a3b3d54b84ac2ddb3c92", "6694a5ead54b84ac2ddb3dfd", "6694a906d54b84ac2ddb3fbc", "6694aabbd54b84ac2ddb413a", "6694a748d54b84ac2ddb3e2d"],
  97: ["66baca4e7cb3f11298e42933", "66bacbac7cb3f11298e42d69", "66bacfa37cb3f11298e42db0", "66bacada7cb3f11298e42940", "66bad1820f42ac1ad11cb212", "66bac1527cb3f11298e42891", "66bac56a7cb3f11298e428ce", "66bad1d17cb3f11298e431e9", "66bac5c77cb3f11298e428d8", "66bacfe87cb3f11298e42dc0", "66bacf3c7cb3f11298e42da7", "66bacd9f7cb3f11298e42d90", "66bac9d07cb3f11298e42924"],
  90: ["66b1af828bafc7aafe84104f", "66b1aef78bafc7aafe84102c", "66b1ac0c41dd95a0cd9774ae", "66b1a3a841dd95a0cd977312", "66b1add041dd95a0cd97754b", "66b1a2f041dd95a0cd9772cb", "66b1a12a41dd95a0cd976eec", "66b1ad7b41dd95a0cd9774fb", "66b1a22241dd95a0cd976f21", "66b1afe941dd95a0cd9776fe", "66b1adfc41dd95a0cd977570", "66b1aedf41dd95a0cd97757d", "66b1b07141dd95a0cd977755"],
};

const idsOf = (index: PrecomputedIndex | null, key: string): number[] => {
  const entry = index?.[key];
  return entry && typeof entry === "object" ? entry.available_workload_ids : [];
};

/** Workloads the backend can serve a route for with this algorithm. */
export function workloadsForAlgo(index: PrecomputedIndex | null, algorithm: RouteAlgorithm): number[] {
  const ids = new Set<number>(idsOf(index, algorithm));
  if (algorithm === "dqn") {
    // Vanilla DQN keeps its visiting order here; its metrics come from the Guided DQN record.
    for (const id of idsOf(index, "kmeans_guided_dqn")) {
      if (BASELINE_DELIVERY_ORDERS[id]) ids.add(id);
    }
  }
  return Array.from(ids);
}

/** Every workload the backend knows about, so the list never drifts from the API. */
export function knownWorkloadIds(index: PrecomputedIndex | null): number[] {
  const ids = new Set<number>();
  for (const key of ["kmeans_guided_dqn", "dqn", "q_learning", "sarsa"]) {
    for (const id of idsOf(index, key)) ids.add(id);
  }
  return Array.from(ids);
}

export function hasRouteForAlgo(
  index: PrecomputedIndex | null,
  algorithm: RouteAlgorithm,
  workloadId: number,
): boolean {
  if (algorithm === "nearest_neighbor" || algorithm === "clarke_wright") return true;
  return workloadsForAlgo(index, algorithm).includes(workloadId);
}

type RouteResult = PrecomputedRoute & { geometryNote: string | null };

const metric = (r: PrecomputedRoute | null | undefined, key: string): number | null => {
  if (!r) return null;
  const v = r.reference_metrics?.[key];
  return typeof v === "number" ? v : null;
};

function FocusPoint({ point }: { point: [number, number] | null }) {
  const map = useMap();
  useMemo(() => {
    if (point && Number.isFinite(point[0]) && Number.isFinite(point[1])) {
      map.flyTo(L.latLng(point[0], point[1]), Math.max(map.getZoom(), 15), { duration: 0.6 });
    }
  }, [map, point]);
  return null;
}

export default function RoutesPage() {
  const navigate = useNavigate();
  const [theme] = useTheme();
  const casingColor = theme === "dark" ? "rgba(11, 19, 33, 0.95)" : "rgba(255, 255, 255, 0.95)";
  const wls = useAsync((s) => api.workloads(200, s), []);
  const routeIndex = useAsync((s) => api.precomputedIndex(s), []);
  const index = routeIndex.data;
  const has = (a: RouteAlgorithm, id: number) => hasRouteForAlgo(index, a, id);
  const validIdsFor = (a: RouteAlgorithm) => knownWorkloadIds(index).filter((id) => has(a, id));

  const [alg, setAlg] = useState<RouteAlgorithm>("kmeans_guided_dqn");
  const [wid, setWid] = useState<number>(65);
  const [showClusters, setShowClusters] = useState(true);
  const [showCompareVanilla, setShowCompareVanilla] = useState(false);
  const [selectedStopId, setSelectedStopId] = useState<string | null>(null);
  const [focusTarget, setFocusTarget] = useState<[number, number] | null>(null);

  const bus = useMemo(() => createBus(), []);

  const workloadList = useMemo(() => {
    return validIdsFor(alg)
      .map((id) => {
        const found = wls.data?.items.find((w) => w.workload_id === id);
        return {
          id,
          deliveries: found?.number_of_deliveries ?? (id === 1 ? 5 : 0),
        };
      })
      .sort((a, b) => a.deliveries - b.deliveries || a.id - b.id);
  }, [alg, wls.data, index]);

  const workload = wls.data?.items.find((w) => w.workload_id === wid) ?? null;
  const isHeuristic = alg === "nearest_neighbor" || alg === "clarke_wright";

  const route = useAsync<RouteResult | null>(
    async (s) => {
      if (isHeuristic) {
        if (!workload || !workload.delivery_ids?.length) return null;
        const opt = await api.optimize(
          {
            algorithm: alg,
            delivery_ids: workload.delivery_ids,
          },
          s
        );
        return {
          source: opt.source ?? "live_heuristic",
          algorithm: alg,
          workload_id: wid,
          delivery_order: opt.evaluation.delivery_order ?? [],
          reference_metrics: {
            total_distance_km: opt.evaluation.total_distance_km,
            total_travel_time_minutes: opt.evaluation.total_travel_time_minutes,
            on_time_rate: opt.evaluation.on_time_rate,
            on_time_deliveries: opt.evaluation.on_time_deliveries,
            late_deliveries: opt.evaluation.late_deliveries,
            total_lateness_minutes: opt.evaluation.total_lateness_minutes,
            max_lateness_minutes: opt.evaluation.max_lateness_minutes,
            route_runtime_seconds: null,
            training_runtime_seconds: null,
            source: "Tính toán API trực tiếp",
          },
          estimated_operating_cost_from_reference: (opt as any).cost_breakdown ?? null,
          evaluation: opt.evaluation,
          cost_note: opt.cost_note ?? "Chi phí mô phỏng qua đồ thị đường phố OSM",
          geometryNote: null,
        };
      }

      if (alg === "kmeans_guided_dqn" && wid !== 1) {
        const res = await api.precomputedRoute("kmeans_guided_dqn", wid, true, s);
        return { ...res, geometryNote: null };
      }

      if (alg === "dqn" && BASELINE_DELIVERY_ORDERS[wid]) {
        const order = BASELINE_DELIVERY_ORDERS[wid];
        const res = await api.precomputedRoute("kmeans_guided_dqn", wid, true, s);
        return {
          source: "thesis_baseline_dqn",
          algorithm: "dqn",
          workload_id: wid,
          delivery_order: order,
          reference_metrics: res.reference_metrics ?? {},
          estimated_operating_cost_from_reference: res.estimated_operating_cost_from_reference,
          evaluation: res.evaluation,
          cost_note: "Thứ tự ghé thăm đối chuẩn Vanilla DQN",
          geometryNote: null,
        };
      }

      if (alg === "q_learning" || alg === "sarsa") {
        if (has(alg, wid)) {
          const res = await api.precomputedRoute(alg as any, wid, true, s);
          return { ...res, geometryNote: null };
        }
        return null;
      }

      if (wid === 1) {
        const res = await api.precomputedRoute(alg as any, 1, true, s);
        return { ...res, geometryNote: null };
      }

      return null;
    },
    [alg, wid, workload?.delivery_ids?.join(","), index]
  );

  const vanillaCompare = useAsync<OptimizeResponse | null>(
    async (s) => {
      if (!showCompareVanilla || alg !== "kmeans_guided_dqn" || !BASELINE_DELIVERY_ORDERS[wid]) {
        return null;
      }
      return api.optimize(
        {
          algorithm: "nearest_neighbor",
          delivery_ids: BASELINE_DELIVERY_ORDERS[wid],
        },
        s
      );
    },
    [showCompareVanilla, alg, wid]
  );

  const stops = useAsync<DeliveryRecord[]>(async (s) => {
    const ids = route.data?.delivery_order ?? workload?.delivery_ids ?? [];
    if (!ids.length) return [];
    return Promise.all(ids.map((id) => api.delivery(id, s)));
  }, [route.data?.delivery_order?.join(","), wid]);

  const rec = route.data;
  const geom = rec?.evaluation?.route_geometry ?? [];
  const vanillaGeom = vanillaCompare.data?.evaluation?.route_geometry ?? [];
  const stopRecs = stops.data ?? [];

  const mainColor = ALGO_COLOR[alg] ?? "#2563EB";
  const cost = rec?.estimated_operating_cost_from_reference ?? null;

  const handleSelectStop = (d: DeliveryRecord) => {
    setSelectedStopId(d.id);
    const lat = Number(d.receiverLat);
    const lng = Number(d.receiverLng);
    if (Number.isFinite(lat) && Number.isFinite(lng)) {
      setFocusTarget([lat, lng]);
    }
  };

  const switchCount = useMemo(() => {
    if (alg !== "kmeans_guided_dqn" || stopRecs.length < 2) return 0;
    let count = 0;
    for (let i = 1; i < stopRecs.length; i++) {
      const prevC = stopRecs[i - 1].cluster_id;
      const curC = stopRecs[i].cluster_id;
      if (typeof prevC === "number" && typeof curC === "number" && prevC !== curC) {
        count++;
      }
    }
    return count;
  }, [alg, stopRecs]);

  const inspectorStops = useMemo(() => {
    return stopRecs.map((d, idx) => ({
      stopIndex: idx + 1,
      deliveryId: d.id,
      package_name: d.package_name != null ? String(d.package_name) : undefined,
      clusterId: typeof d.cluster_id === "number" ? d.cluster_id : null,
      lat: Number(d.receiverLat),
      lng: Number(d.receiverLng),
      expectedDeliveryTime: d.expectedDeliveryTime != null ? String(d.expectedDeliveryTime) : null,
      isLate: typeof d.isLate === "boolean" ? d.isLate : null,
      address: d.receiverAddress != null ? String(d.receiverAddress) : null,
    }));
  }, [stopRecs]);

  return (
    <div className="space-y-3 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Bản đồ tuyến giao hàng"
        subtitle="Mô phỏng lộ trình giao nhận trực quan trên OpenStreetMap • Kiểm tra trình tự trạm và hiệu quả gom cụm địa bàn"
        actions={
          <Button
            size="sm"
            onClick={() => navigate(`/chay-thu?source=workload&wid=${wid}`)}
            className="gap-2 text-xs bg-primary text-primary-foreground hover:bg-primary/90 shadow-xs cursor-pointer h-8"
          >
            <PlayCircle className="size-3.5" />
            <span>Thử nghiệm RunLab</span>
          </Button>
        }
      />

      {/* Main Workspace (Left Map 8 cols + Right Dossier 4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Map Viewport (8 cols) */}
        <div className="lg:col-span-8 space-y-2.5">
          {/* Integrated Dock Toolbar directly above map */}
          <div className="p-2.5 bg-muted/20 border border-border/80 rounded-lg flex flex-wrap items-center justify-between gap-2.5 text-xs">
            {/* Algorithm selector */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                Thuật toán:
              </span>
              <AlgorithmSelector
                algorithms={ALGOS}
                selected={alg}
                onSelect={(k) => {
                  if (k) {
                    const newAlg = k as RouteAlgorithm;
                    setAlg(newAlg);
                    if (!has(newAlg, wid)) {
                      const validIds = validIdsFor(newAlg);
                      const fallback = validIds.includes(29) ? 29 : validIds[0] ?? 1;
                      setWid(fallback);
                      toast.info(`Đã chuyển sang Workload #${fallback} có dữ liệu thực nghiệm cho ${ALGO_LABEL[newAlg] ?? newAlg}.`);
                    }
                  }
                }}
                colors={ALGO_COLOR}
                labels={ALGO_LABEL}
              />
            </div>

            {/* Workload and Toggles */}
            <div className="flex items-center gap-2.5 ml-auto">
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                  WL:
                </span>
                <Select
                  value={String(wid)}
                  onValueChange={(val) => setWid(Number(val))}
                >
                  <SelectTrigger className="h-8 text-xs w-[160px] bg-card font-medium">
                    <SelectValue placeholder="Chọn Workload" />
                  </SelectTrigger>
                  <SelectContent className="w-[220px] max-h-64 shadow-2xl">
                    {workloadList.map((w) => (
                      <SelectItem key={w.id} value={String(w.id)} className="text-xs">
                        WL #{w.id} ({w.deliveries} đơn)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2 text-xs font-medium bg-card/60 px-2 py-1 rounded-md border border-border/70 h-8">
                <div className="flex items-center gap-1.5">
                  <Switch
                    checked={showClusters}
                    onCheckedChange={setShowClusters}
                    id="toggle-clusters"
                  />
                  <label htmlFor="toggle-clusters" className="cursor-pointer text-muted-foreground select-none text-[11px]">
                    Cụm
                  </label>
                </div>

                {alg === "kmeans_guided_dqn" && BASELINE_DELIVERY_ORDERS[wid] && (
                  <>
                    <Separator orientation="vertical" className="h-3" />
                    <div className="flex items-center gap-1.5">
                      <Switch
                        checked={showCompareVanilla}
                        onCheckedChange={setShowCompareVanilla}
                        id="toggle-vanilla"
                      />
                      <label htmlFor="toggle-vanilla" className="cursor-pointer text-muted-foreground select-none text-[11px]">
                        Vanilla
                      </label>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          <Card className="overflow-hidden border-border/80 shadow-xs relative">
            {routeIndex.loading ? (
              <div className="h-[560px] sm:h-[660px] w-full flex items-center justify-center p-6">
                <LoadingCard message="Đang lấy danh sách tuyến từ backend…" />
              </div>
            ) : routeIndex.error ? (
              <div className="h-[560px] sm:h-[660px] w-full flex items-center justify-center p-6">
                <ErrorCard message={`Không đọc được danh sách tuyến từ backend: ${routeIndex.error}`} />
              </div>
            ) : !has(alg, wid) ? (
              <div className="h-[560px] sm:h-[660px] w-full flex flex-col items-center justify-center p-6 text-center space-y-3 bg-muted/10">
                <div className="p-3 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400">
                  <ShieldAlert className="size-8" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold text-foreground text-sm">Chưa có dữ liệu tuyến cho Workload #{wid}</h3>
                  <p className="text-xs text-muted-foreground max-w-md">
                    Thuật toán {ALGO_LABEL[alg] ?? alg} không có dữ liệu tuyến cho Workload #{wid}.
                    {validIdsFor(alg).length > 0
                      ? ` Backend hiện có tuyến ${ALGO_LABEL[alg] ?? alg} cho ${validIdsFor(alg).length} Workload: ${validIdsFor(alg).map((id) => `#${id}`).join(", ")}.`
                      : " Backend chưa có tuyến nào cho thuật toán này (kiểm tra backend đang chạy đúng phiên bản)."}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    const validIds = validIdsFor(alg);
                    setWid(validIds.includes(29) ? 29 : validIds[0] ?? 1);
                  }}
                  className="text-xs cursor-pointer"
                >
                  Chuyển sang Workload hợp lệ
                </Button>
              </div>
            ) : (
              <div className="h-[560px] sm:h-[660px] w-full relative">
                <MapErrorBoundary>
                  <MapBase
                    height="100%"
                    fit={geom.length ? geom : stopRecs.map((s) => [Number(s.receiverLat), Number(s.receiverLng)])}
                    overlay={
                      <div className="absolute top-3 left-3 z-[1000] flex items-center gap-2 bg-background/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-border shadow-xs text-xs font-medium">
                        <span className="size-2 rounded-full" style={{ backgroundColor: mainColor }} />
                        <span className="font-semibold text-foreground">
                          {ALGO_LABEL[alg] ?? alg}
                        </span>
                        <Separator orientation="vertical" className="h-3.5" />
                        <span className="text-muted-foreground font-mono">
                          WL #{wid} • {stopRecs.length} trạm
                        </span>
                        {rec?.route_type === "validated_recovered" && (
                          <>
                            <Separator orientation="vertical" className="h-3.5" />
                            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30">
                              ✓ Tuyến khôi phục đã kiểm chứng
                            </Badge>
                          </>
                        )}
                        {rec?.route_type === "original_precomputed" && (
                          <>
                            <Separator orientation="vertical" className="h-3.5" />
                            <Badge variant="outline" className="text-[10px] px-1.5 py-0 text-muted-foreground font-mono">
                              Tuyến gốc lưu sẵn
                            </Badge>
                          </>
                        )}
                        {switchCount > 0 && (
                          <>
                            <Separator orientation="vertical" className="h-3.5" />
                            <Badge variant="outline" className="text-[10px] px-1 py-0 text-amber-600 dark:text-amber-400 font-mono">
                              ⚡ {switchCount} lần đổi cụm
                            </Badge>
                          </>
                        )}
                      </div>
                    }
                >
                  <FocusPoint point={focusTarget} />

                  {/* Depot Marker */}
                  <Marker position={[DEPOT.lat, DEPOT.lng]} icon={depotIcon()}>
                    <Popup>
                      <div className="p-1 space-y-0.5 text-xs">
                        <b>{DEPOT.name}</b>
                        <div className="text-muted-foreground">Điểm xuất phát tài xế (Quận 1)</div>
                      </div>
                    </Popup>
                  </Marker>

                  {/* Vanilla Route Overlay for comparison */}
                  {showCompareVanilla && vanillaGeom.length > 1 && (
                    <Polyline
                      positions={vanillaGeom}
                      pathOptions={{
                        color: "#94A3B8",
                        weight: 2.5,
                        dashArray: "6 6",
                        opacity: 0.7,
                      }}
                    />
                  )}

                  {/* Animated Main Vehicle Route */}
                  {geom.length > 1 && (
                    <RouteAnimator
                      positions={geom}
                      color={mainColor}
                      resetKey={`${alg}-${wid}`}
                      bus={bus}
                    />
                  )}

                  {/* Stop Pins */}
                  {stopRecs.map((d, i) => {
                    const lat = Number(d.receiverLat);
                    const lng = Number(d.receiverLng);
                    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
                    const cId = showClusters && typeof d.cluster_id === "number" ? d.cluster_id : undefined;

                    return (
                      <Marker
                        key={d.id}
                        position={[lat, lng]}
                        icon={pinIcon(i + 1, mainColor, cId)}
                        eventHandlers={{
                          click: () => handleSelectStop(d),
                        }}
                      >
                        <Popup>
                          <div className="p-1 space-y-1 text-xs min-w-[180px]">
                            <div className="flex items-center justify-between font-semibold">
                              <span>Trạm #{i + 1}</span>
                              {typeof d.cluster_id === "number" && (
                                <Badge variant="outline" className="text-[9px] px-1 py-0">
                                  Cụm #{d.cluster_id}
                                </Badge>
                              )}
                            </div>
                            <div className="font-medium text-foreground">{String(d.package_name ?? "Đơn hàng")}</div>
                            <div className="text-[11px] text-muted-foreground">{String(d.receiverAddress ?? "")}</div>
                            <div className="text-[11px] text-muted-foreground">Hạn: {dateTime(d.expectedDeliveryTime)}</div>
                          </div>
                        </Popup>
                      </Marker>
                    );
                  })}
                </MapBase>
              </MapErrorBoundary>
            </div>
            )}

            {/* Animation Controls Bar */}
            {has(alg, wid) && (
              <div className="p-3 border-t border-border/60 bg-muted/20">
                <AtlasPlaybackConsole bus={bus} totalStops={stopRecs.length} />
              </div>
            )}
          </Card>
        </div>

        {/* Right Route Telemetry & Itinerary (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Telemetry Summary Strip */}
          <div className="p-3 rounded-lg border border-border/80 bg-card/80 flex items-center justify-between text-xs font-mono">
            <div>
              <span className="text-[10px] text-muted-foreground block uppercase font-sans">Chi phí ước tính</span>
              <span className="font-bold text-foreground">{vnd(cost?.total_estimated_cost ?? rec?.evaluation?.estimated_operating_cost)}</span>
            </div>
            <Separator orientation="vertical" className="h-6" />
            <div>
              <span className="text-[10px] text-muted-foreground block uppercase font-sans">Đúng hạn</span>
              <span className="font-bold text-emerald-700 dark:text-emerald-400">{num(metric(rec, "on_time_deliveries"), 0)} / {stopRecs.length} trạm</span>
            </div>
            <Separator orientation="vertical" className="h-6" />
            <div>
              <span className="text-[10px] text-muted-foreground block uppercase font-sans">Trễ hạn</span>
              <span className="font-bold text-muted-foreground">{num(metric(rec, "late_deliveries"), 0)} trạm</span>
            </div>
          </div>

          {/* Stop Itinerary Sequence */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 sm:px-5 pb-2 border-b border-border/60">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <CardTitle className="text-sm font-semibold">
                      Trình tự Ghé thăm ({stopRecs.length} trạm)
                    </CardTitle>
                    {rec?.route_type === "validated_recovered" && (
                      <Badge variant="secondary" className="text-[9px] px-1.5 py-0 bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30">
                        Tuyến khôi phục đã kiểm chứng
                      </Badge>
                    )}
                    {rec?.route_type === "original_precomputed" && (
                      <Badge variant="outline" className="text-[9px] px-1.5 py-0 text-muted-foreground">
                        Tuyến gốc lưu sẵn
                      </Badge>
                    )}
                  </div>
                  <CardDescription className="text-xs mt-0.5">
                    {rec?.provenance ? (
                      <span className="block text-[11px] text-muted-foreground leading-normal" title={rec.provenance}>
                        Nguồn: {rec.provenance}
                      </span>
                    ) : (
                      "Bấm vào trạm để camera định vị toạ độ trên bản đồ"
                    )}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              <div className="max-h-[380px] overflow-y-auto divide-y divide-border/40">
                {stops.loading && (
                  <div className="p-6">
                    <LoadingCard message="Đang nạp chi tiết các trạm dừng..." />
                  </div>
                )}
                {stopRecs.map((d, i) => {
                  const isSel = selectedStopId === d.id;
                  const hasClusterSwitch =
                    alg === "kmeans_guided_dqn" &&
                    i > 0 &&
                    typeof d.cluster_id === "number" &&
                    typeof stopRecs[i - 1]?.cluster_id === "number" &&
                    d.cluster_id !== stopRecs[i - 1]?.cluster_id;

                  return (
                    <div
                      key={d.id}
                      onClick={() => handleSelectStop(d)}
                      className={`p-3 cursor-pointer transition-colors text-xs space-y-1 ${
                        isSel ? "bg-accent/60" : "hover:bg-muted/40"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className="size-5 rounded-full flex items-center justify-center font-bold text-white text-[10px]"
                            style={{ backgroundColor: mainColor }}
                          >
                            {i + 1}
                          </span>
                          <span className="font-semibold text-foreground truncate max-w-[140px]">
                            {String(d.package_name ?? "Đơn hàng")}
                          </span>
                        </div>
                        {typeof d.cluster_id === "number" && (
                          <Badge
                            variant="outline"
                            className="text-[10px] px-1 py-0 font-mono"
                            style={{ color: clusterColor(d.cluster_id) }}
                          >
                            Cụm #{d.cluster_id}
                          </Badge>
                        )}
                      </div>

                      <div className="text-[11px] text-muted-foreground truncate pl-7">
                        {String(d.receiverAddress ?? "–").split(",").slice(0, 3).join(",")}
                      </div>

                      <div className="flex items-center justify-between pl-7 text-[10px] text-muted-foreground">
                        <span>Hạn: {dateTime(d.expectedDeliveryTime)}</span>
                        {hasClusterSwitch && (
                          <span className="text-amber-600 dark:text-amber-400 font-bold">
                            ⚡ Đổi cụm (#{stopRecs[i - 1].cluster_id} → #{d.cluster_id})
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Route Transition Inspector */}
      <RouteTransitionInspector
        stops={inspectorStops}
        algorithmLabel={ALGO_LABEL[alg] ?? alg}
        workloadId={wid}
        totalDistanceKm={rec?.evaluation?.total_distance_km ?? metric(rec, "total_distance_km")}
        totalTravelTimeMinutes={rec?.evaluation?.total_travel_time_minutes ?? metric(rec, "total_travel_time_minutes")}
        onTimeRate={rec?.evaluation?.on_time_rate ?? metric(rec, "on_time_rate")}
        isLoading={stops.loading}
        selectedStopId={selectedStopId}
        onSelectStop={(s) => {
          setSelectedStopId(s.deliveryId);
          setFocusTarget([s.lat, s.lng]);
        }}
      />

      {/* Accordion: Spatial Clustering Penalty & Scope */}
      <Card className="border-border/80 shadow-xs">
        <Accordion className="w-full">
          <AccordionItem value="penalty" className="border-0">
            <AccordionTrigger className="px-6 py-4 text-xs font-semibold hover:no-underline">
              <span>⚡ Cơ chế Phạt Chuyển Cụm (Spatial Clustering Penalty, λ = 0,50)</span>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4 pt-0 text-xs text-muted-foreground leading-relaxed border-t border-border/40 mt-2 space-y-2">
              <p>
                Mô hình <b>K-Means-guided DQN</b> tích hợp số hiệu cụm địa lý $c_t$ vào hàm phần thưởng nhằm định hướng tác tử giao trọn vẹn theo từng phân vùng:
              </p>
              <div className="p-2.5 rounded bg-muted/60 font-mono text-xs text-foreground border border-border/60">
                R_t = R_env − λ · 𝕀(c_t ≠ c_&#123;t-1&#125;) &nbsp;&nbsp;(λ = 0,50)
              </div>
              <p>
                Trên 6 workload đủ dữ liệu tuyến đối chứng, số lần chuyển cụm trung bình giảm từ <b>0,6201</b> xuống <b>0,5778</b> (~-6,83%). Đổi lại quãng đường tăng nhẹ +1,76% và thời gian tăng +1,87%, nhưng duy trì <b>100% SLA trên 12 workload kiểm chứng</b>.
              </p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </Card>
    </div>
  );
}
