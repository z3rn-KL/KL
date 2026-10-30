import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
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
import { Tooltip as UiTooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
  PageHeader,
  LoadingCard,
  ErrorCard,
  ResearchCallout,
} from "@/components/shared/DomainComponents";
import {
  Clock,
  CheckCircle2,
  TrendingDown,
  Layers,
  Network,
  ShieldCheck,
} from "lucide-react";
import { AlgorithmSelector } from "@/components/shared/AlgorithmSelector";
import {
  api,
  type AlgorithmKey,
  type AlgorithmSummary,
} from "../lib/api";
import {
  ALGO_COLOR,
  ALGO_LABEL,
  num,
  pct,
  vnd,
} from "../lib/format";
import { useAsync } from "../lib/useAsync";

type ExtendedAlgoKey = AlgorithmKey | "kmeans_guided_dqn" | "proposed_rl";

const LOCAL_ALGO_LABEL: Record<string, string> = {
  ...ALGO_LABEL,
  kmeans_guided_dqn: "K-Means-guided DQN",
  proposed_rl: "Proposed RL",
};

const LOCAL_ALGO_COLOR: Record<string, string> = {
  ...ALGO_COLOR,
  kmeans_guided_dqn: "var(--algo-guided)",
  proposed_rl: "var(--accent)",
};

const PRIMARY_ALGOS: ExtendedAlgoKey[] = [
  "nearest_neighbor",
  "clarke_wright",
  "dqn",
  "kmeans_guided_dqn",
  "proposed_rl",
];

const ALL_ALGOS: ExtendedAlgoKey[] = [
  "nearest_neighbor",
  "clarke_wright",
  "q_learning",
  "sarsa",
  "dqn",
  "kmeans_guided_dqn",
  "proposed_rl",
];

interface DisplayRow {
  key: ExtendedAlgoKey;
  name: string;
  color: string;
  sourceScope: string;
  workloads: number;
  deliveries: number;
  mean_distance_km: number;
  std_distance_km: number;
  mean_travel_time_minutes: number;
  weighted_on_time_rate: number;
  total_lateness_minutes: number;
  mean_cluster_switches?: number | null;
  route_runtime_seconds: number | null;
  training_runtime_seconds: number | null;
  cost: number | null;
  isProposed?: boolean;
}

// 12-workload exact benchmark stats (108 deliveries)
const BENCHMARK_12_WORKLOADS: Record<ExtendedAlgoKey, Partial<DisplayRow>> = {
  clarke_wright: {
    workloads: 12,
    deliveries: 108,
    mean_distance_km: 70.55,
    std_distance_km: 29.21,
    mean_travel_time_minutes: 76.63,
    weighted_on_time_rate: 0.963,
    total_lateness_minutes: 118.65,
    mean_cluster_switches: null,
    route_runtime_seconds: 4.0543,
    training_runtime_seconds: null,
  },
  dqn: {
    workloads: 12,
    deliveries: 108,
    mean_distance_km: 72.07,
    std_distance_km: 28.80,
    mean_travel_time_minutes: 78.70,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: null,
    route_runtime_seconds: 0.0027,
    training_runtime_seconds: 203.36,
  },
  nearest_neighbor: {
    workloads: 12,
    deliveries: 108,
    mean_distance_km: 77.60,
    std_distance_km: 31.94,
    mean_travel_time_minutes: 84.71,
    weighted_on_time_rate: 0.944,
    total_lateness_minutes: 192.02,
    mean_cluster_switches: null,
    route_runtime_seconds: 4.0406,
    training_runtime_seconds: null,
  },
  q_learning: {
    workloads: 12,
    deliveries: 108,
    mean_distance_km: 77.12,
    std_distance_km: 32.13,
    mean_travel_time_minutes: 84.13,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: null,
    route_runtime_seconds: 0.0006,
    training_runtime_seconds: 3.35,
  },
  sarsa: {
    workloads: 12,
    deliveries: 108,
    mean_distance_km: 79.67,
    std_distance_km: 34.81,
    mean_travel_time_minutes: 86.81,
    weighted_on_time_rate: 0.981,
    total_lateness_minutes: 113.05,
    mean_cluster_switches: null,
    route_runtime_seconds: 0.0006,
    training_runtime_seconds: 3.11,
  },
  kmeans_guided_dqn: {},
  proposed_rl: {},
};

// 16-workload Held-out Test stats (131 deliveries)
const BENCHMARK_16_TEST_WORKLOADS: Record<ExtendedAlgoKey, Partial<DisplayRow>> = {
  clarke_wright: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 55.58,
    std_distance_km: 24.12,
    mean_travel_time_minutes: 60.74,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: 4.25,
    route_runtime_seconds: 4.02,
    training_runtime_seconds: null,
  },
  dqn: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 57.11,
    std_distance_km: 23.95,
    mean_travel_time_minutes: 62.33,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: 4.13,
    route_runtime_seconds: 0.0025,
    training_runtime_seconds: 185.0,
  },
  nearest_neighbor: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 59.58,
    std_distance_km: 25.40,
    mean_travel_time_minutes: 64.95,
    weighted_on_time_rate: 0.9924,
    total_lateness_minutes: 10.46,
    mean_cluster_switches: 4.06,
    route_runtime_seconds: 4.01,
    training_runtime_seconds: null,
  },
  q_learning: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 60.04,
    std_distance_km: 26.10,
    mean_travel_time_minutes: 65.43,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: 4.19,
    route_runtime_seconds: 0.0006,
    training_runtime_seconds: 3.2,
  },
  sarsa: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 60.81,
    std_distance_km: 26.85,
    mean_travel_time_minutes: 66.21,
    weighted_on_time_rate: 1.0,
    total_lateness_minutes: 0,
    mean_cluster_switches: 4.19,
    route_runtime_seconds: 0.0006,
    training_runtime_seconds: 3.0,
  },
  kmeans_guided_dqn: {},
  proposed_rl: {
    workloads: 16,
    deliveries: 131,
    mean_distance_km: 61.90,
    std_distance_km: 25.80,
    mean_travel_time_minutes: 67.52,
    weighted_on_time_rate: 0.9924,
    total_lateness_minutes: 12.72,
    mean_cluster_switches: 4.00,
    route_runtime_seconds: 0.0938,
    training_runtime_seconds: 312.0,
    isProposed: true,
  },
};

export default function Compare() {
  const sum = useAsync((s) => api.benchmarkSummary(s), []);
  const val = useAsync((s) => api.finalValidation(s), []);
  const real = useAsync((s) => api.realData(s), []);

  const [mode, setMode] = useState<"primary" | "all">("primary");
  const [datasetScope, setDatasetScope] = useState<"same12" | "full108" | "proposed16">("full108");
  const [selectedAlgo, setSelectedAlgo] = useState<ExtendedAlgoKey | null>(null);
  const [cost, setCost] = useState<Partial<Record<ExtendedAlgoKey, number>>>({});
  const [costUnit, setCostUnit] = useState<string>("");
  const [costErr, setCostErr] = useState<string | null>(null);

  // Compute guided row from final validation
  const guidedSummary = val.data?.summary;
  const guidedRow = useMemo<DisplayRow | null>(() => {
    if (!guidedSummary) return null;
    return {
      key: "kmeans_guided_dqn",
      name: "K-Means-guided DQN",
      color: "var(--algo-guided)",
      sourceScope: "12 WL (Kiểm chứng)",
      workloads: Number(guidedSummary.workloads ?? 12),
      deliveries: Number(guidedSummary.deliveries ?? 108),
      mean_distance_km: Number(guidedSummary.mean_guided_distance_km ?? 73.34),
      std_distance_km: 28.85,
      mean_travel_time_minutes: Number(guidedSummary.mean_guided_travel_time_minutes ?? 80.17),
      weighted_on_time_rate: Number(guidedSummary.guided_weighted_on_time_rate ?? 1.0),
      total_lateness_minutes: 0,
      mean_cluster_switches: 0.5778,
      route_runtime_seconds: 0.004,
      training_runtime_seconds: Number(guidedSummary.mean_guided_training_runtime_seconds ?? 231.47),
      cost: cost.kmeans_guided_dqn ?? null,
      isProposed: true,
    };
  }, [guidedSummary, cost.kmeans_guided_dqn]);

  // Build combined rows based on active dataset scope
  const allRows = useMemo<DisplayRow[]>(() => {
    if (datasetScope === "proposed16") {
      const algos = mode === "primary"
        ? (["clarke_wright", "dqn", "nearest_neighbor", "proposed_rl"] as ExtendedAlgoKey[])
        : (["clarke_wright", "dqn", "nearest_neighbor", "q_learning", "sarsa", "proposed_rl"] as ExtendedAlgoKey[]);

      return algos.map((k) => {
        const item = BENCHMARK_16_TEST_WORKLOADS[k] ?? {};
        return {
          key: k,
          name: LOCAL_ALGO_LABEL[k] ?? k,
          color: LOCAL_ALGO_COLOR[k] ?? "#8EA3BC",
          sourceScope: "16 WL Held-out Test",
          workloads: item.workloads ?? 16,
          deliveries: item.deliveries ?? 131,
          mean_distance_km: item.mean_distance_km ?? 60.0,
          std_distance_km: item.std_distance_km ?? 25.0,
          mean_travel_time_minutes: item.mean_travel_time_minutes ?? 65.0,
          weighted_on_time_rate: item.weighted_on_time_rate ?? 1.0,
          total_lateness_minutes: item.total_lateness_minutes ?? 0,
          mean_cluster_switches: item.mean_cluster_switches ?? null,
          route_runtime_seconds: item.route_runtime_seconds ?? null,
          training_runtime_seconds: item.training_runtime_seconds ?? null,
          cost: cost[k] ?? null,
          isProposed: k === "proposed_rl",
        };
      });
    }

    if (!sum.data?.items) return [];

    let base: DisplayRow[];
    if (datasetScope === "same12") {
      base = (sum.data.items as AlgorithmSummary[]).map((i) => {
        const d12 = BENCHMARK_12_WORKLOADS[i.algorithm] ?? {};
        return {
          key: i.algorithm,
          name: LOCAL_ALGO_LABEL[i.algorithm] ?? i.algorithm,
          color: LOCAL_ALGO_COLOR[i.algorithm] ?? "#8EA3BC",
          sourceScope: "Cùng 12 WL (Đối sánh chuẩn)",
          workloads: 12,
          deliveries: 108,
          mean_distance_km: d12.mean_distance_km ?? i.mean_distance_km,
          std_distance_km: d12.std_distance_km ?? i.std_distance_km,
          mean_travel_time_minutes: d12.mean_travel_time_minutes ?? i.mean_travel_time_minutes,
          weighted_on_time_rate: d12.weighted_on_time_rate ?? i.weighted_on_time_rate,
          total_lateness_minutes: d12.total_lateness_minutes ?? i.total_lateness_minutes,
          mean_cluster_switches: d12.mean_cluster_switches ?? null,
          route_runtime_seconds: d12.route_runtime_seconds ?? i.mean_route_runtime_seconds,
          training_runtime_seconds: d12.training_runtime_seconds ?? i.mean_training_runtime_seconds,
          cost: cost[i.algorithm] ?? null,
        };
      });
      if (guidedRow) {
        base.push(guidedRow);
      }
    } else {
      base = sum.data.items.map((i: AlgorithmSummary) => ({
        key: i.algorithm,
        name: LOCAL_ALGO_LABEL[i.algorithm] ?? i.algorithm,
        color: LOCAL_ALGO_COLOR[i.algorithm] ?? "#8EA3BC",
        sourceScope: "108 WL (Toàn bộ dataset)",
        workloads: i.workloads,
        deliveries: i.deliveries,
        mean_distance_km: i.mean_distance_km,
        std_distance_km: i.std_distance_km,
        mean_travel_time_minutes: i.mean_travel_time_minutes,
        weighted_on_time_rate: i.weighted_on_time_rate,
        total_lateness_minutes: i.total_lateness_minutes,
        mean_cluster_switches: null,
        route_runtime_seconds: i.mean_route_runtime_seconds,
        training_runtime_seconds: i.mean_training_runtime_seconds,
        cost: cost[i.algorithm] ?? null,
      }));
      // Trên tập 108 WL Benchmark cơ sở, loại trừ hoàn toàn dữ liệu 12 WL / 16 WL để tránh xếp hạng cọc lệch mẫu
    }

    return base;
  }, [sum.data, guidedRow, cost, datasetScope, mode]);

  const activeKeys = useMemo<ExtendedAlgoKey[]>(() => {
    if (datasetScope === "full108") {
      return mode === "primary"
        ? ["clarke_wright", "dqn", "nearest_neighbor"]
        : ["clarke_wright", "dqn", "nearest_neighbor", "q_learning", "sarsa"];
    }
    if (datasetScope === "same12") {
      return mode === "primary"
        ? ["clarke_wright", "dqn", "nearest_neighbor", "kmeans_guided_dqn"]
        : ["clarke_wright", "dqn", "nearest_neighbor", "q_learning", "sarsa", "kmeans_guided_dqn"];
    }
    // proposed16
    return mode === "primary"
      ? ["clarke_wright", "dqn", "nearest_neighbor", "proposed_rl"]
      : ["clarke_wright", "dqn", "nearest_neighbor", "q_learning", "sarsa", "proposed_rl"];
  }, [datasetScope, mode]);

  const rows = useMemo(() => {
    return activeKeys
      .map((k) => allRows.find((r) => r.key === k))
      .filter((x): x is DisplayRow => !!x);
  }, [activeKeys, allRows]);

  // Compute operating costs via API
  useEffect(() => {
    if (!rows.length) return;
    const ctrl = new AbortController();
    Promise.all(
      rows.map((r) =>
        api
          .cost(
            {
              total_distance_km: r.mean_distance_km,
              total_travel_time_minutes: r.mean_travel_time_minutes,
            },
            ctrl.signal,
          )
          .then((c) => [r.key, c] as const),
      ),
    )
      .then((list) => {
        setCost(Object.fromEntries(list.map(([a, c]) => [a, c.result.total_estimated_cost])));
        setCostUnit(list[0]?.[1].result.cost_unit ?? "");
        setCostErr(null);
      })
      .catch((e: Error) => {
        if (e.name !== "AbortError") setCostErr(e.message);
      });
    return () => ctrl.abort();
  }, [rows.length, datasetScope]);

  // Eligibility filter:
  // In full108 mode, Guided DQN is strictly excluded from 108 WL Best-in-Class
  const eligibleRows = useMemo(() => {
    return datasetScope === "full108"
      ? rows.filter((r) => r.key !== "kmeans_guided_dqn")
      : rows;
  }, [rows, datasetScope]);

  // Best metrics derived strictly from eligibleRows
  const costValues = eligibleRows
    .map((r) => r.cost)
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const minCost = costValues.length ? Math.min(...costValues) : NaN;
  const minDist = eligibleRows.length ? Math.min(...eligibleRows.map((r) => r.mean_distance_km)) : 0;
  const minTime = eligibleRows.length ? Math.min(...eligibleRows.map((r) => r.mean_travel_time_minutes)) : 0;
  const maxOn = eligibleRows.length ? Math.max(...eligibleRows.map((r) => r.weighted_on_time_rate)) : 0;

  const bestDistAlgos = eligibleRows.filter((r) => Math.abs(r.mean_distance_km - minDist) < 1e-4);
  const bestTimeAlgos = eligibleRows.filter((r) => Math.abs(r.mean_travel_time_minutes - minTime) < 1e-4);
  const bestCostAlgos = eligibleRows.filter((r) => r.cost !== null && Math.abs(r.cost - minCost) < 1e-4);
  const bestOnAlgos = eligibleRows.filter((r) => Math.abs(r.weighted_on_time_rate - maxOn) < 1e-4);

  // Pairwise with Baseline
  const nnRow = allRows.find((r) => r.key === "nearest_neighbor");
  const pctVsNN = useMemo(() => {
    if (!nnRow) return [];
    return eligibleRows.map((r) => {
      const diffDistPct = ((nnRow.mean_distance_km - r.mean_distance_km) / nnRow.mean_distance_km) * 100;
      const diffTimePct = ((nnRow.mean_travel_time_minutes - r.mean_travel_time_minutes) / nnRow.mean_travel_time_minutes) * 100;
      return {
        key: r.key,
        name: r.name,
        color: r.color,
        diffDistPct: Number(diffDistPct.toFixed(2)),
        diffTimePct: Number(diffTimePct.toFixed(2)),
      };
    });
  }, [nnRow, eligibleRows]);

  const bestDistName = bestDistAlgos.map((a) => a.name).join(", ");
  const bestTimeName = bestTimeAlgos.map((a) => a.name).join(", ");

  // Benchmark data: true 0-100 normalization
  const radar = useMemo(() => {
    if (!eligibleRows.length) return null;
    const axes = [
      {
        axis: "Quãng đường (ngắn hơn)",
        get: (r: DisplayRow) => (Number.isFinite(r.mean_distance_km) ? r.mean_distance_km : null),
        lowerBetter: true,
      },
      {
        axis: "Thời gian đi (nhanh hơn)",
        get: (r: DisplayRow) => (Number.isFinite(r.mean_travel_time_minutes) ? r.mean_travel_time_minutes : null),
        lowerBetter: true,
      },
      {
        axis: "Chi phí mô phỏng (tiết kiệm)",
        get: (r: DisplayRow) => (r.cost !== null && Number.isFinite(r.cost) ? r.cost : null),
        lowerBetter: true,
      },
      {
        axis: "Đúng hạn SLA",
        get: (r: DisplayRow) => (Number.isFinite(r.weighted_on_time_rate) ? r.weighted_on_time_rate : null),
        lowerBetter: false,
      },
      {
        axis: "Tốc độ sinh tuyến",
        get: (r: DisplayRow) =>
          r.route_runtime_seconds !== null && Number.isFinite(r.route_runtime_seconds)
            ? r.route_runtime_seconds
            : null,
        lowerBetter: true,
        log: true,
      },
      {
        axis: "Độ ổn định (std km)",
        get: (r: DisplayRow) => (Number.isFinite(r.std_distance_km) ? r.std_distance_km : null),
        lowerBetter: true,
      },
    ];

    return axes.map((a) => {
      const values = eligibleRows.map((r) => {
        const v = a.get(r);
        if (v === null || !Number.isFinite(v)) return null;
        return a.log ? Math.log10(Math.max(v, 1e-6)) : v;
      });

      const validNums = values.filter((v): v is number => v !== null && Number.isFinite(v));
      const row: Record<string, number | string | null> = { axis: a.axis };

      if (validNums.length === 0) {
        eligibleRows.forEach((r) => {
          row[r.key] = null;
        });
        return row;
      }

      const lo = Math.min(...validNums);
      const hi = Math.max(...validNums);

      eligibleRows.forEach((r, i) => {
        const val = values[i];
        if (val === null) {
          row[r.key] = null;
          return;
        }

        if (Math.abs(hi - lo) < 1e-9) {
          row[r.key] = 100;
          return;
        }

        const score = a.lowerBetter
          ? ((hi - val) / (hi - lo)) * 100
          : ((val - lo) / (hi - lo)) * 100;

        const clamped = Math.max(0, Math.min(100, score));
        row[r.key] = Number(clamped.toFixed(1));
      });

      return row;
    });
  }, [eligibleRows]);

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="So sánh thuật toán định tuyến"
        subtitle="Đối chuẩn thực nghiệm giữa Heuristic cổ điển và Học tăng cường theo từng tập kiểm thử độc lập"
      />

      <ResearchCallout title="Tỷ lệ đúng hạn ở đây là kết quả tính theo mô hình, không phải thực tế" type="warning">
        Các con số SLA trong trang này được tính trên đồ thị đường phố (tốc độ lý tưởng, không tính thời gian phục vụ tại điểm giao).
        {real.data
          ? ` Thực tế quan sát trong dữ liệu XeDu chỉ có ${pct(1 - real.data.observed.late_rate, 1)} đơn giao đúng hạn.`
          : ""}{" "}
        Xem trang <Link to="/du-lieu-that" className="underline font-medium">Dữ liệu thật</Link>.
      </ResearchCallout>

      <div className="stack">
        {/* ─── Scope Switcher & Algorithm Filter Toolbar ─── */}
        <Card className="border-border/80 shadow-xs mb-3">
          <CardContent className="p-3 sm:px-4 space-y-2">
            <div className="flex flex-col gap-2.5 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex items-center gap-3 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                    Tập dữ liệu:
                  </span>
                  <Tabs
                    value={datasetScope}
                    onValueChange={(s) => setDatasetScope(s as "same12" | "full108" | "proposed16")}
                  >
                    <TabsList className="h-8">
                      <TabsTrigger value="same12" className="text-xs">12 WL (Kiểm chứng Paired)</TabsTrigger>
                      <TabsTrigger value="full108" className="text-xs">108 WL (Cơ sở Benchmark)</TabsTrigger>
                      <TabsTrigger value="proposed16" className="text-xs">16 WL Test (Proposed RL)</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                <Separator orientation="vertical" className="h-4 hidden sm:block" />

                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                    Nhóm:
                  </span>
                  <Tabs value={mode} onValueChange={(m) => setMode(m as "primary" | "all")}>
                    <TabsList className="h-8">
                      <TabsTrigger value="primary" className="text-xs">Thuật toán cốt lõi</TabsTrigger>
                      <TabsTrigger value="all" className="text-xs">Đầy đủ (+ QL & SARSA)</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
              </div>

              {selectedAlgo && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedAlgo(null)}
                  className="h-8 text-xs gap-1.5 self-start xl:self-auto font-medium"
                  style={{
                    borderColor: LOCAL_ALGO_COLOR[selectedAlgo],
                    color: LOCAL_ALGO_COLOR[selectedAlgo],
                  }}
                >
                  <span>Lọc: <b>{LOCAL_ALGO_LABEL[selectedAlgo]}</b></span>
                  <span>✕</span>
                </Button>
              )}
            </div>

            {/* Algorithm Chips Pill Strip with clear disabled state and tooltips */}
            <div className="flex items-center gap-2.5 flex-wrap pt-2.5 border-t border-border/50">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                Lọc thuật toán:
              </span>
              <AlgorithmSelector
                algorithms={
                  datasetScope === "full108"
                    ? (mode === "primary" ? ["clarke_wright", "dqn", "nearest_neighbor"] : ["clarke_wright", "dqn", "nearest_neighbor", "q_learning", "sarsa"])
                    : (mode === "primary" ? PRIMARY_ALGOS : ALL_ALGOS)
                }
                selected={selectedAlgo}
                onSelect={(k) => setSelectedAlgo(k as ExtendedAlgoKey | null)}
                isAvailable={(k) => {
                  if (datasetScope === "full108") {
                    return k !== "kmeans_guided_dqn" && k !== "proposed_rl";
                  }
                  if (datasetScope === "same12") {
                    return k !== "proposed_rl";
                  }
                  if (datasetScope === "proposed16") {
                    return k !== "kmeans_guided_dqn";
                  }
                  return true;
                }}
                unavailableReason={(k) => {
                  if (datasetScope === "full108" && (k === "kmeans_guided_dqn" || k === "proposed_rl")) {
                    return `Thuật toán ${LOCAL_ALGO_LABEL[k]} không thuộc tập 108 Workload cơ sở. Dữ liệu đối chuẩn khả dụng trên tập 12 WL hoặc 16 WL Test.`;
                  }
                  if (datasetScope === "same12" && k === "proposed_rl") {
                    return `Proposed RL được đánh giá trên tập 16 Workload Test độc lập.`;
                  }
                  if (datasetScope === "proposed16" && k === "kmeans_guided_dqn") {
                    return `K-Means-guided DQN được đánh giá trên tập kiểm chứng 12 Workload.`;
                  }
                  return "Thuật toán không khả dụng trong phạm vi dữ liệu hiện tại.";
                }}
                badge={(k) => {
                  if (datasetScope === "full108" && (k === "kmeans_guided_dqn" || k === "proposed_rl")) {
                    return k === "kmeans_guided_dqn" ? "12 WL" : "16 WL";
                  }
                  if (datasetScope === "same12" && k === "proposed_rl") return "16 WL";
                  if (datasetScope === "proposed16" && k === "kmeans_guided_dqn") return "12 WL";
                  return undefined;
                }}
                colors={LOCAL_ALGO_COLOR}
                labels={LOCAL_ALGO_LABEL}
              />
            </div>
          </CardContent>
        </Card>

        {/* Banner phạm vi nghiên cứu súc tích */}
        {datasetScope === "full108" && (
          <div className="rounded-lg border border-border/70 bg-card/60 p-3.5 shadow-2xs mb-4 text-xs flex items-center gap-3 border-l-4 border-l-sky-500">
            <span className="text-base shrink-0">ℹ️</span>
            <div className="leading-relaxed text-muted-foreground">
              <b className="text-foreground">Phạm vi 108 Workload Benchmark Cơ sở:</b> Đối chuẩn 5 thuật toán cơ sở (Clarke–Wright, DQN, Q-Learning, SARSA, Nearest Neighbor) trên toàn bộ 108 đợt giao (813 đơn). Bảng và biểu đồ hiển thị thuần nhất 5 thuật toán được đo lường thực tế trên 108 Workload, không gộp lẫn các tập kiểm chứng 12 WL hay 16 WL Test.
            </div>
          </div>
        )}

        {datasetScope === "proposed16" && (
          <div className="rounded-lg border border-border/70 bg-card/60 p-3.5 shadow-2xs mb-4 text-xs flex items-center gap-3 border-l-4 border-l-amber-500">
            <span className="text-base shrink-0">🎯</span>
            <div className="leading-relaxed text-muted-foreground">
              <b className="text-foreground">Phạm vi 16 Workload Held-out Test (Proposed RL):</b> Proposed RL tối ưu hàm mục tiêu đa tiêu chí khóa <code className="font-mono text-foreground font-semibold">J(π)</code>, đạt số lần chuyển cụm trung bình thấp nhất (<b className="text-foreground">4,00 lần</b>) trong các thuật toán được so sánh trên 16 workload Test; mô hình chưa vượt Clarke–Wright (55,58 km) và DQN (57,11 km) về cự ly.
            </div>
          </div>
        )}

        {datasetScope === "same12" && (
          <div className="rounded-lg border border-border/70 bg-card/60 p-3.5 shadow-2xs mb-4 text-xs flex items-center gap-3 border-l-4 border-l-indigo-500">
            <span className="text-base shrink-0">🔬</span>
            <div className="leading-relaxed text-muted-foreground">
              <b className="text-foreground">Phạm vi 12 Workload Paired Validation (Guided DQN vs Vanilla DQN):</b> Đối chuẩn công bằng trên cùng tập 12 workload (108 đơn). Guided DQN bảo toàn <b className="text-foreground">100% SLA đúng hạn</b> và giảm <b className="text-foreground">-6,83%</b> tỷ lệ đổi cụm liên vùng, đánh đổi mức cự ly nhỏ (+1,76%) và thời gian (+1,87%).
            </div>
          </div>
        )}

        {(sum.loading || val.loading) && <LoadingCard message="Đang nạp bảng đối chuẩn thuật toán..." />}
        {sum.error && <ErrorCard message={sum.error} />}
        {val.error && <ErrorCard message={val.error} />}

        {rows.length > 0 && (
          <>
            {/* ─── 1. Biểu đồ Cột Nhóm: Quãng đường & Thời gian ─── */}
            <div className="grid cols-2">
              <Card
                title="Quãng đường trung bình (km)"
                sub={`${bestDistName} ngắn nhất (${num(minDist, 2)} km).`}
              >
                <div style={{ height: 250 }} className="recharts-responsive-container">
                  <ResponsiveContainer>
                    <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 6, left: 0 }}>
                      <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="key"
                        tickFormatter={(k) => LOCAL_ALGO_LABEL[k] ?? k}
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} unit=" km" />
                      <Tooltip formatter={(v: any) => [`${num(v, 2)} km`, "Quãng đường TB"]} />
                      <Bar dataKey="mean_distance_km" radius={[4, 4, 0, 0]}>
                        {rows.map((r) => (
                          <Cell
                            key={r.key}
                            fill={r.color}
                            opacity={selectedAlgo === null || selectedAlgo === r.key ? 1 : 0.25}
                            cursor="pointer"
                            onClick={() => setSelectedAlgo(selectedAlgo === r.key ? null : r.key)}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card
                title="Thời gian di chuyển trung bình (phút)"
                sub={`${bestTimeName} nhanh nhất (${num(minTime, 2)} phút).`}
              >
                <div style={{ height: 250 }} className="recharts-responsive-container">
                  <ResponsiveContainer>
                    <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 6, left: 0 }}>
                      <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="key"
                        tickFormatter={(k) => LOCAL_ALGO_LABEL[k] ?? k}
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} unit=" ph" />
                      <Tooltip formatter={(v: any) => [`${num(v, 2)} phút`, "Thời gian TB"]} />
                      <Bar dataKey="mean_travel_time_minutes" radius={[4, 4, 0, 0]}>
                        {rows.map((r) => (
                          <Cell
                            key={r.key}
                            fill={r.color}
                            opacity={selectedAlgo === null || selectedAlgo === r.key ? 1 : 0.25}
                            cursor="pointer"
                            onClick={() => setSelectedAlgo(selectedAlgo === r.key ? null : r.key)}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            {/* ─── 2. Biểu đồ Chênh lệch % so với Nearest Neighbor (SLA & Tiết kiệm) ─── */}
            <Card
              title="Mức chênh lệch % so với Nearest Neighbor (Baseline)"
              sub="Dương (+) = tối ưu tốt hơn NN (ngắn hơn / nhanh hơn). Giá trị tính trực tiếp từ dữ liệu đối chuẩn."
            >
              <div style={{ height: 240 }} className="recharts-responsive-container">
                <ResponsiveContainer>
                  <BarChart
                    data={pctVsNN.filter((p) => p.key !== "nearest_neighbor")}
                    margin={{ top: 8, right: 12, bottom: 6, left: 0 }}
                  >
                    <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 12 }} unit="%" />
                    <Tooltip
                      formatter={(v: any, name: any) => [
                        `${v > 0 ? "+" : ""}${num(v, 2)}%`,
                        name === "diffDistPct" ? "Quãng đường vs NN" : "Thời gian vs NN",
                      ]}
                    />
                    <Legend />
                    <Bar dataKey="diffDistPct" name="Tiết kiệm cự ly (%)" fill="var(--algo-cw)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="diffTimePct" name="Tiết kiệm thời gian (%)" fill="var(--algo-vanilla)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* ─── 3. Bảng Đối chuẩn Chi tiết Toàn diện (table.tbl) ─── */}
            <Card
              title={
                datasetScope === "proposed16"
                  ? "Bảng đối chuẩn chi tiết 16 Workload Test (Proposed RL vs Baselines)"
                  : datasetScope === "same12"
                  ? "Bảng đối chuẩn chi tiết 12 Workload Paired (Guided DQN vs Vanilla DQN)"
                  : "Bảng đối chuẩn chi tiết 108 Workload Benchmark Toàn diện"
              }
              sub="Bấm chọn hàng để làm nổi bật thuật toán tương ứng trên toàn bộ biểu đồ bên trên."
            >
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Thuật toán</th>
                      <th className="num">Cự ly TB (km)</th>
                      <th className="num">Std (km)</th>
                      <th className="num">Thời gian (ph)</th>
                      <th className="num">Chi phí mô phỏng (₫)*</th>
                      <th className="num">Đúng hạn (SLA)</th>
                      {datasetScope !== "full108" && (
                        <th className="num">Đổi cụm TB (lần)</th>
                      )}
                      <th className="num">Sinh tuyến (s)</th>
                      <th className="num">Huấn luyện (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => {
                      const isEligible = datasetScope !== "full108" || r.key !== "kmeans_guided_dqn";
                      const isBestDist = isEligible && Math.abs(r.mean_distance_km - minDist) < 1e-4;
                      const isBestTime = isEligible && Math.abs(r.mean_travel_time_minutes - minTime) < 1e-4;
                      const isBestCost = isEligible && r.cost !== null && Math.abs(r.cost - minCost) < 1e-4;
                      const isBestOn = isEligible && Math.abs(r.weighted_on_time_rate - maxOn) < 1e-4;
                      const isSelected = selectedAlgo === r.key;
                      const isDimmed = selectedAlgo !== null && !isSelected;

                      return (
                        <tr
                          key={r.key}
                          onClick={() => setSelectedAlgo(selectedAlgo === r.key ? null : r.key)}
                          style={{
                            cursor: "pointer",
                            background: isSelected
                              ? "var(--accent-soft)"
                              : r.isProposed
                              ? "rgba(139, 124, 255, 0.05)"
                              : undefined,
                            opacity: isDimmed ? 0.35 : 1,
                            transition: "all 0.15s ease",
                          }}
                          title="Bấm để chọn/bỏ chọn làm nổi bật"
                        >
                          <td style={{ fontWeight: r.isProposed ? 700 : 500 }}>
                            <span className="swatch" style={{ background: r.color }} />
                            {r.name}
                            {r.isProposed && (
                              <span
                                className="chip ai"
                                style={{
                                  marginLeft: 6,
                                  fontSize: 9.5,
                                  padding: "1px 5px",
                                }}
                              >
                                {datasetScope === "full108" ? "Tham chiếu 12 WL" : "Đề xuất"}
                              </span>
                            )}
                          </td>
                          <td className={`num ${isBestDist ? "best" : ""}`}>
                            {num(r.mean_distance_km, 2)}
                          </td>
                          <td className="num">{num(r.std_distance_km, 2)}</td>
                          <td className={`num ${isBestTime ? "best" : ""}`}>
                            {num(r.mean_travel_time_minutes, 2)}
                          </td>
                          <td className={`num ${isBestCost ? "best" : ""}`}>
                            {r.cost !== null ? vnd(r.cost) : "–"}
                          </td>
                          <td className={`num ${isBestOn ? "best" : ""}`}>
                            {pct(r.weighted_on_time_rate, 2)}
                          </td>
                          {datasetScope !== "full108" && (
                            <td
                              className={`num ${
                                r.mean_cluster_switches != null && r.mean_cluster_switches <= 4.001
                                  ? "best"
                                  : ""
                              }`}
                            >
                              {r.mean_cluster_switches != null
                                ? num(r.mean_cluster_switches, 2)
                                : "–"}
                            </td>
                          )}
                          <td className="num">{num(r.route_runtime_seconds, 4)}</td>
                          <td className="num">
                            {r.training_runtime_seconds
                              ? num(r.training_runtime_seconds, 1)
                              : "–"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="muted small" style={{ marginTop: 8 }}>
                * Chi phí mô phỏng qua <code>/api/evaluation/cost</code>{costUnit ? ` (${costUnit})` : ""}. Nhấp hàng để lọc biểu đồ.
              </div>
              {costErr && <ErrorCard message={`Chưa tính được chi phí: ${costErr}`} />}
            </Card>

            {/* ─── Benchmark Đa Tiêu chí Chuẩn hóa 0–100 ─── */}
            {radar && (
              <Card
                title="Hồ sơ Năng lực Đa tiêu chí (Benchmark Chuẩn hoá 0–100)"
                sub="Thang điểm 0–100 giữa các phương pháp trên cùng tập dữ liệu (100 = tối ưu nhất ở tiêu chí đó)."
              >
                <div style={{ display: "grid", gap: 12 }}>
                  {radar.map((criterion) => {
                    const label = String(criterion.axis).replace(/\n/g, " ");
                    return (
                      <div
                        key={label}
                        className="rounded-lg border border-border/60 bg-card/60 p-3 shadow-2xs mb-2.5"
                      >
                        <div
                          style={{
                            fontSize: 12.5,
                            fontWeight: 700,
                            color: "var(--text)",
                            marginBottom: 6,
                            display: "flex",
                            justifyContent: "space-between",
                          }}
                        >
                          <span>{label}</span>
                          <span style={{ fontSize: 10.5, color: "var(--muted)", fontWeight: 500 }}>
                            Điểm chuẩn hoá (0–100)
                          </span>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                          {eligibleRows.map((r) => {
                            const rawScore = criterion[r.key];
                            const score = typeof rawScore === "number" ? rawScore : null;
                            const isBest = score !== null && score >= 99.9;
                            const isSelected = selectedAlgo === r.key;
                            const isDimmed = selectedAlgo !== null && !isSelected;
                            return (
                              <div
                                key={r.key}
                                onClick={() => setSelectedAlgo(selectedAlgo === r.key ? null : r.key)}
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: 8,
                                  cursor: "pointer",
                                  opacity: isDimmed ? 0.35 : 1,
                                  transition: "all 0.15s ease",
                                }}
                                title="Bấm để làm nổi bật phương pháp này"
                              >
                                <span
                                  style={{
                                    width: 140,
                                    fontSize: 11.5,
                                    color: isSelected ? r.color : "var(--text)",
                                    fontWeight: isSelected || r.isProposed ? 700 : 500,
                                    textAlign: "right",
                                    flexShrink: 0,
                                    whiteSpace: "nowrap",
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                  }}
                                >
                                  {r.name}
                                </span>
                                <div
                                  style={{
                                    flex: 1,
                                    height: 8,
                                    background: "var(--track)",
                                    borderRadius: 9999,
                                    overflow: "hidden",
                                    position: "relative",
                                    boxShadow: isSelected ? `0 0 0 1.5px ${r.color}` : undefined,
                                  }}
                                >
                                  {score !== null ? (
                                    <div
                                      style={{
                                        width: `${score}%`,
                                        height: "100%",
                                        background: r.color,
                                        borderRadius: 9999,
                                        opacity: isBest ? 1 : 0.85,
                                        transition: "width 0.35s ease",
                                      }}
                                    />
                                  ) : null}
                                </div>
                                <span
                                  style={{
                                    width: 40,
                                    fontSize: 11.5,
                                    fontWeight: isBest ? 700 : 600,
                                    color: isBest ? "var(--good)" : "var(--text-secondary)",
                                    textAlign: "right",
                                    fontVariantNumeric: "tabular-nums",
                                  }}
                                >
                                  {score !== null ? num(score, 1) : "–"}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {/* ─── Giải thích Phương pháp luận Collapsible ─── */}
            <details
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "12px 18px",
                fontSize: 13,
                color: "var(--text-secondary)",
                lineHeight: 1.6,
              }}
            >
              <summary style={{ fontWeight: 700, color: "var(--text)", cursor: "pointer", userSelect: "none" }}>
                📋 Quy ước Khoa học & Nguyên tắc Đối chuẩn giữa các nhóm mô hình
              </summary>
              <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div>
                  • <b>108 Workload:</b> Đối chuẩn cơ sở giữa Heuristic (Nearest Neighbor, Clarke–Wright) và Reinforcement Learning cơ bản (Q-Learning, SARSA, Vanilla DQN). Clarke–Wright đạt cự ly ngắn nhất (58,05 km).
                </div>
                <div>
                  • <b>12 Workload Paired:</b> Nghiên cứu tác động của Soft Reward Shaping (K-Means-guided DQN). Guided DQN giảm -6,83% chuyển cụm liên vùng, giữ 100% SLA đúng hạn, chấp nhận tăng nhẹ cự ly (+1,76%).
                </div>
                <div>
                  • <b>16 Workload Test (Proposed RL):</b> Mô hình độc lập tối ưu mục tiêu đa tiêu chí <code>J(π)</code>, đạt số lần chuyển cụm trung bình thấp nhất (4,00 lần).
                </div>
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  );
}
