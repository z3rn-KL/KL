// Thin typed client for the FastAPI backend (python/api).
// All paths are relative (/api/...) and proxied by Vite in dev/preview.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { signal?: AbortSignal },
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(
      0,
      "Không kết nối được tới backend. Hãy chạy FastAPI (uvicorn api.main:app --port 8000).",
    );
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (body?.detail) detail = JSON.stringify(body.detail);
    } catch {
      if (res.status === 500) {
        detail =
          "Backend báo lỗi 500 (thường do thiếu dataset hoặc graph đường đi trên máy chạy backend).";
      }
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const get = <T,>(path: string, signal?: AbortSignal) =>
  request<T>(path, { signal });
const post = <T,>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body), signal });

// ---------------------------------------------------------------- types

export type AlgorithmKey =
  | "nearest_neighbor"
  | "clarke_wright"
  | "q_learning"
  | "sarsa"
  | "dqn";

export type PrecomputedAlgorithm =
  | "q_learning"
  | "sarsa"
  | "dqn"
  | "kmeans_guided_dqn";

export interface Health {
  status: string;
  api_version: string;
  artifacts: Record<string, boolean>;
  core_artifacts_ready: boolean;
  ready_for_frontend: boolean;
}

export interface AlgorithmSummary {
  algorithm: AlgorithmKey;
  workloads: number;
  deliveries: number;
  mean_distance_km: number;
  std_distance_km: number;
  median_distance_km: number;
  mean_travel_time_minutes: number;
  std_travel_time_minutes: number;
  on_time_deliveries: number;
  late_deliveries: number;
  weighted_on_time_rate: number;
  total_lateness_minutes: number;
  max_lateness_minutes: number;
  mean_route_runtime_seconds: number | null;
  mean_matrix_preparation_seconds: number | null;
  mean_training_runtime_seconds: number | null;
  total_training_runtime_seconds: number | null;
  mean_greedy_reward: number | null;
  mean_reward_per_delivery: number | null;
  mean_model_size: number | null;
}

export interface ResultsOverview {
  benchmark: {
    workloads: number;
    deliveries: number;
    algorithm_runs: number;
    algorithms: AlgorithmSummary[];
  };
  [key: string]: unknown;
}

export interface BenchmarkRun {
  scope: string;
  workload_id: number;
  wave_start: string;
  wave_end: string;
  algorithm: AlgorithmKey;
  number_of_deliveries: number;
  on_time_deliveries: number;
  late_deliveries: number;
  on_time_rate: number | null;
  total_lateness_minutes: number;
  max_lateness_minutes: number;
  total_distance_km: number;
  total_travel_time_minutes: number;
  route_runtime_seconds: number | null;
  training_runtime_seconds: number | null;
  greedy_reward: number | null;
  model_size: number | null;
}

export interface Paged<T> {
  total: number;
  offset: number;
  limit: number;
  returned: number;
  items: T[];
}

export interface Workload {
  workload_id: number;
  wave_start: string;
  wave_end: string;
  number_of_deliveries: number;
  delivery_ids: string[];
}

export interface SensitivityRow {
  cluster_switch_penalty: number;
  runs: number;
  mean_distance_km: number;
  mean_travel_time_minutes: number;
  overall_on_time_rate: number;
  total_late_deliveries: number;
  mean_cluster_switches: number;
  mean_cluster_switch_rate: number;
  mean_training_runtime_seconds: number;
  mean_reward_per_delivery: number;
  [key: string]: number;
}

export interface Sensitivity {
  selected_lambda: number;
  summary: SensitivityRow[];
  pairwise_vs_lambda_zero: Record<string, number | string | null>[];
}

export interface FinalValidation {
  summary: Record<string, number | string | null>;
  paired_workloads: Record<string, number | string | null>[];
  scope_note: string;
}

export interface KmeansPoint {
  cluster_id: number;
  delivery_id: string;
  latitude: number;
  longitude: number;
}

export interface KmeansPage {
  algorithm: string;
  source: string;
  n_clusters: number;
  total_deliveries: number;
  cluster_sizes: Record<string, number>;
  offset: number;
  limit: number;
  returned: number;
  items: KmeansPoint[];
}

export interface DeliveryPoint {
  delivery_id: string;
  latitude: number;
  longitude: number;
}

export interface ClusterRunResult {
  algorithm: string;
  parameters: Record<string, number>;
  metrics: Record<string, number | null>;
  clusters: {
    cluster_id: number;
    size: number;
    total_weight: number;
    centroid: { latitude: number; longitude: number };
    delivery_ids: string[];
  }[];
  noise_delivery_ids?: string[];
  assignments: (DeliveryPoint & { cluster_id: number })[];
}

export interface DeliveryStats {
  deliveries: number;
  columns: number;
  missing_expected_delivery_time: number;
  unique_shippers: number;
  service_type_counts: Record<string, number>;
}

export type DeliveryRecord = Record<string, string | number | boolean | null> & {
  id: string;
};

export interface RouteEvaluation {
  total_distance_km: number;
  total_travel_time_minutes: number;
  on_time_deliveries: number;
  late_deliveries: number;
  on_time_rate: number | null;
  total_lateness_minutes: number;
  max_lateness_minutes: number;
  estimated_operating_cost: number;
  route_geometry: [number, number][];
  delivery_order: string[];
}

export interface CostEstimate {
  total_estimated_cost: number;
  distance_cost: number;
  travel_time_cost: number;
  late_delivery_cost: number;
  lateness_cost: number;
  cost_unit: string;
}

export interface PrecomputedRoute {
  source: string;
  route_type?: string;
  provenance?: string;
  data_status?: string;
  algorithm: string;
  workload_id: number;
  delivery_order: string[];
  reference_metrics: Record<string, number | string | null>;
  estimated_operating_cost_from_reference: CostEstimate | null;
  evaluation: RouteEvaluation | null;
  cost_note: string;
}

export interface PrecomputedIndex {
  [algorithm: string]: { available_workload_ids: number[] } | string;
}

export interface DeliveryQuery {
  q?: string;
  service_type?: string;
  is_late?: boolean;
  cluster_id?: number;
  sort_by?: string;
  sort_dir?: "asc" | "desc";
}

export interface StabilitySeed {
  seed: number;
  training_runtime_seconds: number;
  q_table_size: number;
  greedy_reward: number;
  exact_best_reward: number;
  reward_gap: number;
  reward_gap_percent: number;
  optimal_reward_match: boolean;
  exact_route_match: boolean;
  greedy_distance_km: number;
  on_time_rate: number;
  mean_last_100_reward: number;
  std_last_100_reward: number;
  [key: string]: number | boolean | string | null;
}

export interface HyperRow {
  configuration: string;
  learning_rate: number;
  discount_factor: number;
  epsilon_decay: number;
  episodes: number;
  runs: number;
  optimal_hit_count: number;
  optimal_hit_rate: number;
  mean_greedy_reward: number;
  mean_reward_gap: number;
  mean_training_runtime_seconds: number;
  unique_greedy_routes: number;
  sensitivity_rank: number;
  [key: string]: number | string | null;
}

export interface Stability {
  scope_note: string;
  overview: Record<string, number | null>;
  seeds: StabilitySeed[];
  hyperparameters: HyperRow[];
}

export interface ClusteringQuality {
  kmeans: {
    k: number;
    inertia: number;
    silhouette_score: number;
    min_cluster_size: number;
    max_cluster_size: number;
    imbalance_ratio: number;
    is_best_silhouette: boolean;
  }[];
  dbscan: {
    eps_km: number;
    min_samples: number;
    number_of_clusters: number;
    number_of_noise: number;
    noise_ratio: number;
    silhouette_score: number | null;
    max_cluster_size: number;
  }[];
  selected: { kmeans_k: number; dbscan_eps_km: number; dbscan_min_samples: number };
}

export interface CostResult {
  result: CostEstimate;
  estimated: boolean;
  actual_xedu_accounting_cost: boolean;
}

export interface OptimizeResponse {
  source: string;
  start_time_source?: string;
  algorithm: string;
  depot: { name: string; latitude: number; longitude: number };
  evaluation: RouteEvaluation;
  cost_note: string;
}

// ---------------------------------------------------------------- calls

// ------------------------------------------------ daily planning / Proposed RL

export type AllocationPolicy = "dedicated" | "merge_small";
export type OvertimePolicy = "overtime" | "defer" | "hybrid";

export interface DailyPlanningRow {
  load_multiplier: number;
  allocation: AllocationPolicy;
  overtime_policy: OvertimePolicy;
  orders: number;
  delivered: number;
  final_backlog: number;
  on_time_rate: number;
  same_day_rate: number;
  mean_backlog: number;
  max_backlog: number;
  order_days_deferred: number;
  overtime_hours: number;
  over_cap_shipper_days: number;
  mean_shippers_used: number;
  distance_km: number;
  wage_cost: number;
  overtime_cost: number;
  distance_cost: number;
  defer_cost: number;
  late_cost: number;
  total_cost: number;
  cost_per_order: number;
  work_hours: number;
}

export interface DailyPlanningSummary {
  note: string;
  rows: DailyPlanningRow[];
  best_within_cap: DailyPlanningRow[];
}

export interface DailyPlanningDay {
  load_multiplier: number;
  allocation: AllocationPolicy;
  overtime_policy: OvertimePolicy;
  day: number;
  incoming: number;
  backlog_in: number;
  delivered: number;
  deferred_out: number;
  shippers_used: number;
  overtime_minutes: number;
  total_cost: number;
}

export interface DailyPlanningDaily {
  note: string;
  days: DailyPlanningDay[];
}

export interface DailyPlanningSensitivityRow extends DailyPlanningRow {
  parameter: string;
  value: number;
}

export interface DailyPlanningSensitivity {
  note: string;
  rows: DailyPlanningSensitivityRow[];
}

export interface ProposedRlSummaryRow {
  algorithm: string;
  workloads: number;
  deliveries: number;
  mean_distance_km: number;
  mean_travel_time_minutes: number;
  late_deliveries: number;
  weighted_sla: number;
  total_lateness_minutes: number;
  mean_cluster_switches: number;
  mean_cluster_switch_rate: number;
  mean_locked_objective: number;
}

export interface ProposedRlPairwiseRow {
  baseline_algorithm: string;
  baseline_mean_locked_objective: number;
  proposed_mean_locked_objective: number;
  locked_objective_improvement_pct: number;
  baseline_mean_cluster_switches: number;
  proposed_mean_cluster_switches: number;
  cluster_switch_reduction_pct: number;
  sla_change_percentage_points: number;
  late_delivery_change: number;
}

export interface ProposedRlFinalTest {
  scope_note: string;
  summary: ProposedRlSummaryRow[];
  pairwise_vs_baselines: ProposedRlPairwiseRow[];
}

// ------------------------------------------------ real-data replay

export interface RealReplayMethod {
  method: "real" | "nearest" | "nearest_search" | "optimal";
  mean: number;
  mean_excess_pct: number;
  ci_low_pct: number;
  ci_high_pct: number;
  at_optimum: number;
}

export interface RealData {
  observed: {
    orders_delivered: number;
    orders_with_deadline: number;
    late_rate: number;
  };
  time_model: {
    pairs: number;
    max_gap_minutes: number;
    real_median_minutes: number;
    real_mean_minutes: number;
    model_median_minutes: number;
    model_mean_minutes: number;
    spearman: number;
    real_shorter_than_model_share: number;
    median_overhead_minutes: number;
  };
  clean_shifts: {
    count: number;
    min_orders: number;
    median_orders: number;
    max_orders: number;
    size_counts: Record<string, number>;
    skipped_unreachable: number;
  };
  deadline_order: {
    earliest_deadline_first_shifts: number;
    shifts: number;
    mean_spearman_position_vs_deadline: number;
  };
  replay: { time: RealReplayMethod[]; km: RealReplayMethod[] };
  scope_note: string;
}

// ------------------------------------------------ shift-level RL

export interface ShiftRlMethod {
  key: string;
  label: string;
  kind: "heuristic" | "rl";
  mean_gap_pct: number;
  std_gap_pct: number | null;
  at_reference: number;
}

export interface ShiftRlGroup {
  group: string;
  shifts: number;
  methods: ShiftRlMethod[];
}

export interface ShiftRlPair {
  wins: number;
  ties: number;
  losses: number;
  mean_difference_points: number;
}

export interface ShiftRl {
  test_set: string;
  shifts: number;
  headline: {
    variant: string;
    label: string;
    seed: number;
    validation_gap_pct: number;
    late_per_shift: number;
    deferred_per_shift: number;
    overtime_minutes_per_shift: number;
    inference_ms: number;
  };
  local_search_seconds_per_shift: number;
  paired_vs: Record<string, ShiftRlPair>;
  groups: ShiftRlGroup[];
  scope_note: string;
}

export const api = {
  health: (s?: AbortSignal) => get<Health>("/api/health", s),

  overview: (s?: AbortSignal) =>
    get<ResultsOverview>("/api/results/overview", s),
  benchmarkSummary: (s?: AbortSignal) =>
    get<{ items: AlgorithmSummary[] }>("/api/results/benchmark/summary", s),
  benchmarkRuns: (
    q: { workload_id?: number; algorithm?: string; limit?: number },
    s?: AbortSignal,
  ) => {
    const p = new URLSearchParams();
    if (q.workload_id) p.set("workload_id", String(q.workload_id));
    if (q.algorithm) p.set("algorithm", q.algorithm);
    p.set("limit", String(q.limit ?? 100));
    return get<Paged<BenchmarkRun>>(`/api/results/benchmark/runs?${p}`, s);
  },
  workloads: (limit = 200, s?: AbortSignal) =>
    get<Paged<Workload>>(`/api/results/workloads?limit=${limit}`, s),
  sensitivity: (s?: AbortSignal) =>
    get<Sensitivity>("/api/results/sensitivity", s),
  finalValidation: (s?: AbortSignal) =>
    get<FinalValidation>("/api/results/final-validation", s),

  /** Precomputed thesis K-Means (K=10); the endpoint pages at 500 rows. */
  kmeansAll: async (s?: AbortSignal) => {
    const first = await get<KmeansPage>("/api/clustering/kmeans?limit=500", s);
    const pages: Promise<KmeansPage>[] = [];
    for (let off = 500; off < first.total_deliveries; off += 500) {
      pages.push(
        get<KmeansPage>(`/api/clustering/kmeans?limit=500&offset=${off}`, s),
      );
    }
    const rest = await Promise.all(pages);
    return {
      meta: first,
      points: [first, ...rest].flatMap((p) => p.items),
    };
  },
  runDbscan: (
    deliveries: DeliveryPoint[],
    eps_km: number,
    min_samples: number,
    s?: AbortSignal,
  ) =>
    post<ClusterRunResult>(
      "/api/clustering/dbscan/run",
      { deliveries, eps_km, min_samples },
      s,
    ),
  runKmeans: (deliveries: DeliveryPoint[], n_clusters: number, s?: AbortSignal) =>
    post<ClusterRunResult>(
      "/api/clustering/kmeans/run",
      { deliveries, n_clusters },
      s,
    ),

  deliveryStats: (s?: AbortSignal) =>
    get<DeliveryStats>("/api/deliveries/stats", s),
  deliveries: (
    offset: number,
    limit: number,
    query: DeliveryQuery = {},
    s?: AbortSignal,
  ) => {
    const p = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    });
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== "" && v !== null) p.set(k, String(v));
    }
    return get<Paged<DeliveryRecord>>(`/api/deliveries?${p}`, s);
  },
  delivery: (id: string, s?: AbortSignal) =>
    get<DeliveryRecord>(`/api/deliveries/${encodeURIComponent(id)}`, s),

  precomputedIndex: (s?: AbortSignal) =>
    get<PrecomputedIndex>("/api/routing/precomputed", s),
  precomputedRoute: (
    algorithm: PrecomputedAlgorithm,
    workload_id: number,
    include_geometry: boolean,
    s?: AbortSignal,
  ) =>
    get<PrecomputedRoute>(
      `/api/routing/precomputed/${algorithm}?workload_id=${workload_id}&include_geometry=${include_geometry}`,
      s,
    ),
  optimize: (
    body: {
      algorithm: "nearest_neighbor" | "clarke_wright";
      delivery_ids: string[];
      metric?: "distance" | "travel_time";
      return_to_depot?: boolean;
      start_time?: string;
    },
    s?: AbortSignal,
  ) => post<OptimizeResponse>("/api/routing/optimize", body, s),

  stability: (s?: AbortSignal) => get<Stability>("/api/results/stability", s),
  clusteringQuality: (s?: AbortSignal) =>
    get<ClusteringQuality>("/api/results/clustering-quality", s),
  cost: (
    body: {
      total_distance_km: number;
      total_travel_time_minutes: number;
      late_deliveries?: number;
      total_lateness_minutes?: number;
    },
    s?: AbortSignal,
  ) =>
    post<CostResult>(
      "/api/evaluation/cost",
      { late_deliveries: 0, total_lateness_minutes: 0, ...body },
      s,
    ),

  dailyPlanningSummary: (s?: AbortSignal) =>
    get<DailyPlanningSummary>("/api/results/daily-planning/summary", s),
  dailyPlanningDaily: (
    load_multiplier: number,
    allocation: AllocationPolicy,
    overtime_policy: OvertimePolicy,
    s?: AbortSignal,
  ) =>
    get<DailyPlanningDaily>(
      `/api/results/daily-planning/daily?load_multiplier=${load_multiplier}&allocation=${allocation}&overtime_policy=${overtime_policy}`,
      s,
    ),
  dailyPlanningSensitivity: (s?: AbortSignal) =>
    get<DailyPlanningSensitivity>("/api/results/daily-planning/sensitivity", s),
  proposedRlFinalTest: (s?: AbortSignal) =>
    get<ProposedRlFinalTest>("/api/results/proposed-rl/final-test", s),
  realData: (s?: AbortSignal) =>
    get<RealData>("/api/results/real-data", s),
  shiftRl: (s?: AbortSignal) =>
    get<ShiftRl>("/api/results/shift-rl", s),
};
