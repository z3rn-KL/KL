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
const patch = <T,>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: "PATCH", body: JSON.stringify(body), signal });

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
  evaluateRoute: (
    body: {
      delivery_ids: string[];
      delivery_order: string[];
      algorithm_label?: string;
      start_time?: string;
      return_to_depot?: boolean;
    },
    s?: AbortSignal,
  ) =>
    post<{
      start_time_source: string;
      evaluation: RouteEvaluation;
      cost_note: string;
    }>('/api/evaluation/route', body, s),
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

  // ------------------------------------------------------------- dispatch
  dispatchCapabilities: (s?: AbortSignal) =>
    get<DispatchCapabilities>("/api/dispatch/capabilities", s),
  dispatchScenarios: (s?: AbortSignal) =>
    get<Array<{ scenario_id: string; name: string; description: string; data_label: string; total_days: number; orders_per_day_approx: number; total_shippers: number; reproducible_seed: number }>>("/api/dispatch/scenarios", s),
  createDispatchSession: (
    body: {
      scenario_id?: string;
      routing_algorithm?: "nearest_neighbor" | "clarke_wright" | "proposed_rl";
      priority_policy?: string;
      total_days?: number;
      seed?: number;
    },
    s?: AbortSignal,
  ) => post<DispatchSessionState>("/api/dispatch/sessions", body, s),
  getDepotDetails: (sessionId: string, s?: AbortSignal) =>
    get<DepotDetailsResponse>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}/depot`, s),
  updateSessionDepot: (
    sessionId: string,
    data: { name?: string; lat?: number; lng?: number; description?: string },
    s?: AbortSignal
  ) =>
    post<{ success: boolean; depot: any }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/depot`,
      data,
      s
    ),
  getStagingBases: (sessionId: string, s?: AbortSignal) =>
    get<StagingBase[]>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}/bases`, s),
  getServiceAreas: (sessionId: string, s?: AbortSignal) =>
    get<ShipperServiceAreaItem[]>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}/service-areas`, s),
  updateShipperServiceArea: (
    sessionId: string,
    shipperId: string,
    data: {
      shift_start_mode?: "START_AT_DEPOT" | "START_AT_BASE";
      base_id?: string;
      home_cluster_id?: number;
      preferred_cluster_ids?: number[];
    },
    s?: AbortSignal
  ) =>
    patch<{ success: boolean; shipper: ShipperData }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/shippers/${encodeURIComponent(shipperId)}/service-area`,
      data,
      s
    ),
  getDispatchSession: (sessionId: string, s?: AbortSignal) =>
    get<DispatchSessionState>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}`, s),
  getNearbyOrders: (sessionId: string, shipperId: string, limit = 15, s?: AbortSignal) =>
    get<NearbyOrderItem[]>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}/shippers/${encodeURIComponent(shipperId)}/nearby-orders?limit=${limit}`, s),
  assignOrders: (sessionId: string, shipperId: string, orderIds: string[], s?: AbortSignal) =>
    post<{ success: boolean; shipper_id: string; assigned_count: number; total_assigned: number; route_sequence: string[] }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/assign`,
      { shipper_id: shipperId, order_ids: orderIds },
      s,
    ),
  unassignOrders: (sessionId: string, orderIds: string[], s?: AbortSignal) =>
    post<{ success: boolean; unassigned_count: number }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/unassign`,
      { order_ids: orderIds },
      s,
    ),
  autoDispatch: (
    sessionId: string,
    algorithm?: "nearest_neighbor" | "clarke_wright" | "proposed_rl",
    strategy?: "cluster_round_robin" | "location_aware_nearest" | "location_aware_min_cost" | "cluster_aware_rl" | "depot_zone_capacity_aware",
    s?: AbortSignal
  ) =>
    post<{ success: boolean; dispatched_orders: number; algorithm_used?: string; strategy_used?: string }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/auto-dispatch`,
      { algorithm, strategy },
      s,
    ),
  previewDispatch: (
    sessionId: string,
    strategy?: "cluster_round_robin" | "location_aware_nearest" | "location_aware_min_cost" | "cluster_aware_rl" | "depot_zone_capacity_aware",
    algorithm?: "nearest_neighbor" | "clarke_wright" | "proposed_rl",
    s?: AbortSignal
  ) =>
    post<PlanPreviewResponse>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/preview`,
      { strategy, algorithm },
      s
    ),
  confirmPreview: (sessionId: string, previewId: string, s?: AbortSignal) =>
    post<{ success: boolean; dispatched_orders: number; algorithm_used: string; strategy_used: string; preview_id: string }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/confirm-preview`,
      { preview_id: previewId },
      s
    ),
  cancelPreview: (sessionId: string, s?: AbortSignal) =>
    post<{ success: boolean; message: string }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/cancel-preview`,
      {},
      s
    ),
  stepSimulationClock: (sessionId: string, deltaMinutes = 15, s?: AbortSignal) =>
    post<{ simulation_clock: number; current_day: number; is_running: boolean; deliveries_made: number; is_overtime: boolean }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/clock/step`,
      { delta_minutes: deltaMinutes },
      s,
    ),
  endDay: (sessionId: string, s?: AbortSignal) =>
    post<{ success: boolean; day: number; metrics: any; carried_over_orders_count: number }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/end-day`,
      {},
      s,
    ),
  nextDay: (sessionId: string, s?: AbortSignal) =>
    post<{ success: boolean; day: number; simulation_clock: number; carried_over_count: number; new_incoming_count: number; total_available_orders: number }>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/next-day`,
      {},
      s,
    ),
  getDispatchMetrics: (sessionId: string, s?: AbortSignal) =>
    get<any>(`/api/dispatch/sessions/${encodeURIComponent(sessionId)}/metrics`, s),
  getSessionRlDecisionTrace: (
    sessionId: string,
    params?: { shipper_id?: string; order_ids?: string[] },
    s?: AbortSignal,
  ) =>
    post<RlDecisionTraceResponse>(
      `/api/dispatch/sessions/${encodeURIComponent(sessionId)}/rl-decision-trace`,
      params || {},
      s,
    ),
  ablationStudy: (s?: AbortSignal) =>
    get<{
      title: string;
      day1_summary: Array<{
        experiment_id: string;
        assignment_policy: string;
        routing_solver: string;
        total_fleet_distance_km: number;
        mean_distance_per_shipper_km: number;
        distance_per_completed_order_km: number;
        total_travel_time_minutes: number;
        completed_orders: number;
        late_orders: number;
        on_time_rate_pct: number;
        total_late_minutes: number;
        inter_cluster_transitions: number;
        overtime_minutes: number;
        estimated_operating_cost_vnd: number;
        planning_time_seconds: number;
      }>;
      day1_details: Array<any>;
      multiday_summary: Array<{
        experiment_id: string;
        assignment_policy: string;
        routing_solver: string;
        day: number;
        active_orders: number;
        completed_orders: number;
        backlog_remaining: number;
        day_distance_km: number;
        cumulative_distance_km: number;
        day_on_time_rate_pct: number;
        cumulative_on_time_rate_pct?: number;
        day_late_orders: number;
        cumulative_late_orders: number;
        day_cost_vnd: number;
        cumulative_cost_vnd: number;
      }>;
      heldout_summary: Array<{
        algorithm: string;
        display_name: string;
        workloads: number;
        total_deliveries: number;
        mean_distance_km: number;
        mean_travel_time_min: number;
        total_late_deliveries: number;
        on_time_rate_pct: number;
      }>;
      paired_deltas: Array<{
        comparison: string;
        baseline: string;
        candidate: string;
        distance_delta_km: number;
        distance_pct_change: number;
        sla_delta_pct: number;
        late_orders_delta: number;
        transitions_delta: number;
        cost_delta_vnd: number;
      }>;
    }>("/api/results/ablation-study", s),
  rlDecisionTrace: (
    q?: { shipperId?: string; day?: number; seed?: number },
    s?: AbortSignal,
  ) => {
    const p = new URLSearchParams();
    if (q?.shipperId) p.set("shipper_id", q.shipperId);
    if (q?.day) p.set("day", String(q.day));
    if (q?.seed) p.set("seed", String(q.seed));
    return get<RLDecisionTrace>(`/api/results/rl-decision-trace?${p}`, s);
  },

  trainingHistory: (
    algorithm: "proposed_rl" | "dqn" | "sarsa" | "q_learning" = "proposed_rl",
    sampleRate: number = 50,
    s?: AbortSignal,
  ) =>
    get<TrainingHistoryResponse>(
      `/api/results/training-history?algorithm=${algorithm}&sample_rate=${sampleRate}`,
      s,
    ),

};



export interface StagingBase {
  base_id: string;
  name: string;
  lat: number;
  lng: number;
  district: string;
  primary_clusters: number[];
}

export interface ShipperServiceAreaItem {
  shipper_id: string;
  name: string;
  depot_id: string;
  shift_start_mode: "START_AT_DEPOT" | "START_AT_BASE";
  start_lat: number;
  start_lng: number;
  base_id: string;
  base_name: string;
  base_lat: number;
  base_lng: number;
  home_cluster_id: number;
  preferred_cluster_ids: number[];
  capacity_kg: number;
  current_load_kg: number;
  assigned_order_count: number;
  status: string;
}

export interface DepotDetailsResponse {
  depot: {
    depot_id: string;
    name: string;
    lat: number;
    lng: number;
    depot_type?: string;
    status?: string;
    description?: string;
    served_clusters?: number[];
  };
  staging_bases?: StagingBase[];
  linked_shippers_count: number;
  served_clusters: number[];
  total_fleet_capacity_kg: number;
  active_routes_count: number;
}

export interface ShipperData {
  shipper_id: string;
  name: string;
  depot_id: string;
  home_cluster_id: number;
  current_lat: number;
  current_lng: number;
  status: "IDLE" | "IN_TRANSIT" | "DELIVERING" | "RETURNING" | "SHIFT_ENDED";
  assigned_order_ids: string[];
  completed_order_ids: string[];
  current_order_id: string | null;
  capacity_kg: number;
  current_load_kg: number;
  distance_travelled_km: number;
  working_minutes: number;
  overtime_minutes: number;
  route_sequence: string[];
  route_geometry: [number, number][];
  location_source?: "DEPOT" | "SIMULATION" | "REAL_GPS" | "ESTIMATED";
  last_location_update_minute?: number;
  shift_start_mode?: "START_AT_DEPOT" | "START_AT_BASE";
  start_lat?: number;
  start_lng?: number;
  base_id?: string;
  base_name?: string;
  base_lat?: number;
  base_lng?: number;
  preferred_cluster_ids?: number[];
  timeline_stops: Array<{
    order_id: string;
    order_code?: string;
    package_name?: string;
    weight_kg?: number;
    lat: number;
    lng: number;
    cluster_id: number;
    arrival_minute: number;
    departure_minute: number;
    leg_distance_km: number;
    deadline_minutes: number | null;
    is_overdue: boolean;
  }>;
}

export interface DispatchOrderItem {
  order_id: string;
  order_code?: string;
  package_name?: string;
  latitude: number;
  longitude: number;
  weight_kg: number;
  cluster_id: number;
  release_day: number;
  created_at: string;
  deadline_minutes: number | null;
  service_duration_minutes: number;
  assigned_shipper_id: string | null;
  delivery_status: "UNASSIGNED" | "ASSIGNED" | "IN_TRANSIT" | "DELIVERED" | "CARRIED_OVER" | "FAILED";
  sla_status: "ON_TIME" | "AT_RISK" | "OVERDUE";
  carryover_count: number;
  assigned_at_minute?: number | null;
  delivered_at_minute?: number | null;
  is_simulated_scenario: boolean;
  dispatch_reason?: string | null;
  dispatch_distance_km?: number | null;
  dispatch_strategy?: string | null;
}

export interface NearbyOrderItem {
  order_id: string;
  order_code?: string;
  package_name?: string;
  latitude: number;
  longitude: number;
  weight_kg: number;
  cluster_id: number;
  cluster_match: boolean;
  distance_km: number;
  distance_type?: string;
  travel_time_minutes: number;
  travel_time_type?: string;
  deadline_minutes: number | null;
  slack_minutes: number | null;
  is_overdue: boolean;
  carryover_count: number;
  priority_score: number;
  delivery_status: string;
  sla_status: string;
  capacity_feasible?: boolean;
  recommendation_reason?: string;
  recommendation_breakdown?: string[];
  policy_name?: string;
}

export interface DispatchSessionState {
  staging_bases?: StagingBase[];
  session_id: string;
  scenario_id: string;
  scenario_name: string;
  current_day: number;
  total_days: number;
  simulation_clock: number;
  shift_minutes: number;
  max_overtime_minutes: number;
  is_running: boolean;
  is_day_ended: boolean;
  routing_algorithm: "nearest_neighbor" | "clarke_wright" | "proposed_rl";
  priority_policy: string;
  session_revision?: number;
  has_active_preview?: boolean;
  active_preview_id?: string | null;
  depot: {
    depot_id: string;
    name: string;
    lat: number;
    lng: number;
    depot_type?: string;
    status?: string;
    description?: string;
    served_clusters?: number[];
  };
  fleet_summary: {
    total_shippers: number;
    active_shippers: number;
    total_distance_km: number;
  };
  order_summary: {
    total: number;
    delivered: number;
    assigned: number;
    unassigned: number;
    carried_over: number;
  };
  shippers: ShipperData[];
  orders: DispatchOrderItem[];
  daily_metrics: Array<{
    day: number;
    total_orders: number;
    delivered: number;
    delivered_on_time: number;
    delivered_overdue: number;
    carried_over_out: number;
    on_time_rate: number;
    delivery_completion_rate: number;
    shippers_active: number;
    total_distance_km: number;
    total_work_minutes: number;
    total_overtime_minutes: number;
    day_delivered?: number;
    day_on_time_rate?: number;
    day_delivered_overdue?: number;
    shift_status?: string;
    simulation_clock_at_end?: number;
    shift_duration_minutes?: number;
  }>;
  recent_events: Array<{
    event_id: string;
    timestamp_minute: number;
    day: number;
    event_type: string;
    shipper_id: string | null;
    order_id: string | null;
    description: string;
  }>;
}

export interface CandidateKpi {
  total_eligible_orders: number;
  dispatched_orders: number;
  unassigned_orders: number;
  active_shippers: number;
  total_distance_km: number;
  mean_distance_km: number;
  total_load_kg: number;
  sla_on_time_rate: number;
  max_duration_minutes: number;
  total_overtime_minutes: number;
  estimated_cost: number;
}

export interface BaselineKpi {
  assigned_orders: number;
  active_shippers: number;
  total_distance_km: number;
  sla_on_time_rate: number;
  estimated_cost: number;
}

export interface KpiDeltas {
  delta_distance_km: number;
  delta_active_shippers: number;
  delta_orders_dispatched: number;
  delta_cost: number;
  delta_sla_rate?: number;
}

export interface ShipperPreviewItem {
  shipper_id: string;
  name: string;
  home_cluster_id: number;
  status: string;
  current_lat: number;
  current_lng: number;
  location_source?: string;
  assigned_order_ids: string[];
  order_count: number;
  current_load_kg: number;
  capacity_kg: number;
  is_overweight: boolean;
  total_distance_km: number;
  estimated_duration_minutes: number;
  is_overtime: boolean;
  route_sequence: string[];
  route_geometry: [number, number][];
  timeline_stops: Array<any>;
  sla_on_time_count: number;
  sla_overdue_count: number;
  algorithm_used: string;
}

export interface OrderPreviewItem {
  order_id: string;
  order_code?: string;
  package_name?: string;
  cluster_id: number;
  weight_kg: number;
  deadline_minutes: number | null;
  candidate_shipper_id: string | null;
  dispatch_strategy?: string | null;
  dispatch_reason?: string | null;
  dispatch_distance_km?: number | null;
}

export interface PlanPreviewResponse {
  success: boolean;
  preview_id: string;
  session_id: string;
  session_revision: number;
  strategy_used: string;
  algorithm_used: string;
  candidate_kpi: CandidateKpi;
  baseline_kpi: BaselineKpi | null;
  deltas: KpiDeltas | null;
  shipper_previews: ShipperPreviewItem[];
  order_previews: OrderPreviewItem[];
}

export interface DispatchCapabilities {
  live_dispatch_algorithms: Array<{
    id: string;
    name: string;
    description: string;
    type: string;
    is_live_capable: boolean;
    has_checkpoint: boolean;
  }>;
  research_replay_algorithms: Array<{
    id: string;
    name: string;
    reason: string;
    type: string;
    is_live_capable: boolean;
    supported_workloads: number[];
  }>;
  priority_policies: Array<{ id: string; name: string }>;
  default_fleet_size: number;
  default_shift_hours: number;
}


// ============================================================
// RL DECISION TRACE & CANDIDATE SCORING INTERFACES
// ============================================================

export interface RLDecisionCandidate {
  candidate_index: number;
  order_id: string;
  order_code: string;
  package_name: string;
  cluster_id: number;
  deadline_minutes: number | null;
  lat: number;
  lng: number;
  model_candidate_score: number;
  is_chosen: boolean;
  features: {
    elapsed_time_ratio: number;
    remaining_deliveries_ratio: number;
    direct_distance_ratio: number;
    travel_time_ratio: number;
    slack_ratio: number;
    deadline_valid: number;
    lateness_ratio: number;
    cluster_switch: number;
  };
}

export interface RLDecisionStep {
  step_index: number;
  current_node_index: number;
  current_coords: { lat: number; lng: number };
  current_time_minutes: number;
  remaining_deliveries: number;
  chosen_action_index: number;
  chosen_order_id: string;
  chosen_order_code: string;
  step_reward: number;
  leg_distance_km: number;
  leg_time_minutes: number;
  arrival_minute: number;
  is_late: boolean;
  lateness_minutes: number;
  candidates: RLDecisionCandidate[];
}

export interface RLRouteMetrics {
  distance_km: number;
  travel_time_minutes: number;
  total_time_minutes: number;
  late_count: number;
  total_lateness_minutes: number;
  on_time_rate: number;
  delivery_count: number;
}

export interface RLDecisionTrace {
  shipper_id: string;
  shipper_name: string;
  home_cluster_id: number;
  day: number;
  seed: number;
  assigned_orders_count: number;
  session_revision?: number;
  has_active_preview?: boolean;
  active_preview_id?: string | null;
  depot: { lat: number; lng: number };
  feature_names: string[];
  model_architecture: {
    name: string;
    type: string;
    input_dim: number;
    hidden_layers: number[];
    activation: string;
    output_dim: number;
    loss_function: string;
    discount_factor_gamma: number;
    inference_mode: string;
  };
  steps: RLDecisionStep[];
  routes: {
    proposed_rl: string[];
    nearest_neighbor: string[];
    clarke_wright: string[];
  };
  orders_metadata: Array<{
    order_id: string;
    order_code: string;
    package_name: string;
    cluster_id: number;
    deadline_minutes: number | null;
    weight_kg: number;
    lat: number;
    lng: number;
  }>;
  metrics: {
    proposed_rl: RLRouteMetrics;
    nearest_neighbor: RLRouteMetrics;
    clarke_wright: RLRouteMetrics;
  };
  divergence: {
    diverges_from_nn: boolean;
    diverges_from_cw: boolean;
    first_divergence_step: number | null;
  };
  scientific_disclosure: {
    score_interpretation: string;
    step_reward_interpretation: string;
    early_divergence_mechanism: string;
    distance_metric: string;
  };
}

export interface ProposedRLDatapoint {
  epoch: number;
  cumulative_episodes: number;
  mean_training_reward: number;
  mean_training_objective: number;
  epsilon_start: number;
  epsilon_end: number;
  gradient_steps: number;
  validation_ran: boolean;
  validation_weighted_sla_pct: number | null;
  validation_mean_distance_km: number | null;
  validation_late_deliveries: number | null;
  is_best_checkpoint: boolean;
  is_global_best?: boolean;
}

export interface BaselineRLDatapoint {
  episode: number;
  epsilon: number;
  total_reward: number;
  reward_ma50: number;
  total_distance_km: number;
  distance_ma50: number;
  total_travel_time_minutes: number;
  steps: number;
  mean_loss?: number;
}

export interface TrainingHistoryResponse {
  algorithm: "proposed_rl" | "dqn" | "sarsa" | "q_learning";
  name: string;
  dataset_id: string;
  source_file: string;
  total_epochs?: number;
  total_episodes: number;
  sample_rate?: number;
  hyperparameters: Record<string, any>;
  training_vs_inference: {
    training_mode: string;
    inference_mode: string;
    action_space?: string;
  };
  scientific_interpretation: string;
  datapoints: any[];
}

export interface CandidateEvaluation {
  action_index: number;
  delivery_id: string;
  q_score: number;
  features: number[];
  feature_dict: {
    norm_distance?: number;
    norm_travel_time?: number;
    deadline_tightness?: number;
    slack_ratio?: number;
    detour_penalty?: number;
    cluster_alignment?: number;
    return_depot_impact?: number;
    candidate_density?: number;
  };
  package_name?: string;
  order_code?: string;
  cluster_id?: number;
  weight_kg?: number;
  lat?: number;
  lng?: number;
}

export interface DecisionTraceStep {
  step_index: number;
  current_node: string;
  chosen_delivery_id: string;
  chosen_action_index: number;
  chosen_q_score: number;
  candidates_evaluated: CandidateEvaluation[];
  step_reward: number;
  cumulative_distance_km: number;
  cumulative_time_minutes: number;
  lateness_minutes: number;
  cluster_switches: number;
  partial_route: string[];
}

export interface RlDecisionTraceResponse {
  success: boolean;
  session_id: string;
  total_stops: number;
  ordered_ids: string[];
  decision_trace: DecisionTraceStep[];
  objective_weights: {
    distance: number;
    travel_time: number;
    late_delivery: number;
    lateness: number;
    cluster_switch: number;
  };
  algorithm: string;
  model_type: string;
  checkpoint: string;
  shipper_id?: string;
  active_shippers?: Array<{ shipper_id: string; name: string; order_count: number }>;
  message?: string;
}

export interface BenchmarkComparisonCase {
  workload_id: number;
  deliveries_count: number;
  results: Record<string, {
    distance_km: number;
    time_minutes: number;
    on_time_rate: number;
    deliveries_count: number;
  }>;
  rl_vs_nn_diff_km: number;
  rl_vs_nn_diff_pct: number;
  verdict: 'RL_OUTPERFORMS_NN' | 'NN_OUTPERFORMS_RL' | 'NEAR_EQUIVALENT';
}

export interface BenchmarkComparisonCasesResponse {
  success: boolean;
  source_file: string;
  cohort: string;
  cases: BenchmarkComparisonCase[];
}
