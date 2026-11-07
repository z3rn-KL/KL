import type { AlgorithmKey, PrecomputedAlgorithm } from "./api";

export const ALGO_LABEL: Record<AlgorithmKey | PrecomputedAlgorithm, string> = {
  nearest_neighbor: "Nearest Neighbor",
  clarke_wright: "Clarke–Wright",
  q_learning: "Q-Learning",
  sarsa: "SARSA",
  dqn: "DQN",
  kmeans_guided_dqn: "DQN + K-Means",
};

export const ALGO_COLOR: Record<AlgorithmKey | PrecomputedAlgorithm, string> = {
  nearest_neighbor: "#5F7694",
  clarke_wright: "#D97706",
  q_learning: "#059669",
  sarsa: "#8B5CF6",
  dqn: "#0284C7",
  kmeans_guided_dqn: "#6475C9",
};

export const ALGO_ORDER: AlgorithmKey[] = [
  "nearest_neighbor",
  "clarke_wright",
  "q_learning",
  "sarsa",
  "dqn",
];

// Categorical palette for clusters (K-Means K=10 fits exactly).
export const CLUSTER_COLORS = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
];
export const NOISE_COLOR = "#111827";

export const clusterColor = (id: number) =>
  id < 0 ? NOISE_COLOR : CLUSTER_COLORS[id % CLUSTER_COLORS.length];

// Default depot used by the backend scenario (simulation/scenario.py).
export const DEPOT = { lat: 10.7769, lng: 106.7009, name: "Kho (Depot 01)" };
export const MAP_CENTER: [number, number] = [10.7769, 106.7009];

const nf = (minD: number, maxD = minD) =>
  new Intl.NumberFormat("vi-VN", {
    minimumFractionDigits: minD,
    maximumFractionDigits: maxD,
  });

export const num = (v: number | null | undefined, d?: number) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof d === "number") {
    return nf(d, d).format(v);
  }
  // Adaptive formatting: keep up to 3 decimals without trailing zeroes
  return nf(0, 3).format(v);
};

export const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "–"
    : `${nf(d).format(v * 100)}%`;

export const signedPct = (v: number | null | undefined, d = 1) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  // Eliminate negative zero: round to display precision and compare
  const factor = Math.pow(10, d);
  const rounded = Math.round(v * factor) / factor;
  const display = Object.is(rounded, -0) ? 0 : rounded;
  return `${display > 0 ? "+" : ""}${nf(d).format(display)}%`;
};

export const vnd = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "–"
    : `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 }).format(v)} ₫`;

export const dateTime = (v: unknown) => {
  if (typeof v !== "string" || !v) return "–";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleString("vi-VN", {
    timeZone: "Asia/Ho_Chi_Minh",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export const SERVICE_LABEL: Record<string, string> = {
  "3h": "Giao 3 giờ",
  "5h": "Giao 5 giờ",
  ban_tai_4h: "Bán tải 4 giờ",
  ban_tai_nhanh: "Bán tải nhanh",
  luu_kho: "Lưu kho",
};
