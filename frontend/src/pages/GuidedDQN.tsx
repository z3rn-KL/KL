import { useMemo, useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import L from "leaflet";
import { Marker, Polyline, Popup } from "react-leaflet";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MapBase } from "../components/MapBase";
import { MapErrorBoundary } from "../components/ErrorBoundary";
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
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  PageHeader,
  LoadingCard,
  ErrorCard,
  ResearchCallout,
} from "@/components/shared/DomainComponents";
import {
  Network,
  CheckCircle2,
  TrendingDown,
  Clock,
  ShieldCheck,
  Compass,
} from "lucide-react";
import { api, type SensitivityRow, type DeliveryRecord } from "../lib/api";
import { num, pct, signedPct, clusterColor } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { useTheme } from "../lib/theme";
import { validLL } from "../lib/geo";

const n = (v: unknown): number | null => (typeof v === "number" ? v : null);
const GUIDED_COLOR = "#8B8CF8";
const VANILLA_COLOR = "#60A5FA";
const TRADEOFF_COLOR = "#FBBF24";

const DEPOT = { lat: 10.7769, lng: 106.7009, name: "Kho xuất phát Depot (Quận 1)" };

function depotIcon() {
  return L.divIcon({
    className: "",
    html: '<div class="depot-icon" style="box-shadow: 0 0 10px rgba(13, 148, 136, 0.6);"><span style="width: 10px; height: 10px; border-radius: 50%; background: #0d9488;"></span></div>',
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

function pinIcon(label: string | number, color: string) {
  return L.divIcon({
    className: "pin",
    html: `<div style="background: ${color}; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 11px; box-shadow: 0 2px 6px rgba(0,0,0,0.35); border: 1.5px solid #fff;">${label}</div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function rippleIcon() {
  return L.divIcon({
    className: "cluster-ripple-wrap",
    html: '<div class="cluster-ripple-ring"></div><div class="cluster-ripple-core"></div>',
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

const latlng = (d: DeliveryRecord): [number, number] | null => {
  const lat = Number(d.receiverLat ?? d.latitude ?? d.lat);
  const lng = Number(d.receiverLng ?? d.longitude ?? d.lng);
  return Number.isFinite(lat) && Number.isFinite(lng) ? [lat, lng] : null;
};

const BASELINE_DELIVERY_ORDERS: Record<number, string[]> = {
  29: ["66668e28bb9619f5c265adfd", "66667aa0bb9619f5c265a711", "66667a1bbb9619f5c265a6e8", "66667fe6bb9619f5c265a947", "66667d4fbb9619f5c265a839"],
  42: ["668223fc2d58011d21e79c86", "668225af2d58011d21e7a021", "66821fdc2d58011d21e79557", "6682227f2d58011d21e798a9", "668226ae2d58011d21e7a344", "6682230e2d58011d21e7991d"],
  65: ["6696bc45c7250ec426d89b88", "6696b597c7250ec426d89a42", "6696b6e1c7250ec426d89a4b", "6696bb32c7250ec426d89af7", "6696b7e6c7250ec426d89adc", "6696b907c7250ec426d89ae5", "6696bd0ec7250ec426d89b91", "6696ba77c7250ec426d89aee"],
  71: ["669b3571c7250ec426d98692", "669b30b2c7250ec426d97fa4", "669b2bd8c7250ec426d97784", "669b318cc7250ec426d983a7", "669b2de9c7250ec426d977cf", "669b2f46c7250ec426d982e", "669b3249c7250ec426d9840b", "669b347cc7250ec426d984fd", "669b3033c7250ec426d97bf1", "669b329fc7250ec426d98434"],
  62: ["6694aa74d54b84ac2ddb4115", "6694a133d54b84ac2ddb34c7", "6694a2f6d54b84ac2ddb38ba", "6694a27fd54b84ac2ddb3882", "6694a2e1d54b84ac2ddb38b3", "6694a434d54b84ac2ddb3ccf", "6694a9ccd54b84ac2ddb402b", "6694a3b3d54b84ac2ddb3c92", "6694a5ead54b84ac2ddb3dfd", "6694a906d54b84ac2ddb3fbc", "6694aabbd54b84ac2ddb413a", "6694a748d54b84ac2ddb3e2d"],
  90: ["66b1aef78bafc7aafe84102c", "66b1af828bafc7aafe84104f", "66b1a3a841dd95a0cd977312", "66b1add041dd95a0cd97754b", "66b1ac0c41dd95a0cd9774ae", "66b1a2f041dd95a0cd9772cb", "66b1a12a41dd95a0cd976eec", "66b1ad7b41dd95a0cd9774fb", "66b1a22241dd95a0cd976f21", "66b1afe941dd95a0cd9776fe", "66b1adfc41dd95a0cd977570", "66b1aedf41dd95a0cd97757d", "66b1b07141dd95a0cd977755"],
};

// Baseline cluster sequences for workloads from sensitivity (lambda=0.0)
const BASELINE_SEQUENCES: Record<number, number[]> = {
  29: [4, 4, 4, 2, 9],
  42: [6, 4, 4, 8, 7, 7],
  65: [4, 4, 6, 6, 5, 2, 6, 8],
  71: [4, 4, 6, 9, 0, 0, 8, 4, 7, 3],
  62: [6, 2, 4, 4, 4, 4, 0, 0, 1, 8, 8, 7],
  90: [6, 6, 4, 4, 4, 8, 8, 0, 9, 5, 5, 3, 7],
};

const PART_EXPLANATIONS: Record<
  string,
  { title: string; badge: string; desc: string; formulaRole: string }
> = {
  data: {
    title: "Tọa độ giao hàng (Dataset XeDu)",
    badge: "2.394 đơn hàng TP.HCM",
    desc: "Tọa độ kinh độ/vĩ độ và khung giờ giao hàng thực tế của các đơn hàng. Làm đầu vào cho cả phân cụm K-Means lẫn môi trường ra quyết định tuần tự RL.",
    formulaRole: "Cung cấp không gian trạng thái s_t (tọa độ vị trí hiện tại của shipper).",
  },
  kmeans: {
    title: "Phân cụm K-Means (K = 10 cụm)",
    badge: "Tín hiệu không gian địa lý",
    desc: "Mô hình học không giám sát phân vùng 2.394 đơn vào 10 cụm địa lý tự nhiên dựa trên khoảng cách Haversine. Mỗi đơn hàng được gắn một nhãn cluster_id (c_t).",
    formulaRole: "Gán nhãn cụm c_t cho từng điểm giao, làm căn cứ xác định sự thay đổi cụm.",
  },
  env: {
    title: "Môi trường học tăng cường (RL Routing Environment)",
    badge: "Theo dõi bước chuyển c_t → c_{t+1}",
    desc: "Tại mỗi bước hành động khi agent chọn điểm giao tiếp theo, môi trường kiểm tra xem điểm mới có cùng cụm với điểm vừa ghé không. Hàm chỉ thị 𝕀(c_t ≠ c_{t+1}) nhận giá trị 1 nếu khác cụm, 0 nếu cùng cụm.",
    formulaRole: "Kích hoạt toán tử chỉ thị 𝕀(c_t ≠ c_{t+1}) khi agent vượt ranh giới cụm.",
  },
  reward: {
    title: "Reward Shaping mềm với hệ số phạt λ_c = 0,50",
    badge: "R'_t = R_t − λ_c · 𝕀(c_t ≠ c_{t+1})",
    desc: "Bổ sung tín hiệu thưởng phạt mềm: khi agent chuyển sang cụm khác, reward bị trừ thêm λ_c = 0,50. Đây là reward shaping bảo toàn tính tối ưu, không làm thay đổi tính hợp lệ hay ràng buộc SLA của tuyến.",
    formulaRole: "Hệ số phạt λ_c trực tiếp định hình chính sách: ưu tiên gom đơn cùng cụm trước khi sang quận khác.",
  },
};

/* ─── Count-up hook ─── */
function useCountUp(target: number | null, duration = 1200): number {
  const [value, setValue] = useState(0);
  const rafRef = useRef(0);
  useEffect(() => {
    if (target === null || target === undefined || !Number.isFinite(target)) { setValue(0); return; }
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setValue(target * ease);
      if (t < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);
  return value;
}

/* ─── SLA Gauge ─── */
function SLAGauge({ rate }: { rate: number | null }) {
  const v = useCountUp(rate !== null ? rate * 100 : null);
  const r = 30;
  const c = 2 * Math.PI * r;
  const offset = c - (c * (v / 100));
  return (
    <div className="kpi-gauge">
      <div className="gauge-ring">
        <svg viewBox="0 0 72 72">
          <circle className="bg" cx="36" cy="36" r={r} />
          <circle className="fg" cx="36" cy="36" r={r}
            strokeDasharray={c} strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.8s ease" }}
          />
        </svg>
        <span className="gauge-label">{num(v, 0)}%</span>
      </div>
      <div>
        <div style={{ fontSize: 13, color: "var(--good)", fontWeight: 700 }}>Tỷ lệ đúng hạn (SLA)</div>
        <div style={{ fontSize: 12, color: "var(--muted)" }}>Cả Vanilla và Guided đều đạt 100%</div>
      </div>
    </div>
  );
}

/* ─── Diverging Percentage Bar ─── */
function DivergeBar({
  value,
  label,
  hint,
  forceColor,
  customDisplay,
}: {
  value: number | null;
  label: string;
  hint: string;
  forceColor?: string;
  customDisplay?: string;
}) {
  const isNeg = value !== null && value < 0;
  const absV = value !== null ? Math.abs(value) : 0;
  const maxWidth = Math.min(absV * 10, 50); // scale % display
  const color = forceColor ?? (isNeg ? "var(--good)" : "var(--algo-tradeoff)");
  return (
    <div style={{ padding: "12px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{label}</span>
        <span style={{
          fontSize: 14, fontWeight: 700,
          color,
        }}>
          {customDisplay ?? signedPct(value, 2)}
        </span>
      </div>
      <div className="diverge-bar">
        <div className="track">
          <div
            className="fill"
            style={{
              width: `${maxWidth}%`,
              background: color,
            }}
          />
        </div>
      </div>
      <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{hint}</div>
    </div>
  );
}

type TabKey = "mechanism" | "sensitivity" | "validation" | "proposed";

const TABS: { id: TabKey; label: string }[] = [
  { id: "mechanism", label: "1. Cơ chế K-Means-guided DQN" },
  { id: "sensitivity", label: "2. Độ nhạy λ (Guided DQN)" },
  { id: "validation", label: "3. Kiểm chứng 12 Workload" },
  { id: "proposed", label: "4. Proposed RL Studio (Mô hình Mới)" },
];

type ArchNodeKey = "state" | "encoder" | "network" | "action" | "env" | "reward";

interface ArchNodeInfo {
  key: ArchNodeKey;
  step: string;
  name: string;
  subtitle: string;
  shape: string;
  formula: string;
  description: string;
  codeFile: string;
  codeSymbol: string;
}

const ARCH_NODES: ArchNodeInfo[] = [
  {
    key: "state",
    step: "Bước 1",
    name: "Không gian Trạng thái",
    subtitle: "RoutingState (3 trường)",
    shape: "s_t = (idx, mask, elapsed_bucket)",
    formula: "s_t = (current_index, visited_mask, elapsed_time_bucket)",
    description: "Trạng thái MDP cơ sở gồm đúng 3 trường: current_index (vị trí điểm giao hiện tại hoặc Depot=0), visited_mask (bitmask nhị phân biểu diễn các điểm đã giao), và elapsed_time_bucket (phân đoạn thời gian đã trôi qua). Không chứa trường tải trọng hay GPS trực tiếp trong RoutingState.",
    codeFile: "python/rl/routing_environment.py & proposed/environment.py",
    codeSymbol: "RoutingState / ProposedRoutingEnvironment",
  },
  {
    key: "encoder",
    step: "Bước 2",
    name: "Bộ Mã hóa Ứng viên",
    subtitle: "Candidate Encoder 8D",
    shape: "x_(t, i) ∈ ℝ⁸, ∀ a_i ∈ U_t",
    formula: "x_(t, i) = [elapsed_ratio, remaining_ratio, distance_ratio, travel_time_ratio, slack_ratio, deadline_valid, lateness_ratio, cluster_switch]",
    description: "Ánh xạ từng ứng viên khả thi a_i thành vector đặc trưng 8 chiều chuẩn hóa gồm: elapsed_ratio, remaining_ratio, distance_ratio, travel_time_ratio, slack_ratio, deadline_valid, lateness_ratio, cluster_switch. Thiết kế này giúp mô hình đạt tính chất bất biến với kích thước tập đơn N (khắc phục hoàn toàn giới hạn output cố định của DQN cũ).",
    codeFile: "python/rl/proposed/candidate_encoder.py",
    codeSymbol: "CandidateEncoderConfig / FEATURE_NAMES",
  },
  {
    key: "network",
    step: "Bước 3",
    name: "Mạng Nơ-ron Đánh giá",
    subtitle: "CandidateQNetwork",
    shape: "MLP: 8 → 64 → 64 → 1",
    formula: "Q(s_t, a_i) = f_θ(x_(t, i))",
    description: "Kiến trúc mạng Perceptron đa tầng (MLP) với 2 tầng ẩn ReLU (hidden_dim = 64). Nhận đầu vào là vector 8D của từng ứng viên và xuất ra một giá trị vô hướng Q-value ước lượng chi phí kỳ vọng âm.",
    codeFile: "python/rl/proposed/agent.py",
    codeSymbol: "CandidateQNetwork(nn.Module)",
  },
  {
    key: "action",
    step: "Bước 4",
    name: "Lựa chọn Hành động",
    subtitle: "Chính sách Q-Greedy",
    shape: "a_t ∈ U_t",
    formula: "a_t = argmax_{a_i ∈ U_t} Q(s_t, a_i)",
    description: "Hành động được chọn là ứng viên có điểm số Q cao nhất trong tập các đơn hàng còn lại. Khi huấn luyện, sử dụng cơ chế ε-greedy (khởi đầu ε = 1.0, suy giảm 0.998, tối thiểu 0.05).",
    codeFile: "python/rl/proposed/agent.py",
    codeSymbol: "ProposedRoutingAgent.act",
  },
  {
    key: "env",
    step: "Bước 5",
    name: "Chuyển đổi Môi trường",
    subtitle: "Đồ thị đường OSM TP.HCM",
    shape: "s_(t+1) = Transition(s_t, a_t)",
    formula: "p_(t+1) = p_a, t_(t+1) = t_t + τ(p_t, p_a)",
    description: "Môi trường cập nhật vị trí shipper tới điểm giao mới qua đồ thị mạng đường thực tế OpenStreetMap. Cập nhật số phút trễ, kiểm tra deadline và ghi nhận cụm địa lý c_(t+1).",
    codeFile: "python/rl/proposed/environment.py",
    codeSymbol: "ProposedRoutingEnvironment.step",
  },
  {
    key: "reward",
    step: "Bước 6",
    name: "Phần thưởng Đa mục tiêu",
    subtitle: "Reward = -ΔJ_t",
    shape: "r_t = -(J_t - J_(t-1))",
    formula: "r_t = -ΔJ_t, với γ = 1.0 ⇒ Σ r_t = -J(π)",
    description: "Phần thưởng nhận được tại mỗi bước chính là mức suy giảm của hàm chi phí mục tiêu đa tiêu chí. Với hệ số chiết khấu γ = 1.0 trên chân trời hữu hạn, tổng reward toàn hành trình tương đương cực tiểu hóa J(π).",
    codeFile: "python/rl/proposed/objective.py",
    codeSymbol: "ObjectiveConfig / r_t = -ΔJ_t",
  },
];

const PROPOSED_TEST_BENCHMARK = [
  {
    name: "Clarke–Wright",
    key: "cw",
    color: "#F59E0B",
    dist: 55.58,
    time: 60.74,
    late: 0,
    sla: 1.0,
    totalLatenessMinutes: 0.0,
    switches: 4.25,
    switchRate: 60.91,
    obj: 0.4257,
  },
  {
    name: "DQN (Vanilla)",
    key: "dqn",
    color: "#60A5FA",
    dist: 57.11,
    time: 62.33,
    late: 0,
    sla: 1.0,
    totalLatenessMinutes: 0.0,
    switches: 4.13,
    switchRate: 59.19,
    obj: 0.4352,
  },
  {
    name: "Nearest Neighbor",
    key: "nn",
    color: "#94A3B8",
    dist: 59.58,
    time: 64.95,
    late: 1,
    sla: 0.9924,
    totalLatenessMinutes: 10.465,
    switches: 4.06,
    switchRate: 58.67,
    obj: 0.4515,
  },
  {
    name: "Q-Learning",
    key: "q_learning",
    color: "#10B981",
    dist: 60.04,
    time: 65.43,
    late: 0,
    sla: 1.0,
    totalLatenessMinutes: 0.0,
    switches: 4.19,
    switchRate: 59.67,
    obj: 0.4495,
  },
  {
    name: "SARSA",
    key: "sarsa",
    color: "#8B5CF6",
    dist: 60.81,
    time: 66.21,
    late: 0,
    sla: 1.0,
    totalLatenessMinutes: 0.0,
    switches: 4.19,
    switchRate: 59.81,
    obj: 0.4575,
  },
  {
    name: "Proposed RL",
    key: "proposed_rl",
    color: "var(--accent)",
    dist: 61.90,
    time: 67.52,
    late: 1,
    sla: 0.9924,
    totalLatenessMinutes: 12.720,
    switches: 4.00,
    switchRate: 58.10,
    obj: 0.4649,
  },
];

export default function GuidedDQN() {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get("tab");
  const activeTab: TabKey =
    rawTab === "sensitivity" || rawTab === "do-nhay"
      ? "sensitivity"
      : rawTab === "validation" || rawTab === "kiem-chung" || rawTab === "kiem-chung-12"
        ? "validation"
        : rawTab === "proposed" || rawTab === "proposed-rl"
          ? "proposed"
          : "mechanism";

  const setTab = (t: TabKey) => {
    setSearchParams({ tab: t }, { replace: true });
  };

  const [theme] = useTheme();
  const casingColor = theme === "dark" ? "rgba(11, 19, 33, 0.95)" : "rgba(255, 255, 255, 0.95)";

  const sens = useAsync((s) => api.sensitivity(s), []);
  const val = useAsync((s) => api.finalValidation(s), []);

  const s = val.data?.summary;

  // Selected architecture node in Proposed RL Studio
  const [selectedArchNode, setSelectedArchNode] = useState<ArchNodeKey>("network");

  // Hover & selection state for interactive diagram
  const [selectedPart, setSelectedPart] = useState<string | null>("reward");
  const [hoverPart, setHoverPart] = useState<string | null>(null);
  const activePart = hoverPart ?? selectedPart;

  // Selected workload for cluster sequence preview & map
  const [selectedWlId, setSelectedWlId] = useState<number>(65);
  const [routeViewMode, setRouteViewMode] = useState<"both" | "guided" | "vanilla">("both");

  // Route & stops for selected workload map
  const routeData = useAsync(async (signal) => {
    try {
      return await api.precomputedRoute("kmeans_guided_dqn", selectedWlId, true, signal);
    } catch {
      return await api.precomputedRoute("kmeans_guided_dqn", selectedWlId, false, signal);
    }
  }, [selectedWlId]);

  const stopsData = useAsync<DeliveryRecord[]>(async (signal) => {
    if (!routeData.data?.delivery_order) return [];
    return await Promise.all(
      routeData.data.delivery_order.map((id) => api.delivery(id, signal)),
    );
  }, [routeData.data?.delivery_order?.join(",")]);

  const mapData = useMemo(() => {
    const stops = stopsData.data ?? [];
    if (!stops.length) return null;
    const stopMap = new Map(stops.map((st) => [st.id, st]));

    const guidedOrder = routeData.data?.delivery_order ?? [];
    const guidedStops = guidedOrder
      .map((id) => stopMap.get(id))
      .filter(Boolean) as DeliveryRecord[];

    // Identify stops with valid coordinates
    const validGuidedStops = guidedStops.filter((st) => latlng(st) !== null);
    const missingCoordsCount = guidedStops.length - validGuidedStops.length;

    const guidedSegments: {
      from: [number, number];
      to: [number, number];
      cluster: number;
    }[] = [];
    let prevPt: [number, number] = [DEPOT.lat, DEPOT.lng];
    for (let i = 0; i < validGuidedStops.length; i++) {
      const pt = latlng(validGuidedStops[i]);
      if (!pt) continue;
      const cluster = Number(validGuidedStops[i].cluster_id ?? 0);
      if (validLL(prevPt) && validLL(pt)) {
        guidedSegments.push({ from: prevPt, to: pt, cluster });
      }
      prevPt = pt;
    }
    if (validLL(prevPt) && validGuidedStops.length > 0) {
      guidedSegments.push({
        from: prevPt,
        to: [DEPOT.lat, DEPOT.lng],
        cluster: Number(validGuidedStops[validGuidedStops.length - 1]?.cluster_id ?? 0),
      });
    }

    const switchStops: {
      stop: DeliveryRecord;
      index: number;
      prevCluster: number;
      nextCluster: number;
    }[] = [];
    for (let i = 1; i < validGuidedStops.length; i++) {
      const cPrev = Number(validGuidedStops[i - 1].cluster_id);
      const cCurr = Number(validGuidedStops[i].cluster_id);
      if (cPrev !== cCurr) {
        switchStops.push({
          stop: validGuidedStops[i],
          index: i + 1,
          prevCluster: cPrev,
          nextCluster: cCurr,
        });
      }
    }

    const vanillaOrder = BASELINE_DELIVERY_ORDERS[selectedWlId] ?? guidedOrder;
    const vanillaPts: [number, number][] = [[DEPOT.lat, DEPOT.lng]];
    for (const id of vanillaOrder) {
      const st = stopMap.get(id);
      if (st) {
        const pt = latlng(st);
        if (pt && validLL(pt)) vanillaPts.push(pt);
      }
    }
    if (vanillaPts.length > 1) {
      vanillaPts.push([DEPOT.lat, DEPOT.lng]);
    }

    const fitPts: [number, number][] = [
      [DEPOT.lat, DEPOT.lng],
      ...validGuidedStops.map((st) => latlng(st)).filter((pt): pt is [number, number] => pt !== null && validLL(pt)),
    ];

    return {
      guidedStops: validGuidedStops,
      guidedSegments,
      switchStops,
      vanillaPts,
      fitPts,
      missingCoordsCount,
    };
  }, [stopsData.data, routeData.data, selectedWlId]);

  // Lambda slider
  const lambdaValues = useMemo(() =>
    sens.data?.summary.map((r: SensitivityRow) => r.cluster_switch_penalty) ?? [],
    [sens.data],
  );
  const [selectedLambda, setSelectedLambda] = useState(0.5);
  const selectedRow = useMemo(() =>
    sens.data?.summary.find((r: SensitivityRow) => r.cluster_switch_penalty === selectedLambda),
    [sens.data, selectedLambda],
  );

  // Workload details from paired data
  const pairedWorkloads = val.data?.paired_workloads ?? [];
  const currentWl = useMemo(
    () => pairedWorkloads.find((w) => Number(w.workload_id) === selectedWlId) ?? pairedWorkloads[0],
    [pairedWorkloads, selectedWlId],
  );

  // Count-up KPIs
  const distDelta = n(s?.ratio_of_means_distance_improvement_pct);
  const timeDelta = n(s?.ratio_of_means_time_improvement_pct);
  const switchReduction = n(s?.relative_switch_rate_reduction_pct_where_available);

  // Parse cluster sequence for Guided DQN
  const guidedSeq = useMemo(() => {
    if (!currentWl?.cluster_sequence) return [];
    return String(currentWl.cluster_sequence).split("->").map((item) => Number(item.trim()));
  }, [currentWl]);

  // Derive Vanilla DQN sequence for the selected workload
  const vanillaSeq = useMemo(() => {
    if (!currentWl) return null;
    const widNum = Number(currentWl.workload_id);
    if (BASELINE_SEQUENCES[widNum]) {
      return BASELINE_SEQUENCES[widNum];
    }
    // If distance delta is 0, Vanilla route equals Guided route
    if (Math.abs(Number(currentWl.distance_improvement_pct ?? 0)) <= 1e-4 && guidedSeq.length > 0) {
      return guidedSeq;
    }
    return null;
  }, [currentWl, guidedSeq]);

  // Shorter / Equal / Longer counts across 12 paired workloads
  const betterCount = useMemo(
    () => pairedWorkloads.filter((w) => Number(w.distance_improvement_pct ?? 0) > 1e-4).length,
    [pairedWorkloads],
  );
  const tieCount = useMemo(
    () => pairedWorkloads.filter((w) => Math.abs(Number(w.distance_improvement_pct ?? 0)) <= 1e-4).length,
    [pairedWorkloads],
  );
  const worseCount = useMemo(
    () => pairedWorkloads.filter((w) => Number(w.distance_improvement_pct ?? 0) < -1e-4).length,
    [pairedWorkloads],
  );
  const totalCount = pairedWorkloads.length || 12;
  const betterPct = totalCount ? (betterCount / totalCount) * 100 : 0;
  const tiePct = totalCount ? (tieCount / totalCount) * 100 : 0;
  const worsePct = totalCount ? (worseCount / totalCount) * 100 : 0;

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Nghiên cứu học tăng cường Guided DQN"
        subtitle="Cơ chế phạt chuyển cụm địa bàn K-Means λ = 0,50 và mô hình Proposed RL mạng Q đa mục tiêu trên đồ thị TP.HCM"
      />

      <div className="space-y-4">
        {/* Tab switcher using official shadcn Tabs */}
        <div className="flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between border-b border-border/50 pb-2.5">
          <Tabs value={activeTab} onValueChange={(t) => setTab(t as TabKey)}>
            <TabsList className="h-8">
              {TABS.map((t) => (
                <TabsTrigger key={t.id} value={t.id} className="text-xs font-medium">
                  {t.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="outline" className="text-xs py-0.5 px-2 font-mono text-primary border-primary/30">
              {"R' = R − λ_c · 𝕀(c_t ≠ c_t+1)"}
            </Badge>
          </div>
        </div>

        {(sens.loading || val.loading) && <LoadingCard message="Đang nạp dữ liệu thực nghiệm RL..." />}
        {sens.error && <ErrorCard message={sens.error} />}
        {val.error && <ErrorCard message={val.error} />}

        {/* ══════════════ TAB 1: CƠ CHẾ ══════════════ */}
        {activeTab === "mechanism" && (
          <div className="stack">
            {/* Interactive Integration Diagram */}
            <Card
              title="Cơ chế tích hợp K-Means vào hàm thưởng RL"
              sub="Bấm hoặc rê chuột vào từng khối để làm sáng phần tương ứng trong công thức toán học và xem giải thích chi tiết"
            >
              <div className="integ-diagram">
                <div
                  className={`integ-block ${activePart === "data" ? "highlight" : ""}`}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSelectedPart(selectedPart === "data" ? null : "data")}
                  onMouseEnter={() => setHoverPart("data")}
                  onMouseLeave={() => setHoverPart(null)}
                >
                  📦 <b>Tọa độ giao hàng</b>
                  <br /><small style={{ color: "var(--muted)" }}>2.394 đơn XeDu</small>
                </div>
                <span className="integ-arrow">→</span>
                <div
                  className={`integ-block ${activePart === "kmeans" ? "highlight" : ""}`}
                  style={{ borderColor: "var(--algo-cw)", cursor: "pointer" }}
                  onClick={() => setSelectedPart(selectedPart === "kmeans" ? null : "kmeans")}
                  onMouseEnter={() => setHoverPart("kmeans")}
                  onMouseLeave={() => setHoverPart(null)}
                >
                  🔵 <b>K-Means (K=10)</b>
                  <br /><small style={{ color: "var(--algo-cw)" }}>gán cluster_id (c)</small>
                </div>
                <span className="integ-arrow">→</span>
                <div
                  className={`integ-block ${activePart === "env" ? "highlight" : ""}`}
                  style={{ borderColor: GUIDED_COLOR, cursor: "pointer" }}
                  onClick={() => setSelectedPart(selectedPart === "env" ? null : "env")}
                  onMouseEnter={() => setHoverPart("env")}
                  onMouseLeave={() => setHoverPart(null)}
                >
                  🧠 <b>Môi trường RL</b>
                  <br /><small style={{ color: GUIDED_COLOR }}>đo c_t vs c_t+1</small>
                </div>
                <span className="integ-arrow">→</span>
                <div
                  className={`integ-block ${activePart === "reward" ? "highlight" : ""}`}
                  style={{ borderColor: TRADEOFF_COLOR, cursor: "pointer" }}
                  onClick={() => setSelectedPart(selectedPart === "reward" ? null : "reward")}
                  onMouseEnter={() => setHoverPart("reward")}
                  onMouseLeave={() => setHoverPart(null)}
                >
                  ⚡ <b>Reward Shaping</b>
                  <br /><small style={{ color: TRADEOFF_COLOR }}>phạt trừ -λ_c khi đổi cụm</small>
                </div>
              </div>

              <div className="formula-block" style={{ marginTop: 14, overflowX: "auto" }}>
                <span style={{ color: activePart === "reward" ? "var(--good)" : undefined }}>R&apos;<sub>t</sub></span>
                {" = "}
                <span style={{ opacity: activePart && activePart !== "reward" ? 0.4 : 1 }}>R<sub>t</sub></span>
                {" − "}
                <span
                  className="highlight"
                  style={{
                    background: activePart === "reward" ? "rgba(251, 191, 36, 0.25)" : undefined,
                    padding: "2px 6px",
                    borderRadius: 4,
                  }}
                >
                  λ<sub>c</sub>
                </span>
                {" · "}
                <span
                  style={{
                    color: activePart === "env" || activePart === "kmeans" ? "var(--accent)" : undefined,
                    fontWeight: activePart === "env" || activePart === "kmeans" ? 800 : undefined,
                    background: activePart === "env" ? "rgba(45, 212, 191, 0.15)" : undefined,
                    padding: "2px 6px",
                    borderRadius: 4,
                  }}
                >
                  𝕀(c<sub>t</sub> ≠ c<sub>t+1</sub>)
                </span>
                <span
                  className="glossary"
                  data-tip="λc = cluster_switch_penalty. Khi agent chuyển sang đơn thuộc cụm khác, reward bị trừ thêm λc. Đây là reward shaping mềm, không thay đổi tính khả thi của tuyến."
                >
                  i
                </span>
              </div>

              {activePart && PART_EXPLANATIONS[activePart] && (
                <div
                  style={{
                    marginTop: 14,
                    padding: "12px 16px",
                    background: "var(--surface-2)",
                    borderRadius: "var(--radius)",
                    border: "1px solid var(--border)",
                    display: "grid",
                    gap: 6,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                    <span style={{ fontWeight: 700, fontSize: 13.5, color: "var(--text)" }}>
                      {PART_EXPLANATIONS[activePart].title}
                    </span>
                    <span className="chip" style={{ background: "rgba(139, 140, 248, 0.12)", color: GUIDED_COLOR, fontSize: 11 }}>
                      {PART_EXPLANATIONS[activePart].badge}
                    </span>
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    {PART_EXPLANATIONS[activePart].desc}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--muted)", borderTop: "1px solid var(--border)", paddingTop: 6 }}>
                    <b>Ý nghĩa toán học:</b> {PART_EXPLANATIONS[activePart].formulaRole}
                  </div>
                </div>
              )}
            </Card>

            <details
              className="card"
              style={{
                padding: "14px 18px",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                cursor: "pointer",
              }}
            >
              <summary style={{ fontWeight: 700, fontSize: 13.5, color: "var(--text)", outline: "none", userSelect: "none" }}>
                📖 Xem chi tiết 3 luận điểm khoa học & nguyên lý đánh đổi thực nghiệm
              </summary>
              <div className="grid cols-3" style={{ alignItems: "stretch", marginTop: 14 }}>
                <div style={{ padding: 14, background: "var(--surface-2)", borderRadius: "var(--radius)" }}>
                  <div style={{ fontWeight: 700, color: "var(--accent)", marginBottom: 6 }}>1. Tín hiệu không gian</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    K-Means phân vùng 2.394 đơn thành 10 cụm địa lý tự nhiên. Khi agent DQN học cách di chuyển, việc đưa cluster_id vào reward giúp định hướng agent ưu tiên giao hết các đơn trong cùng cụm trước khi sang cụm mới.
                  </div>
                </div>
                <div style={{ padding: 14, background: "var(--surface-2)", borderRadius: "var(--radius)" }}>
                  <div style={{ fontWeight: 700, color: "var(--algo-tradeoff)", marginBottom: 6 }}>2. Bản chất đánh đổi</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Guided DQN <b>không làm tuyến ngắn hơn</b>. Việc gom cụm làm tăng nhẹ quãng đường (+{num(distDelta, 2)}%) và thời gian (+{num(timeDelta, 2)}%) nhưng đổi lại {switchReduction !== null && switchReduction < 0 ? `giảm tương đối ${num(Math.abs(switchReduction), 2)}%` : switchReduction !== null ? `giảm ${signedPct(switchReduction, 2)}` : "–"} số lần đổi cụm, giúp lộ trình shipper gom gọn gàng, hạn chế di chuyển qua lại giữa các quận.
                  </div>
                </div>
                <div style={{ padding: 14, background: "var(--surface-2)", borderRadius: "var(--radius)" }}>
                  <div style={{ fontWeight: 700, color: "var(--good)", marginBottom: 6 }}>3. Bảo toàn chất lượng dịch vụ</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Tỷ lệ giao hàng đúng hạn (SLA) vẫn đạt tuyệt đối <b>100%</b> trên cả {num(n(s?.workloads ?? 12), 0)} workload kiểm chứng ({num(n(s?.deliveries ?? 108), 0)} đơn hàng), chứng minh việc định hướng cụm không vi phạm các ràng buộc cửa sổ thời gian giao hàng.
                  </div>
                </div>
              </div>
            </details>

            {/* Comparative Architectural Specification: RoutingState vs Candidate Encoder */}
            <Card className="border-border/80 shadow-xs">
              <CardHeader className="p-4 sm:px-5 pb-2">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <span className="size-2 rounded-full bg-primary" />
                  <span>Mô hình hóa MDP: So sánh Không gian Trạng thái Giữa Hai Hệ thống RL</span>
                </CardTitle>
                <CardDescription className="text-xs">
                  Phân định tường minh giữa Trạng thái Môi trường MDP Cơ sở (RoutingState) và Bộ Mã hóa Ứng viên 8D (Proposed RL Candidate Encoder).
                </CardDescription>
              </CardHeader>
              <CardContent className="p-4 sm:px-5 pt-2 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* System A: Main RL RoutingState */}
                  <div className="rounded-lg border border-border/70 bg-card p-3.5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground">1. Môi trường MDP Cơ sở (RoutingState)</span>
                      <Badge variant="outline" className="text-[10px] font-mono">3 trường dữ liệu</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Sử dụng cho các mô hình RL truyền thống (Q-Learning, SARSA, Vanilla DQN). Trạng thái được mô hình hóa theo tiến trình rời rạc:
                    </p>
                    <div className="space-y-1.5 font-mono text-[11px]">
                      <div className="p-1.5 rounded bg-muted/40 border border-border/40 flex items-center justify-between">
                        <span className="text-primary font-bold">current_index</span>
                        <span className="text-muted-foreground text-[10px]">Vị trí trạm hiện tại (hoặc 0 = Depot)</span>
                      </div>
                      <div className="p-1.5 rounded bg-muted/40 border border-border/40 flex items-center justify-between">
                        <span className="text-primary font-bold">visited_mask</span>
                        <span className="text-muted-foreground text-[10px]">Bitmask nhị phân các điểm đã giao</span>
                      </div>
                      <div className="p-1.5 rounded bg-muted/40 border border-border/40 flex items-center justify-between">
                        <span className="text-primary font-bold">elapsed_time_bucket</span>
                        <span className="text-muted-foreground text-[10px]">Phân đoạn thời gian hành trình đã trôi qua</span>
                      </div>
                    </div>
                    <div className="text-[10px] text-muted-foreground italic border-t border-border/40 pt-1.5">
                      * Khẳng định khoa học: RoutingState không chứa tải trọng còn lại hay tọa độ GPS thô làm trường trạng thái trực tiếp.
                    </div>
                  </div>

                  {/* System B: Proposed RL Candidate Encoder */}
                  <div className="rounded-lg border border-border/70 bg-card p-3.5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground">2. Bộ Mã hóa Ứng viên (Candidate Encoder)</span>
                      <Badge variant="outline" className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400 border-emerald-500/30">8 Đặc trưng Chuẩn hóa</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Sử dụng cho Proposed RL. Mã hóa vector đặc trưng 8D tương đối cho từng đơn hàng ứng viên khả thi a_i ∈ U_t:
                    </p>
                    <div className="grid grid-cols-2 gap-1.5 font-mono text-[10px]">
                      <div className="p-1 rounded bg-muted/30 border border-border/40">1. elapsed_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">2. remaining_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">3. distance_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">4. travel_time_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">5. slack_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">6. deadline_valid</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">7. lateness_ratio</div>
                      <div className="p-1 rounded bg-muted/30 border border-border/40">8. cluster_switch</div>
                    </div>
                    <div className="text-[10px] text-muted-foreground italic border-t border-border/40 pt-1.5">
                      * Tính bất biến quy mô: Giúp mạng Q xử lý linh hoạt mọi số lượng đơn hàng N mà không bị gò bó output cố định.
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ══════════════ TAB 2: ĐỘ NHẠY λ ══════════════ */}
        {activeTab === "sensitivity" && sens.data && (
          <Card
            title="Khảo sát độ nhạy hệ số phạt λ (cluster_switch_penalty)"
            sub={`Chạy trên 6 workload tiêu biểu, λ ∈ {0, 0.10, 0.25, 0.50}. Mức λ = ${num(sens.data.selected_lambda, 2)} được chọn tối ưu.`}
          >
            <div className="lambda-slider">
              <span style={{ fontSize: 13, color: "var(--muted)", fontWeight: 600 }}>Chọn mức λ:</span>
              <input
                type="range"
                min={0}
                max={lambdaValues.length - 1}
                step={1}
                value={lambdaValues.indexOf(selectedLambda)}
                onChange={(e) => setSelectedLambda(lambdaValues[Number(e.target.value)])}
                aria-label="Chọn hệ số phạt λ"
              />
              <span className="lambda-val">{num(selectedLambda, 2)}</span>
            </div>

            {selectedRow && (
              <div className="grid cols-4" style={{ marginTop: 16 }}>
                <div className="stat tone-teal">
                  <div className="label">Quãng đường TB</div>
                  <div className="value">{num(selectedRow.mean_distance_km, 2)}</div>
                  <div className="hint">km / workload</div>
                </div>
                <div className="stat tone-blue">
                  <div className="label">Thời gian TB</div>
                  <div className="value">{num(selectedRow.mean_travel_time_minutes, 2)}</div>
                  <div className="hint">phút / workload</div>
                </div>
                <div className="stat tone-rose">
                  <div className="label">Đổi cụm TB</div>
                  <div className="value">{num(selectedRow.mean_cluster_switches, 2)}</div>
                  <div className="hint">lần ({pct(selectedRow.mean_cluster_switch_rate, 1)})</div>
                </div>
                <div className="stat tone-violet">
                  <div className="label">Đúng hạn (SLA)</div>
                  <div className="value">{pct(selectedRow.overall_on_time_rate, 0)}</div>
                  <div className="hint">{num(selectedRow.total_late_deliveries, 0)} đơn trễ</div>
                </div>
              </div>
            )}

            <div className="grid cols-2" style={{ marginTop: 20 }}>
              <div>
                <h3 style={{ fontSize: 14, marginBottom: 12, color: "var(--text-secondary)" }}>Quãng đường theo λ</h3>
                <div style={{ height: 220 }}>
                  <ResponsiveContainer>
                    <LineChart data={sens.data.summary} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" />
                      <XAxis
                        dataKey="cluster_switch_penalty"
                        tick={{ fontSize: 12 }}
                        label={{ value: "λ", position: "insideBottomRight", offset: -4, fill: "var(--muted)" }}
                      />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} width={56} />
                      <Tooltip />
                      <Line
                        type="linear"
                        dataKey="mean_distance_km"
                        name="Quãng đường TB (km)"
                        stroke={VANILLA_COLOR}
                        strokeWidth={2.5}
                        dot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="axis-note">Trục không bắt đầu từ 0</div>
              </div>
              <div>
                <h3 style={{ fontSize: 14, marginBottom: 12, color: "var(--text-secondary)" }}>Số lần đổi cụm theo λ</h3>
                <div style={{ height: 220 }}>
                  <ResponsiveContainer>
                    <LineChart data={sens.data.summary} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" />
                      <XAxis
                        dataKey="cluster_switch_penalty"
                        tick={{ fontSize: 12 }}
                        label={{ value: "λ", position: "insideBottomRight", offset: -4, fill: "var(--muted)" }}
                      />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} width={56} />
                      <Tooltip />
                      <Line
                        type="linear"
                        dataKey="mean_cluster_switches"
                        name="Đổi cụm TB (lần)"
                        stroke={GUIDED_COLOR}
                        strokeWidth={2.5}
                        dot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="axis-note">Trục không bắt đầu từ 0</div>
              </div>
            </div>

            {/* Sensitivity table */}
            <div className="table-wrap" style={{ marginTop: 16 }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="num">λ</th>
                    <th className="num">Lượt chạy</th>
                    <th className="num">Quãng đường (km)</th>
                    <th className="num">Thời gian (phút)</th>
                    <th className="num">Đúng hạn</th>
                    <th className="num">Đổi cụm TB</th>
                    <th className="num">Tỷ lệ đổi cụm</th>
                    <th className="num">Huấn luyện (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {sens.data.summary.map((r: SensitivityRow) => (
                    <tr
                      key={r.cluster_switch_penalty}
                      className={r.cluster_switch_penalty === selectedLambda ? "sel" : ""}
                      style={{ cursor: "pointer" }}
                      onClick={() => setSelectedLambda(r.cluster_switch_penalty)}
                    >
                      <td className="num" style={{ fontWeight: r.cluster_switch_penalty === sens.data!.selected_lambda ? 800 : 400 }}>
                        {num(r.cluster_switch_penalty, 2)}
                        {r.cluster_switch_penalty === sens.data!.selected_lambda && (
                          <span style={{ marginLeft: 6, fontSize: 10, color: GUIDED_COLOR }}>✓ chọn</span>
                        )}
                      </td>
                      <td className="num">{num(r.runs, 0)}</td>
                      <td className="num">{num(r.mean_distance_km, 2)}</td>
                      <td className="num">{num(r.mean_travel_time_minutes, 2)}</td>
                      <td className="num">{pct(r.overall_on_time_rate, 0)}</td>
                      <td className="num">{num(r.mean_cluster_switches, 2)}</td>
                      <td className="num">{pct(r.mean_cluster_switch_rate, 1)}</td>
                      <td className="num">{num(r.mean_training_runtime_seconds, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* ══════════════ TAB 3: KIỂM CHỨNG 12 WORKLOAD ══════════════ */}
        {activeTab === "validation" && s && (
          <>
            <Card
              title="Kiểm chứng cuối: 12 workload / 108 đơn"
              sub={`Vanilla DQN (λ=0) so với K-Means-guided DQN (λ=${num(n(s.guided_lambda), 2)})`}
            >
              {/* KPI section with count-up */}
              <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 1fr 1fr", gap: 20, marginBottom: 14 }}>
                <SLAGauge rate={n(s.guided_weighted_on_time_rate)} />
                <DivergeBar
                  value={distDelta !== null ? -distDelta : null}
                  label="Δ Quãng đường (G − V)"
                  hint={`Dương (+) = Guided dài hơn (+${num(distDelta, 2)}% đánh đổi)`}
                  forceColor="var(--algo-tradeoff)"
                />
                <DivergeBar
                  value={timeDelta !== null ? -timeDelta : null}
                  label="Δ Thời gian (G − V)"
                  hint={`Dương (+) = Guided chậm hơn (+${num(timeDelta, 2)}% đánh đổi)`}
                  forceColor="var(--algo-tradeoff)"
                />
                <DivergeBar
                  value={switchReduction !== null ? -switchReduction : null}
                  label="Δ Tỷ lệ đổi cụm (G − V)"
                  hint={switchReduction !== null ? `Giảm ${signedPct(switchReduction, 2)} (trên ${num(Number(s.workloads_with_baseline_sequences ?? 6), 0)} WL đối sánh)` : "–"}
                  forceColor="var(--good)"
                  customDisplay={switchReduction !== null ? signedPct(-switchReduction, 2) : undefined}
                />
              </div>

              {/* Stacked distribution bar: Shorter / Equal / Longer */}
              <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: "var(--radius)", marginBottom: 18 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                  <span>Phân bố so sánh quãng đường trên 12 workload:</span>
                  <span style={{ color: "var(--accent)", fontWeight: 700 }}>
                    {tieCount}/{totalCount} workload cho tuyến giống hệt Vanilla ({pct(tiePct / 100, 1)})
                  </span>
                </div>
                <div style={{ display: "flex", height: 22, borderRadius: 6, overflow: "hidden", gap: 2 }}>
                  <div
                    style={{ width: `${betterPct}%`, background: "var(--good)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 11, fontWeight: 700 }}
                    title={`Ngắn hơn (tốt hơn về km): ${betterCount}/${totalCount} (${num(betterPct, 1)}%)`}
                  >
                    {betterCount > 0 ? `${betterCount} ngắn (${num(betterPct, 1)}%)` : ""}
                  </div>
                  <div
                    style={{ width: `${tiePct}%`, background: "var(--algo-cw)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 11, fontWeight: 700 }}
                    title={`Bằng nhau (trùng tuyến): ${tieCount}/${totalCount} (${num(tiePct, 1)}%)`}
                  >
                    {tieCount > 0 ? `${tieCount} giống hệt (${num(tiePct, 1)}%)` : ""}
                  </div>
                  <div
                    style={{ width: `${worsePct}%`, background: "var(--algo-tradeoff)", display: "flex", alignItems: "center", justifyContent: "center", color: "#000", fontSize: 11, fontWeight: 700 }}
                    title={`Dài hơn (đánh đổi gom cụm): ${worseCount}/${totalCount} (${num(worsePct, 1)}%)`}
                  >
                    {worseCount > 0 ? `${worseCount} dài (${num(worsePct, 1)}%)` : ""}
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 8, fontSize: 12, color: "var(--text-secondary)" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--good)" }} />
                    Ngắn hơn (WL #90): <b>{betterCount} WL ({pct(betterCount / totalCount, 1)})</b>
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--algo-cw)" }} />
                    Bằng nhau (tuyến giống hệt): <b>{tieCount} WL ({pct(tieCount / totalCount, 1)})</b>
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--algo-tradeoff)" }} />
                    Dài hơn (đánh đổi gom cụm): <b>{worseCount} WL ({pct(worseCount / totalCount, 1)})</b>
                  </span>
                </div>
              </div>

              {/* Summary comparison table */}
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Chỉ tiêu</th>
                      <th className="num">
                        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 3, background: VANILLA_COLOR, marginRight: 6 }} />
                        Vanilla DQN (Cơ sở)
                      </th>
                      <th className="num">
                        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 3, background: GUIDED_COLOR, marginRight: 6 }} />
                        K-Means-guided DQN (Đề xuất)
                      </th>
                      <th className="num">Chênh lệch Δ (Guided − Vanilla)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Quãng đường TB (km)</td>
                      <td className="num">{num(n(s.mean_baseline_distance_km), 2)}</td>
                      <td className="num">{num(n(s.mean_guided_distance_km), 2)}</td>
                      <td className="num" style={{ color: "var(--algo-tradeoff)", fontWeight: 700 }}>
                        {signedPct(distDelta !== null ? -distDelta : null, 2)}
                        <span style={{ marginLeft: 4, fontSize: 10 }}>↑ đánh đổi nhẹ</span>
                      </td>
                    </tr>
                    <tr>
                      <td>Thời gian TB (phút)</td>
                      <td className="num">{num(n(s.mean_baseline_travel_time_minutes), 2)}</td>
                      <td className="num">{num(n(s.mean_guided_travel_time_minutes), 2)}</td>
                      <td className="num" style={{ color: "var(--algo-tradeoff)", fontWeight: 700 }}>
                        {signedPct(timeDelta !== null ? -timeDelta : null, 2)}
                        <span style={{ marginLeft: 4, fontSize: 10 }}>↑ đánh đổi nhẹ</span>
                      </td>
                    </tr>
                    <tr>
                      <td>Giao đúng hạn (SLA)</td>
                      <td className="num" style={{ color: "var(--good)", fontWeight: 700 }}>{pct(n(s.baseline_weighted_on_time_rate), 1)}</td>
                      <td className="num" style={{ color: "var(--good)", fontWeight: 700 }}>{pct(n(s.guided_weighted_on_time_rate), 1)}</td>
                      <td className="num" style={{ color: "var(--good)" }}>= 100% giữ nguyên</td>
                    </tr>
                    <tr>
                      <td>
                        Tỷ lệ đổi cụm
                        <span className="glossary" data-tip="Chỉ tính trên 6 workload có lưu tuyến DQN thường. Tỷ lệ = số lần chuyển cụm / (số điểm giao - 1).">i</span>
                        <span style={{ fontSize: 11, color: "var(--muted)", marginLeft: 4 }}>
                          ({num(n(s.switch_comparison_workloads), 0)} WL đối sánh)
                        </span>
                      </td>
                      <td className="num">{pct(n(s.mean_baseline_switch_rate_where_available), 1)}</td>
                      <td className="num" style={{ color: "var(--good)", fontWeight: 700 }}>
                        {pct(n(s.mean_guided_switch_rate_where_available), 1)}
                      </td>
                      <td className="num" style={{ color: "var(--good)", fontWeight: 700 }}>
                        {signedPct(switchReduction !== null ? -switchReduction : null, 1)}
                        <span style={{ marginLeft: 4, fontSize: 10 }}>↓ giảm đổi cụm</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Card>

            {/* ─── Cluster Sequence Visualizer: 2 Rows Side-by-Side ─── */}
            {currentWl && (
              <Card
                title={`Khảo sát chuỗi cụm thực tế: Workload #${currentWl.workload_id}`}
                sub={`Số điểm giao: ${currentWl.number_of_deliveries} đơn | Δ quãng đường (G − V): ${signedPct(n(currentWl.distance_improvement_pct) !== null ? -(n(currentWl.distance_improvement_pct) as number) : null, 2)}${(n(currentWl.distance_improvement_pct) ?? 0) < 0 ? " (Guided dài hơn)" : (n(currentWl.distance_improvement_pct) ?? 0) > 0 ? " (Guided ngắn hơn)" : " (Tuyến trùng khớp)"}`}
              >
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 13, color: "var(--muted)", fontWeight: 600 }}>Xem workload:</span>
                  {[65, 71, 90, 105, 42, 29].map((id) => (
                    <button
                      key={id}
                      className={`btn ${selectedWlId === id ? "" : "ghost"}`}
                      style={{ padding: "4px 10px", fontSize: 12 }}
                      onClick={() => setSelectedWlId(id)}
                    >
                      WL #{id}
                    </button>
                  ))}
                </div>

                <div style={{ display: "grid", gap: 14 }}>
                  {/* Row 1: Vanilla DQN Sequence */}
                  <div style={{ background: "var(--surface-2)", padding: 14, borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: VANILLA_COLOR }}>
                        1. Vanilla DQN (Cơ sở, không hướng dẫn cụm):
                      </span>
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        {currentWl.baseline_cluster_switches !== undefined && currentWl.baseline_cluster_switches !== null
                          ? `Đổi cụm: ${currentWl.baseline_cluster_switches} lần`
                          : vanillaSeq
                            ? `Đổi cụm: ${vanillaSeq.filter((c, i) => i > 0 && c !== vanillaSeq[i - 1]).length} lần`
                            : "Không lưu chuỗi cụm chi tiết"}
                      </span>
                    </div>

                    {vanillaSeq ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                        <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 11.5, fontWeight: 700 }}>
                          Depot
                        </span>
                        {vanillaSeq.map((cId, idx) => {
                          const isSwitched = idx > 0 && vanillaSeq[idx - 1] !== cId;
                          return (
                            <span key={`vanilla-${idx}`} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                              <span style={{ color: "var(--muted)", fontSize: 12 }}>→</span>
                              <span
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: 5,
                                  padding: "3px 8px",
                                  borderRadius: 6,
                                  background: "var(--surface)",
                                  border: isSwitched ? "1px solid var(--border)" : "1px solid var(--border)",
                                }}
                              >
                                <span className="swatch" style={{ background: clusterColor(cId), width: 9, height: 9, borderRadius: 2 }} />
                                <span style={{ fontSize: 12 }}>Cụm {cId}</span>
                                {isSwitched && <span style={{ fontSize: 10, color: "var(--muted)" }}>⚡</span>}
                              </span>
                            </span>
                          );
                        })}
                        <span style={{ color: "var(--muted)", fontSize: 12 }}>→</span>
                        <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 11.5, fontWeight: 700 }}>
                          Depot
                        </span>
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: "var(--muted)", fontStyle: "italic" }}>
                        Chỉ lưu tổng quãng đường và thời gian đối sánh, không lưu chuỗi tọa độ chi tiết.
                      </div>
                    )}
                  </div>

                  {/* Row 2: Guided DQN Sequence */}
                  <div style={{ background: "var(--surface-2)", padding: 14, borderRadius: "var(--radius)", border: `1px solid ${GUIDED_COLOR}40` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: GUIDED_COLOR }}>
                        2. K-Means-guided DQN (Đề xuất, λ = 0,50):
                      </span>
                      <span style={{ fontSize: 12, color: "var(--accent)", fontWeight: 600 }}>
                        Đổi cụm: {currentWl.guided_cluster_switches} lần ({pct(n(currentWl.guided_cluster_switch_rate), 1)})
                      </span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                      <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 11.5, fontWeight: 700 }}>
                        Depot
                      </span>
                      {guidedSeq.map((cId, idx) => {
                        const isSwitched = idx > 0 && guidedSeq[idx - 1] !== cId;
                        return (
                          <span key={`guided-${idx}`} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                            <span style={{ color: "var(--muted)", fontSize: 12 }}>→</span>
                            <span
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 5,
                                padding: "3px 8px",
                                borderRadius: 6,
                                background: "var(--surface)",
                                border: isSwitched ? "1px solid var(--algo-tradeoff)" : "1px solid var(--border)",
                                boxShadow: isSwitched ? "0 0 6px rgba(251, 191, 36, 0.25)" : undefined,
                              }}
                            >
                              <span className="swatch" style={{ background: clusterColor(cId), width: 9, height: 9, borderRadius: 2 }} />
                              <b style={{ fontSize: 12 }}>Cụm {cId}</b>
                              {isSwitched && (
                                <span style={{ fontSize: 10, color: "var(--algo-tradeoff)", fontWeight: 700 }}>⚡ Đổi</span>
                              )}
                            </span>
                          </span>
                        );
                      })}
                      <span style={{ color: "var(--muted)", fontSize: 12 }}>→</span>
                      <span style={{ padding: "3px 8px", borderRadius: 6, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 11.5, fontWeight: 700 }}>
                        Depot
                      </span>
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 20, marginTop: 12, fontSize: 12.5, color: "var(--text-secondary)" }}>
                  <span>Số đơn giao: <b>{currentWl.number_of_deliveries}</b></span>
                  <span>Quãng đường Guided: <b>{num(n(currentWl.guided_distance_km), 2)} km</b></span>
                  <span>Quãng đường Vanilla: <b>{num(n(currentWl.baseline_distance_km), 2)} km</b></span>
                  {Math.abs(Number(currentWl.distance_improvement_pct ?? 0)) <= 1e-4 && (
                    <span style={{ color: "var(--accent)", fontWeight: 600 }}>
                      ✓ Tuyến hai thuật toán trùng khớp hoàn toàn
                    </span>
                  )}
                </div>
              </Card>
            )}

            {/* ─── Interactive Comparison Map ─── */}
            <Card
              title={`Bản đồ đối sánh lộ trình thực nghiệm: Workload #${selectedWlId}`}
              sub="So sánh không gian giữa lộ trình Vanilla DQN và K-Means-guided DQN. Các đoạn tuyến Guided được tô màu theo cụm; điểm đổi cụm có hiệu ứng ripple."
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 10 }}>
                <Tabs
                  value={routeViewMode}
                  onValueChange={(v) => setRouteViewMode(v as any)}
                >
                  <TabsList className="h-8">
                    <TabsTrigger value="both" className="text-xs">
                      Cả hai (chồng tuyến)
                    </TabsTrigger>
                    <TabsTrigger value="guided" className="text-xs">
                      Guided DQN (tô màu cụm)
                    </TabsTrigger>
                    <TabsTrigger value="vanilla" className="text-xs">
                      Vanilla DQN (nét đứt xám)
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                <div style={{ display: "flex", gap: 14, alignItems: "center", fontSize: 12, color: "var(--text-secondary)", flexWrap: "wrap" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 14, height: 3, borderTop: "2px dashed #94a3b8", display: "inline-block" }} />
                    Vanilla DQN (Cơ sở)
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 14, height: 4, background: GUIDED_COLOR, borderRadius: 2 }} />
                    Guided DQN (Theo cụm)
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--algo-tradeoff)", boxShadow: "0 0 4px var(--algo-tradeoff)" }} />
                    Điểm đổi cụm (ripple)
                  </span>
                </div>
              </div>

              {routeData.error ? (
                <div className="note warn" style={{ margin: "12px 0" }}>
                  Workload #{selectedWlId} không có tuyến của kmeans_guided_dqn.
                  ({String(routeData.error)})
                </div>
              ) : routeData.loading || stopsData.loading ? (
                <div style={{ height: 500, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <LoadingCard message="Đang nạp tọa độ các điểm giao thực tế..." />
                </div>
              ) : mapData ? (
                <MapErrorBoundary>
                  {mapData.missingCoordsCount > 0 && (
                    <div className="note warn" style={{ marginBottom: 12 }}>
                      Workload này có {mapData.missingCoordsCount} điểm giao thiếu tọa độ và đã được loại khỏi tuyến vẽ.
                    </div>
                  )}

                  {mapData.guidedStops.length < 2 ? (
                    <div style={{ padding: 36, textAlign: "center", color: "var(--muted)", background: "var(--surface-2)", borderRadius: "var(--radius)" }}>
                      Workload này không đủ số điểm có tọa độ hợp lệ (tối thiểu 2 điểm) để vẽ tuyến giao hàng.
                    </div>
                  ) : (
                    <div style={{ width: "100%", height: 520, position: "relative" }}>
                      <MapBase height="520px" fit={mapData.fitPts.length > 1 ? mapData.fitPts : undefined}>
                        {/* Kho xuất phát */}
                        <Marker position={[DEPOT.lat, DEPOT.lng]} icon={depotIcon()}>
                          <Popup>
                            <div style={{ padding: 4 }}>
                              <b style={{ color: "#0d9488" }}>{DEPOT.name}</b>
                              <p style={{ margin: "4px 0 0", fontSize: "12px", color: "var(--muted)" }}>
                                Kho tập kết xuất phát và kết thúc lộ trình giao hàng
                              </p>
                            </div>
                          </Popup>
                        </Marker>

                        {/* Tuyến Vanilla DQN (nét đứt kèm viền Casing nổi bật) */}
                        {(routeViewMode === "vanilla" || routeViewMode === "both") && mapData.vanillaPts.length > 1 && (
                          <>
                            <Polyline
                              positions={mapData.vanillaPts}
                              pathOptions={{
                                color: casingColor,
                                weight: 6,
                                opacity: 0.85,
                                lineJoin: "round",
                              }}
                            />
                            <Polyline
                              positions={mapData.vanillaPts}
                              pathOptions={{
                                color: VANILLA_COLOR,
                                weight: 3.6,
                                opacity: 0.95,
                                dashArray: "6 6",
                                lineJoin: "round",
                              }}
                            />
                          </>
                        )}

                        {/* Tuyến Guided DQN: các đoạn được tô màu theo cụm kèm Casing viền sáng */}
                        {(routeViewMode === "guided" || routeViewMode === "both") &&
                          mapData.guidedSegments.map((seg, sIdx) => (
                            <div key={`seg-wrap-${sIdx}`} style={{ display: "contents" }}>
                              <Polyline
                                positions={[seg.from, seg.to]}
                                pathOptions={{
                                  color: casingColor,
                                  weight: 7.5,
                                  opacity: 0.9,
                                  lineCap: "round",
                                  lineJoin: "round",
                                }}
                              />
                              <Polyline
                                positions={[seg.from, seg.to]}
                                pathOptions={{
                                  color: clusterColor(seg.cluster),
                                  weight: 4.8,
                                  opacity: 1.0,
                                  lineCap: "round",
                                  lineJoin: "round",
                                }}
                              />
                            </div>
                          ))}

                        {/* Điểm đổi cụm: Hiệu ứng ripple */}
                        {(routeViewMode === "guided" || routeViewMode === "both") &&
                          mapData.switchStops.map((sw) => {
                            const swPos = latlng(sw.stop);
                            if (!swPos) return null;
                            return (
                              <Marker
                                key={`ripple-${sw.stop.id}`}
                                position={swPos}
                                icon={rippleIcon()}
                                interactive={false}
                                zIndexOffset={500}
                              />
                            );
                          })}

                        {/* Điểm giao hàng */}
                        {mapData.guidedStops.map((st, i) => {
                          const pos = latlng(st);
                          if (!pos) return null;
                          const cluster = Number(st.cluster_id ?? 0);
                          return (
                            <Marker
                              key={`st-${st.id}`}
                              position={pos}
                              icon={pinIcon(i + 1, clusterColor(cluster))}
                              zIndexOffset={100}
                            >
                              <Popup>
                                <div style={{ minWidth: 200, padding: 4 }}>
                                  <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                                    Điểm #{i + 1}: {String(st.package_name ?? "Đơn hàng")}
                                  </div>
                                  <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                                    {String(st.receiverAddress ?? "")}
                                  </div>
                                  <div style={{ fontSize: 11.5 }}>
                                    Cụm K-Means: <b style={{ color: clusterColor(cluster) }}>Cụm {cluster}</b>
                                  </div>
                                </div>
                              </Popup>
                            </Marker>
                          );
                        })}
                      </MapBase>
                    </div>
                  )}
                </MapErrorBoundary>
              ) : (
                <div style={{ padding: 24, textAlign: "center", color: "var(--muted)" }}>
                  Không tìm thấy dữ liệu tọa độ cho Workload #{selectedWlId}
                </div>
              )}
            </Card>

            {/* ─── Per-workload detail table ─── */}
            <Card title="Bảng đối chuẩn chi tiết 12 workload thực tế" sub="12 workload được chọn deterministically (median + final workload cho từng quy mô đơn)">
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th className="num">Workload</th>
                      <th className="num">Số đơn</th>
                      <th className="num">Vanilla DQN (km)</th>
                      <th className="num">Guided DQN (km)</th>
                      <th className="num">Δ Quãng đường (G − V)</th>
                      <th className="num">Δ Thời gian (G − V)</th>
                      <th className="num">Đổi cụm (Guided)</th>
                      <th className="num">Đổi cụm (Vanilla)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairedWorkloads.map((w) => {
                      const distImpr = n(w.distance_improvement_pct);
                      const timeImpr = n(w.time_improvement_pct);
                      // Delta = Guided - Vanilla: positive means Guided is longer/slower
                      const deltaDistPct = distImpr !== null ? -distImpr : null;
                      const deltaTimePct = timeImpr !== null ? -timeImpr : null;
                      return (
                        <tr
                          key={String(w.workload_id)}
                          className={selectedWlId === Number(w.workload_id) ? "sel" : ""}
                          style={{ cursor: "pointer" }}
                          onClick={() => setSelectedWlId(Number(w.workload_id))}
                        >
                          <td className="num">#{String(w.workload_id)}</td>
                          <td className="num">{num(n(w.number_of_deliveries), 0)}</td>
                          <td className="num">{num(n(w.baseline_distance_km), 2)}</td>
                          <td className="num">{num(n(w.guided_distance_km), 2)}</td>
                          <td
                            className="num"
                            style={{
                              color: deltaDistPct !== null && deltaDistPct > 0 ? "var(--algo-tradeoff)" : deltaDistPct !== null && deltaDistPct < 0 ? "var(--good)" : undefined,
                              fontWeight: 700,
                            }}
                          >
                            {signedPct(deltaDistPct, 2)}
                          </td>
                          <td
                            className="num"
                            style={{
                              color: deltaTimePct !== null && deltaTimePct > 0 ? "var(--algo-tradeoff)" : deltaTimePct !== null && deltaTimePct < 0 ? "var(--good)" : undefined,
                              fontWeight: 700,
                            }}
                          >
                            {signedPct(deltaTimePct, 2)}
                          </td>
                          <td className="num">{num(n(w.guided_cluster_switches), 0)}</td>
                          <td className="num">
                            {w.baseline_cluster_switches !== undefined && w.baseline_cluster_switches !== null
                              ? num(n(w.baseline_cluster_switches), 0)
                              : "–"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* ─── Conclusion Callout ─── */}
            <ResearchCallout
              type="warning"
              title="Kết luận khoa học (Trung thực với số liệu thực nghiệm)"
            >
              1. <b>Tăng tính liên tục không gian:</b> K-Means giúp giảm {switchReduction !== null ? signedPct(-switchReduction, 1) : "–"} tỷ lệ đổi cụm trên các workload đối sánh ({num(Number(s.workloads_with_baseline_sequences ?? 6), 0)} workload), hạn chế di chuyển chéo giữa các quận.
              <br />
              2. <b>Bản chất đánh đổi kỹ thuật:</b> Mô hình <strong>không làm tuyến ngắn hơn hay nhanh hơn</strong> Vanilla DQN: quãng đường tăng +{num(distDelta, 2)}%, thời gian tăng +{num(timeDelta, 2)}% ({num(n(s.workloads), 0)} workload kiểm chứng).
              <br />
              3. <b>Bảo toàn dịch vụ:</b> Tỷ lệ giao hàng đúng hạn đạt tuyệt đối <b>100%</b> trên toàn bộ {num(n(s.workloads), 0)} workload ({num(n(s.deliveries), 0)} đơn hàng).
            </ResearchCallout>

            {/* Source metadata */}
            <div className="source-meta">
              <span>📁 Nguồn file: <code>results/chapter4/tables/table_final_guided_dqn_summary.csv</code></span>
              <span>🔢 Quy mô: {num(n(s.workloads), 0)} workload / {num(n(s.deliveries), 0)} đơn</span>
              <span>λ = {num(n(s.guided_lambda), 2)}</span>
              <span>Seed = 42 (reproducible)</span>
            </div>
          </>
        )}

        {/* ══════════════ TAB 4: MÔ HÌNH PROPOSED RL ══════════════ */}
        {activeTab === "proposed" && (
          <div className="stack" style={{ gap: 20 }}>
            {/* Header Callout & Protocol Status */}
            <div className="rounded-xl border border-border/70 bg-card p-5 sm:p-6 shadow-2xs">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 14 }}>
                <div style={{ maxWidth: 860 }}>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11.5, fontWeight: 700, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)" }} />
                    Mô hình Reinforcement Learning Đa mục tiêu Đề xuất mới
                  </div>
                  <h2 style={{ fontSize: 22, fontWeight: 700, margin: "0 0 8px", color: "var(--text)" }}>
                    Proposed RL Research Studio (Candidate-scoring Q-Network)
                  </h2>
                  <p style={{ margin: 0, fontSize: 13.5, lineHeight: 1.6, color: "var(--text-secondary)" }}>
                    Mô hình mở rộng do nhóm nghiên cứu phát triển độc lập (mã nguồn trong <code>python/rl/proposed/</code>). Sử dụng mạng nơ-ron học sâu chấm điểm từng ứng viên giao hàng (Candidate-scoring), kết hợp vector đặc trưng 8 chiều và hàm mục tiêu đa tiêu chí khóa để giải quyết bài toán định tuyến xe với cửa sổ thời gian (VRPTW) trên mạng đường phố OpenStreetMap TP.HCM.
                  </p>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
                  <span className="chip good" style={{ fontSize: 11.5, fontWeight: 700, padding: "4px 10px" }}>
                    ✓ Đánh giá độc lập trên 16 WL Test
                  </span>
                  <span className="chip" style={{ fontSize: 11, background: "var(--surface-2)", color: "var(--muted)" }}>
                    Checkpoint Epoch 30 · Lock Objective
                  </span>
                </div>
              </div>
              <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--border)", display: "flex", gap: 12, flexWrap: "wrap", fontSize: 12 }}>
                <span style={{ color: "var(--text-secondary)" }}>
                  <b>Phạm vi nghiên cứu:</b> 76 Workload Train · 16 Workload Validation · 16 Workload Held-out Test (131 đơn hàng)
                </span>
                <span style={{ color: "var(--muted)" }}>|</span>
                <span style={{ color: "var(--text-secondary)" }}>
                  <b>Hàm mục tiêu:</b> J(π) = 0,375D̄ + 0,375T̄ + 0,10M̄ + 0,10L̄ + 0,05S̄
                </span>
                <span style={{ color: "var(--muted)" }}>|</span>
                <span style={{ color: "var(--text-secondary)" }}>
                  <b>Phần thưởng bước:</b> r_t = -ΔJ_t (γ = 1,0)
                </span>
              </div>
            </div>

            {/* 1. Interactive Architectural Flowchart */}
            <Card
              title="1. Kiến trúc Tổng thể Mô hình (Interactive Model Architecture Pipeline)"
              sub="Quy trình ra quyết định tuần tự: State → Candidate Encoder → Q-Network → Action → Environment → Reward. Nhấp vào từng khối để xem chi tiết tensor và thuật toán."
            >
              <div className="rl-arch-pipeline">
                {ARCH_NODES.map((node, idx) => {
                  const isAct = node.key === selectedArchNode;
                  return (
                    <div key={node.key} style={{ display: "flex", alignItems: "center", flex: 1 }}>
                      <div
                        className={`rl-node-card ${isAct ? "active" : ""}`}
                        onClick={() => setSelectedArchNode(node.key)}
                        style={{
                          borderColor: isAct ? "var(--accent)" : undefined,
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                          <span style={{ fontSize: 10, fontWeight: 700, color: isAct ? "var(--accent)" : "var(--muted)", textTransform: "uppercase" }}>
                            {node.step}
                          </span>
                          <span className="badge" style={{ fontSize: 9.5 }}>#{idx + 1}</span>
                        </div>
                        <div style={{ fontWeight: 700, fontSize: 13, color: isAct ? "var(--accent)" : "var(--text)" }}>
                          {node.name}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2, marginBottom: 8 }}>
                          {node.subtitle}
                        </div>
                        <div style={{ marginTop: "auto", padding: "4px 6px", background: "var(--surface-2)", borderRadius: 4, fontFamily: "ui-monospace, monospace", fontSize: 10.5, color: "var(--text-secondary)", textAlign: "center", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {node.shape}
                        </div>
                      </div>
                      {idx < ARCH_NODES.length - 1 && (
                        <div className="rl-node-arrow">→</div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Detail Inspector for Selected Node */}
              {(() => {
                const activeNode = ARCH_NODES.find((n) => n.key === selectedArchNode) ?? ARCH_NODES[2];
                return (
                  <div className="rl-detail-inspector">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, marginBottom: 10 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="badge" style={{ fontSize: 11, fontWeight: 700, background: "var(--accent-soft)", color: "var(--accent)" }}>
                          {activeNode.step}: {activeNode.name}
                        </span>
                        <code style={{ fontSize: 12, fontWeight: 700, color: "var(--text)" }}>
                          {activeNode.formula}
                        </code>
                      </div>
                      <span style={{ fontSize: 11.5, color: "var(--muted)" }}>
                        Mã nguồn: <code>{activeNode.codeFile}</code> ({activeNode.codeSymbol})
                      </span>
                    </div>
                    <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "var(--text-secondary)" }}>
                      {activeNode.description}
                    </p>
                  </div>
                );
              })()}
            </Card>

            {/* 2. Candidate Encoder 8D Feature Breakdown */}
            <Card
              title="2. Không gian Biểu diễn Ứng viên: 8 Chiều Đặc trưng (Candidate Encoder)"
              sub="Mỗi đơn hàng hợp lệ a_i ∈ U_t được featurize thành vector 8D x_(t, i) theo đúng khai báo trong python/rl/proposed/candidate_encoder.py."
            >
              <div className="feature-badge-grid">
                {[
                  { name: "elapsed_ratio", idx: 0, math: "t_elapsed / T_ref (clip 2.0)", desc: "Tỷ lệ thời gian đã trôi qua so với tổng thời gian dự kiến cả hành trình." },
                  { name: "remaining_ratio", idx: 1, math: "|U_t| / N_total", desc: "Tỷ lệ số lượng đơn hàng chưa được giao còn lại trong đợt giao." },
                  { name: "distance_ratio", idx: 2, math: "d(p_t, p_i) / d_ref (ref=10 km)", desc: "Khoảng cách đường đi từ vị trí hiện tại đến điểm giao ứng viên trên đồ thị OSM." },
                  { name: "travel_time_ratio", idx: 3, math: "τ(p_t, p_i) / τ_ref (ref=15 ph)", desc: "Thời gian di chuyển ước tính theo vận tốc nội đô đến điểm giao ứng viên." },
                  { name: "slack_ratio", idx: 4, math: "(t_deadline - t_arrival) / τ_ref", desc: "Độ dư dả thời gian trước hạn giao cam kết (clip 2.0; số âm nếu đã trễ hạn)." },
                  { name: "deadline_valid", idx: 5, math: "𝕀(deadline ≠ None)", desc: "Cờ nhị phân (1,0 nếu đơn có hạn cam kết SLA; 0,0 nếu đơn giao linh hoạt)." },
                  { name: "lateness_ratio", idx: 6, math: "max(0, t_arr - t_dead) / L_ref", desc: "Mức độ trễ phát sinh nếu chọn đơn này giao tiếp theo (ref=30 phút)." },
                  { name: "cluster_switch", idx: 7, math: "𝕀(c_i ≠ c_t)", desc: "Cờ nhị phân (1,0 nếu đơn này khác cụm K-Means với điểm vừa ghé; 0,0 nếu cùng cụm)." },
                ].map((feat) => (
                  <div key={feat.name} className="weight-card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <code style={{ fontSize: 12.5, color: "var(--accent)", fontWeight: 700 }}>{feat.name}</code>
                      <span className="badge" style={{ fontSize: 10 }}>#{feat.idx}</span>
                    </div>
                    <div style={{ fontFamily: "ui-monospace, monospace", fontSize: 11, color: "var(--text)", padding: "3px 6px", background: "var(--surface-2)", borderRadius: 4, marginBottom: 6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {feat.math}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.45 }}>
                      {feat.desc}
                    </div>
                  </div>
                ))}
              </div>
              <div className="muted small" style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
                <span>💡 <b>Ưu điểm lý thuyết:</b> Khắc phục nhược điểm cố định N output của mô hình DQN truyền thống; mạng chấm điểm Candidate-scoring hoạt động độc lập cho từng ứng viên nên có thể xử lý các workload có số lượng đơn biến thiên linh hoạt.</span>
              </div>
            </Card>

            {/* 3. Objective Formulation & Locked Weights */}
            <Card
              title="3. Hàm mục tiêu Đa tiêu chí J(π) & Bộ trọng số Khóa (Locked Weights)"
              sub="Hàm chi phí kết hợp tuyến tính 5 mục tiêu, cấu hình D_moderate_sla được chọn lọc qua khảo sát độ nhạy và khóa trước khi đánh giá trên tập Test."
            >
              <div className="formula-block" style={{ marginBottom: 16 }}>
                <span>J(π) = </span>
                <span style={{ color: "var(--accent)", fontWeight: 700 }}>0,375</span>·D̄
                {" + "}
                <span style={{ color: "var(--accent)", fontWeight: 700 }}>0,375</span>·T̄
                {" + "}
                <span style={{ color: "var(--algo-tradeoff)", fontWeight: 700 }}>0,100</span>·M̄
                {" + "}
                <span style={{ color: "var(--algo-tradeoff)", fontWeight: 700 }}>0,100</span>·L̄
                {" + "}
                <span style={{ color: "var(--algo-guided)", fontWeight: 700 }}>0,050</span>·S̄
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
                {[
                  { weight: "w_d = 37,5%", name: "Quãng đường (D̄)", ref: "ref = 10,0 km / chặng", color: "var(--accent)", desc: "Chuẩn hóa cự ly di chuyển shipper trên mạng đường OSM." },
                  { weight: "w_t = 37,5%", name: "Thời gian (T̄)", ref: "ref = 15,0 phút / chặng", color: "var(--accent)", desc: "Chuẩn hóa thời gian lăn bánh theo vận tốc đường phố TP.HCM." },
                  { weight: "w_m = 10,0%", name: "Số đơn trễ (M̄)", ref: "Phạt bậc thang SLA", color: "var(--algo-tradeoff)", desc: "Đếm số đơn vi phạm mốc deadline để bảo vệ tỷ lệ đúng hạn." },
                  { weight: "w_l = 10,0%", name: "Tổng phút trễ (L̄)", ref: "ref = 30,0 phút trễ", color: "var(--algo-tradeoff)", desc: "Hạn chế mức độ trễ nghiêm trọng của từng đơn hàng." },
                  { weight: "w_s = 5,0%", name: "Đổi cụm (S̄)", ref: "Tỷ lệ đổi cụm K-Means", color: "var(--algo-guided)", desc: "Khuyến khích hoàn tất đơn cùng cụm trước khi chuyển vùng." },
                ].map((wItem) => (
                  <div key={wItem.name} className="weight-card">
                    <div style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>{wItem.weight}</div>
                    <div style={{ fontWeight: 700, fontSize: 13.5, color: wItem.color, marginTop: 4 }}>{wItem.name}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{wItem.ref}</div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 6, lineHeight: 1.4 }}>{wItem.desc}</div>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 14, padding: "10px 14px", background: "var(--surface-2)", borderRadius: 8, fontSize: 12.5, lineHeight: 1.6, color: "var(--text-secondary)" }}>
                <b>Cơ chế Phần thưởng Từng bước:</b> <code>r_t = -ΔJ_t = -(J_t - J_&#123;t-1&#125;)</code>. Khi sử dụng hệ số chiết khấu <code>γ = 1,0</code> trên chân trời hữu hạn (finite-horizon episode), tổng phần thưởng tích lũy toàn bộ hành trình chính xác bằng <code>Σ r_t = -J(π)</code>. Cực đại hóa tổng phần thưởng của thuật toán RL tương đương với việc cực tiểu hóa hàm mục tiêu đa tiêu chí <code>J(π)</code>.
              </div>
            </Card>

            {/* 4. Training Split Protocol */}
            <Card
              title="4. Giao thức Phân tách Dữ liệu Thực nghiệm (Dataset Split Protocol)"
              sub="Quy trình chuẩn mực học máy nhằm chống rò rỉ dữ liệu (Data Leakage): Train (76 WL) → Validation (16 WL) → Held-out Test (16 WL)."
            >
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                <div style={{ padding: 16, background: "var(--surface-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>Tập Huấn luyện (Train)</span>
                    <span className="chip" style={{ fontSize: 10.5 }}>76 Workloads</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)", margin: "8px 0 4px" }}>
                    3.040 Episodes
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Mỗi mô hình được huấn luyện 40 epoch trên 76 workload. Replay buffer: 10.000 transitions; Batch size: 64; Optimizer: Adam (lr = 10⁻³).
                  </div>
                </div>

                <div style={{ padding: 16, background: "var(--surface-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>Tập Kiểm chuẩn (Validation)</span>
                    <span className="chip" style={{ fontSize: 10.5 }}>16 Workloads</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)", margin: "8px 0 4px" }}>
                    Checkpoint Epoch 30
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Dùng để chọn lọc checkpoint tối ưu trong quá trình training và tinh chỉnh trọng số cấu hình D_moderate_sla trước khi đóng băng mô hình.
                  </div>
                </div>

                <div style={{ padding: 16, background: "var(--surface-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--good)", textTransform: "uppercase" }}>Tập Kiểm chứng Ẩn (Test)</span>
                    <span className="chip good" style={{ fontSize: 10.5 }}>16 Workloads</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)", margin: "8px 0 4px" }}>
                    131 Đơn hàng
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    Mô hình bị KHÓA HOÀN TOÀN trước khi đánh giá trên tập Test. Tuyệt đối không tinh chỉnh thêm tham số hoặc học tiếp trên tập này.
                  </div>
                </div>
              </div>
            </Card>

            {/* 5. Head-to-Head Comparison Table on 16 Held-out Test Workloads */}
            <Card
              title="5. Bảng Đối chuẩn Trực diện trên 16 Workload Test (Final Test Performance)"
              sub="Tất cả các phương pháp được đánh giá trên cùng tập 16 workload Held-out Test (131 đơn hàng). Nguồn số liệu chính thức: results/proposed_rl_final_locked_objective_summary.csv."
            >
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Thuật toán</th>
                      <th className="num">Quãng đường TB</th>
                      <th className="num">Thời gian TB</th>
                      <th className="num">Đơn trễ</th>
                      <th className="num">Tỷ lệ SLA</th>
                      <th className="num">Phút trễ TB/WL</th>
                      <th className="num">Đổi cụm TB</th>
                      <th className="num">Tỷ lệ đổi cụm</th>
                      <th className="num">Mục tiêu J(π)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {PROPOSED_TEST_BENCHMARK.map((b) => {
                      const isProposed = b.key === "proposed_rl";
                      const isBestSwitches = b.switches === 4.00;
                      const isBestDist = b.dist === 55.58;
                      return (
                        <tr key={b.key} style={{ background: isProposed ? "var(--hover-row)" : undefined }}>
                          <td>
                            <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                              <span className="swatch" style={{ background: b.color }} />
                              <b style={{ color: isProposed ? "var(--accent)" : "var(--text)" }}>{b.name}</b>
                              {isProposed && (
                                <span className="chip" style={{ fontSize: 10, padding: "1px 5px", background: "var(--accent-soft)", color: "var(--accent)" }}>
                                  Proposed
                                </span>
                              )}
                            </span>
                          </td>
                          <td className={`num ${isBestDist ? "best" : ""}`}>
                            {num(b.dist, 2)} km
                          </td>
                          <td className="num">{num(b.time, 1)} ph</td>
                          <td className="num">{b.late} / 131</td>
                          <td className="num">
                            <span style={{ color: b.sla === 1 ? "var(--good)" : undefined, fontWeight: 700 }}>
                              {pct(b.sla, 2)}
                            </span>
                          </td>
                          <td className="num">{num(b.totalLatenessMinutes / 16, 2)} ph/WL</td>
                          <td className={`num ${isBestSwitches ? "best" : ""}`}>
                            <b>{num(b.switches, 2)}</b>
                          </td>
                          <td className={`num ${isBestSwitches ? "best" : ""}`}>
                            {pct(b.switchRate / 100, 1)}
                          </td>
                          <td className="num">{num(b.obj, 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Visual Benchmark Horizontal Bars */}
              <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div style={{ padding: 14, background: "var(--surface-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text)", marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
                    <span>Quãng đường trung bình (km) — Ngắn hơn là tốt hơn</span>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Clarke–Wright tốt nhất (55,58 km)</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {PROPOSED_TEST_BENCHMARK.map((b) => (
                      <div key={b.key} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                        <span style={{ width: 110, color: "var(--text-secondary)", textAlign: "right", flexShrink: 0 }}>
                          {b.name}
                        </span>
                        <div style={{ flex: 1, height: 16, background: "var(--surface)", borderRadius: 4, overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${(b.dist / 65) * 100}%`,
                              height: "100%",
                              background: b.color,
                              opacity: 0.85,
                              borderRadius: 4,
                            }}
                          />
                        </div>
                        <span style={{ width: 56, fontWeight: 700, textAlign: "right" }}>
                          {num(b.dist, 2)} km
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ padding: 14, background: "var(--surface-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text)", marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
                    <span>Số lần đổi cụm trung bình — Ít hơn là tốt hơn</span>
                    <span style={{ fontSize: 11, color: "var(--accent)" }}>Proposed RL thấp nhất (4,00 lần)</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {PROPOSED_TEST_BENCHMARK.map((b) => (
                      <div key={b.key} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                        <span style={{ width: 110, color: "var(--text-secondary)", textAlign: "right", flexShrink: 0 }}>
                          {b.name}
                        </span>
                        <div style={{ flex: 1, height: 16, background: "var(--surface)", borderRadius: 4, overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${(b.switches / 4.5) * 100}%`,
                              height: "100%",
                              background: b.color,
                              opacity: 0.85,
                              borderRadius: 4,
                            }}
                          />
                        </div>
                        <span style={{ width: 56, fontWeight: 700, textAlign: "right", color: b.switches === 4.00 ? "var(--accent)" : undefined }}>
                          {num(b.switches, 2)} lần
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            {/* 6. Scientific Limitations & Research Transparency Callout */}
            <ResearchCallout
              type="warning"
              title="Minh bạch Khoa học & Giới hạn Thực nghiệm (Proposed RL)"
            >
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, fontSize: 12.5, lineHeight: 1.5, color: "var(--text-secondary)" }}>
                <div>
                  <b style={{ color: "var(--text)" }}>1. Cự ly & Thời gian:</b>
                  <p style={{ margin: "4px 0 0" }}>
                    Trên 16 WL Test, Proposed RL đạt <b>61,90 km · 67,52 phút</b>. Mô hình <b>chưa vượt Clarke–Wright (55,58 km)</b> và <b>DQN (57,11 km)</b> về khoảng cách.
                  </p>
                </div>
                <div>
                  <b style={{ color: "var(--text)" }}>2. Khả năng Gom Cụm:</b>
                  <p style={{ margin: "4px 0 0" }}>
                    Đạt <b>4,00 lần đổi cụm trung bình</b> (thấp nhất trong 6 thuật toán so sánh trên Test). Tuy nhiên, kết quả này <b>không đồng nghĩa với tối ưu toàn cục</b>.
                  </p>
                </div>
                <div>
                  <b style={{ color: "var(--text)" }}>3. Độc lập Dữ liệu:</b>
                  <p style={{ margin: "4px 0 0" }}>
                    Guided DQN đánh giá trên <b>12 WL kiểm chứng</b>; Proposed RL đánh giá độc lập trên <b>16 WL Test</b>. Hai tập độc lập, không trộn lẫn.
                  </p>
                </div>
              </div>
            </ResearchCallout>
          </div>
        )}
      </div>
    </div>
  );
}
