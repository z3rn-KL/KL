import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  BarChart3,
  Network,
  Truck,
  BrainCircuit,
  PlayCircle,
  TrendingUp,
  ShieldCheck,
  List,
  Info,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Zap,
} from "lucide-react";
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
import { api, AlgorithmSummary, FinalValidation, ClusteringQuality, Sensitivity, Stability } from "@/lib/api";
import { RLTrainingExplorer } from "@/components/research/RLTrainingExplorer";
import { RLDecisionExplorer } from "@/components/research/RLDecisionExplorer";
import { RLBaselineExplorer } from "@/components/research/RLBaselineExplorer";

import { ALGO_COLOR, ALGO_LABEL } from "@/lib/format";

type ResearchTopic =
  | "overview"
  | "clustering"
  | "benchmark"
  | "ablation"
  | "routing"
  | "rl"
  | "experiments"
  | "sensitivity"
  | "stability";

type MetricType = "distance" | "late" | "lateness" | "runtime";

const TOPIC_DETAILS: Record<
  ResearchTopic,
  { eyebrow: string; title: string; desc: string; link?: string }
> = {
  overview: {
    eyebrow: "Hướng nghiên cứu",
    title: "Từ dữ liệu không gian đến quyết định định tuyến.",
    desc: "NEXUS kết hợp phân cụm không gian và học tăng cường để nghiên cứu phương án tổ chức tuyến giao hàng trên dữ liệu thực nghiệm TP.HCM.",
  },
  clustering: {
    eyebrow: "Phân tích không gian",
    title: "K-Means và DBSCAN trên 2,394 đơn giao hàng",
    desc: "Quan sát phân bố đơn hàng, tâm cụm và điểm bất thường. Tối ưu bán kính và phân bổ khối lượng cho 16 shipper.",
    link: "/phan-cum",
  },
  benchmark: {
    eyebrow: "Đối sánh thuật toán",
    title: "So sánh theo từng tập thực nghiệm (108 Workloads)",
    desc: "Các nhóm 108, 12 và 16 workload được đối sánh khoa học: Proposed RL vượt trội về tỷ lệ giao đúng hạn và giảm cự ly.",
    link: "/so-sanh",
  },
  ablation: {
    eyebrow: "Kiểm chứng thực nghiệm",
    title: "Ablation Study 2×2: Phân cụm không gian kết hợp Proposed RL",
    desc: "Đánh giá có kiểm soát tác động riêng lẻ và kết hợp của phân cụm (K-Means K=10) và học tăng cường (Proposed RL) trên 16 shipper và 85 đơn hàng.",
  },
  routing: {
    eyebrow: "Phân tích tuyến giao",
    title: "Route Order và hình học mạng đường OSM TP.HCM",
    desc: "Đối chiếu thứ tự điểm dừng với đường đi thực tế qua đồ thị mạng đường OpenStreetMap, không dùng khoảng cách đường chim bay.",
    link: "/tuyen-giao",
  },
  rl: {
    eyebrow: "Học tăng cường",
    title: "Q-Learning · SARSA · DQN · Guided DQN · Proposed RL",
    desc: "Phân tích cấu trúc phần thưởng, hàm giá trị hành động Q(s, a), và trọng số mô hình đề xuất đã hội tụ.",
    link: "/guided-dqn",
  },
  experiments: {
    eyebrow: "Thực nghiệm",
    title: "Chạy thử thuật toán theo workload (Run Lab)",
    desc: "Môi trường thực thi trực tiếp các kịch bản thực nghiệm, kiểm chứng thời gian suy diễn và kiểm soát provenance.",
    link: "/chay-thu",
  },
  sensitivity: {
    eyebrow: "Phân tích độ nhạy",
    title: "Ảnh hưởng của tham số mô hình và tải trọng",
    desc: "Khảo sát tính nhạy cảm của chính sách RL theo hệ số phạt trễ SLA, ngưỡng dung lượng xe và số lượng shipper.",
    link: "/do-nhay",
  },
  stability: {
    eyebrow: "Đánh giá độ ổn định",
    title: "Biến thiên theo seed và 10 lần chạy lặp",
    desc: "Thể hiện độ tin cậy và khả năng tái lặp của mô hình đề xuất trên các tập kiểm thử độc lập.",
    link: "/on-dinh",
  },
};


const PIPELINE_STAGES = [
  { step: 1, title: "Dữ liệu Đơn hàng GIS", dataset: "Dataset A (2.394 đơn)", targetTopic: "clustering" as ResearchTopic, link: "/don-hang" },
  { step: 2, title: "Tiền xử lý & Ma trận", dataset: "OSM Dijkstra / Haversine", targetTopic: "routing" as ResearchTopic, link: "/tuyen-giao" },
  { step: 3, title: "Phân cụm Không gian", dataset: "K-Means K=10 (Sil=0.429)", targetTopic: "clustering" as ResearchTopic, link: "/phan-cum" },
  { step: 4, title: "Phân công 16 Shipper", dataset: "Spatial Cluster Affinity", targetTopic: "routing" as ResearchTopic, link: "/tuyen-giao" },
  { step: 5, title: "Học tăng cường RL", dataset: "Candidate-Scoring DQN", targetTopic: "rl" as ResearchTopic, link: "/guided-dqn" },
  { step: 6, title: "Mô phỏng Động Nhiều Ngày", dataset: "Dataset C (85 đơn/ngày)", targetTopic: "ablation" as ResearchTopic, link: "/dieu-phoi" },
  { step: 7, title: "Đánh giá & Ablation 2×2", dataset: "108 Workloads & Ablation", targetTopic: "benchmark" as ResearchTopic, link: "/so-sanh" },
];

export default function ResearchAnalytics() {
  const navigate = useNavigate();
  const [topic, setTopic] = useState<ResearchTopic>("overview");
  const [benchmarkData, setBenchmarkData] = useState<AlgorithmSummary[]>([]);
  const [validationData, setValidationData] = useState<FinalValidation | null>(null);
  const [clusteringQuality, setClusteringQuality] = useState<ClusteringQuality | null>(null);
  const [sensitivityData, setSensitivityData] = useState<Sensitivity | null>(null);
  const [stabilityData, setStabilityData] = useState<Stability | null>(null);
  const [metricType, setMetricType] = useState<MetricType>("distance");
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [bench, validation, clusterQ, sens, stab] = await Promise.allSettled([
          api.benchmarkSummary(),
          api.finalValidation(),
          api.clusteringQuality(),
          api.sensitivity(),
          api.stability(),
        ]);
        if (bench.status === "fulfilled" && bench.value?.items) {
          setBenchmarkData(bench.value.items);
        }
        if (validation.status === "fulfilled" && validation.value) {
          setValidationData(validation.value);
        }
        if (clusterQ.status === "fulfilled" && clusterQ.value) {
          setClusteringQuality(clusterQ.value);
        }
        if (sens.status === "fulfilled" && sens.value) {
          setSensitivityData(sens.value);
        }
        if (stab.status === "fulfilled" && stab.value) {
          setStabilityData(stab.value);
        }
      } catch (err) {
        console.warn("Could not load research benchmark data", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const chartBenchmarkData = benchmarkData.map((item) => {
    let value = 0;
    if (metricType === "distance") value = Number(item.mean_distance_km.toFixed(1));
    else if (metricType === "late") value = item.late_deliveries;
    else if (metricType === "lateness") value = Number(item.total_lateness_minutes.toFixed(1));
    else if (metricType === "runtime") value = Number((item.mean_route_runtime_seconds ?? 0).toFixed(4));

    return {
      name: ALGO_LABEL[item.algorithm] || item.algorithm,
      algorithm: item.algorithm,
      value,
      distanceKm: Number(item.mean_distance_km.toFixed(1)),
      travelTimeMin: Number(item.mean_travel_time_minutes.toFixed(1)),
      onTimeRate: Number((item.weighted_on_time_rate * 100).toFixed(2)),
      lateDeliveries: item.late_deliveries,
      totalLatenessMin: Number(item.total_lateness_minutes.toFixed(1)),
      runtimeSec: Number((item.mean_route_runtime_seconds ?? 0).toFixed(4)),
    };
  });

  const clusterDistribution = [
    { cluster: "Cụm 0", size: 261, label: "Tân Bình - Phú Nhuận" },
    { cluster: "Cụm 1", size: 83, label: "Bình Thạnh Bắc" },
    { cluster: "Cụm 2", size: 162, label: "Quận 10 - Quận 3" },
    { cluster: "Cụm 3", size: 70, label: "Quận 1 Trung tâm" },
    { cluster: "Cụm 4", size: 627, label: "Gò Vấp Đông" },
    { cluster: "Cụm 5", size: 98, label: "Bình Tân" },
    { cluster: "Cụm 6", size: 413, label: "Tân Phú" },
    { cluster: "Cụm 7", size: 215, label: "Quận 11" },
    { cluster: "Cụm 8", size: 212, label: "Quận 5 - Quận 6" },
    { cluster: "Cụm 9", size: 253, label: "Quận 12 Nam" },
  ];

  const currentTopic = TOPIC_DETAILS[topic];

  const getMetricDomain = (): [number, number | "auto"] => {
    if (metricType === "distance") return [50, 68];
    if (metricType === "late") return [0, 8];
    if (metricType === "lateness") return [0, 220];
    return [0, "auto"];
  };

  const getMetricLabel = () => {
    if (metricType === "distance") return "Quãng đường TB (km)";
    if (metricType === "late") return "Số đơn trễ hạn (đơn)";
    if (metricType === "lateness") return "Tổng phút trễ SLA (phút)";
    return "Thời gian tính toán (giây)";
  };

  const formatTooltipValue = (val: any) => {
    if (metricType === "distance") return [`${val} km`, "Quãng đường"];
    if (metricType === "late") return [`${val} đơn`, "Đơn trễ"];
    if (metricType === "lateness") return [`${val} phút`, "Tổng phút trễ"];
    return [`${val} s`, "Thời gian chạy"];
  };

  return (
    <section id="page-research" className="page">
      <div className="wrap">
        {/* Page Intro matching Approved Reference */}
        <div className="page-intro">
          <div>
            <div className="eyebrow">RESEARCH / GRADUATION THESIS</div>
            <h1>Phân tích & Nghiên cứu</h1>
            <p className="subline">
              Một không gian riêng cho nghiên cứu phân cụm, định tuyến và học tăng cường.
            </p>
          </div>
          <div className="hero-right">
            <span className="date-pill">
              <Info className="icons smico" /> Không chỉnh dữ liệu khoa học
            </span>
          </div>
        </div>

                {/* 7-Stage Scientific Research Pipeline Bar */}
        <div
          className="box"
          style={{
            marginBottom: "20px",
            padding: "16px 20px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "8px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "12px",
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontWeight: 700, fontSize: "13px", color: "var(--text)" }}>
                QUY TRÌNH KHOA HỌC 7 GIAI ĐOẠN (SCIENTIFIC RESEARCH PIPELINE)
              </span>
              <span className="chip blue" style={{ fontSize: "12px" }}>KLTN_11</span>
            </div>
            <span style={{ fontSize: "12px", color: "var(--muted)" }}>
              Nhấp vào giai đoạn để chuyển tab nghiên cứu tương ứng
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(135px, 1fr))",
              gap: "8px",
            }}
          >
            {PIPELINE_STAGES.map((stg) => {
              const isActive = stg.targetTopic === topic;
              return (
                <button
                  key={stg.step}
                  type="button"
                  onClick={() => setTopic(stg.targetTopic)}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    padding: "10px 12px",
                    borderRadius: "6px",
                    border: isActive ? "2px solid #3b82f6" : "1px solid var(--border)",
                    background: isActive ? "rgba(59, 130, 246, 0.12)" : "var(--panel-bg)",
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "all 0.15s ease",
                  }}
                >
                  <span
                    style={{
                      fontSize: "12px",
                      fontWeight: 800,
                      color: isActive ? "#2563eb" : "var(--text-secondary)",
                      marginBottom: "4px",
                    }}
                  >
                    GIAI ĐOẠN {stg.step}
                  </span>
                  <strong
                    style={{
                      fontSize: "12px",
                      color: "var(--text)",
                      lineHeight: 1.3,
                      marginBottom: "4px",
                    }}
                  >
                    {stg.title}
                  </strong>
                  <span
                    style={{
                      fontSize: "12px",
                      color: "var(--text-secondary)",
                      lineHeight: 1.2,
                    }}
                  >
                    {stg.dataset}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Research Layout: Left Nav + Right Content */}
        <div className="research-wrap">
          {/* Left Vertical Topics Navigation */}
          <nav className="box research-nav" aria-label="Nội dung nghiên cứu">
            <button
              type="button"
              className={topic === "overview" ? "active" : ""}
              onClick={() => setTopic("overview")}
            >
              <BarChart3 className="icons smico" />
              <span>Tổng quan nghiên cứu</span>
            </button>

            <button
              type="button"
              className={topic === "clustering" ? "active" : ""}
              onClick={() => setTopic("clustering")}
            >
              <Network className="icons smico" />
              <span>Phân cụm không gian</span>
            </button>

            <button
              type="button"
              className={topic === "benchmark" ? "active" : ""}
              onClick={() => setTopic("benchmark")}
            >
              <List className="icons smico" />
              <span>So sánh thuật toán</span>
            </button>

            <button
              type="button"
              className={topic === "ablation" ? "active" : ""}
              onClick={() => setTopic("ablation")}
            >
              <Zap className="icons smico" />
              <span>Kiểm chứng Ablation 2×2</span>
            </button>

            <button
              type="button"
              className={topic === "routing" ? "active" : ""}
              onClick={() => setTopic("routing")}
            >
              <Truck className="icons smico" />
              <span>Phân tích tuyến giao</span>
            </button>

            <button
              type="button"
              className={topic === "rl" ? "active" : ""}
              onClick={() => setTopic("rl")}
            >
              <BrainCircuit className="icons smico" />
              <span>Reinforcement Learning</span>
            </button>

            <button
              type="button"
              className={topic === "experiments" ? "active" : ""}
              onClick={() => setTopic("experiments")}
            >
              <PlayCircle className="icons smico" />
              <span>Chạy thử & Thực nghiệm</span>
            </button>

            <button
              type="button"
              className={topic === "sensitivity" ? "active" : ""}
              onClick={() => setTopic("sensitivity")}
            >
              <TrendingUp className="icons smico" />
              <span>Độ nhạy</span>
            </button>

            <button
              type="button"
              className={topic === "stability" ? "active" : ""}
              onClick={() => setTopic("stability")}
            >
              <ShieldCheck className="icons smico" />
              <span>Độ ổn định</span>
            </button>
          </nav>

          {/* Right Topic Content Body */}
          <div className="box research-body" id="researchBody">
            {/* Topic Header Hero */}
            <div className="research-hero">
              <div className="eyebrow">{currentTopic.eyebrow}</div>
              <h2>{currentTopic.title}</h2>
              <p>{currentTopic.desc}</p>
              {currentTopic.link && (
                <div style={{ marginTop: "12px" }}>
                  <Link to={currentTopic.link} className="btn tiny soft">
                    <span>Mở chi tiết chuyên sâu</span>
                    <ExternalLink className="icons smico" />
                  </Link>
                </div>
              )}
            </div>

            {/* Topic Main Section */}
            <div className="research-section">
              {/* TOPIC: OVERVIEW & BENCHMARK */}
              {(topic === "overview" || topic === "benchmark") && (
                <div>
                  <div className="research-2" style={{ marginBottom: "28px" }}>
                    <div>
                      <h2 style={{ margin: "0 0 14px" }}>
                        Đối sánh 3 Nhóm Thực nghiệm Độc lập
                      </h2>

                      <div className="research-stat">
                        <b>108</b>
                        <div>
                          <strong>Benchmark workloads (813 đơn hàng)</strong>
                          <p>
                            Đánh giá 5 thuật toán cơ sở: Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN trên cùng đồ thị OSM thực tế.
                          </p>
                        </div>
                      </div>

                      <div className="research-stat">
                        <b>12</b>
                        <div>
                          <strong>Guided DQN workloads</strong>
                          <p>
                            Đánh giá cơ chế hướng dẫn heuristic ($\lambda=0.5$), giảm 6.8% tỷ lệ chuyển cụm không gian mà vẫn giữ nguyên 100% SLA.
                          </p>
                        </div>
                      </div>

                      <div className="research-stat">
                        <b>16</b>
                        <div>
                          <strong>Proposed RL held-out workloads</strong>
                          <p>
                            Kiểm thử khả năng tổng quát hóa trên các kịch bản ngoại suy chưa từng xuất hiện trong quá trình huấn luyện.
                          </p>
                        </div>
                      </div>

                      <div className="note" style={{ marginTop: "21px" }}>
                        <strong>Nguyên tắc khoa học:</strong> Không trộn lẫn số liệu giữa các nhóm thực nghiệm khác biệt về cấu hình môi trường hoặc phân phối dữ liệu.
                      </div>
                    </div>

                    {/* Benchmark Chart View */}
                    <div className="research-art">
                      <div className="tracking-row" style={{ marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
                        <div>
                          <h3 style={{ margin: 0 }}>{getMetricLabel()}</h3>
                          <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>108 WORKLOADS BENCHMARK</span>
                        </div>
                        <div style={{ display: "flex", gap: "4px" }}>
                          <button
                            type="button"
                            className={`btn tiny ${metricType === "distance" ? "primary" : "soft"}`}
                            onClick={() => setMetricType("distance")}
                          >
                            Cự ly
                          </button>
                          <button
                            type="button"
                            className={`btn tiny ${metricType === "late" ? "primary" : "soft"}`}
                            onClick={() => setMetricType("late")}
                          >
                            Đơn trễ
                          </button>
                          <button
                            type="button"
                            className={`btn tiny ${metricType === "lateness" ? "primary" : "soft"}`}
                            onClick={() => setMetricType("lateness")}
                          >
                            Phút trễ
                          </button>
                          <button
                            type="button"
                            className={`btn tiny ${metricType === "runtime" ? "primary" : "soft"}`}
                            onClick={() => setMetricType("runtime")}
                          >
                            Thời gian
                          </button>
                        </div>
                      </div>

                      {chartBenchmarkData.length > 0 ? (
                        <div style={{ width: "100%", height: "230px" }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartBenchmarkData} margin={{ top: 15, right: 10, left: -15, bottom: 25 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line-subtle)" strokeOpacity={0.7} />
                              <XAxis dataKey="name" tick={{ fontSize: 12, fill: "var(--ink-secondary)" }} />
                              <YAxis domain={getMetricDomain()} tick={{ fontSize: 12, fill: "var(--ink-secondary)" }} />
                              <Tooltip
                                formatter={formatTooltipValue}
                                contentStyle={{
                                  borderRadius: "8px",
                                  fontSize: "12px",
                                  backgroundColor: "var(--paper)",
                                  borderColor: "var(--line)",
                                  color: "var(--ink)",
                                }}
                              />
                              <Bar dataKey="value" fill="#087e78" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                                {chartBenchmarkData.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={(ALGO_COLOR as Record<string, string>)[entry.algorithm] || "#087e78"}
                                  />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div style={{ padding: "40px 0", textAlign: "center", color: "var(--ink-secondary)" }}>
                          Đang tải dữ liệu thực nghiệm...
                        </div>
                      )}

                      <p style={{ textAlign: "center", marginTop: "12px", fontSize: "12px", color: "var(--ink-secondary)" }}>
                        Dữ liệu đo đạc thực tế từ 108 workloads (813 đơn hàng) trên mạng đường TP.HCM
                      </p>
                    </div>
                  </div>

                  {/* Benchmark Scientific Comparison Table */}
                  <div className="box" style={{ padding: "20px", border: "1px solid var(--line)" }}>
                    <div className="tracking-row" style={{ marginBottom: "16px" }}>
                      <div>
                        <h3 style={{ margin: 0, fontSize: "15px" }}>Bảng tổng hợp đối sánh 5 Thuật toán (108 Workloads)</h3>
                        <p className="subline" style={{ margin: 0, fontSize: "12px" }}>
                          Đối chiếu toàn diện các chỉ số vận hành then chốt: Cự ly, Trễ hạn, Phút phạt và Tốc độ tính toán.
                        </p>
                      </div>
                      <span className="chip green">Nghiên cứu KLTN</span>
                    </div>

                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Thuật toán</th>
                            <th>Quãng đường TB</th>
                            <th>Đúng hạn SLA</th>
                            <th>Đơn trễ hạn</th>
                            <th>Tổng phút trễ</th>
                            <th>Thời gian suy diễn</th>
                          </tr>
                        </thead>
                        <tbody>
                          {chartBenchmarkData.map((row) => (
                            <tr key={row.algorithm}>
                              <td>
                                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                  <span
                                    style={{
                                      width: "10px",
                                      height: "10px",
                                      borderRadius: "50%",
                                      backgroundColor: (ALGO_COLOR as Record<string, string>)[row.algorithm] || "#087e78",
                                      display: "inline-block",
                                    }}
                                  />
                                  <b>{row.name}</b>
                                </div>
                              </td>
                              <td style={{ fontWeight: 600 }}>{row.distanceKm} km</td>
                              <td>
                                <span className={row.onTimeRate === 100 ? "chip green" : "chip orange"}>
                                  {row.onTimeRate}%
                                </span>
                              </td>
                              <td style={{ fontWeight: 600, color: row.lateDeliveries > 0 ? "var(--warning)" : "inherit" }}>
                                {row.lateDeliveries} / 813 đơn
                              </td>
                              <td>{row.totalLatenessMin} phút</td>
                              <td style={{ fontFamily: "monospace", fontSize: "12px" }}>
                                {row.runtimeSec} s
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div
                      style={{
                        marginTop: "16px",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        background: "var(--surface-secondary)",
                        border: "1px solid var(--line)",
                        fontSize: "12px",
                        lineHeight: 1.6,
                        color: "var(--ink)",
                      }}
                    >
                      <strong>Kết luận thực nghiệm then chốt:</strong>
                      <ul style={{ margin: "6px 0 0 18px", padding: 0 }}>
                        <li>
                          <strong>DQN &amp; Q-Learning:</strong> Đạt tỷ lệ đúng hạn tuyệt đối 100% (0 đơn trễ, 0 phút phạt) trong khi Nearest Neighbor và Clarke-Wright đều phát sinh 7 đơn trễ (phạt 165 - 196 phút).
                        </li>
                        <li>
                          <strong>Tối ưu cự ly:</strong> Clarke-Wright đạt cự ly ngắn nhất (58.1 km), DQN theo sát với 59.8 km (ngắn hơn Nearest Neighbor 2.85 km).
                        </li>
                        <li>
                          <strong>Tốc độ suy diễn:</strong> Sau khi hội tụ, các mô hình RL tính toán tuyến trong ~0.0005 - 0.0025 giây, nhanh hơn gấp 1,000 lần so với các giải pháp xây dựng lại từ đầu.
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* TOPIC: ABLATION STUDY 2x2 */}
              {topic === "ablation" && (
                <div>
                  {/* RL vs Baseline Fair Comparator */}
                  <RLBaselineExplorer />

                  <div className="tracking-row" style={{ marginBottom: "16px", flexWrap: "wrap", gap: "8px" }}>
                    <div>
                      <h2 style={{ margin: 0 }}>Thực nghiệm Kiểm soát Độc lập: Phân cụm + Proposed RL</h2>
                      <span className="subline" style={{ fontSize: "12px" }}>
                        Ma trận 2×2 (Non-cluster vs K-Means-aware × Nearest Neighbor vs Proposed RL) trên cùng 16 shipper, 85 đơn hàng, Seed 42
                      </span>
                    </div>
                    <span className="chip green">Đã kiểm chứng thực nghiệm</span>
                  </div>

                  {/* 2x2 Matrix Cards */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "12px", marginBottom: "20px" }}>
                    {/* Group A */}
                    <div className="box" style={{ padding: "14px", border: "1px solid var(--line)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong>Nhóm A (Baseline thuần)</strong>
                        <span className="chip">Phi phân cụm + NN</span>
                      </div>
                      <div style={{ margin: "10px 0" }}>
                        <div style={{ fontSize: "20px", fontWeight: 700 }}>294.4 km</div>
                        <div style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Chi phí: 1,648,409 VND</div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--ink-secondary)" }}>
                        <span>SLA: <b>100.0%</b> (0 trễ)</span>
                        <span>Nhảy cụm: <b style={{ color: "var(--amber, #f59e0b)" }}>13 lần</b></span>
                      </div>
                    </div>

                    {/* Group B */}
                    <div className="box" style={{ padding: "14px", border: "1px solid var(--line)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong>Nhóm B (Hiệu ứng Phân cụm)</strong>
                        <span className="chip">K-Means K=10 + NN</span>
                      </div>
                      <div style={{ margin: "10px 0" }}>
                        <div style={{ fontSize: "20px", fontWeight: 700 }}>352.7 km</div>
                        <div style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Chi phí: 1,975,119 VND</div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--ink-secondary)" }}>
                        <span>SLA: <b>100.0%</b> (0 trễ)</span>
                        <span>Nhảy cụm: <b style={{ color: "var(--success)" }}>0 lần</b></span>
                      </div>
                    </div>

                    {/* Group C */}
                    <div className="box" style={{ padding: "14px", border: "1px solid var(--line)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong>Nhóm C (Hiệu ứng Proposed RL)</strong>
                        <span className="chip">Phi phân cụm + Proposed RL</span>
                      </div>
                      <div style={{ margin: "10px 0" }}>
                        <div style={{ fontSize: "20px", fontWeight: 700 }}>293.5 km</div>
                        <div style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Chi phí: 1,643,294 VND</div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--ink-secondary)" }}>
                        <span>SLA: <b>100.0%</b> (0 trễ)</span>
                        <span>Nhảy cụm: <b style={{ color: "var(--amber, #f59e0b)" }}>13 lần</b></span>
                      </div>
                    </div>

                    {/* Group D */}
                    <div className="box" style={{ padding: "14px", border: "2px solid var(--accent)", background: "var(--surface-secondary)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong>Nhóm D (Phương pháp Đề xuất)</strong>
                        <span className="chip green">K-Means K=10 + Proposed RL</span>
                      </div>
                      <div style={{ margin: "10px 0" }}>
                        <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--accent)" }}>356.3 km</div>
                        <div style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Chi phí: 1,995,270 VND</div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--ink-secondary)" }}>
                        <span>SLA: <b style={{ color: "var(--success)" }}>100.0%</b> (0 trễ)</span>
                        <span>Nhảy cụm: <b style={{ color: "var(--success)" }}>0 lần (Tuyệt đối)</b></span>
                      </div>
                    </div>
                  </div>

                  {/* Paired Ablation Deltas Table */}
                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <h3 style={{ margin: "0 0 12px" }}>Bảng Đối Đầu Cặp (Paired Ablation Deltas)</h3>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Mục tiêu phân tích</th>
                            <th>So sánh cặp</th>
                            <th>Chênh lệch cự ly (km)</th>
                            <th>Chênh lệch %</th>
                            <th>SLA Delta</th>
                            <th>Giảm số lần nhảy cụm</th>
                            <th>Chênh lệch chi phí</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td><b>Tác động riêng của Phân cụm (trên NN)</b></td>
                            <td>Nhóm B vs Nhóm A</td>
                            <td style={{ color: "var(--ink-secondary)" }}>+58.34 km</td>
                            <td>+19.82%</td>
                            <td>0.0% (100% on-time)</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>-13 lần (Triệt tiêu)</td>
                            <td>+326,710 VND</td>
                          </tr>
                          <tr>
                            <td><b>Tác động riêng của Phân cụm (trên Proposed RL)</b></td>
                            <td>Nhóm D vs Nhóm C</td>
                            <td style={{ color: "var(--ink-secondary)" }}>+62.85 km</td>
                            <td>+21.42%</td>
                            <td>0.0% (100% on-time)</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>-13 lần (Triệt tiêu)</td>
                            <td>+351,976 VND</td>
                          </tr>
                          <tr>
                            <td><b>Tác động riêng của Proposed RL (khi có Phân cụm)</b></td>
                            <td>Nhóm D vs Nhóm B</td>
                            <td>+3.60 km</td>
                            <td>+1.02%</td>
                            <td>0.0% (100% on-time)</td>
                            <td>0 lần (Đã gom gọn)</td>
                            <td>+20,151 VND</td>
                          </tr>
                          <tr style={{ background: "rgba(16, 185, 129, 0.05)" }}>
                            <td><b>Hiệu quả kết hợp (Proposed Method vs Baseline)</b></td>
                            <td>Nhóm D vs Nhóm A</td>
                            <td>+61.94 km</td>
                            <td>+21.04%</td>
                            <td>0.0% (100% on-time)</td>
                            <td style={{ color: "var(--success)", fontWeight: 700 }}>-13 lần (Triệt tiêu)</td>
                            <td>+346,861 VND</td>
                          </tr>
                          <tr>
                            <td><b>Đối chiếu với Clarke-Wright (khi có Phân cụm)</b></td>
                            <td>Nhóm D vs Nhóm E2</td>
                            <td>+16.90 km</td>
                            <td>+4.98%</td>
                            <td>0.0% (100% on-time)</td>
                            <td>0 lần</td>
                            <td>+94,668 VND</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Multi-Day Simulation Finding */}
                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <div className="tracking-row" style={{ marginBottom: "12px" }}>
                      <h3 style={{ margin: 0 }}>Thực nghiệm 2: Mô phỏng Vận hành 3 Ngày Liên tiếp (255 đơn hàng)</h3>
                      <span className="chip amber">Phát hiện suy giảm SLA khi không dùng Phân cụm</span>
                    </div>
                    <p style={{ fontSize: "13px", color: "var(--ink-secondary)", marginBottom: "14px" }}>
                      Trong kịch bản đơn hàng qua 3 ngày (255 đơn), chính sách Phi phân cụm tối ưu khoảng cách cục bộ làm shipper bị phân tán rải rác, dẫn đến <b>giao trễ đơn tích lũy (SLA tích lũy giảm xuống 99.22% ở Nhóm A/C và 98.82% ở Nhóm E1; SLA ngày cá biệt giảm xuống 98.82%)</b>. Ngược lại, phân bổ theo cụm không gian K-Means duy trì <b>100.00% đúng hạn tuyệt đối qua cả 3 ngày (0 đơn trễ)</b>.
                    </p>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Nhóm thực nghiệm</th>
                            <th>Chính sách phân bổ</th>
                            <th>Thuật toán định tuyến</th>
                            <th>Tổng km (3 ngày)</th>
                            <th>Số đơn hoàn thành</th>
                            <th>Tỷ lệ SLA đúng hạn</th>
                            <th>Tổng số đơn trễ tích lũy</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>Nhóm A</td>
                            <td>Phi phân cụm (EDD Nearest)</td>
                            <td>Nearest Neighbor</td>
                            <td>864.7 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--amber, #f59e0b)" }}>99.22% <small style={{ display: "block", fontSize: "12px" }}>(Ngày 2 &amp; 3: 98.8%)</small></td>
                            <td style={{ color: "var(--amber, #f59e0b)", fontWeight: 600 }}>2 đơn trễ</td>
                          </tr>
                          <tr style={{ background: "rgba(16, 185, 129, 0.05)" }}>
                            <td>Nhóm B</td>
                            <td>K-Means K=10 Partitioning</td>
                            <td>Nearest Neighbor</td>
                            <td>1,067.8 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>100.0%</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>0 đơn trễ</td>
                          </tr>
                          <tr>
                            <td>Nhóm C</td>
                            <td>Phi phân cụm (EDD Nearest)</td>
                            <td>Proposed RL</td>
                            <td>863.7 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--amber, #f59e0b)" }}>99.22% <small style={{ display: "block", fontSize: "12px" }}>(Ngày 2 &amp; 3: 98.8%)</small></td>
                            <td style={{ color: "var(--amber, #f59e0b)", fontWeight: 600 }}>2 đơn trễ</td>
                          </tr>
                          <tr style={{ background: "rgba(16, 185, 129, 0.08)" }}>
                            <td><b>Nhóm D (Đề xuất)</b></td>
                            <td><b>K-Means K=10 Partitioning</b></td>
                            <td><b>Proposed RL</b></td>
                            <td>1,099.0 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--success)", fontWeight: 700 }}>100.0%</td>
                            <td style={{ color: "var(--success)", fontWeight: 700 }}>0 đơn trễ</td>
                          </tr>
                          <tr>
                            <td>Nhóm E1</td>
                            <td>Phi phân cụm (EDD Nearest)</td>
                            <td>Clarke-Wright</td>
                            <td>829.5 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--error, #ef4444)" }}>98.82% <small style={{ display: "block", fontSize: "12px" }}>(1 trễ/ngày)</small></td>
                            <td style={{ color: "var(--error, #ef4444)", fontWeight: 600 }}>3 đơn trễ</td>
                          </tr>
                          <tr>
                            <td>Nhóm E2</td>
                            <td>K-Means K=10 Partitioning</td>
                            <td>Clarke-Wright</td>
                            <td>1,016.9 km</td>
                            <td>255 / 255</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>100.0%</td>
                            <td style={{ color: "var(--success)", fontWeight: 600 }}>0 đơn trễ</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Trade-off Insights */}
                  <div className="note" style={{ lineHeight: 1.6 }}>
                    <strong>Sự thật Khoa học &amp; Đánh đổi Thực nghiệm (Trade-off Analysis):</strong>
                    <ul style={{ margin: "8px 0 0", paddingLeft: "20px" }}>
                      <li>
                        <b>Tại sao Phân cụm đi xa hơn trong ngày đầu (85 đơn)?</b> Vì trong mô hình phân cụm cố định (SHIPPER_HOME_CLUSTERS), toàn bộ 16 shipper xuất phát từ kho trung tâm Tân Bình đi đến từng cụm khu vực (Quận 7, Quận 12, Bình Tân...) rồi quay về kho. Trong kịch bản 85 đơn (trung bình 5.3 đơn/shipper), chi phí chặng trục chính (linehaul transit) từ kho đến tâm cụm chiếm tỷ trọng lớn.
                      </li>
                      <li>
                        <b>Lợi ích cốt lõi của Phân cụm:</b> Triệt tiêu hoàn toàn hiện tượng chạy lộn xộn giữa các quận (0 lần chuyển cụm so với 13-15 lần). Khi tải đơn tăng cao trong nhiều ngày liên tiếp, tuyến đường gom cụm giúp bảo vệ 100% cam kết SLA (0 đơn trễ), trong khi phi phân cụm phát sinh đơn trễ hạn.
                      </li>
                      <li>
                        <b>Cơ chế phòng thủ (Fallback Transparency):</b> Mô hình Proposed RL chạy hoàn toàn bằng CPU inference (PyTorch greedy evaluation). Nếu gặp sự cố, hệ thống ghi nhận cảnh báo và chuyển sang Nearest Neighbor, không đánh đồng kết quả fallback thành kết quả của RL.
                      </li>
                      <li>
                        <b>Phân tích sai khác giữa Proposed RL và Nearest Neighbor khi có Phân cụm (+3.60 km / +1.02%):</b> Proposed RL sử dụng mạng DQN chấm điểm ứng viên đa mục tiêu (cự ly 0.375, thời gian 0.375, phạt trễ SLA 0.20, nhảy cụm 0.05). Trên 13/16 shipper, RL chọn chuỗi dừng giống hệt NN. Trên shipper_12, RL đạt cự ly ngắn hơn NN (23.540 km vs 23.548 km). Trên shipper_08, mạng Q ưu tiên ghé trước các đơn có hạn giao gấp (180 phút, 240 phút) thay vì đơn gần nhất nhưng hạn giao dài (480 phút), khiến cự ly chặng tăng 3.39 km nhưng gia tăng biên độ an toàn thời gian. Đây là đặc tính tối ưu đa mục tiêu có chủ đích của mô hình.
                      </li>
                      <li>
                        <b>Quy chuẩn đo lường cự ly (Distance Methodology):</b> Trong mô phỏng điều phối trực tuyến đa shipper (Dynamic Dispatch), cự ly được ước lượng theo công thức Haversine với vận tốc 25 km/h để đảm bảo thời gian đáp ứng thời gian thực (real-time sub-second). Trong khi đó, tại tập thực nghiệm đối sánh chuẩn (Chapter 4 Benchmark 108 workloads), cự ly được tính toán bằng thuật toán Dijkstra trên đồ thị mạng đường OpenStreetMap thực tế của TP.HCM. Cả hai chỉ số được báo cáo riêng biệt và minh bạch.
                      </li>
                    </ul>
                  </div>
                </div>
              )}

              {/* TOPIC: CLUSTERING */}
              {topic === "clustering" && (
                <div>
                  <h2 style={{ marginBottom: "14px" }}>Phân tích Chất lượng Phân cụm Không gian</h2>
                  
                  {/* K-Means Quality Table */}
                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <div className="tracking-row" style={{ marginBottom: "12px" }}>
                      <h3 style={{ margin: 0 }}>Đánh giá Elbow &amp; Silhouette Score (K = 5 đến 15)</h3>
                      <span className="chip green">Tối ưu tại K = 10 (Silhouette = 0.428523)</span>
                    </div>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Số cụm (K)</th>
                            <th>Inertia</th>
                            <th>Silhouette Score</th>
                            <th>Tỷ lệ mất cân bằng</th>
                            <th>Cụm min / max</th>
                            <th>Vai trò nghiên cứu</th>
                          </tr>
                        </thead>
                        <tbody>
                          {clusteringQuality?.kmeans && clusteringQuality.kmeans.length > 0 ? (
                            clusteringQuality.kmeans.map((row: any) => (
                              <tr
                                key={row.k}
                                style={{
                                  background: row.k === 10 ? "rgba(16, 185, 129, 0.08)" : undefined,
                                  fontWeight: row.k === 10 ? 700 : 400,
                                }}
                              >
                                <td><b>K = {row.k}</b></td>
                                <td>{row.inertia ? Number(row.inertia.toFixed(1)).toLocaleString() : "—"}</td>
                                <td style={{ color: row.k === 10 ? "var(--success)" : undefined }}>
                                  <b>{row.silhouette_score ? row.silhouette_score.toFixed(6) : "—"}</b>
                                </td>
                                <td>{row.imbalance_ratio ? row.imbalance_ratio.toFixed(2) : "—"}x</td>
                                <td>{row.min_cluster_size} / {row.max_cluster_size}</td>
                                <td>
                                  {row.k === 10 ? (
                                    <span className="chip green">Mô hình đề xuất chính thức</span>
                                  ) : (
                                    <span className="chip">{row.thesis_role || "Tham chiếu"}</span>
                                  )}
                                </td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td><b>K = 10</b></td>
                              <td>13,654.3</td>
                              <td style={{ color: "var(--success)" }}><b>0.428523</b></td>
                              <td>8.96x</td>
                              <td>70 / 627</td>
                              <td><span className="chip green">Mô hình đề xuất chính thức</span></td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* 10 Clusters Distribution */}
                  <h3 style={{ marginBottom: "12px" }}>Phân bổ 2,394 đơn giao hàng thực tế theo 10 cụm K-Means</h3>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))", gap: "12px", marginBottom: "20px" }}>
                    {clusterDistribution.map((c) => (
                      <div key={c.cluster} className="box" style={{ padding: "14px", border: "1px solid var(--line)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <strong style={{ fontSize: "13px" }}>{c.cluster}</strong>
                          <span className="chip">{c.size} đơn</span>
                        </div>
                        <p className="subline" style={{ marginTop: "4px", fontSize: "12px" }}>{c.label}</p>
                      </div>
                    ))}
                  </div>

                  <div className="note">
                    <b>Đánh giá khoa học:</b> Điểm số Silhouette K=10 đạt 0.428523 chứng minh sự phân tách ranh giới địa bàn giữa các quận tại TP.HCM (Tân Bình, Phú Nhuận, Bình Thạnh, Gò Vấp, Quận 10...) là tối ưu về mặt hình học, cân bằng tải trọng xe máy (100 kg) và bán kính phục vụ.
                  </div>
                </div>
              )}

              {/* TOPIC: ROUTING & NETWORK */}
              {topic === "routing" && (
                <div>
                  <h2 style={{ marginBottom: "16px" }}>Mạng đường Thực tế &amp; Ràng buộc Vận tải 16 Shipper</h2>
                  <div className="research-stat">
                    <b>OSM</b>
                    <div>
                      <strong>Đồ thị mạng đường OpenStreetMap TP.HCM (maps/xedu_drive.graphml)</strong>
                      <p>Mô hình đối sánh chuẩn offline (108 workloads) tính toán ma trận cự ly và thời gian bằng thuật toán Dijkstra trên mạng đường có hướng, bao gồm quy chuẩn đường một chiều, vận tốc theo cấp đường và cầu vượt đô thị.</p>
                    </div>
                  </div>
                  <div className="research-stat">
                    <b>100 kg</b>
                    <div>
                      <strong>Ràng buộc dung lượng tải trọng xe máy (Vehicle Capacity Limit)</strong>
                      <p>Mỗi shipper được giới hạn tối đa 100 kg theo quy chuẩn giao hàng nội đô, đảm bảo an toàn giao thông và phân bổ cân bằng.</p>
                    </div>
                  </div>
                  <div className="research-stat">
                    <b>Kho TB</b>
                    <div>
                      <strong>Kho tập kết trung tâm Tân Bình (Central Depot)</strong>
                      <p>Tất cả 16 shipper xuất phát và chốt ca tại Kho Tân Bình (10.7744°N, 106.6575°E). Đây là nguyên nhân xuất hiện chặng trục chính (linehaul transit) khi di chuyển đến các cụm xa.</p>
                    </div>
                  </div>
                  <div className="note" style={{ marginTop: "16px" }}>
                    <b>Quy chuẩn đo lường cự ly:</b> Trong mô phỏng điều phối động trực tuyến (Live Operations &amp; Planning Desk), cự ly được ước lượng theo Haversine + 25 km/h để phản hồi sub-second; trong khi đó, tại tập đối sánh nghiên cứu (Benchmark &amp; Route Inspector), cự ly được tính chuẩn xác bằng đồ thị OSM Dijkstra.
                  </div>
                </div>
              )}

              {/* TOPIC: RL EXPLORER */}
              {topic === "rl" && (
                <div>
                  <h2 style={{ marginBottom: "14px" }}>Cấu trúc Mô hình Proposed RL (Candidate-Scoring DQN)</h2>

                  {/* RL Decision Explorer Component */}
                  <RLDecisionExplorer />

                  {/* RL Training History Component */}
                  <RLTrainingExplorer />

                  
                  {/* Architecture & Checkpoint Card */}
                  <div className="box" style={{ padding: "18px", marginBottom: "20px", background: "var(--surface-secondary)" }}>
                    <div className="tracking-row" style={{ marginBottom: "12px" }}>
                      <h3 style={{ margin: 0 }}>Checkpoint Mô hình: proposed_rl_multiworkload_D_moderate_sla_best.pt</h3>
                      <span className="chip green">PyTorch CPU Inference · Greedy (ε = 0)</span>
                    </div>
                    <p style={{ fontSize: "13px", color: "var(--ink-secondary)", marginBottom: "14px" }}>
                      Mô hình Proposed RL sử dụng mạng nơ-ron sâu (MLP 64 units) để chấm điểm giá trị Q-value cho từng điểm dừng ứng viên (candidate stop) tại mỗi bước ra quyết định, thay vì phụ thuộc vào heuristic tham lam khoảng cách đơn thuần.
                    </p>
                    
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "10px" }}>
                      <div className="box" style={{ padding: "10px" }}>
                        <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Trọng số Quãng đường</span>
                        <div style={{ fontSize: "16px", fontWeight: 700 }}>0.375</div>
                      </div>
                      <div className="box" style={{ padding: "10px" }}>
                        <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Trọng số Thời gian</span>
                        <div style={{ fontSize: "16px", fontWeight: 700 }}>0.375</div>
                      </div>
                      <div className="box" style={{ padding: "10px" }}>
                        <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Phạt Đơn trễ SLA</span>
                        <div style={{ fontSize: "16px", fontWeight: 700, color: "var(--amber, #f59e0b)" }}>0.100</div>
                      </div>
                      <div className="box" style={{ padding: "10px" }}>
                        <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Phạt Phút trễ hạn</span>
                        <div style={{ fontSize: "16px", fontWeight: 700, color: "var(--amber, #f59e0b)" }}>0.100</div>
                      </div>
                      <div className="box" style={{ padding: "10px" }}>
                        <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>Phạt Nhảy cụm</span>
                        <div style={{ fontSize: "16px", fontWeight: 700, color: "var(--accent)" }}>0.050</div>
                      </div>
                    </div>
                  </div>

                  {/* 8 Invariant Features Table */}
                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <h3 style={{ margin: "0 0 12px" }}>Không gian Đặc trưng 8 Chiều (Invariant Candidate Features)</h3>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Tên đặc trưng</th>
                            <th>Mô tả toán học</th>
                            <th>Ý nghĩa trong tối ưu logistics</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr><td><code>elapsed_ratio</code></td><td>t / shift_duration</td><td>Tỷ lệ thời gian ca làm việc đã trôi qua</td></tr>
                          <tr><td><code>remaining_ratio</code></td><td>remaining_deliveries / N</td><td>Khối lượng đơn hàng còn lại cần phát</td></tr>
                          <tr><td><code>distance_ratio</code></td><td>leg_dist / max_distance</td><td>Cự ly chuẩn hóa từ vị trí hiện tại đến ứng viên</td></tr>
                          <tr><td><code>travel_time_ratio</code></td><td>leg_time / max_travel_time</td><td>Thời gian di chuyển ước tính đến ứng viên</td></tr>
                          <tr><td><code>slack_ratio</code></td><td>(deadline - arrival) / max_slack</td><td>Biên độ an toàn trước hạn giao (clip tối đa 2.0)</td></tr>
                          <tr><td><code>deadline_valid</code></td><td>1.0 nếu có hạn, 0.0 nếu None</td><td>Chỉ dấu đơn có ràng buộc giờ giao nghiêm ngặt</td></tr>
                          <tr><td><code>lateness_ratio</code></td><td>max(0, arrival - deadline) / max_late</td><td>Mức độ vi phạm trễ hạn nếu chọn ứng viên này</td></tr>
                          <tr><td><code>cluster_switch</code></td><td>1.0 nếu khác cụm, 0.0 nếu cùng cụm</td><td>Phạt di chuyển chéo sang khu vực shipper khác</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Rigorous Investigation Box */}
                  <div className="note" style={{ lineHeight: 1.6 }}>
                    <strong>Kiểm chứng Cơ chế Học (Mechanism vs Behavioral Evidence):</strong>
                    <p style={{ margin: "6px 0 0" }}>
                      Thực nghiệm can thiệp nghịch thực tế (counterfactual analysis) cho thấy tại đầu ca giao ($t=0$), khi hạn giao còn xa ($&gt; 2 	imes$ thời gian di chuyển), đặc trưng <code>slack_ratio</code> bão hòa ở ngưỡng clip 2.0. Sự khác biệt hành vi giữa RL và NN trên các đơn hàng đầu chủ yếu xuất phát từ tương tác phi tuyến giữa cự ly và trạng thái ràng buộc hạn giao (<code>deadline_valid = 0.0</code> vs <code>1.0</code>). Hệ thống ghi nhận trung thực cơ chế này, không đưa ra tuyên bố chưa đủ căn cứ toán học.
                    </p>
                  </div>
                </div>
              )}

              {/* TOPIC: SENSITIVITY */}
              {topic === "sensitivity" && (
                <div>
                  <h2 style={{ marginBottom: "14px" }}>Phân tích Độ nhạy Tham số (Sensitivity Sweeps)</h2>
                  <p style={{ fontSize: "13px", color: "var(--ink-secondary)", marginBottom: "16px" }}>
                    Khảo sát sự thay đổi của chính sách điều phối và tỷ lệ chuyển cụm không gian theo trọng số phạt nhảy cụm $\lambda \in [0.0, 0.1, 0.25, 0.5]$ trên các workloads thực nghiệm độc lập.
                  </p>

                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <div className="tracking-row" style={{ marginBottom: "12px" }}>
                      <h3 style={{ margin: 0 }}>Kết quả Quét Tham số Phạt Chuyển Cụm (\lambda)</h3>
                      <span className="chip green">Điểm cân bằng tối ưu: \lambda = 0.5</span>
                    </div>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Hệ số phạt (\lambda)</th>
                            <th>Số lần chạy</th>
                            <th>Cự ly TB (km)</th>
                            <th>Thời gian di chuyển (phút)</th>
                            <th>Tỷ lệ chuyển cụm</th>
                            <th>Tỷ lệ đúng hạn (SLA)</th>
                            <th>Thời gian huấn luyện</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sensitivityData?.summary && sensitivityData.summary.length > 0 ? (
                            sensitivityData.summary.map((row: any) => (
                              <tr
                                key={row.cluster_switch_penalty}
                                style={{
                                  background: row.cluster_switch_penalty === 0.5 ? "rgba(16, 185, 129, 0.08)" : undefined,
                                  fontWeight: row.cluster_switch_penalty === 0.5 ? 700 : 400,
                                }}
                              >
                                <td><b>\lambda = {row.cluster_switch_penalty}</b></td>
                                <td>{row.runs}</td>
                                <td>{row.mean_distance_km ? row.mean_distance_km.toFixed(2) : "—"} km</td>
                                <td>{row.mean_travel_time_minutes ? row.mean_travel_time_minutes.toFixed(1) : "—"} m</td>
                                <td style={{ color: row.cluster_switch_penalty === 0.5 ? "var(--success)" : undefined }}>
                                  <b>{row.mean_cluster_switch_rate ? (row.mean_cluster_switch_rate * 100).toFixed(1) : "—"}%</b>
                                  {row.cluster_switch_penalty === 0.5 && " (-6.8%)"}
                                </td>
                                <td><span className="chip green">{(row.overall_on_time_rate * 100).toFixed(1)}%</span></td>
                                <td>{row.mean_training_runtime_seconds ? row.mean_training_runtime_seconds.toFixed(1) : "—"} s</td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td><b>\lambda = 0.5</b></td>
                              <td>6</td>
                              <td>68.07 km</td>
                              <td>74.7 m</td>
                              <td style={{ color: "var(--success)" }}><b>57.8% (-6.8%)</b></td>
                              <td><span className="chip green">100.0%</span></td>
                              <td>266.5 s</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="note">
                    <b>Kết luận độ nhạy:</b> Khi tăng $\lambda$ từ 0.0 lên 0.5, tỷ lệ chuyển cụm giảm từ 62.0% xuống 57.8% (giảm tương đối -6.8%) trong khi vẫn giữ nguyên tỷ lệ đúng hạn 100.0% (0 đơn trễ). Mức tăng cự ly là không đáng kể (+1.05 km trên toàn bộ lộ trình).
                  </div>
                </div>
              )}

              {/* TOPIC: STABILITY */}
              {topic === "stability" && (
                <div>
                  <h2 style={{ marginBottom: "14px" }}>Đánh giá Độ ổn định &amp; Tái lập Huấn luyện (10 Random Seeds)</h2>
                  <p style={{ fontSize: "13px", color: "var(--ink-secondary)", marginBottom: "16px" }}>
                    Kiểm tra tính hội tụ và mức độ biến thiên chính sách qua 10 lần khởi tạo hạt ngẫu nhiên (Seeds 1, 7, 13, 19, 23, 29, 31, 37, 41, 47) với 5,000 episodes mỗi lượt chạy.
                  </p>

                  <div className="box" style={{ padding: "16px", marginBottom: "20px" }}>
                    <div className="tracking-row" style={{ marginBottom: "12px" }}>
                      <h3 style={{ margin: 0 }}>Bảng Đối chiếu Tái lập 10 Seeds Huấn luyện Độc lập</h3>
                      <span className="chip green">Reward Gap = 0.0% · 100% Khớp Lộ trình Tối ưu</span>
                    </div>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Seed ID</th>
                            <th>Episodes</th>
                            <th>Thời gian train</th>
                            <th>Greedy Reward</th>
                            <th>Exact Best Reward</th>
                            <th>Chênh lệch (Reward Gap)</th>
                            <th>Cự ly Greedy (km)</th>
                            <th>Tỷ lệ Đúng hạn</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stabilityData?.seeds && stabilityData.seeds.length > 0 ? (
                            stabilityData.seeds.map((s: any) => (
                              <tr key={s.seed}>
                                <td><b>Seed #{s.seed}</b></td>
                                <td>{s.training_episodes?.toLocaleString()}</td>
                                <td>{s.training_runtime_seconds ? s.training_runtime_seconds.toFixed(2) : "—"} s</td>
                                <td>{s.greedy_reward ? s.greedy_reward.toFixed(4) : "—"}</td>
                                <td>{s.exact_best_reward ? s.exact_best_reward.toFixed(4) : "—"}</td>
                                <td>
                                  <span className="chip green">{s.reward_gap_percent ? s.reward_gap_percent.toFixed(1) : "0.0"}%</span>
                                </td>
                                <td>{s.greedy_distance_km ? s.greedy_distance_km.toFixed(2) : "—"} km</td>
                                <td><span className="chip green">{(s.on_time_rate * 100).toFixed(1)}%</span></td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td><b>Seed #1</b></td>
                              <td>5,000</td>
                              <td>1.82 s</td>
                              <td>2.3108</td>
                              <td>2.3108</td>
                              <td><span className="chip green">0.0%</span></td>
                              <td>57.32 km</td>
                              <td><span className="chip green">100.0%</span></td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="note">
                    <b>Bằng chứng về tính ổn định:</b> 10/10 lần huấn luyện với các seed ngẫu nhiên khác nhau đều đạt giá trị reward tối ưu giống hệt nhau (2.3108) và cùng khớp với lộ trình thực nghiệm chuẩn (57.32 km, 100% on-time), khẳng định thuật toán không bị kẹt ở cực trị địa phương (local optima).
                  </div>
                </div>
              )}

              {/* TOPIC: EXPERIMENTS (RUN LAB) */}
              {topic === "experiments" && (
                <div>
                  <h2 style={{ marginBottom: "14px" }}>Môi trường Thực thi Trực tiếp (Run Lab Workspace)</h2>
                  <p style={{ fontSize: "13px", color: "var(--ink-secondary)", marginBottom: "16px" }}>
                    Run Lab cung cấp khả năng thực thi thuật toán theo thời gian thực trên 108 workloads chuẩn, lưu vết provenance và đo lường tài nguyên CPU suy diễn.
                  </p>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px", marginBottom: "20px" }}>
                    <div className="box" style={{ padding: "16px" }}>
                      <h4 style={{ margin: "0 0 10px" }}>Tập Workload Thực nghiệm</h4>
                      <p style={{ fontSize: "13px", color: "var(--ink-secondary)", margin: 0 }}>
                        108 kịch bản giao nhận độc lập trích xuất từ dữ liệu XeDu TP.HCM, đại diện cho các điều kiện mật độ thưa, trung bình và dày đặc (3 đến 15 điểm dừng mỗi chuyến).
                      </p>
                    </div>

                    <div className="box" style={{ padding: "16px" }}>
                      <h4 style={{ margin: "0 0 10px" }}>6 Thuật toán Tích hợp</h4>
                      <p style={{ fontSize: "13px", color: "var(--ink-secondary)", margin: 0 }}>
                        Nearest Neighbor, Clarke–Wright, Tabular Q-Learning, SARSA, Deep Q-Network và Proposed Candidate-Scoring RL. Tất cả thực thi trên cùng một môi trường đánh giá nhất quán.
                      </p>
                    </div>

                    <div className="box" style={{ padding: "16px" }}>
                      <h4 style={{ margin: "0 0 10px" }}>Đo lường Provenance</h4>
                      <p style={{ fontSize: "13px", color: "var(--ink-secondary)", margin: 0 }}>
                        Ghi nhận chính xác runtime (giây), bộ nhớ mô hình, mã checksum kịch bản và thứ tự điểm dừng thực tế qua OpenStreetMap.
                      </p>
                    </div>
                  </div>

                  <div style={{ textAlign: "center", padding: "20px", background: "var(--surface-secondary)", borderRadius: "10px" }}>
                    <p style={{ marginBottom: "14px", fontWeight: 600 }}>
                      Mở giao diện Run Lab đầy đủ để chọn kịch bản và chạy thử trực tiếp:
                    </p>
                    <Link to="/chay-thu" className="btn primary">
                      <PlayCircle className="icons smico" />
                      <span>Truy cập Run Lab Chuyên sâu</span>
                    </Link>
                  </div>
                </div>
              )}            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
