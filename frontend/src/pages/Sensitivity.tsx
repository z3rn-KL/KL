import { useState, useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  MetricCard,
  PageHeader,
  LoadingCard,
  ErrorCard,
  ResearchCallout,
} from "@/components/shared/DomainComponents";
import {
  Sliders,
  MapPin,
  Clock,
  TrendingDown,
  CheckCircle2,
  ShieldCheck,
} from "lucide-react";
import { api, SensitivityRow } from "../lib/api";
import { num, pct, signedPct } from "../lib/format";
import { useAsync } from "../lib/useAsync";

const n = (v: unknown): number | null => (typeof v === "number" ? v : null);

export default function Sensitivity() {
  const sens = useAsync((s) => api.sensitivity(s), []);
  const val = useAsync((s) => api.finalValidation(s), []);

  const [selectedLambda, setSelectedLambda] = useState<number>(0.5);
  const [workloadFilter, setWorkloadFilter] = useState<"all" | "shorter" | "equal" | "longer">("all");

  const s = val.data?.summary;
  const sensitivityRows = sens.data?.summary ?? [];

  // Active lambda row data
  const activeLambdaRow = useMemo<SensitivityRow | undefined>(() => {
    return sensitivityRows.find((r) => Math.abs(r.cluster_switch_penalty - selectedLambda) < 0.01) || sensitivityRows[sensitivityRows.length - 1];
  }, [sensitivityRows, selectedLambda]);

  // Baseline lambda = 0.0 row for relative diff comparison
  const lambdaZeroRow = useMemo<SensitivityRow | undefined>(() => {
    return sensitivityRows.find((r) => Math.abs(r.cluster_switch_penalty - 0.0) < 0.01);
  }, [sensitivityRows]);

  // Filtered paired workloads
  const pairedWorkloads = useMemo(() => {
    const list = val.data?.paired_workloads ?? [];
    if (workloadFilter === "all") return list;
    return list.filter((w) => {
      const delta = -(n(w.distance_improvement_pct) ?? 0);
      if (workloadFilter === "shorter") return delta < -1e-4;
      if (workloadFilter === "equal") return Math.abs(delta) <= 1e-4;
      if (workloadFilter === "longer") return delta > 1e-4;
      return true;
    });
  }, [val.data?.paired_workloads, workloadFilter]);

  // Breakdown counts for paired workloads
  const workloadStats = useMemo(() => {
    const list = val.data?.paired_workloads ?? [];
    let shorter = 0;
    let equal = 0;
    let longer = 0;
    for (const w of list) {
      const delta = -(n(w.distance_improvement_pct) ?? 0);
      if (delta < -1e-4) shorter++;
      else if (Math.abs(delta) <= 1e-4) equal++;
      else longer++;
    }
    return { total: list.length, shorter, equal, longer };
  }, [val.data?.paired_workloads]);

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Phân tích độ nhạy hệ số phạt λ"
        subtitle="Khảo sát đánh đổi giữa cự ly di chuyển và tỷ lệ gom cụm qua 4 giá trị λ (6 WL, 24 lượt chạy thực nghiệm)"
      />

      <div className="space-y-4">
        {(sens.loading || val.loading) && <LoadingCard message="Đang nạp phân tích độ nhạy..." />}
        {sens.error && <ErrorCard message={sens.error} />}
        {val.error && <ErrorCard message={val.error} />}

        {/* 1. Interactive Lambda Selector Strip */}
        <Card className="border-border/80 shadow-xs">
          <CardHeader className="p-3.5 sm:px-4 pb-2">
            <CardTitle className="text-sm font-bold tracking-tight">
              Lựa chọn Hệ số Phạt Chuyển Cụm λ
            </CardTitle>
            <CardDescription className="text-xs">
              4 giá trị λ rời rạc trên 6 Workload khảo sát (24 lượt chạy) • λ = 0,50 là giá trị chọn thực nghiệm
            </CardDescription>
          </CardHeader>
          <CardContent className="p-3.5 sm:px-4 pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {sensitivityRows.map((r) => {
                const valNum = r.cluster_switch_penalty;
                const isSelected = Math.abs(valNum - selectedLambda) < 0.01;
                const isChosenOptimal = Math.abs(valNum - 0.5) < 0.01;
                return (
                  <button
                    key={valNum}
                    onClick={() => setSelectedLambda(valNum)}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                      isSelected
                        ? "border-primary bg-primary/10 shadow-xs ring-1 ring-primary/40"
                        : "border-border/70 bg-card hover:bg-muted/40 hover:border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`text-sm font-bold font-mono ${
                          isSelected ? "text-primary font-extrabold" : "text-foreground"
                        }`}
                      >
                        λ = {num(valNum, 2)}
                      </span>
                      {isChosenOptimal ? (
                        <Badge
                          variant="outline"
                          className="text-[10px] px-1 py-0 font-semibold bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
                        >
                          ★ Chọn
                        </Badge>
                      ) : (
                        <span className="text-[10px] text-muted-foreground">{r.runs} lượt</span>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground mt-1 truncate">
                      Đổi cụm: <b>{num(r.mean_cluster_switches, 2)}</b> ({pct(r.mean_cluster_switch_rate, 1)})
                    </div>
                    <div className="text-[10px] text-muted-foreground/80 mt-0.5 truncate font-mono">
                      {num(r.mean_distance_km, 1)} km · SLA {pct(r.overall_on_time_rate, 0)}
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* 3. Trade-off Dynamics Analytical Charts */}
        <div className="grid cols-2">
          <Card
            title="Phân tích đánh đổi (Trade-off Analysis): Quãng đường & Thời gian theo λ"
            sub="Quãng đường (km) và Thời gian (phút) tăng nhẹ khi hệ số phạt λ tăng dần từ 0,0 lên 0,50."
          >
            <div>
              <div style={{ height: 260 }}>
                <ResponsiveContainer>
                  <LineChart data={sensitivityRows} margin={{ top: 12, right: 24, bottom: 8, left: 4 }}>
                    <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="cluster_switch_penalty"
                      tick={{ fontSize: 12, fill: "var(--muted)" }}
                      tickFormatter={(v: number) => `λ=${num(v, 2)}`}
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: "var(--muted)" }}
                      domain={["auto", "auto"]}
                      width={52}
                      tickFormatter={(v: number) => num(v, 1)}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as SensitivityRow;
                        return (
                          <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", fontSize: 12, color: "var(--text)" }}>
                            <div style={{ fontWeight: 700, marginBottom: 4, color: "var(--accent)" }}>Hệ số λ = {num(d.cluster_switch_penalty, 2)}</div>
                            <div>Quãng đường: <b>{num(d.mean_distance_km, 2)} km</b></div>
                            <div>Thời gian: <b>{num(d.mean_travel_time_minutes, 2)} phút</b></div>
                            <div style={{ color: "var(--muted)", fontSize: 11, marginTop: 4 }}>Huấn luyện: {num(d.mean_training_runtime_seconds, 1)}s</div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
                    <ReferenceLine x={0.5} stroke="var(--accent)" strokeDasharray="4 4" label={{ value: "λ*=0,50", fill: "var(--accent)", fontSize: 11 }} />
                    <Line
                      type="monotone"
                      dataKey="mean_distance_km"
                      name="Quãng đường TB (km)"
                      stroke="#38BDF8"
                      strokeWidth={2.5}
                      dot={{ r: 4, fill: "#38BDF8" }}
                      activeDot={{ r: 6 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="mean_travel_time_minutes"
                      name="Thời gian TB (phút)"
                      stroke="#98A2F5"
                      strokeWidth={2.5}
                      dot={{ r: 4, fill: "#98A2F5" }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="axis-note" style={{ marginTop: 8 }}>
                <span>ℹ️</span> Cả 4 mức λ đều giữ 100% SLA. Tại λ = 0,50, cự ly tăng +1,57% (+1,06 km) so với λ = 0.
              </div>
            </div>
          </Card>

          <Card
            title="Hiệu quả Gom cụm Không gian theo λ"
            sub="Số lần chuyển cụm và Tỷ lệ đổi cụm (%) giảm rõ rệt khi áp dụng mức phạt λ = 0,50."
          >
            <div>
              <div style={{ height: 260 }}>
                <ResponsiveContainer>
                  <LineChart data={sensitivityRows} margin={{ top: 12, right: 24, bottom: 8, left: 4 }}>
                    <CartesianGrid stroke="var(--grid-color)" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="cluster_switch_penalty"
                      tick={{ fontSize: 12, fill: "var(--muted)" }}
                      tickFormatter={(v: number) => `λ=${num(v, 2)}`}
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: "var(--muted)" }}
                      domain={[4.0, 5.5]}
                      width={44}
                      tickFormatter={(v: number) => num(v, 1)}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as SensitivityRow;
                        return (
                          <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px", fontSize: 12, color: "var(--text)" }}>
                            <div style={{ fontWeight: 700, marginBottom: 4, color: "var(--algo-guided)" }}>Hệ số λ = {num(d.cluster_switch_penalty, 2)}</div>
                            <div>Đổi cụm TB: <b>{num(d.mean_cluster_switches, 2)} lần</b></div>
                            <div>Tỷ lệ đổi cụm: <b>{pct(d.mean_cluster_switch_rate, 1)}</b></div>
                            <div style={{ color: "var(--good)", fontSize: 11, marginTop: 4 }}>SLA: {pct(d.overall_on_time_rate, 0)} (100% đúng hẹn)</div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
                    <ReferenceLine x={0.5} stroke="var(--good)" strokeDasharray="4 4" label={{ value: "Giảm 6,83%", fill: "var(--good)", fontSize: 11 }} />
                    <Line
                      type="monotone"
                      dataKey="mean_cluster_switches"
                      name="Số lần đổi cụm TB (lần)"
                      stroke="#34D399"
                      strokeWidth={2.5}
                      dot={{ r: 4, fill: "#34D399" }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="axis-note" style={{ marginTop: 8 }}>
                <span>🎯</span> Tại λ = 0,50, số lần đổi cụm trung bình giảm từ 5,00 xuống 4,67 lần (tỷ lệ đổi cụm giảm từ 62,01% xuống 57,78%).
              </div>
            </div>
          </Card>
        </div>

        {/* 4. Sensitivity Summary Matrix Table */}
        <Card
          title="Bảng Đối sánh Chi tiết 4 Mức Hệ số λ"
          sub="Số liệu thực nghiệm thu được từ 6 workload khảo sát (mỗi workload 9 đơn hàng)."
        >
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th className="num">Hệ số λ</th>
                  <th className="num">Số lượt chạy</th>
                  <th className="num">Quãng đường (km)</th>
                  <th className="num">Thời gian (phút)</th>
                  <th className="num">Tỷ lệ đúng hạn</th>
                  <th className="num">Đổi cụm TB</th>
                  <th className="num">Tỷ lệ đổi cụm</th>
                  <th className="num">Thời gian train</th>
                  <th>Đánh giá khoa học</th>
                </tr>
              </thead>
              <tbody>
                {sensitivityRows.map((r) => {
                  const isSel = Math.abs(r.cluster_switch_penalty - selectedLambda) < 0.01;
                  const isBest = Math.abs(r.cluster_switch_penalty - 0.50) < 0.01;
                  return (
                    <tr
                      key={r.cluster_switch_penalty}
                      className={isSel ? "sel" : ""}
                      onClick={() => setSelectedLambda(r.cluster_switch_penalty)}
                      style={{ cursor: "pointer" }}
                    >
                      <td className="num" style={{ fontWeight: 700, color: isBest ? "var(--accent)" : undefined }}>
                        {num(r.cluster_switch_penalty, 2)}
                        {isBest ? " ★" : ""}
                      </td>
                      <td className="num">{num(r.runs, 0)}</td>
                      <td className="num">{num(r.mean_distance_km, 2)}</td>
                      <td className="num">{num(r.mean_travel_time_minutes, 2)}</td>
                      <td className="num" style={{ color: "var(--good)", fontWeight: 600 }}>
                        {pct(r.overall_on_time_rate, 1)}
                      </td>
                      <td className="num" style={{ fontWeight: isBest ? 700 : undefined, color: isBest ? "var(--good)" : undefined }}>
                        {num(r.mean_cluster_switches, 2)}
                      </td>
                      <td className="num" style={{ fontWeight: isBest ? 700 : undefined, color: isBest ? "var(--good)" : undefined }}>
                        {pct(r.mean_cluster_switch_rate, 1)}
                      </td>
                      <td className="num">{num(r.mean_training_runtime_seconds, 1)}s</td>
                      <td style={{ fontSize: 12 }}>
                        {isBest ? (
                          <span className="chip good" style={{ fontSize: 11 }}>
                            Cân bằng thực nghiệm: Giảm -6,83% đổi cụm
                          </span>
                        ) : r.cluster_switch_penalty === 0.0 ? (
                          <span className="chip" style={{ fontSize: 11 }}>
                            Vanilla DQN (Không phạt cụm)
                          </span>
                        ) : (
                          <span className="muted">Tín hiệu phạt chưa đủ mạnh để thay đổi lộ trình</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-3">
            <ResearchCallout
              type="info"
              title="Ghi chú khoa học về hệ số phạt λ = 0,50"
            >
              Hệ số λ = 0,50 là điểm cân bằng thực nghiệm (empirical trade-off) được lựa chọn trong tập khảo sát 4 giá trị rời rạc λ ∈ &#123;0.0, 0.1, 0.25, 0.50&#125; (24 lượt chạy trên 6 workload). Đây là điểm cân bằng thực nghiệm tối ưu trong tập khảo sát, không phải là nghiệm tối ưu Pareto toàn cục trên toàn không gian tham số liên tục ℝ⁺. Tại mức này, mô hình đạt mức giảm đổi cụm lớn nhất (-6,83%) với mức gia tăng cự ly khiêm tốn (+1,76%) và bảo toàn 100% SLA đúng hạn.
            </ResearchCallout>
          </div>
        </Card>

        {/* 5. 12-Workload Paired Validation Section */}
        {s && (
          <Card
            title="Kiểm chứng Cặp đôi trên 12 Workload Thực tế (DQN thường vs DQN + K-Means λ = 0,50)"
            sub="Đối chiếu từng cặp tuyến đường (Paired-sample validation) trên 12 workload độc lập (108 đơn hàng)."
          >
            {/* Top Aggregate Stats Strip */}
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4 mb-4">
              <MetricCard
                label="Hệ số λ được chọn"
                value="λ = 0,50"
                hint={`${num(n(s.workloads), 0)} workload · ${num(n(s.deliveries), 0)} đơn`}
              />
              <MetricCard
                label="Δ Quãng đường (12 Workload)"
                value="+1,76%"
                hint="Guided dài hơn Vanilla (+1,27 km đánh đổi) trên toàn bộ 12 workload"
              />
              <MetricCard
                label="Phân bố cự ly (12 Workload)"
                value={`${workloadStats.shorter} / ${workloadStats.equal} / ${workloadStats.longer}`}
                hint={`${pct(workloadStats.shorter / workloadStats.total, 0)} ngắn · ${pct(workloadStats.equal / workloadStats.total, 0)} bằng · ${pct(workloadStats.longer / workloadStats.total, 0)} dài`}
              />
              <MetricCard
                label="Δ Tỷ lệ đổi cụm"
                value="-6,83%"
                hint="0,6201 → 0,5778 trên 6 workload đối sánh"
                trend={{ value: "Gom cụm", isPositive: true }}
              />
            </div>

            {/* Filter Pills for Paired Workloads */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
              <Tabs
                value={workloadFilter}
                onValueChange={(v) => setWorkloadFilter(v as any)}
              >
                <TabsList className="h-8">
                  <TabsTrigger value="all" className="text-xs">
                    Tất cả ({workloadStats.total})
                  </TabsTrigger>
                  <TabsTrigger value="shorter" className="text-xs">
                    Ngắn hơn ({workloadStats.shorter})
                  </TabsTrigger>
                  <TabsTrigger value="equal" className="text-xs">
                    Tương đương ({workloadStats.equal})
                  </TabsTrigger>
                  <TabsTrigger value="longer" className="text-xs">
                    Dài hơn ({workloadStats.longer})
                  </TabsTrigger>
                </TabsList>
              </Tabs>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                Đang hiển thị {pairedWorkloads.length}/{workloadStats.total} workload
              </div>
            </div>

            {/* Paired Workload Table */}
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="num">Workload</th>
                    <th className="num">Số đơn</th>
                    <th className="num">QĐ thường (km)</th>
                    <th className="num">QĐ Guided (km)</th>
                    <th className="num">Δ Quãng đường</th>
                    <th className="num">Δ Thời gian</th>
                    <th className="num">Đổi cụm (Guided)</th>
                    <th>Trạng thái đánh đổi</th>
                    <th>Nguồn số liệu</th>
                  </tr>
                </thead>
                <tbody>
                  {pairedWorkloads.map((w) => {
                    const distImpr = n(w.distance_improvement_pct);
                    const timeImpr = n(w.time_improvement_pct);
                    const deltaDist = distImpr !== null ? -distImpr : null;
                    const deltaTime = timeImpr !== null ? -timeImpr : null;
                    const isShorter = deltaDist !== null && deltaDist < -1e-4;
                    const isEqual = deltaDist !== null && Math.abs(deltaDist) <= 1e-4;
                    const isLonger = deltaDist !== null && deltaDist > 1e-4;

                    return (
                      <tr key={String(w.workload_id)}>
                        <td className="num" style={{ fontWeight: 700 }}>
                          #{String(w.workload_id)}
                        </td>
                        <td className="num">{num(n(w.number_of_deliveries), 0)}</td>
                        <td className="num">{num(n(w.baseline_distance_km), 2)}</td>
                        <td className="num" style={{ fontWeight: 600 }}>
                          {num(n(w.guided_distance_km), 2)}
                        </td>
                        <td
                          className="num"
                          style={{
                            fontWeight: 700,
                            color: isShorter ? "var(--good)" : isLonger ? "var(--algo-tradeoff)" : "var(--muted)",
                          }}
                        >
                          {signedPct(deltaDist, 2)}
                        </td>
                        <td
                          className="num"
                          style={{
                            fontWeight: 700,
                            color: deltaTime !== null && deltaTime < -1e-4 ? "var(--good)" : deltaTime !== null && deltaTime > 1e-4 ? "var(--algo-tradeoff)" : "var(--muted)",
                          }}
                        >
                          {signedPct(deltaTime, 2)}
                        </td>
                        <td className="num" style={{ fontWeight: 600 }}>
                          {num(n(w.guided_cluster_switches), 0)}
                        </td>
                        <td>
                          {isShorter ? (
                            <span className="chip good" style={{ fontSize: 11 }}>
                              ✓ Ngắn hơn (-{num(Math.abs(deltaDist!), 2)}%)
                            </span>
                          ) : isEqual ? (
                            <span className="chip" style={{ fontSize: 11 }}>
                              = Tương đương (0,00%)
                            </span>
                          ) : (
                            <span className="chip warn" style={{ fontSize: 11 }}>
                              Đánh đổi (+{num(deltaDist!, 2)}%)
                            </span>
                          )}
                        </td>
                        <td className="muted" style={{ fontSize: 12 }}>
                          {w.result_source === "new_training"
                            ? "Huấn luyện mới"
                            : w.result_source === "reused_sensitivity"
                              ? "Dùng lại từ độ nhạy"
                              : String(w.result_source ?? "–")}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <details style={{ marginTop: 14, padding: "12px 16px", background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.6 }}>
              <summary style={{ fontWeight: 700, color: "var(--text)", cursor: "pointer", userSelect: "none" }}>
                💡 Xem chi tiết phân tích đánh đổi (Trade-off Analysis) từ Dữ liệu Thực nghiệm 12 Workload
              </summary>
              <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div>
                  • <b>Quãng đường & Thời gian (12 Workload, 108 đơn hàng):</b> Đánh giá trên toàn bộ 12 workload so sánh giữa Guided DQN và Vanilla DQN. Guided DQN làm tăng trung bình +1,76% quãng đường (72,07 km → 73,34 km) và +1,87% thời gian lăn bánh (78,70 ph → 80,17 ph), bảo toàn 100% SLA đúng hạn.
                </div>
                <div>
                  • <b>Chuyển cụm Không gian (6 Workload đối sánh tuyến):</b> Chỉ có 6 workload lưu đầy đủ dữ liệu tuyến chi tiết của Vanilla DQN để tính toán số lần chuyển cụm. Trên 6 workload này, tỷ lệ chuyển cụm giảm từ 62,01% xuống 57,78% (giảm tương đối -6,83%), giúp shipper gom đơn theo từng cụm địa lý.
                </div>
                <div>
                  • <b>Phân bố 12 workload:</b> 7/12 workload (58,3%) cho ra đúng tuyến đường của DQN thường (Δ = 0,00%); 4/12 workload (33,3%) chấp nhận tăng cự ly (+0,8% đến +14,3%) để gom đơn trong cùng cụm; 1/12 workload (Workload #10) cự ly ngắn hơn -0,79%.
                </div>
              </div>
            </details>
          </Card>
        )}

        {/* 6. Protocol & Architecture Comparison (Collapsible) */}
        <details
          className="card"
          style={{
            padding: "16px 20px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            cursor: "pointer",
          }}
        >
          <summary style={{ fontWeight: 700, fontSize: 13.5, color: "var(--text)", outline: "none", userSelect: "none" }}>
            📖 Xem luận chứng phương pháp nghiên cứu & so sánh cơ chế hàm thưởng (Guided DQN vs Proposed RL)
          </summary>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 14 }}>
            <div
              style={{
                padding: "14px 16px",
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text)" }}>
                  1. Guided DQN: Reward Shaping Mềm
                </span>
                <span className="chip" style={{ fontSize: 10.5, background: "rgba(139, 140, 248, 0.15)", color: "var(--algo-guided)" }}>
                  Kiểm chứng 12 WL
                </span>
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                <div style={{ marginBottom: 4 }}>
                  <b>Hàm phần thưởng:</b>{" "}
                  <code>R&apos;_t = R_t − λ_c · 𝕀(c_t ≠ c_t+1)</code>
                </div>
                <div>
                  Bổ sung tín hiệu không gian phạt mềm <code>λ_c = 0,50</code> khi shipper chuyển giữa 2 cụm địa lý khác nhau để thúc đẩy gom đơn trong cùng một cụm.
                </div>
              </div>
            </div>

            <div
              style={{
                padding: "14px 16px",
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text)" }}>
                  2. Proposed RL: Mạng Đa Tiêu chí
                </span>
                <span className="chip" style={{ fontSize: 10.5, background: "rgba(54, 181, 197, 0.15)", color: "var(--accent)" }}>
                  Held-out Test 16 WL
                </span>
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                <div style={{ marginBottom: 4 }}>
                  <b>Hàm mục tiêu tuyến:</b>{" "}
                  <code>J(π) = 0,375D̄ + 0,375T̄ + 0,10M̄ + 0,10L̄ + 0,05S̄</code>
                </div>
                <div>
                  Chấm điểm từng ứng viên giao hàng qua mạng MLP 8D với hàm phần thưởng bước <code>r_t = −ΔJ_t</code> (γ = 1,0). Số liệu độc lập trên 16 workload test.
                </div>
              </div>
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}
