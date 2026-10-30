import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
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
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  MetricCard,
  ResearchCallout,
} from "@/components/shared/DomainComponents";
import {
  api,
  AllocationPolicy,
  DailyPlanningRow,
  DailyPlanningSensitivityRow,
  OvertimePolicy,
} from "../lib/api";
import { num, pct, vnd } from "../lib/format";
import { useAsync } from "../lib/useAsync";

const POLICIES: OvertimePolicy[] = ["overtime", "defer", "hybrid"];

const POLICY_LABEL: Record<OvertimePolicy, string> = {
  overtime: "Luôn làm thêm giờ",
  defer: "Luôn dời đơn",
  hybrid: "Chính sách lai",
};

const POLICY_COLOR: Record<OvertimePolicy, string> = {
  overtime: "#f59e0b",
  defer: "#ef4444",
  hybrid: "#10b981",
};

const ALLOCATION_LABEL: Record<AllocationPolicy, string> = {
  dedicated: "Mỗi cụm một shipper",
  merge_small: "Gộp cụm nhỏ",
};

const PARAMETER_LABEL: Record<string, string> = {
  defer_penalty_per_order_day: "Mức phạt dời đơn (₫/đơn/ngày)",
  overtime_multiplier: "Hệ số lương làm thêm giờ",
  max_overtime_minutes: "Trần làm thêm giờ (phút/ngày)",
  n_shippers: "Số shipper",
  small_cluster_threshold: "Ngưỡng cụm nhỏ (đơn)",
};

interface SensitivityBest {
  key: string;
  parameter: string;
  value: number;
  best: DailyPlanningSensitivityRow | null;
}

export default function DailyPlanning() {
  const summary = useAsync((s) => api.dailyPlanningSummary(s), []);
  const sensitivity = useAsync((s) => api.dailyPlanningSensitivity(s), []);

  const [load, setLoad] = useState(4);
  const [allocation, setAllocation] = useState<AllocationPolicy>("dedicated");

  const daily = useAsync(
    (s) =>
      Promise.all(
        POLICIES.map((p) => api.dailyPlanningDaily(load, allocation, p, s)),
      ),
    [load, allocation],
  );

  const loads = useMemo(() => {
    const values = (summary.data?.rows ?? []).map((r) => r.load_multiplier);
    return Array.from(new Set(values)).sort((a, b) => a - b);
  }, [summary.data]);

  const rows = useMemo(() => {
    const all = summary.data?.rows ?? [];
    return POLICIES.map((p) =>
      all.find(
        (r) =>
          r.load_multiplier === load &&
          r.allocation === allocation &&
          r.overtime_policy === p,
      ),
    ).filter((r): r is DailyPlanningRow => r !== undefined);
  }, [summary.data, load, allocation]);

  const best = (summary.data?.best_within_cap ?? []).find(
    (r) => r.load_multiplier === load,
  );

  const overCapRows = rows.filter((r) => r.over_cap_shipper_days > 0);

  const backlogSeries = useMemo(() => {
    const series = daily.data;
    if (!series) return [];
    return series[0].days.map((d, i) => ({
      day: d.day + 1,
      overtime: series[0].days[i].deferred_out,
      defer: series[1].days[i].deferred_out,
      hybrid: series[2].days[i].deferred_out,
    }));
  }, [daily.data]);

  const sensitivityBest = useMemo<SensitivityBest[]>(() => {
    const groups = new Map<string, DailyPlanningSensitivityRow[]>();
    for (const r of sensitivity.data?.rows ?? []) {
      const key = `${r.parameter}|${r.value}`;
      const group = groups.get(key);
      if (group) group.push(r);
      else groups.set(key, [r]);
    }
    return Array.from(groups.entries()).map(([key, group]) => {
      const legal = group.filter((r) => r.over_cap_shipper_days === 0);
      const winner = legal.reduce<DailyPlanningSensitivityRow | null>(
        (a, b) => (a === null || b.total_cost < a.total_cost ? b : a),
        null,
      );
      return {
        key,
        parameter: group[0].parameter,
        value: group[0].value,
        best: winner,
      };
    });
  }, [sensitivity.data]);

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Kế hoạch giao hàng theo ngày"
        subtitle="Khi shipper hết ca 10 giờ mà còn đơn: làm thêm giờ hay dời sang hôm sau? Cụm ít đơn: giữ shipper riêng hay gộp?"
        badge="Mô phỏng 30 ngày"
      />

      {summary.loading && <LoadingCard message="Đang nạp kết quả mô phỏng…" />}
      {summary.error && <ErrorCard message={summary.error} />}

      {summary.data && (
        <>
          <ResearchCallout title="Phạm vi của mô phỏng" type="warning">
            {summary.data.note}
          </ResearchCallout>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-muted-foreground mr-1">
                Mức tải
              </span>
              {loads.map((l) => (
                <Button
                  key={l}
                  size="sm"
                  variant={l === load ? "default" : "outline"}
                  className="h-7 px-2.5 text-xs font-mono"
                  onClick={() => setLoad(l)}
                >
                  x{l}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-muted-foreground mr-1">
                Phân bổ cụm
              </span>
              {(Object.keys(ALLOCATION_LABEL) as AllocationPolicy[]).map((a) => (
                <Button
                  key={a}
                  size="sm"
                  variant={a === allocation ? "default" : "outline"}
                  className="h-7 px-2.5 text-xs"
                  onClick={() => setAllocation(a)}
                >
                  {ALLOCATION_LABEL[a]}
                </Button>
              ))}
            </div>
          </div>

          {best && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Tốt nhất trong trần làm thêm giờ"
                value={POLICY_LABEL[best.overtime_policy]}
                hint={ALLOCATION_LABEL[best.allocation]}
                badge={`x${best.load_multiplier}`}
              />
              <MetricCard
                label="Chi phí mỗi đơn"
                value={vnd(best.cost_per_order)}
              />
              <MetricCard
                label="Đơn tồn cuối kỳ"
                value={num(best.final_backlog, 0)}
                hint={`Trên tổng ${num(best.orders, 0)} đơn`}
              />
              <MetricCard
                label="Giờ làm thêm"
                value={`${num(best.overtime_hours, 1)} giờ`}
                hint={`${num(best.mean_shippers_used, 2)} shipper/ngày`}
              />
            </div>
          )}

          {overCapRows.length > 0 && (
            <ResearchCallout title="Chính sách vi phạm trần làm thêm giờ" type="warning">
              {overCapRows
                .map(
                  (r) =>
                    `${POLICY_LABEL[r.overtime_policy]}: ${num(r.over_cap_shipper_days, 0)} ca vượt trần`,
                )
                .join(" · ")}
              . Chính sách này có thể rẻ nhất nhưng không dùng được nếu coi 2 giờ
              mỗi ngày là mức tối đa, nên không được chọn là &quot;tốt nhất&quot;.
            </ResearchCallout>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                So sánh ba chính sách ở mức tải x{load}
              </CardTitle>
              <CardDescription className="text-xs">
                {ALLOCATION_LABEL[allocation]} · chi phí tính bằng giả định của
                mô phỏng, không phải chi phí thật của XeDu.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Chính sách</TableHead>
                      <TableHead className="text-right">Chi phí/đơn</TableHead>
                      <TableHead className="text-right">Đúng hạn</TableHead>
                      <TableHead className="text-right">Giờ làm thêm</TableHead>
                      <TableHead className="text-right">Ca vượt trần</TableHead>
                      <TableHead className="text-right">Đơn tồn cuối</TableHead>
                      <TableHead className="text-right">Shipper TB</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.map((r) => {
                      const isBest =
                        best !== undefined &&
                        best.allocation === r.allocation &&
                        best.overtime_policy === r.overtime_policy;
                      return (
                        <TableRow key={r.overtime_policy}>
                          <TableCell className="font-medium">
                            {POLICY_LABEL[r.overtime_policy]}
                            {isBest && (
                              <Badge
                                variant="outline"
                                className="ml-2 text-[10px] px-1.5 py-0 text-emerald-600 border-emerald-500/40"
                              >
                                tốt nhất
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {vnd(r.cost_per_order)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {pct(r.on_time_rate, 1)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {num(r.overtime_hours, 1)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {num(r.over_cap_shipper_days, 0)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {num(r.final_backlog, 0)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">
                            {num(r.mean_shippers_used, 2)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              <div style={{ height: 240 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={rows}
                    margin={{ top: 8, right: 8, bottom: 6, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="overtime_policy"
                      tickFormatter={(k: OvertimePolicy) => POLICY_LABEL[k] ?? k}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      width={64}
                      tickFormatter={(v: number) => num(v, 0)}
                    />
                    <Tooltip
                      formatter={(v: any) => [vnd(v), "Chi phí mỗi đơn"]}
                      labelFormatter={(k: any) => POLICY_LABEL[k as OvertimePolicy] ?? k}
                    />
                    <Bar dataKey="cost_per_order" radius={[4, 4, 0, 0]}>
                      {rows.map((r) => (
                        <Cell
                          key={r.overtime_policy}
                          fill={POLICY_COLOR[r.overtime_policy]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            Đơn tồn theo từng ngày (x{load}, {ALLOCATION_LABEL[allocation]})
          </CardTitle>
          <CardDescription className="text-xs">
            Số đơn chưa giao được cuối mỗi ngày. Chính sách dời đơn cộng dồn dần
            khi tải vượt năng lực.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {daily.loading && <LoadingCard message="Đang nạp số liệu từng ngày…" />}
          {daily.error && <ErrorCard message={daily.error} />}
          {daily.data && (
            <div style={{ height: 260 }}>
              <ResponsiveContainer>
                <LineChart
                  data={backlogSeries}
                  margin={{ top: 8, right: 16, bottom: 6, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(d: number) => `N${d}`}
                  />
                  <YAxis tick={{ fontSize: 12 }} width={48} />
                  <Tooltip
                    labelFormatter={(d: any) => `Ngày ${d}`}
                    formatter={(v: any, name: any) => [
                      `${num(v, 0)} đơn`,
                      POLICY_LABEL[name as OvertimePolicy] ?? name,
                    ]}
                  />
                  <Legend
                    formatter={(k: any) => POLICY_LABEL[k as OvertimePolicy] ?? k}
                    wrapperStyle={{ fontSize: 12 }}
                  />
                  {POLICIES.map((p) => (
                    <Line
                      key={p}
                      type="monotone"
                      dataKey={p}
                      stroke={POLICY_COLOR[p]}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            Độ nhạy theo tham số (mức tải x12)
          </CardTitle>
          <CardDescription className="text-xs">
            Với mỗi giá trị tham số: chính sách rẻ nhất trong số những chính sách
            không vượt trần làm thêm giờ.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sensitivity.loading && <LoadingCard message="Đang nạp phân tích độ nhạy…" />}
          {sensitivity.error && <ErrorCard message={sensitivity.error} />}
          {sensitivity.data && (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Tham số</TableHead>
                    <TableHead className="text-right">Giá trị</TableHead>
                    <TableHead>Chính sách tốt nhất</TableHead>
                    <TableHead className="text-right">Chi phí/đơn</TableHead>
                    <TableHead className="text-right">Đơn tồn cuối</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sensitivityBest.map((s) => (
                    <TableRow key={s.key}>
                      <TableCell>
                        {PARAMETER_LABEL[s.parameter] ?? s.parameter}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {num(s.value)}
                      </TableCell>
                      <TableCell>
                        {s.best
                          ? `${POLICY_LABEL[s.best.overtime_policy]} · ${ALLOCATION_LABEL[s.best.allocation]}`
                          : "–"}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {s.best ? vnd(s.best.cost_per_order) : "–"}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {s.best ? num(s.best.final_backlog, 0) : "–"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
