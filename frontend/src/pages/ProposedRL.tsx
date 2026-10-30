import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { api, type ShiftRlMethod } from "../lib/api";
import { num, pct, signedPct } from "../lib/format";
import { useAsync } from "../lib/useAsync";

const PROPOSED = "Proposed RL";

export default function ProposedRL() {
  const test = useAsync((s) => api.proposedRlFinalTest(s), []);
  const shift = useAsync((s) => api.shiftRl(s), []);

  // Lower objective J is better.
  const ranked = useMemo(
    () =>
      [...(test.data?.summary ?? [])].sort(
        (a, b) => a.mean_locked_objective - b.mean_locked_objective,
      ),
    [test.data],
  );

  const proposed = ranked.find((r) => r.algorithm === PROPOSED);
  const rank = proposed ? ranked.indexOf(proposed) + 1 : null;

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Proposed RL: kết quả kiểm tra cuối"
        subtitle="Mô hình và hàm mục tiêu J được khóa bằng train và validation trước khi chạm vào tập test."
        badge="16 workload · 131 đơn"
      />

      {test.loading && <LoadingCard message="Đang nạp kết quả kiểm tra cuối…" />}
      {test.error && <ErrorCard message={test.error} />}

      {test.data && proposed && (
        <>
          <ResearchCallout title="Kết luận" type="warning">
            Proposed RL <b>không thắng</b> các thuật toán đối chứng trên tập test
            (xếp hạng {rank}/{ranked.length} theo J). Mỗi workload chỉ có 5 đến 16
            điểm nên các heuristic đã gần tối ưu; đây là nhận định từ kết quả đo.
            {" "}
            {test.data.scope_note}
          </ResearchCallout>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Hàm mục tiêu J"
              value={num(proposed.mean_locked_objective, 4)}
              hint="Càng thấp càng tốt"
            />
            <MetricCard
              label="Quãng đường TB"
              value={`${num(proposed.mean_distance_km, 1)} km`}
            />
            <MetricCard
              label="SLA có trọng số"
              value={pct(proposed.weighted_sla, 2)}
              hint={`${num(proposed.late_deliveries, 0)} đơn trễ`}
            />
            <MetricCard
              label="Đổi cụm TB"
              value={num(proposed.mean_cluster_switches, 2)}
              hint="Số lần đổi cụm mỗi workload"
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                Hàm mục tiêu J trên tập test (càng thấp càng tốt)
              </CardTitle>
              <CardDescription className="text-xs">
                Mọi thuật toán được đo bằng cùng một hàm mục tiêu đã khóa.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div style={{ height: 240 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={ranked}
                    margin={{ top: 8, right: 8, bottom: 6, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="algorithm" tick={{ fontSize: 11 }} />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      width={52}
                      domain={[0, "auto"]}
                      tickFormatter={(v: number) => num(v, 2)}
                    />
                    <Tooltip
                      formatter={(v: any) => [num(v, 4), "J"]}
                    />
                    <Bar dataKey="mean_locked_objective" radius={[4, 4, 0, 0]}>
                      {ranked.map((r) => (
                        <Cell
                          key={r.algorithm}
                          fill={r.algorithm === PROPOSED ? "#ef4444" : "#64748b"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Thuật toán</TableHead>
                      <TableHead className="text-right">J</TableHead>
                      <TableHead className="text-right">Quãng đường (km)</TableHead>
                      <TableHead className="text-right">Thời gian (phút)</TableHead>
                      <TableHead className="text-right">Đơn trễ</TableHead>
                      <TableHead className="text-right">SLA</TableHead>
                      <TableHead className="text-right">Đổi cụm TB</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ranked.map((r) => (
                      <TableRow key={r.algorithm}>
                        <TableCell className="font-medium">
                          {r.algorithm}
                          {r.algorithm === PROPOSED && (
                            <Badge
                              variant="outline"
                              className="ml-2 text-[10px] px-1.5 py-0"
                            >
                              đề xuất
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.mean_locked_objective, 4)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.mean_distance_km, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.mean_travel_time_minutes, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.late_deliveries, 0)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {pct(r.weighted_sla, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.mean_cluster_switches, 2)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Proposed RL so với từng thuật toán</CardTitle>
              <CardDescription className="text-xs">
                Mức cải thiện J âm nghĩa là Proposed RL kém hơn.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>So với</TableHead>
                      <TableHead className="text-right">Cải thiện J</TableHead>
                      <TableHead className="text-right">Giảm đổi cụm</TableHead>
                      <TableHead className="text-right">Thay đổi SLA (điểm %)</TableHead>
                      <TableHead className="text-right">Chênh đơn trễ</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {test.data.pairwise_vs_baselines.map((p) => (
                      <TableRow key={p.baseline_algorithm}>
                        <TableCell className="font-medium">
                          {p.baseline_algorithm}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {signedPct(p.locked_objective_improvement_pct, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {signedPct(p.cluster_switch_reduction_pct, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(p.sla_change_percentage_points, 2)}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(p.late_delivery_change, 0)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {shift.data && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Agent RL theo ca làm (thưởng phạt mới), kiểm tra cuối ({shift.data.test_set})
            </CardTitle>
            <CardDescription className="text-xs">
              {shift.data.scope_note} Tập kiểm tra này chưa được dùng khi phát triển
              mô hình; mô hình và seed được chọn bằng validation. Khoảng cách tới tham chiếu: càng thấp càng
              tốt. Tham chiếu là nghiệm tối ưu chính xác khi ca có tối đa 14 đơn,
              còn lại là nghiệm tốt nhất tìm được bằng local search nhiều lần.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Phương pháp</TableHead>
                    {shift.data.groups.map((g) => (
                      <TableHead key={g.group} className="text-right">
                        {g.group} ({num(g.shifts, 0)} ca)
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {shift.data.groups[0].methods.map((m, row) => (
                    <TableRow key={m.key}>
                      <TableCell className="font-medium">
                        {m.label}
                        {m.kind === "rl" && (
                          <Badge variant="outline" className="ml-2 text-[10px] px-1.5 py-0">
                            RL
                          </Badge>
                        )}
                      </TableCell>
                      {shift.data!.groups.map((g) => {
                        const cell: ShiftRlMethod = g.methods[row];
                        return (
                          <TableCell
                            key={g.group}
                            className="text-right font-mono tabular-nums"
                          >
                            {num(cell.mean_gap_pct, 1)}%
                            {cell.std_gap_pct !== null &&
                              ` ± ${num(cell.std_gap_pct, 1)}`}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Mô hình RL chính (chọn theo validation): {shift.data.headline.label}, seed{" "}
              {shift.data.headline.seed}. Suy luận mất{" "}
              {num(shift.data.headline.inference_ms, 1)} ms mỗi ca, so với{" "}
              {num(shift.data.local_search_seconds_per_shift, 2)} giây của NN + local
              search. So với NN + local search trên cùng các ca: thắng{" "}
              {num(shift.data.paired_vs.nn_all_ls.wins, 0)}, hòa{" "}
              {num(shift.data.paired_vs.nn_all_ls.ties, 0)}, thua{" "}
              {num(shift.data.paired_vs.nn_all_ls.losses, 0)}. Mọi hệ số tiền là giả
              định; dữ liệu XeDu không có giá đơn hàng.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
