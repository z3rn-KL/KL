import { useState } from "react";
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
import { api, type RealReplayMethod } from "../lib/api";
import { num, pct } from "../lib/format";
import { useAsync } from "../lib/useAsync";

const METHOD_LABEL: Record<RealReplayMethod["method"], string> = {
  real: "Thứ tự giao thật của shipper",
  nearest: "Nearest Neighbor",
  nearest_search: "NN + local search",
  optimal: "Tối ưu chính xác",
};

const METHOD_COLOR: Record<RealReplayMethod["method"], string> = {
  real: "#ef4444",
  nearest: "#f59e0b",
  nearest_search: "#3b82f6",
  optimal: "#10b981",
};

const METRIC = {
  time: { label: "Thời gian đường phố", unit: "phút" },
  km: { label: "Quãng đường", unit: "km" },
} as const;

export default function RealData() {
  const data = useAsync((s) => api.realData(s), []);
  const [metric, setMetric] = useState<keyof typeof METRIC>("time");

  const d = data.data;
  const rows = d?.replay[metric] ?? [];

  return (
    <div className="space-y-4 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      <PageHeader
        title="Dữ liệu thật"
        subtitle="Kết quả quan sát từ dữ liệu giao hàng XeDu và phát lại trên đồ thị đường phố thật."
        badge="Không mô phỏng đơn"
      />

      {data.loading && <LoadingCard message="Đang nạp kết quả dữ liệu thật…" />}
      {data.error && <ErrorCard message={data.error} />}

      {d && (
        <>
          <ResearchCallout title="Cái gì là thật, cái gì là tính toán" type="warning">
            {d.scope_note} Hệ số tiền (phí, hoa hồng, lương) không có trong dữ liệu nên
            không xuất hiện ở trang này.
          </ResearchCallout>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Giao trễ thực tế"
              value={pct(d.observed.late_rate, 1)}
              hint={`${num(d.observed.orders_with_deadline, 0)} đơn có hạn giao (quan sát)`}
              badge="Quan sát"
            />
            <MetricCard
              label="Thực tế giữa hai lần giao"
              value={`${num(d.time_model.real_median_minutes, 1)} phút`}
              hint={`Mô hình đường phố chỉ cho ${num(d.time_model.model_median_minutes, 1)} phút`}
              badge="Trung vị"
            />
            <MetricCard
              label="Phần mô hình không có"
              value={`${num(d.time_model.median_overhead_minutes, 1)} phút`}
              hint="Phục vụ, lấy hàng, chờ, kẹt xe (mỗi điểm giao)"
              badge="Trung vị"
            />
            <MetricCard
              label="Ca phát lại"
              value={num(d.clean_shifts.count, 0)}
              hint={`${num(d.clean_shifts.min_orders, 0)} đến ${num(d.clean_shifts.max_orders, 0)} đơn mỗi ca, một điểm lấy hàng`}
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Mô hình đường phố so với thực tế</CardTitle>
              <CardDescription className="text-xs">
                {num(d.time_model.pairs, 0)} cặp giao liên tiếp của cùng một shipper (cách nhau không quá{" "}
                {num(d.time_model.max_gap_minutes, 0)} phút).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Chỉ số</TableHead>
                      <TableHead className="text-right">Giá trị</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[
                      ["Thời gian thật giữa hai lần giao (trung vị / trung bình)", `${num(d.time_model.real_median_minutes, 1)} / ${num(d.time_model.real_mean_minutes, 1)} phút`],
                      ["Thời gian di chuyển theo mô hình (trung vị / trung bình)", `${num(d.time_model.model_median_minutes, 1)} / ${num(d.time_model.model_mean_minutes, 1)} phút`],
                      ["Tương quan hạng Spearman", num(d.time_model.spearman, 2)],
                      ["Cặp mà thực tế nhanh hơn mô hình", pct(d.time_model.real_shorter_than_model_share, 1)],
                    ].map(([label, value]) => (
                      <TableRow key={label}>
                        <TableCell>{label}</TableCell>
                        <TableCell className="text-right font-mono tabular-nums">{value}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Mô hình là một cận dưới nhất quán nhưng thấp hơn thực tế nhiều lần. Vì vậy tỷ lệ
                đúng hạn tính theo mô hình ở trang So sánh cao hơn nhiều so với thực tế.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle className="text-sm">
                    Phát lại {num(d.clean_shifts.count, 0)} ca giao thật
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Mỗi ca: một shipper, một ngày, một điểm lấy hàng (dùng làm kho), mọi đơn tạo trước khi
                    bắt đầu giao. Tuyến đi từ điểm lấy hàng đến đơn cuối, không quay về.
                  </CardDescription>
                </div>
                <div className="flex gap-1.5">
                  {(Object.keys(METRIC) as (keyof typeof METRIC)[]).map((key) => (
                    <Button
                      key={key}
                      size="sm"
                      variant={key === metric ? "default" : "outline"}
                      className="h-7 px-2.5 text-xs"
                      onClick={() => setMetric(key)}
                    >
                      {METRIC[key].label}
                    </Button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div style={{ height: 240 }}>
                <ResponsiveContainer>
                  <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 6, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="method"
                      tickFormatter={(k: RealReplayMethod["method"]) => METHOD_LABEL[k] ?? k}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis
                      tick={{ fontSize: 12 }}
                      width={52}
                      tickFormatter={(v: number) => `${num(v, 0)}%`}
                    />
                    <Tooltip
                      formatter={(v: any) => [`${num(v, 2)}%`, "Dài hơn tối ưu (trung bình)"]}
                      labelFormatter={(k: any) => METHOD_LABEL[k as RealReplayMethod["method"]] ?? k}
                    />
                    <Bar dataKey="mean_excess_pct" radius={[4, 4, 0, 0]}>
                      {rows.map((r) => (
                        <Cell key={r.method} fill={METHOD_COLOR[r.method]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Phương pháp</TableHead>
                      <TableHead className="text-right">Trung bình ({METRIC[metric].unit})</TableHead>
                      <TableHead className="text-right">Dài hơn tối ưu</TableHead>
                      <TableHead className="text-right">Khoảng tin cậy 95%</TableHead>
                      <TableHead className="text-right">Đúng tối ưu</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.map((r) => (
                      <TableRow key={r.method}>
                        <TableCell className="font-medium">{METHOD_LABEL[r.method]}</TableCell>
                        <TableCell className="text-right font-mono tabular-nums">{num(r.mean, 2)}</TableCell>
                        <TableCell className="text-right font-mono tabular-nums">{num(r.mean_excess_pct, 2)}%</TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.ci_low_pct, 2)}% đến {num(r.ci_high_pct, 2)}%
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {num(r.at_optimum, 0)}/{num(d.clean_shifts.count, 0)}
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
              <CardTitle className="text-sm">Đọc kết quả này thế nào</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs leading-relaxed text-muted-foreground">
              <p>
                Thứ tự giao thật theo hạn giao sớm nhất ở {num(d.deadline_order.earliest_deadline_first_shifts, 0)}/
                {num(d.deadline_order.shifts, 0)} ca (tương quan hạng {num(d.deadline_order.mean_spearman_position_vs_deadline, 2)}),
                nên phần dài hơn tối ưu không giải thích được chỉ bằng việc ưu tiên đơn gấp.
              </p>
              <p>
                Mẫu nhỏ và đã chọn lọc: ca chỉ có {num(d.clean_shifts.min_orders, 0)} đến {num(d.clean_shifts.max_orders, 0)} đơn
                (trung vị {num(d.clean_shifts.median_orders, 0)}). Nghiệm tối ưu là cận dưới cho bài toán tĩnh; shipper
                còn chịu kẹt xe, thời gian phục vụ và khách vắng nhà mà dữ liệu không có, nên khoảng cách tới tối ưu
                không phải mức tiết kiệm có thể hứa.
              </p>
              <p>
                Với quy mô ca thật như vậy, Nearest Neighbor kết hợp local search gần như đạt tối ưu, nên thuật toán học
                tăng cường không có chỗ để cải thiện thêm trên các ca tĩnh này.
              </p>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
