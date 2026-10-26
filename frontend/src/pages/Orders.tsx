import React, { useEffect, useState, useMemo } from "react";
import { CircleMarker, Marker, Popup } from "react-leaflet";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  PageHeader,
  LoadingCard,
  ErrorCard,
} from "@/components/shared/DomainComponents";
import { MapBase } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { api, type DeliveryQuery, type DeliveryRecord } from "@/lib/api";
import {
  DEPOT,
  SERVICE_LABEL,
  clusterColor,
  dateTime,
  num,
  pct,
} from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import {
  Package,
  Clock,
  Layers,
  Network,
  Search,
  ArrowUpDown,
  Filter,
  X,
  ExternalLink,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  MapPin,
  Calendar,
  Weight,
  Download,
  Share2,
} from "lucide-react";

const PAGE_SIZE = 25;

const DETAIL_FIELDS: [keyof DeliveryRecord, string][] = [
  ["mdh", "Mã vận đơn"],
  ["package_name", "Kiện hàng"],
  ["serviceType", "Gói dịch vụ"],
  ["orderStatus", "Trạng thái vận hành"],
  ["weight", "Khối lượng"],
  ["createdAt", "Thời điểm tạo đơn"],
  ["expectedDeliveryTime", "Hạn cam kết (SLA)"],
  ["deliveredAt", "Thời điểm giao thực tế"],
  ["leadTimeMinutes", "Thời gian hoàn tất"],
  ["isLate", "Trạng thái đúng hạn"],
  ["senderAddress", "Địa chỉ người gửi"],
  ["receiverAddress", "Địa chỉ người nhận"],
  ["shippingDistance", "Quãng đường ghi nhận"],
  ["haversineDistanceKm", "Khoảng cách chim bay"],
  ["cluster_id", "Phân cụm K-Means"],
];

function formatDetailValue(key: keyof DeliveryRecord, v: unknown): string {
  if (v === null || v === undefined || v === "") return "–";
  if (key === "serviceType") return SERVICE_LABEL[String(v)] ?? String(v);
  if (key === "createdAt" || key === "expectedDeliveryTime" || key === "deliveredAt")
    return dateTime(v as string);
  if (key === "isLate") return v ? "Trễ hạn" : "Đúng hạn";
  if (key === "weight") return `${num(Number(v), 2)} kg`;
  if (key === "shippingDistance") return `${num(Number(v), 0)} m`;
  if (key === "haversineDistanceKm") return `${num(Number(v), 2)} km`;
  if (key === "leadTimeMinutes") return `${num(Number(v), 1)} phút`;
  if (key === "cluster_id") return `Cụm #${v}`;
  if (typeof v === "number") return num(v, Number.isInteger(v) ? 0 : 2);
  return String(v);
}

export default function Orders() {
  const [page, setPage] = useState(0);
  const [sel, setSel] = useState<DeliveryRecord | null>(null);
  const [text, setText] = useState("");
  const [q, setQ] = useState("");
  const [service, setService] = useState("all");
  const [late, setLate] = useState("all");
  const [cluster, setCluster] = useState("all");
  const [sort, setSort] = useState("default");
  const [dir, setDir] = useState<"asc" | "desc">("asc");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setQ(text.trim()), 300);
    return () => clearTimeout(t);
  }, [text]);

  // Reset page on filter changes
  useEffect(() => {
    setPage(0);
  }, [q, service, late, cluster, sort, dir]);

  const query: DeliveryQuery = useMemo(
    () => ({
      q: q || undefined,
      service_type: service === "all" ? undefined : service,
      is_late: late === "all" ? undefined : late === "1",
      cluster_id: cluster === "all" ? undefined : Number(cluster),
      sort_by: sort === "default" ? undefined : sort,
      sort_dir: dir,
    }),
    [q, service, late, cluster, sort, dir]
  );

  const list = useAsync(
    (signal) => api.deliveries(page * PAGE_SIZE, PAGE_SIZE, query, signal),
    [page, q, service, late, cluster, sort, dir]
  );

  const stats = useAsync((signal) => api.deliveryStats(signal), []);
  const serviceKeys = Object.keys(stats.data?.service_type_counts ?? {});
  const isFiltered = Boolean(
    q || service !== "all" || late !== "all" || cluster !== "all" || sort !== "default"
  );

  const total = list.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const lat = sel ? Number(sel.receiverLat) : NaN;
  const lng = sel ? Number(sel.receiverLng) : NaN;
  const hasPoint = Number.isFinite(lat) && Number.isFinite(lng);

  const handleCopy = (val: string) => {
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(val);
      setCopiedId(val);
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  const handleClearFilters = () => {
    setText("");
    setService("all");
    setLate("all");
    setCluster("all");
    setSort("default");
    setDir("asc");
  };

  const handleExportData = () => {
    if (!list.data?.items) return;
    const blob = new Blob([JSON.stringify(list.data.items, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kltn11_orders_page_${page + 1}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3 max-w-[1720px] mx-auto p-3 sm:p-5 lg:p-6">
      {/* 1. Compact Header with inline statline */}
      <PageHeader
        title="Quản lý đơn hàng"
        subtitle="2.394 đơn hàng XeDu TP.HCM • 95,6% có hạn cam kết SLA (2.289 đơn) • 16 shipper phụ trách"
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportData}
              className="gap-1.5 text-xs h-8"
              disabled={!list.data?.items}
            >
              <Download className="size-3.5" />
              <span>Xuất JSON</span>
            </Button>
          </div>
        }
      />

      {/* 2. Table-First Main Card with Integrated Toolbar */}
      <Card className="border-border/80 shadow-xs overflow-hidden">
        {/* Integrated Compact Single-Row Toolbar */}
        <div className="p-3 bg-muted/20 border-b border-border/60 flex flex-wrap items-center gap-2">
          {/* Search */}
          <div className="relative min-w-[220px] flex-1">
            <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Tra cứu mã đơn, kiện hàng, địa chỉ..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="pl-8 h-8 text-xs bg-background"
            />
          </div>

          {/* Service filter */}
          <Select value={service} onValueChange={(val) => setService(val ?? "all")}>
            <SelectTrigger className="h-8 text-xs w-[130px] bg-background">
              <SelectValue placeholder="Dịch vụ" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Tất cả dịch vụ</SelectItem>
              {serviceKeys.map((k) => (
                <SelectItem key={k} value={k} className="text-xs">
                  {SERVICE_LABEL[k] ?? k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* SLA filter */}
          <Select value={late} onValueChange={(val) => setLate(val ?? "all")}>
            <SelectTrigger className="h-8 text-xs w-[125px] bg-background">
              <SelectValue placeholder="SLA" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Tất cả SLA</SelectItem>
              <SelectItem value="0" className="text-xs">Đúng hạn</SelectItem>
              <SelectItem value="1" className="text-xs">Trễ hạn</SelectItem>
            </SelectContent>
          </Select>

          {/* Cluster filter */}
          <Select value={cluster} onValueChange={(val) => setCluster(val ?? "all")}>
            <SelectTrigger className="h-8 text-xs w-[120px] bg-background">
              <SelectValue placeholder="Cụm" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Tất cả 10 cụm</SelectItem>
              {Array.from({ length: 10 }, (_, i) => (
                <SelectItem key={i} value={String(i)} className="text-xs">
                  Cụm #{i}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Sort */}
          <div className="flex items-center gap-1">
            <Select value={sort} onValueChange={(val) => setSort(val ?? "default")}>
              <SelectTrigger className="h-8 text-xs w-[125px] bg-background">
                <SelectValue placeholder="Sắp xếp" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Mặc định</SelectItem>
                <SelectItem value="createdAt" className="text-xs">Thời điểm tạo</SelectItem>
                <SelectItem value="expectedDeliveryTime" className="text-xs">Hạn SLA</SelectItem>
                <SelectItem value="weight" className="text-xs">Khối lượng</SelectItem>
                <SelectItem value="haversineDistanceKm" className="text-xs">Cự ly Depot</SelectItem>
                <SelectItem value="cluster_id" className="text-xs">Mã cụm</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              size="icon"
              className="size-8 shrink-0 bg-background"
              disabled={sort === "default"}
              onClick={() => setDir((d) => (d === "asc" ? "desc" : "asc"))}
              title={dir === "asc" ? "Tăng dần" : "Giảm dần"}
            >
              <ArrowUpDown className="size-3 text-muted-foreground" />
            </Button>
          </div>

          {/* Reset Filters */}
          {isFiltered && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearFilters}
              className="h-8 text-xs text-muted-foreground hover:text-foreground gap-1 px-2"
            >
              <X className="size-3" />
              <span>Xóa lọc</span>
            </Button>
          )}

          {/* Pagination controls in toolbar */}
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-xs font-mono text-muted-foreground">
              {page + 1}/{pages} ({num(total, 0)} đơn)
            </span>
            <Button
              variant="outline"
              size="icon"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              className="size-8 text-xs"
            >
              <ChevronLeft className="size-3.5" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              disabled={page >= pages - 1}
              onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
              className="size-8 text-xs"
            >
              <ChevronRight className="size-3.5" />
            </Button>
          </div>
        </div>

        <CardContent className="p-0 overflow-x-auto">
          {list.loading ? (
            <div className="p-8">
              <LoadingCard message="Đang tải danh sách đơn hàng..." />
            </div>
          ) : list.error ? (
            <div className="p-8">
              <ErrorCard message={list.error} />
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent bg-muted/40">
                  <TableHead className="w-[110px] text-xs font-semibold">Mã đơn</TableHead>
                  <TableHead className="text-xs font-semibold">Kiện hàng</TableHead>
                  <TableHead className="text-xs font-semibold">Gói dịch vụ</TableHead>
                  <TableHead className="text-xs font-semibold">Cụm K-Means</TableHead>
                  <TableHead className="text-xs font-semibold">Hạn cam kết SLA</TableHead>
                  <TableHead className="text-right text-xs font-semibold">Khối lượng</TableHead>
                  <TableHead className="text-right text-xs font-semibold">Cự ly Depot</TableHead>
                  <TableHead className="w-[90px] text-center text-xs font-semibold">Trạng thái</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {list.data?.items.map((row) => {
                  const isSelected = sel?.id === row.id;
                  const rowClusterId =
                    typeof row.cluster_id === "number" ? row.cluster_id : Number(row.cluster_id ?? 0);
                  const clr = clusterColor(rowClusterId);
                  const rowMdh = row.mdh ? String(row.mdh) : "";
                  const rowService = row.serviceType ? String(row.serviceType) : "";

                  return (
                    <TableRow
                      key={row.id}
                      onClick={() => setSel(isSelected ? null : row)}
                      className={`cursor-pointer transition-colors text-xs ${
                        isSelected
                          ? "bg-primary/10 border-l-2 border-l-primary font-medium"
                          : "hover:bg-muted/30"
                      }`}
                    >
                      <TableCell className="font-mono font-medium text-foreground">
                        <div className="flex items-center gap-1.5">
                          <span className="truncate max-w-[90px]">{rowMdh || `#${row.id}`}</span>
                          {rowMdh && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCopy(rowMdh);
                              }}
                              className="text-muted-foreground hover:text-foreground opacity-60 hover:opacity-100"
                              title="Sao chép mã đơn"
                            >
                              {copiedId === rowMdh ? (
                                <Check className="size-3 text-emerald-600" />
                              ) : (
                                <Copy className="size-3" />
                              )}
                            </button>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="max-w-[200px] truncate text-muted-foreground">
                        {row.package_name ? String(row.package_name) : "–"}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-[10px] font-normal">
                          {SERVICE_LABEL[rowService] ?? rowService}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <span
                          className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-semibold"
                          style={{
                            backgroundColor: `${clr}18`,
                            color: clr,
                            border: `1px solid ${clr}40`,
                          }}
                        >
                          <span
                            className="size-1.5 rounded-full"
                            style={{ backgroundColor: clr }}
                          />
                          Cụm #{rowClusterId}
                        </span>
                      </TableCell>
                      <TableCell className="font-mono text-[11px] text-muted-foreground">
                        {row.expectedDeliveryTime ? dateTime(String(row.expectedDeliveryTime)) : "Linh hoạt"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-[11px]">
                        {row.weight ? `${num(Number(row.weight), 2)} kg` : "–"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-[11px] text-foreground font-semibold">
                        {row.haversineDistanceKm ? `${num(Number(row.haversineDistanceKm), 2)} km` : "–"}
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant={row.isLate ? "destructive" : "secondary"}
                          className="text-[10px] font-mono px-1.5 py-0"
                        >
                          {row.isLate ? "Trễ hạn" : "Đúng hạn"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>

        <CardFooter className="p-3 sm:px-5 border-t border-border/60 flex items-center justify-between text-xs text-muted-foreground">
          <span>Tổng cộng {num(total, 0)} đơn XeDu nội đô</span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              className="h-7 text-xs px-2"
            >
              <ChevronLeft className="size-3.5 mr-1" />
              <span>Trước</span>
            </Button>
            <span className="text-xs font-mono font-medium px-1">
              Trang {page + 1} / {pages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= pages - 1}
              onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
              className="h-7 text-xs px-2"
            >
              <span>Sau</span>
              <ChevronRight className="size-3.5 ml-1" />
            </Button>
          </div>
        </CardFooter>
      </Card>

      {/* 5. Slide-Over Detail Drawer when an Order is Selected */}
      {sel && (
        <div className="fixed inset-y-0 right-0 z-50 w-full max-w-[460px] bg-card border-l border-border/80 shadow-2xl flex flex-col animate-in slide-in-from-right duration-200">
          <div className="p-4 border-b border-border/60 flex items-center justify-between bg-muted/20">
            <div className="flex items-center gap-2">
              <Package className="size-4 text-primary" />
              <span className="text-sm font-bold tracking-tight text-foreground">
                Hồ sơ Đơn hàng {sel.mdh ? String(sel.mdh) : `#${sel.id}`}
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSel(null)}
              className="size-7 rounded-full text-muted-foreground hover:text-foreground"
            >
              <X className="size-4" />
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Mini Map */}
            <div className="h-[200px] w-full rounded-lg overflow-hidden border border-border/60 relative">
              {hasPoint ? (
                <MapErrorBoundary>
                  <MapBase fit={[[lat - 0.015, lng - 0.015], [lat + 0.015, lng + 0.015]]} height="100%">
                    <CircleMarker
                      center={[lat, lng]}
                      radius={7}
                      pathOptions={{
                        color: "#ffffff",
                        fillColor: clusterColor(typeof sel.cluster_id === "number" ? sel.cluster_id : 0),
                        fillOpacity: 1,
                        weight: 2,
                      }}
                    >
                      <Popup>
                        <div className="text-xs">
                          <p className="font-bold">{String(sel.package_name ?? "")}</p>
                          <p>{String(sel.receiverAddress ?? "")}</p>
                        </div>
                      </Popup>
                    </CircleMarker>
                    <CircleMarker
                      center={[DEPOT.lat, DEPOT.lng]}
                      radius={5}
                      pathOptions={{
                        color: "#ffffff",
                        fillColor: "#171717",
                        fillOpacity: 1,
                        weight: 2,
                      }}
                    />
                  </MapBase>
                </MapErrorBoundary>
              ) : (
                <div className="h-full flex items-center justify-center text-xs text-muted-foreground bg-muted/30">
                  Không có dữ liệu toạ độ GPS
                </div>
              )}
            </div>

            {/* Core Specifications Grid */}
            <div className="space-y-2">
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                Thông tin chi tiết kiện hàng
              </h4>
              <div className="divide-y divide-border/40 rounded-lg border border-border/60 bg-muted/10 text-xs">
                {DETAIL_FIELDS.map(([key, label]) => {
                  const val = sel[key];
                  return (
                    <div key={key} className="flex items-center justify-between p-2.5">
                      <span className="text-muted-foreground text-[11px]">{label}</span>
                      <span className="font-mono font-medium text-foreground text-right max-w-[240px] truncate">
                        {formatDetailValue(key, val)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="p-3 border-t border-border/60 bg-muted/20 flex items-center justify-between">
            <span className="text-[11px] text-muted-foreground font-mono">
              GPS: {String(sel.receiverLat ?? "").slice(0, 7)}, {String(sel.receiverLng ?? "").slice(0, 7)}
            </span>
            <Button
              size="sm"
              onClick={() => setSel(null)}
              className="text-xs h-7 bg-primary text-primary-foreground"
            >
              Đóng hồ sơ
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
