import React, { useEffect, useMemo, useState } from "react";
import { CircleMarker, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MapBase } from "@/components/MapBase";
import { MapErrorBoundary } from "@/components/ErrorBoundary";
import { api, type DeliveryRecord, type KmeansPoint } from "@/lib/api";
import { DEPOT, clusterColor, num } from "@/lib/format";
import { useAsync } from "@/lib/useAsync";
import {
  Network,
  Search,
  Sparkles,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const PAGE_SIZE = 15;

function MapFlyTo({ target }: { target: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (map && typeof map.getZoom === "function" && target && Number.isFinite(target[0]) && Number.isFinite(target[1])) {
      try {
        map.flyTo(L.latLng(target[0], target[1]), Math.max(map.getZoom(), 13), {
          duration: 0.8,
        });
      } catch {
        // Map instance unmounting or transitioning
      }
    }
  }, [target, map]);
  return null;
}

export default function DataClustering() {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [page, setPage] = useState(0);
  const [flyTarget, setFlyTarget] = useState<[number, number] | null>(null);
  const [selectedOrder, setSelectedOrder] = useState<DeliveryRecord | null>(null);

  // Fetch K-Means precomputed data
  const km = useAsync((s) => api.kmeansAll(s), []);
  // Fetch deliveries with filter
  const deliveriesQuery = useAsync(
    (s) =>
      api.deliveries(
        page * PAGE_SIZE,
        PAGE_SIZE,
        {
          cluster_id: selectedCluster !== null ? selectedCluster : undefined,
          q: searchTerm || undefined,
        },
        s
      ),
    [page, selectedCluster, searchTerm]
  );

  // Cluster points
  const points: KmeansPoint[] = km.data?.points ?? [];
  const nClusters = km.data?.meta?.n_clusters ?? 10;
  const silhouette = 0.429;
  const totalDeliveries = km.data?.meta?.total_deliveries ?? 2394;

  // Compute centroids dynamically from points
  const centroids = useMemo(() => {
    const sums: Record<number, { lat: number; lng: number; count: number }> = {};
    for (let i = 0; i < nClusters; i++) sums[i] = { lat: 0, lng: 0, count: 0 };
    points.forEach((p) => {
      if (sums[p.cluster_id]) {
        sums[p.cluster_id].lat += p.latitude;
        sums[p.cluster_id].lng += p.longitude;
        sums[p.cluster_id].count += 1;
      }
    });
    const res: Record<number, { lat: number; lng: number }> = {};
    for (let i = 0; i < nClusters; i++) {
      if (sums[i] && sums[i].count > 0) {
        res[i] = { lat: sums[i].lat / sums[i].count, lng: sums[i].lng / sums[i].count };
      }
    }
    return res;
  }, [points, nClusters]);

  // Filtered points on map
  const visiblePoints = useMemo(() => {
    if (selectedCluster === null) return points;
    return points.filter((p) => p.cluster_id === selectedCluster);
  }, [points, selectedCluster]);

  // Cluster counts
  const clusterCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    for (let i = 0; i < nClusters; i++) counts[i] = 0;
    points.forEach((p) => {
      counts[p.cluster_id] = (counts[p.cluster_id] || 0) + 1;
    });
    return counts;
  }, [points, nClusters]);

  const handleClusterSelect = (cId: number | null) => {
    setSelectedCluster(cId);
    setPage(0);
    if (cId !== null && centroids[cId]) {
      setFlyTarget([centroids[cId].lat, centroids[cId].lng]);
    } else {
      setFlyTarget([DEPOT.lat, DEPOT.lng]);
    }
  };

  const orders = deliveriesQuery.data?.items ?? [];
  const totalOrders = deliveriesQuery.data?.total ?? 0;
  const totalPages = Math.ceil(totalOrders / PAGE_SIZE);

  return (
    <div className="space-y-2.5 sm:space-y-3 flex-1 flex flex-col min-h-0">
      {/* ============================================================== */}
      {/* 1. SCREEN HEADER WITH SCIENTIFIC CONTEXT                       */}
      {/* ============================================================== */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 border-b border-border/70 pb-2.5">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground">
              Phân cụm không gian
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground">
            2.394 đơn hàng · 10 cụm K-Means · Silhouette 0,429
          </p>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge variant="outline" className="border-teal-600/40 bg-teal-600/10 text-teal-700 dark:text-teal-300 font-mono text-xs font-semibold">
            Tập dữ liệu XeDu TP.HCM
          </Badge>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. PHƯƠNG PHÁP PHÂN CỤM & HỖ TRỢ ĐỊNH TUYẾN                   */}
      {/* ============================================================== */}
      <details className="group rounded-2xl border border-teal-500/30 bg-card p-3.5 sm:p-4 text-xs shadow-xs space-y-3">
        <summary className="flex items-center justify-between cursor-pointer list-none select-none">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-teal-600 dark:text-teal-400 shrink-0" />
            <span className="font-bold text-sm text-foreground">
              Phương pháp phân cụm và hỗ trợ định tuyến (K-Means &amp; Candidate Pruning)
            </span>
          </div>
          <div className="flex items-center gap-2">
            {selectedCluster !== null && (
              <span className="px-2.5 py-0.5 rounded-lg bg-teal-600 text-white font-mono text-xs font-bold">
                Cụm #{selectedCluster} ({clusterCounts[selectedCluster] || 0} đơn)
              </span>
            )}
            <span className="text-xs text-teal-600 dark:text-teal-400 font-semibold group-open:hidden">
              [Xem chi tiết ▼]
            </span>
            <span className="text-xs text-teal-600 dark:text-teal-400 font-semibold hidden group-open:inline">
              [Thu gọn ▲]
            </span>
          </div>
        </summary>

        <div className="pt-3 border-t border-border/70 grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs text-muted-foreground">
          {/* Cột A: Kết quả phân cụm */}
          <div className="p-3 rounded-xl border border-border/70 bg-muted/20 space-y-1.5">
            <span className="font-bold text-foreground text-xs block text-teal-700 dark:text-teal-300">
              A. Kết quả phân cụm không gian
            </span>
            <div className="space-y-1 text-muted-foreground">
              <div>• Số cụm tối ưu: <strong className="font-mono text-foreground">K = 10</strong></div>
              <div>• Hệ số phân tách: <strong className="font-mono text-foreground">Silhouette ≈ 0,429</strong></div>
              <div>• Phạm vi phân hoạch: <strong className="font-mono text-foreground">2.394 đơn hàng</strong></div>
            </div>
          </div>

          {/* Cột B: Thu hẹp tập ứng viên */}
          <div className="p-3 rounded-xl border border-border/70 bg-muted/20 space-y-1.5">
            <span className="font-bold text-foreground text-xs block text-teal-700 dark:text-teal-300">
              B. Thu hẹp tập ứng viên (Pruning)
            </span>
            <p className="leading-relaxed">
              Phân cụm giúp giảm số ứng viên được xem xét tại mỗi bước định tuyến, từ khoảng <strong className="text-foreground">43</strong> xuống <strong className="text-foreground">4,5 điểm/bước</strong> trong thiết lập thực nghiệm đã đo, hạ thấp độ phức tạp không gian hành động của mô hình RL.
            </p>
          </div>

          {/* Cột C: Liên hệ Candidate-Scoring DQN */}
          <div className="p-3 rounded-xl border border-border/70 bg-muted/20 space-y-2">
            <span className="font-bold text-foreground text-xs block text-teal-700 dark:text-teal-300">
              C. Liên hệ Candidate-Scoring DQN
            </span>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs items-baseline">
              <span className="text-muted-foreground font-medium whitespace-nowrap">Cơ sở đổi cụm:</span>
              <strong className="font-mono text-foreground font-semibold">cluster_switch (Feature #8)</strong>

              <span className="text-muted-foreground font-medium whitespace-nowrap">Trọng số thành phần đổi cụm:</span>
              <strong className="font-mono text-teal-600 dark:text-teal-400 font-bold">w_c = 0,050</strong>

              <span className="text-muted-foreground font-medium whitespace-nowrap">Ý nghĩa nghiệp vụ:</span>
              <span className="text-foreground leading-snug">Hạn chế shipper chuyển cụm không cần thiết giữa các khu vực giao hàng.</span>
            </div>
          </div>
        </div>

        {/* Locked scientific objective, checked against python/routing/proposed_rl_solver.py. */}
        <div className="p-3.5 rounded-xl border border-teal-500/20 bg-teal-500/5 space-y-3">
          <h3 className="font-bold text-sm text-foreground">Hàm mục tiêu định tuyến của mô hình Proposed RL</h3>
          <div
            className="rounded-xl border border-border/70 bg-card p-4 overflow-x-auto font-mono text-sm leading-relaxed shadow-2xs"
            role="math"
            aria-label="Hàm mục tiêu định tuyến J(π)"
          >
            <div className="grid grid-cols-[auto_auto_1fr] items-baseline gap-x-2 gap-y-1.5 whitespace-nowrap min-w-max">
              {/* Dòng 1: Khởi đầu hàm mục tiêu và hai thành phần đầu tiên */}
              <span className="text-foreground font-semibold">
                J(π) = Σ<sub>t</sub> [
              </span>
              <span className="w-4 text-center"></span>
              <span className="text-foreground">
                0,375 · d̄<sub>t</sub> + 0,375 · t̄<sub>t</sub>
              </span>

              {/* Dòng 2: Hai thành phần vi phạm thời gian, dấu cộng thẳng hàng */}
              <span></span>
              <span className="text-teal-600 dark:text-teal-400 font-bold text-center">+</span>
              <span className="text-foreground">
                0,100 · L̄<sub>t</sub> + 0,100 · Δt̄<sub>late,t</sub>
              </span>

              {/* Dòng 3: Thành phần đổi cụm và đóng ngoặc, dấu cộng thẳng hàng */}
              <span></span>
              <span className="text-teal-600 dark:text-teal-400 font-bold text-center">+</span>
              <span className="text-foreground">
                0,050 · c<sub>switch,t</sub> ]
              </span>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
            {[
              ["Cự ly", "0,375"],
              ["Thời gian", "0,375"],
              ["Chỉ dấu giao trễ", "0,100"],
              ["Thời lượng trễ", "0,100"],
              ["Đổi cụm", "0,050"],
            ].map(([label, weight]) => (
              <div key={label} className="p-2 rounded-md bg-card border border-border/60">
                <span className="text-muted-foreground block leading-snug">{label}</span>
                <strong className="font-mono text-foreground text-sm">{weight}</strong>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Trọng số được đối chiếu với cấu hình LOCKED_OBJECTIVE trong bộ giải Proposed RL.
            Các ký hiệu có dấu gạch trên (d̄, t̄, L̄, Δt̄_late) biểu thị thành phần chuẩn hóa; không cộng trực tiếp các đại lượng thô km, phút và số lần đổi cụm.
            Kết quả cụ thể của một bước ra quyết định xem tại Vết quyết định RL.
          </p>
        </div>
      </details>

      {/* ============================================================== */}
      {/* 3. CLUSTER SELECTOR PILLS                                      */}
      {/* ============================================================== */}
      <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-1">
        <button
          type="button"
          onClick={() => handleClusterSelect(null)}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
            selectedCluster === null
              ? "bg-teal-600 text-white shadow-xs"
              : "border border-border bg-card text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          Tất cả các cụm ({totalDeliveries})
        </button>
        {Array.from({ length: nClusters }, (_, i) => {
          const isSelected = selectedCluster === i;
          const count = clusterCounts[i] || 0;
          return (
            <button
              key={i}
              type="button"
              onClick={() => handleClusterSelect(i)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                isSelected
                  ? "bg-teal-600 text-white shadow-xs"
                  : "border border-border bg-card text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <span
                className="size-2.5 rounded-full shrink-0 border border-white/50"
                style={{ backgroundColor: clusterColor(i) }}
              />
              <span>Cụm {i}</span>
              <span className={`text-[11px] font-mono px-1 rounded-full ${isSelected ? "bg-white/20 text-white font-bold" : "bg-muted text-muted-foreground font-semibold"}`}>
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* ============================================================== */}
      {/* 4. SPLIT LAYOUT: INTERACTIVE MAP (60%) + DATA TABLE (40%)      */}
      {/* ============================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 sm:gap-4 items-stretch flex-1 min-h-0">
        {/* Left Column: Interactive GIS Map */}
        <div className="lg:col-span-7 flex flex-col">
          <Card className="border-border/80 shadow-xs overflow-hidden flex flex-col h-[500px] sm:h-[530px] lg:h-[550px] xl:h-[570px] 2xl:h-[610px]">
            <CardHeader className="p-2.5 sm:px-3.5 border-b border-border/60 flex flex-row items-center justify-between shrink-0">
              <div>
                <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                  Bản đồ phân cụm GIS
                </CardTitle>
                <CardDescription className="text-xs">
                  {selectedCluster !== null
                    ? `Đang hiển thị Cụm #${selectedCluster} (${visiblePoints.length} đơn hàng) • Bấm tâm cụm để xem tọa độ`
                    : `Hiển thị toàn bộ ${points.length} đơn hàng theo 10 màu cụm phân biệt`}
                </CardDescription>
              </div>
              {selectedCluster !== null && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleClusterSelect(null)}
                  className="text-xs text-teal-600 h-7 cursor-pointer"
                >
                  Xóa lọc cụm
                </Button>
              )}
            </CardHeader>

            <div className="flex-1 w-full relative min-h-0">
              <MapErrorBoundary>
                <MapBase height="100%">
                  <MapFlyTo target={flyTarget} />

                  {/* Delivery points */}
                  {visiblePoints.map((pt, i) => (
                    <CircleMarker
                      key={`pt-${pt.delivery_id || i}`}
                      center={[pt.latitude, pt.longitude]}
                      radius={selectedCluster === pt.cluster_id ? 5 : 3.5}
                      pathOptions={{
                        color: "#ffffff",
                        weight: selectedCluster === pt.cluster_id ? 1.5 : 0.8,
                        fillColor: clusterColor(pt.cluster_id),
                        fillOpacity: selectedCluster !== null && selectedCluster !== pt.cluster_id ? 0.2 : 0.85,
                      }}
                    >
                      <Popup>
                        <div className="text-xs font-sans space-y-1">
                          <div className="font-bold text-foreground">Đơn hàng: {pt.delivery_id}</div>
                          <div>Cụm K-Means: <strong style={{ color: clusterColor(pt.cluster_id) }}>Cụm #{pt.cluster_id}</strong></div>
                          <div className="text-muted-foreground">Tọa độ: {pt.latitude.toFixed(4)}, {pt.longitude.toFixed(4)}</div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}

                  {/* Centroids */}
                  {Object.entries(centroids).map(([clusterIdxStr, c]) => {
                    const i = Number(clusterIdxStr);
                    if (selectedCluster !== null && selectedCluster !== i) return null;
                    return (
                      <CircleMarker
                        key={`cent-${i}`}
                        center={[c.lat, c.lng]}
                        radius={9}
                        pathOptions={{
                          color: "#ffffff",
                          weight: 2.5,
                          fillColor: clusterColor(i),
                          fillOpacity: 1,
                        }}
                      >
                        <Popup>
                          <div className="text-xs font-sans space-y-1">
                            <div className="font-bold text-foreground">Tâm cụm Cụm #{i} (Centroid)</div>
                            <div>Tọa độ trung tâm: {c.lat.toFixed(4)}, {c.lng.toFixed(4)}</div>
                            <div>Số đơn trong cụm: {clusterCounts[i]} đơn</div>
                          </div>
                        </Popup>
                      </CircleMarker>
                    );
                  })}

                  {/* Depot Marker */}
                  <CircleMarker
                    center={[DEPOT.lat, DEPOT.lng]}
                    radius={9}
                    pathOptions={{
                      color: "#ffffff",
                      weight: 2.5,
                      fillColor: "#0f766e",
                      fillOpacity: 1,
                    }}
                  >
                    <Popup>
                      <div className="text-xs font-sans font-bold">DEPOT TÂN BÌNH (Kho Trung Tâm)</div>
                    </Popup>
                  </CircleMarker>
                </MapBase>
              </MapErrorBoundary>

              {/* Map Floating Legend */}
              <div className="absolute bottom-3 left-3 z-[1000] bg-background/90 backdrop-blur-md px-3 py-2 rounded-xl border border-border/80 shadow-md text-[11px] space-y-1">
                <div className="font-bold text-muted-foreground uppercase text-[10px]">Chú giải bản đồ</div>
                <div className="flex items-center gap-2">
                  <span className="size-3 rounded-full bg-teal-600 border border-white" />
                  <span>Kho Depot</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="size-3 rounded-full border-2 border-white bg-amber-500" />
                  <span>Tâm cụm Centroid</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-slate-400" />
                  <span>Đơn hàng theo cụm</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Delivery Orders Table & Search */}
        <div className="lg:col-span-5 flex flex-col">
          <Card className="border-border/80 shadow-xs flex flex-col h-[500px] sm:h-[530px] lg:h-[550px] xl:h-[570px] 2xl:h-[610px] overflow-hidden">
            <CardHeader className="p-2.5 sm:px-3.5 border-b border-border/60 shrink-0">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                    Danh sách đơn hàng ({totalOrders})
                  </CardTitle>
                </div>
                {/* Search input */}
                <div className="relative w-full sm:w-52">
                  <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
                  <Input
                    type="text"
                    placeholder="Tìm mã đơn, địa chỉ…"
                    value={searchTerm}
                    onChange={(e) => {
                      setSearchTerm(e.target.value);
                      setPage(0);
                    }}
                    className="h-7 pl-8 text-xs bg-background"
                  />
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0 flex-1 min-h-0 flex flex-col">
              <div className="flex-1 min-h-0 overflow-y-auto overflow-x-auto">
                <Table>
                  <TableHeader className="sticky top-0 bg-card/95 backdrop-blur-xs z-10 shadow-2xs border-b border-border/80">
                    <TableRow className="text-xs bg-muted/40 hover:bg-muted/40 font-bold">
                      <TableHead className="w-20 font-bold text-foreground">Mã đơn</TableHead>
                      <TableHead className="font-bold text-foreground">Địa chỉ giao</TableHead>
                      <TableHead className="w-16 text-center font-bold text-foreground">Cụm</TableHead>
                      <TableHead className="w-24 text-right font-bold text-foreground">Khối lượng</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {deliveriesQuery.loading ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8 text-xs text-muted-foreground">
                          Đang tải dữ liệu đơn hàng…
                        </TableCell>
                      </TableRow>
                    ) : orders.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8 text-xs text-muted-foreground">
                          Không tìm thấy đơn hàng nào phù hợp với bộ lọc.
                        </TableCell>
                      </TableRow>
                    ) : (
                      orders.map((o) => {
                        const cId = Number(o.cluster_id ?? 0);
                        const mdhStr = o.mdh ? String(o.mdh) : o.id;
                        const destStr = o.destinationAddress ? String(o.destinationAddress) : "TP. Hồ Chí Minh";
                        return (
                          <TableRow
                            key={o.id}
                            onClick={() => {
                              setSelectedOrder(o);
                              if (o.latitude && o.longitude) {
                                setFlyTarget([Number(o.latitude), Number(o.longitude)]);
                              }
                            }}
                            className="cursor-pointer hover:bg-muted/50 text-xs sm:text-sm transition-colors"
                          >
                            <TableCell className="font-mono font-bold text-foreground text-xs">
                              {mdhStr}
                            </TableCell>
                            <TableCell className="max-w-[160px] truncate text-muted-foreground text-xs" title={destStr}>
                              {destStr}
                            </TableCell>
                            <TableCell className="text-center">
                              <span
                                className="inline-flex items-center justify-center size-5.5 rounded-full text-xs font-bold text-white shadow-2xs"
                                style={{ backgroundColor: clusterColor(cId) }}
                              >
                                {cId}
                              </span>
                            </TableCell>
                            <TableCell className="text-right font-mono text-xs font-medium">
                              {o.weight ? `${o.weight}kg` : "—"}
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </div>

              {/* Selected Order Detail Bar (Contained within table card to preserve viewport-fit) */}
              {selectedOrder && (
                <div className="border-t border-teal-500/30 bg-teal-500/10 p-2 sm:px-3 text-xs flex items-center justify-between gap-2 shrink-0">
                  <div className="flex items-center gap-2 truncate">
                    <span className="font-mono font-bold text-teal-800 dark:text-teal-200">
                      {selectedOrder.mdh ? String(selectedOrder.mdh) : selectedOrder.id}
                    </span>
                    <Badge variant="outline" className="text-xs font-mono h-4.5 px-1.5">
                      Cụm #{Number(selectedOrder.cluster_id ?? 0)}
                    </Badge>
                    <span className="truncate text-muted-foreground text-xs" title={String(selectedOrder.destinationAddress)}>
                      {selectedOrder.destinationAddress ? String(selectedOrder.destinationAddress) : "TP. Hồ Chí Minh"}
                    </span>
                    <span className="font-mono text-xs text-muted-foreground shrink-0">
                      ({selectedOrder.weight ? `${selectedOrder.weight}kg` : "—"})
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedOrder(null)}
                    className="size-5 p-0 text-muted-foreground hover:text-foreground shrink-0 cursor-pointer"
                  >
                    ×
                  </Button>
                </div>
              )}

              {/* Table pagination */}
              <div className="flex items-center justify-between p-2 sm:px-3 border-t border-border/60 text-xs shrink-0 bg-muted/20">
                <span className="text-muted-foreground text-xs font-medium">
                  Trang {page + 1} / {Math.max(1, totalPages)}
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page === 0}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    className="size-7 p-0 cursor-pointer"
                  >
                    <ChevronLeft className="size-3.5" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page >= totalPages - 1}
                    onClick={() => setPage((p) => p + 1)}
                    className="size-7 p-0 cursor-pointer"
                  >
                    <ChevronRight className="size-3.5" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
