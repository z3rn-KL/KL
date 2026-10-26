import React, { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { LoadingCard } from "@/components/shared/DomainComponents";
import { clusterColor, dateTime, num, pct } from "@/lib/format";
import {
  Compass,
  ArrowRight,
  Zap,
  Clock,
  MapPin,
  CheckCircle2,
  AlertTriangle,
  Info,
  ShieldCheck,
  Layers,
} from "lucide-react";

export interface RouteStop {
  stopIndex: number;
  deliveryId: string;
  package_name?: string;
  clusterId?: number | null;
  lat: number;
  lng: number;
  expectedDeliveryTime?: string | null;
  isLate?: boolean | null;
  address?: string | null;
}

export interface RouteTransitionInspectorProps {
  stops: RouteStop[];
  algorithmLabel?: string;
  workloadId?: number | string;
  totalDistanceKm?: number | null;
  totalTravelTimeMinutes?: number | null;
  onTimeRate?: number | null;
  isLoading?: boolean;
  selectedStopId?: string | null;
  onSelectStop?: (stop: RouteStop) => void;
  className?: string;
}

/**
 * Calculates Haversine great-circle distance in kilometers between two lat/lng coordinates.
 * Explicitly documented: this is straight-line air distance, not road network distance.
 */
function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  if (!Number.isFinite(lat1) || !Number.isFinite(lon1) || !Number.isFinite(lat2) || !Number.isFinite(lon2)) {
    return 0;
  }
  const R = 6371.0088;
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function RouteTransitionInspector({
  stops,
  algorithmLabel,
  workloadId,
  totalDistanceKm,
  totalTravelTimeMinutes,
  onTimeRate,
  isLoading = false,
  selectedStopId,
  onSelectStop,
  className = "",
}: RouteTransitionInspectorProps) {
  // Normalize transitions and count genuine cluster changes
  const transitionData = useMemo(() => {
    let clusterSwitches = 0;
    const enriched = stops.map((curr, idx) => {
      const prev = idx > 0 ? stops[idx - 1] : null;
      let isClusterSwitch = false;
      let haversineLegKm: number | null = null;

      if (prev) {
        // Only count switch when both consecutive deliveries have valid numeric cluster IDs
        if (
          typeof curr.clusterId === "number" &&
          typeof prev.clusterId === "number" &&
          curr.clusterId !== prev.clusterId
        ) {
          isClusterSwitch = true;
          clusterSwitches++;
        }
        haversineLegKm = haversineKm(prev.lat, prev.lng, curr.lat, curr.lng);
      }

      return {
        ...curr,
        prevClusterId: prev?.clusterId ?? null,
        isClusterSwitch,
        haversineLegKm,
      };
    });

    return {
      stops: enriched,
      totalSwitches: clusterSwitches,
    };
  }, [stops]);

  if (isLoading) {
    return (
      <Card className={`border-border/80 shadow-xs ${className}`}>
        <CardContent className="p-6">
          <LoadingCard message="Đang nạp dữ liệu chuỗi hành trình và phân tích chuyển vùng..." />
        </CardContent>
      </Card>
    );
  }

  if (!stops || stops.length === 0) {
    return (
      <Card className={`border-border/80 shadow-xs ${className}`}>
        <CardHeader className="p-4 sm:px-5">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Compass className="size-4 text-muted-foreground" />
            <span>Thanh tra Chuỗi Giao hàng (Route Transition Inspector)</span>
          </CardTitle>
          <CardDescription className="text-xs">
            Chưa có dữ liệu lộ trình để thanh tra. Vui lòng chọn một Workload hoặc chạy tối ưu tuyến.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className={`border-border/80 shadow-xs overflow-hidden ${className}`}>
      {/* Header with Title and Overall Route Indicators */}
      <CardHeader className="p-4 sm:px-5 pb-3 border-b border-border/60 bg-muted/20">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Compass className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">
                Thanh tra Chuỗi Giao hàng Thực tế
              </CardTitle>
              {algorithmLabel && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {algorithmLabel}
                </Badge>
              )}
              {workloadId !== undefined && (
                <Badge variant="secondary" className="text-[10px] font-mono">
                  WL #{workloadId}
                </Badge>
              )}
            </div>
            <CardDescription className="text-xs">
              Giám sát thứ tự ghé thăm, hạn cam kết SLA và các điểm chuyển cụm không gian từ kết quả tính toán thực tế.
            </CardDescription>
          </div>

          {/* Quick Route Highlights */}
          <div className="flex items-center gap-2 text-xs flex-wrap">
            <Badge
              variant="outline"
              className={`px-2 py-0.5 font-mono text-xs font-semibold ${
                transitionData.totalSwitches > 0
                  ? "border-amber-500/40 text-amber-700 dark:text-amber-400 bg-amber-500/10"
                  : "border-emerald-500/40 text-emerald-700 dark:text-emerald-400 bg-emerald-500/10"
              }`}
            >
              <Zap className="size-3 mr-1 inline" />
              {transitionData.totalSwitches} lần đổi cụm
            </Badge>

            {totalDistanceKm !== null && totalDistanceKm !== undefined && (
              <span className="text-muted-foreground font-mono">
                {num(totalDistanceKm, 1)} km (OSRM)
              </span>
            )}

            {onTimeRate !== null && onTimeRate !== undefined && (
              <Badge variant="outline" className="text-[10px] text-emerald-600 dark:text-emerald-400 border-emerald-500/30">
                SLA {pct(onTimeRate, 0)}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      {/* Main Table: Sequential Inspections */}
      <CardContent className="p-0">
        <div className="max-h-[380px] overflow-y-auto">
          <Table className="tbl">
            <TableHeader>
              <TableRow className="bg-muted/40 text-[11px] border-b border-border/60 hover:bg-transparent">
                <TableHead className="w-[50px] text-center font-bold">Trạm</TableHead>
                <TableHead className="font-bold">Mã vận đơn & Kiện</TableHead>
                <TableHead className="font-bold">Cụm K-Means</TableHead>
                <TableHead className="font-bold">Chuyển vùng</TableHead>
                <TableHead className="font-bold">Hạn giao SLA</TableHead>
                <TableHead className="text-right font-bold">Cự ly chim bay</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {transitionData.stops.map((item) => {
                const isSelected = selectedStopId === item.deliveryId;
                const cColor =
                  typeof item.clusterId === "number" ? clusterColor(item.clusterId) : "#6B7280";

                return (
                  <TableRow
                    key={item.deliveryId}
                    onClick={() => onSelectStop?.(item)}
                    className={`cursor-pointer transition-colors text-xs border-b border-border/40 ${
                      isSelected ? "bg-accent/70 font-medium" : "hover:bg-muted/30"
                    } ${item.isClusterSwitch ? "bg-amber-500/[0.04]" : ""}`}
                  >
                    {/* Step Number */}
                    <TableCell className="text-center font-mono font-bold text-foreground py-2">
                      <span className="inline-flex size-5 items-center justify-center rounded-full bg-muted border border-border text-[11px]">
                        {item.stopIndex}
                      </span>
                    </TableCell>

                    {/* Order ID & Package */}
                    <TableCell className="py-2">
                      <div className="font-mono font-semibold text-foreground">
                        {item.deliveryId.slice(-8)}
                      </div>
                      <div className="text-[11px] text-muted-foreground truncate max-w-[180px]">
                        {item.package_name ?? item.address ?? "Kiện hàng"}
                      </div>
                    </TableCell>

                    {/* Cluster Badge */}
                    <TableCell className="py-2">
                      {typeof item.clusterId === "number" ? (
                        <Badge
                          variant="outline"
                          className="font-mono text-[10px] px-1.5 py-0"
                          style={{
                            borderColor: `${cColor}66`,
                            backgroundColor: `${cColor}15`,
                            color: cColor,
                          }}
                        >
                          Cụm #{item.clusterId}
                        </Badge>
                      ) : (
                        <span className="text-muted-foreground text-[11px]">Chưa có dữ liệu</span>
                      )}
                    </TableCell>

                    {/* Transition Indicator */}
                    <TableCell className="py-2">
                      {item.isClusterSwitch ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-bold text-amber-600 dark:text-amber-400">
                          <Zap className="size-3" />
                          <span>#{item.prevClusterId} → #{item.clusterId}</span>
                        </span>
                      ) : item.stopIndex > 1 && typeof item.clusterId === "number" ? (
                        <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                          <CheckCircle2 className="size-2.5 text-emerald-500" />
                          <span>Cùng cụm #{item.clusterId}</span>
                        </span>
                      ) : (
                        <span className="text-[11px] text-muted-foreground">–</span>
                      )}
                    </TableCell>

                    {/* Expected SLA Deadline */}
                    <TableCell className="py-2 font-mono text-[11px]">
                      {item.expectedDeliveryTime ? (
                        <span className="text-muted-foreground">
                          {dateTime(item.expectedDeliveryTime)}
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-[10px]">Linh hoạt</span>
                      )}
                    </TableCell>

                    {/* Haversine distance with explicit indicator */}
                    <TableCell className="py-2 text-right font-mono text-[11px]">
                      {item.haversineLegKm !== null ? (
                        <span className="text-muted-foreground" title="Khoảng cách đường chim bay (Haversine)">
                          +{num(item.haversineLegKm, 2)} km*
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-[10px]">Xuất phát</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>

      {/* Scientific Methodology Disclaimer Footer */}
      <CardFooter className="p-3 px-4 sm:px-5 border-t border-border/60 bg-muted/20 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-[11px] text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <Info className="size-3.5 text-primary shrink-0" />
          <span>
            <b>*Ghi chú khoa học:</b> Cự ly chặng đánh dấu (*) là <i>khoảng cách đường chim bay (Haversine)</i> giữa 2 tọa độ liên tiếp, không phải khoảng cách mạng đường bộ OpenStreetMap.
          </span>
        </div>
        <div className="text-[10px] font-mono text-muted-foreground/80 shrink-0">
          Chuyển cụm = 𝕀(c_t ≠ c_&#123;t-1&#125;)
        </div>
      </CardFooter>
    </Card>
  );
}

