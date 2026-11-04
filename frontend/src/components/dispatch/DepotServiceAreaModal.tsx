import React, { useState, useEffect } from "react";
import {
  Building2,
  MapPin,
  Truck,
  Layers,
  CheckCircle2,
  AlertTriangle,
  X,
  Compass,
  RefreshCw,
  ShieldCheck,
  ChevronRight,
  Info,
  Radio,
  Sliders,
} from "lucide-react";
import {
  api,
  DispatchSessionState,
  StagingBase,
  ShipperServiceAreaItem,
  ShipperData,
} from "@/lib/api";
import { clusterColor } from "@/lib/format";

interface DepotServiceAreaModalProps {
  isOpen: boolean;
  onClose: () => void;
  session: DispatchSessionState;
  onSessionUpdated?: (updatedSession: DispatchSessionState) => void;
  onFocusCoordinates?: (coords: [number, number]) => void;
}

export function DepotServiceAreaModal({
  isOpen,
  onClose,
  session,
  onSessionUpdated,
  onFocusCoordinates,
}: DepotServiceAreaModalProps) {
  const [loading, setLoading] = useState<boolean>(false);
  const [stagingBases, setStagingBases] = useState<StagingBase[]>([]);
  const [serviceAreas, setServiceAreas] = useState<ShipperServiceAreaItem[]>([]);
  const [selectedShipperId, setSelectedShipperId] = useState<string>("shipper_01");
  const [editMode, setEditMode] = useState<"START_AT_DEPOT" | "START_AT_BASE">("START_AT_DEPOT");
  const [editBaseId, setEditBaseId] = useState<string>("base_tb");
  const [editHomeCluster, setEditHomeCluster] = useState<number>(0);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Load staging bases and service areas
  const loadData = async () => {
    try {
      setLoading(true);
      const [bases, areas] = await Promise.all([
        api.getStagingBases(session.session_id),
        api.getServiceAreas(session.session_id),
      ]);
      setStagingBases(bases);
      setServiceAreas(areas);

      const s = areas.find((a) => a.shipper_id === selectedShipperId) || areas[0];
      if (s) {
        setEditMode(s.shift_start_mode);
        setEditBaseId(s.base_id);
        setEditHomeCluster(s.home_cluster_id);
      }
    } catch (err: any) {
      console.warn("Could not load service areas:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen, session.session_id]);

  useEffect(() => {
    const s = serviceAreas.find((a) => a.shipper_id === selectedShipperId);
    if (s) {
      setEditMode(s.shift_start_mode);
      setEditBaseId(s.base_id);
      setEditHomeCluster(s.home_cluster_id);
    }
  }, [selectedShipperId, serviceAreas]);

  if (!isOpen) return null;

  const currentShipper = serviceAreas.find((a) => a.shipper_id === selectedShipperId);

  const handleSaveShipper = async () => {
    try {
      setLoading(true);
      setSaveError(null);
      setSaveSuccess(null);
      const res = await api.updateShipperServiceArea(session.session_id, selectedShipperId, {
        shift_start_mode: editMode,
        base_id: editBaseId,
        home_cluster_id: editHomeCluster,
      });
      if (res.success) {
        setSaveSuccess(`Đã cập nhật chế độ cho ${selectedShipperId}!`);
        // Refresh session state
        const updated = await api.getDispatchSession(session.session_id);
        if (onSessionUpdated) onSessionUpdated(updated);
        await loadData();
      }
    } catch (err: any) {
      setSaveError(err.message || "Lỗi khi lưu cấu hình shipper.");
    } finally {
      setLoading(false);
    }
  };

  const handleFocusDepot = () => {
    if (onFocusCoordinates) {
      onFocusCoordinates([session.depot.lat, session.depot.lng]);
      onClose();
    }
  };

  const handleFocusBase = (lat: number, lng: number) => {
    if (onFocusCoordinates) {
      onFocusCoordinates([lat, lng]);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-3 sm:p-4">
      <div className="flex flex-col max-h-[92vh] w-full max-w-5xl rounded-2xl border border-border/80 bg-background text-foreground shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border/80 bg-muted/40 px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="flex size-9 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
              <Building2 className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-extrabold tracking-tight">
                  Quản Lý Kho Trung Tâm & Điểm Trực Ca Shipper
                </h2>
                <span className="rounded-full bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  GIS Logistics Hub
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Định vị kho trung tâm, cấu hình trạm trực ca ngoại vi và phân bổ cụm phụ trách cho đội xe.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors cursor-pointer"
            aria-label="Đóng cửa sổ"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Section 1: Central Depot Landmark Card */}
          <div className="rounded-xl border border-border/80 bg-card p-4 shadow-2xs">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex items-start gap-3">
                <div className="flex size-11 items-center justify-center rounded-xl bg-sky-600 text-white shadow-md shrink-0">
                  <Building2 className="size-6" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-sm text-foreground">
                      {session.depot.name}
                    </h3>
                    <span className="rounded-md bg-sky-500/15 border border-sky-500/30 px-2.5 py-0.5 font-mono text-xs font-bold text-sky-700 dark:text-sky-300">
                      CENTRAL HUB (KHO CHÍNH)
                    </span>
                    <span className="rounded-md bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 font-mono text-xs font-bold text-emerald-700 dark:text-emerald-300">
                      HOẠT ĐỘNG
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {session.depot.description || "Kho điều phối hàng hóa trung tâm Tân Bình, TP.HCM (Mô phỏng)"}
                  </p>
                  <div className="flex flex-wrap items-center gap-3 mt-2 text-xs text-muted-foreground font-mono">
                    <span>
                      Tọa độ: <strong>{session.depot.lat.toFixed(4)}, {session.depot.lng.toFixed(4)}</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Đội xe liên kết: <strong>{session.shippers.length} Shipper</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Phục vụ: <strong>10 Cụm K-Means TP.HCM</strong>
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleFocusDepot}
                  className="flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-semibold text-primary hover:bg-primary/20 transition-colors cursor-pointer shadow-2xs"
                  title="Di chuyển bản đồ đến vị trí Kho trung tâm"
                >
                  <MapPin className="size-3.5" />
                  <span>Định vị trên bản đồ</span>
                </button>
              </div>
            </div>

            {/* Note on logistics physics */}
            <div className="mt-3 rounded-lg bg-muted/40 border border-border/60 p-3 text-xs text-muted-foreground flex items-center gap-2.5">
              <Info className="size-4 text-sky-500 shrink-0" />
              <span>
                <strong>Nguyên lý luân chuyển:</strong> Toàn bộ kiện hàng được tập kết tại Kho trung tâm Tân Bình. Shipper bắt đầu ca từ trạm trực ca ngoại vi (Mode B) bắt buộc phải di chuyển ghé Kho trung tâm nhận hàng trước khi tiến hành giao đến các điểm khách hàng.
              </span>
            </div>
          </div>

          {/* Section 2: Staging Operating Bases Overview */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                Mạng Lưới Trạm Trực Ca Ngoại Vi TP.HCM ({stagingBases.length} Trạm)
              </h4>
              <span className="text-xs text-muted-foreground">
                Điểm tập kết nghiệp vụ • Không phải địa chỉ cá nhân
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
              {stagingBases.map((base) => {
                const assignedShippers = serviceAreas.filter((s) => s.base_id === base.base_id);
                return (
                  <div
                    key={base.base_id}
                    className="flex flex-col justify-between rounded-xl border border-border/70 bg-card p-3 shadow-2xs hover:border-primary/50 transition-colors"
                  >
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs text-foreground flex items-center gap-1.5">
                          <span className="size-2 rounded-full bg-purple-500" />
                          {base.name}
                        </span>
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground">
                          {base.district}
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Tọa độ: <span className="font-mono">{base.lat.toFixed(4)}, {base.lng.toFixed(4)}</span>
                      </div>
                      <div className="flex items-center gap-1 mt-1.5">
                        <span className="text-xs text-muted-foreground">Cụm phụ trách:</span>
                        {base.primary_clusters.map((c) => (
                          <span
                            key={c}
                            className="inline-block size-3 rounded-full border border-white dark:border-black"
                            style={{ backgroundColor: clusterColor(c) }}
                            title={`Cụm #${c}`}
                          />
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-between border-t border-border/40 pt-2 mt-2 text-xs">
                      <span className="text-muted-foreground">
                        Trực thuộc: <strong>{assignedShippers.length} Shipper</strong>
                      </span>
                      <button
                        type="button"
                        onClick={() => handleFocusBase(base.lat, base.lng)}
                        className="text-primary hover:underline font-semibold"
                      >
                        Định vị
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section 3: Shipper Service Area Configuration */}
          <div className="rounded-xl border border-border/80 bg-card p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-border/60 pb-2">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                  Cấu Hình Điểm Xuất Phát & Cụm Phụ Trách Shipper
                </h4>
                <p className="text-xs text-muted-foreground">
                  Chọn shipper để điều chỉnh chế độ ca (Tại Kho vs Tại Trạm) và khu vực địa lý ưu tiên.
                </p>
              </div>

              {saveSuccess && (
                <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-500/20 flex items-center gap-1">
                  <CheckCircle2 className="size-3.5" />
                  {saveSuccess}
                </span>
              )}
              {saveError && (
                <span className="text-xs font-semibold text-rose-600 dark:text-rose-400 bg-rose-500/10 px-2.5 py-1 rounded-md border border-rose-500/20 flex items-center gap-1">
                  <AlertTriangle className="size-3.5" />
                  {saveError}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* Left: Shipper Selector List */}
              <div className="lg:col-span-5 max-h-[260px] overflow-y-auto space-y-1 pr-1">
                {serviceAreas.map((s) => {
                  const isSelected = s.shipper_id === selectedShipperId;
                  return (
                    <button
                      key={s.shipper_id}
                      type="button"
                      onClick={() => setSelectedShipperId(s.shipper_id)}
                      className={`w-full flex items-center justify-between p-2 rounded-lg text-left text-xs transition-colors cursor-pointer border ${
                        isSelected
                          ? "bg-primary/10 border-primary/50 font-semibold text-foreground shadow-2xs"
                          : "bg-muted/20 border-transparent hover:bg-muted text-muted-foreground"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="size-5 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs">
                          {s.shipper_id.replace("shipper_", "")}
                        </span>
                        <span>{s.name}</span>
                      </div>

                      <div className="flex items-center gap-2">
                        <span
                          className="inline-block size-3 rounded-full border border-white dark:border-black"
                          style={{ backgroundColor: clusterColor(s.home_cluster_id) }}
                          title={`Cụm chính #${s.home_cluster_id}`}
                        />
                        <span className={`text-xs px-2 py-0.5 rounded font-mono font-semibold ${s.shift_start_mode === "START_AT_BASE" ? "bg-purple-500/15 text-purple-700 dark:text-purple-300" : "bg-sky-500/15 text-sky-700 dark:text-sky-300"}`}>
                          {s.shift_start_mode === "START_AT_BASE" ? "Tại Trạm" : "Tại Kho"}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Right: Active Shipper Edit Form */}
              <div className="lg:col-span-7 rounded-xl border border-border/60 bg-muted/20 p-3.5 flex flex-col justify-between">
                {currentShipper ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between border-b border-border/40 pb-2">
                      <div className="flex items-center gap-2">
                        <Truck className="size-4 text-primary" />
                        <span className="font-bold text-sm text-foreground">
                          {currentShipper.name} ({currentShipper.shipper_id})
                        </span>
                      </div>
                      <span className="text-xs text-muted-foreground font-mono">
                        Tải: {currentShipper.current_load_kg.toFixed(1)} / {currentShipper.capacity_kg} kg
                      </span>
                    </div>

                    {/* Start Mode Radio */}
                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        Chế độ xuất phát đầu ca:
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <label className={`flex items-start gap-2 p-2 rounded-lg border cursor-pointer transition-colors ${editMode === "START_AT_DEPOT" ? "bg-primary/10 border-primary text-foreground" : "bg-card border-border/70 text-muted-foreground"}`}>
                          <input
                            type="radio"
                            name="start_mode"
                            value="START_AT_DEPOT"
                            checked={editMode === "START_AT_DEPOT"}
                            onChange={() => setEditMode("START_AT_DEPOT")}
                            className="mt-0.5"
                          />
                          <div>
                            <div className="font-bold text-sm text-foreground">Mode A: Tại Kho trung tâm</div>
                            <div className="text-xs text-muted-foreground mt-0.5">Nhận hàng và đi giao trực tiếp từ Tân Bình</div>
                          </div>
                        </label>

                        <label className={`flex items-start gap-2 p-2 rounded-lg border cursor-pointer transition-colors ${editMode === "START_AT_BASE" ? "bg-primary/10 border-primary text-foreground" : "bg-card border-border/70 text-muted-foreground"}`}>
                          <input
                            type="radio"
                            name="start_mode"
                            value="START_AT_BASE"
                            checked={editMode === "START_AT_BASE"}
                            onChange={() => setEditMode("START_AT_BASE")}
                            className="mt-0.5"
                          />
                          <div>
                            <div className="font-bold text-sm text-foreground">Mode B: Tại Trạm trực ca</div>
                            <div className="text-xs text-muted-foreground mt-0.5">Xuất phát từ Trạm → ghé Kho nhận hàng</div>
                          </div>
                        </label>
                      </div>
                    </div>

                    {/* Staging Base Selector */}
                    <div className="space-y-1">
                      <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        Trạm trực ca phân công:
                      </label>
                      <select
                        value={editBaseId}
                        onChange={(e) => setEditBaseId(e.target.value)}
                        className="w-full h-9 px-3 rounded-lg border border-border bg-background text-xs font-semibold text-foreground focus:ring-1 focus:ring-primary cursor-pointer"
                      >
                        {stagingBases.map((b) => (
                          <option key={b.base_id} value={b.base_id}>
                            {b.name} ({b.district})
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Home Cluster Selector */}
                    <div className="space-y-1">
                      <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        Cụm địa lý ưu tiên chính:
                      </label>
                      <select
                        value={editHomeCluster}
                        onChange={(e) => setEditHomeCluster(parseInt(e.target.value, 10))}
                        className="w-full h-9 px-3 rounded-lg border border-border bg-background text-xs font-semibold text-foreground focus:ring-1 focus:ring-primary cursor-pointer"
                      >
                        {Array.from({ length: 10 }).map((_, c) => (
                          <option key={c} value={c}>
                            Cụm K-Means #{c} (Khu vực địa lý #{c})
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Action Button */}
                    <div className="pt-2 flex justify-end">
                      <button
                        type="button"
                        onClick={handleSaveShipper}
                        disabled={loading}
                        className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90 transition-colors shadow-xs cursor-pointer disabled:opacity-50"
                      >
                        <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
                        <span>Áp dụng cấu hình</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-xs text-muted-foreground">
                    Chọn một shipper bên trái để cấu hình.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="border-t border-border/80 bg-muted/30 px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <ShieldCheck className="size-3.5 text-emerald-500" />
            <span>Mọi thay đổi vị trí chỉ áp dụng trong phiên mô phỏng hiện hành, bảo vệ 100% dữ liệu gốc của Leader.</span>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-border px-4 py-1.5 text-xs font-semibold hover:bg-muted transition-colors cursor-pointer"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}

