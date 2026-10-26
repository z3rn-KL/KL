import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import {
  Search,
  Sun,
  Moon,
  Maximize2,
  Minimize2,
  Download,
  MapPin,
  Route,
  Activity,
  Layers,
  BarChart3,
  Sliders,
  ShieldCheck,
  Package,
  Compass,
  LayoutDashboard,
  Network,
  PlayCircle,
  BrainCircuit,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface NavCommandItem {
  to: string;
  label: string;
  shortLabel?: string;
  category?: string;
  icon?: React.ElementType;
}

export interface AtlasWorkspaceItemRef {
  id: string;
  to: string;
  title: string;
  shortTitle: string;
  category: "spatial" | "engine" | "rl" | "validation";
  icon: React.ElementType;
  badge?: string;
}

export interface CommandItem {
  id: string;
  title: string;
  subtitle?: string;
  section: "Phân hệ làm việc" | "Tùy biến môi trường" | "Hành động dữ liệu";
  icon: React.ElementType;
  badge?: string;
  keywords?: string[];
  action: () => void;
  disabled?: boolean;
}

interface AtlasCommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onToggleTheme: () => void;
  isDarkTheme: boolean;
  onTogglePresentation: () => void;
  isPresentationMode: boolean;
  navItems?: NavCommandItem[];
  workspaces?: AtlasWorkspaceItemRef[];
}

const WORKSPACE_DESCRIPTIONS: Record<string, { desc: string; badge: string }> = {
  "/tong-quan": { desc: "Toàn cảnh GIS và đối chuẩn cự ly baseline", badge: "Tổng quan" },
  "/don-hang": { desc: "Tra cứu 2.394 đơn nội đô và cam kết SLA", badge: "Dữ liệu" },
  "/phan-cum": { desc: "10 cụm K-Means và phân tích điểm ngoại lai", badge: "Dữ liệu" },
  "/tuyen-giao": { desc: "Bản đồ tuyến 13 workload và thanh tra đổi cụm", badge: "Định tuyến" },
  "/chay-thu": { desc: "Thử nghiệm giải thuật song song NN và Clarke-Wright", badge: "Định tuyến" },
  "/guided-dqn": { desc: "Nghiên cứu hàm phần thưởng gom cụm λ=0,50", badge: "Nghiên cứu" },
  "/so-sanh": { desc: "Đối chuẩn thuật toán trên 3 cohort độc lập", badge: "Nghiên cứu" },
  "/do-nhay": { desc: "Khảo sát đánh đổi cự ly qua 24 lần chạy", badge: "Nghiên cứu" },
  "/on-dinh": { desc: "Kiểm chứng 10 seed khớp 120 hoán vị tối ưu", badge: "Nghiên cứu" },
};

export function AtlasCommandPalette({
  isOpen,
  onClose,
  onToggleTheme,
  isDarkTheme,
  onTogglePresentation,
  isPresentationMode,
  navItems,
  workspaces,
}: AtlasCommandPaletteProps) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // Focus trap, restoration & body scroll lock
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      setSearch("");
      setSelectedIndex(0);
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      const timer = setTimeout(() => inputRef.current?.focus(), 40);
      return () => {
        clearTimeout(timer);
        document.body.style.overflow = originalOverflow;
      };
    } else if (previousFocusRef.current) {
      previousFocusRef.current.focus();
    }
  }, [isOpen]);

  const handleExportData = () => {
    const summary = {
      project: "KLTN_11 Logistics Delivery Route Optimization (Civic Atlas Rebuild)",
      dataset: "XeDu TP.HCM (2.394 orders)",
      clustering: "K-Means (K=10) & DBSCAN Spatial Density",
      algorithms: ["Nearest Neighbor", "Clarke-Wright", "Q-Learning", "SARSA", "Vanilla DQN", "K-Means-guided DQN"],
      key_contributions: {
        sla_compliance: "100%",
        cluster_switch_reduction: "-6.83% (12 Validation Workloads)",
        distance_tradeoff: "+1.76%",
        time_tradeoff: "+1.87%",
      },
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kltn11_atlas_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    onClose();
  };

  const navList = workspaces
    ? workspaces.map((w) => ({
        to: w.to,
        label: w.title,
        icon: w.icon,
      }))
    : navItems || [
        { to: "/tong-quan", label: "Tổng quan", icon: LayoutDashboard },
        { to: "/don-hang", label: "Đơn hàng", icon: Package },
        { to: "/phan-cum", label: "Phân cụm", icon: Network },
        { to: "/tuyen-giao", label: "Tuyến giao", icon: MapPin },
        { to: "/chay-thu", label: "Chạy thử", icon: PlayCircle },
        { to: "/guided-dqn", label: "Guided DQN", icon: BrainCircuit },
        { to: "/so-sanh", label: "So sánh", icon: BarChart3 },
        { to: "/do-nhay", label: "Độ nhạy", icon: Sliders },
        { to: "/on-dinh", label: "Độ ổn định", icon: ShieldCheck },
      ];

  const commands: CommandItem[] = [
    // 1. Phân hệ làm việc
    ...navList.map((n) => {
      const meta = WORKSPACE_DESCRIPTIONS[n.to] || { desc: `Chuyển tới ${n.label}`, badge: "Trang" };
      return {
        id: `nav-${n.to.replace("/", "")}`,
        title: n.label,
        subtitle: meta.desc,
        section: "Phân hệ làm việc" as const,
        icon: n.icon || Compass,
        badge: meta.badge,
        keywords: [n.label.toLowerCase(), n.to, meta.desc.toLowerCase()],
        action: () => {
          navigate(n.to);
          onClose();
        },
      };
    }),

    // 2. Tùy biến môi trường
    {
      id: "theme-toggle",
      title: isDarkTheme ? "Chuyển sang giao diện Sáng (Mineral Light)" : "Chuyển sang giao diện Tối (Deep Jade Dark)",
      subtitle: "Thay đổi giao diện màu khoáng ấm hoặc lục thạch tối",
      section: "Tùy biến môi trường" as const,
      icon: isDarkTheme ? Sun : Moon,
      badge: isDarkTheme ? "Light Mode" : "Dark Mode",
      keywords: ["theme", "giao diện", "sáng", "tối", "dark", "light"],
      action: () => {
        onToggleTheme();
        onClose();
      },
    },
    {
      id: "presentation-toggle",
      title: isPresentationMode ? "Thoát chế độ trình chiếu (Phím F / Esc)" : "Bật chế độ trình chiếu toàn màn hình (Phím F)",
      subtitle: "Ẩn các thanh công cụ, mở rộng bản đồ để báo cáo",
      section: "Tùy biến môi trường" as const,
      icon: isPresentationMode ? Minimize2 : Maximize2,
      badge: "Phím F",
      keywords: ["presentation", "trình chiếu", "toàn màn hình", "fullscreen", "báo cáo"],
      action: () => {
        onTogglePresentation();
        onClose();
      },
    },

    // 3. Hành động dữ liệu
    {
      id: "export-data",
      title: "Xuất tóm tắt nghiên cứu (JSON)",
      subtitle: "Tải file dữ liệu 4 cohort và các chỉ số tối ưu hóa",
      section: "Hành động dữ liệu" as const,
      icon: Download,
      badge: "JSON",
      keywords: ["export", "xuất", "download", "tải", "json", "dữ liệu"],
      action: handleExportData,
    },
  ];

  // Filter commands
  const filtered = commands.filter((cmd) => {
    if (!search.trim()) return true;
    const q = search.toLowerCase().trim();
    if (cmd.title.toLowerCase().includes(q)) return true;
    if (cmd.subtitle?.toLowerCase().includes(q)) return true;
    if (cmd.keywords?.some((k) => k.includes(q))) return true;
    return false;
  });

  useEffect(() => {
    setSelectedIndex(0);
  }, [search]);

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (filtered.length === 0) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % filtered.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filtered.length) % filtered.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const current = filtered[selectedIndex];
      if (current && !current.disabled) {
        current.action();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  const activeOptionId = filtered[selectedIndex]?.id || undefined;

  const dialogContent = (
    <div
      className="fixed inset-0 z-[9999] flex items-start justify-center pt-16 sm:pt-24 bg-black/60 backdrop-blur-xs p-3 sm:p-4 animate-in fade-in-0 duration-150"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Bảng điều hướng và lệnh"
    >
      <div
        className="cmd-dialog relative w-full max-w-xl mx-auto overflow-hidden rounded-xl border border-border bg-card text-foreground shadow-2xl animate-in zoom-in-95 duration-150 flex flex-col max-h-[80vh]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
        data-slot="command-dialog"
      >
        {/* Search Input Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border bg-muted/30">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-autocomplete="list"
            aria-controls="command-list-options"
            aria-activedescendant={activeOptionId}
            className="cmd-input flex-1 bg-transparent text-sm font-normal text-foreground placeholder:text-muted-foreground focus:outline-none"
            placeholder="Tìm trang hoặc lệnh toàn cục (Ctrl+K)…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Tìm kiếm trang hoặc lệnh toàn cục"
            data-slot="command-input"
          />
          <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono font-medium rounded border border-border bg-card text-muted-foreground shadow-2xs">
            Esc
          </kbd>
        </div>

        {/* Global Scope Clarification Hint */}
        <div className="px-4 py-1.5 bg-muted/20 border-b border-border/50 flex items-center justify-between text-[10.5px] text-muted-foreground select-none">
          <span className="font-medium">4 nhóm phân hệ: Tổng quan • Dữ liệu • Định tuyến • Nghiên cứu</span>
          <span className="hidden sm:inline opacity-70">Tra cứu đơn hàng tại phân hệ Đơn hàng</span>
        </div>

        {/* Command List */}
        <div
          id="command-list-options"
          role="listbox"
          aria-label="Danh sách lệnh và trang"
          className="cmd-list flex-1 overflow-y-auto p-2 space-y-1 max-h-[420px]"
          ref={listRef}
          data-slot="command-list"
        >
          {filtered.length === 0 ? (
            <div
              className="cmd-empty py-10 text-center text-xs font-medium text-muted-foreground"
              data-slot="command-empty"
            >
              Không tìm thấy trang hoặc lệnh nào phù hợp.
            </div>
          ) : (
            filtered.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              const isDisabled = !!item.disabled;
              const IconComponent = item.icon;
              return (
                <div
                  key={item.id}
                  id={item.id}
                  data-slot="command-item"
                  data-selected={isSelected ? "true" : undefined}
                  role="option"
                  tabIndex={-1}
                  aria-selected={isSelected}
                  aria-disabled={isDisabled}
                  className={`cmd-item flex items-center gap-3 px-3 py-2 rounded-lg text-xs transition-colors select-none cursor-pointer ${
                    isSelected
                      ? "selected bg-primary/15 border border-primary/40 font-medium"
                      : "hover:bg-muted/40 border border-transparent"
                  }`}
                  onClick={() => {
                    if (!isDisabled) item.action();
                  }}
                  onMouseEnter={() => {
                    if (!isDisabled) setSelectedIndex(idx);
                  }}
                >
                  <div
                    className={`cmd-item-icon flex size-7 shrink-0 items-center justify-center rounded-md border transition-colors ${
                      isSelected
                        ? "border-primary bg-primary text-primary-foreground shadow-2xs"
                        : "border-border/80 bg-muted/40 text-muted-foreground"
                    }`}
                  >
                    <IconComponent className="size-3.5" />
                  </div>
                  <div className="cmd-item-text flex flex-col flex-1 min-w-0">
                    <div
                      className={`cmd-item-title text-xs tracking-tight truncate font-semibold ${
                        isSelected
                          ? "text-primary"
                          : "text-foreground"
                      }`}
                    >
                      {item.title}
                    </div>
                    {item.subtitle && (
                      <div className="cmd-item-subtitle text-[11px] text-muted-foreground truncate mt-0.5">
                        {item.subtitle}
                      </div>
                    )}
                  </div>
                  {item.badge && (
                    <Badge
                      variant="outline"
                      className={`cmd-item-badge text-[10px] font-mono shrink-0 px-1.5 py-0.2 transition-colors ${
                        isSelected
                          ? "border-primary/40 bg-primary/10 text-primary font-semibold"
                          : "border-border/60 bg-muted/20 text-muted-foreground"
                      }`}
                    >
                      {item.badge}
                    </Badge>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer Keyboard Hints */}
        <div
          className="cmd-footer flex items-center justify-between px-4 py-2 border-t border-border bg-muted/20 text-[11px] text-muted-foreground font-medium"
          data-slot="command-footer"
        >
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.2 rounded border border-border bg-card text-[10px] font-mono font-semibold text-foreground">↑</kbd>
            <kbd className="px-1.5 py-0.2 rounded border border-border bg-card text-[10px] font-mono font-semibold text-foreground">↓</kbd>
            <span className="ml-1">Chọn</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.2 rounded border border-border bg-card text-[10px] font-mono font-semibold text-foreground">↵ Enter</kbd>
            <span className="ml-1">Mở</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.2 rounded border border-border bg-card text-[10px] font-mono font-semibold text-foreground">Esc</kbd>
            <span className="ml-1">Đóng</span>
          </div>
        </div>
      </div>
    </div>
  );

  return typeof document !== "undefined"
    ? createPortal(dialogContent, document.body)
    : dialogContent;
}
