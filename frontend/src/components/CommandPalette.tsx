import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Search,
  Sun,
  Moon,
  Maximize2,
  Minimize2,
  Sparkles,
  Download,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface NavCommandItem {
  to: string;
  label: string;
  shortLabel?: string;
  num: number;
  icon: React.ElementType;
}

export interface CommandItem {
  id: string;
  title: string;
  subtitle?: string;
  section: "Trang hệ thống" | "Tùy biến giao diện" | "Hành động dữ liệu";
  icon: React.ElementType;
  badge?: string;
  keywords?: string[];
  action: () => void;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onToggleTheme: () => void;
  isDarkTheme: boolean;
  onToggleEffects: () => void;
  isEffectsOn: boolean;
  onTogglePresentation: () => void;
  isPresentationMode: boolean;
  navItems: NavCommandItem[];
}

export function CommandPalette({
  isOpen,
  onClose,
  onToggleTheme,
  isDarkTheme,
  onToggleEffects,
  isEffectsOn,
  onTogglePresentation,
  isPresentationMode,
  navItems,
}: CommandPaletteProps) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Export dataset helper
  const handleExportData = () => {
    const summary = {
      project: "KLTN_11 Logistics Delivery Route Optimization",
      dataset: "XeDu TP.HCM (2.394 orders)",
      clustering: "K-Means (K=10) & DBSCAN",
      algorithms: ["Nearest Neighbor", "Clarke-Wright", "Q-Learning", "SARSA", "DQN", "K-Means-guided DQN"],
      key_contributions: {
        sla_rate: "100%",
        cluster_switch_reduction: "-6.83%",
        distance_tradeoff: "+1.76%",
        time_tradeoff: "+1.87%",
      },
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kltn11_summary_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    onClose();
  };

  const commands: CommandItem[] = [
    // 1. Navigation items
    ...navItems.map((n) => ({
      id: `nav-${n.to}`,
      title: `${n.num}. ${n.label}`,
      subtitle: `Chuyển nhanh tới trang ${n.label}`,
      section: "Trang hệ thống" as const,
      icon: n.icon,
      badge: `Trang ${n.num}`,
      keywords: [n.label.toLowerCase(), n.to, `trang ${n.num}`],
      action: () => {
        navigate(n.to);
        onClose();
      },
    })),

    // 2. Interface settings
    {
      id: "theme-toggle",
      title: isDarkTheme ? "Chuyển sang Giao diện Sáng (Light Mode)" : "Chuyển sang Giao diện Tối (Dark Mode)",
      subtitle: "Đổi màu nền graphite / trắng kem tiêu chuẩn",
      section: "Tùy biến giao diện" as const,
      icon: isDarkTheme ? Sun : Moon,
      badge: isDarkTheme ? "Dark" : "Light",
      keywords: ["theme", "giao diện", "dark", "light", "sáng", "tối", "màu"],
      action: () => {
        onToggleTheme();
        onClose();
      },
    },
    {
      id: "presentation-toggle",
      title: isPresentationMode ? "Thoát Chế độ Trình chiếu (Esc / F)" : "Bật Chế độ Trình chiếu Báo cáo (Phím F)",
      subtitle: "Ẩn thanh điều hướng bên, mở rộng toàn màn hình, cỡ chữ lớn thuyết trình",
      section: "Tùy biến giao diện" as const,
      icon: isPresentationMode ? Minimize2 : Maximize2,
      badge: "Phím F",
      keywords: ["presentation", "trình chiếu", "toàn màn hình", "fullscreen", "báo cáo"],
      action: () => {
        onTogglePresentation();
        onClose();
      },
    },
    {
      id: "effects-toggle",
      title: isEffectsOn ? "Tắt Hiệu ứng chuyển động (Tăng tốc độ)" : "Bật Hiệu ứng chuyển động (Animation)",
      subtitle: "Bật/Tắt hiệu ứng chuyển cảnh mượt và hiệu ứng ánh sáng nền",
      section: "Tùy biến giao diện" as const,
      icon: Sparkles,
      badge: isEffectsOn ? "Đang bật" : "Đang tắt",
      keywords: ["effects", "hiệu ứng", "motion", "animation", "giảm chuyển động"],
      action: () => {
        onToggleEffects();
        onClose();
      },
    },

    // 3. Data actions
    {
      id: "export-json",
      title: "Xuất Dữ liệu Tóm tắt Nghiên cứu (JSON)",
      subtitle: "Tải file số liệu kết quả tối ưu và đóng góp khoa học của đề tài",
      section: "Hành động dữ liệu" as const,
      icon: Download,
      badge: "Tải JSON",
      keywords: ["export", "xuất dữ liệu", "download", "tải", "json", "kết quả"],
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

  // Focus input on open
  useEffect(() => {
    if (isOpen) {
      setSearch("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Keep selected index in bound
  useEffect(() => {
    setSelectedIndex(0);
  }, [search]);

  // Keyboard navigation inside palette
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1 < filtered.length ? prev + 1 : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 >= 0 ? prev - 1 : filtered.length - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        filtered[selectedIndex].action();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 bg-black/60 backdrop-blur-sm p-4 animate-in fade-in-0 duration-150"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Bảng điều khiển lệnh nhanh"
    >
      <div
        className="cmd-dialog relative w-full max-w-xl overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-2xl animate-in zoom-in-95 duration-150 flex flex-col max-h-[80vh]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Input Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border bg-card/40">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            className="flex-1 bg-transparent text-sm font-normal text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
            placeholder="Nhập lệnh hoặc tìm trang (nhấn ↑↓ để chọn, Enter để chạy, Esc để đóng)…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Tìm kiếm lệnh"
          />
          <kbd className="hidden sm:inline-flex items-center px-2 py-0.5 text-[10px] font-mono font-medium rounded border border-border bg-muted/60 text-muted-foreground shadow-2xs">
            Esc
          </kbd>
        </div>

        {/* Command List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1" ref={listRef}>
          {filtered.length === 0 ? (
            <div className="py-10 text-center text-sm font-medium text-muted-foreground">
              Không tìm thấy lệnh hoặc trang nào phù hợp.
            </div>
          ) : (
            filtered.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              const IconComponent = item.icon;
              return (
                <div
                  key={item.id}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs transition-colors cursor-pointer select-none ${
                    isSelected
                      ? "bg-muted/80 text-foreground font-medium shadow-2xs border border-border"
                      : "text-foreground/80 hover:bg-muted/30 border border-transparent"
                  }`}
                  onClick={() => item.action()}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  role="button"
                  tabIndex={0}
                  aria-selected={isSelected}
                >
                  <div
                    className={`flex size-8 shrink-0 items-center justify-center rounded-md border transition-colors ${
                      isSelected
                        ? "border-border bg-card text-foreground shadow-2xs"
                        : "border-border/60 bg-muted/40 text-muted-foreground"
                    }`}
                  >
                    <IconComponent className="size-4" />
                  </div>
                  <div className="flex flex-col flex-1 min-w-0">
                    <div className={`text-xs tracking-tight truncate ${isSelected ? "text-foreground font-semibold" : "text-foreground/90 font-medium"}`}>
                      {item.title}
                    </div>
                    {item.subtitle && (
                      <div className="text-[11px] text-muted-foreground/80 truncate mt-0.5">
                        {item.subtitle}
                      </div>
                    )}
                  </div>
                  {item.badge && (
                    <Badge
                      variant="outline"
                      className={`text-[10px] font-mono shrink-0 px-1.5 py-0.5 ${
                        isSelected
                          ? "bg-card text-foreground font-semibold border-border shadow-2xs"
                          : "bg-muted/40 text-muted-foreground font-medium border-border/50"
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

        {/* Footer Shortcuts */}
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-border bg-card/60 text-[11px] text-muted-foreground font-medium">
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted/60 text-[10px] font-mono font-medium text-foreground shadow-2xs">↑</kbd>
            <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted/60 text-[10px] font-mono font-medium text-foreground shadow-2xs">↓</kbd>
            <span className="ml-1 text-muted-foreground">Di chuyển</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted/60 text-[10px] font-mono font-medium text-foreground shadow-2xs">↵ Enter</kbd>
            <span className="ml-1 text-muted-foreground">Thực hiện</span>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted/60 text-[10px] font-mono font-medium text-foreground shadow-2xs">Esc</kbd>
            <span className="ml-1 text-muted-foreground">Đóng</span>
          </div>
        </div>
      </div>
    </div>
  );
}
