import React, { useState, useEffect, useRef } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Package,
  Network,
  MapPin,
  PlayCircle,
  BrainCircuit,
  BarChart3,
  Sliders,
  ShieldCheck,
  Search,
  Sun,
  Moon,
  Maximize2,
  Minimize2,
  Radio,
  Truck,
  Sparkles,
  Database,
  ChevronDown,
  Layers,
} from "lucide-react";
import { useTheme } from "@/lib/theme";
import { api, Health } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { AtlasCommandPalette } from "./AtlasCommandPalette";

export interface AtlasWorkspaceItem {
  id: string;
  to: string;
  title: string;
  shortTitle: string;
  category: "spatial" | "engine" | "rl" | "validation";
  icon: React.ElementType;
}

export const CORE_PILLARS: AtlasWorkspaceItem[] = [
  {
    id: "dispatch",
    to: "/dieu-phoi",
    title: "Điều phối",
    shortTitle: "Điều phối",
    category: "engine",
    icon: Truck,
  },
  {
    id: "orders",
    to: "/don-hang",
    title: "Đơn hàng",
    shortTitle: "Đơn hàng",
    category: "spatial",
    icon: Package,
  },
  {
    id: "research",
    to: "/nghien-cuu",
    title: "Phân tích",
    shortTitle: "Phân tích",
    category: "rl",
    icon: BarChart3,
  },
];

export interface SpecializedGroup {
  id: string;
  label: string;
  items: AtlasWorkspaceItem[];
}

export const SPECIALIZED_MODULE_GROUPS: SpecializedGroup[] = [
  {
    id: "gis-ops",
    label: "Vận hành GIS & Mô phỏng",
    items: [
      {
        id: "routes",
        to: "/tuyen-giao",
        title: "Tuyến giao",
        shortTitle: "Tuyến giao",
        category: "engine",
        icon: MapPin,
      },
      {
        id: "run-lab",
        to: "/chay-thu",
        title: "Chạy thử (Run Lab)",
        shortTitle: "Chạy thử",
        category: "engine",
        icon: PlayCircle,
      },
      {
        id: "daily-planning",
        to: "/ke-hoach-ngay",
        title: "Kế hoạch ca",
        shortTitle: "Kế hoạch ca",
        category: "engine",
        icon: Truck,
      },
      {
        id: "real-data",
        to: "/du-lieu-that",
        title: "Dữ liệu thật",
        shortTitle: "Dữ liệu thật",
        category: "spatial",
        icon: Database,
      },
      {
        id: "overview",
        to: "/tong-quan",
        title: "Tổng quan GIS",
        shortTitle: "Tổng quan",
        category: "spatial",
        icon: LayoutDashboard,
      },
    ],
  },
  {
    id: "algorithms-rl",
    label: "Thuật toán & Kiểm chứng",
    items: [
      {
        id: "clusters",
        to: "/phan-cum",
        title: "Phân cụm K-Means",
        shortTitle: "Phân cụm",
        category: "spatial",
        icon: Network,
      },
      {
        id: "proposed-rl",
        to: "/proposed-rl",
        title: "Proposed RL",
        shortTitle: "Proposed RL",
        category: "rl",
        icon: Sparkles,
      },
      {
        id: "compare",
        to: "/so-sanh",
        title: "So sánh Cohort",
        shortTitle: "So sánh",
        category: "rl",
        icon: BarChart3,
      },
      {
        id: "guided-dqn",
        to: "/guided-dqn",
        title: "Guided DQN",
        shortTitle: "Guided DQN",
        category: "rl",
        icon: BrainCircuit,
      },
      {
        id: "sensitivity",
        to: "/do-nhay",
        title: "Độ nhạy",
        shortTitle: "Độ nhạy",
        category: "validation",
        icon: Sliders,
      },
      {
        id: "stability",
        to: "/on-dinh",
        title: "Độ ổn định",
        shortTitle: "Độ ổn định",
        category: "validation",
        icon: ShieldCheck,
      },
    ],
  },
];

export const ATLAS_WORKSPACES: AtlasWorkspaceItem[] = [
  ...CORE_PILLARS,
  ...SPECIALIZED_MODULE_GROUPS.flatMap((g) => g.items),
];

export function AtlasShell({ children }: { children: React.ReactNode }) {
  const [isCmdOpen, setIsCmdOpen] = useState(false);
  const [isPresentation, setIsPresentation] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const health = useAsync((s) => api.health(s), []);
  const [theme, toggleTheme] = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  const activeWorkspace = ATLAS_WORKSPACES.find((item) =>
    location.pathname.startsWith(item.to)
  ) || ATLAS_WORKSPACES[0];

  const activeSpecializedItem = SPECIALIZED_MODULE_GROUPS.flatMap((g) => g.items).find(
    (item) => location.pathname.startsWith(item.to)
  );

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isInput =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsCmdOpen((prev) => !prev);
        return;
      }

      if (isInput) return;

      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        setIsPresentation((prev) => !prev);
        return;
      }

      if (e.key === "Escape") {
        if (isDropdownOpen) {
          setIsDropdownOpen(false);
          return;
        }
        if (isPresentation) {
          e.preventDefault();
          setIsPresentation(false);
          return;
        }
      }

      if (isPresentation) {
        const curIdx = ATLAS_WORKSPACES.findIndex((n) => location.pathname.startsWith(n.to));
        if (e.key === "ArrowRight") {
          e.preventDefault();
          const nextIdx = (curIdx + 1) % ATLAS_WORKSPACES.length;
          navigate(ATLAS_WORKSPACES[nextIdx].to);
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          const prevIdx = (curIdx - 1 + ATLAS_WORKSPACES.length) % ATLAS_WORKSPACES.length;
          navigate(ATLAS_WORKSPACES[prevIdx].to);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPresentation, isDropdownOpen, location.pathname, navigate]);

  const isHealthy = !health.loading && !health.error && health.data?.core_artifacts_ready;
  const isConnecting = health.loading;

  return (
    <div className={`atlas-app shell min-h-screen flex flex-col bg-background text-foreground antialiased selection:bg-primary/20 ${isPresentation ? "presentation-mode" : ""}`}>
      {/* ============================================================== */}
      {/* SINGLE COMPACT APP HEADER (~56px) */}
      {/* ============================================================== */}
      {!isPresentation && (
        <header className="sticky top-0 z-40 w-full h-14 border-b border-border/80 bg-card/95 backdrop-blur-md">
          <div className="mx-auto flex h-full max-w-[1720px] items-center justify-between px-3 md:px-6 gap-3">
            {/* Left: Brand Logo & Title */}
            <div className="flex items-center gap-2.5 shrink-0">
              <button
                type="button"
                onClick={() => navigate("/dieu-phoi")}
                className="group flex items-center gap-2 text-left cursor-pointer"
                aria-label="Về trang điều phối"
              >
                <div className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-xs">
                  <Radio className="size-3.5 text-primary-foreground" />
                </div>
                <div className="hidden sm:flex flex-col">
                  <span className="font-mono text-xs font-bold tracking-wider uppercase text-foreground leading-none">
                    NEXUS DELIVERY
                  </span>
                  <span className="text-[9px] font-mono font-semibold tracking-widest uppercase text-muted-foreground leading-none mt-0.5">
                    INTELLIGENCE
                  </span>
                </div>
              </button>
            </div>

            {/* Center: 3 Primary Pillars + Specialized Modules Dropdown */}
            <nav
              aria-label="Điều hướng chính"
              className="flex items-center gap-2"
            >
              <div className="flex items-center gap-1 bg-muted/40 p-0.5 rounded-lg border border-border/50">
                {CORE_PILLARS.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname.startsWith(item.to);

                  return (
                    <NavLink
                      key={item.id}
                      to={item.to}
                      className={`atlas-nav-item flex h-8 items-center gap-1.5 rounded-md px-2.5 sm:px-3 text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                        isActive
                          ? "active bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                      }`}
                    >
                      <Icon className="size-3.5 shrink-0" />
                      <span className="hidden sm:inline">{item.title}</span>
                    </NavLink>
                  );
                })}
              </div>

              {/* Specialized Modules Dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  type="button"
                  onClick={() => setIsDropdownOpen((prev) => !prev)}
                  className={`flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs font-medium transition-colors cursor-pointer ${
                    activeSpecializedItem
                      ? "border-primary/50 bg-primary/10 text-primary font-semibold"
                      : "border-border/70 bg-muted/30 text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                  aria-expanded={isDropdownOpen}
                  aria-haspopup="true"
                  title="Khám phá 11 chuyên trang GIS & Nghiên cứu nâng cao"
                >
                  <Layers className="size-3.5 shrink-0" />
                  <span className="hidden lg:inline">
                    {activeSpecializedItem ? `Chuyên mục: ${activeSpecializedItem.title}` : "Phân hệ chuyên sâu"}
                  </span>
                  <ChevronDown className={`size-3 transition-transform ${isDropdownOpen ? "rotate-180" : ""}`} />
                </button>

                {isDropdownOpen && (
                  <div className="absolute left-0 mt-1.5 w-80 md:w-96 rounded-xl border border-border/80 bg-card/98 p-3 shadow-xl backdrop-blur-lg z-50 animate-in fade-in zoom-in-95">
                    <div className="space-y-3">
                      {SPECIALIZED_MODULE_GROUPS.map((group) => (
                        <div key={group.id} className="space-y-1">
                          <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-2">
                            {group.label}
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-1">
                            {group.items.map((item) => {
                              const Icon = item.icon;
                              const isActive = location.pathname.startsWith(item.to);
                              return (
                                <button
                                  key={item.id}
                                  type="button"
                                  onClick={() => {
                                    navigate(item.to);
                                    setIsDropdownOpen(false);
                                  }}
                                  className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-left transition-colors cursor-pointer ${
                                    isActive
                                      ? "bg-primary text-primary-foreground font-semibold"
                                      : "text-foreground hover:bg-muted"
                                  }`}
                                >
                                  <Icon className="size-3.5 shrink-0 opacity-70" />
                                  <span className="truncate">{item.title}</span>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </nav>

            {/* Right: Actions, Search, Health & Theme */}
            <div className="flex items-center gap-2 shrink-0">
              {/* Global Search / Command Trigger */}
              <button
                type="button"
                onClick={() => setIsCmdOpen(true)}
                className="flex h-8 items-center gap-2 rounded-md border border-border/80 bg-background/80 px-2.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-muted hover:text-foreground cursor-pointer"
                aria-label="Đi đến trang hoặc lệnh (Ctrl+K)"
              >
                <Search className="size-3.5" />
                <span className="hidden xl:inline font-medium">Đi đến trang hoặc lệnh…</span>
                <kbd className="hidden lg:inline-flex items-center rounded border border-border bg-muted/60 px-1.5 py-0.2 font-mono text-[10px] font-semibold text-muted-foreground">
                  Ctrl K
                </kbd>
              </button>

              {/* Status Pill (Quiet, Static Dot, No blinking) */}
              <div
                className="hidden lg:flex h-8 items-center gap-1.5 rounded-md border border-border/80 bg-background/60 px-2 text-xs font-medium text-muted-foreground"
                title={
                  isConnecting
                    ? "Đang thăm dò trạng thái FastAPI Backend..."
                    : isHealthy
                    ? "Backend sẵn sàng — Core Artifacts sẵn sàng"
                    : "Backend ngoại tuyến — Chế độ mô phỏng từ Checkpoints"
                }
              >
                <span
                  className={`size-2 rounded-full shrink-0 ${
                    isHealthy
                      ? "bg-emerald-500"
                      : isConnecting
                      ? "bg-amber-400"
                      : "bg-rose-500"
                  }`}
                />
                <span className="text-[11px] font-medium text-foreground/80">
                  {isConnecting ? "Kiểm tra…" : isHealthy ? "Backend sẵn sàng" : "Chế độ Offline"}
                </span>
              </div>

              {/* Presentation Mode Toggle */}
              <button
                type="button"
                onClick={() => setIsPresentation((prev) => !prev)}
                className="flex size-8 items-center justify-center rounded-md border border-border/80 bg-background/80 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground cursor-pointer"
                title="Chế độ trình chiếu toàn màn hình (Phím F)"
                aria-label="Chế độ trình chiếu"
              >
                <Maximize2 className="size-3.5" />
              </button>

              {/* Theme Toggle Button */}
              <button
                type="button"
                onClick={toggleTheme}
                className="flex size-8 items-center justify-center rounded-md border border-border/80 bg-background/80 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground cursor-pointer"
                title={theme === "dark" ? "Chuyển sang giao diện Sáng" : "Chuyển sang giao diện Tối"}
                aria-label="Đổi giao diện sáng/tối"
              >
                {theme === "dark" ? (
                  <Sun className="size-3.5 text-amber-400" />
                ) : (
                  <Moon className="size-3.5 text-slate-700" />
                )}
              </button>
            </div>
          </div>
        </header>
      )}

      {/* Presentation Mode Floating Return Pill */}
      {isPresentation && (
        <div className="fixed top-3 right-3 z-50 flex items-center gap-2 rounded-full border border-border/80 bg-card/90 px-3 py-1.5 shadow-lg backdrop-blur-md">
          <span className="font-mono text-xs font-semibold text-primary">
            {activeWorkspace.title}
          </span>
          <span className="text-[11px] text-muted-foreground">| Phím mũi tên ← → chuyển trang</span>
          <button
            type="button"
            onClick={() => setIsPresentation(false)}
            className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
          >
            <Minimize2 className="size-3" />
            <span>Thoát (Esc)</span>
          </button>
        </div>
      )}

      {/* Main Workspace Content */}
      <main className="flex-1 min-w-0 w-full overflow-x-clip">
        {children}
      </main>

      {/* Command Palette Modal */}
      <AtlasCommandPalette
        isOpen={isCmdOpen}
        onClose={() => setIsCmdOpen(false)}
        onToggleTheme={toggleTheme}
        isDarkTheme={theme === "dark"}
        onTogglePresentation={() => setIsPresentation((p) => !p)}
        isPresentationMode={isPresentation}
        workspaces={ATLAS_WORKSPACES}
      />
    </div>
  );
}
