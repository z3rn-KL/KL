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
  Minimize,
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

export interface AtlasNavGroup {
  id: "overview" | "data" | "routing" | "research";
  label: string;
  items: AtlasWorkspaceItem[];
}

export const ATLAS_NAV_GROUPS: AtlasNavGroup[] = [
  {
    id: "overview",
    label: "Tổng quan",
    items: [
      {
        id: "overview",
        to: "/tong-quan",
        title: "Tổng quan",
        shortTitle: "Tổng quan",
        category: "spatial",
        icon: LayoutDashboard,
      },
    ],
  },
  {
    id: "data",
    label: "Dữ liệu",
    items: [
      {
        id: "orders",
        to: "/don-hang",
        title: "Đơn hàng",
        shortTitle: "Đơn hàng",
        category: "spatial",
        icon: Package,
      },
      {
        id: "clusters",
        to: "/phan-cum",
        title: "Phân cụm",
        shortTitle: "Phân cụm",
        category: "spatial",
        icon: Network,
      },
    ],
  },
  {
    id: "routing",
    label: "Định tuyến",
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
        title: "Chạy thử",
        shortTitle: "Chạy thử",
        category: "engine",
        icon: PlayCircle,
      },
    ],
  },
  {
    id: "research",
    label: "Nghiên cứu",
    items: [
      {
        id: "guided-dqn",
        to: "/guided-dqn",
        title: "Guided DQN",
        shortTitle: "Guided DQN",
        category: "rl",
        icon: BrainCircuit,
      },
      {
        id: "compare",
        to: "/so-sanh",
        title: "So sánh",
        shortTitle: "So sánh",
        category: "rl",
        icon: BarChart3,
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

export const ATLAS_WORKSPACES: AtlasWorkspaceItem[] = ATLAS_NAV_GROUPS.flatMap((g) => g.items);

export function AtlasShell({ children }: { children: React.ReactNode }) {
  const [isCmdOpen, setIsCmdOpen] = useState(false);
  const [isPresentation, setIsPresentation] = useState(false);
  const health = useAsync((s) => api.health(s), []);
  const [theme, toggleTheme] = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const navScrollRef = useRef<HTMLDivElement>(null);

  const activeWorkspace = ATLAS_WORKSPACES.find((item) =>
    location.pathname.startsWith(item.to)
  ) || ATLAS_WORKSPACES[0];

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

      if (e.key === "Escape" && isPresentation) {
        e.preventDefault();
        setIsPresentation(false);
        return;
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
  }, [isPresentation, location.pathname, navigate]);

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
                onClick={() => navigate("/tong-quan")}
                className="group flex items-center gap-2 text-left cursor-pointer"
                aria-label="Về trang tổng quan"
              >
                <div className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground shadow-xs">
                  <Radio className="size-3.5 text-primary-foreground" />
                </div>
                <span className="font-mono text-xs font-bold tracking-wider uppercase text-foreground">
                  NEXUS ROUTE
                </span>
              </button>
            </div>

            {/* Center: Clean 4-Category Segmented Navigation */}
            <nav
              aria-label="Điều hướng 4 nhóm phân hệ làm việc"
              className="flex-1 overflow-x-auto no-scrollbar mx-2 py-1"
              ref={navScrollRef}
            >
              <div className="flex items-center gap-1.5 min-w-max">
                {ATLAS_NAV_GROUPS.map((group, gIdx) => (
                  <React.Fragment key={group.id}>
                    {gIdx > 0 && (
                      <div className="h-4 w-px bg-border/60 shrink-0 mx-0.5" aria-hidden="true" />
                    )}
                    <div className="flex items-center gap-1 bg-muted/30 px-1 py-0.5 rounded-lg border border-border/40">
                      <span className="hidden 2xl:inline text-[9px] font-bold text-muted-foreground/75 uppercase tracking-wider px-1 select-none">
                        {group.label}
                      </span>
                      {group.items.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname.startsWith(item.to);

                        return (
                          <NavLink
                            key={item.id}
                            to={item.to}
                            className={`atlas-nav-item flex h-7 items-center gap-1.5 rounded-md px-2 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                              isActive
                                ? "active bg-primary text-primary-foreground font-semibold shadow-xs"
                                : "text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                            }`}
                          >
                            <Icon className="size-3.5 shrink-0" />
                            <span>{item.title}</span>
                          </NavLink>
                        );
                      })}
                    </div>
                  </React.Fragment>
                ))}
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
                <kbd className="hidden sm:inline-flex items-center rounded border border-border bg-muted/60 px-1.5 py-0.2 font-mono text-[10px] font-semibold text-muted-foreground">
                  Ctrl K
                </kbd>
              </button>

              {/* Status Pill (Quiet, Static Dot, No blinking) */}
              <div
                className="hidden sm:flex h-8 items-center gap-1.5 rounded-md border border-border/80 bg-background/60 px-2 text-xs font-medium text-muted-foreground"
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
      <main className="flex-1 min-w-0 w-full">
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
