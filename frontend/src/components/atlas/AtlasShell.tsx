import React, { useState, useEffect } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Network,
  BrainCircuit,
  Truck,
  FileText,
  BarChart3,
  Sun,
  Moon,
  Maximize2,
  Minimize2,
  Cpu,
  Layers,
} from "lucide-react";
import { useTheme } from "@/lib/theme";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { LiveClock } from "@/components/shared/LiveClock";

export interface NavItem {
  id: string;
  to: string;
  title: string;
  shortTitle: string;
  step: number;
  icon: React.ElementType;
}

export const OFFICIAL_NAV_ITEMS: NavItem[] = [
  {
    id: "overview",
    to: "/tong-quan",
    title: "Tổng quan",
    shortTitle: "Tổng quan",
    step: 1,
    icon: LayoutDashboard,
  },
  {
    id: "data-clustering",
    to: "/phan-cum",
    title: "Dữ liệu & Phân cụm",
    shortTitle: "Phân cụm",
    step: 2,
    icon: Network,
  },
  {
    id: "rl-optimization",
    to: "/toi-uu-rl",
    title: "Tối ưu tuyến bằng RL",
    shortTitle: "Tối ưu RL",
    step: 3,
    icon: BrainCircuit,
  },
  {
    id: "simulation",
    to: "/mo-phong",
    title: "Mô phỏng giao hàng",
    shortTitle: "Mô phỏng",
    step: 4,
    icon: Truck,
  },
  {
    id: "shift-report",
    to: "/bao-cao",
    title: "Báo cáo kết ca",
    shortTitle: "Báo cáo",
    step: 5,
    icon: FileText,
  },
  {
    id: "experiments",
    to: "/thuc-nghiem",
    title: "Thực nghiệm & So sánh",
    shortTitle: "Thực nghiệm",
    step: 6,
    icon: BarChart3,
  },
];

export function AtlasShell({ children }: { children: React.ReactNode }) {
  const [isPresentation, setIsPresentation] = useState(false);
  const health = useAsync((s) => api.health(s), []);
  const [theme, toggleTheme] = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  // Keyboard navigation for presentation mode and fast route switching
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isInput =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);

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
        const curIdx = OFFICIAL_NAV_ITEMS.findIndex((n) => location.pathname.startsWith(n.to));
        if (e.key === "ArrowRight") {
          e.preventDefault();
          const nextIdx = (curIdx + 1) % OFFICIAL_NAV_ITEMS.length;
          navigate(OFFICIAL_NAV_ITEMS[nextIdx].to);
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          const prevIdx = (curIdx - 1 + OFFICIAL_NAV_ITEMS.length) % OFFICIAL_NAV_ITEMS.length;
          navigate(OFFICIAL_NAV_ITEMS[prevIdx].to);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPresentation, location.pathname, navigate]);

  const isHealthy = !health.loading && !health.error && health.data?.core_artifacts_ready;
  const isConnecting = health.loading;

  const currentNav = OFFICIAL_NAV_ITEMS.find((item) =>
    location.pathname.startsWith(item.to)
  ) || OFFICIAL_NAV_ITEMS[0];

  return (
    <div className={`atlas-app shell min-h-screen flex flex-col bg-background text-foreground antialiased selection:bg-teal-500/20 ${isPresentation ? "presentation-mode" : ""}`}>
      {/* ============================================================== */}
      {/* PRECISION APP HEADER (Inspired by Linear & Vercel Geist)      */}
      {/* ============================================================== */}
      {!isPresentation && (
        <header className="sticky top-0 z-40 w-full h-14 border-b border-border/70 bg-card/90 backdrop-blur-md transition-colors">
          <div className="mx-auto flex h-full max-w-[1780px] items-center justify-between px-3 md:px-6 gap-3">
            {/* Left: Brand & Title */}
            <div className="flex items-center gap-3 shrink-0">
              <button
                type="button"
                onClick={() => navigate("/tong-quan")}
                className="group flex items-center gap-2.5 text-left cursor-pointer focus:outline-none"
                aria-label="Về màn hình Tổng quan"
              >
                <div className="flex size-8 items-center justify-center rounded-lg bg-teal-600 text-white shadow-xs group-hover:bg-teal-700 transition-colors">
                  <Layers className="size-4.5 text-white" />
                </div>
                <div className="flex flex-col">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-sm font-black tracking-wider uppercase text-foreground leading-none">
                      NEXUS
                    </span>
                    <span className="text-[10px] font-sans font-bold tracking-wide text-teal-700 dark:text-teal-300 bg-teal-500/10 px-1.5 py-0.5 rounded border border-teal-500/20 leading-none">
                      Delivery Intelligence
                    </span>
                  </div>
                  <span className="text-[10px] font-medium text-muted-foreground leading-none mt-1 hidden sm:inline">
                    Tối ưu &amp; điều phối giao hàng
                  </span>
                </div>
              </button>
            </div>

            {/* Center: EXACT 6 MAIN NAVIGATION ITEMS */}
            <nav
              aria-label="Điều hướng chính khóa luận"
              className="flex items-center gap-1 overflow-x-auto no-scrollbar py-1"
            >
              <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-xl border border-border/50">
                {OFFICIAL_NAV_ITEMS.map((item) => {
                  const Icon = item.icon;
                  const isActive =
                    location.pathname.startsWith(item.to) ||
                    (item.to === "/tong-quan" && (location.pathname === "/" || location.pathname === "/overview"));

                  return (
                    <NavLink
                      key={item.id}
                      to={item.to}
                      className={`flex h-8 items-center gap-1.5 rounded-lg px-2.5 lg:px-3 text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                        isActive
                          ? "bg-teal-600 text-white shadow-xs font-bold"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                      }`}
                    >
                      <span className={`inline-flex items-center justify-center size-4 rounded-full text-[10px] font-bold ${
                        isActive ? "bg-white/20 text-white" : "bg-muted text-muted-foreground/80"
                      }`}>
                        {item.step}
                      </span>
                      <Icon className="size-3.5 shrink-0" />
                      <span className="hidden md:inline">{item.shortTitle}</span>
                    </NavLink>
                  );
                })}
              </div>
            </nav>

            {/* Right: Health Status, Checkpoint info, Fullscreen & Theme */}
            <div className="flex items-center gap-2 shrink-0">
              {/* Real-time System Clock (Asia/Ho_Chi_Minh • UTC+7) */}
              <LiveClock className="hidden sm:inline-flex h-8" />

              {/* Backend Status Dot */}
              <div
                className="hidden lg:flex h-8 items-center gap-1.5 rounded-lg border border-border/70 bg-background/60 px-2.5 text-xs font-medium text-muted-foreground"
                title={
                  isConnecting
                    ? "Đang kết nối FastAPI (127.0.0.1:8011)..."
                    : isHealthy
                    ? "Backend sẵn sàng (Port 8011) — PyTorch Model Checkpoint đã khóa"
                    : "Backend ngoại tuyến — Chế độ dự phòng"
                }
              >
                <span
                  className={`size-2 rounded-full shrink-0 ${
                    isHealthy
                      ? "bg-emerald-500 shadow-xs"
                      : isConnecting
                      ? "bg-amber-400 animate-ping"
                      : "bg-rose-500"
                  }`}
                />
                <span className="text-xs font-medium text-foreground/85">
                  {isConnecting ? "Đang kết nối…" : isHealthy ? "Backend Sẵn Sàng" : "Mô phỏng Offline"}
                </span>
              </div>

              {/* Presentation Mode Toggle (F) */}
              <button
                type="button"
                onClick={() => setIsPresentation((prev) => !prev)}
                className="flex size-8 items-center justify-center rounded-lg border border-border/70 bg-background/70 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground cursor-pointer"
                title="Chế độ trình chiếu toàn màn hình (Phím F)"
                aria-label="Chế độ trình chiếu"
              >
                <Maximize2 className="size-3.5" />
              </button>

              {/* Theme Toggle Button */}
              <button
                type="button"
                onClick={toggleTheme}
                className="flex size-8 items-center justify-center rounded-lg border border-border/70 bg-background/70 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground cursor-pointer"
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

      {/* Floating Exit Button for Presentation Mode */}
      {isPresentation && (
        <div className="fixed top-3 right-4 z-50 flex items-center gap-2">
          <div className="bg-background/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-border/80 text-xs font-medium shadow-lg">
            Màn hình {currentNav.step}/6: <strong className="text-teal-600 dark:text-teal-400">{currentNav.title}</strong> (Dùng phím ← → để chuyển trang)
          </div>
          <button
            type="button"
            onClick={() => setIsPresentation(false)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 text-white font-semibold text-xs shadow-lg hover:bg-teal-700 transition-colors cursor-pointer"
            title="Thoát trình chiếu (Esc)"
          >
            <Minimize2 className="size-3.5" />
            <span>Thoát</span>
          </button>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-[1780px] mx-auto p-3 sm:p-4 lg:p-4 flex flex-col min-h-0 transition-all">
        {children}
      </main>
    </div>
  );
}
