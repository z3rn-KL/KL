import React, { useState, useEffect } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarFooter,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  LayoutDashboard,
  Package,
  Network,
  Map,
  PlayCircle,
  BrainCircuit,
  BarChart3,
  Sliders,
  ShieldCheck,
  FileCode,
  Search,
  Sun,
  Moon,
  Maximize2,
  Minimize2,
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Radio,
  Clock,
  Sparkles,
} from "lucide-react";
import { useTheme } from "@/lib/theme";
import { api, Health } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { CommandPalette } from "@/components/CommandPalette";

export interface NavItemConfig {
  to: string;
  label: string;
  shortLabel: string;
  num: number;
  icon: React.ElementType;
  badge?: string;
  accent?: boolean;
}

export interface NavGroupConfig {
  id: string;
  label: string;
  items: NavItemConfig[];
}

export const NAVIGATION_STRUCTURE: NavGroupConfig[] = [
  {
    id: "overview",
    label: "TỔNG QUAN & DỮ LIỆU KHÔNG GIAN",
    items: [
      {
        to: "/tong-quan",
        label: "Tổng quan hệ thống",
        shortLabel: "Tổng quan",
        num: 1,
        icon: LayoutDashboard,
      },
      {
        to: "/don-hang",
        label: "Quản lý đơn hàng",
        shortLabel: "Đơn hàng",
        num: 2,
        icon: Package,
      },
      {
        to: "/phan-cum",
        label: "Phân cụm không gian",
        shortLabel: "Phân cụm",
        num: 3,
        icon: Network,
      },
    ],
  },
  {
    id: "gis_routing",
    label: "VẬN HÀNH & MÔ PHỎNG TUYẾN",
    items: [
      {
        to: "/tuyen-giao",
        label: "Bản đồ tuyến giao hàng",
        shortLabel: "Bản đồ tuyến GIS",
        num: 4,
        icon: Map,
      },
      {
        to: "/chay-thu",
        label: "Thử nghiệm định tuyến",
        shortLabel: "Thử nghiệm tuyến",
        num: 5,
        icon: PlayCircle,
      },
    ],
  },
  {
    id: "ai_rl",
    label: "MÔ HÌNH HỌC TĂNG CƯỜNG (RL)",
    items: [
      {
        to: "/guided-dqn",
        label: "Nghiên cứu học tăng cường",
        shortLabel: "Nghiên cứu RL",
        num: 6,
        icon: BrainCircuit,
        badge: "RL",
        accent: true,
      },
      {
        to: "/so-sanh",
        label: "So sánh thuật toán định tuyến",
        shortLabel: "So sánh thuật toán",
        num: 7,
        icon: BarChart3,
      },
    ],
  },
  {
    id: "evaluation",
    label: "THÍ NGHIỆM CHUYÊN SÂU & ĐỘ TIN CẬY",
    items: [
      {
        to: "/do-nhay",
        label: "Phân tích độ nhạy hệ số phạt λ",
        shortLabel: "Phân tích độ nhạy",
        num: 8,
        icon: Sliders,
      },
      {
        to: "/on-dinh",
        label: "Độ ổn định và siêu tham số",
        shortLabel: "Độ ổn định",
        num: 9,
        icon: ShieldCheck,
      },
    ],
  },
];

export const ALL_NAV_ITEMS = NAVIGATION_STRUCTURE.flatMap((g) => g.items);

function LiveClockDisplay() {
  const [time, setTime] = useState(() => new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeStr = time.toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  return (
    <div className="hidden h-8 items-center gap-1.5 px-2.5 text-xs font-medium text-foreground/90 bg-card/80 rounded-md border border-border/80 shadow-2xs lg:flex">
      <Clock className="size-3.5 text-primary" />
      <span className="font-mono text-xs font-semibold text-foreground tabular-nums tracking-tight">{timeStr}</span>
      <span className="text-[10px] font-mono font-semibold text-muted-foreground ml-0.5">ICT</span>
    </div>
  );
}

function AppSidebar({ health }: { health: ReturnType<typeof useAsync<Health>> }) {
  const location = useLocation();
  const navigate = useNavigate();

  const isHealthy = !health.loading && !health.error && health.data?.core_artifacts_ready;
  const isConnecting = health.loading;

  return (
    <Sidebar variant="sidebar" collapsible="icon" className="border-r border-border/80 bg-sidebar">
      {/* Brand Header */}
      <SidebarHeader className="border-b border-sidebar-border px-3.5 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-xs">
            <Radio className="size-4 animate-pulse text-white" />
          </div>
          <div className="flex flex-col overflow-hidden text-left group-data-[collapsible=icon]:hidden">
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-bold tracking-tight text-sidebar-foreground">NEXUS ROUTE</span>
              <Badge variant="outline" className="px-1 py-0 text-[9px] font-mono font-medium text-primary border-primary/30">
                KLTN-11
              </Badge>
            </div>
            <span className="text-[11px] font-medium text-muted-foreground/90 truncate">
              Tối ưu tuyến giao hàng
            </span>
          </div>
        </div>
      </SidebarHeader>

      {/* Navigation Groups */}
      <SidebarContent className="px-2 py-2">
        {NAVIGATION_STRUCTURE.map((group) => (
          <SidebarGroup key={group.id} className="py-1.5">
            <SidebarGroupLabel className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 px-2 group-data-[collapsible=icon]:hidden">
              {group.label}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname.startsWith(item.to);
                  return (
                    <SidebarMenuItem key={item.to}>
                      <SidebarMenuButton
                        isActive={isActive}
                        tooltip={item.label}
                        onClick={() => navigate(item.to)}
                        className={`h-9 px-2.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                          isActive
                            ? "bg-accent font-semibold text-accent-foreground shadow-xs"
                            : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                        }`}
                      >
                        <Icon className={`size-4 shrink-0 ${isActive ? "text-primary" : "text-muted-foreground"}`} />
                        <span className="truncate group-data-[collapsible=icon]:hidden flex-1">
                          {item.label}
                        </span>
                        {item.badge && (
                          <Badge
                            variant="outline"
                            className="ml-auto px-1 py-0 text-[10px] font-mono font-semibold bg-primary/5 text-primary border-primary/20 group-data-[collapsible=icon]:hidden"
                          >
                            {item.badge}
                          </Badge>
                        )}
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      {/* Sidebar Footer with System Status */}
      <SidebarFooter className="border-t border-sidebar-border p-2">
        <div className="flex flex-col gap-1.5">
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors group-data-[collapsible=icon]:justify-center"
            title="FastAPI Swagger Documentation"
          >
            <FileCode className="size-4 shrink-0" />
            <span className="truncate group-data-[collapsible=icon]:hidden">FastAPI Swagger</span>
          </a>

          <div className="flex items-center justify-between rounded-md bg-muted/40 p-2 text-xs group-data-[collapsible=icon]:p-1.5">
            <div className="flex items-center gap-2 overflow-hidden">
              <span className="relative flex size-2 shrink-0">
                {isHealthy ? (
                  <>
                    <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
                  </>
                ) : isConnecting ? (
                  <span className="relative inline-flex size-2 rounded-full bg-amber-400 animate-pulse" />
                ) : (
                  <span className="relative inline-flex size-2 rounded-full bg-rose-500" />
                )}
              </span>
              <span className="truncate font-medium text-[11px] group-data-[collapsible=icon]:hidden">
                {isConnecting ? "Kiểm tra API…" : isHealthy ? "Backend Trực tuyến" : "Mất kết nối API"}
              </span>
            </div>
            {isHealthy && (
              <Badge variant="outline" className="text-[10px] font-mono px-1 py-0 group-data-[collapsible=icon]:hidden">
                v12.2
              </Badge>
            )}
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}

function AppHeader({
  onOpenCmd,
  isPresentation,
  onTogglePresentation,
  health,
}: {
  onOpenCmd: () => void;
  isPresentation: boolean;
  onTogglePresentation: () => void;
  health: ReturnType<typeof useAsync<Health>>;
}) {
  const location = useLocation();
  const [theme, toggleTheme] = useTheme();

  // Find current item & group
  let currentGroup = "TỔNG QUAN";
  let currentItem = ALL_NAV_ITEMS[0];

  for (const group of NAVIGATION_STRUCTURE) {
    const found = group.items.find((item) => location.pathname.startsWith(item.to));
    if (found) {
      currentGroup = group.label;
      currentItem = found;
      break;
    }
  }

  const isHealthy = !health.loading && !health.error && health.data?.core_artifacts_ready;

  return (
    <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center justify-between gap-2 border-b border-border/80 bg-background/95 px-4 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <SidebarTrigger className="size-8 rounded-md border border-border/60 hover:bg-muted transition-colors" />
        <Separator orientation="vertical" className="h-4" />

        {/* Clean Breadcrumb */}
        <Breadcrumb className="hidden sm:flex">
          <BreadcrumbList className="text-xs">
            <BreadcrumbItem>
              <span className="text-muted-foreground/70 font-medium">{currentGroup}</span>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold text-foreground">
                {currentItem.label}
              </BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>

      <div className="flex items-center gap-2">
        <LiveClockDisplay />

        {/* Global Search / Command Trigger */}
        <Button
          variant="outline"
          size="sm"
          onClick={onOpenCmd}
          className="h-8 gap-2 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground bg-card border-border/80 shadow-2xs hover:bg-accent/60"
        >
          <Search className="size-3.5 text-muted-foreground" />
          <span className="hidden md:inline font-medium">Tìm lệnh & điều hướng…</span>
          <kbd className="hidden md:inline-flex items-center rounded border border-border/80 bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-muted-foreground shadow-2xs">
            Ctrl K
          </kbd>
        </Button>

        {/* Presentation Toggle */}
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant={isPresentation ? "secondary" : "ghost"}
                size="icon"
                onClick={onTogglePresentation}
                className="size-8"
                aria-label="Chế độ trình chiếu"
              >
                {isPresentation ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
              </Button>
            }
          />
          <TooltipContent side="bottom">
            {isPresentation ? "Thoát trình chiếu (Esc / F)" : "Chế độ trình chiếu (F)"}
          </TooltipContent>
        </Tooltip>

        {/* Theme Toggle */}
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                className="size-8"
                aria-label="Đổi giao diện sáng/tối"
              >
                {theme === "dark" ? <Sun className="size-4 text-amber-400" /> : <Moon className="size-4 text-slate-700" />}
              </Button>
            }
          />
          <TooltipContent side="bottom">
            {theme === "dark" ? "Chuyển sang giao diện Sáng" : "Chuyển sang giao diện Tối"}
          </TooltipContent>
        </Tooltip>
      </div>
    </header>
  );
}

export function AppLayout({ children }: { children: React.ReactNode }) {
  const [isCmdOpen, setIsCmdOpen] = useState(false);
  const [isPresentation, setIsPresentation] = useState(false);
  const health = useAsync((s) => api.health(s), []);
  const [theme, toggleTheme] = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  // Keyboard shortcut listener
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
        const curIdx = ALL_NAV_ITEMS.findIndex((n) => location.pathname.startsWith(n.to));
        if (e.key === "ArrowRight") {
          e.preventDefault();
          const nextIdx = (curIdx + 1) % ALL_NAV_ITEMS.length;
          navigate(ALL_NAV_ITEMS[nextIdx].to);
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          const prevIdx = (curIdx - 1 + ALL_NAV_ITEMS.length) % ALL_NAV_ITEMS.length;
          navigate(ALL_NAV_ITEMS[prevIdx].to);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPresentation, location.pathname, navigate]);

  return (
    <SidebarProvider defaultOpen={true}>
      <div className={`shell flex min-h-screen w-full bg-background ${isPresentation ? "presentation-mode" : ""}`}>
        {!isPresentation && <AppSidebar health={health} />}
        <div className="flex flex-1 flex-col overflow-hidden min-w-0">
          {!isPresentation && (
            <AppHeader
              onOpenCmd={() => setIsCmdOpen(true)}
              isPresentation={isPresentation}
              onTogglePresentation={() => setIsPresentation((p) => !p)}
              health={health}
            />
          )}
          <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
            <div className="mx-auto max-w-[1600px] w-full">{children}</div>
          </main>
        </div>

        {/* Command Palette */}
        <CommandPalette
          isOpen={isCmdOpen}
          onClose={() => setIsCmdOpen(false)}
          onToggleTheme={toggleTheme}
          isDarkTheme={theme === "dark"}
          onToggleEffects={() => {}}
          isEffectsOn={true}
          onTogglePresentation={() => setIsPresentation((p) => !p)}
          isPresentationMode={isPresentation}
          navItems={ALL_NAV_ITEMS.map((n) => ({
            to: n.to,
            label: n.label,
            shortLabel: n.shortLabel,
            num: n.num,
            icon: n.icon,
          }))}
        />
      </div>
    </SidebarProvider>
  );
}
