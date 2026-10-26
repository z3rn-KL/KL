import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TrendingUp, TrendingDown, Info, ArrowUpRight, RotateCcw, AlertCircle } from "lucide-react";
import { ALGO_COLOR, ALGO_LABEL } from "@/lib/format";

export interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ElementType;
  trend?: {
    value: string;
    isPositive?: boolean;
    isNeutral?: boolean;
  };
  badge?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  trend,
  badge,
  className = "",
}: MetricCardProps) {
  return (
    <Card className={`overflow-hidden border-border/80 shadow-xs bg-card transition-all hover:border-border hover:shadow-sm ${className}`}>
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground truncate">
            {label}
          </span>
          <div className="flex items-center gap-1.5 shrink-0">
            {badge && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-medium">
                {badge}
              </Badge>
            )}
            {Icon && (
              <div className="flex size-7 items-center justify-center rounded-md bg-muted/60 text-muted-foreground">
                <Icon className="size-3.5" />
              </div>
            )}
          </div>
        </div>

        <div className="mt-2.5 flex items-baseline justify-between gap-2">
          <div className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl font-mono tabular-nums">
            {value}
          </div>
          {trend && (
            <div
              className={`flex items-center text-xs font-semibold ${
                trend.isNeutral
                  ? "text-muted-foreground"
                  : trend.isPositive
                  ? "text-emerald-600 dark:text-emerald-400"
                  : "text-rose-600 dark:text-rose-400"
              }`}
            >
              {trend.isPositive ? (
                <TrendingUp className="mr-0.5 size-3" />
              ) : trend.isNeutral ? null : (
                <TrendingDown className="mr-0.5 size-3" />
              )}
              <span>{trend.value}</span>
            </div>
          )}
        </div>

        {hint && (
          <div className="mt-1.5 text-xs text-muted-foreground/80 leading-normal line-clamp-2">
            {hint}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
  badge,
}: {
  title: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  badge?: string;
}) {
  return (
    <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-border/50 pb-3">
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <h1 className="text-lg font-bold tracking-tight text-foreground sm:text-xl font-sans">
            {title}
          </h1>
          {badge && (
            <Badge variant="outline" className="font-mono text-[10px] font-medium px-1.5 py-0">
              {badge}
            </Badge>
          )}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground leading-normal">
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function AlgorithmBadge({ algorithm }: { algorithm: string }) {
  const label = ALGO_LABEL[algorithm as keyof typeof ALGO_LABEL] || algorithm;
  const color = ALGO_COLOR[algorithm as keyof typeof ALGO_COLOR] || "#64748B";

  return (
    <Badge
      variant="outline"
      className="gap-1.5 px-2 py-0.5 text-xs font-medium border-border/80 shadow-2xs"
    >
      <span className="size-2 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </Badge>
  );
}

export function LoadingCard({ message = "Đang tải dữ liệu…" }: { message?: string }) {
  return (
    <Card className="p-8 flex flex-col items-center justify-center text-center space-y-4">
      <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">{message}</p>
        <p className="text-xs text-muted-foreground">Đang truy vấn FastAPI backend</p>
      </div>
    </Card>
  );
}

export function ErrorCard({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <Card className="border-destructive/30 bg-destructive/5 p-6 space-y-4">
      <div className="flex items-start gap-3">
        <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
        <div className="space-y-1 flex-1">
          <h4 className="text-sm font-semibold text-destructive">Lỗi truy xuất dữ liệu</h4>
          <p className="text-xs text-muted-foreground">{message}</p>
        </div>
      </div>
      {onRetry && (
        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={onRetry} className="gap-1.5 text-xs">
            <RotateCcw className="size-3.5" />
            <span>Thử lại</span>
          </Button>
        </div>
      )}
    </Card>
  );
}

export function ResearchCallout({
  title,
  children,
  type = "info",
}: {
  title: string;
  children: React.ReactNode;
  type?: "info" | "warning" | "success";
}) {
  const borderClass =
    type === "warning"
      ? "border-amber-500/30 bg-amber-500/5 text-foreground"
      : type === "success"
      ? "border-emerald-500/30 bg-emerald-500/5 text-foreground"
      : "border-border/70 bg-card text-foreground shadow-2xs";

  return (
    <div className={`rounded-lg border p-4 text-xs leading-relaxed ${borderClass}`}>
      <div className="font-semibold flex items-center gap-1.5 text-foreground">
        <Info className="size-3.5 text-primary" />
        <span>{title}</span>
      </div>
      <div className="text-muted-foreground leading-relaxed mt-1.5">{children}</div>
    </div>
  );
}

