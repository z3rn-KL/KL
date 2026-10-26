import React from "react";

export interface MetricItem {
  id?: string;
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  sub?: string;
  badge?: string;
  intent?: "positive" | "warning" | "violation" | "neutral";
  status?: "positive" | "warning" | "violation" | "neutral";
  icon?: React.ElementType;
  mono?: boolean;
}

export type AtlasMetricItem = MetricItem;

interface AtlasMetricStripProps {
  metrics: MetricItem[];
  className?: string;
  scope?: string;
}

export function AtlasMetricStrip({ metrics, className = "", scope }: AtlasMetricStripProps) {
  const getStatusColor = (intent?: MetricItem["intent"], status?: MetricItem["status"]) => {
    const s = intent || status;
    switch (s) {
      case "positive":
        return "text-emerald-700 dark:text-emerald-400";
      case "warning":
        return "text-amber-700 dark:text-amber-400";
      case "violation":
        return "text-rose-700 dark:text-rose-400";
      default:
        return "text-foreground";
    }
  };

  return (
    <div
      className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 p-3 rounded-lg border border-border/80 bg-card/90 shadow-2xs text-xs ${className}`}
    >
      {metrics.map((m, idx) => {
        const Icon = m.icon;
        const colorClass = getStatusColor(m.intent, m.status);

        return (
          <div
            key={m.id || idx}
            className="flex flex-col justify-between p-2 rounded-md bg-muted/20 border border-border/50 hover:bg-muted/30 transition-colors"
          >
            <div className="flex items-center justify-between gap-1 mb-1">
              <span className="text-[10.5px] font-bold uppercase tracking-wider text-muted-foreground truncate">
                {m.label}
              </span>
              {m.badge && (
                <span className="rounded bg-primary/10 px-1 py-0.2 font-mono text-[9px] font-semibold text-primary border border-primary/20 shrink-0">
                  {m.badge}
                </span>
              )}
            </div>

            <div className="flex items-baseline gap-1.5 my-0.5">
              {Icon && <Icon className="size-3.5 text-muted-foreground shrink-0 self-center" />}
              <span className={`text-base font-bold font-mono tracking-tight tabular-nums ${colorClass}`}>
                {m.value}
              </span>
              {m.unit && <span className="text-[11px] text-muted-foreground font-medium">{m.unit}</span>}
            </div>

            {(m.hint || m.sub) && (
              <span className="text-[10px] text-muted-foreground/80 truncate mt-0.5">
                {m.hint || m.sub}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

