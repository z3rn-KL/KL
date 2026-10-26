import React from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Check } from "lucide-react";

export interface AlgorithmSelectorProps {
  algorithms: string[];
  selected: string | null;
  onSelect: (key: string | null) => void;
  isAvailable?: (key: string) => boolean;
  unavailableReason?: (key: string) => string;
  badge?: (key: string) => string | undefined;
  colors: Record<string, string>;
  labels: Record<string, string>;
  className?: string;
}

export function AlgorithmSelector({
  algorithms,
  selected,
  onSelect,
  isAvailable = () => true,
  unavailableReason = () => "Thuật toán không khả dụng trong phạm vi dữ liệu hiện tại.",
  badge = () => undefined,
  colors,
  labels,
  className = "",
}: AlgorithmSelectorProps) {
  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      {algorithms.map((key) => {
        const isSel = selected === key;
        const avail = isAvailable(key);
        const reason = !avail ? unavailableReason(key) : "";
        const badgeText = badge(key);
        const color = colors[key] ?? "#3B82F6";
        const label = labels[key] ?? key;

        const buttonElement = (
          <button
            key={key}
            type="button"
            aria-disabled={!avail}
            onClick={() => {
              if (avail) {
                onSelect(isSel ? null : key);
              }
            }}
            className={`h-8 px-3 text-xs rounded-lg inline-flex items-center gap-1.5 transition-all select-none shrink-0 ${
              isSel
                ? "font-medium border-2 border-primary bg-primary/10 text-primary dark:bg-primary/20 dark:text-foreground dark:border-primary shadow-2xs"
                : avail
                ? "font-medium border border-border/80 bg-card text-foreground hover:bg-muted/50 hover:text-foreground shadow-2xs cursor-pointer"
                : "font-medium border border-border/60 bg-transparent text-muted-foreground/75 cursor-not-allowed hover:bg-transparent"
            }`}
          >
            {/* Dot or Active Check Icon */}
            {isSel ? (
              <span
                className="flex size-4 shrink-0 items-center justify-center rounded-full text-white text-[10px]"
                style={{ backgroundColor: color }}
              >
                <Check className="size-2.5 stroke-[3]" />
              </span>
            ) : (
              <span
                className={`size-2.5 rounded-full shrink-0 ${!avail ? "opacity-40" : ""}`}
                style={{ backgroundColor: color }}
              />
            )}

            <span className="truncate">{label}</span>

            {/* Badge for unavailable workload hint */}
            {!avail && badgeText && (
              <span className="text-[10px] font-mono font-medium text-muted-foreground/90 bg-muted/60 px-1.5 py-0.5 rounded border border-border/60 ml-1 shrink-0 whitespace-nowrap">
                {badgeText}
              </span>
            )}
          </button>
        );

        if (!avail) {
          return (
            <Tooltip key={key}>
              <TooltipTrigger render={buttonElement} />
              <TooltipContent side="bottom" className="text-xs max-w-xs shadow-xl">
                {reason}
              </TooltipContent>
            </Tooltip>
          );
        }

        return buttonElement;
      })}
    </div>
  );
}
