import React, { useEffect } from "react";
import { Play, Pause, RotateCcw, FastForward, StepBack, StepForward } from "lucide-react";
import { AnimBus, useBus } from "../RouteAnimator";

interface AtlasPlaybackConsoleProps {
  bus: AnimBus;
  totalStops?: number;
  className?: string;
}

export function AtlasPlaybackConsole({ bus, totalStops = 0, className = "" }: AtlasPlaybackConsoleProps) {
  const s = useBus(bus);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        if (bus.s.progress >= 1) {
          bus.set({ progress: 0, playing: true, run: bus.s.run + 1 });
        } else {
          bus.set({ playing: !bus.s.playing });
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [bus]);

  const toggle = () => {
    if (s.progress >= 1) {
      bus.set({ progress: 0, playing: true, run: s.run + 1 });
    } else {
      bus.set({ playing: !s.playing });
    }
  };

  const reset = () => {
    bus.set({ progress: 0, playing: true, run: s.run + 1 });
  };

  const stepBack = () => {
    const nextP = Math.max(0, s.progress - 0.05);
    bus.set({ progress: nextP, playing: false });
  };

  const stepForward = () => {
    const nextP = Math.min(1, s.progress + 0.05);
    bus.set({ progress: nextP, playing: false });
  };

  const speeds = [0.5, 1, 2, 4, 8];
  const pct = Math.round(s.progress * 100);
  const currentStop = totalStops > 0 ? Math.min(totalStops, Math.floor(s.progress * totalStops) + 1) : null;

  return (
    <div
      className={`flex items-center gap-3 px-3.5 py-2 rounded-lg border border-border bg-card/95 backdrop-blur-md shadow-md text-xs select-none ${className}`}
      role="region"
      aria-label="Điều khiển lộ trình mô phỏng"
    >
      {/* Primary Action: Play/Pause */}
      <button
        onClick={toggle}
        className="flex items-center justify-center size-8 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-medium shadow-2xs"
        title={s.playing ? "Tạm dừng (Phím Space)" : "Phát lộ trình (Phím Space)"}
        aria-label={s.playing ? "Tạm dừng" : "Phát"}
      >
        {s.playing ? <Pause className="size-4" /> : <Play className="size-4 ml-0.5" />}
      </button>

      {/* Step Buttons */}
      <div className="flex items-center gap-1 border-r border-border pr-2.5">
        <button
          onClick={stepBack}
          className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="Lùi 5%"
          aria-label="Lùi một chặng"
        >
          <StepBack className="size-3.5" />
        </button>
        <button
          onClick={reset}
          className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="Phát lại từ đầu"
          aria-label="Phát lại từ đầu"
        >
          <RotateCcw className="size-3.5" />
        </button>
        <button
          onClick={stepForward}
          className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="Tiến 5%"
          aria-label="Tiến một chặng"
        >
          <StepForward className="size-3.5" />
        </button>
      </div>

      {/* Scrubber slider */}
      <div className="flex items-center gap-2 flex-1 min-w-[120px]">
        <input
          type="range"
          min={0}
          max={1000}
          value={Math.round(s.progress * 1000)}
          onChange={(e) => bus.set({ progress: Number(e.target.value) / 1000, playing: false })}
          className="w-full accent-[#176B5B] dark:accent-[#72C5AC] cursor-pointer h-1.5 bg-muted rounded-lg"
          aria-label="Thanh trượt tiến độ lộ trình"
        />
        <span className="font-mono tabular-nums text-[11px] text-muted-foreground w-9 text-right font-semibold">
          {pct}%
        </span>
      </div>

      {/* Stop counter if provided */}
      {currentStop !== null && (
        <div className="hidden sm:flex items-center gap-1 font-mono text-[11px] px-2 py-0.5 rounded bg-muted text-foreground border border-border shrink-0">
          <span className="text-muted-foreground">Chặng:</span>
          <span className="font-semibold text-primary">
            {currentStop}/{totalStops}
          </span>
        </div>
      )}

      {/* Speed Selector */}
      <div className="flex items-center rounded-md bg-muted p-0.5 border border-border shrink-0">
        {speeds.map((v) => (
          <button
            key={v}
            onClick={() => bus.set({ speed: v })}
            className={`px-1.5 py-0.5 text-[10.5px] font-mono rounded transition-colors ${
              s.speed === v
                ? "bg-card text-foreground font-semibold shadow-2xs border border-border"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {v}x
          </button>
        ))}
      </div>
    </div>
  );
}

