import React, { useState, useEffect } from "react";
import { Clock } from "lucide-react";

export interface LiveClockProps {
  className?: string;
  showDate?: boolean;
  showTimezone?: boolean;
}

/**
 * LiveClock component providing real-time system clock formatted for Asia/Ho_Chi_Minh (UTC+7).
 *
 * CRITICAL DISAMBIGUATION:
 * - Real Time (Giờ thực tế): System wall-clock from browser, formatted in Asia/Ho_Chi_Minh.
 * - Simulation Time (Giờ mô phỏng): Discrete logical shift clock (e.g. 08:00, +15m) on Screen 04.
 * - Data Timestamp (Thời điểm cập nhật): Point-in-time timestamp from backend dataset/report.
 */
export const LiveClock: React.FC<LiveClockProps> = ({
  className = "",
  showDate = true,
  showTimezone = true,
}) => {
  const [time, setTime] = useState<Date>(() => new Date());

  useEffect(() => {
    let timerId: any = null;

    const tick = () => {
      // Avoid state updates if page is hidden to save CPU and battery
      if (typeof document !== "undefined" && document.hidden) return;
      setTime(new Date());
    };

    timerId = setInterval(tick, 1000);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        setTime(new Date());
      }
    };

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }

    return () => {
      if (timerId) clearInterval(timerId);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
    };
  }, []);

  // Format explicitly for Asia/Ho_Chi_Minh timezone
  const timeFormatter = new Intl.DateTimeFormat("vi-VN", {
    timeZone: "Asia/Ho_Chi_Minh",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  const dateFormatter = new Intl.DateTimeFormat("vi-VN", {
    timeZone: "Asia/Ho_Chi_Minh",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });

  const timeStr = timeFormatter.format(time);
  const dateStr = dateFormatter.format(time);

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border/70 bg-background/70 text-xs text-muted-foreground font-mono shadow-2xs select-none ${className}`}
      title="Đồng hồ thời gian thực trình duyệt (Asia/Ho_Chi_Minh • UTC+7) — Khác với đồng hồ mô phỏng điều phối"
    >
      <Clock className="size-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
      <span className="font-bold text-foreground tracking-tight">{timeStr}</span>
      {showDate && <span className="hidden xl:inline text-muted-foreground/80">· {dateStr}</span>}
      {showTimezone && (
        <span className="text-[10px] px-1 py-0.2 rounded bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 font-sans font-semibold">
          UTC+7
        </span>
      )}
    </div>
  );
};

