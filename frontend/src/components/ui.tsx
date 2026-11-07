import type { ReactNode } from "react";

export function Icon({ d, size = 18 }: { d: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={d} />
    </svg>
  );
}

export function PageHead({
  title,
  eyebrow,
  children,
}: {
  title: string;
  eyebrow?: string;
  children?: ReactNode;
}) {
  return (
    <header className="page-head">
      {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
      <h1>{title}</h1>
      {children ? <p>{children}</p> : null}
    </header>
  );
}

export function Card({
  title,
  sub,
  children,
  className = "",
}: {
  title?: string;
  sub?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {title ? <h2 className="card-title">{title}</h2> : null}
      {sub ? <div className="card-sub">{sub}</div> : null}
      {children}
    </section>
  );
}

export type Tone = "teal" | "amber" | "violet" | "blue" | "rose";

export function Stat({
  label,
  value,
  hint,
  icon,
  tone = "teal",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: string;
  tone?: Tone;
}) {
  return (
    <div className={`stat tone-${tone}`}>
      {icon ? (
        <span className="stat-icon">
          <Icon d={icon} size={20} />
        </span>
      ) : null}
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export function Loading({ text = "Đang tải dữ liệu…" }: { text?: string }) {
  return (
    <div className="skeleton-loading-wrap" role="status" aria-label={text}>
      <div className="skeleton-head">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-sub" />
      </div>
      <div className="skeleton-grid">
        <div className="skeleton-card" />
        <div className="skeleton-card" />
        <div className="skeleton-card" />
        <div className="skeleton-card" />
      </div>
      <div className="skeleton-main-card">
        <div className="skeleton-inner-line" />
        <div className="skeleton-inner-line short" />
        <div className="skeleton-chart-box" />
        <div className="loading-inline-hint">
          <span className="spinner" />
          <span>{text}</span>
        </div>
      </div>
    </div>
  );
}

export function ErrorNote({
  message,
  kind = "bad",
  onRetry,
}: {
  message: string;
  kind?: "bad" | "warn";
  onRetry?: () => void;
}) {
  return (
    <div className={`note ${kind} error-note-card`} role="alert">
      <div className="error-note-icon">
        <Icon
          d={
            kind === "warn"
              ? "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              : "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          }
          size={20}
        />
      </div>
      <div className="error-note-body">
        <div className="error-note-message">{message}</div>
      </div>
      <button
        type="button"
        className="btn ghost error-retry-btn"
        onClick={() => {
          if (onRetry) onRetry();
          else window.location.reload();
        }}
        title="Thử lại"
      >
        <Icon d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" size={13} />
        <span>Thử lại</span>
      </button>
    </div>
  );
}

export function Callout({
  type = "note",
  title,
  children,
  className = "",
  style,
}: {
  type?: "note" | "tradeoff" | "warning";
  title?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div className={`callout ${type} ${className}`} style={style} role="note">
      {title ? <div className="callout-title">{title}</div> : null}
      <div className="callout-body">{children}</div>
    </div>
  );
}

export function Seg<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={o.value === value}
          className={o.value === value ? "on" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
