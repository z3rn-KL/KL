import { Component, type ReactNode } from "react";

/* ─── State & Props ─── */
interface EBState {
  hasError: boolean;
  error: Error | null;
}

interface EBProps {
  children: ReactNode;
  /** Custom fallback UI; receives a reset callback. */
  fallback?: (props: { error: Error | null; reset: () => void }) => ReactNode;
}

/* ─── Generic ErrorBoundary ─── */
export class ErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  reset = () => this.setState({ hasError: false, error: null });

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback({ error: this.state.error, reset: this.reset });
      }
      return (
        <div
          style={{
            padding: 24,
            textAlign: "center",
            color: "var(--warn, #f59e0b)",
            background: "var(--surface-2, #1e1e2e)",
            borderRadius: 10,
            border: "1px solid var(--border, #333)",
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
            Không thể hiển thị nội dung này
          </div>
          <div style={{ fontSize: 12, color: "var(--muted, #888)", marginBottom: 12 }}>
            {this.state.error?.message ?? "Đã xảy ra lỗi không xác định."}
          </div>
          <button
            onClick={this.reset}
            style={{
              padding: "6px 18px",
              borderRadius: 6,
              border: "1px solid var(--border, #444)",
              background: "var(--accent, #8B8CF8)",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: 13,
            }}
          >
            Thử lại
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ─── Pre-built fallback for map blocks ─── */
export function MapErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={({ error, reset }) => (
        <div
          style={{
            height: 420,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
            color: "var(--warn, #f59e0b)",
            background: "var(--surface-2, #1e1e2e)",
            borderRadius: 10,
            border: "1px solid var(--border, #333)",
          }}
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <div style={{ fontSize: 14, fontWeight: 600 }}>Không thể hiển thị bản đồ này</div>
          <div style={{ fontSize: 12, color: "var(--muted, #888)", maxWidth: 360, textAlign: "center" }}>
            {error?.message ?? "Lỗi không xác định khi khởi tạo bản đồ."}
          </div>
          <button
            onClick={reset}
            style={{
              marginTop: 4,
              padding: "6px 18px",
              borderRadius: 6,
              border: "1px solid var(--border, #444)",
              background: "var(--accent, #8B8CF8)",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: 13,
            }}
          >
            Thử lại
          </button>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

/* ─── Pre-built fallback for full-page routes ─── */
export function PageErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={({ error, reset }) => (
        <div
          style={{
            padding: 48,
            textAlign: "center",
            color: "var(--warn, #f59e0b)",
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>
            Không thể hiển thị trang này
          </div>
          <div style={{ fontSize: 13, color: "var(--muted, #888)", marginBottom: 16, maxWidth: 480, margin: "0 auto 16" }}>
            {error?.message ?? "Đã xảy ra lỗi không mong muốn."}
          </div>
          <button
            onClick={reset}
            style={{
              padding: "8px 24px",
              borderRadius: 8,
              border: "1px solid var(--border, #444)",
              background: "var(--accent, #8B8CF8)",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: 14,
            }}
          >
            Thử lại
          </button>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

