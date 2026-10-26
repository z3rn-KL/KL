import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AtlasShell } from "@/components/atlas/AtlasShell";
import { PageErrorBoundary } from "@/components/ErrorBoundary";
import { LoadingCard } from "@/components/shared/DomainComponents";

const Overview = lazy(() => import("./pages/Overview"));
const Orders = lazy(() => import("./pages/Orders"));
const Clusters = lazy(() => import("./pages/Clusters"));
const RoutesPage = lazy(() => import("./pages/RoutesPage"));
const RunLab = lazy(() => import("./pages/RunLab"));
const Compare = lazy(() => import("./pages/Compare"));
const GuidedDQN = lazy(() => import("./pages/GuidedDQN"));
const Sensitivity = lazy(() => import("./pages/Sensitivity"));
const Stability = lazy(() => import("./pages/Stability"));

export default function App() {
  return (
    <AtlasShell>
      <Suspense fallback={<LoadingCard message="Đang tải trang…" />}>
        <Routes>
          <Route path="/" element={<Navigate to="/tong-quan" replace />} />
          <Route
            path="/tong-quan"
            element={
              <PageErrorBoundary>
                <Overview />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/don-hang"
            element={
              <PageErrorBoundary>
                <Orders />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/phan-cum"
            element={
              <PageErrorBoundary>
                <Clusters />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/tuyen-giao"
            element={
              <PageErrorBoundary>
                <RoutesPage />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/chay-thu"
            element={
              <PageErrorBoundary>
                <RunLab />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/so-sanh"
            element={
              <PageErrorBoundary>
                <Compare />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/guided-dqn"
            element={
              <PageErrorBoundary>
                <GuidedDQN />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/do-nhay"
            element={
              <PageErrorBoundary>
                <Sensitivity />
              </PageErrorBoundary>
            }
          />
          <Route
            path="/on-dinh"
            element={
              <PageErrorBoundary>
                <Stability />
              </PageErrorBoundary>
            }
          />

          {/* English URL aliases */}
          <Route path="/overview" element={<Navigate to="/tong-quan" replace />} />
          <Route path="/orders" element={<Navigate to="/don-hang" replace />} />
          <Route path="/clusters" element={<Navigate to="/phan-cum" replace />} />
          <Route path="/routes" element={<Navigate to="/tuyen-giao" replace />} />
          <Route path="/playground" element={<Navigate to="/chay-thu" replace />} />
          <Route path="/compare" element={<Navigate to="/so-sanh" replace />} />
          <Route path="/sensitivity" element={<Navigate to="/do-nhay" replace />} />
          <Route path="/stability" element={<Navigate to="/on-dinh" replace />} />
          <Route path="*" element={<Navigate to="/tong-quan" replace />} />
        </Routes>
      </Suspense>
    </AtlasShell>
  );
}
