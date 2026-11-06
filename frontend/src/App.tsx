import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AtlasShell } from "@/components/atlas/AtlasShell";
import { PageErrorBoundary } from "@/components/ErrorBoundary";
import { LoadingCard } from "@/components/shared/DomainComponents";
import { DispatchProvider } from "@/context/DispatchContext";

// The 6 Official Thesis-Aligned Screens
const Overview = lazy(() => import("./pages/Overview"));
const DataClustering = lazy(() => import("./pages/DataClustering"));
const RLOptimization = lazy(() => import("./pages/RLOptimization"));
const SimulationOps = lazy(() => import("./pages/SimulationOps"));
const ShiftReport = lazy(() => import("./pages/ShiftReport"));
const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage"));

export default function App() {
  return (
    <DispatchProvider>
      <AtlasShell>
        <Suspense fallback={<LoadingCard message="Đang nạp màn hình…" />}>
          <Routes>
            {/* Root entrypoint — Renders Screen 01: Overview directly to eliminate race conditions */}
            <Route
              path="/"
              element={
                <PageErrorBoundary>
                  <Overview />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 01 — TỔNG QUAN */}
            <Route
              path="/tong-quan"
              element={
                <PageErrorBoundary>
                  <Overview />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 02 — DỮ LIỆU & PHÂN CỤM */}
            <Route
              path="/phan-cum"
              element={
                <PageErrorBoundary>
                  <DataClustering />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 03 — TỐI ƯU TUYẾN BẰNG RL */}
            <Route
              path="/toi-uu-rl"
              element={
                <PageErrorBoundary>
                  <RLOptimization />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 04 — MÔ PHỎNG GIAO HÀNG */}
            <Route
              path="/mo-phong"
              element={
                <PageErrorBoundary>
                  <SimulationOps />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 05 — BÁO CÁO KẾT CA */}
            <Route
              path="/bao-cao"
              element={
                <PageErrorBoundary>
                  <ShiftReport />
                </PageErrorBoundary>
              }
            />

            {/* SCREEN 06 — THỰC NGHIỆM & SO SÁNH */}
            <Route
              path="/thuc-nghiem"
              element={
                <PageErrorBoundary>
                  <ExperimentsPage />
                </PageErrorBoundary>
              }
            />

            {/* ============================================================== */}
            {/* LEGACY ROUTE ALIASES (100% COVERAGE — NO DEAD ROUTES)          */}
            {/* ============================================================== */}
            {/* Screen 01 aliases */}
            <Route path="/overview" element={<PageErrorBoundary><Overview /></PageErrorBoundary>} />

            {/* Screen 02 aliases */}
            <Route path="/don-hang" element={<PageErrorBoundary><DataClustering /></PageErrorBoundary>} />
            <Route path="/orders" element={<PageErrorBoundary><DataClustering /></PageErrorBoundary>} />
            <Route path="/clusters" element={<PageErrorBoundary><DataClustering /></PageErrorBoundary>} />
            <Route path="/du-lieu-that" element={<PageErrorBoundary><DataClustering /></PageErrorBoundary>} />
            <Route path="/real-data" element={<PageErrorBoundary><DataClustering /></PageErrorBoundary>} />

            {/* Screen 03 aliases */}
            <Route path="/planning" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/rl-optimization" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/ke-hoach-ngay" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/daily-planning" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/proposed-rl" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/tuyen-giao" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />
            <Route path="/routes" element={<PageErrorBoundary><RLOptimization /></PageErrorBoundary>} />

            {/* Screen 04 aliases */}
            <Route path="/simulation" element={<PageErrorBoundary><SimulationOps /></PageErrorBoundary>} />
            <Route path="/live-ops" element={<PageErrorBoundary><SimulationOps /></PageErrorBoundary>} />
            <Route path="/dieu-phoi" element={<PageErrorBoundary><SimulationOps /></PageErrorBoundary>} />
            <Route path="/dispatch" element={<PageErrorBoundary><SimulationOps /></PageErrorBoundary>} />

            {/* Screen 05 aliases */}
            <Route path="/day-report" element={<PageErrorBoundary><ShiftReport /></PageErrorBoundary>} />
            <Route path="/report" element={<PageErrorBoundary><ShiftReport /></PageErrorBoundary>} />

            {/* Screen 06 aliases */}
            <Route path="/experiments" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/so-sanh" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/compare" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/nghien-cuu" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/research" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/research-analytics" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/guided-dqn" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/do-nhay" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/sensitivity" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/on-dinh" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/stability" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/chay-thu" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/playground" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />
            <Route path="/run-lab" element={<PageErrorBoundary><ExperimentsPage /></PageErrorBoundary>} />

            {/* Fallback to Overview */}
            <Route path="*" element={<PageErrorBoundary><Overview /></PageErrorBoundary>} />
          </Routes>
        </Suspense>
      </AtlasShell>
    </DispatchProvider>
  );
}
