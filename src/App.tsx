import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ErrorBoundary from "@/components/truthlens/ErrorBoundary";
import RouteFallback from "@/components/truthlens/RouteFallback";

// The landing page is the entry point, so it stays in the main bundle. Every other route is split
// out: a visitor who only reads the home page should not download pdf.js, the report generator or
// four dashboards. This took the initial chunk from 651 kB to a fraction of it.
import TruthLensHome from "./pages/TruthLensHome";

const TruthLensVerify = lazy(() => import("./pages/TruthLensVerify"));
const TruthLensBatch = lazy(() => import("./pages/TruthLensBatch"));
const TruthLensReview = lazy(() => import("./pages/TruthLensReview"));
const TruthLensDashboard = lazy(() => import("./pages/TruthLensDashboard"));
const TruthLensBenchmark = lazy(() => import("./pages/TruthLensBenchmark"));
const TruthLensAdmin = lazy(() => import("./pages/TruthLensAdmin"));

// RLVO research demos — kept reachable so existing links do not break, but no longer part of the
// TruthLens product surface or its primary navigation.
const ImageRefinement = lazy(() => import("./pages/ImageRefinement"));
const VideoRefinement = lazy(() => import("./pages/VideoRefinement"));
const Proctoring = lazy(() => import("./pages/Proctoring"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 },
  },
});

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              {/* TruthLens AI Enterprise Routes */}
              <Route path="/" element={<TruthLensHome />} />
              <Route path="/verify" element={<TruthLensVerify />} />
              <Route path="/batch" element={<TruthLensBatch />} />
              <Route path="/review" element={<TruthLensReview />} />
              <Route path="/dashboard" element={<TruthLensDashboard />} />
              <Route path="/benchmark" element={<TruthLensBenchmark />} />
              <Route path="/admin" element={<TruthLensAdmin />} />

              {/* RLVO Lab & Proctoring research demos */}
              <Route path="/image-refinement" element={<ImageRefinement />} />
              <Route path="/video-refinement" element={<VideoRefinement />} />
              <Route path="/proctoring" element={<Proctoring />} />

              {/* Catch-all */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;
