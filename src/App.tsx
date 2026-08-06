import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ErrorBoundary from "@/components/truthlens/ErrorBoundary";
import { AuthProvider } from "@/integrations/auth";
import RouteFallback from "@/components/truthlens/RouteFallback";

// The landing page is the entry point, so it stays in the main bundle. Every other route is split
// out: a visitor who only reads the home page should not download pdf.js, the report generator or
// four dashboards. This took the initial chunk from 651 kB to a fraction of it.
import TruthLensHome from "./pages/TruthLensHome";

const TruthLensLogin = lazy(() => import("./pages/TruthLensLogin"));
const TruthLensVerify = lazy(() => import("./pages/TruthLensVerify"));
const TruthLensReview = lazy(() => import("./pages/TruthLensReview"));
const TruthLensDashboard = lazy(() => import("./pages/TruthLensDashboard"));
const TruthLensBenchmark = lazy(() => import("./pages/TruthLensBenchmark"));
const TruthLensAdmin = lazy(() => import("./pages/TruthLensAdmin"));
const TruthLensHistory = lazy(() => import("./pages/TruthLensHistory"));

// Batch verification is intentionally not routed: the page and its API remain in the tree so the
// feature can be restored, but an unfinished surface must not be reachable.

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
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              {/* TruthLens AI Enterprise Routes */}
              <Route path="/" element={<TruthLensHome />} />
              <Route path="/login" element={<TruthLensLogin />} />
              <Route path="/verify" element={<TruthLensVerify />} />
              <Route path="/review" element={<TruthLensReview />} />
              <Route path="/dashboard" element={<TruthLensDashboard />} />
              <Route path="/benchmark" element={<TruthLensBenchmark />} />
              <Route path="/admin" element={<TruthLensAdmin />} />
              <Route path="/history" element={<TruthLensHistory />} />

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
    </AuthProvider>
  </ErrorBoundary>
);

export default App;
