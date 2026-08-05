import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Crosshair, Loader2, FileWarning } from "lucide-react";
import * as pdfjs from "pdfjs-dist";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { Evidence } from "@/types/truthlens";

// Bundle the worker with the app rather than fetching it from a CDN — the published app must
// work offline and behind a strict CSP.
pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

interface DocumentEvidenceViewerProps {
  /** Data URL of the uploaded file — PDF or raster image. */
  source: string | null;
  fileName: string;
  evidence: Evidence[];
  activeEvidence: Evidence | null;
  onSelectEvidence: (evidence: Evidence) => void;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4;

/**
 * Renders the actual document and draws normalised evidence boxes over it.
 *
 * PDFs previously fell through to `<img src="data:application/pdf...">`, which is a broken image
 * in every browser — so the evidence viewer showed nothing for the primary enterprise format.
 * Selecting evidence now navigates to its page and zooms to its region.
 */
export default function DocumentEvidenceViewer({
  source,
  fileName,
  evidence,
  activeEvidence,
  onSelectEvidence,
}: DocumentEvidenceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [pageCount, setPageCount] = useState(1);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  /*
   * Detect PDFs by content type OR by extension. Checking only for the `data:application/pdf`
   * prefix meant any PDF supplied as a plain URL — a stored document, a signed storage link —
   * fell through to the <img> branch and rendered as a broken image with no error. The upload
   * path happens to produce data URLs, which hid the gap.
   */
  const isPdf = Boolean(
    source?.startsWith("data:application/pdf") ||
      /\.pdf($|[?#])/i.test(source ?? "") ||
      /\.pdf$/i.test(fileName),
  );

  /* ── Load the PDF once per source ── */
  useEffect(() => {
    if (!source || !isPdf) {
      setPdf(null);
      setStatus(source ? "ready" : "idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setError(null);

    // Data URL → decode in place; anything else → let pdf.js fetch it.
    const task = source.startsWith("data:")
      ? pdfjs.getDocument({ data: Uint8Array.from(atob(source.slice(source.indexOf(",") + 1)), (char) => char.charCodeAt(0)) })
      : pdfjs.getDocument(source);

    task.promise
      .then((doc) => {
        if (cancelled) {
          void doc.destroy();
          return;
        }
        setPdf(doc);
        setPageCount(doc.numPages);
        setStatus("ready");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "This PDF could not be rendered.");
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [source, isPdf]);

  /* ── Render the current page whenever page or zoom changes ── */
  useEffect(() => {
    if (!pdf || !canvasRef.current) return;
    let cancelled = false;
    let task: { cancel: () => void } | null = null;

    void pdf.getPage(Math.min(page, pdf.numPages)).then((pdfPage) => {
      const canvas = canvasRef.current;
      if (cancelled || !canvas) return;
      // Render at device resolution so text stays crisp when zoomed into a small region.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      /*
       * Fit the page to the container at 100% instead of a fixed 1.5x. A letter page is ~1188px
       * tall; the old fixed scale showed only its top third inside the viewer and forced the
       * reviewer to scroll hunting for the highlight. "100%" should mean "the whole page fits".
       */
      const container = scrollRef.current;
      const unscaled = pdfPage.getViewport({ scale: 1 });
      // Fit the *whole* page: constrain by whichever axis runs out first. Fitting width alone
      // still left a portrait page overflowing by ~500px, which is the problem being solved.
      const fitScale = Math.max(
        0.2,
        Math.min(
          ((container?.clientWidth ?? 640) - 4) / unscaled.width,
          ((container?.clientHeight ?? 480) - 4) / unscaled.height,
        ),
      );
      const viewport = pdfPage.getViewport({ scale: zoom * fitScale * dpr });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / dpr}px`;
      canvas.style.height = `${viewport.height / dpr}px`;
      const context = canvas.getContext("2d");
      if (!context) return;
      task = pdfPage.render({ canvasContext: context, viewport });
    });

    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [pdf, page, zoom]);

  /* ── Selecting evidence navigates to its page and scrolls its box into view ── */
  useEffect(() => {
    if (!activeEvidence) return;
    if (activeEvidence.pageNumber !== page && activeEvidence.pageNumber <= pageCount) {
      setPage(activeEvidence.pageNumber);
    }
    const container = scrollRef.current;
    const box = activeEvidence.boundingBox;
    if (!container || !box) return;
    // Defer so the page render has laid out before we scroll to the region.
    const timer = window.setTimeout(() => {
      container.scrollTo({
        left: (box.x / 100) * container.scrollWidth - container.clientWidth / 2 + ((box.width / 100) * container.scrollWidth) / 2,
        top: (box.y / 100) * container.scrollHeight - container.clientHeight / 2 + ((box.height / 100) * container.scrollHeight) / 2,
        behavior: "smooth",
      });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [activeEvidence, page, pageCount]);

  const zoomToEvidence = useCallback(() => {
    const box = activeEvidence?.boundingBox;
    if (!box) return;
    // Fit the region to roughly half the viewport, clamped to sane bounds.
    const target = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, 45 / Math.max(box.width, box.height, 4)));
    setZoom(Math.round(target * 10) / 10);
  }, [activeEvidence]);

  const pageEvidence = useMemo(
    () => evidence.filter((item) => item.boundingBox && item.pageNumber === page),
    [evidence, page],
  );

  if (!source) {
    return (
      <EmptyState icon={<FileWarning className="w-9 h-9 text-primary/50" />} title="No document available">
        The uploaded file is not attached to this result, so evidence regions cannot be drawn.
      </EmptyState>
    );
  }

  if (status === "error") {
    return (
      <EmptyState icon={<FileWarning className="w-9 h-9 text-danger/60" />} title="Document could not be rendered">
        {error} Evidence coordinates are still listed below and remain part of the record.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-2">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setPage((current) => Math.max(1, current - 1))}
            disabled={page <= 1}
            className="p-1.5 rounded-lg glass-light border border-border disabled:opacity-30 disabled:cursor-not-allowed hover:text-primary"
            aria-label="Previous page"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono text-muted-foreground px-1.5 tabular-nums">
            {page} / {pageCount}
          </span>
          <button
            onClick={() => setPage((current) => Math.min(pageCount, current + 1))}
            disabled={page >= pageCount}
            className="p-1.5 rounded-lg glass-light border border-border disabled:opacity-30 disabled:cursor-not-allowed hover:text-primary"
            aria-label="Next page"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={zoomToEvidence}
            disabled={!activeEvidence?.boundingBox}
            className="px-2 py-1.5 rounded-lg glass-light border border-border flex items-center gap-1.5 disabled:opacity-30 disabled:cursor-not-allowed hover:text-accent"
            title="Zoom to the selected evidence region"
          >
            <Crosshair className="w-3.5 h-3.5" /> Zoom to evidence
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(MIN_ZOOM, Math.round((z - 0.25) * 100) / 100))}
            className="p-1.5 rounded-lg glass-light border border-border hover:text-primary"
            aria-label="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono text-muted-foreground w-10 text-center tabular-nums">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom((z) => Math.min(MAX_ZOOM, Math.round((z + 0.25) * 100) / 100))}
            className="p-1.5 rounded-lg glass-light border border-border hover:text-primary"
            aria-label="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Canvas + overlay */}
      {/* Tall enough to show a full page at 100% without scrolling — the point of an evidence
          viewer is that the reviewer sees the highlight in the context of the page. */}
      <div ref={scrollRef} className="relative w-full h-[30rem] rounded-xl overflow-auto glass-light border border-border/80 text-center">
        {status === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center z-30 bg-background/40">
            <Loader2 className="w-6 h-6 text-primary animate-spin" />
          </div>
        )}

        {/*
          The positioning context must be exactly the page, not the scroll container. With
          `min-w-full` this div stayed container-width while the canvas shrank to fit, so every
          percentage-positioned box drifted right — measured at 24% for a box that belonged at
          11.5%. Shrink-wrapping the canvas makes the percentages mean what they say.
        */}
        <div className="relative inline-block">
          {isPdf ? (
            <canvas ref={canvasRef} className="block" />
          ) : (
            <img
              src={source}
              alt={`${fileName} preview`}
              style={{ width: `${zoom * 100}%` }}
              className="block max-w-none"
              onLoad={() => setStatus("ready")}
            />
          )}

          {/* Evidence regions. Coordinates are percentages, normalised server-side, so they
              scale with zoom without further conversion. */}
          {pageEvidence.map((item, position) => {
            const box = item.boundingBox!;
            const isActive = activeEvidence?.id === item.id;
            return (
              <motion.button
                key={item.id}
                type="button"
                initial={{ opacity: 0 }}
                animate={{ opacity: isActive ? 1 : item.cited ? 0.75 : 0.4 }}
                onClick={() => onSelectEvidence(item)}
                title={item.text.slice(0, 160)}
                className={`absolute rounded flex items-start p-0.5 transition-colors ${
                  isActive
                    ? "border-2 border-accent bg-accent/25 shadow-lg shadow-accent/30 z-20"
                    : item.cited
                    ? "border border-success/70 bg-success/10 z-10"
                    : "border border-dashed border-primary/50 bg-primary/5 z-0"
                }`}
                style={{
                  left: `${box.x}%`,
                  top: `${box.y}%`,
                  width: `${box.width}%`,
                  height: `${box.height}%`,
                }}
              >
                {isActive && (
                  <span className="bg-accent text-accent-foreground text-[8px] font-bold px-1 py-0.5 rounded shadow whitespace-nowrap">
                    {item.layoutRegion?.toUpperCase() || "EVIDENCE"} #{position + 1}
                  </span>
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm border border-success/70 bg-success/20" /> Cited by verifier</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm border border-dashed border-primary/50 bg-primary/10" /> Retrieved, not cited</span>
      </div>
    </div>
  );
}

function EmptyState({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="w-full h-[30rem] rounded-xl glass-light border border-border/80 flex flex-col items-center justify-center text-center p-6">
      {icon}
      <p className="text-xs font-semibold mt-2">{title}</p>
      <p className="text-[11px] text-muted-foreground leading-relaxed mt-1 max-w-sm">{children}</p>
    </div>
  );
}
