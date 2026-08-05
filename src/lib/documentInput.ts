/**
 * Document intake: everything becomes an image before it leaves the browser.
 *
 * PDFs are rasterised here with pdf.js, which the evidence viewer already loads. Three reasons
 * this belongs client-side rather than in the pipeline:
 *
 *   1. The verification engine then has exactly one input shape. No PDF branch, no vendor-specific
 *      file handling, no "this provider can read PDFs and that one cannot".
 *   2. Several vendors genuinely cannot accept a PDF — OpenRouter charges for its file-parser
 *      plugin and returned 402 on a zero balance, while the same page as an image worked free.
 *   3. Rasterising on the client costs the server nothing and keeps the upload inside the request
 *      body limit, because a rendered page is usually smaller than the source PDF.
 */

const MAX_PAGES = 10;
/** 2x renders small print legibly; below this OCR loses digits on dense invoices. */
const RENDER_SCALE = 2;
const JPEG_QUALITY = 0.92;

export interface PreparedPage {
  /** Data URL of the rendered page. */
  dataUrl: string;
  pageNumber: number;
  width: number;
  height: number;
}

export interface PreparedDocument {
  pages: PreparedPage[];
  /** Original file name, preserved for the audit trail. */
  fileName: string;
  sourceType: string;
  /** True when a PDF was rasterised rather than uploaded as-is. */
  converted: boolean;
  /** Set when the document had more pages than we render. */
  truncatedFrom?: number;
}

const readAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`${file.name} could not be read.`));
    reader.readAsDataURL(file);
  });

const readAsArrayBuffer = (file: File) =>
  new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(new Error(`${file.name} could not be read.`));
    reader.readAsArrayBuffer(file);
  });

/** Detect a PDF from its bytes, not its extension — users rename and re-save files constantly. */
async function isPdf(file: File): Promise<boolean> {
  const head = new Uint8Array(await file.slice(0, 5).arrayBuffer());
  return head[0] === 0x25 && head[1] === 0x50 && head[2] === 0x44 && head[3] === 0x46; // %PDF
}

/**
 * Prepare any supported document for verification.
 * Images pass through untouched; PDFs are rendered page by page.
 */
export async function prepareDocument(file: File): Promise<PreparedDocument> {
  if (!(await isPdf(file))) {
    const dataUrl = await readAsDataUrl(file);
    const size = await imageSize(dataUrl);
    return {
      pages: [{ dataUrl, pageNumber: 1, width: size.width, height: size.height }],
      fileName: file.name,
      sourceType: file.type || "image",
      converted: false,
    };
  }

  // Loaded lazily so a user who never uploads a PDF never downloads pdf.js.
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

  const data = new Uint8Array(await readAsArrayBuffer(file));
  const document = await pdfjs.getDocument({ data }).promise;
  const total = document.numPages;
  const renderCount = Math.min(total, MAX_PAGES);
  const pages: PreparedPage[] = [];

  for (let pageNumber = 1; pageNumber <= renderCount; pageNumber++) {
    const page = await document.getPage(pageNumber);
    const viewport = page.getViewport({ scale: RENDER_SCALE });
    const canvas = window.document.createElement("canvas");
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("This browser could not render the PDF to an image.");
    // White background: a transparent canvas encodes to black in JPEG and destroys the text.
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    await page.render({ canvasContext: context, viewport }).promise;
    pages.push({
      dataUrl: canvas.toDataURL("image/jpeg", JPEG_QUALITY),
      pageNumber,
      width: canvas.width,
      height: canvas.height,
    });
  }

  await document.destroy();

  return {
    pages,
    fileName: file.name,
    sourceType: "application/pdf",
    converted: true,
    truncatedFrom: total > renderCount ? total : undefined,
  };
}

function imageSize(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => resolve({ width: 0, height: 0 });
    image.src = dataUrl;
  });
}

/** Formats the intake accepts. TIFF is included per requirement, though browser support varies. */
export const ACCEPTED_EXTENSIONS = ".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff";
export const ACCEPTED_PATTERN = /\.(pdf|png|jpe?g|webp|tiff?)$/i;
