// Where the Tesseract engine is loaded from. Its own module, like consent: a host that packages
// the engine can point OCR at it without pulling tesseract.js into its startup bundle.
//
// The setting lives on globalThis for the same reason the consent handler does: a host and the
// libraries it embeds can end up with two copies of this module and must still see one answer.

/** URLs of a packaged Tesseract engine. Anything left out is fetched from the CDN as before. */
export interface OcrEnginePaths {
  /** tesseract.js's worker script. */
  workerPath?: string;
  /** A tesseract.js-core build, or the folder holding them. */
  corePath?: string;
}

const KEY = "__localmlOcrEnginePaths";
type Slot = { [KEY]?: OcrEnginePaths };

/**
 * Load the engine from the host's own URLs instead of the CDN. A host that packages
 * tesseract.js's worker and one tesseract.js-core build calls this once at startup; with nothing
 * set, tesseract.js fetches both from jsDelivr as before. Language data is not affected: it
 * still comes from the CDN.
 */
export function setOcrEnginePaths(paths: OcrEnginePaths | null): void {
  (globalThis as Slot)[KEY] = paths ?? undefined;
}

export function ocrEnginePaths(): OcrEnginePaths {
  return (globalThis as Slot)[KEY] ?? {};
}
