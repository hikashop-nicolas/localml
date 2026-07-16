// On-device OCR via Tesseract.js. Tesseract runs its own WebAssembly worker internally and
// fetches its core + language data from a CDN on first use (then browser-cached), the same
// way the translation models download from a CDN. So this is just a thin driver: it creates
// a worker, streams progress, recognizes one image and tears the worker down.
import { createWorker, type ImageLike } from "tesseract.js";

export interface OcrResult {
  text: string;
  confidence: number; // overall page confidence, 0..100
}

export type OcrProgress =
  | { stage: "load"; ratio: number } // downloading/initializing the engine + language data
  | { stage: "recognize"; ratio: number }; // running recognition on the image

export interface OcrOptions {
  // Tesseract language code(s): "eng", "fra", "jpn", or a "+"-joined combo like "eng+fra".
  lang?: string;
  onProgress?: (p: OcrProgress) => void;
}

export interface OcrRun {
  done: Promise<OcrResult>;
  cancel(): void; // terminate the worker; `done` rejects with an "cancelled" error
}

// Recognize text in a single image. Returns a cancellable run.
export function runOcr(image: ImageLike, opts: OcrOptions = {}): OcrRun {
  const lang = opts.lang && opts.lang.trim() ? opts.lang.trim() : "eng";
  let cancelled = false;
  let terminate: (() => void) | null = null;

  const done = (async (): Promise<OcrResult> => {
    const worker = await createWorker(lang, 1, {
      logger: (m: { status?: string; progress?: number }) => {
        const ratio = typeof m.progress === "number" ? m.progress : 0;
        const stage: OcrProgress["stage"] = m.status?.includes("recogniz") ? "recognize" : "load";
        opts.onProgress?.({ stage, ratio });
      },
    });
    terminate = () => void worker.terminate();
    if (cancelled) {
      await worker.terminate();
      throw new Error("cancelled");
    }
    try {
      const { data } = await worker.recognize(image);
      return { text: data.text, confidence: data.confidence };
    } finally {
      await worker.terminate();
    }
  })();

  return {
    done,
    cancel: () => {
      cancelled = true;
      terminate?.();
    },
  };
}
