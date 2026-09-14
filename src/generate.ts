// Main-thread driver for on-device text generation (summarize / elaborate / shorten / write).
// Spawns the lazy worker, streams cumulative output text and download progress, and resolves
// with the final text. A run can be cancelled (drops the worker). Mirrors runTranslate.
import { genModel, TASK_MODEL, type GenTask, type GenProgress } from "./gen-backend";

export {
  GEN_MODELS,
  TASK_MODEL,
  genModel,
  buildMessages,
  capFor,
  type GenModelInfo,
  type GenTask,
  type GenProgress,
  type DtypeSpec,
} from "./gen-backend";

export interface GenerateOptions {
  task: GenTask;
  /** Override the default model for the task. */
  model?: string;
  /** A domain-specific system prompt for the chat tasks (overrides the task default), so a
      consumer can specialise (e.g. a spreadsheet-formula assistant) without a new task. */
  system?: string;
  /** Force a backend; omit to auto-detect (WebGPU, else WASM). */
  device?: "webgpu" | "wasm";
}

export interface GenerateCallbacks {
  onProgress?: (p: GenProgress) => void;
  /** The generated text so far (cumulative), fired as tokens stream in. */
  onPartial?: (text: string) => void;
  /** Which backend the run settled on. */
  onDevice?: (device: "webgpu" | "wasm") => void;
}

export interface GenerateRun {
  cancel(): void;
  /** Resolves with the final text when generation finishes (or the last text on cancel). */
  done: Promise<{ text: string; stopped: boolean }>;
}

/** Run one assist task over `input` (selected text, or the instruction for "write"). */
export function runGenerate(input: string, opts: GenerateOptions, cb: GenerateCallbacks = {}): GenerateRun {
  const model = opts.model ?? TASK_MODEL[opts.task];
  const info = genModel(model);
  const engine = info?.engine ?? "chat";

  let worker: Worker | null = null;
  let last = "";
  let settled = false;

  const done = new Promise<{ text: string; stopped: boolean }>((resolve, reject) => {
    /** One attempt in its own worker; `device` undefined lets the worker pick WebGPU when it can. */
    const attempt = (device: "webgpu" | "wasm" | undefined): void => {
      const w = new Worker(new URL("./generate.worker.ts", import.meta.url), { type: "module" });
      worker = w;
      let ranOn: "webgpu" | "wasm" | undefined;
      const fail = (message: string): void => {
        w.terminate();
        if (settled) return;
        // A GPU that fails, while loading or part way through (a phone short of memory loses its
        // device), leaves that worker unusable. Start over on the CPU in a fresh one, unless the
        // caller chose the backend or the CPU is what just failed.
        if (!opts.device && ranOn !== "wasm" && device !== "wasm") {
          console.warn("[localml] generation failed on WebGPU, retrying on the CPU:", message);
          last = "";
          cb.onPartial?.("");
          attempt("wasm");
          return;
        }
        settled = true;
        reject(new Error(message));
      };
      w.onmessage = (e: MessageEvent) => {
        const m = e.data;
        switch (m.type) {
          case "progress":
            cb.onProgress?.({ stage: m.stage, ratio: m.ratio, file: m.file });
            break;
          case "partial":
            last = m.text;
            cb.onPartial?.(m.text);
            break;
          case "device":
            ranOn = m.device;
            cb.onDevice?.(m.device);
            break;
          case "done":
            settled = true;
            resolve({ text: (m.text ?? last) as string, stopped: false });
            w.terminate();
            break;
          case "error":
            fail(m.message);
            break;
        }
      };
      w.onerror = (e) => fail(e.message || "worker error");
      w.postMessage({
        type: "run",
        task: opts.task,
        input,
        system: opts.system,
        model,
        engine,
        device: device ?? opts.device,
        dtype: info?.dtype,
      });
    };
    attempt(undefined);
  });

  return {
    cancel: () => {
      if (!settled) {
        settled = true;
        (worker as Worker | null)?.terminate();
      }
    },
    done,
  };
}
