// Text-generation worker: runs a summarizer (encoder-decoder) or a small instruction-tuned
// chat model (causal) off the main thread via transformers.js, streaming the output token by
// token. Talks to generate.ts via messages. WebGPU with a WASM fallback, like the translator.
import { pipeline, TextStreamer, env } from "@huggingface/transformers";
import { post, onMessage, hasWebGpu, downloadProgress } from "./worker-common";
import { buildMessages, capFor, cleanReply, genModel, type DtypeSpec, type GenTask } from "./gen-backend";

env.allowLocalModels = false;

interface RunMsg {
  type: "run";
  task: GenTask;
  input: string;
  system?: string;
  model: string;
  engine: "summarization" | "chat";
  device?: "webgpu" | "wasm";
  dtype?: { webgpu: DtypeSpec; wasm: DtypeSpec };
}

// Minimal shape of a transformers.js pipeline we use (callable + a tokenizer for token counts).
type Pipe = ((input: unknown, opts: Record<string, unknown>) => Promise<unknown>) & {
  tokenizer?: (t: string) => { input_ids?: { size?: number; dims?: number[] } };
};

let cached: { key: string; fn: Pipe } | null = null;

// Whether the GPU can run f16 shaders. Many phone GPUs cannot, and an f16 build fails on them.
async function gpuHasF16(): Promise<boolean> {
  try {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
    return !!(await gpu?.requestAdapter())?.features.has("shader-f16");
  } catch {
    return false;
  }
}

async function getPipe(engine: "summarization" | "chat", model: string, device: "webgpu" | "wasm", dtypeSpec?: { webgpu: DtypeSpec; wasm: DtypeSpec }): Promise<Pipe> {
  const dtype: DtypeSpec = dtypeSpec ? dtypeSpec[device] : "q8";
  const key = `${engine}:${model}@${device}:${JSON.stringify(dtype)}`;
  if (cached && cached.key === key) return cached.fn;
  const options = { device, dtype, progress_callback: downloadProgress() };
  const kind = engine === "summarization" ? "summarization" : "text-generation";
  const fn = (await pipeline(kind, model, options as never)) as unknown as Pipe;
  cached = { key, fn };
  return fn;
}

// Estimate the input's token count (for the max_new_tokens cap) from the model's tokenizer,
// falling back to a char-based guess.
function tokenCount(pipe: Pipe, text: string): number {
  try {
    const ids = pipe.tokenizer?.(text).input_ids;
    const n = ids?.size ?? ids?.dims?.[ids.dims.length - 1];
    if (typeof n === "number" && n > 0) return n;
  } catch {
    /* fall through */
  }
  return Math.ceil(text.length / 4);
}

onMessage(async (e: MessageEvent) => {
  const msg = e.data as { type: string };
  if (msg.type !== "run") return;
  const run = e.data as RunMsg;

  // A model that needs WebGPU hands a CPU run to its fallback, which brings its own dtypes.
  const onCpu = (): { model: string; dtype?: { webgpu: DtypeSpec; wasm: DtypeSpec } } => {
    const fallback = genModel(run.model)?.wasmFallback;
    return fallback ? { model: fallback, dtype: genModel(fallback)?.dtype } : { model: run.model, dtype: run.dtype };
  };

  /** Load the model on one backend and generate the whole answer there. */
  const runOn = async (device: "webgpu" | "wasm"): Promise<string> => {
    const target = device === "wasm" ? onCpu() : { model: run.model, dtype: run.dtype };
    const pipe = await getPipe(run.engine, target.model, device, target.dtype);
    post({ type: "device", device });

    // Stream generated text as cumulative snapshots so the UI just renders the latest.
    let acc = "";
    const streamer = new TextStreamer((pipe as unknown as { tokenizer: unknown }).tokenizer as never, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        acc += text;
        post({ type: "partial", text: acc });
      },
    } as never);

    const cap = capFor(run.task, tokenCount(pipe, run.input));

    if (run.engine === "summarization") {
      const out = (await pipe(run.input, { max_new_tokens: cap, streamer, no_repeat_ngram_size: 3 })) as { summary_text?: string }[] | { summary_text?: string };
      if (!acc) {
        const arr = Array.isArray(out) ? out : [out];
        acc = arr[0]?.summary_text ?? "";
        post({ type: "partial", text: acc });
      }
    } else {
      const messages = buildMessages(run.task, run.input, run.system);
      // No repetition penalty and no no-repeat n-grams: both count the prompt, so they forbid the
      // very words a rewrite has to keep, and pushed every model tried into paraphrase or nonsense.
      const out = (await pipe(messages, {
        max_new_tokens: cap,
        streamer,
        do_sample: false,
      })) as { generated_text?: unknown }[] | { generated_text?: unknown };
      if (!acc) {
        // Fallback if streaming produced nothing: dig the assistant turn out of the result.
        const arr = Array.isArray(out) ? out : [out];
        const gen = arr[0]?.generated_text;
        if (Array.isArray(gen)) {
          const last = gen[gen.length - 1] as { content?: string };
          acc = last?.content ?? "";
        } else if (typeof gen === "string") {
          acc = gen;
        }
        post({ type: "partial", text: acc });
      }
    }
    return acc;
  };

  try {
    let device: "webgpu" | "wasm" = run.device ?? ((await hasWebGpu()) ? "webgpu" : "wasm");
    // A model that needs f16 shaders skips a GPU without them: on the phones tested it loaded and
    // then lost the device part way through, after downloading the model for nothing.
    if (!run.device && device === "webgpu" && genModel(run.model)?.needsF16 && !(await gpuHasF16())) device = "wasm";
    // One backend per worker. A GPU that fails leaves onnxruntime unusable in this worker, so the
    // driver (generate.ts) retries on the CPU in a fresh one rather than here.
    const text = await runOn(device);
    // Only the answer belongs in the document, not the wrapping a small model adds around it.
    post({ type: "done", text: run.engine === "chat" ? cleanReply(text) : text.trim() });
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
});
