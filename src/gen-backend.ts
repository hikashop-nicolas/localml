// Text-generation model catalog and prompt building, kept engine-agnostic (the worker in
// generate.worker.ts runs transformers.js). Two engines: a purpose-built summarizer
// (encoder-decoder) and a small instruction-tuned chat model that handles the open-ended
// tasks (elaborate / shorten / write) from a task-specific prompt.

export type DtypeSpec = string | Record<string, string>;

/** The assist tasks. `summarize` uses the summarizer; the rest use the chat model. */
export type GenTask = "summarize" | "elaborate" | "shorten" | "write";

export type GenProgress =
  | { stage: "download"; ratio: number; file?: string }
  | { stage: "generate"; ratio: number };

export interface GenModelInfo {
  id: string;
  label: string;
  sizeMb: number;
  /** Which transformers.js pipeline the worker builds for this model. */
  engine: "summarization" | "chat";
  dtype: { webgpu: DtypeSpec; wasm: DtypeSpec };
}

// distilbart for summarize (q8 keeps the download small, quality holds for short summaries).
// Qwen2.5-0.5B-Instruct for the chat tasks: multilingual and small. q8 on both backends:
// q4f16 collapsed a 0.5B model into repeated tokens/gibberish on WebGPU (same failure the
// translator saw with 4-bit), so q8 is the quality floor here. (Alternates if it
// underperforms: SmolLM2-360M-Instruct, Llama-3.2-1B-Instruct.)
export const GEN_MODELS: GenModelInfo[] = [
  { id: "Xenova/distilbart-cnn-6-6", label: "DistilBART", sizeMb: 150, engine: "summarization", dtype: { webgpu: "q8", wasm: "q8" } },
  { id: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen2.5 0.5B", sizeMb: 550, engine: "chat", dtype: { webgpu: "q8", wasm: "q8" } },
];

const SUMMARIZER = "Xenova/distilbart-cnn-6-6";
const CHAT = "onnx-community/Qwen2.5-0.5B-Instruct";

/** The default model for each task. */
export const TASK_MODEL: Record<GenTask, string> = {
  summarize: SUMMARIZER,
  elaborate: CHAT,
  shorten: CHAT,
  write: CHAT,
};

export function genModel(id: string): GenModelInfo | undefined {
  return GEN_MODELS.find((m) => m.id === id);
}

export interface ChatMessage {
  role: "system" | "user";
  content: string;
}

// Task -> chat prompt. Each keeps the model on a tight leash: reply with only the result, in
// the same language as the input (Qwen is multilingual). `input` is the selected text for
// elaborate/shorten and the user's instruction for write.
export function buildMessages(task: GenTask, input: string): ChatMessage[] {
  const sys: Record<Exclude<GenTask, "summarize">, string> = {
    elaborate:
      "You expand the user's text, adding relevant detail and depth while keeping its meaning, tone and language. Reply with only the expanded text, no preamble.",
    shorten:
      "You rewrite the user's text to be more concise while keeping its meaning, tone and language. Reply with only the shortened text, no preamble.",
    write:
      "You are a concise writing assistant. Write exactly what the user asks for, in their language. Reply with only the requested text, no preamble.",
  };
  return [
    { role: "system", content: sys[task as Exclude<GenTask, "summarize">] },
    { role: "user", content: input },
  ];
}

// A token budget for the chat tasks, sized to the input so the model can't ramble forever.
// elaborate can grow, shorten must shrink, write is open-ended with a flat cap.
export function capFor(task: GenTask, approxInputTokens: number): number {
  const n = Math.max(0, Math.round(approxInputTokens));
  switch (task) {
    case "elaborate":
      return clamp(n * 2 + 80, 96, 1024);
    case "shorten":
      return clamp(Math.round(n * 0.9) + 24, 32, 512);
    case "write":
      return 384;
    case "summarize":
      return clamp(Math.round(n * 0.5) + 24, 32, 256);
  }
}

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

/** Rough token estimate when a tokenizer isn't handy (~4 chars/token). */
export const approxTokens = (text: string): number => Math.ceil(text.length / 4);
