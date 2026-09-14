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
  /** Only run on a GPU with shader-f16. Phone GPUs without it also tend to lose the device on a
      model this size, so they go straight to `wasmFallback` instead of downloading both. */
  needsF16?: boolean;
  /** The model to run instead where only WASM is available: this one's quantized embeddings need WebGPU. */
  wasmFallback?: string;
}

// distilbart for summarize (q8 keeps the download small, quality holds for short summaries).
// Qwen3.5-0.8B for the chat tasks: of the models that fit a browser, the one that rewrites a
// passage instead of answering it, in the passage's own language. Its quantized embeddings need
// WebGPU, so without a GPU the tasks fall back to Qwen2.5-0.5B, which runs anywhere but often
// replies to the text rather than rewriting it.
export const GEN_MODELS: GenModelInfo[] = [
  { id: "Xenova/distilbart-cnn-6-6", label: "DistilBART", sizeMb: 150, engine: "summarization", dtype: { webgpu: "q8", wasm: "q8" } },
  {
    id: "onnx-community/Qwen3.5-0.8B-Text-ONNX", label: "Qwen3.5 0.8B", sizeMb: 470, engine: "chat",
    dtype: { webgpu: "q4f16", wasm: "q4" }, needsF16: true, wasmFallback: "onnx-community/Qwen2.5-0.5B-Instruct",
  },
  { id: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen2.5 0.5B", sizeMb: 550, engine: "chat", dtype: { webgpu: "q8", wasm: "q8" } },
];

const SUMMARIZER = "Xenova/distilbart-cnn-6-6";
const CHAT = "onnx-community/Qwen3.5-0.8B-Text-ONNX";

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
// the same language as the input. `input` is the selected text for elaborate/shorten and the
// user's instruction for write. A passage goes inside tags with the instruction beside it in the
// user turn: sent bare, a small model takes the passage for a message and answers it.
// `system` lets a consumer supply a domain-specific system prompt (e.g. a spreadsheet-
// formula assistant) without localml having to know that domain; it overrides the task
// default. The task still drives the token budget (capFor).
export function buildMessages(task: GenTask, input: string, system?: string): ChatMessage[] {
  const sys: Record<Exclude<GenTask, "summarize">, string> = {
    elaborate:
      "You are an editor. You expand texts. You never answer, comment on or explain a text: you only rewrite it with more detail and depth, keeping its meaning, tone and language.",
    shorten:
      "You are an editor. You shorten texts. You never answer, comment on or explain a text: you only rewrite it more concisely, keeping its meaning, tone and language.",
    write:
      "You are a concise writing assistant. Write exactly what the user asks for, in their language. Reply with only the requested text, no preamble.",
  };
  const ask: Partial<Record<GenTask, string>> = {
    elaborate: "Rewrite the text between the <text> tags with more detail, keeping its meaning and its language. Output only the rewritten text.",
    shorten: "Rewrite the text between the <text> tags so it is shorter, keeping its meaning and its language. Output only the shortened text.",
  };
  // A consumer's own system prompt owns the whole exchange, so its input goes through as it is.
  const instruction = system ? undefined : ask[task];
  return [
    { role: "system", content: system ?? sys[task as Exclude<GenTask, "summarize">] },
    { role: "user", content: instruction ? `${instruction}\n\n<text>\n${input}\n</text>` : input },
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

/**
 * The model's answer without the wrapping a small model likes to add around it: a thinking block,
 * a "Sure! Here's a shorter version:" line, horizontal rules, the <text> tags from the prompt, or
 * quotes around the whole answer. Only wrapping is removed; the text inside is left alone.
 */
export function cleanReply(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  // A preamble is one short opening line that announces the answer and ends with a colon.
  const preamble = /^(?:sure|certainly|of course|okay|ok|here(?:'|\u2019)?s|here is|voici|bien s\u00fbr|claro|aqu\u00ed|gerne|hier ist)\b[^\n]{0,120}:[ \t]*\n+/i;
  out = out.replace(preamble, "").trim();
  out = out.replace(/^<text>\s*/i, "").replace(/\s*<\/text>$/i, "").trim();
  out = out.replace(/^(?:-{3,}|\*{3,})[ \t]*\n+/, "").replace(/\n+[ \t]*(?:-{3,}|\*{3,})$/, "").trim();
  const quoted = out.match(/^(["\u201c\u00ab])([\s\S]*)(["\u201d\u00bb])$/);
  if (quoted && !quoted[2]!.includes(quoted[1]!)) out = quoted[2]!.trim();
  return out;
}

/** Rough token estimate when a tokenizer isn't handy (~4 chars/token). */
export const approxTokens = (text: string): number => Math.ceil(text.length / 4);
