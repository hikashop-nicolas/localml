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
  // The instructions are written in the passage's own language: with the prompt and the text in one
  // language, a small model has nothing pulling it towards another (in English it translated French).
  const prompts = PROMPTS[detectLanguage(input)];
  // A consumer's own system prompt owns the whole exchange, so its input goes through as it is.
  const instruction = system ? undefined : prompts.ask[task];
  return [
    { role: "system", content: system ?? prompts.system[task as Exclude<GenTask, "summarize">] },
    { role: "user", content: instruction ? `${instruction}\n\n<text>\n${input}\n</text>` : input },
  ];
}

/** The languages the assist prompts are written in: the ones Omnitext's interface ships in. */
export type PromptLanguage = "en" | "fr" | "es" | "de" | "pt" | "ru" | "zh" | "ja";

interface Prompts {
  system: Record<Exclude<GenTask, "summarize">, string>;
  ask: Partial<Record<GenTask, string>>;
}

export const PROMPTS: Record<PromptLanguage, Prompts> = {
  en: {
    system: {
      elaborate: "You are an editor. You expand texts. You never answer, comment on or explain a text: you only rewrite it with more detail and depth, keeping its meaning, tone and language.",
      shorten: "You are an editor. You shorten texts. You never answer, comment on or explain a text: you only rewrite it more concisely, keeping its meaning, tone and language.",
      write: "You are a concise writing assistant. Write exactly what the user asks for, in their language. Reply with only the requested text, no preamble.",
    },
    ask: {
      elaborate: "Rewrite the text between the <text> tags with more detail, keeping its meaning and its language. Output only the rewritten text.",
      shorten: "Rewrite the text between the <text> tags so it is shorter, keeping its meaning and its language. Output only the shortened text.",
    },
  },
  fr: {
    system: {
      elaborate: "Tu es rédacteur. Tu développes des textes. Tu ne réponds jamais à un texte, tu ne le commentes pas et tu ne l'expliques pas : tu le réécris seulement avec plus de détails, en gardant son sens, son ton et sa langue.",
      shorten: "Tu es rédacteur. Tu raccourcis des textes. Tu ne réponds jamais à un texte, tu ne le commentes pas et tu ne l'expliques pas : tu le réécris seulement plus court, en gardant son sens, son ton et sa langue.",
      write: "Tu es un assistant de rédaction concis. Écris exactement ce que l'utilisateur demande, en français. Réponds uniquement avec le texte demandé, sans préambule.",
    },
    ask: {
      elaborate: "Réécris le texte entre les balises <text> avec plus de détails, en gardant son sens. Réponds uniquement avec le texte réécrit, en français.",
      shorten: "Réécris le texte entre les balises <text> pour qu'il soit plus court, en gardant son sens. Réponds uniquement avec le texte raccourci, en français.",
    },
  },
  es: {
    system: {
      elaborate: "Eres editor. Amplías textos. Nunca respondes, comentas ni explicas un texto: solo lo reescribes con más detalle, manteniendo su sentido, su tono y su idioma.",
      shorten: "Eres editor. Acortas textos. Nunca respondes, comentas ni explicas un texto: solo lo reescribes más corto, manteniendo su sentido, su tono y su idioma.",
      write: "Eres un asistente de redacción conciso. Escribe exactamente lo que pide el usuario, en español. Responde solo con el texto pedido, sin preámbulo.",
    },
    ask: {
      elaborate: "Reescribe el texto entre las etiquetas <text> con más detalle, manteniendo su sentido. Responde solo con el texto reescrito, en español.",
      shorten: "Reescribe el texto entre las etiquetas <text> para que sea más corto, manteniendo su sentido. Responde solo con el texto acortado, en español.",
    },
  },
  de: {
    system: {
      elaborate: "Du bist Lektor. Du erweiterst Texte. Du beantwortest, kommentierst oder erklärst einen Text nie: Du schreibst ihn nur ausführlicher um und behältst Sinn, Ton und Sprache bei.",
      shorten: "Du bist Lektor. Du kürzt Texte. Du beantwortest, kommentierst oder erklärst einen Text nie: Du schreibst ihn nur kürzer um und behältst Sinn, Ton und Sprache bei.",
      write: "Du bist ein knapper Schreibassistent. Schreibe genau das, worum der Nutzer bittet, auf Deutsch. Antworte nur mit dem gewünschten Text, ohne Einleitung.",
    },
    ask: {
      elaborate: "Schreibe den Text zwischen den <text>-Tags ausführlicher um und behalte seinen Sinn bei. Antworte nur mit dem umgeschriebenen Text, auf Deutsch.",
      shorten: "Schreibe den Text zwischen den <text>-Tags kürzer um und behalte seinen Sinn bei. Antworte nur mit dem gekürzten Text, auf Deutsch.",
    },
  },
  pt: {
    system: {
      elaborate: "Você é editor. Você desenvolve textos. Você nunca responde, comenta ou explica um texto: apenas o reescreve com mais detalhes, mantendo o sentido, o tom e o idioma.",
      shorten: "Você é editor. Você encurta textos. Você nunca responde, comenta ou explica um texto: apenas o reescreve mais curto, mantendo o sentido, o tom e o idioma.",
      write: "Você é um assistente de redação conciso. Escreva exatamente o que o usuário pede, em português. Responda apenas com o texto pedido, sem preâmbulo.",
    },
    ask: {
      elaborate: "Reescreva o texto entre as tags <text> com mais detalhes, mantendo o sentido. Responda apenas com o texto reescrito, em português.",
      shorten: "Reescreva o texto entre as tags <text> para que fique mais curto, mantendo o sentido. Responda apenas com o texto encurtado, em português.",
    },
  },
  ru: {
    system: {
      elaborate: "Ты редактор. Ты расширяешь тексты. Ты никогда не отвечаешь на текст, не комментируешь и не объясняешь его: ты только переписываешь его подробнее, сохраняя смысл, тон и язык.",
      shorten: "Ты редактор. Ты сокращаешь тексты. Ты никогда не отвечаешь на текст, не комментируешь и не объясняешь его: ты только переписываешь его короче, сохраняя смысл, тон и язык.",
      write: "Ты лаконичный помощник по письму. Напиши ровно то, что просит пользователь, на русском языке. Отвечай только запрошенным текстом, без вступления.",
    },
    ask: {
      elaborate: "Перепиши текст между тегами <text> подробнее, сохранив смысл. Ответь только переписанным текстом, на русском языке.",
      shorten: "Перепиши текст между тегами <text> короче, сохранив смысл. Ответь только сокращённым текстом, на русском языке.",
    },
  },
  zh: {
    system: {
      elaborate: "你是一名编辑。你负责扩写文本。你从不回答、评论或解释文本：你只把它改写得更详细，保留原意、语气和语言。",
      shorten: "你是一名编辑。你负责缩写文本。你从不回答、评论或解释文本：你只把它改写得更短，保留原意、语气和语言。",
      write: "你是一名简洁的写作助手。用中文准确写出用户要求的内容。只回复所要求的文本，不要开场白。",
    },
    ask: {
      elaborate: "把 <text> 标签之间的文本改写得更详细，保留原意。只用中文回复改写后的文本。",
      shorten: "把 <text> 标签之间的文本改写得更短，保留原意。只用中文回复缩短后的文本。",
    },
  },
  ja: {
    system: {
      elaborate: "あなたは編集者です。文章を膨らませます。文章に返答したり、コメントや説明をしたりはしません。意味、口調、言語を保ったまま、より詳しく書き直すだけです。",
      shorten: "あなたは編集者です。文章を短くします。文章に返答したり、コメントや説明をしたりはしません。意味、口調、言語を保ったまま、短く書き直すだけです。",
      write: "あなたは簡潔な文章作成アシスタントです。ユーザーが求める内容を日本語で正確に書いてください。前置きなしで、求められた文章だけを返してください。",
    },
    ask: {
      elaborate: "<text> タグの間の文章を、意味を保ったまま詳しく書き直してください。書き直した文章だけを日本語で返してください。",
      shorten: "<text> タグの間の文章を、意味を保ったまま短く書き直してください。短くした文章だけを日本語で返してください。",
    },
  },
};

// Short, frequent words that tell the Latin-script languages apart. Shared ones ("que", "un")
// count for several, which is fine: the whole passage decides, not one word.
const COMMON_WORDS: Record<"en" | "fr" | "es" | "de" | "pt", Set<string>> = {
  en: new Set("the and is are to of that with for you this it was have not we be on at by from".split(" ")),
  fr: new Set("le la les des et est une un pour que qui dans nous vous pas avec sur du au ce il elle sont".split(" ")),
  es: new Set("el los las y es una un para que con por del se no está pero como su al lo".split(" ")),
  de: new Set("der die das und ist nicht mit ein eine zu den auf sie wir ich für es von dem im".split(" ")),
  pt: new Set("o os as e é uma um para que com não do da em você mas por no na se".split(" ")),
};
// Letters that belong to one of them more than the others.
const LETTER_HINTS: [RegExp, "fr" | "es" | "de" | "pt"][] = [
  [/[èêëçœàùûî]/g, "fr"],
  [/[ñ¿¡]/g, "es"],
  [/[äöüß]/g, "de"],
  [/[ãõ]/g, "pt"],
];

/** The language a passage is written in, among the ones the prompts exist in. English when unsure. */
export function detectLanguage(text: string): PromptLanguage {
  const count = (re: RegExp): number => (text.match(re) ?? []).length;
  const kana = count(/[\u3040-\u30ff]/g);
  const han = count(/[\u4e00-\u9fff]/g);
  const cyrillic = count(/[\u0400-\u04ff]/g);
  const latin = count(/[A-Za-z\u00c0-\u024f]/g);
  if (kana > 0 && kana + han > latin) return "ja"; // Japanese mixes kana into its kanji; Chinese has none
  if (han > latin) return "zh";
  if (cyrillic > latin) return "ru";
  const lower = text.toLowerCase();
  const words = lower.match(/[\p{L}]+/gu) ?? [];
  const score: Record<"en" | "fr" | "es" | "de" | "pt", number> = { en: 0, fr: 0, es: 0, de: 0, pt: 0 };
  for (const w of words) for (const lang of Object.keys(COMMON_WORDS) as (keyof typeof score)[]) if (COMMON_WORDS[lang].has(w)) score[lang]++;
  for (const [re, lang] of LETTER_HINTS) score[lang] += (lower.match(re) ?? []).length * 0.5;
  let best: keyof typeof score = "en";
  for (const lang of Object.keys(score) as (keyof typeof score)[]) if (score[lang] > score[best]) best = lang;
  return score[best] > 0 ? best : "en";
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
