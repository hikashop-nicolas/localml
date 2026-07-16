// Translation model catalog and language-code mapping, extracted from subedit. The engine
// (transformers.js, in translate.worker.ts) turns an array of strings into their
// translations; this module just describes the available models and the languages they
// support, keeping the worker/driver engine-agnostic.

// A transformers.js dtype spec: one dtype for all sub-models, or per sub-model.
export type DtypeSpec = string | Record<string, string>;

export type TranscribeProgress =
  | { stage: "download"; ratio: number; file?: string }
  | { stage: "translate"; ratio: number };

export interface TranslateModelInfo {
  id: string;
  label: string;
  sizeMb: number;
  descKey: string;
  scheme: "iso" | "flores"; // which language-code scheme the model expects
  dtype: { webgpu: DtypeSpec; wasm: DtypeSpec };
}

// q8 for both GPU and CPU. Tested q4f16 on WebGPU (4-bit decoder): it wrecked translation
// quality (output degenerated into punctuation and repeated words). fp16 keeps quality but
// roughly doubles the download for no quality gain over q8, which already runs fine on WebGPU.
// So q8 is the sweet spot on both backends.
export const TRANSLATE_MODELS: TranslateModelInfo[] = [
  { id: "Xenova/m2m100_418M", label: "M2M-100", sizeMb: 500, descKey: "mtDescM2m", scheme: "iso", dtype: { webgpu: "q8", wasm: "q8" } },
  { id: "Xenova/nllb-200-distilled-600M", label: "NLLB-200", sizeMb: 800, descKey: "mtDescNllb", scheme: "flores", dtype: { webgpu: "q8", wasm: "q8" } },
];

export const DEFAULT_TRANSLATE_MODEL = "Xenova/m2m100_418M";

export function translateModel(id: string): TranslateModelInfo | undefined {
  return TRANSLATE_MODELS.find((m) => m.id === id);
}

// The languages offered for source/target, with each model's code for them.
export const TRANSLATE_LANGS: { code: string; label: string; iso: string; flores: string }[] = [
  { code: "en", label: "English", iso: "en", flores: "eng_Latn" },
  { code: "fr", label: "Français", iso: "fr", flores: "fra_Latn" },
  { code: "ja", label: "日本語", iso: "ja", flores: "jpn_Jpan" },
  { code: "es", label: "Español", iso: "es", flores: "spa_Latn" },
  { code: "de", label: "Deutsch", iso: "de", flores: "deu_Latn" },
  { code: "it", label: "Italiano", iso: "it", flores: "ita_Latn" },
  { code: "pt", label: "Português", iso: "pt", flores: "por_Latn" },
  { code: "nl", label: "Nederlands", iso: "nl", flores: "nld_Latn" },
  { code: "ru", label: "Русский", iso: "ru", flores: "rus_Cyrl" },
  { code: "zh", label: "中文", iso: "zh", flores: "zho_Hans" },
  { code: "ko", label: "한국어", iso: "ko", flores: "kor_Hang" },
  { code: "ar", label: "العربية", iso: "ar", flores: "arb_Arab" },
];

// Map a common code (e.g. "fr") to what the chosen translation model expects.
export function mtLangCode(scheme: "iso" | "flores", common: string): string {
  const l = TRANSLATE_LANGS.find((x) => x.code === common);
  return l ? l[scheme] : common;
}
