import { runOcr } from "../src/ocr";
import { runTranslate, TRANSLATE_LANGS, TRANSLATE_MODELS, DEFAULT_TRANSLATE_MODEL } from "../src/translate";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- OCR ----
const ocrCanvas = $<HTMLCanvasElement>("ocrcanvas");
const ocrProg = $("ocrprog");
const ocrOut = $("ocrout");

function drawText(text: string): void {
  const ctx = ocrCanvas.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, ocrCanvas.width, ocrCanvas.height);
  ctx.fillStyle = "#000";
  ctx.font = "56px Georgia, serif";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 20, ocrCanvas.height / 2);
}

async function ocr(image: CanvasImageSource | Blob): Promise<void> {
  ocrOut.textContent = "";
  const lang = $<HTMLSelectElement>("ocrlang").value;
  const run = runOcr(image as never, {
    lang,
    onProgress: (p) => (ocrProg.textContent = `${p.stage} ${Math.round(p.ratio * 100)}%`),
  });
  try {
    const { text, confidence } = await run.done;
    ocrProg.textContent = `done (confidence ${Math.round(confidence)}%)`;
    ocrOut.textContent = text;
  } catch (e) {
    ocrProg.textContent = `error: ${(e as Error).message}`;
  }
}

$("ocrrender").addEventListener("click", () => {
  drawText($<HTMLInputElement>("ocrtext").value);
  void ocr(ocrCanvas);
});

$<HTMLInputElement>("ocrfile").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void ocr(f);
});

// ?src=<url> loads and OCRs an image straight away (handy for testing).
const src = new URLSearchParams(location.search).get("src");
if (src) void fetch(src).then((r) => r.blob()).then((b) => ocr(b));

// ---- Translate ----
const tFrom = $<HTMLSelectElement>("tfrom");
const tTo = $<HTMLSelectElement>("tto");
const tModel = $<HTMLSelectElement>("tmodel");
for (const l of TRANSLATE_LANGS) {
  tFrom.add(new Option(l.label, l.code));
  tTo.add(new Option(l.label, l.code));
}
tFrom.value = "fr";
tTo.value = "en";
for (const m of TRANSLATE_MODELS) tModel.add(new Option(m.label, m.id));
tModel.value = DEFAULT_TRANSLATE_MODEL;

$("tgo").addEventListener("click", () => {
  const tProg = $("tprog");
  const tOut = $("tout");
  const lines = $<HTMLTextAreaElement>("tsrc").value.split("\n");
  tOut.textContent = "";
  const run = runTranslate(
    lines,
    { model: tModel.value, srcLang: tFrom.value, tgtLang: tTo.value },
    {
      onProgress: (p) => (tProg.textContent = `${p.stage} ${Math.round(p.ratio * 100)}%`),
      onPartial: (start, texts) => {
        const cur = tOut.textContent ?? "";
        tOut.textContent = cur + (start > 0 ? "\n" : "") + texts.join("\n");
      },
      onDevice: (d) => (tProg.textContent = `running on ${d}…`),
    },
  );
  void run.done.then(() => (tProg.textContent = "done"));
});
