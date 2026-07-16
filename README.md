# localml

Shared **on-device** (in-browser) machine learning for text, used by
[imageview](https://github.com/hikashop-nicolas/imageview) and
[subedit](https://github.com/hikashop-nicolas/subedit). Everything runs on the user's
machine, no server. Two independent entry points, each loading its engine on demand:

- `localml/ocr` : OCR via [Tesseract.js](https://github.com/naptha/tesseract.js/). Tesseract
  runs its own WASM worker and fetches its core + language data from a CDN on first use
  (then browser-cached).
- `localml/translate` : text translation via [transformers.js](https://github.com/huggingface/transformers.js)
  (m2m100-418M or NLLB-200-distilled-600M), in a worker, WebGPU with a WASM (CPU) fallback.
  Models download from the Hugging Face CDN on first use and are cached.

## Usage

### OCR

```ts
import { runOcr } from "localml/ocr";

const run = runOcr(imageBlobOrImgElement, {
  lang: "eng",                       // or "fra", "jpn", "eng+fra", …
  onProgress: (p) => console.log(p), // { stage: "load" | "recognize", ratio }
});
const { text, words } = await run.done;
// run.cancel() terminates early
```

### Translate

```ts
import { runTranslate, TRANSLATE_LANGS } from "localml/translate";

const run = runTranslate(["Bonjour", "le monde"], { model: "Xenova/m2m100_418M", srcLang: "fr", tgtLang: "en" }, {
  onProgress: (p) => console.log(p),
  onPartial: (start, texts) => console.log(start, texts), // streamed per line
});
await run.done;      // { stopped }
// run.pause() / run.resume() / run.cancel()
```

## Develop

```bash
npm install
npm run dev        # demo at the printed URL
npm run typecheck
npm test
npm run build      # emits dist/ (also runs on install via prepare)
```

## License

MIT
