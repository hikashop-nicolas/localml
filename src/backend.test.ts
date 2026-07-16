import { describe, it, expect } from "vitest";
import { mtLangCode, translateModel, TRANSLATE_LANGS, TRANSLATE_MODELS, DEFAULT_TRANSLATE_MODEL } from "./backend";

describe("translate catalog", () => {
  it("maps common codes to the model scheme", () => {
    expect(mtLangCode("iso", "fr")).toBe("fr");
    expect(mtLangCode("flores", "fr")).toBe("fra_Latn");
    expect(mtLangCode("flores", "ja")).toBe("jpn_Jpan");
  });

  it("passes through unknown codes unchanged", () => {
    expect(mtLangCode("iso", "xx")).toBe("xx");
    expect(mtLangCode("flores", "xx")).toBe("xx");
  });

  it("resolves a model by id and knows the default", () => {
    expect(translateModel(DEFAULT_TRANSLATE_MODEL)?.scheme).toBe("iso");
    expect(translateModel("Xenova/nllb-200-distilled-600M")?.scheme).toBe("flores");
    expect(translateModel("nope")).toBeUndefined();
  });

  it("every language has both scheme codes", () => {
    for (const l of TRANSLATE_LANGS) {
      expect(l.iso).toBeTruthy();
      expect(l.flores).toMatch(/_/);
    }
    expect(TRANSLATE_MODELS.length).toBeGreaterThanOrEqual(2);
  });
});
