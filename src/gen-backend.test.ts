import { describe, expect, it } from "vitest";
import { GEN_MODELS, TASK_MODEL, approxTokens, buildMessages, capFor, cleanReply, genModel } from "./gen-backend";

describe("gen-backend catalog", () => {
  it("maps every task to a model that exists in the catalog", () => {
    for (const model of Object.values(TASK_MODEL)) {
      expect(genModel(model)).toBeDefined();
    }
  });

  it("routes summarize to the summarizer and the rest to the chat model", () => {
    expect(genModel(TASK_MODEL.summarize)?.engine).toBe("summarization");
    for (const task of ["elaborate", "shorten", "write"] as const) {
      expect(genModel(TASK_MODEL[task])?.engine).toBe("chat");
    }
  });

  it("gives a model that needs WebGPU a fallback that runs on the CPU", () => {
    for (const m of GEN_MODELS) {
      if (!m.wasmFallback) continue;
      const fallback = genModel(m.wasmFallback);
      expect(fallback, m.wasmFallback).toBeDefined();
      expect(fallback!.engine).toBe(m.engine);
      expect(fallback!.wasmFallback).toBeUndefined(); // one hop, never a chain
    }
    expect(genModel(TASK_MODEL.shorten)?.webgpuNoF16).toBeTruthy(); // phones without f16 shaders
  });

  it("declares a webgpu and wasm dtype for every model", () => {
    for (const m of GEN_MODELS) {
      expect(m.dtype.webgpu).toBeTruthy();
      expect(m.dtype.wasm).toBeTruthy();
    }
  });
});

describe("buildMessages", () => {
  it("puts a task-specific system prompt first, then the user input", () => {
    const msgs = buildMessages("elaborate", "the cat sat");
    expect(msgs).toHaveLength(2);
    expect(msgs[0]!.role).toBe("system");
    expect(msgs[0]!.content.toLowerCase()).toContain("expand");
    // The passage goes inside tags beside its instruction: sent bare, a small model answers it.
    expect(msgs[1]!.role).toBe("user");
    expect(msgs[1]!.content).toContain("<text>\nthe cat sat\n</text>");
    expect(msgs[1]!.content.toLowerCase()).toContain("rewrite");
  });

  it("wraps a passage to shorten the same way, and leaves a write instruction as it is", () => {
    expect(buildMessages("shorten", "long words here")[1]!.content).toContain("<text>\nlong words here\n</text>");
    expect(buildMessages("write", "a thank-you note")[1]).toEqual({ role: "user", content: "a thank-you note" });
  });

  it("uses different system prompts per task", () => {
    const sys = (t: "elaborate" | "shorten" | "write") => buildMessages(t, "x")[0]!.content.toLowerCase();
    expect(sys("shorten")).toContain("concise");
    expect(sys("write")).toContain("writing assistant");
    expect(sys("elaborate")).not.toBe(sys("shorten"));
  });

  it("lets a caller override the system prompt (domain specialisation)", () => {
    const msgs = buildMessages("write", "sum column B", "You write spreadsheet formulas.");
    expect(msgs[0]).toEqual({ role: "system", content: "You write spreadsheet formulas." });
    expect(msgs[1]).toEqual({ role: "user", content: "sum column B" });
  });
});

describe("cleanReply", () => {
  it("drops an announcing line and the rules a small model puts around its answer", () => {
    const raw = "Sure! Here's a shortened version of your text:\n\n---\n\nThanks for the quick reply. Please fix the date.\n\n---";
    expect(cleanReply(raw)).toBe("Thanks for the quick reply. Please fix the date.");
  });

  it("drops a thinking block, echoed tags and quotes around the whole answer", () => {
    expect(cleanReply("<think>\nplan\n</think>\n<text>\nShort text.\n</text>")).toBe("Short text.");
    expect(cleanReply("\u201cNous avons re\u00e7u votre demande.\u201d")).toBe("Nous avons re\u00e7u votre demande.");
  });

  it("leaves an answer alone when nothing wraps it", () => {
    const plain = "Here is the fix: add two lines of CSS.";
    expect(cleanReply(plain)).toBe(plain); // a colon inside the sentence, no line break: not a preamble
    expect(cleanReply('He said "no" and "yes".')).toBe('He said "no" and "yes".');
  });
});

describe("capFor", () => {
  it("gives elaborate room to grow and keeps shorten near input length, within bounds", () => {
    expect(capFor("elaborate", 100)).toBeGreaterThan(100); // room to expand
    // The cap is a safety ceiling (the prompt drives shortening); a shorten cap below the
    // input would truncate a valid concise rewrite, so it sits near input, not below.
    expect(capFor("shorten", 100)).toBeLessThan(capFor("elaborate", 100));
    expect(capFor("shorten", 100)).toBeLessThanOrEqual(140);
    expect(capFor("elaborate", 100000)).toBeLessThanOrEqual(1024); // clamped
    expect(capFor("shorten", 0)).toBeGreaterThanOrEqual(32); // floor
  });

  it("gives write a flat budget regardless of input size", () => {
    expect(capFor("write", 5)).toBe(capFor("write", 5000));
  });
});

describe("approxTokens", () => {
  it("estimates ~4 chars per token", () => {
    expect(approxTokens("abcd")).toBe(1);
    expect(approxTokens("a".repeat(40))).toBe(10);
  });
});
