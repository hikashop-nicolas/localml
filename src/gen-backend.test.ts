import { describe, expect, it } from "vitest";
import { GEN_MODELS, TASK_MODEL, approxTokens, buildMessages, capFor, genModel } from "./gen-backend";

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
    expect(msgs[1]).toEqual({ role: "user", content: "the cat sat" });
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
