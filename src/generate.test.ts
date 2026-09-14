import { afterEach, describe, expect, it, vi } from "vitest";
import { chatTasksAvailable } from "./generate";

// The chat tasks are only offered where their model runs: a GPU with f16 shaders.
describe("chatTasksAvailable", () => {
  afterEach(() => vi.unstubAllGlobals());

  const withGpu = (gpu: unknown): void => vi.stubGlobal("navigator", { ...globalThis.navigator, gpu });

  it("offers them on a GPU with f16 shaders", async () => {
    withGpu({ requestAdapter: async () => ({ features: new Set(["shader-f16"]) }) });
    expect(await chatTasksAvailable()).toBe(true);
  });

  it("withholds them on a GPU without f16 shaders, as most phones have", async () => {
    withGpu({ requestAdapter: async () => ({ features: new Set<string>() }) });
    expect(await chatTasksAvailable()).toBe(false);
  });

  it("withholds them with no WebGPU at all, or when asking for an adapter fails", async () => {
    withGpu(undefined);
    expect(await chatTasksAvailable()).toBe(false);
    withGpu({ requestAdapter: async () => null });
    expect(await chatTasksAvailable()).toBe(false);
    withGpu({ requestAdapter: async () => { throw new Error("blocked"); } });
    expect(await chatTasksAvailable()).toBe(false);
  });
});
