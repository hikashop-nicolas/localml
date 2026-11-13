import { afterEach, describe, expect, it, vi } from "vitest";
import { setRemoteConsentHandler } from "./consent";
import { fetchModelFiles, ModelFilesDeclined, modelFileUrl } from "./files";

const files = [{ name: "a.onnx", bytes: 3 }, { name: "b.json", bytes: 2 }];

afterEach(() => {
  setRemoteConsentHandler(null);
  vi.unstubAllGlobals();
});

describe("fetchModelFiles", () => {
  it("builds Hugging Face resolve URLs", () => {
    expect(modelFileUrl("me/models", "dir/a b.onnx")).toBe("https://huggingface.co/me/models/resolve/main/dir/a%20b.onnx");
  });

  it("asks once with the total size, then downloads every file with progress", async () => {
    const asked: unknown[] = [];
    setRemoteConsentHandler((req) => (asked.push(req), true));
    vi.stubGlobal("fetch", async (url: string) => new Response(url.endsWith("a.onnx") ? new Uint8Array([1, 2, 3]) : new Uint8Array([4, 5])));
    const seen: number[] = [];
    const got = await fetchModelFiles({ feature: "scan", label: "oemer", repo: "me/models", files, alsoFrom: ["cdn.jsdelivr.net"], onProgress: (n) => seen.push(n) });
    expect([...got.get("a.onnx")!]).toEqual([1, 2, 3]);
    expect([...got.get("b.json")!]).toEqual([4, 5]);
    expect(asked).toEqual([{ feature: "scan", hosts: ["huggingface.co", "cdn.jsdelivr.net"], sizeMb: 5e-6, label: "oemer" }]);
    expect(seen.at(-1)).toBe(5);
  });

  it("throws ModelFilesDeclined and fetches nothing when the host refuses", async () => {
    setRemoteConsentHandler(() => false);
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    await expect(fetchModelFiles({ feature: "scan", repo: "me/models", files })).rejects.toBeInstanceOf(ModelFilesDeclined);
    expect(fetch).not.toHaveBeenCalled();
  });
});
