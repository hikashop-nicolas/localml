import { afterEach, describe, expect, it, vi } from "vitest";
import { afterConsent, requestRemote, setRemoteConsentHandler, type RemoteRequest } from "./consent";

const req: RemoteRequest = { feature: "translate", hosts: ["huggingface.co"], sizeMb: 500 };
const fakeRun = (value: string) => ({ done: Promise.resolve(value), cancel: vi.fn() });

afterEach(() => setRemoteConsentHandler(null));

describe("remote consent", () => {
  it("lets runs through when no host has installed a handler", async () => {
    expect(await requestRemote(req)).toBe(true);
    const start = vi.fn(() => fakeRun("ran"));
    expect(await afterConsent(req, async () => "declined", start).done).toBe("ran");
  });

  it("passes the request to the host and starts only on yes", async () => {
    const handler = vi.fn(async () => true);
    setRemoteConsentHandler(handler);
    const start = vi.fn(() => fakeRun("ran"));
    expect(await afterConsent(req, async () => "declined", start).done).toBe("ran");
    expect(handler).toHaveBeenCalledWith(req);
  });

  it("ends a refused run like a cancel, without starting it", async () => {
    setRemoteConsentHandler(() => false);
    const start = vi.fn(() => fakeRun("ran"));
    expect(await afterConsent(req, async () => "declined", start).done).toBe("declined");
    expect(start).not.toHaveBeenCalled();
  });

  it("treats a handler that throws as a refusal", async () => {
    setRemoteConsentHandler(() => {
      throw new Error("dialog broke");
    });
    expect(await requestRemote(req)).toBe(false);
  });

  it("does not start a run cancelled while the host was still asking", async () => {
    let answer!: (ok: boolean) => void;
    setRemoteConsentHandler(() => new Promise<boolean>((r) => (answer = r)));
    const start = vi.fn(() => fakeRun("ran"));
    const run = afterConsent(req, async () => "declined", start);
    run.cancel();
    answer(true);
    expect(await run.done).toBe("declined");
    expect(start).not.toHaveBeenCalled();
  });
});
