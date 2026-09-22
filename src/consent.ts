// Consent before anything is fetched from outside the device. Every feature here runs on the
// user's machine, but its model or engine is downloaded on first use: a host app can ask the
// user once (and remember the answer) by installing a handler. With no handler, runs proceed
// as before, so the demo pages and hosts that do not care keep working unchanged.
//
// The handler lives on globalThis rather than in a module variable: a host and the libraries it
// embeds can end up with two copies of this module, and they must still share one answer.

export type RemoteFeature = "ocr" | "translate" | "generate" | "transcribe";

export interface RemoteRequest {
  feature: RemoteFeature;
  /** Hosts the download comes from, for the host to name in its prompt. */
  hosts: string[];
  /** Approximate size of the first download in MB, when known. */
  sizeMb?: number;
  /** The model or engine, e.g. "M2M-100". */
  label?: string;
}

export type RemoteConsentHandler = (req: RemoteRequest) => boolean | Promise<boolean>;

const KEY = "__localmlRemoteConsent";
type Slot = { [KEY]?: RemoteConsentHandler };

export function setRemoteConsentHandler(handler: RemoteConsentHandler | null): void {
  (globalThis as Slot)[KEY] = handler ?? undefined;
}

/** Ask the host whether this download may happen. True when no host has an opinion. A handler
 *  that throws counts as a refusal: when in doubt, nothing leaves the device. */
export async function requestRemote(req: RemoteRequest): Promise<boolean> {
  const handler = (globalThis as Slot)[KEY];
  if (!handler) return true;
  try {
    return (await handler(req)) === true;
  } catch {
    return false;
  }
}

/**
 * Start a run only once the host agrees. A refusal ends the run the way a cancel does (the
 * `declined` outcome), so the hosts' existing cancel handling covers it and needs no new code.
 */
export function afterConsent<R extends { done: Promise<T>; cancel(): void }, T>(
  req: RemoteRequest,
  declined: () => Promise<T>,
  start: () => R,
): { done: Promise<T>; cancel(): void; current(): R | null } {
  let run: R | null = null;
  let cancelled = false;
  const done = requestRemote(req).then((ok) => {
    if (!ok || cancelled) return declined();
    run = start();
    return run.done;
  });
  return {
    done,
    cancel: () => {
      cancelled = true;
      run?.cancel();
    },
    current: () => run,
  };
}

/** Where the model-based features fetch from on first use: the model from the Hugging Face hub,
 *  and the onnxruntime engine files transformers.js loads from jsDelivr (measured, not assumed). */
export const MODEL_HOSTS = ["huggingface.co", "cdn.jsdelivr.net"];
