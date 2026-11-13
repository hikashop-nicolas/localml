// Model files a feature downloads as they are (ONNX graphs, JSON weights), for libraries that
// run their own models: asked for once through the host's consent handler, fetched from a
// Hugging Face repository with progress, and kept in Cache Storage for the next run.
import { requestRemote, type RemoteFeature } from "./consent";

export interface ModelFilesRequest {
  /** The feature asking, as the host's consent prompt names it. */
  feature: RemoteFeature;
  /** The model's name for the prompt, e.g. "oemer". */
  label?: string;
  /** A Hugging Face repository, "owner/name", and its revision (main by default). */
  repo: string;
  revision?: string;
  /** File names in the repository, with their sizes in bytes when known (for the prompt and progress). */
  files: { name: string; bytes?: number }[];
  /** Called with bytes received and the total, while downloading. */
  onProgress?(loaded: number, total: number): void;
  signal?: AbortSignal;
  /** Cache Storage name; one cache serves every feature. */
  cacheName?: string;
  /** Other hosts the feature downloads from (an engine), named in the same question. */
  alsoFrom?: string[];
}

export const MODEL_FILES_HOST = "huggingface.co";
const CACHE = "localml-model-files";

export class ModelFilesDeclined extends Error {
  constructor() {
    super("The download was declined");
    this.name = "ModelFilesDeclined";
  }
}

/** The URL of a file in a Hugging Face repository. */
export function modelFileUrl(repo: string, name: string, revision = "main"): string {
  return `https://${MODEL_FILES_HOST}/${repo}/resolve/${encodeURIComponent(revision)}/${name.split("/").map(encodeURIComponent).join("/")}`;
}

/**
 * The files' bytes, by name. Cached files are read with no question asked; when some must be
 * downloaded the host is asked first, and a refusal throws ModelFilesDeclined.
 */
export async function fetchModelFiles(req: ModelFilesRequest): Promise<Map<string, Uint8Array>> {
  const out = new Map<string, Uint8Array>();
  const cache = await openCache(req.cacheName ?? CACHE);
  const urls = req.files.map((f) => modelFileUrl(req.repo, f.name, req.revision));
  const missing: number[] = [];
  for (let i = 0; i < urls.length; i++) {
    const hit = await cache?.match(urls[i]).catch(() => undefined);
    if (hit) out.set(req.files[i].name, new Uint8Array(await hit.arrayBuffer()));
    else missing.push(i);
  }
  if (!missing.length) return out;
  const total = missing.reduce((s, i) => s + (req.files[i].bytes ?? 0), 0);
  const ok = await requestRemote({ feature: req.feature, hosts: [MODEL_FILES_HOST, ...(req.alsoFrom ?? [])], sizeMb: total ? total / 1e6 : undefined, label: req.label });
  if (!ok) throw new ModelFilesDeclined();
  let loaded = 0;
  for (const i of missing) {
    const res = await fetch(urls[i], { signal: req.signal });
    if (!res.ok) throw new Error(`${req.files[i].name}: HTTP ${res.status}`);
    const bytes = await readAll(res, (n) => {
      loaded += n;
      req.onProgress?.(loaded, total);
    });
    out.set(req.files[i].name, bytes);
    await cache?.put(urls[i], new Response(bytes as BlobPart)).catch(() => undefined);
  }
  return out;
}

/** Whether every file is already cached (nothing would be downloaded or asked). */
export async function hasModelFiles(req: Pick<ModelFilesRequest, "repo" | "revision" | "files" | "cacheName">): Promise<boolean> {
  const cache = await openCache(req.cacheName ?? CACHE);
  if (!cache) return false;
  for (const f of req.files) if (!(await cache.match(modelFileUrl(req.repo, f.name, req.revision)).catch(() => undefined))) return false;
  return true;
}

async function openCache(name: string): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(name);
  } catch {
    return null;
  }
}

async function readAll(res: Response, progress: (n: number) => void): Promise<Uint8Array> {
  if (!res.body) {
    const b = new Uint8Array(await res.arrayBuffer());
    progress(b.length);
    return b;
  }
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    size += value.length;
    progress(value.length);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
