// tsc transpiles each file 1:1 and leaves `new URL("./x.worker.ts", import.meta.url)`
// pointing at the .ts source. In the published dist only the compiled .worker.js exists,
// so a consuming bundler (Vite) can't resolve the .ts entry. Rewrite the worker URLs to
// .js across the built dist. Source keeps .ts so localml's own Vite dev/build still works.
import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";

const root = new URL("../dist/", import.meta.url).pathname;
const rx = /(new URL\(\s*["']\.\/[\w.-]+\.worker)\.ts(["'])/g;

function walk(dir) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      walk(full);
    } else if (name.endsWith(".js")) {
      const src = readFileSync(full, "utf8");
      const out = src.replace(rx, "$1.js$2");
      if (out !== src) {
        writeFileSync(full, out);
        console.log(`fixed worker URLs in ${full.slice(root.length)}`);
      }
    }
  }
}

walk(root);
