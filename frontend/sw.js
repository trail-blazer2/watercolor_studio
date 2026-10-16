const VERSION = "watercolor-v2";
const SHELL_CACHE = `${VERSION}-shell`;
const ASSET_CACHE = `${VERSION}-assets`;
const scope = new URL("./", self.location.href);
const local = (path) => new URL(path, scope).href;
const SHELL = [
  "./",
  "index.html",
  "studio.css",
  "manifest.json",
  "assets/icon.svg",
  "assets/icon-192.png",
  "assets/icon-512.png",
  "js/main.js",
  "js/app.js",
  "js/document.js",
  "js/editor.js",
  "js/pyrandom.js",
  "js/state.js",
  "js/pipeline/paint.js",
  "js/pipeline/color.js",
  "js/pipeline/compositor.js",
  "js/pipeline/layers.js",
  "js/pipeline/presets.js",
  "js/pipeline/sampling.js",
  "js/pipeline/segment.js",
  "js/pipeline/export.js",
  "js/gpu/context.js",
  "js/gpu/stamp.js",
  "js/gpu/shaders/stamp.js",
  "dev/benchmark.html",
  "dev/compare.html",
  "dev/regression.html",
  ...[
    "bgbrush", "brush2", "colordrop", "darkbrush", "darkdots", "largebrush",
    "papertexture", "sharpbrush", "square", "watercolor_texture",
    "watercolortexture", "whitebrush", "whitedots",
  ].map((name) => `assets/preview/${name}.png`),
].map(local);

// Keep this exact version in sync with the segmentation runtime loader.
const ORT_PREFIX = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await cache.addAll(SHELL);
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith("watercolor-") && key !== SHELL_CACHE && key !== ASSET_CACHE) {
        await caches.delete(key);
      }
    }
    await self.clients.claim();
  })());
});

function cacheable(request, response) {
  return response.ok && response.type !== "opaque" && response.status !== 206 &&
    !request.headers.has("range") && !/\bno-store\b/i.test(response.headers.get("cache-control") || "");
}

async function store(request, response) {
  try {
    const cache = await caches.open(ASSET_CACHE);
    await cache.put(request, response);
  } catch {
    // Model downloads can exceed storage quota; never fail the inference fetch.
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET" || request.headers.has("range")) return;
  const url = new URL(request.url);
  const sameApp = url.origin === scope.origin && url.pathname.startsWith(scope.pathname);
  const ortAsset = request.url.startsWith(ORT_PREFIX) &&
    /\.(?:m?js|wasm)$/.test(url.pathname);
  if (!sameApp && !ortAsset) return;

  event.respondWith((async () => {
    const cached = await caches.match(request).catch(() => undefined);
    if (cached && request.mode !== "navigate") return cached;
    try {
      const response = await fetch(request);
      if (cacheable(request, response)) event.waitUntil(store(request, response.clone()));
      return response;
    } catch (error) {
      if (cached) return cached;
      if (request.mode === "navigate" && sameApp) {
        const shell = await caches.match(local("index.html"));
        if (shell) return shell;
      }
      throw error;
    }
  })());
});
