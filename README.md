# Watercolor Studio

An experimental static, on-device photo-to-watercolor editor. The app in
`frontend/` is the GitHub Pages entry point. Python reference/backend code lives
in `backend/` and is not deployed or required by the browser app. Photos and
painting edits stay in your browser. Initial model and ONNX Runtime downloads
require a network connection; they do not contain your photo.

**Site:** https://trail-blazer2.github.io/watercolor_studio/

## Run locally

Use a current browser with WebGL2, WebAssembly, and workers. Chrome/Edge are the
primary development targets. HTTPS (or localhost) is needed for service workers.

```sh
mkdir -p frontend/models
curl --fail --location --retry 3 \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
  --output frontend/models/u2net.onnx
curl --fail --location --retry 3 \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx \
  --output frontend/models/u2netp.onnx
python -m http.server 8000 --directory frontend
```

Open **http://localhost:8000/**. The official
rembg release weights are approximately 176 MB (`u2net`) and 4.6 MB (`u2netp`);
they are ignored by Git and downloaded into the Pages artifact during deployment.
Respect the upstream model and runtime licenses.

## Enable GitHub Pages (repository owner)

1. In `trail-blazer2/watercolor_studio`, push these changes to `main` (or change the workflow's branch filter if your
   default branch has a different name).
2. Open **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. In **Actions**, enable workflows if prompted. Run **Deploy on-device studio to
   Pages** manually once, or push a change under `frontend/`.
4. Wait for its build and deploy jobs to finish, then open
   **https://trail-blazer2.github.io/watercolor_studio/**.

The workflow uploads **only `frontend/`**, uses SHA-pinned Pages actions, and
grants the deploy job `pages: write` and `id-token: write`. No Modal token or
other application secret is required. Repository/environment policy can still
require an owner to approve deployment.

## Offline use and privacy

The service worker and manifest use relative paths, including on the Pages
`/watercolor_studio/` subpath. Open the app online first, finish a successful
segmentation, wait for the full-model upgrade to finish, then reopen it while
offline. Model/runtime files are cached only after successful fetches. Available storage, browser
eviction, and private-browsing restrictions can prevent offline model reuse;
caching failures must not prevent an otherwise successful online inference.
The small model can retain its initial mask if the full model is unavailable.
Export once while online as well if you want offline JPG export: this fetches and
caches the separate HD brush assets.

An install option depends on browser/platform support (the manifest supplies
192/512-pixel PNG icons and an SVG icon). Clearing site data removes downloaded
caches and local settings.
Images chosen in the editor and developer pages are processed locally. Network
requests fetch static assets, ONNX Runtime, and models, not selected photos.
The current UI also requests Google Fonts. Model downloads begin after choosing
a photo, not merely opening the editor. Use developer tools/network logs to
verify that no photo upload occurs.

## Validation and developer pages

```sh
node --experimental-default-type=module --test frontend/tests/*.test.mjs
```

Open **http://localhost:8000/dev/regression.html** and click **Run browser
regressions**. Expect eight PASS lines. These use a synthetic photo/mask and the
real GPU compositor and JPG encoder, without model downloads. They cover bounded
photo decoding, stamp orientation/alpha, deterministic rendering, layer-parameter
history, paint/eraser history and resizing, mask history, tile consistency, and
JPEG decoding. Add `?run` to run automatically.

Document/history is connected to the mobile editor and GPU rendering. Mask
rebuilds replay edits from the original mask and mark the result as already
edited, so preview and export do not apply those edits a second time.

### Owner smoke test after deployment

1. Open **https://trail-blazer2.github.io/watercolor_studio/dev/regression.html**,
   run the checks, and confirm eight PASS lines.
2. Open the deployed app online and upload a small photo. Verify initial
   segmentation/painting completes, then allow the full-model upgrade to finish.
3. Select a layer and change its opacity. Confirm the visible result updates.
4. Try paint, detail, eraser, and mask-add/mask-erase strokes. Undo and redo each
   operation; check that both the stroke and affected controls return correctly.
5. Pan and zoom, then export a JPG. Open the downloaded file and compare its
   placement, layers, and edits with the preview. Test share only on a browser
   that offers it.
6. Reload once to ensure service-worker control, enable offline mode, and reload
   again. Upload a photo and confirm cached inference works. If storage caching
   was denied, restore the network; online inference should still work.
7. Check the network panel: photo bytes must not be posted to any endpoint.
   Repeat the editing/export checks on a target phone; desktop success does not
   establish mobile performance or memory limits.

- **`/dev/benchmark.html`** (relative to the app root): real 5,000/20,000-stamp
  WebGL2 draws, plus photo decoding, segmentation, painting, and available
  per-stage pipeline timings. Values are measured at run time, not canned.
  Stamp timing includes CPU submission and GPU completion; it is not a hardware
  GPU timer-query benchmark.
- **`/dev/compare.html`**: choose an original photo and a separately produced
  Python animals-preset reference, then generate a local browser export for
  side-by-side inspection. Match input, seed, preset, dimensions, and placement
  when assessing differences.
  Both developer pages use the editor's 2048-pixel photo decode limit.

### Results from this continuation (2026-10-16)

- All **40 Node tests** and **eight browser regressions** passed. The eight
  browser checks also passed offline after the service worker cached the app.
- Headless Chromium 155 on Linux, using **SwiftShader software WebGL2**, drew
  5,000 stamps in **4.2 ms** and 20,000 in **4.4 ms** (submission plus finish;
  generation took 6.3/14.9 ms, setup 14.2 ms). These are sandbox measurements,
  **not phone/GPU performance claims**.
- Mobile-sized (390 × 844) editor startup, cached offline shell, and recovery
  after an inference-load error passed. Comparison input validation and reference
  preview passed.
- Official model downloads succeeded, but **`cdn.jsdelivr.net` does not resolve
  in this sandbox**. The pipeline benchmark, comparison generation, and editor
  upload correctly reported the unavailable ONNX Runtime and recovered.
  **Real browser inference, offline inference, full comparison generation,
  Python-reference fidelity, and target-phone timings remain unverified.**

## Scope and known limitations

This is an **animals-preset port, not a pixel-identical Python renderer**.
Watercolor wash/diffusion, interior texture, edge bleed, stamp distribution, and
paper/compositing are approximations; stochastic marks may differ. In particular:

- **Oil detail** uses Canvas2D contrast/saturation adjustments, not Python's oil
  filtering and sharpening algorithms. Faithful structure-tensor/flow filtering
  and high-pass blending are deferred; unsupported flow controls are not shown.
- **Accent strokes** use contrast/color-thresholded edge patches, capped at 180
  candidates, rather than Python's adaptive interest/weight maps, image-statistic
  thresholds, and spatial weight balancing. A faithful port is deferred.
- Browser image resampling differs from Pillow Lanczos. Guided mask refinement
  can be skipped when float render targets or the GPU memory budget are
  unavailable; the mask without guided refinement is retained and the limitation
  is reported.
- Layers labeled **(approx)** are intentionally not fidelity claims. Ported
  preset data does not establish equivalent rendering for every Python preset.

The comparison page does not calculate or certify perceptual fidelity.

Cold starts include model transfer, runtime initialization, and inference.
**Target-device cold/warm timing and fidelity review are pending**; no measured
three-second guarantee is claimed. Large outputs can exceed browser memory,
canvas, or GPU limits even when rendering is tiled. The bounded-stripe JPEG
encoder writes quality-95, 4:4:4 output with 300-DPI JFIF metadata; compressed
output bytes still occupy memory, and print software may ignore that metadata.
WebGPU availability varies; the WASM fallback can be slower and does not depend
on cross-origin isolation or multithreaded shared memory on GitHub Pages.
