# Watercolor Studio

Watercolor Studio is an experimental photo-to-watercolor editor. The current
painting generator and most of the editor still use the existing Modal service.
The browser-side document and WebGL files are an initial foundation for moving
that work onto the artist's device; they do **not** yet replace Modal rendering.

## Try the current prototype

1. Start the static site from the repository folder with `python -m http.server 8000`.
2. Open `http://localhost:8000` in a browser.
3. Upload a photo. The prototype sends it to the configured Modal endpoint to
   create the painting. Do not upload private photos unless you are comfortable
   sending them to that service.

The endpoint can be changed from the cloud button in the top bar. The optional
Modal `/cutout` endpoint is currently used by the server, not by a browser AI
model. This prototype does not have an offline generation path yet.

## New browser foundations

- `frontend/js/document.js` defines a reference-canvas document, relative pixel
  scaling, normalized edit strokes, and print-size data. The size picker is
  intentionally disabled until high-resolution rendering and export are ready.
- `frontend/js/state.js` keeps up to 50 vector edit operations for undo and redo.
- `frontend/js/gpu/` provides a WebGL2 texture pool and an instanced stamp
  renderer to build on. The current editor still displays its Canvas2D renderer.

## Not implemented yet

Full on-device u2net/u2netp segmentation, model caching and service worker,
guided-filter edge refinement, a WebGL port of the complete animals pipeline,
GPU-worker edge sampling, replay of vector edits during painting, resolution-
independent tiled JPG export, GitHub Pages deployment, and mobile bottom-sheet
controls remain to be done. The new modules are foundations, not a claim that
the production requirements or the 1–3 second cold-start goal are met.

## Check the document foundation

Run `node --experimental-default-type=module --test frontend/tests/document.test.mjs`
to check print-size scaling, vector stroke persistence, and edit-history limits.
