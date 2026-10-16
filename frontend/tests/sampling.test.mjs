import assert from "node:assert/strict";
import test from "node:test";
import { poissonSample, maskBounds, sampleMaskEdges, ellipsePoints } from "../js/pipeline/sampling.js";
import { generateStamps, createLayers } from "../js/pipeline/layers.js";
import { getPresetCopy } from "../js/pipeline/presets.js";
import { WatercolorDocument } from "../js/document.js";
import { PaintingCompositor } from "../js/pipeline/compositor.js";
import { createPainting } from "../js/pipeline/paint.js";

test("Poisson edge sampling is deterministic and respects minimum separation", () => {
  const points = Array.from({ length: 1600 }, (_, i) => ({ x: i % 40 - 20, y: Math.floor(i / 40) - 20 }));
  const sampled = poissonSample(points, 5);
  assert.deepEqual(poissonSample(points, 5), sampled);
  assert.ok(sampled.length > 30);
  for (let i = 0; i < sampled.length; i++) for (let j = i + 1; j < sampled.length; j++) {
    assert.ok(Math.hypot(sampled[i].x - sampled[j].x, sampled[i].y - sampled[j].y) >= 5);
  }
});

test("Poisson retains row-major first candidates and accepts exact-spacing neighbors", () => {
  assert.deepEqual(poissonSample([{ x: 0, y: 0 }, { x: 3, y: 0 }, { x: 4, y: 0 }, { x: 8, y: 0 }], 4),
    [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 8, y: 0 }]);
  assert.deepEqual(poissonSample([], 4), []);
  assert.throws(() => poissonSample([], 0), RangeError);
  assert.throws(() => poissonSample([], NaN), RangeError);
});

test("mask edges, bounds and outward normals handle empty, inset and border masks", () => {
  const alpha = new Uint8Array(49);
  assert.equal(maskBounds(alpha, 7, 7), null);
  assert.deepEqual(sampleMaskEdges(alpha, 7, 7), []);
  for (let y = 1; y < 6; y++) for (let x = 1; x < 6; x++) alpha[y * 7 + x] = 255;
  assert.deepEqual(maskBounds(alpha, 7, 7), { x: 1, y: 1, width: 5, height: 5 });
  const edges = sampleMaskEdges(alpha, 7, 7, 2);
  assert.equal(edges.length, 16);
  const left = edges.find((p) => p.x === 1 && p.y === 3);
  assert.ok(left.nx < -0.99 && Math.abs(left.ny) < 1e-8);
  alpha.fill(255);
  assert.equal(sampleMaskEdges(alpha, 7, 7).length, 24);
});

test("ellipse sampling honors bottom-only and inset geometry", () => {
  const points = ellipsePoints(1260, 1782, 200, { bottomOnly: true, bottomPercent: 0.15 });
  assert.ok(points.length > 0);
  assert.ok(points.every((p) => p.y >= 1782 * 0.85));
  assert.ok(ellipsePoints(1260, 1782, 200, { inset: -200 }).some((p) => p.x > 1260));
});

test("normalized stamp layout and reference lengths survive document output scaling", () => {
  const document = new WatercolorDocument(), cfg = getPresetCopy();
  const layer = createLayers(cfg).find((l) => l.id === "color_brush");
  const edges = Array.from({ length: 40 }, (_, i) => ({
    x: 0.2 + i / 100, y: 0.4, nx: 0, ny: -1,
  }));
  const model = { edges, average: [0.5, 0.4, 0.2], dominant: [0.5, 0.4, 0.2], colorAt: () => [0.5, 0.4, 0.2] };
  const first = generateStamps(layer, layer.params, model, document, cfg);
  assert.deepEqual(first, generateStamps(layer, layer.params, model, document, cfg));
  document.resize(2520, 3564);
  assert.deepEqual(first, generateStamps(layer, layer.params, model, document, cfg));
});

test("painting rejects missing decoded photos before allocating graphics resources", async () => {
  await assert.rejects(createPainting({}), /decoded document.photo/);
});

test("compositor memory budget includes resident brush textures and working surfaces", () => {
  const canvas = { width: 1, height: 1 };
  const compositor = { canvas, assetBytes: 100 * 1024 * 1024,
    gl: { MAX_TEXTURE_SIZE: 1, getParameter: () => 8192, viewport: () => {} } };
  PaintingCompositor.prototype.resize.call(compositor, 1024, 1024);
  assert.equal(canvas.width, 1024);
  assert.throws(() => PaintingCompositor.prototype.resize.call(compositor, 8192, 8192), /memory budget/);
  assert.throws(() => PaintingCompositor.prototype.resize.call(compositor, 9000, 1), /memory budget/);
  compositor.assetBytes = 250 * 1024 * 1024;
  assert.throws(() => PaintingCompositor.prototype.resize.call(compositor, 1, 1), /memory budget/);
});
