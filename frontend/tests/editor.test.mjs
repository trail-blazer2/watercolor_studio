import test from "node:test";
import assert from "node:assert/strict";
import { maskDiameter, subjectPoint, replayParameters } from "../js/editor.js";
import { WatercolorDocument } from "../js/document.js";
import { EditHistory } from "../js/state.js";

test("mask pointer mapping uses subject bounds, not canvas dimensions", () => {
  assert.deepEqual(subjectPoint({ x: 0.4, y: 0.6, pressure: 0.5 },
    { x: 0.2, y: 0.2, width: 0.4, height: 0.8 }),
  { x: 0.5, y: 0.49999999999999994, pressure: 0.5 });
});

test("parameter edits replay through undo and redo without changing strokes", () => {
  const document = new WatercolorDocument();
  const history = new EditHistory(document);
  history.addStroke({ type: "parameter", layerId: "bg_brush", key: "opacity", value: 0.6 });
  history.addStroke({ type: "stroke", layerId: "bg_brush", points: [{ x: 0.5, y: 0.3 }] });
  history.addStroke({ type: "parameter", layerId: "bg_brush", key: "opacity", value: 0.2 });
  replayParameters(document);
  assert.equal(document.layerParams.bg_brush.opacity, 0.2);
  history.undo();
  replayParameters(document);
  assert.equal(document.layerParams.bg_brush.opacity, 0.6);
  assert.equal(document.edits[1].type, "stroke");
  history.redo();
  replayParameters(document);
  assert.equal(document.layerParams.bg_brush.opacity, 0.2);
});

test("mask brush footprint stays in photo space when placement changes or source resolution doubles", () => {
  const bounds = { width: 0.5, height: 0.5 };
  const record = { size: 0.02 };
  record.sourceSize = maskDiameter(record, 1000, 800, bounds) / 1000;
  assert.equal(maskDiameter(record, 1000, 800, { width: 0.25, height: 0.25 }), 40);
  assert.equal(maskDiameter(record, 2000, 1600, bounds), 80);
});
