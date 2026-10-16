import assert from "node:assert/strict";
import test from "node:test";
import { ENABLE_SIZE_PICKER, OUTPUT_SIZES, WatercolorDocument } from "../js/document.js";
import { EditHistory } from "../js/state.js";

test("pixel-valued settings scale from the reference canvas", () => {
  const document = new WatercolorDocument();
  assert.equal(document.px(100), 100);
  document.resize(2835, 1982);
  assert.equal(document.px(100), 2835 / 1782 * 100);
});

test("size options include requested print dimensions while picker remains disabled", () => {
  assert.equal(ENABLE_SIZE_PICKER, false);
  assert.ok(OUTPUT_SIZES.some(({ widthPx, heightPx }) => widthPx === 2835 && heightPx === 1982));
  assert.ok(OUTPUT_SIZES.some(({ widthPx, heightPx }) => widthPx === 7559 && heightPx === 5039));
});

test("normalized vector strokes survive document resizing", () => {
  const document = new WatercolorDocument();
  const stroke = document.addStroke({
    points: [{ x: 0.2, y: 0.7, pressure: 0.4 }],
    size: 50,
    color: "#ffffff",
  });
  document.resize(7559, 5039);
  assert.deepEqual(stroke.points, [{ x: 0.2, y: 0.7, pressure: 0.4 }]);
  assert.equal(stroke.size, 50 / 1782);
});

test("edit history supports undo, redo, and a bounded 50-step history", () => {
  const document = new WatercolorDocument();
  const history = new EditHistory(document);
  for (let index = 0; index < 51; index++) {
    history.addStroke({ points: [{ x: index / 100, y: 0.5 }] });
  }
  assert.equal(history.undoStack.length, 50);
  assert.equal(document.edits.length, 51);
  history.undo();
  assert.equal(document.edits.length, 50);
  history.redo();
  assert.equal(document.edits.length, 51);
});
