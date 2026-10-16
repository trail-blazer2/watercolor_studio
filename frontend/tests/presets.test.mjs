import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { PRESETS, getPresetCopy } from "../js/pipeline/presets.js";
import { createLayers } from "../js/pipeline/layers.js";

test("all six presets match every Python AST literal value", () => {
  const source = fileURLToPath(new URL("../../backend/presets.py", import.meta.url));
  const script = `import ast,json,sys
m=ast.parse(open(sys.argv[1],encoding="utf-8").read())
print(json.dumps(next(ast.literal_eval(n.value) for n in m.body if isinstance(n,ast.AnnAssign) and getattr(n.target,"id","")=="PRESETS")))`;
  const expected = JSON.parse(execFileSync("python", ["-c", script, source], { encoding: "utf8" }));
  assert.deepEqual(PRESETS, expected);
  assert.equal(PRESETS.animals.MASK_BRUSH_SIZE, 900);
  assert.equal(PRESETS.animals.CC_WARM_BOOST, 2.2);
  assert.equal(PRESETS.cars.OUTPUT_WIDTH, 1782);
});

test("preset copies are independent and unknown names default to animals", () => {
  const copy = getPresetCopy("missing");
  assert.equal(copy._preset_name, "animals");
  copy.DARK_BACKGROUND_COLOR[0] = 0;
  copy.ACCENT_STROKES_SETTINGS.changed = true;
  assert.equal(PRESETS.animals.DARK_BACKGROUND_COLOR[0], 255);
  assert.deepEqual(getPresetCopy().ACCENT_STROKES_SETTINGS, {});
});

test("painting stack matches Python order, seeds, controls and approximation labels", () => {
  const layers = createLayers(getPresetCopy());
  assert.deepEqual(layers.map((l) => l.id), [
    "bg_paper", "bg_brush", "sharp_brush", "mask_blur", "color_brush",
    "accent_strokes", "white_dots", "dark_dots", "glow", "accent_strokes_2",
    "colordrop", "shadow", "subject", "oil_paint", "edge_curve",
    "dark_ellipse", "white_ellipse", "outer_white", "small_dots", "custom_detail",
  ]);
  assert.deepEqual(layers.filter((l) => ["bg_brush", "sharp_brush", "mask_blur", "color_brush",
    "white_dots", "dark_dots", "dark_ellipse", "white_ellipse", "outer_white", "small_dots",
    "shadow", "colordrop"].includes(l.id)).map((l) => l.seedOffset),
  [200, 150, 0, 100, 300, 400, 850, 800, 600, 500, 550, 700]);
  for (const layer of layers) {
    if (layer.approx) assert.match(layer.name, /\(approx\)/);
    for (const key of ["brightness", "contrast", "saturation", "spacing", "size", "jiggle", "brush"]) {
      assert.ok(Object.hasOwn(layer.params, key), `${layer.id}: ${key}`);
    }
  }
});
