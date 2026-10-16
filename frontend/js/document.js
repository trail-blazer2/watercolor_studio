export const REFERENCE_CANVAS = Object.freeze({
  width: 1260,
  height: 1782,
});

export const ENABLE_SIZE_PICKER = false;

export const OUTPUT_SIZES = Object.freeze([
  { id: "a4-portrait", label: "A4 portrait", widthPx: 2480, heightPx: 3508, dpi: 300 },
  { id: "a4-landscape", label: "A4 landscape", widthPx: 3508, heightPx: 2480, dpi: 300 },
  { id: "40x30-landscape", label: "40 × 30 cm landscape", widthPx: 2835, heightPx: 1982, dpi: 180 },
  { id: "40x30-portrait", label: "40 × 30 cm portrait", widthPx: 1982, heightPx: 2835, dpi: 180 },
  { id: "large-landscape", label: "Large landscape", widthPx: 7559, heightPx: 5039, dpi: 300 },
  { id: "large-portrait", label: "Large portrait", widthPx: 5039, heightPx: 7559, dpi: 300 },
  { id: "custom", label: "Custom", widthPx: null, heightPx: null, dpi: 300 },
]);

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

export class WatercolorDocument {
  constructor({
    photo = null,
    mask = null,
    preset = "animals",
    outputWidth = REFERENCE_CANVAS.width,
    outputHeight = REFERENCE_CANVAS.height,
    referenceWidth = REFERENCE_CANVAS.width,
    referenceHeight = REFERENCE_CANVAS.height,
    presetParams = {},
    layerParams = {},
    placement = { cx: 0.5, cy: 0.5, scale: 1 },
    edits = [],
  } = {}) {
    this.photo = photo;
    this.mask = mask;
    this.preset = preset;
    this.outputWidth = outputWidth;
    this.outputHeight = outputHeight;
    this.referenceWidth = referenceWidth;
    this.referenceHeight = referenceHeight;
    this.presetParams = { ...presetParams };
    this.layerParams = { ...layerParams };
    this.placement = {
      cx: clamp(placement.cx, 0, 1),
      cy: clamp(placement.cy, 0, 1),
      scale: Math.max(0, placement.scale),
    };
    this.edits = edits.map((edit) => structuredClone(edit));
    this.outputSizeId = "reference";
  }

  get scale() {
    return Math.max(this.outputWidth, this.outputHeight) /
      Math.max(this.referenceWidth, this.referenceHeight);
  }

  px(valueAtReference) {
    return valueAtReference * this.scale;
  }

  resize(width, height, sizeId = "custom") {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      throw new RangeError("Document dimensions must be positive finite numbers.");
    }
    this.outputWidth = Math.round(width);
    this.outputHeight = Math.round(height);
    this.outputSizeId = sizeId;
    return this;
  }

  addStroke({ points, space = "canvas", tool = "paint", tip = "round_hard", size = 1,
    color = "#000000", opacity = 1, pressure = 1, layerId = null }) {
    if (!Array.isArray(points) || points.length === 0) {
      throw new TypeError("A stroke needs at least one point.");
    }
    const normalizedPoints = points.map(({ x, y, pressure: pointPressure }) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new TypeError("Stroke points must contain finite x and y coordinates.");
      }
      const point = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
      if (Number.isFinite(pointPressure)) point.pressure = clamp(pointPressure, 0, 1);
      return point;
    });
    const record = {
      type: "stroke",
      space,
      tool,
      tip,
      points: normalizedPoints,
      size: Math.max(0, size) / Math.max(this.outputWidth, this.outputHeight),
      color,
      opacity: clamp(opacity, 0, 1),
      pressure: clamp(pressure, 0, 1),
      layerId,
    };
    this.edits.push(record);
    return record;
  }
}
