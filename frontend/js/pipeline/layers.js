import { PythonRandom } from "../pyrandom.js";
import { ellipsePoints, poissonSample } from "./sampling.js";
import { brushColor, mix } from "./color.js";

const definitions = [
  ["bg_paper", "Paper & canvas", "Canvas Base", "paper", "PAPER_TEXTURE", 0],
  ["bg_brush", "Background wash", "Under Subject", "edge", "BG_BRUSH", 200],
  ["sharp_brush", "Sharp edge brush", "Under Subject", "edge", "SHARP_BRUSH", 150],
  ["mask_blur", "Layered blur & square mask", "Under Subject", "blur", "MASK_BRUSH", 0],
  ["color_brush", "Color wash", "Under Subject", "edge", "COLOR_BRUSH", 100],
  ["accent_strokes", "Accent strokes — pass 1", "Under Subject", "accent", "ACCENT_STROKES", 1000],
  ["white_dots", "White splatter", "Under Subject", "dots", "WHITE_DOTS", 300],
  ["dark_dots", "Dark splatter", "Under Subject", "dots", "DARK_DOTS", 400],
  ["glow", "Subject rim glow", "Under Subject", "glow", "GLOW", 0],
  ["accent_strokes_2", "Accent strokes — pass 2", "Under Subject", "accent", "ACCENT_STROKES", 1000],
  ["colordrop", "Bottom color drops", "Under Subject", "bottom", "COLORDROP_BRUSH", 850],
  ["shadow", "Bottom shadow", "Under Subject", "bottom", "SHADOW_BRUSH", 800],
  ["subject", "Subject & color correction", "Subject & Oil", "subject", "CC", 0],
  ["oil_paint", "Oil detail", "Subject & Oil", "oil", "OIL_PAINT", 0],
  ["edge_curve", "Edge tri-tone", "Over Subject", "curve", "EDGE_CURVE", 0],
  ["dark_ellipse", "Bottom dark ellipse", "Over Subject", "ellipse", "DARK_ELLIPSE", 600],
  ["white_ellipse", "Inner frame brush", "Over Subject", "ellipse", "WHITE_BRUSH", 500],
  ["outer_white", "Outer frame brush", "Over Subject", "ellipse", "OUTER_WHITE_BRUSH", 550],
  ["small_dots", "Small white dots", "Over Subject", "dots", "SMALL_WHITE_DOTS", 700],
  ["custom_detail", "Painted details", "Over Subject", "custom", "", 0],
];

export function createLayers(cfg) {
  return definitions.map(([id, name, group, kind, prefix, seedOffset]) => ({
    id, name: name + (kind === "custom" ? "" : " (approx)"), group,
    approx: kind !== "custom", kind, prefix, seedOffset,
    visible: id === "subject" ? cfg.SHOW_SUBJECT : (cfg[`${prefix}_ENABLED`] ?? true),
    opacity: kind === "paper" ? 1 : cfg[`${prefix}_OPACITY`] ?? 1,
    params: {
      brightness: 1, contrast: 1, saturation: 1,
      spacing: cfg[`${prefix}_SPACING`] ?? 100, size: cfg[`${prefix}_SIZE`] ?? 300,
      jiggle: cfg[`${prefix}_JIGGLE`] ?? cfg.BRUSH_DIRECTION_JIGGLE,
      brush: cfg[`${prefix}_FILE`] ?? (
        id.includes("white") ? cfg.ELLIPSE_BRUSH_FILE : "bgbrush.png"
      ),
    },
  }));
}

export function generateStamps(layer, params, model, document, cfg) {
  const { edges, average, dominant } = model;
  const max = Math.max(document.outputWidth, document.outputHeight);
  const px = (n) => document.px(n);
  const length = (n) => px(n) / max;
  const rng = new PythonRandom(cfg.RANDOM_SEED + layer.seedOffset);
  let points;
  if (layer.kind === "ellipse") {
    points = ellipsePoints(document.outputWidth, document.outputHeight, Math.max(1, px(params.spacing)), {
      inset: px(cfg[`${layer.prefix}_ELLIPSE_INSET`] ?? 0),
      bottomOnly: layer.id === "dark_ellipse", bottomPercent: cfg.DARK_ELLIPSE_BOTTOM_PERCENT,
    }).map((p) => ({ ...p, x: p.x / document.outputWidth, y: p.y / document.outputHeight }));
    const blend = cfg[`${layer.prefix}_OUTLINE_BLEND`] ?? 0;
    if (blend > 0 && edges.length) points = points.map((point) => {
      let best = edges[0], distance = Infinity;
      for (const edge of edges) {
        const d = ((edge.x - point.x) * document.outputWidth) ** 2 +
          ((edge.y - point.y) * document.outputHeight) ** 2;
        if (d < distance) { best = edge; distance = d; }
      }
      return { ...point, x: point.x + (best.x - point.x) * blend, y: point.y + (best.y - point.y) * blend };
    });
  } else {
    points = poissonSample(edges.map((p) => ({
      ...p, x: p.x * document.outputWidth, y: p.y * document.outputHeight,
    })), Math.max(1, px(params.spacing))).map((p) => ({
      ...p, x: p.x / document.outputWidth, y: p.y / document.outputHeight,
    }));
  }
  if (layer.kind === "bottom") {
    const bounds = model.subjectBounds;
    points = points.filter((p) => p.y > bounds.y + bounds.height * 0.75);
  }
  if (layer.kind === "accent") {
    // Contrast-ranked edge color patches stand in for Python's adaptive interest/weight maps.
    points = points.filter((p) => {
      const c = model.colorAt(p), l = (c[0] + c[1] + c[2]) / 3;
      return l < 0.4 || l > 0.75 || Math.max(...c) - Math.min(...c) > 0.18;
    }).slice(0, 180);
  }
  const stamps = [];
  for (const point of points) {
    let color = model.colorAt(point), colorize = true;
    let rotation, size, opacity = 1;
    if (layer.kind === "ellipse") {
      const jiggle = params.jiggle;
      rotation = Math.atan2(-(0.5 - point.y) * document.outputHeight,
        (0.5 - point.x) * document.outputWidth) +
        (90 + (cfg[`${layer.prefix}_ROTATION_OFFSET`] ?? 180)) * Math.PI / 180;
      if (jiggle > 0) rotation += rng.uniform(-Math.PI * jiggle, Math.PI * jiggle);
      size = length(params.size) * rng.uniform(0.7, 1.3);
      color = cfg.ELLIPSE_BRUSH_MODE === "dark" ? mix(average, [0, 0, 0], 0.7) : [1, 1, 1];
      if (layer.id === "dark_ellipse") color = mix(average, [0, 0, 0], 0.65);
      const falloff = cfg[`${layer.prefix}_OPACITY_FALLOFF`] ?? 0;
      opacity *= 1 - falloff * (1 - point.y);
    } else if (layer.kind === "dots" || layer.id === "sharp_brush") {
      rotation = rng.uniform(0, 2 * Math.PI);
      size = length(params.size) * rng.uniform(0.7, 1.3);
      if (layer.kind === "dots") {
        colorize = layer.id === "dark_dots" && cfg.DARK_DOTS_USE_DOMINANT_COLOR;
        color = colorize ? dominant.map((v) => v * (1 - cfg.DARK_DOTS_COLOR_DARKEN)) : [1, 1, 1];
      } else color = brushColor(color, cfg);
    } else {
      rotation = Math.atan2(-point.ny, point.nx) + Math.PI * 1.5;
      if (params.jiggle > 0) rotation += rng.uniform(-Math.PI * params.jiggle, Math.PI * params.jiggle);
      size = length(params.size) * rng.uniform(0.7, 1.3);
      color = brushColor(color, cfg, layer.id === "bg_brush", average, dominant);
      if (layer.kind === "accent") {
        size *= 0.8; opacity = 0.6;
        if (layer.id === "accent_strokes_2") color = mix(color, [1, 1, 1], 0.12);
      }
      if (layer.kind === "bottom") {
        color = layer.id === "shadow" ? [0, 0, 0] : color;
      }
    }
    let { x, y } = point;
    if (layer.id === "small_dots") {
      if (cfg.SMALL_WHITE_DOTS_AVOID_BOTTOM && y > 1 - px(cfg.SMALL_WHITE_DOTS_BOTTOM_MARGIN) / document.outputHeight) continue;
      const offset = px(cfg.SMALL_WHITE_DOTS_DISTANCE_JITTER);
      x += (point.nx ?? 0) * offset / document.outputWidth;
      y += (point.ny ?? 0) * offset / document.outputHeight;
    }
    stamps.push({ x, y, size: Math.max(length(20), size), rotation, opacity, color, colorize });
  }
  return stamps;
}
