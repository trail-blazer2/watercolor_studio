import { getPresetCopy } from "./presets.js";
import { createLayers, generateStamps } from "./layers.js";
import { clamp, correctPixels } from "./color.js";
import { maskBounds, sampleMaskEdges } from "./sampling.js";
import { makeCanvas, PaintingCompositor } from "./compositor.js";

const assetNames = [
  "bgbrush.png", "brush2.png", "square.png", "sharpbrush.png", "whitedots.png",
  "darkdots.png", "whitebrush.png", "darkbrush.png", "colordrop.png",
  "papertexture.png", "watercolortexture.png",
];
const context = (canvas) => {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas2D is unavailable.");
  return ctx;
};
const round = (n) => Math.max(1, Math.round(n));

async function loadAssets(quality) {
  const folder = quality === "full" || quality === "hd" ? "hd" : "preview";
  const loaded = new Map();
  const results = await Promise.allSettled(assetNames.map(async (name) => {
    const response = await fetch(new URL(`../../assets/${folder}/${name}`, import.meta.url));
    if (!response.ok) throw new Error(`Unable to load brush ${name}: ${response.status}`);
    let image = await createImageBitmap(await response.blob());
    const limit = folder === "hd" ? 2048 : 768;
    if (Math.max(image.width, image.height) > limit) {
      const factor = limit / Math.max(image.width, image.height);
      const resized = await createImageBitmap(image, {
        resizeWidth: round(image.width * factor), resizeHeight: round(image.height * factor), resizeQuality: "high",
      });
      image.close(); image = resized;
    }
    loaded.set(name, image);
  }));
  const failure = results.find((result) => result.status === "rejected");
  if (failure) {
    for (const image of loaded.values()) image.close();
    throw failure.reason;
  }
  return loaded;
}

function placement(document, bounds, width, height) {
  const p = document.placement ?? { cx: 0.5, cy: 0.5, scale: 1 };
  const fit = 0.82 * Math.min(document.outputWidth / bounds.width, document.outputHeight / bounds.height) * p.scale;
  const subjectBounds = {
    x: p.cx - bounds.width * fit / document.outputWidth / 2,
    y: p.cy - bounds.height * fit / document.outputHeight / 2,
    width: bounds.width * fit / document.outputWidth, height: bounds.height * fit / document.outputHeight,
  };
  return {
    subjectBounds,
    imageBounds: {
      x: subjectBounds.x - bounds.x * fit / document.outputWidth,
      y: subjectBounds.y - bounds.y * fit / document.outputHeight,
      width: width * fit / document.outputWidth, height: height * fit / document.outputHeight,
    },
  };
}

function strokePoints(stroke, model) {
  return stroke.points.map((p) => {
    if (stroke.space === "subject") {
      const b = stroke.tool?.startsWith("mask-") ? model.imageBounds : model.subjectBounds;
      return { ...p, x: b.x + p.x * b.width, y: b.y + p.y * b.height };
    }
    if (stroke.space === "photo") return {
      ...p, x: model.imageBounds.x + p.x * model.imageBounds.width,
      y: model.imageBounds.y + p.y * model.imageBounds.height,
    };
    return p;
  });
}

function drawStroke(ctx, stroke, model, view, { mask = false } = {}) {
  if (!stroke.points?.length || !(stroke.size > 0)) return;
  const points = strokePoints(stroke, model);
  const max = Math.max(view.fullWidth, view.fullHeight);
  const radius = stroke.size * max / 2;
  const erase = stroke.tool === "erase" || stroke.tool === "mask-erase";
  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.globalAlpha = clamp(stroke.opacity ?? 1);
  ctx.fillStyle = ctx.strokeStyle = mask ? "#ffffff" : stroke.color;
  ctx.lineCap = ctx.lineJoin = "round";
  const soft = /soft|watercolor/.test(stroke.tip);
  if (soft) {
    ctx.shadowColor = mask ? "#ffffff" : stroke.color;
    ctx.shadowBlur = radius * 0.5;
  }
  const pixel = (p) => ({ x: p.x * view.fullWidth - view.x, y: p.y * view.fullHeight - view.y });
  for (let i = 0; i < points.length; i++) {
    const p = pixel(points[i]), previous = pixel(points[Math.max(0, i - 1)]);
    const pressure = clamp(points[i].pressure ?? stroke.pressure ?? 1);
    const r = Math.max(0.1, radius * pressure);
    ctx.lineWidth = r * 2;
    ctx.beginPath();
    if (i === 0) { ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); }
    else { ctx.moveTo(previous.x, previous.y); ctx.lineTo(p.x, p.y); ctx.stroke(); }
  }
  ctx.restore();
}

function createModel(document, cfg, sourceLimit) {
  const photo = document.photo;
  const factor = Math.min(1, sourceLimit / Math.max(photo.width, photo.height));
  const width = round(photo.width * factor), height = round(photo.height * factor);
  const photoCanvas = makeCanvas(width, height), photoContext = context(photoCanvas);
  photoContext.drawImage(photo, 0, 0, width, height);
  const pixels = photoContext.getImageData(0, 0, width, height);
  const mask = makeCanvas(width, height), maskContext = context(mask);
  const alpha = new Uint8Array(width * height);
  if (document.mask) {
    maskContext.drawImage(document.mask, 0, 0, width, height);
    const data = maskContext.getImageData(0, 0, width, height);
    // Segmentation returns opaque grayscale; also support transparent white cutout masks.
    let transparent = false;
    for (let i = 3; i < data.data.length; i += 4) if (data.data[i] < 255) { transparent = true; break; }
    for (let i = 0; i < alpha.length; i++) {
      alpha[i] = transparent ? data.data[i * 4 + 3] : data.data[i * 4];
    }
  } else {
    for (let i = 0; i < alpha.length; i++) alpha[i] = pixels.data[i * 4 + 3];
  }
  const bounds = maskBounds(alpha, width, height) ?? { x: 0, y: 0, width, height };
  const model = { ...placement(document, bounds, width, height), width, height };
  const maskPixels = maskContext.createImageData(width, height);
  for (let i = 0; i < alpha.length; i++) {
    maskPixels.data.set([255, 255, 255, alpha[i]], i * 4);
  }
  maskContext.putImageData(maskPixels, 0, 0);
  // Mask operations replay in source space without mutating the original mask.
  const b = model.imageBounds;
  const maskView = { fullWidth: width / b.width, fullHeight: height / b.height,
    x: b.x * width / b.width, y: b.y * height / b.height };
  for (const stroke of document.maskEditsApplied || !b.width || !b.height ? [] : document.edits ?? []) {
    if (stroke.tool === "mask-add" || stroke.tool === "mask-erase") drawStroke(maskContext, stroke, model, maskView, { mask: true });
  }
  const edited = maskContext.getImageData(0, 0, width, height).data;
  for (let i = 0; i < alpha.length; i++) {
    alpha[i] = Math.round(edited[i * 4 + 3] * pixels.data[i * 4 + 3] / 255);
    pixels.data[i * 4 + 3] = alpha[i];
  }
  const params = document.layerParams?.subject ?? {};
  const subjectCfg = { ...cfg };
  for (const [key, target] of [
    ["shadowLift", "CC_SHADOW_LIFT"], ["highlightReduce", "CC_HIGHLIGHT_REDUCE"],
    ["vibrance", "CC_VIBRANCE"], ["warmth", "CC_WARM_FILTER"], ["warmBoost", "CC_WARM_BOOST"],
  ]) if (Number.isFinite(params[key])) subjectCfg[target] = params[key];
  const strength = clamp(params.ccStrength ?? 1);
  const original = strength < 1 ? pixels.data.slice() : null;
  correctPixels(pixels, subjectCfg);
  if (original) for (let i = 0; i < pixels.data.length; i++) {
    if (i % 4 !== 3) pixels.data[i] = original[i] * (1 - strength) + pixels.data[i] * strength;
  }
  photoContext.putImageData(pixels, 0, 0);
  model.subject = photoCanvas;
  const edgeScale = Math.min(1, 1024 / Math.max(width, height));
  const edgeWidth = round(width * edgeScale), edgeHeight = round(height * edgeScale);
  const edgeAlpha = new Uint8Array(edgeWidth * edgeHeight);
  for (let y = 0; y < edgeHeight; y++) for (let x = 0; x < edgeWidth; x++) {
    edgeAlpha[y * edgeWidth + x] = alpha[Math.min(height - 1, Math.floor(y / edgeScale)) * width +
      Math.min(width - 1, Math.floor(x / edgeScale))];
  }
  const sourceEdges = sampleMaskEdges(edgeAlpha, edgeWidth, edgeHeight);
  model.edges = !b.width || !b.height ? [] : sourceEdges.map((p) => ({
    ...p, x: b.x + (p.x + 0.5) / edgeWidth * b.width, y: b.y + (p.y + 0.5) / edgeHeight * b.height,
  }));
  const colorAt = ({ x, y, nx = 0, ny = 0 }) => {
    if (!b.width || !b.height) return [0.5, 0.5, 0.5];
    const inset = document.px(cfg.EDGE_SAMPLE_WIDTH) / 2;
    const sx = Math.round(((x - nx * inset / document.outputWidth - b.x) / b.width) * width);
    const sy = Math.round(((y - ny * inset / document.outputHeight - b.y) / b.height) * height);
    const index = (clamp(sy, 0, height - 1) * width + clamp(sx, 0, width - 1)) * 4;
    return [pixels.data[index], pixels.data[index + 1], pixels.data[index + 2]].map((v) => v / 255);
  };
  model.colorAt = colorAt;
  const histogram = new Map(), sum = [0, 0, 0];
  let count = 0;
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] < 128) continue;
    const color = [pixels.data[i * 4], pixels.data[i * 4 + 1], pixels.data[i * 4 + 2]];
    for (let c = 0; c < 3; c++) sum[c] += color[c];
    const key = color.map((v) => v >> 5).join(",");
    histogram.set(key, (histogram.get(key) ?? 0) + 1); count++;
  }
  model.average = sum.map((v) => count ? v / count / 255 : 0.5);
  const dominant = [...histogram.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  model.dominant = dominant ? dominant.split(",").map((v) => (Number(v) * 32 + 16) / 255) : model.average;
  return model;
}

export async function createPainting(document, { onProgress, assetsQuality = "preview" } = {}) {
  if (!document?.photo?.width || !document.photo.height) throw new TypeError("A decoded document.photo is required.");
  const start = performance.now();
  const cfg = { ...getPresetCopy(document.preset), ...document.presetParams };
  onProgress?.({ stage: "brushes", progress: 0.1, message: "Loading watercolor brushes" });
  const assets = await loadAssets(assetsQuality);
  let compositor;
  try { compositor = new PaintingCompositor(assets); }
  catch (error) { for (const image of assets.values()) image.close(); throw error; }
  const layers = createLayers(cfg), cache = new Map();
  let model, modelKey, disposed = false;
  const scratch = makeCanvas(), stage = makeCanvas();
  const modelSignature = () => JSON.stringify([
    document.placement, document.maskEditsApplied,
    (document.edits ?? []).filter((s) => s.tool === "mask-add" || s.tool === "mask-erase"),
    ...["ccStrength", "shadowLift", "highlightReduce", "vibrance", "warmth", "warmBoost"]
      .map((key) => document.layerParams?.subject?.[key]),
  ]);
  const refresh = () => {
    const key = modelSignature();
    if (key !== modelKey) {
      model = createModel(document, cfg, assetsQuality === "hd" || assetsQuality === "full" ? 2048 : 1024);
      modelKey = key; cache.clear();
      painting.subjectBounds = { ...model.subjectBounds };
      painting.imageBounds = { ...model.imageBounds };
    }
  };
  const effective = (layer) => ({ ...layer.params, ...document.layerParams?.[layer.id] });
  const stampsFor = (layer, params) => {
    const key = JSON.stringify([params.size, params.spacing, params.jiggle, document.outputWidth / document.outputHeight]);
    const cached = cache.get(layer.id);
    if (cached?.key === key) return cached.stamps;
    const stamps = generateStamps(layer, params, model, document, cfg);
    cache.set(layer.id, { key, stamps });
    return stamps;
  };
  const drawSubject = (ctx, view, scale = 1) => {
    const b = model.imageBounds;
    const cx = model.subjectBounds.x + model.subjectBounds.width / 2;
    const cy = model.subjectBounds.y + model.subjectBounds.height / 2;
    ctx.drawImage(model.subject,
      (cx + (b.x - cx) * scale) * view.fullWidth - view.x,
      (cy + (b.y - cy) * scale) * view.fullHeight - view.y,
      b.width * view.fullWidth * scale, b.height * view.fullHeight * scale);
  };
  const drawLayer = (layer, params, ctx, view) => {
    const { width, height } = scratch;
    const scale = Math.max(view.fullWidth, view.fullHeight) / Math.max(document.referenceWidth, document.referenceHeight);
    if (layer.kind === "paper") {
      ctx.fillStyle = `rgb(${cfg.DARK_BACKGROUND_COLOR.join(",")})`; ctx.fillRect(0, 0, width, height);
      if (cfg.PAPER_TEXTURE_ENABLED) {
        ctx.globalAlpha = cfg.PAPER_TEXTURE_OPACITY;
        ctx.drawImage(assets.get("papertexture.png"), -view.x, -view.y, view.fullWidth, view.fullHeight);
        ctx.globalAlpha = 1;
        ctx.fillStyle = `rgba(255,220,160,${cfg.PAPER_TEXTURE_WARMTH * 0.15})`; ctx.fillRect(0, 0, width, height);
      }
    } else if (["subject", "oil", "glow"].includes(layer.kind)) {
      if (layer.kind === "oil") {
        const detail = params.detail ?? 1, sharpening = params.sharpening ?? 1;
        ctx.filter = `contrast(${cfg.OIL_FINAL_CONTRAST * (1 + 0.04 * detail + 0.04 * sharpening)}) saturate(${1 + detail * 0.08})`;
      }
      if (layer.kind === "glow") ctx.filter = `blur(${cfg.GLOW_BLUR * scale}px)`;
      drawSubject(ctx, view, layer.kind === "glow" ? 1 + cfg.GLOW_WIDTH / 500 : 1);
      ctx.filter = "none";
      if (layer.kind === "glow") {
        ctx.globalCompositeOperation = "source-in"; ctx.fillStyle = `rgb(${cfg.GLOW_COLOR.join(",")})`;
        ctx.fillRect(0, 0, width, height); ctx.globalCompositeOperation = "source-over";
      }
      if (layer.kind === "subject" && cfg.WATERCOLOR_TEXTURE_ENABLED) {
        ctx.globalCompositeOperation = "source-atop"; ctx.globalAlpha = cfg.WATERCOLOR_TEXTURE_OPACITY * 0.25;
        ctx.drawImage(assets.get("watercolortexture.png"), -view.x, -view.y, view.fullWidth, view.fullHeight);
        ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
      }
    } else if (layer.kind === "curve") {
      ctx.strokeStyle = "rgba(65,45,32,.45)"; ctx.lineWidth = 2 * scale;
      for (const p of model.edges.filter((_, i) => i % 12 === 0)) {
        ctx.beginPath(); ctx.arc(p.x * view.fullWidth - view.x, p.y * view.fullHeight - view.y, scale, 0, 2 * Math.PI); ctx.stroke();
      }
    } else if (layer.kind !== "custom") {
      const stamps = stampsFor(layer, params).map((s) => ({
        ...s, x: s.x * view.fullWidth - view.x, y: s.y * view.fullHeight - view.y,
        size: s.size * Math.max(view.fullWidth, view.fullHeight),
      })).filter((s) => s.x + s.size > 0 && s.y + s.size > 0 && s.x - s.size < width && s.y - s.size < height);
      if (layer.kind === "blur") {
        ctx.filter = `blur(${cfg.BLUR_AMOUNT * scale}px) saturate(${cfg.SATURATION_BOOST}) contrast(${cfg.CONTRAST_BOOST})`;
        for (let i = cfg.NUM_BLUR_LAYERS; i >= 1; i--) drawSubject(ctx, view, 1 + i * cfg.LAYER_SCALE_STEP);
        ctx.filter = "none";
        ctx.globalCompositeOperation = "destination-in";
        ctx.drawImage(compositor.drawStamps(params.brush, stamps, width, height), 0, 0);
        ctx.globalCompositeOperation = "source-over";
      } else ctx.drawImage(compositor.drawStamps(params.brush, stamps, width, height), 0, 0);
    }
  };
  const painting = {
    layers, subjectBounds: null, imageBounds: null, assetsQuality, timings: {},
    rerenderLayer(id) {
      if (!layers.some((layer) => layer.id === id)) throw new RangeError(`Unknown painting layer: ${id}`);
      cache.delete(id);
      if (id === "subject") modelKey = null;
    },
    render({
      canvas,
      width = canvas?.width ?? document.outputWidth, height = canvas?.height ?? document.outputHeight, x = 0, y = 0,
      fullWidth = width, fullHeight = height,
    } = {}) {
      if (disposed) throw new Error("Painting has been disposed.");
      for (const value of [width, height, fullWidth, fullHeight]) {
        if (!Number.isFinite(value) || value <= 0) throw new RangeError("Render dimensions must be positive.");
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new RangeError("Tile coordinates must be finite.");
      canvas ??= makeCanvas(round(width), round(height));
      refresh();
      const begun = performance.now();
      width = round(width); height = round(height);
      const scale = Math.max(fullWidth, fullHeight) / Math.max(document.referenceWidth, document.referenceHeight);
      const softRadius = Math.max(0, ...(document.edits ?? []).filter((s) => /soft|watercolor/.test(s.tip)).map((s) =>
        s.size * Math.max(fullWidth, fullHeight) * 0.75));
      const pad = Math.ceil(Math.max(cfg.BLUR_AMOUNT * scale * 3, cfg.GLOW_BLUR * scale * 3, softRadius) + 4);
      const paddedWidth = width + pad * 2, paddedHeight = height + pad * 2;
      compositor.resize(paddedWidth, paddedHeight);
      scratch.width = stage.width = paddedWidth; scratch.height = stage.height = paddedHeight;
      const ctx = context(scratch), target = context(stage);
      const view = { x: x - pad, y: y - pad, fullWidth, fullHeight };
      target.fillStyle = cfg.DARK_BACKGROUND_ENABLED ? `rgb(${cfg.DARK_BACKGROUND_COLOR.join(",")})` : "#ffffff";
      target.fillRect(0, 0, paddedWidth, paddedHeight);
      for (const layer of layers) {
        const params = effective(layer);
        if ((params.visible ?? layer.visible) === false) continue;
        const opacity = clamp(params.opacity ?? layer.opacity);
        if (!opacity) continue;
        ctx.clearRect(0, 0, paddedWidth, paddedHeight);
        drawLayer(layer, params, ctx, view);
        for (const stroke of document.edits ?? []) {
          if (stroke.tool === "mask-add" || stroke.tool === "mask-erase") continue;
          if ((stroke.layerId ?? "custom_detail") === layer.id) drawStroke(ctx, stroke, model, view);
        }
        target.globalAlpha = opacity;
        target.drawImage(compositor.correct(scratch, params), 0, 0);
      }
      canvas.width = width; canvas.height = height;
      context(canvas).drawImage(stage, pad, pad, width, height, 0, 0, width, height);
      painting.timings.lastRenderMs = performance.now() - begun;
      return canvas;
    },
    renderTile(options) { return painting.render(options); },
    dispose() {
      if (disposed) return;
      disposed = true; compositor.dispose(); cache.clear();
      for (const image of assets.values()) image.close();
      model.subject.width = model.subject.height = scratch.width = scratch.height = stage.width = stage.height = 1;
    },
  };
  onProgress?.({ stage: "sampling", progress: 0.6, message: "Sampling subject edges and colors" });
  try { refresh(); }
  catch (error) { compositor.dispose(); for (const image of assets.values()) image.close(); throw error; }
  painting.timings.createMs = performance.now() - start;
  onProgress?.({ stage: "painting", progress: 1, message: "Painting ready — approximate browser effects" });
  return painting;
}
