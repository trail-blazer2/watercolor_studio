/**
 * 🎨 Watercolor Artist Studio — Real-Time 60 FPS Layer & Brush Engine
 * Guarantees 100% pixel-identical initial output + live per-layer editing
 */

// ============================================================================
// 1. PYTHON MERSENNE TWISTER (Exact match for Python random.Random(seed))
// ============================================================================
class PythonRandom {
  constructor(seed = 42) {
    this.MT = new Uint32Array(624);
    this.index = 624;
    this.seed(seed);
  }
  seed(n) {
    n = Math.abs(Math.floor(n)) >>> 0;
    this.init_genrand(19650218);
    let i = 1, j = 0;
    const key = [n];
    for (let k = 624; k > 0; k--) {
      const prev = this.MT[i - 1];
      const x = prev ^ (prev >>> 30);
      const mul = Math.imul(x, 1664525) >>> 0;
      this.MT[i] = ((this.MT[i] ^ mul) + key[j] + j) >>> 0;
      i++; j = (j + 1) % key.length;
      if (i >= 624) { this.MT[0] = this.MT[623]; i = 1; }
    }
    for (let k = 623; k > 0; k--) {
      const prev = this.MT[i - 1];
      const x = prev ^ (prev >>> 30);
      const mul = Math.imul(x, 1566083941) >>> 0;
      this.MT[i] = ((this.MT[i] ^ mul) - i) >>> 0;
      i++;
      if (i >= 624) { this.MT[0] = this.MT[623]; i = 1; }
    }
    this.MT[0] = 0x80000000 >>> 0;
  }
  init_genrand(s) {
    this.MT[0] = s >>> 0;
    for (let i = 1; i < 624; i++) {
      const x = this.MT[i - 1] ^ (this.MT[i - 1] >>> 30);
      this.MT[i] = (Math.imul(x, 1812433253) + i) >>> 0;
    }
    this.index = 624;
  }
  genrand_int32() {
    if (this.index >= 624) {
      const mag01 = [0x0, 0x9908b0df];
      for (let kk = 0; kk < 227; kk++) {
        const y = (this.MT[kk] & 0x80000000) | (this.MT[kk + 1] & 0x7fffffff);
        this.MT[kk] = this.MT[kk + 397] ^ (y >>> 1) ^ mag01[y & 0x1];
      }
      for (let kk = 227; kk < 623; kk++) {
        const y = (this.MT[kk] & 0x80000000) | (this.MT[kk + 1] & 0x7fffffff);
        this.MT[kk] = this.MT[kk - 227] ^ (y >>> 1) ^ mag01[y & 0x1];
      }
      const y = (this.MT[623] & 0x80000000) | (this.MT[0] & 0x7fffffff);
      this.MT[623] = this.MT[396] ^ (y >>> 1) ^ mag01[y & 0x1];
      this.index = 0;
    }
    let y = this.MT[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  random() {
    const a = this.genrand_int32() >>> 5;
    const b = this.genrand_int32() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }
  uniform(a, b) {
    return a + (b - a) * this.random();
  }
}

// ============================================================================
// 2. LAYER STACK DEFINITIONS (Exact Z-Order from watercolor.py)
// ============================================================================
const LAYER_DEFINITIONS = [
  // OVER-SUBJECT LAYERS (Top of stack)
  { id: "custom_detail", name: "✨ Eye Reflections & Detail Brush", group: "Over Subject", type: "custom", defaultVis: true },
  { id: "small_dots", name: "⚪ Small White Dots", group: "Over Subject", type: "brush_edge", seedOffset: 700, prefix: "SMALL_WHITE_DOTS" },
  { id: "outer_white", name: "⭕ Outer Frame Brush", group: "Over Subject", type: "brush_ellipse", seedOffset: 550, prefix: "OUTER_WHITE_BRUSH" },
  { id: "white_ellipse", name: "⭕ Inner Frame Brush", group: "Over Subject", type: "brush_ellipse", seedOffset: 500, prefix: "WHITE_BRUSH" },
  { id: "dark_ellipse", name: "🌘 Bottom Dark Ellipse", group: "Over Subject", type: "brush_ellipse_dark", seedOffset: 600, prefix: "DARK_ELLIPSE" },
  { id: "edge_curve", name: "📐 Edge Curve Tri-Tone", group: "Over Subject", type: "effect", prefix: "EDGE_CURVE" },
  { id: "oil_paint", name: "🖼️ Oil Paint Effect", group: "Subject & Oil", type: "oil", prefix: "OIL_PAINT" },

  // SUBJECT LAYER
  { id: "subject", name: "🦁 Subject & Color Correction", group: "Subject & Oil", type: "subject", prefix: "CC" },

  // UNDER-SUBJECT LAYERS (Fully editable & erasable!)
  { id: "shadow", name: "🌑 Bottom Shadow Brush", group: "Under Subject", type: "effect", prefix: "SHADOW_BRUSH" },
  { id: "colordrop", name: "💧 Bottom Color Drops", group: "Under Subject", type: "effect", prefix: "COLORDROP_BRUSH" },
  { id: "accent_strokes_2", name: "🔥 Accent Strokes (2nd pass)", group: "Under Subject", type: "effect", prefix: "ACCENT_STROKES" },
  { id: "glow", name: "☀️ Subject Rim Glow", group: "Under Subject", type: "glow", prefix: "GLOW" },
  { id: "dark_dots", name: "⚫ Dark Splatter Dots", group: "Under Subject", type: "brush_dots", seedOffset: 400, prefix: "DARK_DOTS" },
  { id: "white_dots", name: "⚪ White Splatter Dots", group: "Under Subject", type: "brush_dots", seedOffset: 300, prefix: "WHITE_DOTS" },
  { id: "accent_strokes", name: "🔥 Accent Strokes (Smart)", group: "Under Subject", type: "accent", prefix: "ACCENT_STROKES" },
  { id: "color_brush", name: "🌈 Color Wash Brush (brush2)", group: "Under Subject", type: "brush_edge", seedOffset: 100, prefix: "COLOR_BRUSH" },
  { id: "mask_blur", name: "🌫️ Blurred Aura + Square Mask", group: "Under Subject", type: "mask_blur", seedOffset: 0, prefix: "MASK_BRUSH" },
  { id: "sharp_brush", name: "📐 Sharp Edge Brush", group: "Under Subject", type: "brush_edge", seedOffset: 150, prefix: "SHARP_BRUSH" },
  { id: "bg_brush", name: "🎨 Large Background Wash", group: "Under Subject", type: "brush_edge", seedOffset: 200, prefix: "BG_BRUSH" },
  { id: "bg_paper", name: "📜 Paper Texture / Canvas BG", group: "Canvas Base", type: "paper", prefix: "PAPER_TEXTURE" },
];

// ============================================================================
// 3. STUDIO STATE
// ============================================================================
const STATE = {
  apiUrl: localStorage.getItem("wc_modal_api") || "https://autorauto--watercolor-studio-gpu-watercolorservice-api.modal.run",
  uploadedFile: null,
  originalImg: null,
  payload: null,
  brushImages: {}, // filename -> HTMLImageElement
  layers: {},      // id -> { vis, baseCanvas, paintCanvas, params, dirtyProcedural }
  selectedLayerId: "bg_brush",
  activeTool: "pan", // 'pan', 'brush', 'eraser', 'mask'
  maskModePaintKeep: true,
  maskCanvas: null,
  zoom: 0.45,
  panX: 40,
  panY: 20,
  isDragging: false,
  lastMouse: { x: 0, y: 0 },
};

// DOM Elements
const displayCanvas = document.getElementById("displayCanvas");
const displayCtx = displayCanvas.getContext("2d");
const canvasContainer = document.getElementById("canvasContainer");
const viewport = document.getElementById("viewport");
const brushCursor = document.getElementById("brushCursor");

// ============================================================================
// 4. INITIALIZATION & API CONNECTION
// ============================================================================
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("apiUrlInput").value = STATE.apiUrl;
  bindUIEvents();
  fetchBrushTextures();
});

async function fetchBrushTextures() {
  try {
    const res = await fetch(`${STATE.apiUrl.replace(/\/$/, "")}/brushes`);
    if (!res.ok) return;
    const data = await res.json();
    for (const [name, b64] of Object.entries(data)) {
      STATE.brushImages[name] = await loadImage(b64);
    }
    console.log("✅ Loaded brush textures for real-time rendering:", Object.keys(STATE.brushImages));
  } catch (err) {
    console.warn("Brush textures will load on first generation:", err);
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function createCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

// ============================================================================
// 5. UPLOAD & MODAL GPU GENERATION
// ============================================================================
function bindUIEvents() {
  const fileInput = document.getElementById("fileInput");
  const generateBtn = document.getElementById("generateBtn");
  const presetSelect = document.getElementById("presetSelect");

  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    STATE.uploadedFile = file;
    const url = URL.createObjectURL(file);
    STATE.originalImg = await loadImage(url);
    window.dispatchEvent(new CustomEvent("watercolor:source", {
      detail: { photo: file, preset: presetSelect.value },
    }));
    STATE.maskCanvas = null; // Reset custom mask on new photo
    generateBtn.disabled = false;
    runModalGeneration();
  });

  presetSelect.addEventListener("change", () => {
    if (STATE.uploadedFile) {
      runModalGeneration();
    }
  });

  generateBtn.addEventListener("click", () => {
    if (STATE.uploadedFile) runModalGeneration();
  });

  // API Bar Toggle
  document.getElementById("apiConfigBtn").addEventListener("click", () => {
    document.getElementById("apiBar").classList.toggle("hidden");
  });

  document.getElementById("saveApiBtn").addEventListener("click", async () => {
    const val = document.getElementById("apiUrlInput").value.trim().replace(/\/$/, "");
    STATE.apiUrl = val;
    localStorage.setItem("wc_modal_api", val);
    const statusEl = document.getElementById("apiStatusText");
    statusEl.textContent = "Testing...";
    try {
      const r = await fetch(`${val}/health`);
      if (r.ok) {
        statusEl.textContent = "✅ Connected!";
        statusEl.style.color = "#059669";
        fetchBrushTextures();
      } else {
        statusEl.textContent = "❌ Error " + r.status;
        statusEl.style.color = "#dc2626";
      }
    } catch (e) {
      statusEl.textContent = "❌ Failed to reach URL";
      statusEl.style.color = "#dc2626";
    }
  });

  // Tool Buttons
  document.querySelectorAll(".tool-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setActiveTool(btn.dataset.tool);
    });
  });

  // Manual Brush Sliders
  const sizeSlider = document.getElementById("manualBrushSize");
  sizeSlider.addEventListener("input", () => {
    document.getElementById("manualSizeVal").textContent = sizeSlider.value;
    updateCursorPreview();
  });

  const opSlider = document.getElementById("manualBrushOpacity");
  opSlider.addEventListener("input", () => {
    document.getElementById("manualOpacityVal").textContent = opSlider.value;
  });

  document.querySelectorAll(".swatch").forEach((sw) => {
    sw.addEventListener("click", () => {
      document.getElementById("manualBrushColor").value = sw.dataset.color;
    });
  });

  // Cutout Mask Mode Buttons
  document.getElementById("maskModeBtn").addEventListener("click", () => {
    setActiveTool(STATE.activeTool === "mask" ? "pan" : "mask");
  });
  document.getElementById("maskKeepBtn").addEventListener("click", () => {
    STATE.maskModePaintKeep = true;
    document.getElementById("maskKeepBtn").className = "btn btn-sm btn-primary";
    document.getElementById("maskRemoveBtn").className = "btn btn-sm btn-outline";
  });
  document.getElementById("maskRemoveBtn").addEventListener("click", () => {
    STATE.maskModePaintKeep = false;
    document.getElementById("maskRemoveBtn").className = "btn btn-sm btn-primary";
    document.getElementById("maskKeepBtn").className = "btn btn-sm btn-outline";
  });
  document.getElementById("applyMaskRegenBtn").addEventListener("click", () => {
    setActiveTool("pan");
    runModalGeneration(true);
  });

  // Zoom Controls
  document.getElementById("zoomInBtn").addEventListener("click", () => setZoom(STATE.zoom * 1.2));
  document.getElementById("zoomOutBtn").addEventListener("click", () => setZoom(STATE.zoom / 1.2));
  document.getElementById("zoomFitBtn").addEventListener("click", fitCanvasToViewport);

  // Reset Active Layer
  document.getElementById("resetLayerBtn").addEventListener("click", () => {
    resetLayerToOriginal(STATE.selectedLayerId);
  });

  // Export Buttons
  document.getElementById("exportPngBtn").addEventListener("click", () => exportImage("png"));
  document.getElementById("exportJpgBtn").addEventListener("click", () => exportImage("jpeg"));

  // Viewport Mouse Events
  setupCanvasInteractions();
}

async function runModalGeneration(useEditedMask = false) {
  if (!STATE.uploadedFile) return;

  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  loadingOverlay.classList.remove("hidden");
  loadingText.textContent = "Generating multi-layer watercolor on Modal GPU...";

  try {
    if (Object.keys(STATE.brushImages).length === 0) {
      fetchBrushTextures();
    }

    const formData = new FormData();
    formData.append("image_file", STATE.uploadedFile);
    formData.append("preset_key", document.getElementById("presetSelect").value);

    if (useEditedMask && STATE.maskCanvas) {
      formData.append("mask_base64", STATE.maskCanvas.toDataURL("image/png"));
    }

    const res = await fetch(`${STATE.apiUrl.replace(/\/$/, "")}/generate`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Modal API returned status ${res.status}`);
    }

    const payload = await res.json();
    await initializeStudioFromPayload(payload);
  } catch (err) {
    alert("Error generating painting: " + err.message);
    console.error(err);
  } finally {
    loadingOverlay.classList.add("hidden");
  }
}

// ============================================================================
// 6. INITIALIZE UN-FLATTENED LAYERS FROM PAYLOAD
// ============================================================================
async function initializeStudioFromPayload(payload) {
  invalidateStack();
  _sampleCache.clear();
  STATE.payload = payload;
  const [w, h] = payload.canvas_size;
  displayCanvas.width = w;
  displayCanvas.height = h;

  document.getElementById("emptyState").classList.add("hidden");
  canvasContainer.classList.remove("hidden");
  document.getElementById("maskModeBtn").disabled = false;
  document.getElementById("exportPngBtn").disabled = false;
  document.getElementById("exportJpgBtn").disabled = false;
  document.getElementById("resetLayerBtn").disabled = false;
  document.getElementById("renderTimeBadge").textContent = `⚡ ${payload.elapsed_sec}s GPU`;

  // Update Quick Swatches with Subject Average & Dominant Color
  if (payload.average_color) {
    const hex = rgbToHex(...payload.average_color);
    const sw = document.getElementById("avgColorSwatch");
    sw.style.background = hex;
    sw.dataset.color = hex;
  }
  if (payload.dominant_color) {
    const hex = rgbToHex(...payload.dominant_color);
    const sw = document.getElementById("domColorSwatch");
    sw.style.background = hex;
    sw.dataset.color = hex;
  }

  // Load Cutout Mask into editable maskCanvas
  const maskImg = await loadImage(payload.mask_png);
  STATE.maskCanvas = createCanvas(maskImg.width, maskImg.height);
  STATE.maskCanvas.getContext("2d").drawImage(maskImg, 0, 0);
  const maskBlob = await new Promise((resolve) => STATE.maskCanvas.toBlob(resolve, "image/png"));
  window.dispatchEvent(new CustomEvent("watercolor:generated", {
    detail: { payload, mask: maskBlob },
  }));

  // Preload all server-rendered layer PNGs in parallel
  const rawImgs = {};
  await Promise.all(
    Object.entries(payload.layers).map(async ([key, b64]) => {
      rawImgs[key] = await loadImage(b64);
    })
  );
  STATE.serverImages = rawImgs;

  const cfg = payload.preset_settings;
  STATE.layers = {};

  for (const def of LAYER_DEFINITIONS) {
    const baseCanvas = createCanvas(w, h);
    const baseCtx = baseCanvas.getContext("2d");

    // Paint & Erase buffers for manual brush tool on EVERY layer
    const paintCanvas = createCanvas(w, h);
    const eraseCanvas = createCanvas(w, h);

    let enabled = true;
    if (def.id === "custom_detail") {
      enabled = true;
    } else if (def.id === "mask_blur") {
      enabled = !!cfg.MASK_BRUSH_ENABLED;
      if (enabled) {
        // Exact composite: blur_bg masked by mask_brush
        baseCtx.drawImage(rawImgs.blur_bg, 0, 0);
        baseCtx.globalCompositeOperation = "destination-in";
        baseCtx.drawImage(rawImgs.mask_brush, 0, 0);
        baseCtx.globalCompositeOperation = "source-over";
      }
    } else if (def.id === "subject") {
      enabled = !!cfg.SHOW_SUBJECT;
      baseCtx.drawImage(rawImgs.subject_cc, 0, 0);
    } else if (def.id === "oil_paint") {
      enabled = !!cfg.OIL_PAINT_ENABLED;
      baseCtx.drawImage(rawImgs.oil_paint_full, 0, 0);
    } else if (def.id === "bg_paper") {
      enabled = true;
      baseCtx.drawImage(rawImgs.bg_paper, 0, 0);
    } else {
      const enKey = `${def.prefix}_ENABLED`;
      enabled = cfg[enKey] !== undefined ? !!cfg[enKey] : true;
      if (rawImgs[def.id]) {
        baseCtx.drawImage(rawImgs[def.id], 0, 0);
      }
    }

    STATE.layers[def.id] = {
      def,
      vis: enabled,
      origImage: rawImgs[def.id] || null,
      baseCanvas,
      paintCanvas,
      eraseCanvas,
      params: buildInitialLayerParams(def, cfg),
    };
  }

  renderLayerListUI();
  selectLayer(STATE.selectedLayerId || "bg_brush");
  fitCanvasToViewport();
  compositeAllLayers();
}

const _pendingRerender = new Set();
let _rafId = 0;
function scheduleRerender(layerId) {
  _pendingRerender.add(layerId);
  if (_rafId) return;
  _rafId = requestAnimationFrame(() => {
    _rafId = 0;
    for (const id of _pendingRerender) rerenderProceduralLayer(id);
    _pendingRerender.clear();
  });
}


function buildInitialLayerParams(def, cfg) {
  const p = def.prefix;
  if (def.type === "brush_edge" || def.type === "brush_dots" || def.type === "brush_ellipse" || def.type === "brush_ellipse_dark") {
    return {
      opacity: cfg[`${p}_OPACITY`] ?? 0.6,
      baseOpacity: cfg[`${p}_OPACITY`] ?? 0.6,
      size: cfg[`${p}_SIZE`] ?? 600,
      spacing: cfg[`${p}_SPACING`] ?? 200,
      jiggle: cfg[`${p}_JIGGLE`] ?? cfg.BRUSH_DIRECTION_JIGGLE ?? 0.6,
      brightness: 100,
      contrast: 100,
      saturation: 100,
      brushFile: cfg[`${p}_FILE`] || defaultBrushForLayer(def.id, cfg),
    };
  }
  if (def.type === "mask_blur") {
    return {
      opacity: cfg.MASK_BRUSH_OPACITY ?? 0.5,
      baseOpacity: cfg.MASK_BRUSH_OPACITY ?? 0.5,
      size: cfg.MASK_BRUSH_SIZE ?? 900,
      spacing: cfg.MASK_BRUSH_SPACING ?? 200,
      jiggle: cfg.BRUSH_DIRECTION_JIGGLE ?? 0.8,
      brightness: 100,
      contrast: 100,
      saturation: 100,
      brushFile: "square.png",
    };
  }
  if (def.type === "accent") {
    return {
      opacity: 1.0,
      baseOpacity: 1.0,
      size: 300,
      spacing: 50,
      brightness: 100,
      contrast: 100,
      saturation: 100,
      brushFile: "bgbrush.png",
    };
  }
  if (def.type === "oil") {
    return {
      opacity: cfg.OIL_PAINT_OPACITY ?? 0.65,
      baseOpacity: 1.0, // oil_paint_full is 100% opacity so slider scales 0..100% smoothly!
      brightness: 100,
      contrast: 100,
      saturation: 100,
    };
  }
  if (def.type === "subject") {
    return {
      opacity: 1.0,
      baseOpacity: 1.0,
      ccBlend: 100, // 100% = Preset Color Correction, 0% = Raw Uncorrected
      brightness: 100,
      contrast: 100,
      saturation: 100,
    };
  }
  return {
    opacity: 1.0,
    baseOpacity: 1.0,
    brightness: 100,
    contrast: 100,
    saturation: 100,
  };
}

function defaultBrushForLayer(id, cfg) {
  const map = {
    bg_brush: "bgbrush.png",
    sharp_brush: "sharpbrush.png",
    color_brush: "brush2.png",
    white_dots: "whitedots.png",
    dark_dots: "darkdots.png",
    small_dots: "whitedots.png",
    dark_ellipse: "bgbrush.png",
    white_ellipse: cfg.ELLIPSE_BRUSH_MODE === "dark" ? (cfg.ELLIPSE_BRUSH_FILE || "square.png") : "whitebrush.png",
    outer_white: cfg.ELLIPSE_BRUSH_MODE === "dark" ? (cfg.ELLIPSE_BRUSH_FILE || "square.png") : "whitebrush.png",
  };
  return map[id] || "bgbrush.png";
}

// ============================================================================
// 7. REAL-TIME COMPOSITOR (60 FPS)
// ============================================================================
const _tempLayerCanvas = document.createElement("canvas");

const _tempLayerCanvas = document.createElement("canvas");
const _stack = { id: null, dirty: true, below: null, above: null };
function invalidateStack() { _stack.dirty = true; }

function drawLayerInto(ctx, def) {
  const lyr = STATE.layers[def.id];
  if (!lyr || !lyr.vis) return;
  const p = lyr.params;
  const op = p.baseOpacity > 0 ? Math.min(2.5, p.opacity / p.baseOpacity) : p.opacity;
  if (op <= 0.001) return;
  const [w, h] = STATE.payload.canvas_size;
  if (_tempLayerCanvas.width !== w || _tempLayerCanvas.height !== h) {
    _tempLayerCanvas.width = w; _tempLayerCanvas.height = h;
  }
  const t = _tempLayerCanvas.getContext("2d");
  t.clearRect(0, 0, w, h);
  const f = p.brightness !== 100 || p.contrast !== 100 || p.saturation !== 100;
  t.filter = f ? `brightness(${p.brightness}%) contrast(${p.contrast}%) saturate(${p.saturation}%)` : "none";
  if (def.id === "subject" && p.ccBlend < 100 && STATE.serverImages.subject_raw) {
    t.drawImage(STATE.serverImages.subject_raw, 0, 0);
    t.globalAlpha = p.ccBlend / 100;
    t.drawImage(lyr.baseCanvas, 0, 0);
    t.globalAlpha = 1;
  } else {
    t.drawImage(lyr.baseCanvas, 0, 0);
  }
  t.filter = "none";
  t.drawImage(lyr.paintCanvas, 0, 0);
  t.globalCompositeOperation = "destination-out";
  t.drawImage(lyr.eraseCanvas, 0, 0);
  t.globalCompositeOperation = "source-over";
  ctx.save();
  ctx.globalAlpha = Math.min(1, op);
  ctx.drawImage(_tempLayerCanvas, 0, 0);
  if (op > 1) { ctx.globalAlpha = Math.min(1, op - 1); ctx.drawImage(_tempLayerCanvas, 0, 0); }
  ctx.restore();
}

function rebuildStack(selId) {
  const [w, h] = STATE.payload.canvas_size;
  _stack.below = _stack.below || createCanvas(w, h);
  _stack.above = _stack.above || createCanvas(w, h);
  [_stack.below, _stack.above].forEach((c) => { c.width = w; c.height = h; });
  const bCtx = _stack.below.getContext("2d"), aCtx = _stack.above.getContext("2d");
  const selIdx = LAYER_DEFINITIONS.findIndex((d) => d.id === selId);
  for (let i = LAYER_DEFINITIONS.length - 1; i >= 0; i--) {
    if (i === selIdx) continue;
    drawLayerInto(i > selIdx ? bCtx : aCtx, LAYER_DEFINITIONS[i]);
  }
  _stack.id = selId; _stack.dirty = false;
}

function compositeAllLayers() {
  if (!STATE.payload) return;
  if (STATE.activeTool === "mask" && STATE.originalImg && STATE.maskCanvas) {
    renderMaskEditorView();
    return;
  }
  const [w, h] = STATE.payload.canvas_size;
  const sel = STATE.selectedLayerId;
  if (_stack.dirty || _stack.id !== sel) rebuildStack(sel);
  displayCtx.clearRect(0, 0, w, h);
  displayCtx.drawImage(_stack.below, 0, 0);
  drawLayerInto(displayCtx, LAYER_DEFINITIONS.find((d) => d.id === sel));
  displayCtx.drawImage(_stack.above, 0, 0);
}

// ============================================================================
// 8. REAL-TIME PROCEDURAL BRUSH RE-RENDERER (Frequency, Size, Texture, Jiggle)
// ============================================================================
const _stampCanvas = document.createElement("canvas");
const _stampCtx = _stampCanvas.getContext("2d");

function colorizeAndDrawBrush(ctx, brushImg, x, y, size, angleDeg, rgbColor, opacity = 1.0) {
  if (!brushImg || size < 4) return;
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.abs(Math.cos(rad));
  const sin = Math.abs(Math.sin(rad));
  const bound = Math.ceil(size * (cos + sin)) + 4;

  if (_stampCanvas.width < bound || _stampCanvas.height < bound) {
    _stampCanvas.width = bound;
    _stampCanvas.height = bound;
  } else {
    _stampCtx.clearRect(0, 0, bound, bound);
  }

  _stampCtx.save();
  _stampCtx.translate(bound / 2, bound / 2);
  _stampCtx.rotate(-rad); // PIL rotate is counter-clockwise
  _stampCtx.drawImage(brushImg, -size / 2, -size / 2, size, size);

  if (rgbColor) {
    _stampCtx.globalCompositeOperation = "source-in";
    _stampCtx.fillStyle = `rgb(${rgbColor[0]}, ${rgbColor[1]}, ${rgbColor[2]})`;
    _stampCtx.fillRect(-bound / 2, -bound / 2, bound, bound);
  }
  _stampCtx.restore();

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.drawImage(_stampCanvas, 0, 0, bound, bound, Math.round(x - bound / 2), Math.round(y - bound / 2), bound, bound);
  ctx.restore();
}

const _sampleCache = new Map();
function sampleEdgePointsJS(edges, spacing) {
  if (!edges || edges.length === 0) return [];
  if (_sampleCache.has(spacing)) return _sampleCache.get(spacing);
  const s2 = spacing * spacing, cell = spacing / Math.SQRT2, W = 100000;
  const grid = new Map(), out = [];
  for (let i = 0; i < edges.length; i++) {
    const pt = edges[i], px = pt[0], py = pt[1];
    const gx = Math.floor(px / cell), gy = Math.floor(py / cell);
    let close = false;
    for (let dx = -2; dx <= 2 && !close; dx++) {
      for (let dy = -2; dy <= 2; dy++) {
        const nb = grid.get((gx + dx) * W + (gy + dy));
        if (nb !== undefined) {
          const ex = px - nb[0], ey = py - nb[1];
          if (ex * ex + ey * ey < s2) { close = true; break; }
        }
      }
    }
    if (!close) { out.push(pt); grid.set(gx * W + gy, pt); }
  }
  _sampleCache.set(spacing, out);
  return out;
}

function rerenderProceduralLayer(layerId) {
  const lyr = STATE.layers[layerId];
  if (!lyr || !STATE.payload) return;

  const def = lyr.def;
  const p = lyr.params;
  const cfg = STATE.payload.preset_settings;
  const [w, h] = STATE.payload.canvas_size;
  const [offX, offY] = STATE.payload.offset;
  const edges = STATE.payload.stroke_data.edges;
  const brushImg = STATE.brushImages[p.brushFile] || Object.values(STATE.brushImages)[0];
  if (!brushImg) return;

  const ctx = lyr.baseCanvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // 1. Edge-based Watercolor Layers (bg_brush, color_brush, sharp_brush, white_dots, dark_dots, small_dots)
  if (def.type === "brush_edge" || def.type === "brush_dots") {
    const rng = new PythonRandom((cfg.RANDOM_SEED || 42) + (def.seedOffset || 0));
    const points = sampleEdgePointsJS(edges, p.spacing);
    const avgColor = STATE.payload.average_color || [128, 128, 128];
    const domColor = STATE.payload.dominant_color;

    for (const pt of points) {
      const [x, y, nx, ny, cr, cg, cb, valid] = pt;
      if (!valid && def.type === "brush_edge" && def.id !== "small_dots") continue;

      let rgb = null;
      if (def.id === "bg_brush" || def.id === "color_brush" || def.id === "sharp_brush") {
        rgb = [cr, cg, cb];
        if (def.id === "bg_brush") {
          if (cfg.BG_BRUSH_MIX_AVERAGE && avgColor) {
            const amt = cfg.BG_BRUSH_MIX_AMOUNT ?? 0.2;
            rgb = rgb.map((v, idx) => Math.round(v * (1 - amt) + avgColor[idx] * amt));
          }
          if (cfg.BG_BRUSH_MIX_DOMINANT && domColor) {
            const amt = cfg.BG_BRUSH_MIX_DOMINANT_AMOUNT ?? 0.3;
            rgb = rgb.map((v, idx) => Math.round(v * (1 - amt) + domColor[idx] * amt));
          }
        }
        rgb = applyBrushWarmthJS(rgb, cfg);
        if (def.id === "bg_brush") {
          rgb = adjustBrightSatJS(rgb, cfg.BG_BRUSH_BRIGHTNESS ?? 0.6, cfg.BG_BRUSH_SATURATION ?? 1.0);
        }
      } else if (def.id === "dark_dots" && cfg.DARK_DOTS_USE_DOMINANT_COLOR && domColor) {
        const dk = cfg.DARK_DOTS_COLOR_DARKEN ?? 0.3;
        rgb = domColor.map((v) => Math.round(v * (1 - dk)));
      }

      let rotAngle = 0;
      if (def.id === "sharp_brush" || def.type === "brush_dots" || def.id === "small_dots") {
        rotAngle = rng.uniform(0, 360);
      } else {
        const deg = (Math.atan2(-ny, nx) * 180) / Math.PI;
        rotAngle = deg + 270 + rng.uniform(-180 * p.jiggle, 180 * p.jiggle);
      }

      let drawX = x + offX;
      let drawY = y + offY;
      if (def.id === "small_dots") {
        const jit = cfg.SMALL_WHITE_DOTS_DISTANCE_JITTER ?? -50;
        const jAmt = rng.uniform(-jit, jit);
        drawX += nx * jAmt;
        drawY += ny * jAmt;
      }

      const variedSize = Math.max(15, Math.floor(p.size * rng.uniform(0.7, 1.3)));
      colorizeAndDrawBrush(ctx, brushImg, drawX, drawY, variedSize, rotAngle, rgb, p.baseOpacity);
    }
  }

  // 2. Mask + Blurred Aura Layer
  else if (def.type === "mask_blur" && STATE.serverImages.blur_bg) {
    const rng = new PythonRandom(cfg.RANDOM_SEED || 42);
    const points = sampleEdgePointsJS(edges, p.spacing);
    const maskTemp = createCanvas(w, h);
    const mCtx = maskTemp.getContext("2d");

    for (const pt of points) {
      const [x, y, nx, ny] = pt;
      const deg = (Math.atan2(-ny, nx) * 180) / Math.PI;
      const rotAngle = deg + 270 + rng.uniform(-180 * p.jiggle, 180 * p.jiggle);
      const variedSize = Math.max(20, Math.floor(p.size * rng.uniform(0.7, 1.3)));
      colorizeAndDrawBrush(mCtx, brushImg, x + offX, y + offY, variedSize, rotAngle, null, p.baseOpacity);
    }

    ctx.drawImage(STATE.serverImages.blur_bg, 0, 0);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(maskTemp, 0, 0);
    ctx.globalCompositeOperation = "source-over";
  }

  // 3. Smart Accent Strokes Layer
  else if (def.type === "accent") {
    const cands = STATE.payload.stroke_data.accent_candidates || [];
    const spots = STATE.payload.stroke_data.accent_spots || [];
    const step = Math.max(1, Math.round(p.spacing / 50));

    for (let i = 0; i < cands.length; i += step) {
      const pt = cands[i];
      const deg = (Math.atan2(-pt.ny, pt.nx) * 180) / Math.PI + 270 + (i % 5 - 2) * 25;
      const sz = Math.max(20, Math.round(p.size * (0.75 + (i % 4) * 0.15)));
      colorizeAndDrawBrush(ctx, brushImg, pt.x, pt.y, sz, deg, pt.color, Math.min(1, 0.55 * pt.interest));
    }
    for (let i = 0; i < spots.length; i += step) {
      const sp = spots[i];
      const deg = (Math.atan2(-sp.ny, sp.nx) * 180) / Math.PI + 270;
      const sz = Math.max(25, Math.round(p.size * 1.05));
      const lightRgb = sp.color.map((c) => Math.min(255, Math.round(c + (255 - c) * 0.65)));
      colorizeAndDrawBrush(ctx, brushImg, sp.x, sp.y, sz, deg, lightRgb, 0.35);
    }
  }

  // 4. Ellipse Frame Layers (white_ellipse, outer_white, dark_ellipse)
  else if (def.type === "brush_ellipse" || def.type === "brush_ellipse_dark") {
    const rng = new PythonRandom((cfg.RANDOM_SEED || 42) + (def.seedOffset || 500));
    const isOuter = def.id === "outer_white";
    const isDark = def.id === "dark_ellipse";
    const inset = isDark ? 0 : cfg[isOuter ? "OUTER_WHITE_BRUSH_ELLIPSE_INSET" : "WHITE_BRUSH_ELLIPSE_INSET"] ?? -40;
    const cx = Math.floor(w / 2), cy = Math.floor(h / 2);
    const rx = Math.max(10, cx - inset), ry = Math.max(10, cy - inset);
    const circum = 2 * Math.PI * Math.max(rx, ry);
    const numPts = Math.max(1, Math.floor(circum / p.spacing));
    const bottomThresh = h * (1 - (cfg.DARK_ELLIPSE_BOTTOM_PERCENT ?? 0.15));

    for (let i = 0; i < numPts; i++) {
      const angle = (2 * Math.PI * i) / numPts;
      const ex = Math.floor(cx + rx * Math.cos(angle));
      const ey = Math.floor(cy + ry * Math.sin(angle));
      if (isDark && ey < bottomThresh) continue;

      const degToCenter = (Math.atan2(-(cy - ey), cx - ex) * 180) / Math.PI;
      const rotAngle = degToCenter + 90 + 180 + rng.uniform(-180 * p.jiggle, 180 * p.jiggle);
      const variedSize = Math.max(20, Math.floor(p.size * rng.uniform(0.7, 1.3)));

      let rgb = null;
      if (isDark && edges.length > 0) {
        const near = edges[i % edges.length];
        rgb = [near[4], near[5], near[6]];
      } else if (cfg.ELLIPSE_BRUSH_MODE === "dark") {
        rgb = [0, 0, 0];
      }

      colorizeAndDrawBrush(ctx, brushImg, ex, ey, variedSize, rotAngle, rgb, p.baseOpacity);
    }
  }

  compositeAllLayers();
}

function applyBrushWarmthJS(rgb, cfg) {
  let [r, g, b] = rgb;
  const amount = cfg.BRUSH_WARMTH || 0;
  if (amount > 0) {
    const baseWarm = amount * 0.3;
    const coolness = Math.max(0, (b - r) / 255.0);
    const cyanFactor = Math.max(0, (b + g - 2 * r) / 510.0);
    const warmStrength = baseWarm + Math.max(coolness, cyanFactor * 0.7) * amount * 0.7;
    r = Math.min(255, Math.max(0, r + Math.floor(warmStrength * 35)));
    g = Math.min(255, Math.max(0, g + Math.floor(warmStrength * 10)));
    b = Math.min(255, Math.max(0, b - Math.floor(warmStrength * 20)));
  }
  if (cfg.DARK_BACKGROUND_ENABLED && cfg.BRUSH_LIGHTEN_AMOUNT > 0) {
    const la = cfg.BRUSH_LIGHTEN_AMOUNT;
    r = Math.min(255, Math.floor(r + (255 - r) * la));
    g = Math.min(255, Math.floor(g + (255 - g) * la));
    b = Math.min(255, Math.floor(b + (255 - b) * la));
  }
  return [r, g, b];
}

function adjustBrightSatJS(rgb, brightness, saturation) {
  let r = (rgb[0] / 255) * brightness;
  let g = (rgb[1] / 255) * brightness;
  let b = (rgb[2] / 255) * brightness;
  const gray = (r + g + b) / 3;
  r = gray + (r - gray) * saturation;
  g = gray + (g - gray) * saturation;
  b = gray + (b - gray) * saturation;
  return [
    Math.max(0, Math.min(255, Math.floor(r * 255))),
    Math.max(0, Math.min(255, Math.floor(g * 255))),
    Math.max(0, Math.min(255, Math.floor(b * 255))),
  ];
}

// ============================================================================
// 9. RIGHT INSPECTOR: LAYER STACK & LIVE SLIDERS UI
// ============================================================================
function renderLayerListUI() {
  const listEl = document.getElementById("layerList");
  listEl.innerHTML = "";
  let currentGroup = "";

  for (const def of LAYER_DEFINITIONS) {
    const lyr = STATE.layers[def.id];
    if (!lyr) continue;

    if (def.group !== currentGroup) {
      currentGroup = def.group;
      const gHeader = document.createElement("div");
      gHeader.className = "layer-group-header";
      gHeader.textContent = currentGroup;
      listEl.appendChild(gHeader);
    }

    const row = document.createElement("div");
    row.className = `layer-item ${STATE.selectedLayerId === def.id ? "selected" : ""}`;
    row.dataset.id = def.id;

    const visBtn = document.createElement("button");
    visBtn.className = "layer-vis-btn";
    visBtn.textContent = lyr.vis ? "👁️" : "🚫";
    visBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      lyr.vis = !lyr.vis;
      visBtn.textContent = lyr.vis ? "👁️" : "🚫";
      invalidateStack();
      compositeAllLayers();
    });

    const nameSpan = document.createElement("span");
    nameSpan.className = "layer-name";
    nameSpan.textContent = def.name;

    row.appendChild(visBtn);
    row.appendChild(nameSpan);
    row.addEventListener("click", () => selectLayer(def.id));
    listEl.appendChild(row);
  }
}

function selectLayer(layerId) {
  STATE.selectedLayerId = layerId;
  document.querySelectorAll(".layer-item").forEach((el) => {
    el.classList.toggle("selected", el.dataset.id === layerId);
  });

  // Auto-select appropriate brush tip when clicking Eye Reflections vs Watercolor layers
  const tipSelect = document.getElementById("manualBrushTip");
  if (layerId === "custom_detail") {
    tipSelect.value = "round_hard";
    document.getElementById("manualBrushSize").value = 18;
    document.getElementById("manualSizeVal").textContent = "18";
  }

  const lyr = STATE.layers[layerId];
  if (!lyr) return;

  document.getElementById("activeLayerTitle").textContent = lyr.def.name;
  renderSlidersForLayer(lyr);
}

function renderSlidersForLayer(lyr) {
  const container = document.getElementById("layerSlidersContainer");
  container.innerHTML = "";
  const p = lyr.params;
  const def = lyr.def;

  // 1. Layer Opacity Slider (All layers)
  addSlider(container, "Layer Opacity", Math.round(p.opacity * 100), 0, 100, "%", (val) => {
    p.opacity = val / 100;
    compositeAllLayers();
  });

  // 2. Procedural Brush Controls (Spacing/Frequency, Size, Texture, Jiggle)
  if (["brush_edge", "brush_dots", "brush_ellipse", "brush_ellipse_dark", "mask_blur", "accent"].includes(def.type)) {
    addBrushFileDropdown(container, p.brushFile, (newFile) => {
      p.brushFile = newFile;
      scheduleRerender(def.id);
    });

    addSlider(container, "Stroke Spacing (Lower = More)", p.spacing, 25, 900, "px", (val) => {
      p.spacing = val;
      scheduleRerender(def.id);
    });

    addSlider(container, "Brush Size", p.size, 40, 1600, "px", (val) => {
      p.size = val;
      scheduleRerender(def.id);
    });

    if (p.jiggle !== undefined) {
      addSlider(container, "Angle Jiggle", Math.round(p.jiggle * 100), 0, 100, "%", (val) => {
        p.jiggle = val / 100;
        scheduleRerender(def.id);
      });
    }
  }

  // 3. Subject Color Correction Blend
  if (def.type === "subject") {
    addSlider(container, "Color Correction Strength", p.ccBlend, 0, 100, "%", (val) => {
      p.ccBlend = val;
      compositeAllLayers();
    });
  }

  // 4. Live Tonal Sliders (Brightness, Contrast, Saturation)
  addSlider(container, "Brightness", p.brightness, 30, 200, "%", (val) => {
    p.brightness = val;
    compositeAllLayers();
  });

  addSlider(container, "Contrast", p.contrast, 30, 200, "%", (val) => {
    p.contrast = val;
    compositeAllLayers();
  });

  addSlider(container, "Saturation", p.saturation, 0, 250, "%", (val) => {
    p.saturation = val;
    compositeAllLayers();
  });
}

function addSlider(parent, labelText, value, min, max, unit, onInput) {
  const row = document.createElement("div");
  row.className = "control-row";
  const lbl = document.createElement("label");
  const valSpan = document.createElement("span");
  valSpan.textContent = `${value}${unit}`;
  lbl.innerHTML = `<span>${labelText}</span>`;
  lbl.appendChild(valSpan);

  const input = document.createElement("input");
  input.type = "range";
  input.min = min;
  input.max = max;
  input.value = value;
  input.step = unit === "px" ? 5 : 1;
  input.addEventListener("input", () => {
    valSpan.textContent = `${input.value}${unit}`;
    onInput(Number(input.value));
  });

  row.appendChild(lbl);
  row.appendChild(input);
  parent.appendChild(row);
}

function addBrushFileDropdown(parent, currentFile, onChange) {
  const row = document.createElement("div");
  row.className = "control-row";
  const lbl = document.createElement("label");
  lbl.textContent = "Layer Brush Texture";

  const sel = document.createElement("select");
  sel.className = "select-box full-width";
  const options = [
    ["bgbrush.png", "🎨 Watercolor Wash (bgbrush)"],
    ["brush2.png", "🖌️ Color Brush (brush2)"],
    ["sharpbrush.png", "📐 Sharp Brush (sharpbrush)"],
    ["whitebrush.png", "⚪ Soft Frame (whitebrush)"],
    ["square.png", "⬛ Square Texture (square)"],
    ["whitedots.png", "💦 White Splatter (whitedots)"],
    ["darkdots.png", "⚫ Dark Splatter (darkdots)"],
  ];
  for (const [val, text] of options) {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = text;
    if (val === currentFile) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => onChange(sel.value));
  row.appendChild(lbl);
  row.appendChild(sel);
  parent.appendChild(row);
}

function resetLayerToOriginal(layerId) {
  invalidateStack();
  const lyr = STATE.layers[layerId];
  if (!lyr || !STATE.payload) return;
  const [w, h] = STATE.payload.canvas_size;
  const cfg = STATE.payload.preset_settings;

  lyr.params = buildInitialLayerParams(lyr.def, cfg);
  lyr.paintCanvas.getContext("2d").clearRect(0, 0, w, h);
  lyr.eraseCanvas.getContext("2d").clearRect(0, 0, w, h);

  const bCtx = lyr.baseCanvas.getContext("2d");
  bCtx.clearRect(0, 0, w, h);
  if (layerId === "mask_blur" && STATE.serverImages.blur_bg) {
    bCtx.drawImage(STATE.serverImages.blur_bg, 0, 0);
    bCtx.globalCompositeOperation = "destination-in";
    bCtx.drawImage(STATE.serverImages.mask_brush, 0, 0);
    bCtx.globalCompositeOperation = "source-over";
  } else if (layerId === "subject") {
    bCtx.drawImage(STATE.serverImages.subject_cc, 0, 0);
  } else if (layerId === "oil_paint") {
    bCtx.drawImage(STATE.serverImages.oil_paint_full, 0, 0);
  } else if (lyr.origImage) {
    bCtx.drawImage(lyr.origImage, 0, 0);
  }

  renderSlidersForLayer(lyr);
  compositeAllLayers();
}

// ============================================================================
// 10. INTERACTIVE BRUSH, ERASER, EYEDROPPER & CUTOUT MASK TOOLS
// ============================================================================
function setActiveTool(tool) {
  STATE.activeTool = tool;
  document.querySelectorAll(".tool-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tool === tool);
  });
  document.getElementById("maskEditControls").classList.toggle("hidden", tool !== "mask");
  updateCursorPreview();
  compositeAllLayers();
}

function setupCanvasInteractions() {
  let isPickingColor = false;
  let vectorStroke = null;

  document.getElementById("eyedropperBtn").addEventListener("click", () => {
    isPickingColor = true;
    viewport.style.cursor = "crosshair";
  });

  viewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    setZoom(STATE.zoom * factor, e.clientX, e.clientY);
  }, { passive: false });

  viewport.addEventListener("pointerdown", (e) => {
    if (!STATE.payload) return;

    if (isPickingColor) {
      pickColorFromCanvas(e);
      isPickingColor = false;
      viewport.style.cursor = "default";
      return;
    }

    STATE.isDragging = true;
    STATE.lastMouse = { x: e.clientX, y: e.clientY };

    if (STATE.activeTool === "brush" || STATE.activeTool === "eraser" || STATE.activeTool === "mask") {
      vectorStroke = {
        space: "canvas",
        tool: STATE.activeTool === "mask"
          ? (STATE.maskModePaintKeep ? "mask-keep" : "mask-remove")
          : STATE.activeTool,
        tip: document.getElementById("manualBrushTip").value,
        size: Number(document.getElementById("manualBrushSize").value),
        color: document.getElementById("manualBrushColor").value,
        opacity: Number(document.getElementById("manualBrushOpacity").value) / 100,
        pressure: e.pressure || 1,
        layerId: STATE.selectedLayerId,
        points: [],
      };
      applyManualStrokeAtMouse(e);
    }
  });

  window.addEventListener("pointermove", (e) => {
    updateCursorPosition(e);
    if (!STATE.isDragging || !STATE.payload) return;

    if (STATE.activeTool === "pan") {
      STATE.panX += e.clientX - STATE.lastMouse.x;
      STATE.panY += e.clientY - STATE.lastMouse.y;
      STATE.lastMouse = { x: e.clientX, y: e.clientY };
      updateCanvasTransform();
    } else {
      applyManualStrokeAtMouse(e);
      STATE.lastMouse = { x: e.clientX, y: e.clientY };
    }
  });

  window.addEventListener("pointerup", () => {
    STATE.isDragging = false;
    if (vectorStroke?.points.length) {
      window.dispatchEvent(new CustomEvent("watercolor:vector-edit", { detail: vectorStroke }));
    }
    vectorStroke = null;
  });
}

function screenToCanvasCoords(clientX, clientY) {
  const rect = displayCanvas.getBoundingClientRect();
  const x = (clientX - rect.left) * (displayCanvas.width / rect.width);
  const y = (clientY - rect.top) * (displayCanvas.height / rect.height);
  return { x, y };
}

function applyManualStrokeAtMouse(e) {
  const { x, y } = screenToCanvasCoords(e.clientX, e.clientY);
  const size = Number(document.getElementById("manualBrushSize").value);
  const opacity = Number(document.getElementById("manualBrushOpacity").value) / 100;
  const colorHex = document.getElementById("manualBrushColor").value;
  const tip = document.getElementById("manualBrushTip").value;
  if (vectorStroke) {
    vectorStroke.points.push({
      x: x / displayCanvas.width,
      y: y / displayCanvas.height,
      pressure: e.pressure || vectorStroke.pressure,
    });
    vectorStroke.pressure = e.pressure || vectorStroke.pressure;
  }

  // Cutout Mask Editing Mode
  if (STATE.activeTool === "mask" && STATE.maskCanvas) {
    const mx = (x / displayCanvas.width) * STATE.maskCanvas.width;
    const my = (y / displayCanvas.height) * STATE.maskCanvas.height;
    const mCtx = STATE.maskCanvas.getContext("2d");
    mCtx.save();
    mCtx.fillStyle = STATE.maskModePaintKeep ? "#ffffff" : "#000000";
    mCtx.beginPath();
    mCtx.arc(mx, my, size / 2, 0, Math.PI * 2);
    mCtx.fill();
    mCtx.restore();
    renderMaskEditorView();
    return;
  }

  const lyr = STATE.layers[STATE.selectedLayerId];
  if (!lyr || !lyr.vis) return;

  // Target either the selected layer's Paint canvas or Erase canvas
  const targetCanvas = STATE.activeTool === "eraser" ? lyr.eraseCanvas : lyr.paintCanvas;
  const ctx = targetCanvas.getContext("2d");

  if (STATE.activeTool === "eraser") {
    ctx.save();
    ctx.globalAlpha = opacity;
    const grad = ctx.createRadialGradient(x, y, size * 0.15, x, y, size / 2);
    grad.addColorStop(0, "rgba(0,0,0,1)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, size / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  } else {
    // Painting with Standard Detail Brush (Eye Reflections) OR Watercolor Stamp
    const rgb = hexToRgb(colorHex);
    if (tip === "round_hard") {
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = colorHex;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (tip === "round_soft") {
      ctx.save();
      ctx.globalAlpha = opacity;
      const grad = ctx.createRadialGradient(x, y, 0, x, y, size / 2);
      grad.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},1)`);
      grad.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (tip === "specular_star") {
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = colorHex;
      ctx.beginPath();
      ctx.arc(x, y, size * 0.28, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = colorHex;
      ctx.lineWidth = Math.max(1.5, size * 0.08);
      ctx.beginPath();
      ctx.moveTo(x - size * 0.6, y); ctx.lineTo(x + size * 0.6, y);
      ctx.moveTo(x, y - size * 0.6); ctx.lineTo(x, y + size * 0.6);
      ctx.stroke();
      ctx.restore();
    } else {
      // Stamp actual watercolor brush texture
      const bImg = STATE.brushImages[tip] || Object.values(STATE.brushImages)[0];
      const randAngle = Math.random() * 360;
      colorizeAndDrawBrush(ctx, bImg, x, y, size, randAngle, rgb, opacity);
    }
  }

  compositeAllLayers();
}

function renderMaskEditorView() {
  const [w, h] = STATE.payload.canvas_size;
  const [offX, offY] = STATE.payload.offset;
  const [subW, subH] = STATE.payload.subject_size;

  displayCtx.clearRect(0, 0, w, h);
  displayCtx.fillStyle = "#f8fafc";
  displayCtx.fillRect(0, 0, w, h);

  // Draw original photo at subject position
  displayCtx.drawImage(STATE.originalImg, offX, offY, subW, subH);

  // Draw translucent red tint over removed areas (where mask is black)
  const overlay = createCanvas(subW, subH);
  const oCtx = overlay.getContext("2d");
  oCtx.fillStyle = "rgba(239, 68, 68, 0.55)";
  oCtx.fillRect(0, 0, subW, subH);
  oCtx.globalCompositeOperation = "destination-out";
  oCtx.drawImage(STATE.maskCanvas, 0, 0, subW, subH);

  displayCtx.drawImage(overlay, offX, offY);
}

function pickColorFromCanvas(e) {
  const { x, y } = screenToCanvasCoords(e.clientX, e.clientY);
  const pixel = displayCtx.getImageData(Math.floor(x), Math.floor(y), 1, 1).data;
  document.getElementById("manualBrushColor").value = rgbToHex(pixel[0], pixel[1], pixel[2]);
}

// ============================================================================
// 11. ZOOM, PAN & HIGH-RES EXPORT
// ============================================================================
function fitCanvasToViewport() {
  if (!STATE.payload) return;
  const [w, h] = STATE.payload.canvas_size;
  const vw = viewport.clientWidth - 60;
  const vh = viewport.clientHeight - 60;
  STATE.zoom = Math.min(vw / w, vh / h);
  STATE.panX = Math.floor((viewport.clientWidth - w * STATE.zoom) / 2);
  STATE.panY = Math.floor((viewport.clientHeight - h * STATE.zoom) / 2);
  updateCanvasTransform();
}

function setZoom(newZoom, anchorX, anchorY) {
  const clamped = Math.max(0.15, Math.min(3.5, newZoom));
  if (anchorX !== undefined && anchorY !== undefined) {
    const vRect = viewport.getBoundingClientRect();
    const mx = anchorX - vRect.left;
    const my = anchorY - vRect.top;
    STATE.panX = mx - ((mx - STATE.panX) / STATE.zoom) * clamped;
    STATE.panY = my - ((my - STATE.panY) / STATE.zoom) * clamped;
  }
  STATE.zoom = clamped;
  updateCanvasTransform();
}

function updateCanvasTransform() {
  canvasContainer.style.transform = `translate(${STATE.panX}px, ${STATE.panY}px) scale(${STATE.zoom})`;
  document.getElementById("zoomLabel").textContent = `${Math.round(STATE.zoom * 100)}%`;
  updateCursorPreview();
}

function updateCursorPosition(e) {
  if (STATE.activeTool === "pan" || !STATE.payload) {
    brushCursor.classList.add("hidden");
    return;
  }
  const vRect = viewport.getBoundingClientRect();
  if (e.clientX < vRect.left || e.clientX > vRect.right || e.clientY < vRect.top || e.clientY > vRect.bottom) {
    brushCursor.classList.add("hidden");
    return;
  }
  brushCursor.classList.remove("hidden");
  brushCursor.style.left = `${e.clientX}px`;
  brushCursor.style.top = `${e.clientY}px`;
  updateCursorPreview();
}

function updateCursorPreview() {
  const size = Number(document.getElementById("manualBrushSize").value) * STATE.zoom;
  brushCursor.style.width = `${Math.max(4, size)}px`;
  brushCursor.style.height = `${Math.max(4, size)}px`;
}

function exportImage(fmt = "jpeg") {
  invalidateStack();
  if (!STATE.payload) return;
  if (STATE.activeTool === "mask") setActiveTool("pan");
  compositeAllLayers();

  const mime = fmt === "png" ? "image/png" : "image/jpeg";
  const ext = fmt === "png" ? "png" : "jpg";
  const dataUrl = displayCanvas.toDataURL(mime, 0.96);

  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = `watercolor_${STATE.payload.preset_key}_${Date.now()}.${ext}`;
  a.click();
}

function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0")).join("");
}

function hexToRgb(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}