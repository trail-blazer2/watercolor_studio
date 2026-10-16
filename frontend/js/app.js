import { WatercolorDocument, ENABLE_SIZE_PICKER, OUTPUT_SIZES } from "./document.js";
import { EditHistory } from "./state.js";
import { maskDiameter, replayMask, replayParameters, subjectPoint } from "./editor.js";
import { segmentPhoto } from "./pipeline/segment.js";
import { createPainting } from "./pipeline/paint.js";
import { exportJPG } from "./pipeline/export.js";

const $ = (id) => document.getElementById(id);
const canvas = $("painting");
const workspace = $("workspace");
const studio = {
  doc: null, history: null, painting: null, baseMask: null, selected: "bg_brush",
  tool: "pan", zoom: 1, panX: 0, panY: 0, pointers: new Map(), stroke: null,
  frame: 0, rendering: false, needsRender: false, busy: false, job: 0,
  lastTap: 0, blobURL: null, blob: null, start: 0, timer: null,
  gestureStart: null, gestureMoved: false,
};

function notice(message) { $("notice").textContent = message; }
function progress(value = {}) {
  if (typeof value === "string") value = { stage: value };
  const stages = { download: "Loading model", model: "Loading model", segment: "Cutting out",
    brushes: "Painting", painting: "Painting", export: "Exporting JPG" };
  $("stage").textContent = stages[value.stage] || value.stage || "Painting";
  $("progress").value = value.progress ?? 0;
  $("stage-message").textContent = value.message || "";
}
function setBusy(busy, stage = "Painting") {
  studio.busy = busy;
  $("busy").hidden = !busy;
  for (const id of ["export-top", "export-mobile"]) $(id).disabled = busy || !studio.painting;
  for (const id of ["photo", "camera"]) $(id).disabled = busy;
  $("tools").inert = $("inspector").inert = busy;
  clearInterval(studio.timer);
  if (busy) {
    studio.start = performance.now();
    progress({ stage });
    studio.timer = setInterval(() => {
      $("elapsed").textContent = `${((performance.now() - studio.start) / 1000).toFixed(1)} s`;
    }, 100);
  }
  historyButtons();
}
function historyButtons() {
  $("undo").disabled = studio.busy || !studio.history?.undoStack.length;
  $("redo").disabled = studio.busy || !studio.history?.redoStack.length;
}
function transform() {
  canvas.style.transform = `translate(calc(-50% + ${studio.panX}px),calc(-50% + ${studio.panY}px)) scale(${studio.zoom})`;
}
function fit() {
  if (!studio.doc) return;
  studio.zoom = Math.min((workspace.clientWidth - 36) / canvas.width, (workspace.clientHeight - 60) / canvas.height);
  studio.panX = studio.panY = 0;
  transform();
}
function scheduleRender() {
  studio.needsRender = true;
  if (!studio.frame) studio.frame = requestAnimationFrame(render);
}
async function render() {
  studio.frame = 0;
  if (!studio.painting || studio.rendering) return;
  studio.rendering = true;
  studio.needsRender = false;
  try {
    await studio.painting.render({
      canvas, width: canvas.width, height: canvas.height,
      fullWidth: canvas.width, fullHeight: canvas.height,
    });
  } catch (error) { notice(`Rendering failed: ${error.message}`); }
  finally {
    studio.rendering = false;
    if (studio.needsRender) scheduleRender();
  }
}
async function rebuild() {
  const old = studio.painting;
  const painting = await createPainting(studio.doc, { onProgress: progress });
  studio.painting = painting;
  old?.dispose();
  window.watercolorPainting = painting;
  window.watercolorDocument = studio.doc;
  renderLayers();
  renderAdjustments();
  if (studio.frame) cancelAnimationFrame(studio.frame);
  studio.frame = 0;
  await render();
}
async function decodePhoto(file) {
  let image;
  try { image = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch {
    const url = URL.createObjectURL(file);
    try {
      image = new Image();
      image.src = url;
      await image.decode();
    } finally { URL.revokeObjectURL(url); }
  }
  const scale = Math.min(1, 2048 / Math.max(image.width, image.height));
  const source = document.createElement("canvas");
  source.width = Math.max(1, Math.round(image.width * scale));
  source.height = Math.max(1, Math.round(image.height * scale));
  source.getContext("2d").drawImage(image, 0, 0, source.width, source.height);
  image.close?.();
  return source;
}
async function upload(file) {
  if (!file || studio.busy) return;
  closeSheets();
  const job = ++studio.job;
  setBusy(true, "Loading model");
  notice("");
  try {
    const photo = await decodePhoto(file);
    const doc = new WatercolorDocument({ photo, preset: "animals" });
    studio.doc = doc;
    studio.history = new EditHistory(doc);
    studio.painting?.dispose();
    studio.painting = null;
    studio.baseMask = null;
    canvas.width = doc.outputWidth;
    canvas.height = doc.outputHeight;
    canvas.hidden = true;
    studio.baseMask = await segmentPhoto(photo, {
      trim: Math.round(doc.px(6)),
      onProgress: (value) => {
        if (job !== studio.job) return;
        progress(value);
        $("model-status").textContent = value.message || value.stage || "Loading AI model";
      },
      onUpgrade: async (mask) => {
        if (job !== studio.job) return;
        // Wait until the initial painting/export has finished before replacing its mask.
        while (studio.busy || studio.rendering || studio.pointers.size > 0) {
          await new Promise((resolve) => setTimeout(resolve, 100));
          if (job !== studio.job) return;
        }
        setBusy(true, "Cutting out");
        try {
          studio.baseMask = mask;
          await updateMask();
          $("model-status").textContent = "Full u2net model ready and cached. Cutout upgraded.";
          notice("Full-quality cutout ready. Your edits have been kept.");
        } catch (error) { notice(`Cutout upgrade: ${error.message}`); }
        finally { setBusy(false); historyButtons(); }
      },
    });
    doc.mask = studio.baseMask;
    progress({ stage: "Painting", progress: 0 });
    await rebuild();
    $("empty").hidden = true;
    canvas.hidden = false;
    fit();
    progress({ stage: "Done", progress: 1 });
    notice(`Done in ${((performance.now() - studio.start) / 1000).toFixed(1)} s · Painted on your device`);
  } catch (error) {
    notice(`Could not make the painting: ${error.message}. Please try again with internet connected.`);
  } finally {
    setBusy(false);
    historyButtons();
    $("photo").value = $("camera").value = "";
  }
}
function layerParams(layer) {
  return { ...layer.params, opacity: layer.opacity, visible: layer.visible, ...studio.doc.layerParams[layer.id] };
}
function renderLayers() {
  const list = $("layers");
  list.replaceChildren();
  let group;
  for (const layer of [...studio.painting.layers].reverse()) {
    if (group !== layer.group) {
      group = layer.group;
      const heading = document.createElement("div");
      heading.className = "layer-group";
      heading.textContent = group;
      list.append(heading);
    }
    const row = document.createElement("div");
    row.className = `layer-row${layer.id === studio.selected ? " active" : ""}`;
    const label = document.createElement("label");
    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = layerParams(layer).visible !== false;
    visible.setAttribute("aria-label", `Show ${layer.name}`);
    visible.addEventListener("change", () => {
      commitParameter(layer.id, "visible", visible.checked);
      scheduleRender();
    });
    label.append(visible);
    const button = document.createElement("button");
    button.textContent = layer.name + (layer.approx && !layer.name.includes("approx") ? " (approx)" : "");
    button.addEventListener("click", () => {
      studio.selected = layer.id;
      renderLayers();
      renderAdjustments();
      if (innerWidth < 820) openSheet("adjust");
    });
    row.append(label, button);
    list.append(row);
  }
}
function commitParameter(layerId, key, value) {
  studio.history.addStroke({ type: "parameter", layerId, key, value });
  (studio.doc.layerParams[layerId] ??= {})[key] = value;
  studio.painting.rerenderLayer?.(layerId);
  historyButtons();
}
function slider(layer, key, name, min, max, step, fallback = 1) {
  const label = document.createElement("label");
  const output = document.createElement("output");
  const input = document.createElement("input");
  input.type = "range";
  input.min = min; input.max = max; input.step = step;
  input.value = layerParams(layer)[key] ?? fallback;
  output.value = input.value;
  label.append(document.createTextNode(name), output, input);
  input.addEventListener("input", () => {
    output.value = input.value;
    (studio.doc.layerParams[layer.id] ??= {})[key] = Number(input.value);
    studio.painting.rerenderLayer?.(layer.id);
    scheduleRender();
  });
  input.addEventListener("change", () => commitParameter(layer.id, key, Number(input.value)));
  return label;
}
function renderAdjustments() {
  const layer = studio.painting.layers.find((item) => item.id === studio.selected) || studio.painting.layers[0];
  studio.selected = layer.id;
  $("selected-name").textContent = layer.name;
  const panel = $("adjustments");
  panel.replaceChildren();
  const controls = [
    ["opacity", "Opacity", 0, 1, 0.01, 1],
    ["brightness", "Brightness", 0, 2, 0.01, 1],
    ["contrast", "Contrast", 0, 2, 0.01, 1],
    ["saturation", "Saturation", 0, 2, 0.01, 1],
  ];
  if (!["subject", "oil_paint", "bg_paper", "custom_detail"].includes(layer.id)) {
    controls.push(["spacing", "Spacing", 5, 1000, 5, 100], ["size", "Brush size", 20, 1600, 10, 500], ["jiggle", "Direction jiggle", 0, 1, 0.01, 0.8]);
    const label = document.createElement("label");
    label.textContent = "Brush texture";
    const select = document.createElement("select");
    for (const name of ["bgbrush.png", "brush2.png", "sharpbrush.png", "square.png", "whitebrush.png", "whitedots.png", "darkdots.png", "colordrop.png"]) {
      const option = document.createElement("option");
      option.value = name; option.textContent = name.replace(".png", "");
      select.append(option);
    }
    select.value = layerParams(layer).brush || "bgbrush.png";
    select.addEventListener("change", () => { commitParameter(layer.id, "brush", select.value); scheduleRender(); });
    label.append(select);
    panel.append(label);
  }
  if (layer.id === "oil_paint") controls.push(
    ["flowLength", "Flow length", 0, 30, 1, 8], ["flowThickness", "Flow thickness", 0, 10, 0.5, 2],
    ["detail", "Detail", 0, 2, 0.05, 1], ["sharpening", "Sharpening", 0, 3, 0.05, 1]);
  if (layer.id === "subject") controls.push(
    ["ccStrength", "Color correction strength", 0, 1, 0.01, 1],
    ["shadowLift", "Lift shadows", 0, 40, 1, 7], ["highlightReduce", "Soften highlights", 0, 40, 1, 12],
    ["vibrance", "Vibrance", 0, 2, 0.01, 1.08], ["warmth", "Warm filter", 0, 0.3, 0.01, 0.05],
    ["warmBoost", "Warm boost", 0, 4, 0.1, 2.2]);
  for (const args of controls) panel.append(slider(layer, ...args));
}
async function updateMask() {
  if (!studio.baseMask || !studio.painting) return;
  studio.doc.mask = replayMask(studio.baseMask, studio.doc.edits,
    studio.painting.imageBounds || studio.painting.subjectBounds);
  await rebuild();
}
async function undo(redo = false) {
  if (studio.busy || !studio.history) return;
  const edit = redo ? studio.history.redo() : studio.history.undo();
  if (!edit) return;
  replayParameters(studio.doc);
  if (edit.tool?.startsWith("mask-")) {
    setBusy(true, "Painting");
    try { await updateMask(); } catch (error) { notice(error.message); }
    finally { setBusy(false); }
  } else {
    studio.painting.rerenderLayer?.(edit.layerId);
    renderLayers();
    renderAdjustments();
    scheduleRender();
  }
  historyButtons();
}
function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height,
    pressure: event.pointerType === "pen" ? Math.max(0.05, event.pressure) : 1 };
}
function pointerStart(event) {
  if (!studio.painting || studio.busy || event.button > 0) return;
  event.preventDefault();
  workspace.setPointerCapture(event.pointerId);
  studio.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (studio.pointers.size > 1) {
    studio.stroke = null;
    studio.gestureMoved = true;
    return;
  }
  studio.gestureStart = { x: event.clientX, y: event.clientY };
  studio.gestureMoved = false;
  const p = canvasPoint(event);
  if (studio.tool === "eyedropper") {
    if (p.x >= 0 && p.x < 1 && p.y >= 0 && p.y < 1) {
      const pixel = canvas.getContext("2d").getImageData(Math.floor(p.x * canvas.width), Math.floor(p.y * canvas.height), 1, 1).data;
      $("color").value = `#${[...pixel].slice(0, 3).map((v) => v.toString(16).padStart(2, "0")).join("")}`;
      notice("Color picked. Choose a brush to paint.");
    }
    return;
  }
  if (studio.tool === "pan") return;
  const mask = studio.tool.startsWith("mask-");
  const mapped = mask ? subjectPoint(p, studio.painting.imageBounds || studio.painting.subjectBounds) : p;
  if (mapped.x < 0 || mapped.y < 0 || mapped.x > 1 || mapped.y > 1) return;
  studio.stroke = {
    points: [mapped], space: mask ? "subject" : "canvas", tool: studio.tool === "detail" ? "paint" : studio.tool,
    tip: studio.tool === "paint" ? "watercolor" : $("tip").value,
    size: studio.doc.px(Number($("brush-size").value)),
    color: $("color").value, opacity: Number($("brush-opacity").value), pressure: p.pressure,
    layerId: studio.tool === "detail" ? "custom_detail" : studio.selected,
  };
}
function pointerMove(event) {
  const previous = studio.pointers.get(event.pointerId);
  if (!previous) return;
  const current = { x: event.clientX, y: event.clientY };
  if (studio.gestureStart && Math.hypot(current.x - studio.gestureStart.x,
    current.y - studio.gestureStart.y) > 8) studio.gestureMoved = true;
  if (studio.pointers.size >= 2) {
    const other = [...studio.pointers.entries()].find(([id]) => id !== event.pointerId)[1];
    const oldDistance = Math.hypot(previous.x - other.x, previous.y - other.y);
    const newDistance = Math.hypot(current.x - other.x, current.y - other.y);
    const rect = workspace.getBoundingClientRect();
    const center = { x: (previous.x + other.x) / 2 - rect.left - rect.width / 2,
      y: (previous.y + other.y) / 2 - rect.top - rect.height / 2 };
    const zoom = Math.max(0.05, Math.min(5, studio.zoom * newDistance / Math.max(1, oldDistance)));
    const ratio = zoom / studio.zoom;
    studio.panX = center.x - (center.x - studio.panX) * ratio + (current.x - previous.x) / 2;
    studio.panY = center.y - (center.y - studio.panY) * ratio + (current.y - previous.y) / 2;
    studio.zoom = zoom;
    transform();
  } else if (studio.tool === "pan") {
    studio.panX += current.x - previous.x;
    studio.panY += current.y - previous.y;
    transform();
  } else if (studio.stroke) {
    const point = canvasPoint(event);
    const p = studio.stroke.space === "subject"
      ? subjectPoint(point, studio.painting.imageBounds || studio.painting.subjectBounds) : point;
    studio.stroke.points.push({ ...p, x: Math.max(0, Math.min(1, p.x)), y: Math.max(0, Math.min(1, p.y)) });
    // Immediate visual feedback; the finished stroke is replayed by the layer renderer.
    if (studio.stroke.space === "canvas" && studio.stroke.tool === "paint") {
      const ctx = canvas.getContext("2d");
      ctx.save();
      ctx.globalAlpha = studio.stroke.opacity * 0.25;
      ctx.fillStyle = studio.stroke.color;
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, studio.stroke.size * p.pressure / 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();
    }
  }
  studio.pointers.set(event.pointerId, current);
}
async function pointerEnd(event) {
  studio.pointers.delete(event.pointerId);
  if (event.type === "pointercancel") { studio.stroke = null; scheduleRender(); return; }
  if (studio.stroke && studio.pointers.size === 0) {
    const record = studio.doc.addStroke(studio.stroke);
    if (record.space === "subject") {
      record.sourceSize = maskDiameter(record, studio.baseMask.width, studio.baseMask.height,
        studio.painting.imageBounds || studio.painting.subjectBounds) /
        Math.max(studio.baseMask.width, studio.baseMask.height);
    }
    studio.history.track(record);
    studio.stroke = null;
    if (record.space === "subject") {
      setBusy(true, "Painting");
      try { await updateMask(); } catch (error) { notice(error.message); }
      finally { setBusy(false); }
    } else {
      studio.painting.rerenderLayer?.(record.layerId);
      scheduleRender();
    }
    historyButtons();
  } else if (studio.tool === "pan" && studio.pointers.size === 0 && !studio.gestureMoved) {
    const now = performance.now();
    if (now - studio.lastTap < 300) fit();
    studio.lastTap = now;
  }
}
function closeSheets() {
  for (const panel of document.querySelectorAll(".sheet")) panel.classList.remove("open");
  $("backdrop").hidden = true;
}
function openSheet(name) {
  closeSheets();
  const panel = name === "photo" || name === "tools" ? $("tools") : $("inspector");
  panel.classList.toggle("show-layers", name === "layers");
  panel.classList.toggle("show-adjust", name === "adjust");
  panel.classList.add("open");
  $("backdrop").hidden = false;
}
async function exportPainting() {
  if (studio.busy || !studio.painting) return;
  closeSheets();
  setBusy(true, "Exporting JPG");
  historyButtons();
  try {
    const painting = await createPainting(studio.doc, { assetsQuality: "hd", onProgress: progress });
    let blob;
    try {
      blob = await exportJPG({
        document: studio.doc,
        onProgress: (value) => progress(typeof value === "number" ? { stage: "Exporting JPG", progress: value } : value),
        renderTile: async ({ width, height, x, y, tileWidth, tileHeight, margin = 0 }) => {
          const tile = document.createElement("canvas");
          await painting.render({
            canvas: tile, width: tileWidth + margin * 2, height: tileHeight + margin * 2,
            x: x - margin, y: y - margin, fullWidth: width, fullHeight: height,
          });
          return tile;
        },
      });
    } finally { painting.dispose(); }
    if (studio.blobURL) URL.revokeObjectURL(studio.blobURL);
    studio.blob = blob;
    studio.blobURL = URL.createObjectURL(blob);
    $("download").href = studio.blobURL;
    const file = new File([blob], "watercolor.jpg", { type: "image/jpeg" });
    $("share").hidden = !navigator.canShare?.({ files: [file] });
    $("export-dialog").showModal();
  } catch (error) { notice(`Export failed: ${error.message}`); }
  finally { setBusy(false); historyButtons(); }
}

for (const id of ["photo", "camera"]) $(id).addEventListener("change", (event) => upload(event.target.files[0]));
for (const button of document.querySelectorAll("[data-tool]")) button.addEventListener("click", () => {
  studio.tool = button.dataset.tool;
  for (const item of document.querySelectorAll("[data-tool]")) item.setAttribute("aria-pressed", String(item === button));
  if (studio.tool === "detail" && studio.painting) {
    studio.selected = "custom_detail"; renderLayers(); renderAdjustments();
  }
  notice(studio.tool.startsWith("mask-") ? "Paint on the subject to keep or remove part of the cutout." : "");
});
for (const button of document.querySelectorAll("[data-sheet]")) button.addEventListener("click", () => openSheet(button.dataset.sheet));
for (const button of document.querySelectorAll(".close-sheet")) button.addEventListener("click", closeSheets);
$("backdrop").addEventListener("click", closeSheets);
$("fit").addEventListener("click", fit);
$("fit-bottom").addEventListener("click", fit);
$("undo").addEventListener("click", () => undo());
$("redo").addEventListener("click", () => undo(true));
$("brush-size").addEventListener("input", (event) => { $("brush-size-label").value = event.target.value; });
for (const id of ["export-top", "export-mobile"]) $(id).addEventListener("click", exportPainting);
$("close-export").addEventListener("click", () => $("export-dialog").close());
$("share").addEventListener("click", async () => {
  try { await navigator.share({ files: [new File([studio.blob], "watercolor.jpg", { type: "image/jpeg" })] }); }
  catch (error) { if (error.name !== "AbortError") notice("Sharing unavailable. Use Download JPG instead."); }
});
workspace.addEventListener("pointerdown", pointerStart);
workspace.addEventListener("pointermove", pointerMove);
workspace.addEventListener("pointerup", pointerEnd);
workspace.addEventListener("pointercancel", pointerEnd);
workspace.addEventListener("wheel", (event) => {
  if (!studio.painting) return;
  event.preventDefault();
  const ratio = Math.exp(-event.deltaY * 0.001);
  studio.zoom = Math.max(0.05, Math.min(5, studio.zoom * ratio));
  transform();
}, { passive: false });
window.addEventListener("keydown", (event) => {
  const typing = event.target.isContentEditable || event.target.tagName === "TEXTAREA" ||
    (event.target.tagName === "INPUT" && ["text", "email", "password", "search", "url", "number"].includes(event.target.type));
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" &&
      !typing) {
    event.preventDefault(); undo(event.shiftKey);
  }
  if (event.key === "Escape") closeSheets();
});
window.addEventListener("resize", fit);
$("size-section").hidden = !ENABLE_SIZE_PICKER;
for (const size of OUTPUT_SIZES.filter((size) => size.widthPx)) {
  const option = document.createElement("option");
  option.value = size.id; option.textContent = size.label;
  $("size").append(option);
}
$("size").addEventListener("change", async () => {
  if (!studio.doc || studio.busy) return;
  const size = OUTPUT_SIZES.find((size) => size.id === $("size").value);
  studio.doc.resize(size.widthPx, size.heightPx, size.id);
  // Preview stays within the iOS canvas budget; export uses the document's full dimensions.
  const scale = Math.min(1, 1782 / Math.max(size.widthPx, size.heightPx));
  canvas.width = Math.round(size.widthPx * scale); canvas.height = Math.round(size.heightPx * scale);
  $("dimensions").textContent = `${size.widthPx} × ${size.heightPx} · Animals`;
  await rebuild(); fit();
});
if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {
  $("model-status").textContent = "Offline caching unavailable. Keep this tab online to load models.";
});
