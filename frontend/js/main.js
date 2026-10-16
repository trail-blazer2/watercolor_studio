import { WatercolorDocument } from "./document.js";
import { EditHistory } from "./state.js";

let history = null;

function setCurrentDocument(document) {
  window.watercolorDocument = document;
  history = new EditHistory(document);
  window.dispatchEvent(new CustomEvent("watercolor:document-ready", { detail: document }));
}

window.addEventListener("watercolor:source", (event) => {
  const { photo, preset } = event.detail;
  setCurrentDocument(new WatercolorDocument({ photo, preset }));
});

window.addEventListener("watercolor:generated", (event) => {
  const { payload, mask } = event.detail;
  const document = window.watercolorDocument;
  if (!document) return;
  const [width, height] = payload.canvas_size;
  const [offsetX, offsetY] = payload.offset;
  const [subjectWidth, subjectHeight] = payload.subject_size;
  document.resize(width, height, "reference");
  document.mask = mask;
  document.preset = payload.preset_key || document.preset;
  document.presetParams = { ...payload.preset_settings };
  document.placement = {
    cx: (offsetX + subjectWidth / 2) / width,
    cy: (offsetY + subjectHeight / 2) / height,
    scale: Math.max(subjectWidth / width, subjectHeight / height),
  };
  window.dispatchEvent(new CustomEvent("watercolor:document-ready", { detail: document }));
});

window.addEventListener("watercolor:vector-edit", (event) => {
  if (!history || !window.watercolorDocument) return;
  const record = window.watercolorDocument.addStroke(event.detail);
  history.track(record);
});

window.addEventListener("watercolor:resize", (event) => {
  if (!window.watercolorDocument) return;
  const { width, height, sizeId } = event.detail;
  window.watercolorDocument.resize(width, height, sizeId);
  window.dispatchEvent(new CustomEvent("watercolor:document-ready", {
    detail: window.watercolorDocument,
  }));
});
