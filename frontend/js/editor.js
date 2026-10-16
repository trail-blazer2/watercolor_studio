export function subjectPoint(point, bounds) {
  return { x: (point.x - bounds.x) / bounds.width, y: (point.y - bounds.y) / bounds.height,
    pressure: point.pressure };
}

export function replayParameters(document) {
  document.layerParams = {};
  for (const edit of document.edits) {
    if (edit.type !== "parameter") continue;
    (document.layerParams[edit.layerId] ??= {})[edit.key] = edit.value;
  }
}

export function maskDiameter(edit, width, height, bounds) {
  return edit.sourceSize !== undefined ? edit.sourceSize * Math.max(width, height)
    : edit.size * Math.max(width / bounds.width, height / bounds.height);
}

export function replayMask(base, edits, bounds) {
  const canvas = document.createElement("canvas");
  canvas.width = base.width;
  canvas.height = base.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(base, 0, 0);
  for (const edit of edits) {
    if (edit.space !== "subject" || !edit.tool?.startsWith("mask-")) continue;
    ctx.fillStyle = edit.tool === "mask-add" ? "#fff" : "#000";
    ctx.strokeStyle = ctx.fillStyle;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.globalAlpha = edit.opacity;
    const diameter = maskDiameter(edit, canvas.width, canvas.height, bounds);
    for (let i = 0; i < edit.points.length; i++) {
      const p = edit.points[i], previous = edit.points[Math.max(0, i - 1)];
      const size = diameter * (p.pressure ?? edit.pressure ?? 1);
      ctx.lineWidth = Math.max(0.5, size);
      ctx.beginPath();
      ctx.moveTo(previous.x * canvas.width, previous.y * canvas.height);
      ctx.lineTo(p.x * canvas.width, p.y * canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, size / 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  return canvas;
}
