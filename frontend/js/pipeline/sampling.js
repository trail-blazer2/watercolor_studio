// Deterministic, input-order Poisson rejection, matching EdgeCache's grid search.
export function poissonSample(points, spacing) {
  if (!Number.isFinite(spacing) || spacing <= 0) throw new RangeError("Spacing must be positive.");
  const cell = spacing / Math.SQRT2, squared = spacing * spacing;
  const grid = new Map(), sampled = [];
  for (const point of points) {
    const { x, y } = point;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
    let close = false;
    for (let dx = -2; dx <= 2 && !close; dx++) {
      for (let dy = -2; dy <= 2 && !close; dy++) {
        const neighbor = grid.get(`${gx + dx},${gy + dy}`);
        close = !!neighbor && (x - neighbor.x) ** 2 + (y - neighbor.y) ** 2 < squared;
      }
    }
    if (!close) {
      sampled.push(point);
      grid.set(`${gx},${gy}`, point);
    }
  }
  return sampled;
}

export function maskBounds(alpha, width, height, threshold = 20) {
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (alpha[y * width + x] > threshold) {
      left = Math.min(left, x); right = Math.max(right, x);
      top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
  }
  return right < left ? null : { x: left, y: top, width: right - left + 1, height: bottom - top + 1 };
}

// Summed moments provide the Python 21x21 foreground-centroid outward normal in O(1).
export function sampleMaskEdges(alpha, width, height, radius = 10) {
  const stride = width + 1, length = stride * (height + 1);
  const count = new Float64Array(length), xs = new Float64Array(length), ys = new Float64Array(length);
  for (let y = 0; y < height; y++) {
    let c = 0, sx = 0, sy = 0;
    for (let x = 0; x < width; x++) {
      const inside = alpha[y * width + x] > 128 ? 1 : 0;
      c += inside; sx += inside * x; sy += inside * y;
      const i = (y + 1) * stride + x + 1;
      count[i] = count[i - stride] + c; xs[i] = xs[i - stride] + sx; ys[i] = ys[i - stride] + sy;
    }
  }
  const sum = (table, x0, y0, x1, y1) =>
    table[y1 * stride + x1] - table[y0 * stride + x1] -
    table[y1 * stride + x0] + table[y0 * stride + x0];
  const edges = [];
  radius = Math.max(1, Math.round(radius));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (alpha[y * width + x] <= 20) continue;
    let edge = false;
    for (let dy = -1; dy <= 1 && !edge; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height ||
        alpha[(y + dy) * width + x + dx] <= 20) { edge = true; break; }
    }
    if (!edge) continue;
    const x0 = Math.max(0, x - radius), x1 = Math.min(width, x + radius + 1);
    const y0 = Math.max(0, y - radius), y1 = Math.min(height, y + radius + 1);
    const n = sum(count, x0, y0, x1, y1);
    const nx = n ? x - sum(xs, x0, y0, x1, y1) / n : 0;
    const ny = n ? y - sum(ys, x0, y0, x1, y1) / n : 0;
    const magnitude = Math.hypot(nx, ny) + 1e-10;
    edges.push({ x, y, nx: nx / magnitude, ny: ny / magnitude });
  }
  return edges;
}

export function ellipsePoints(width, height, spacing, { inset = 0, bottomOnly = false, bottomPercent = 0.25 } = {}) {
  if (!(spacing > 0)) throw new RangeError("Spacing must be positive.");
  const cx = width / 2, cy = height / 2;
  const rx = Math.max(10, cx - inset), ry = Math.max(10, cy - inset);
  const count = Math.max(1, Math.floor(2 * Math.PI * Math.max(rx, ry) / spacing));
  const points = [];
  for (let i = 0; i < count; i++) {
    const angle = 2 * Math.PI * i / count;
    const x = Math.trunc(cx + rx * Math.cos(angle)), y = Math.trunc(cy + ry * Math.sin(angle));
    if (!bottomOnly || y >= height * (1 - bottomPercent)) points.push({ x, y, angle });
  }
  return points;
}
