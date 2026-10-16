export const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
export const mix = (a, b, amount) => a.map((value, i) => value * (1 - amount) + b[i] * amount);

export function brushColor(color, cfg, background = false, average = color, dominant = average) {
  if (background && cfg.BG_BRUSH_MIX_AVERAGE) color = mix(color, average, cfg.BG_BRUSH_MIX_AMOUNT);
  if (background && cfg.BG_BRUSH_MIX_DOMINANT) color = mix(color, dominant, cfg.BG_BRUSH_MIX_DOMINANT_AMOUNT);
  let [r, g, b] = color;
  const amount = cfg.BRUSH_WARMTH;
  if (amount > 0) {
    const strength = amount * 0.3 + Math.max(0, b - r, (b + g - 2 * r) * 0.35) * amount * 0.7;
    r += Math.trunc(strength * 35) / 255;
    g += Math.trunc(strength * 10) / 255;
    b -= Math.trunc(strength * 20) / 255;
  }
  color = [r, g, b].map((v) => clamp(v));
  if (cfg.DARK_BACKGROUND_ENABLED) color = mix(color, [1, 1, 1], cfg.BRUSH_LIGHTEN_AMOUNT);
  if (background) {
    color = color.map((v) => v * cfg.BG_BRUSH_BRIGHTNESS);
    const gray = color.reduce((a, v) => a + v, 0) / 3;
    color = color.map((v) => gray + (v - gray) * cfg.BG_BRUSH_SATURATION);
  }
  return color.map((v) => clamp(v));
}

// Local luminance replaces Python's blurred shadow mask; exposed as an approximate layer.
export function correctPixels(imageData, cfg) {
  const data = imageData.data;
  if (!cfg.COLOR_CORRECTION_ENABLED) return imageData;
  for (let i = 0; i < data.length; i += 4) {
    let color = [data[i], data[i + 1], data[i + 2]];
    const lum = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255;
    const delta = (1 - lum) ** 0.7 * cfg.CC_SHADOW_LIFT -
      clamp((lum ** 0.7 - 0.5) * 2) * cfg.CC_HIGHLIGHT_REDUCE;
    color = color.map((v) => clamp((v + delta) / 255));
    const max = Math.max(...color), min = Math.min(...color), gray = color.reduce((a, v) => a + v, 0) / 3;
    const boost = 1 + (cfg.CC_VIBRANCE - 1) * Math.sqrt(1 - (max ? (max - min) / max : 0));
    color = color.map((v) => clamp(gray + (v - gray) * boost));
    const l = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114;
    color = color.map((v) => clamp(((l + (v - l) * cfg.CC_SATURATION) * cfg.CC_BRIGHTNESS - 0.5) * cfg.CC_CONTRAST + 0.5));
    color = mix(color, [1, 197 / 255, 143 / 255], cfg.CC_WARM_FILTER);
    const warmth = clamp(color[0] - color[2]) * (1 - clamp(color[1] - color[0]));
    color[0] += warmth * (cfg.CC_WARM_BOOST - 1) * 20 / 255;
    color[2] -= warmth * (cfg.CC_WARM_BOOST - 1) * 10 / 255;
    for (let c = 0; c < 3; c++) data[i + c] = clamp(color[c]) * 255;
  }
  return imageData;
}
