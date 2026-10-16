"""
🎨 Accent Stroke Placer — Optimized & Simplified
==================================================
Adds accent strokes to watercolor paintings.

Features:
  - adaptive_strength (0-10): how much adaptive stats influence thresholds
  - mid_stroke_saturation / mid_stroke_size_mult / mid_stroke_opacity
  - weight_balance: auto-rebalances stroke placement + promotes underrepresented patches
  - bottom_fade: gradual opacity reduction in bottom 30%
  - CLI generates original (no-strokes) image for comparison
  - 100% deterministic (same seed = same pixels every time)
  - 100% ready for watercolor.py integration
"""

from __future__ import annotations
import os, sys, math, random, zipfile, glob, argparse
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from scipy import ndimage


# ====================================================================
# DETERMINISTIC PASS OFFSETS (replaces hash() which is non-deterministic)
# ====================================================================
_PASS_OFFSETS = {"main": 1000, "dark": 2000, "bright": 3000}


# ====================================================================
# DEFAULT SETTINGS
# ====================================================================
DEFAULTS = {
    "random_seed":             42,
    "adaptive_strength":       7,

    # --- interest map ---
    "dark_threshold":          90,
    "bright_threshold":        200,
    "saturation_threshold":    0.1,
    "edge_contrast_radius":    20,
    "edge_contrast_weight":    0.2,
    "dark_weight":             1.5,
    "bright_weight":           1.2,
    "saturation_weight":       2,
    "interest_blur_sigma":     6,
    "interest_threshold":      0.002,

    # --- candidate selection ---
    "edge_inset_px":           10,
    "min_point_spacing":       50,
    "max_candidates":          180,

    # --- stroke rendering ---
    "brush_file":              "bgbrush.png",
    "stroke_base_size":        300,
    "stroke_size_jitter":      0.5,
    "stroke_opacity_base":     0.6,
    "stroke_opacity_jitter":   0.1,
    "stroke_warmth":           0.8,
    "stroke_saturation_boost": 1.5,
    "stroke_brightness_adj":   0.0,
    "rotation_jiggle":         0.8,
    "interior_scatter_px":     40,

    # --- per-category ---
    "dark_stroke_saturation":  0.9,
    "bright_stroke_saturation": 1.1,
    "mid_stroke_saturation":   1.2,
    "mid_stroke_size_mult":    1.0,
    "mid_stroke_opacity":      None,

    # --- dark-accent pass ---
    "dark_accent_enabled":     True,
    "dark_accent_size_mult":   1.0,
    "dark_accent_opacity":     0.6,
    "dark_accent_darken":      0.3,

    # --- bright-accent pass ---
    "bright_accent_enabled":   True,
    "bright_accent_size_mult": 0.75,
    "bright_accent_opacity":   0.3,
    "bright_accent_lighten":   0.8,

    # --- ray-cast bright spot scanner ---
    "bright_spot_enabled":          True,
    "bright_spot_max_depth":        40,
    "bright_spot_step":             4,
    "bright_spot_sample_radius":    1,
    "bright_spot_score_bonus":      0.9,
    "bright_spot_stroke_size":      320,
    "bright_spot_size_jitter":      0.2,
    "bright_spot_opacity":          0.3,
    "bright_spot_lighten":          0.7,
    "bright_spot_warmth":           0.15,
    "bright_spot_min_spacing":      40,

    # --- patch size filtering ---
    "bright_spot_min_patch_area":   600,
    "bright_spot_max_patch_area":   50000,
    "bright_spot_flood_lum_tolerance": 10,
    "bright_spot_flood_max_radius":  20,

    # --- bright spot area → size scaling ---
    "bright_spot_area_size_min":    0.9,
    "bright_spot_area_size_max":    1.6,
    "bright_spot_area_ref_small":   600,
    "bright_spot_area_ref_large":   5000,

    # --- bright edge promotion ---
    "bright_edge_promotion_enabled": True,
    "bright_edge_global_contrast_min": 30,
    "bright_edge_promotion_radius":  5,
    "bright_edge_min_spacing":       50,
    "bright_edge_score_bonus":       0.85,
    "bright_edge_min_patch_area":    120,

    # --- visual weight balance ---
    "weight_balance_enabled":        True,
    "weight_balance_strength":       5,
    "weight_balance_grid":           20,
    "weight_balance_dark_adj":       0.15,
    "weight_balance_light_adj":      0.12,
    "weight_balance_opacity_adj":    0.9,
    "weight_balance_promote_dark":   True,
    "weight_balance_promote_max":    8,

    # --- bottom fade ---
    "bottom_fade_enabled":           True,
    "bottom_fade_start":             0.70,
    "bottom_fade_min_opacity":       0.0,

    # --- compositing ---
    "layer_order":             "between",
}


# ====================================================================
# SHARED UTILITIES
# ====================================================================

def _lum_from_rgb(rgb_f32: np.ndarray) -> np.ndarray:
    return 0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]


def _lum_from_color(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _sample_area_color(img: np.ndarray, x: int, y: int, radius: int):
    h, w = img.shape[:2]
    x, y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    if patch.shape[2] == 4:
        vis = patch[:, :, 3] > 30
        if not np.any(vis):
            return None
        rgb = patch[:, :, :3][vis]
    else:
        rgb = patch[:, :, :3].reshape(-1, 3)
    if len(rgb) == 0:
        return None
    return tuple(np.mean(rgb, axis=0).astype(int))


def _colorize(brush: Image.Image, colour: tuple) -> Image.Image:
    arr = np.array(brush)
    out = np.empty_like(arr)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = colour[0], colour[1], colour[2]
    out[:, :, 3] = arr[:, :, 3]
    return Image.fromarray(out, "RGBA")


def _apply_opacity(image: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return image
    arr = np.array(image)
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * opacity).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _composite_safe(layer: Image.Image, brush_img: Image.Image, px: int, py: int):
    bw, bh = brush_img.size
    lw, lh = layer.size
    s_x1, s_y1 = max(0, -px), max(0, -py)
    s_x2, s_y2 = min(bw, lw - px), min(bh, lh - py)
    if s_x1 >= s_x2 or s_y1 >= s_y2:
        return
    layer.alpha_composite(brush_img.crop((s_x1, s_y1, s_x2, s_y2)),
                          (max(0, px), max(0, py)))


def _poisson_thin(points: list, spacing: float,
                  key_x="x", key_y="y") -> list:
    spacing_sq = spacing * spacing
    cell = spacing / math.sqrt(2)
    grid = {}
    result = []
    for p in points:
        px, py = p[key_x], p[key_y]
        gx, gy = int(px / cell), int(py / cell)
        too_close = False
        for dx in range(-2, 3):
            if too_close:
                break
            for dy in range(-2, 3):
                nb = grid.get((gx + dx, gy + dy))
                if nb and (px - nb[0]) ** 2 + (py - nb[1]) ** 2 < spacing_sq:
                    too_close = True
                    break
        if not too_close:
            result.append(p)
            grid[(gx, gy)] = (px, py)
    return result


def _load_brush(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        print(f"⚠️  Brush '{path}' not found — generating fallback")
        sz = 120
        img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse([8, 8, sz - 8, sz - 8], fill=(255, 255, 255, 220))
        return img.filter(ImageFilter.GaussianBlur(radius=6))


def _bottom_fade_mult(canvas_y: int, canvas_h: int, cfg: dict) -> float:
    """Return opacity multiplier for bottom fade. 1.0 above fade zone, ramps to min at bottom."""
    if not cfg.get("bottom_fade_enabled", True):
        return 1.0
    start = cfg.get("bottom_fade_start", 0.70)
    min_op = cfg.get("bottom_fade_min_opacity", 0.15)
    frac = canvas_y / max(1, canvas_h)
    if frac <= start:
        return 1.0
    t = (frac - start) / (1.0 - start)
    return 1.0 - t * (1.0 - min_op)


# IMPROVED: Brush resize/rotate cache — avoids re-computing identical transforms
_BRUSH_TRANSFORM_CACHE = {}
_BRUSH_CACHE_MAX = 256


def _get_transformed_brush(brush_img: Image.Image, sz: int, angle: float) -> Image.Image:
    """Cache brush resize+rotate by (size, quantized_angle). Huge speedup for repeated sizes."""
    # Quantize angle to nearest 2° to increase cache hits
    q_angle = round(angle / 2.0) * 2.0
    key = (id(brush_img), sz, q_angle)
    cached = _BRUSH_TRANSFORM_CACHE.get(key)
    if cached is not None:
        return cached
    sized = brush_img.resize((sz, sz), Image.Resampling.BILINEAR)
    rotated = sized.rotate(q_angle, expand=True, resample=Image.Resampling.BILINEAR)
    # Evict oldest if cache is full
    if len(_BRUSH_TRANSFORM_CACHE) >= _BRUSH_CACHE_MAX:
        _BRUSH_TRANSFORM_CACHE.pop(next(iter(_BRUSH_TRANSFORM_CACHE)))
    _BRUSH_TRANSFORM_CACHE[key] = rotated
    return rotated


def precompute_area_colors(subject_rgba: np.ndarray, radius: int) -> np.ndarray:
    """Precomputes area averages ignoring transparent pixels using fast uniform filters."""
    rgb = subject_rgba[:, :, :3].astype(np.float32)
    alpha = (subject_rgba[:, :, 3] > 30).astype(np.float32)
    
    ksz = radius * 2 + 1
    # Sum of alpha in the neighborhood
    alpha_sum = ndimage.uniform_filter(alpha, size=ksz, mode='constant', cval=0.0)
    safe_alpha = np.maximum(alpha_sum, 1e-10)
    
    # Return RGBA so we can use the alpha channel to check if the area was empty
    out = np.zeros((subject_rgba.shape[0], subject_rgba.shape[1], 4), dtype=np.uint8)
    for c in range(3):
        c_sum = ndimage.uniform_filter(rgb[:, :, c] * alpha, size=ksz, mode='constant', cval=0.0)
        out[:, :, c] = (c_sum / safe_alpha).astype(np.uint8)
    out[:, :, 3] = (alpha_sum > 0).astype(np.uint8) * 255
    return out


# ====================================================================
# IMAGE-ADAPTIVE STATISTICS
# ====================================================================

class ImageStats:
    def __init__(self):
        self.dark_thresh = 80.0
        self.bright_thresh = 140.0
        self.lum_median = 128.0
        self.lum_iqr = 100.0
        self.bright_scan_lum_jump = 50.0
        self.bright_scan_min_lum = 200.0
        self.n_samples = 0

    def analyze(self, lum_values: np.ndarray, adaptive_strength: float = 1.0):
        if len(lum_values) == 0:
            return
        self.n_samples = len(lum_values)
        self.lum_median = float(np.median(lum_values))
        p25 = float(np.percentile(lum_values, 25))
        p75 = float(np.percentile(lum_values, 75))
        self.lum_iqr = p75 - p25

        ad = float(np.clip(p25, 20, 120))
        ab = float(np.clip(p75, 100, 240))
        if ad >= ab - 20:
            mid = (ad + ab) / 2
            ad, ab = mid - 10, mid + 10

        aj = max(20.0, self.lum_iqr * 0.5)
        am = min(240.0, max(140.0, self.lum_median + 30))

        s = adaptive_strength
        self.dark_thresh = 80 * (1 - s) + ad * s
        self.bright_thresh = 140 * (1 - s) + ab * s
        self.bright_scan_lum_jump = 50 * (1 - s) + aj * s
        self.bright_scan_min_lum = 200 * (1 - s) + am * s

    def log(self):
        print(f"   📊 Stats ({self.n_samples} samples): "
              f"median={self.lum_median:.0f}, IQR={self.lum_iqr:.0f}, "
              f"dark<{self.dark_thresh:.0f}, bright>{self.bright_thresh:.0f}")


# ====================================================================
# VISUAL WEIGHT BALANCE (with dark-patch promotion)
# ====================================================================

class WeightBalanceMap:
    """
    Analyzes visual "heaviness" in an NxN grid.
    Heavy zones → lighter strokes / lower opacity.
    Light zones → slightly darker strokes AND promotes nearby ignored dark candidates.
    """

    def __init__(self):
        self.enabled = False
        self.strength = 0.5
        self.grid_n = 4
        self.zone_weights = None
        self.zone_bias = None
        self.canvas_size = (0, 0)
        self.dark_adj = 0.15
        self.light_adj = 0.12
        self.opacity_adj = 0.3
        # IMPROVED: Pre-compute inverse cell dimensions for fast zone lookups
        self._inv_cell_w = 0.0
        self._inv_cell_h = 0.0

    def analyze(self, base_image: Image.Image, cfg: dict):
        self.enabled = cfg.get("weight_balance_enabled", True)
        self.strength = cfg.get("weight_balance_strength", 5) / 10.0
        self.grid_n = cfg.get("weight_balance_grid", 4)
        self.dark_adj = cfg.get("weight_balance_dark_adj", 0.15)
        self.light_adj = cfg.get("weight_balance_light_adj", 0.12)
        self.opacity_adj = cfg.get("weight_balance_opacity_adj", 0.3)
        self.canvas_size = base_image.size

        if not self.enabled or self.strength <= 0:
            return

        # IMPROVED: Single vectorized luminance computation
        rgb = np.array(base_image.convert("RGB")).astype(np.float32)
        lum = _lum_from_rgb(rgb)
        h, w = lum.shape
        n = self.grid_n

        # IMPROVED: Pre-compute inverse cell size for O(1) zone lookups
        self._inv_cell_w = n / max(1, w)
        self._inv_cell_h = n / max(1, h)

        self.zone_weights = np.zeros((n, n), dtype=np.float32)
        cell_h, cell_w = h // n, w // n

        for gy in range(n):
            for gx in range(n):
                y1, y2 = gy * cell_h, min((gy + 1) * cell_h, h)
                x1, x2 = gx * cell_w, min((gx + 1) * cell_w, w)
                self.zone_weights[gy, gx] = 1.0 - (np.mean(lum[y1:y2, x1:x2]) / 255.0)

        mean_w = np.mean(self.zone_weights)
        raw_bias = self.zone_weights - mean_w
        max_dev = max(abs(raw_bias.max()), abs(raw_bias.min()), 0.01)
        self.zone_bias = np.clip(raw_bias / max_dev, -1.0, 1.0)

    def _get_zone(self, canvas_x: int, canvas_y: int) -> tuple:
        """IMPROVED: Fast zone lookup using pre-computed inverse dimensions."""
        n = self.grid_n
        gx = min(n - 1, max(0, int(canvas_x * self._inv_cell_w)))
        gy = min(n - 1, max(0, int(canvas_y * self._inv_cell_h)))
        return gy, gx

    def get_adjustment(self, canvas_x: int, canvas_y: int) -> dict:
        if not self.enabled or self.zone_bias is None or self.strength <= 0:
            return {"brightness_adj": 0.0, "opacity_mult": 1.0, "zone_bias": 0.0}

        gy, gx = self._get_zone(canvas_x, canvas_y)
        bias = float(self.zone_bias[gy, gx]) * self.strength

        if bias > 0:
            return {"brightness_adj": bias * self.light_adj,
                    "opacity_mult": max(0.3, 1.0 - bias * self.opacity_adj),
                    "zone_bias": bias}
        else:
            return {"brightness_adj": bias * self.dark_adj,
                    "opacity_mult": 1.0,
                    "zone_bias": bias}

    def get_light_zones(self, threshold: float = -0.2) -> list:
        if self.zone_bias is None:
            return []
        # IMPROVED: Vectorized zone search
        ys, xs = np.where(self.zone_bias < threshold)
        return list(zip(ys.tolist(), xs.tolist()))

    def point_in_zone(self, x: int, y: int, gy: int, gx: int) -> bool:
        zy, zx = self._get_zone(x, y)
        return zy == gy and zx == gx

    def log(self):
        if not self.enabled or self.zone_weights is None:
            print("   ⚖️  Weight balance: disabled")
            return
        print(f"   ⚖️  Weight balance: {self.strength:.0%} strength, "
              f"{self.grid_n}×{self.grid_n} grid")
        n = self.grid_n
        for gy in range(n):
            row = " ".join(f"{self.zone_weights[gy, gx]:.2f}" for gx in range(n))
            bias_row = " ".join(
                "↑" if self.zone_bias[gy, gx] > 0.1 else
                "↓" if self.zone_bias[gy, gx] < -0.1 else "·"
                for gx in range(n))
            print(f"       row{gy}: [{row}]  bias: [{bias_row}]")


def promote_dark_in_light_zones(
        all_edge_pixels, interest_map, normal_x, normal_y,
        lum_map, alpha_crop, subject_crop,
        existing_candidates, weight_map, offset, cfg):
    """
    Find dark patches in zones the weight map considers 'too light'.
    """
    if not cfg.get("weight_balance_promote_dark", True):
        return []
    if not weight_map.enabled or weight_map.zone_bias is None:
        return []

    light_zones = weight_map.get_light_zones(threshold=-0.15)
    if not light_zones:
        return []

    max_promote = cfg.get("weight_balance_promote_max", 8)
    h, w = lum_map.shape
    dark_thresh = cfg["dark_threshold"]
    inset = cfg["edge_inset_px"]

    rng = random.Random(cfg["random_seed"] + 5555)

    # IMPROVED: Use a set of coarse grid cells for fast dedup
    existing_set = set()
    for c in existing_candidates:
        existing_set.add((c["x"] // 10, c["y"] // 10))

    # IMPROVED: Convert light_zones to a set for O(1) lookup
    light_zone_set = set(light_zones)

    zone_candidates = []
    for ex, ey in all_edge_pixels:
        canvas_x, canvas_y = ex + offset[0], ey + offset[1]

        # IMPROVED: Direct zone lookup instead of iterating all light zones
        zone = weight_map._get_zone(canvas_x, canvas_y)
        if zone not in light_zone_set:
            continue

        ey_c = max(0, min(h - 1, ey))
        ex_c = max(0, min(w - 1, ex))
        lum_val = float(lum_map[ey_c, ex_c])

        if lum_val >= dark_thresh:
            continue

        coarse = (ex // 10, ey // 10)
        if coarse in existing_set:
            continue

        nx_v = float(normal_x[ey_c, ex_c])
        ny_v = float(normal_y[ey_c, ex_c])
        sx = max(0, min(w - 1, int(ex + (-nx_v) * inset)))
        sy = max(0, min(h - 1, int(ey + (-ny_v) * inset)))

        darkness_score = (dark_thresh - lum_val) / max(1, dark_thresh)
        gy, gx = zone
        zone_need = abs(float(weight_map.zone_bias[gy, gx]))

        interest = max(0.3, min(1.0, darkness_score * 0.5 + zone_need * 0.5))

        zone_candidates.append({
            "x": sx, "y": sy, "edge_x": ex, "edge_y": ey,
            "nx": nx_v, "ny": ny_v,
            "interest": interest, "lum": lum_val,
            "is_dark": True, "is_bright": False,
            "_promoted": True,
        })

    if not zone_candidates:
        return []

    zone_candidates.sort(key=lambda p: p["interest"], reverse=True)
    thinned = _poisson_thin(zone_candidates, cfg["min_point_spacing"])
    result = thinned[:max_promote]

    print(f"   ⚖️  Promoted {len(result)} dark points in light zones "
          f"(lums: {[int(p['lum']) for p in result]})")
    return result


# ====================================================================
# EDGE DETECTION & NORMALS
# ====================================================================

def compute_edge_pixels(alpha: np.ndarray, threshold: int = 20):
    binary = (alpha > threshold).astype(np.uint8) * 255
    eroded = ndimage.minimum_filter(binary, size=3)
    # IMPROVED: Compute difference in-place to avoid extra allocation
    diff = binary.astype(np.int16)
    diff -= eroded.astype(np.int16)
    ys, xs = np.where(diff > 0)
    return list(zip(xs.tolist(), ys.tolist()))


def compute_normals(alpha: np.ndarray, radius: int = 10):
    binary = (alpha > 128).astype(np.float32)
    ksz = radius * 2 + 1
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    count = ndimage.uniform_filter(binary, size=ksz, mode='constant', cval=0)
    safe = np.maximum(count, 1e-10)
    avg_x = ndimage.uniform_filter(binary * xx, size=ksz, mode='constant', cval=0) / safe
    avg_y = ndimage.uniform_filter(binary * yy, size=ksz, mode='constant', cval=0) / safe
    nx, ny = -(avg_x - xx), -(avg_y - yy)
    mag = np.sqrt(nx ** 2 + ny ** 2) + 1e-10
    return (nx / mag).astype(np.float32), (ny / mag).astype(np.float32)


# ====================================================================
# INTEREST MAP
# ====================================================================

def build_interest_map(subject_rgba: np.ndarray, cfg: dict) -> np.ndarray:
    rgb = subject_rgba[:, :, :3].astype(np.float32)
    alpha = subject_rgba[:, :, 3].astype(np.float32) / 255.0
    lum = _lum_from_rgb(rgb)

    mx, mn = np.max(rgb, axis=2), np.min(rgb, axis=2)
    sat01 = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)

    # IMPROVED: Vectorized gradient magnitude across all 3 channels at once
    r = cfg["edge_contrast_radius"]
    ksz = r * 2 + 1
    contrast = np.zeros_like(lum)
    for ch in range(3):
        gy, gx = np.gradient(rgb[:, :, ch])
        # IMPROVED: In-place operations to reduce allocations
        mag_sq = gx * gx
        mag_sq += gy * gy
        np.sqrt(mag_sq, out=mag_sq)
        contrast += ndimage.uniform_filter(mag_sq, size=ksz)
    contrast *= (1.0 / 3.0)
    c_max = contrast.max()
    if c_max > 0:
        contrast *= (1.0 / c_max)

    dark_t = cfg["dark_threshold"]
    bright_t = cfg["bright_threshold"]
    dark_mask = np.clip((dark_t - lum) / max(1, dark_t), 0, 1)
    bright_mask = np.clip((lum - bright_t) / max(1, 255 - bright_t), 0, 1)
    sat_mask = np.clip((sat01 - cfg["saturation_threshold"])
                       / (1.0 - cfg["saturation_threshold"] + 1e-6), 0, 1)

    interest = (cfg["edge_contrast_weight"] * contrast
                + cfg["dark_weight"] * dark_mask
                + cfg["bright_weight"] * bright_mask
                + cfg["saturation_weight"] * sat_mask)
    interest *= alpha

    sigma = cfg["interest_blur_sigma"]
    if sigma > 0:
        interest = ndimage.gaussian_filter(interest, sigma=sigma)

    i_max = interest.max()
    if i_max > 0:
        interest *= (1.0 / i_max)
    return interest.astype(np.float32)


# ====================================================================
# CANDIDATE SELECTION & CLASSIFICATION
# ====================================================================

def select_candidates(edge_pixels, interest_map, normal_x, normal_y, cfg):
    h, w = interest_map.shape
    inset = cfg["edge_inset_px"]
    threshold = cfg["interest_threshold"]
    max_pts = cfg["max_candidates"]
    spacing = cfg["min_point_spacing"]

    rng = random.Random(cfg["random_seed"])

    # IMPROVED: Vectorized pre-filter — avoid Python loop over all edge pixels
    if not edge_pixels:
        return []
    edge_arr = np.array(edge_pixels, dtype=np.int32)  # (N, 2) — x, y
    ex_all = edge_arr[:, 0]
    ey_all = edge_arr[:, 1]
    ex_c = np.clip(ex_all, 0, w - 1)
    ey_c = np.clip(ey_all, 0, h - 1)
    interest_vals = interest_map[ey_c, ex_c]
    above_mask = interest_vals >= threshold
    above_indices = np.where(above_mask)[0]

    # Shuffle deterministically
    idx_list = above_indices.tolist()
    rng.shuffle(idx_list)

    # Poisson-disk selection
    spacing_sq = spacing * spacing
    cell = spacing / math.sqrt(2)
    grid = {}
    candidates = []

    for idx in idx_list:
        if len(candidates) >= max_pts:
            break
        ex = int(ex_all[idx])
        ey = int(ey_all[idx])
        exc = int(ex_c[idx])
        eyc = int(ey_c[idx])

        gx_i, gy_i = int(ex / cell), int(ey / cell)
        too_close = False
        for ddx in range(-2, 3):
            if too_close:
                break
            for ddy in range(-2, 3):
                nb = grid.get((gx_i + ddx, gy_i + ddy))
                if nb and (ex - nb[0]) ** 2 + (ey - nb[1]) ** 2 < spacing_sq:
                    too_close = True
                    break
        if too_close:
            continue
        grid[(gx_i, gy_i)] = (ex, ey)
        nx_v = float(normal_x[eyc, exc])
        ny_v = float(normal_y[eyc, exc])
        sx = max(0, min(w - 1, int(ex + (-nx_v) * inset)))
        sy = max(0, min(h - 1, int(ey + (-ny_v) * inset)))
        candidates.append({
            "x": sx, "y": sy, "edge_x": ex, "edge_y": ey,
            "nx": nx_v, "ny": ny_v,
            "interest": float(interest_vals[idx]),
        })
    return candidates


def classify_points(candidates, lum_map, cfg):
    if not candidates:
        return []
    
    h, w = lum_map.shape
    dark_t = cfg["dark_threshold"]
    bright_t = cfg["bright_threshold"]
    
    # Bulk extract coords
    xs = np.clip([c["x"] for c in candidates], 0, w - 1)
    ys = np.clip([c["y"] for c in candidates], 0, h - 1)
    
    # Bulk lookup (C-speed)
    lums = lum_map[ys, xs]
    
    for i, pt in enumerate(candidates):
        L = float(lums[i])
        pt["lum"] = L
        pt["is_dark"] = L < dark_t
        pt["is_bright"] = L > bright_t
        
    return candidates


# ====================================================================
# PATCH SIZE ESTIMATION
# ====================================================================

def estimate_patch_area(lum_map, alpha, cx, cy, spot_lum, cfg):
    h, w = lum_map.shape
    tolerance = cfg.get("bright_spot_flood_lum_tolerance", 40)
    max_r = cfg.get("bright_spot_flood_max_radius", 80)
    min_lum = spot_lum - tolerance

    x1, x2 = max(0, cx - max_r), min(w, cx + max_r + 1)
    y1, y2 = max(0, cy - max_r), min(h, cy + max_r + 1)

    # IMPROVED: Bounds check before indexing
    seed_lx, seed_ly = cx - x1, cy - y1
    local_bright = (lum_map[y1:y2, x1:x2] >= min_lum) & (alpha[y1:y2, x1:x2] > 128)

    if seed_ly < 0 or seed_ly >= local_bright.shape[0] or \
       seed_lx < 0 or seed_lx >= local_bright.shape[1] or \
       not local_bright[seed_ly, seed_lx]:
        return 0, None

    labelled, _ = ndimage.label(local_bright)
    seed_label = labelled[seed_ly, seed_lx]
    if seed_label == 0:
        return 0, None

    blob = labelled == seed_label
    area = int(np.sum(blob))

    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y1:y2, x1:x2] = blob
    return area, full_mask


# ====================================================================
# RAY-CAST BRIGHT SPOT SCANNER
# ====================================================================

def scan_bright_spots(subject_rgba, edge_pixels, normal_x, normal_y,
                      stats, lum_map, cfg):
    if not cfg.get("bright_spot_enabled", True):
        return []

    max_depth = cfg["bright_spot_max_depth"]
    step = cfg["bright_spot_step"]
    sample_r = cfg["bright_spot_sample_radius"]
    min_jump = stats.bright_scan_lum_jump
    min_lum = stats.bright_scan_min_lum
    bonus = cfg["bright_spot_score_bonus"]
    min_patch = cfg.get("bright_spot_min_patch_area", 150)
    max_patch = cfg.get("bright_spot_max_patch_area", 50000)

    h, w = subject_rgba.shape[:2]
    alpha = subject_rgba[:, :, 3]

    step_edge = max(1, len(edge_pixels) // 2000)
    sampled = list(edge_pixels[::step_edge])
    rng = random.Random(cfg["random_seed"] + 111)
    rng.shuffle(sampled)

    # --- THE SMART FIX: Precompute all colors instantly ---
    edge_color_map = precompute_area_colors(subject_rgba, 4)
    ray_color_map = precompute_area_colors(subject_rgba, sample_r)
    # ------------------------------------------------------

    raw_spots = []
    for ex, ey in sampled:
        ey_c = max(0, min(h - 1, ey))
        ex_c = max(0, min(w - 1, ex))
        inx = -float(normal_x[ey_c, ex_c])
        iny = -float(normal_y[ey_c, ex_c])

        # Fast O(1) lookup instead of slicing
        ec_data = edge_color_map[ey_c, ex_c]
        if ec_data[3] == 0:  # Check if alpha is 0 (empty area)
            continue
        edge_color = tuple(ec_data[:3])
        edge_lum = _lum_from_color(edge_color)

        best = None
        for d in range(step, max_depth + 1, step):
            sx = int(ex + inx * d)
            sy = int(ey + iny * d)
            
            # Fast O(1) lookup instead of slicing
            sx_c = max(0, min(w - 1, sx))
            sy_c = max(0, min(h - 1, sy))
            rc_data = ray_color_map[sy_c, sx_c]
            
            if rc_data[3] == 0: # Check if alpha is 0
                continue
                
            color = tuple(rc_data[:3])
            lv = _lum_from_color(color)
            
            if lv >= min_lum and (lv - edge_lum) >= min_jump:
                if best is None or lv > best["lum"]:
                    best = {"x": sx, "y": sy, "color": color, "lum": lv,
                            "lum_jump": lv - edge_lum, "depth": d,
                            "edge_x": ex, "edge_y": ey,
                            "nx": float(normal_x[ey_c, ex_c]),
                            "ny": float(normal_y[ey_c, ex_c])}
        if best:
            raw_spots.append(best)

    print(f"   🔦 Ray-cast: {len(raw_spots)} raw → ", end="")
    deduped = _poisson_thin(raw_spots, 20)
    print(f"{len(deduped)} unique")

    sized = []
    rej_s = rej_l = 0
    for s in deduped:
        area, mask = estimate_patch_area(lum_map, alpha, s["x"], s["y"], s["lum"], cfg)
        s["patch_area"], s["patch_mask"] = area, mask

        if area < min_patch:
            rej_s += 1
            s["rejected"], s["_accepted"] = "too_small", False
        elif area > max_patch:
            rej_l += 1
            s["rejected"], s["_accepted"] = "too_large", False
        else:
            jump_score = min(1.0, s["lum_jump"] / (min_jump * 2))
            lum_score = min(1.0, (s["lum"] - stats.lum_median) / max(1, 255 - stats.lum_median))
            size_score = min(1.0, math.log1p(area) / math.log1p(5000))
            s["interest"] = max(0.3, min(1.0,
                                         jump_score * 0.35 + lum_score * 0.25
                                         + size_score * 0.2 + bonus * 0.2))
            s["category"], s["rejected"], s["_accepted"] = "bright_spot", None, True
        sized.append(s)

    accepted = sorted([s for s in sized if s["_accepted"]],
                      key=lambda s: s["interest"], reverse=True)
    min_sp = cfg["bright_spot_min_spacing"]

    thinned = []
    thinned_indices = set()
    for i, s in enumerate(accepted):
        if not any((s["x"] - t["x"]) ** 2 + (s["y"] - t["y"]) ** 2 < min_sp ** 2
                   for t in thinned):
            thinned.append(s)
            thinned_indices.add(i)

    thinned_coords = set((s["x"], s["y"]) for s in thinned)
    for s in sized:
        s["_accepted"] = (s["x"], s["y"]) in thinned_coords and s.get("rejected") is None

    print(f"   🔦 Final: {len(thinned)} spots, {rej_s} small, {rej_l} large")
    return sized


# ====================================================================
# BRIGHT EDGE PROMOTION
# ====================================================================

def promote_bright_edges(candidates, subject_rgba, stats, existing_spots,
                         lum_map, cfg):
    if not cfg.get("bright_edge_promotion_enabled", True):
        return []

    min_contrast = cfg["bright_edge_global_contrast_min"]
    promo_r = cfg["bright_edge_promotion_radius"]
    min_spacing = cfg["bright_edge_min_spacing"]
    bonus = cfg["bright_edge_score_bonus"]
    min_patch = cfg.get("bright_edge_min_patch_area", 150)

    h, w = subject_rgba.shape[:2]
    alpha = subject_rgba[:, :, 3]
    median = stats.lum_median
    min_spacing_sq = min_spacing * min_spacing

    promoted = []
    for pt in candidates:
        if not pt.get("is_bright", False):
            continue
        color = _sample_area_color(subject_rgba, pt["x"], pt["y"], promo_r)
        if color is None:
            continue
        area_lum = _lum_from_color(color)
        gc = area_lum - median
        if gc < min_contrast:
            continue

        area, mask = estimate_patch_area(lum_map, alpha, pt["x"], pt["y"], area_lum, cfg)
        if area < min_patch:
            continue

        contrast_score = min(1.0, gc / max(1, 255 - median))
        size_score = min(1.0, math.log1p(area) / math.log1p(5000))
        interest = max(0.4, min(1.0,
                                contrast_score * 0.4 + pt["interest"] * 0.2
                                + size_score * 0.2 + bonus * 0.2))

        promoted.append({
            "x": pt["x"], "y": pt["y"], "color": color,
            "lum": area_lum, "lum_jump": gc, "depth": 0,
            "edge_x": pt.get("edge_x", pt["x"]),
            "edge_y": pt.get("edge_y", pt["y"]),
            "nx": pt["nx"], "ny": pt["ny"],
            "interest": interest, "category": "bright_spot",
            "source": "edge_promotion",
            "patch_area": area, "patch_mask": mask,
            "rejected": None, "_accepted": True,
        })

    if not promoted:
        return []

    promoted.sort(key=lambda s: s["interest"], reverse=True)
    existing_accepted = [s for s in existing_spots if s.get("_accepted")]

    thinned = []
    for p in promoted:
        if not any((p["x"] - t["x"]) ** 2 + (p["y"] - t["y"]) ** 2 < min_spacing_sq
                   for t in existing_accepted + thinned):
            thinned.append(p)

    print(f"   ✨ Promoted {len(thinned)} bright edges")
    return thinned


# ====================================================================
# COLOUR ADJUSTMENT
# ====================================================================

def adjust_colour(colour, cfg, is_dark=False, is_bright=False,
                  accent_pass="main", weight_adj: float = 0.0):
    r, g, b = float(colour[0]), float(colour[1]), float(colour[2])

    w = cfg["stroke_warmth"]
    if w > 0:
        r += w * 25; g += w * 5; b -= w * 15

    gray = (r + g + b) / 3.0
    s = cfg["stroke_saturation_boost"]
    r, g, b = gray + (r - gray) * s, gray + (g - gray) * s, gray + (b - gray) * s

    adj = cfg["stroke_brightness_adj"] + weight_adj
    if accent_pass == "dark":
        adj -= cfg.get("dark_accent_darken", 0.25)
    elif accent_pass == "bright":
        adj += cfg.get("bright_accent_lighten", 0.15)

    if adj > 0:
        r += (255 - r) * adj; g += (255 - g) * adj; b += (255 - b) * adj
    elif adj < 0:
        f = 1.0 + adj; r *= f; g *= f; b *= f

    if is_dark:
        cat_sat = cfg.get("dark_stroke_saturation", 1.0)
    elif is_bright:
        cat_sat = cfg.get("bright_stroke_saturation", 1.0)
    else:
        cat_sat = cfg.get("mid_stroke_saturation", 1.0)

    if cat_sat != 1.0:
        r = gray + (r - gray) * cat_sat
        g = gray + (g - gray) * cat_sat
        b = gray + (b - gray) * cat_sat

    return (int(max(0, min(255, r))),
            int(max(0, min(255, g))),
            int(max(0, min(255, b))))


# ====================================================================
# STROKE RENDERERS
# ====================================================================

def render_strokes(candidates, subject_rgba, canvas_size, offset,
                   brush_img, cfg, accent_pass="main",
                   weight_map: WeightBalanceMap = None):
    rng = random.Random(cfg["random_seed"] + _PASS_OFFSETS.get(accent_pass, 0))
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas_h = canvas_size[1]
    _no_weight = {"brightness_adj": 0.0, "opacity_mult": 1.0}

    for pt in candidates:
        if accent_pass == "dark" and not pt.get("is_dark", False):
            continue
        if accent_pass == "bright" and not pt.get("is_bright", False):
            continue

        colour = _sample_area_color(subject_rgba, pt["x"], pt["y"], 12)
        if colour is None:
            colour = tuple(subject_rgba[
                max(0, min(subject_rgba.shape[0] - 1, pt["y"])),
                max(0, min(subject_rgba.shape[1] - 1, pt["x"]))
            ][:3])

        canvas_x = pt["x"] + offset[0]
        canvas_y = pt["y"] + offset[1]
        w_adj = weight_map.get_adjustment(canvas_x, canvas_y) if weight_map else _no_weight

        colour = adjust_colour(colour, cfg,
                               is_dark=pt.get("is_dark", False),
                               is_bright=pt.get("is_bright", False),
                               accent_pass=accent_pass,
                               weight_adj=w_adj["brightness_adj"])

        base = cfg["stroke_base_size"]
        if accent_pass == "dark":
            base = int(base * cfg.get("dark_accent_size_mult", 0.7))
        elif accent_pass == "bright":
            base = int(base * cfg.get("bright_accent_size_mult", 0.85))
        else:
            base = int(base * cfg.get("mid_stroke_size_mult", 1.0))
        jitter = cfg["stroke_size_jitter"]
        sz = max(20, int(base * rng.uniform(1 - jitter, 1 + jitter)))

        nx, ny = pt["nx"], pt["ny"]
        angle = math.degrees(math.atan2(-ny, nx)) + 270
        angle += rng.uniform(-180 * cfg["rotation_jiggle"], 180 * cfg["rotation_jiggle"])

        scatter = cfg["interior_scatter_px"]
        sx = pt["x"] + int(-nx * rng.uniform(0, scatter))
        sy = pt["y"] + int(-ny * rng.uniform(0, scatter))

        # IMPROVED: Use cached brush transform
        rotated = _get_transformed_brush(brush_img, sz, angle)
        colored = _colorize(rotated, colour)

        if accent_pass == "dark":
            op = cfg.get("dark_accent_opacity", 0.55)
        elif accent_pass == "bright":
            op = cfg.get("bright_accent_opacity", 0.40)
        else:
            mid_op = cfg.get("mid_stroke_opacity")
            op = mid_op if mid_op is not None else cfg["stroke_opacity_base"]
        op *= pt["interest"]
        op += rng.uniform(-cfg["stroke_opacity_jitter"], cfg["stroke_opacity_jitter"])
        op = max(0.05, min(1.0, op))
        op *= w_adj["opacity_mult"]
        op *= _bottom_fade_mult(canvas_y, canvas_h, cfg)
        op = max(0.02, min(1.0, op))

        colored = _apply_opacity(colored, op)
        px = sx + offset[0] - rotated.width // 2
        py = sy + offset[1] - rotated.height // 2
        _composite_safe(layer, colored, px, py)

    return layer


def render_spot_strokes(spots, subject_rgba, canvas_size, offset,
                        brush_img, cfg,
                        weight_map: WeightBalanceMap = None):
    accepted = [s for s in spots if s.get("_accepted", False)]
    if not accepted:
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    rng = random.Random(cfg["random_seed"] + 8888)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas_h = canvas_size[1]
    _no_weight = {"brightness_adj": 0.0, "opacity_mult": 1.0}

    base_size = cfg["bright_spot_stroke_size"]
    size_jitter = cfg["bright_spot_size_jitter"]
    opacity_base = cfg["bright_spot_opacity"]
    lighten = cfg["bright_spot_lighten"]
    warmth = cfg["bright_spot_warmth"]
    sat_b = cfg["stroke_saturation_boost"]

    area_min = cfg.get("bright_spot_area_size_min", 0.6)
    area_max = cfg.get("bright_spot_area_size_max", 1.6)
    ref_s = cfg.get("bright_spot_area_ref_small", 700)
    ref_l = cfg.get("bright_spot_area_ref_large", 5000)
    log_s, log_l = math.log1p(ref_s), math.log1p(ref_l)
    log_range = max(0.01, log_l - log_s)

    for spot in accepted:
        colour = _sample_area_color(subject_rgba, spot["x"], spot["y"], 15)
        if colour is None:
            continue

        canvas_x = spot["x"] + offset[0]
        canvas_y = spot["y"] + offset[1]
        w_adj = weight_map.get_adjustment(canvas_x, canvas_y) if weight_map else _no_weight

        r, g, b = float(colour[0]), float(colour[1]), float(colour[2])
        if warmth > 0:
            r += warmth * 25; g += warmth * 5; b -= warmth * 15
        gray = (r + g + b) / 3.0
        r = gray + (r - gray) * sat_b
        g = gray + (g - gray) * sat_b
        b = gray + (b - gray) * sat_b

        total_lighten = lighten + w_adj["brightness_adj"]
        if total_lighten > 0:
            r += (255 - r) * total_lighten
            g += (255 - g) * total_lighten
            b += (255 - b) * total_lighten
        elif total_lighten < 0:
            f = 1.0 + total_lighten; r *= f; g *= f; b *= f

        colour = (int(max(0, min(255, r))),
                  int(max(0, min(255, g))),
                  int(max(0, min(255, b))))

        area = spot.get("patch_area", 500)
        t = float(np.clip((math.log1p(area) - log_s) / log_range, 0.0, 1.0))
        area_factor = area_min + (area_max - area_min) * t
        sz = max(20, int(base_size * area_factor * rng.uniform(1 - size_jitter, 1 + size_jitter)))

        nx, ny = spot["nx"], spot["ny"]
        angle = math.degrees(math.atan2(-ny, nx)) + 270 + rng.uniform(-60, 60)

        # IMPROVED: Use cached brush transform
        rotated = _get_transformed_brush(brush_img, sz, angle)
        colored = _colorize(rotated, colour)

        op = max(0.05, min(1.0, opacity_base * spot["interest"] + rng.uniform(-0.1, 0.1)))
        op *= w_adj["opacity_mult"]
        op *= _bottom_fade_mult(canvas_y, canvas_h, cfg)
        op = max(0.02, min(1.0, op))
        colored = _apply_opacity(colored, op)

        px = spot["x"] + offset[0] - rotated.width // 2
        py = spot["y"] + offset[1] - rotated.height // 2
        _composite_safe(layer, colored, px, py)

    return layer


# ====================================================================
# DEBUG
# ====================================================================

def save_debug_interest(interest_map, alpha, out_path):
    h, w = interest_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 0] = (interest_map * 255).astype(np.uint8)
    vis[:, :, 2] = ((1 - interest_map) * 255).astype(np.uint8)
    vis[:, :, 1] = (np.minimum(interest_map, 1 - interest_map) * 360).astype(np.uint8)
    a = (alpha / 255.0)[:, :, np.newaxis]
    vis = (vis.astype(np.float32) * a * 0.8 + 40).astype(np.uint8)
    Image.fromarray(vis).save(out_path)
    print(f"   💾 {out_path}")


def save_debug_candidates(subject_rgba, candidates, bright_spots, out_path):
    vis = Image.fromarray(subject_rgba).copy()
    draw = ImageDraw.Draw(vis)

    for pt in candidates:
        x, y = pt["x"], pt["y"]
        r = int(4 + pt["interest"] * 6)
        promoted = pt.get("_promoted", False)
        if promoted:
            col = (255, 165, 0, 230)
        elif pt.get("is_dark"):
            col = (255, 50, 50, 200)
        elif pt.get("is_bright"):
            col = (50, 255, 50, 200)
        else:
            col = (50, 150, 255, 200)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255, 150))

    for spot in bright_spots:
        x, y = spot["x"], spot["y"]
        area = spot.get("patch_area", 0)
        accepted = spot.get("_accepted", False)
        rejected = spot.get("rejected")
        if accepted:
            r = max(8, min(30, int(math.sqrt(area) / 3)))
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 0, 255), width=3)
            draw.text((x + r + 4, y - 6), f"{area}px²", fill=(255, 255, 0, 255))
        elif rejected == "too_small":
            r = max(4, int(math.sqrt(area) / 4))
            draw.line([x - r, y - r, x + r, y + r], fill=(255, 80, 80, 150), width=2)
            draw.line([x - r, y + r, x + r, y - r], fill=(255, 80, 80, 150), width=2)
        elif rejected == "too_large":
            r = 15
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(180, 50, 255, 150), width=2)

    legend_y = 10
    for col, text in [
        ((50, 150, 255), "Edge (mid)"), ((255, 50, 50), "Edge (dark)"),
        ((50, 255, 50), "Edge (bright)"), ((255, 165, 0), "Promoted (weight balance)"),
        ((255, 255, 0), "Bright spot ✓"), ((255, 80, 80), "Bright spot ✗"),
    ]:
        draw.rectangle([10, legend_y, 22, legend_y + 12], fill=col + (220,))
        draw.text((26, legend_y - 1), text, fill=(255, 255, 255, 220))
        legend_y += 16

    vis.save(out_path)
    print(f"   💾 {out_path}")


def save_debug_weight_map(weight_map: WeightBalanceMap, canvas_size, out_path):
    if not weight_map.enabled or weight_map.zone_bias is None:
        return
    w, h = canvas_size
    n = weight_map.grid_n
    vis = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(vis)
    cell_w, cell_h = w // n, h // n

    for gy in range(n):
        for gx in range(n):
            bias = weight_map.zone_bias[gy, gx]
            x1, y1 = gx * cell_w, gy * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            if bias > 0.1:
                fill = (255, 100, 100, int(abs(bias) * 60))
                label = f"heavy ↑{bias:.2f}"
            elif bias < -0.1:
                fill = (100, 100, 255, int(abs(bias) * 60))
                label = f"light ↓{bias:.2f}"
            else:
                fill = (200, 200, 200, 20)
                label = f"·{bias:.2f}"
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=(255, 255, 255, 80))
            draw.text((x1 + 5, y1 + 5), label, fill=(255, 255, 255, 200))

    vis.save(out_path)
    print(f"   💾 {out_path}")


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def create_accent_strokes(bottom: Image.Image,
                          top: Image.Image,
                          subject_oil: Image.Image,
                          cfg: dict | None = None,
                          debug_dir: str | None = None,
                          return_strokes_only: bool = False) -> Image.Image:
    settings = {**DEFAULTS}
    if cfg:
        settings.update(cfg)

    # IMPROVED: Clear brush transform cache at start of each run
    _BRUSH_TRANSFORM_CACHE.clear()

    so_arr = np.array(subject_oil.convert("RGBA"))
    alpha = so_arr[:, :, 3]

    ys, xs = np.where(alpha > 20)
    if len(ys) == 0:
        print("⚠️  No visible subject!")
        result = bottom.copy()
        result.alpha_composite(top)
        return result

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    subject_crop = so_arr[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    alpha_crop = alpha[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = (bbox[0], bbox[1])
    canvas_size = (bottom.width, bottom.height)

    print(f"   Subject: {subject_crop.shape[1]}×{subject_crop.shape[0]} "
          f"at ({bbox[0]},{bbox[1]})")

    lum_map = _lum_from_rgb(subject_crop[:, :, :3].astype(np.float32))

    adaptive_strength = settings.get("adaptive_strength", 7) / 10.0
    print(f"   📊 Adaptive: {settings.get('adaptive_strength', 7)}/10")

    edge_pixels = compute_edge_pixels(alpha_crop)
    print(f"   {len(edge_pixels)} edge pixels")

    # IMPROVED: Fully vectorized edge luminance sampling (was a Python list comprehension)
    if edge_pixels:
        edge_arr = np.array(edge_pixels, dtype=np.int32)
        step_e = max(1, len(edge_pixels) // 2000)
        sampled_edges = edge_arr[::step_e]
        ex_s = np.clip(sampled_edges[:, 0], 0, alpha_crop.shape[1] - 1)
        ey_s = np.clip(sampled_edges[:, 1], 0, alpha_crop.shape[0] - 1)
        edge_lums = lum_map[ey_s, ex_s]
    else:
        edge_lums = np.array([], dtype=np.float32)

    stats = ImageStats()
    stats.analyze(edge_lums, adaptive_strength)
    stats.log()

    print("   🔍 Interest map + normals…")
    interest = build_interest_map(subject_crop, settings)
    normal_x, normal_y = compute_normals(alpha_crop)

    print("   📍 Candidates…")
    candidates = select_candidates(edge_pixels, interest, normal_x, normal_y, settings)
    candidates = classify_points(candidates, lum_map, settings)
    n_dark = sum(1 for p in candidates if p["is_dark"])
    n_bright = sum(1 for p in candidates if p["is_bright"])
    print(f"   {len(candidates)} pts ({n_dark} dark, {n_bright} bright, "
          f"{len(candidates) - n_dark - n_bright} mid)")

    print("   ⚖️  Analyzing visual weight…")
    base_composite = bottom.copy()
    base_composite.alpha_composite(top)
    weight_map = WeightBalanceMap()
    weight_map.analyze(base_composite, settings)
    weight_map.log()

    promoted_dark = promote_dark_in_light_zones(
        edge_pixels, interest, normal_x, normal_y,
        lum_map, alpha_crop, subject_crop,
        candidates, weight_map, offset, settings)
    candidates.extend(promoted_dark)

    print("   🔦 Bright spots…")
    all_spots = scan_bright_spots(
        subject_crop, edge_pixels, normal_x, normal_y, stats, lum_map, settings)
    promoted_bright = promote_bright_edges(
        candidates, subject_crop, stats, all_spots, lum_map, settings)
    all_spots.extend(promoted_bright)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        save_debug_interest(interest, alpha_crop,
                            os.path.join(debug_dir, "debug_interest.png"))
        save_debug_candidates(subject_crop, candidates, all_spots,
                              os.path.join(debug_dir, "debug_candidates.png"))
        save_debug_weight_map(weight_map, canvas_size,
                              os.path.join(debug_dir, "debug_weight.png"))

    brush_path = settings["brush_file"]
    if not os.path.isabs(brush_path):
        brush_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), brush_path)
    settings["brush_file"] = brush_path
    brush_img = _load_brush(brush_path)

    print("   🖌️  Rendering…")
    main_layer = render_strokes(candidates, subject_crop, canvas_size, offset,
                                brush_img, settings, "main", weight_map)

    dark_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                 brush_img, settings, "dark", weight_map)
                  if settings["dark_accent_enabled"] else None)

    bright_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                   brush_img, settings, "bright", weight_map)
                    if settings["bright_accent_enabled"] else None)

    spot_layer = render_spot_strokes(all_spots, subject_crop, canvas_size, offset,
                                     brush_img, settings, weight_map)

    strokes = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    strokes.alpha_composite(main_layer)
    if dark_layer:
        strokes.alpha_composite(dark_layer)
    if bright_layer:
        strokes.alpha_composite(bright_layer)
    strokes.alpha_composite(spot_layer)

    if debug_dir:
        strokes.save(os.path.join(debug_dir, "debug_strokes.png"))

    if return_strokes_only:
        return strokes

    print("   🎨 Compositing…")
    result = bottom.copy()
    if settings["layer_order"] == "between":
        result.alpha_composite(strokes)
        result.alpha_composite(top)
    else:
        result.alpha_composite(top)
        result.alpha_composite(strokes)
    return result




def create_accent_stroke_layer(canvas_below: Image.Image,
                                canvas_above: Image.Image,
                                subject_oil: Image.Image,
                                cfg: dict | None = None,
                                debug_dir: str | None = None) -> Image.Image:
    """
    Same pipeline as create_accent_strokes, but returns ONLY the strokes
    RGBA layer (no compositing).  watercolor.py handles layer ordering.
    
    canvas_below: everything composited BELOW the stroke insertion point
    canvas_above: everything composited ABOVE (used only for weight-balance analysis)
    subject_oil:  the subject with oil-paint effect (RGBA, full canvas size)
    """
    settings = {**DEFAULTS}
    if cfg:
        settings.update(cfg)

    _BRUSH_TRANSFORM_CACHE.clear()

    so_arr = np.array(subject_oil.convert("RGBA"))
    alpha = so_arr[:, :, 3]

    ys, xs = np.where(alpha > 20)
    if len(ys) == 0:
        print("   ⚠️  Accent strokes: no visible subject")
        return Image.new("RGBA", (canvas_below.width, canvas_below.height), (0, 0, 0, 0))

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    subject_crop = so_arr[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    alpha_crop = alpha[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = (bbox[0], bbox[1])
    canvas_size = (canvas_below.width, canvas_below.height)

    print(f"   🎨 Accent strokes: subject {subject_crop.shape[1]}×{subject_crop.shape[0]} at ({bbox[0]},{bbox[1]})")

    lum_map = _lum_from_rgb(subject_crop[:, :, :3].astype(np.float32))

    adaptive_strength = settings.get("adaptive_strength", 7) / 10.0

    edge_pixels = compute_edge_pixels(alpha_crop)

    if edge_pixels:
        edge_arr = np.array(edge_pixels, dtype=np.int32)
        step_e = max(1, len(edge_pixels) // 2000)
        sampled_edges = edge_arr[::step_e]
        ex_s = np.clip(sampled_edges[:, 0], 0, alpha_crop.shape[1] - 1)
        ey_s = np.clip(sampled_edges[:, 1], 0, alpha_crop.shape[0] - 1)
        edge_lums = lum_map[ey_s, ex_s]
    else:
        edge_lums = np.array([], dtype=np.float32)

    stats = ImageStats()
    stats.analyze(edge_lums, adaptive_strength)

    interest = build_interest_map(subject_crop, settings)
    normal_x, normal_y = compute_normals(alpha_crop)

    candidates = select_candidates(edge_pixels, interest, normal_x, normal_y, settings)
    candidates = classify_points(candidates, lum_map, settings)

    # Weight balance uses the full composite (below + above) for analysis
    weight_ref = canvas_below.copy()
    weight_ref.alpha_composite(canvas_above)
    weight_map = WeightBalanceMap()
    weight_map.analyze(weight_ref, settings)
    del weight_ref  # free memory

    promoted_dark = promote_dark_in_light_zones(
        edge_pixels, interest, normal_x, normal_y,
        lum_map, alpha_crop, subject_crop,
        candidates, weight_map, offset, settings)
    candidates.extend(promoted_dark)

    all_spots = scan_bright_spots(
        subject_crop, edge_pixels, normal_x, normal_y, stats, lum_map, settings)
    promoted_bright = promote_bright_edges(
        candidates, subject_crop, stats, all_spots, lum_map, settings)
    all_spots.extend(promoted_bright)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        save_debug_interest(interest, alpha_crop,
                            os.path.join(debug_dir, "debug_accent_interest.png"))
        save_debug_candidates(subject_crop, candidates, all_spots,
                              os.path.join(debug_dir, "debug_accent_candidates.png"))
        save_debug_weight_map(weight_map, canvas_size,
                              os.path.join(debug_dir, "debug_accent_weight.png"))

    brush_path = settings["brush_file"]
    if not os.path.isabs(brush_path):
        brush_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), brush_path)
    settings["brush_file"] = brush_path
    brush_img = _load_brush(brush_path)

    main_layer = render_strokes(candidates, subject_crop, canvas_size, offset,
                                brush_img, settings, "main", weight_map)
    dark_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                 brush_img, settings, "dark", weight_map)
                  if settings["dark_accent_enabled"] else None)
    bright_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                   brush_img, settings, "bright", weight_map)
                    if settings["bright_accent_enabled"] else None)
    spot_layer = render_spot_strokes(all_spots, subject_crop, canvas_size, offset,
                                     brush_img, settings, weight_map)

    strokes = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    strokes.alpha_composite(main_layer)
    if dark_layer:
        strokes.alpha_composite(dark_layer)
    if bright_layer:
        strokes.alpha_composite(bright_layer)
    strokes.alpha_composite(spot_layer)

    if debug_dir:
        strokes.save(os.path.join(debug_dir, "debug_accent_strokes.png"))

    n_acc = len(candidates) + sum(1 for s in all_spots if s.get("_accepted"))
    print(f"   🎨 Accent strokes: {n_acc} total strokes rendered")
    return strokes


# ====================================================================
# CLI
# ====================================================================

def load_layers_from_zip(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        def find(keyword):
            return next((n for n in names
                         if keyword in n.lower() and n.lower().endswith(".png")), None)
        bn, tn, sn = find("bottom"), find("top"), find("subject_oil") or find("subject")
        if not all([bn, tn, sn]):
            raise FileNotFoundError(
                f"ZIP must contain bottom.png, top.png, subject_oil.png. Found: {names}")
        return (Image.open(BytesIO(zf.read(bn))).convert("RGBA"),
                Image.open(BytesIO(zf.read(tn))).convert("RGBA"),
                Image.open(BytesIO(zf.read(sn))).convert("RGBA"))


def main():
    parser = argparse.ArgumentParser(description="🎨 Accent Stroke Placer")
    parser.add_argument("zip_path", nargs="?", default=None)
    parser.add_argument("--brush", default="bgbrush.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--layer-order", choices=["between", "above"], default="between")
    parser.add_argument("--adaptive", type=int, default=7, choices=range(0, 11),
                        metavar="0-10")
    parser.add_argument("--min-patch", type=int, default=None)
    parser.add_argument("--max-patch", type=int, default=None)
    parser.add_argument("--mid-sat", type=float, default=None)
    parser.add_argument("--mid-size", type=float, default=None)
    parser.add_argument("--mid-opacity", type=float, default=None)
    parser.add_argument("--weight-strength", type=int, default=None, choices=range(0, 11),
                        metavar="0-10", help="Weight balance strength")
    parser.add_argument("--no-bottom-fade", action="store_true")
    args = parser.parse_args()

    zip_path = args.zip_path
    if not zip_path:
        zips = sorted(glob.glob("*.zip"))
        zip_path = zips[0] if zips else None
    if not zip_path:
        print("❌ No ZIP found.")
        sys.exit(1)

    print(f"📦 Loading: {zip_path}")
    bottom, top, subject_oil = load_layers_from_zip(zip_path)
    print(f"   Canvas: {bottom.width}×{bottom.height}")

    out_dir = os.path.dirname(os.path.abspath(zip_path))

    original_path = os.path.join(out_dir, "result_original.png")
    print(f"   📸 Saving original…")
    original = bottom.copy()
    original.alpha_composite(top)
    original.save(original_path)
    print(f"   💾 {original_path}")

    user_cfg = {
        "brush_file": args.brush,
        "random_seed": args.seed,
        "layer_order": args.layer_order,
        "adaptive_strength": args.adaptive,
    }
    if args.min_patch is not None:
        user_cfg["bright_spot_min_patch_area"] = args.min_patch
        user_cfg["bright_edge_min_patch_area"] = args.min_patch
    if args.max_patch is not None:
        user_cfg["bright_spot_max_patch_area"] = args.max_patch
    if args.mid_sat is not None:
        user_cfg["mid_stroke_saturation"] = args.mid_sat
    if args.mid_size is not None:
        user_cfg["mid_stroke_size_mult"] = args.mid_size
    if args.mid_opacity is not None:
        user_cfg["mid_stroke_opacity"] = args.mid_opacity
    if args.weight_strength is not None:
        user_cfg["weight_balance_strength"] = args.weight_strength
    if args.no_bottom_fade:
        user_cfg["bottom_fade_enabled"] = False

    debug_dir = None if args.no_debug else out_dir

    result = create_accent_strokes(bottom, top, subject_oil,
                                   cfg=user_cfg, debug_dir=debug_dir)

    result_path = os.path.join(out_dir, "result.png")
    result.save(result_path)
    print(f"✅ Saved: {result_path}")
    print(f"   Compare: {original_path}")


def create_accent_strokes_detailed(
    bottom: Image.Image,
    top: Image.Image,
    subject_oil: Image.Image,
    cfg: dict | None = None,
    assets_dir: str | None = None,
) -> dict:
    """
    100% pixel-identical to create_accent_strokes(..., return_strokes_only=True),
    while also returning the individual sub-layers and candidate points for live studio editing.
    """
    settings = {**DEFAULTS}
    if cfg:
        settings.update(cfg)

    _BRUSH_TRANSFORM_CACHE.clear()

    so_arr = np.array(subject_oil.convert("RGBA"))
    alpha = so_arr[:, :, 3]

    ys, xs = np.where(alpha > 20)
    canvas_size = (bottom.width, bottom.height)
    empty_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if len(ys) == 0:
        return {
            "combined": empty_layer,
            "main": empty_layer,
            "dark": empty_layer,
            "bright": empty_layer,
            "spot": empty_layer,
            "candidates": [],
            "spots": [],
            "settings": settings,
        }

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    subject_crop = so_arr[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    alpha_crop = alpha[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = (bbox[0], bbox[1])

    lum_map = _lum_from_rgb(subject_crop[:, :, :3].astype(np.float32))
    adaptive_strength = settings.get("adaptive_strength", 7) / 10.0
    edge_pixels = compute_edge_pixels(alpha_crop)

    if edge_pixels:
        edge_arr = np.array(edge_pixels, dtype=np.int32)
        step_e = max(1, len(edge_pixels) // 2000)
        sampled_edges = edge_arr[::step_e]
        ex_s = np.clip(sampled_edges[:, 0], 0, alpha_crop.shape[1] - 1)
        ey_s = np.clip(sampled_edges[:, 1], 0, alpha_crop.shape[0] - 1)
        edge_lums = lum_map[ey_s, ex_s]
    else:
        edge_lums = np.array([], dtype=np.float32)

    stats = ImageStats()
    stats.analyze(edge_lums, adaptive_strength)

    interest = build_interest_map(subject_crop, settings)
    normal_x, normal_y = compute_normals(alpha_crop)

    candidates = select_candidates(edge_pixels, interest, normal_x, normal_y, settings)
    candidates = classify_points(candidates, lum_map, settings)

    base_composite = bottom.copy()
    base_composite.alpha_composite(top)
    weight_map = WeightBalanceMap()
    weight_map.analyze(base_composite, settings)

    promoted_dark = promote_dark_in_light_zones(
        edge_pixels, interest, normal_x, normal_y,
        lum_map, alpha_crop, subject_crop,
        candidates, weight_map, offset, settings
    )
    candidates.extend(promoted_dark)

    all_spots = scan_bright_spots(
        subject_crop, edge_pixels, normal_x, normal_y, stats, lum_map, settings
    )
    promoted_bright = promote_bright_edges(
        candidates, subject_crop, stats, all_spots, lum_map, settings
    )
    all_spots.extend(promoted_bright)

    brush_name = os.path.basename(settings["brush_file"])
    if assets_dir and os.path.exists(os.path.join(assets_dir, brush_name)):
        brush_path = os.path.join(assets_dir, brush_name)
    else:
        brush_path = settings["brush_file"]
        if not os.path.isabs(brush_path):
            brush_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), brush_path)
    settings["brush_file"] = brush_path
    brush_img = _load_brush(brush_path)

    main_layer = render_strokes(candidates, subject_crop, canvas_size, offset,
                                brush_img, settings, "main", weight_map)
    dark_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                 brush_img, settings, "dark", weight_map)
                  if settings["dark_accent_enabled"] else empty_layer.copy())
    bright_layer = (render_strokes(candidates, subject_crop, canvas_size, offset,
                                   brush_img, settings, "bright", weight_map)
                    if settings["bright_accent_enabled"] else empty_layer.copy())
    spot_layer = render_spot_strokes(all_spots, subject_crop, canvas_size, offset,
                                     brush_img, settings, weight_map)

    strokes = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    strokes.alpha_composite(main_layer)
    if settings["dark_accent_enabled"]:
        strokes.alpha_composite(dark_layer)
    if settings["bright_accent_enabled"]:
        strokes.alpha_composite(bright_layer)
    strokes.alpha_composite(spot_layer)

    # Serialize lightweight stroke points for live browser re-rendering
    serializable_candidates = []
    for pt in candidates:
        c_rgb = _sample_area_color(subject_crop, pt["x"], pt["y"], 12)
        if c_rgb is None:
            c_rgb = (128, 128, 128)
        w_adj = weight_map.get_adjustment(pt["x"] + offset[0], pt["y"] + offset[1])
        serializable_candidates.append({
            "x": int(pt["x"] + offset[0]),
            "y": int(pt["y"] + offset[1]),
            "nx": round(float(pt["nx"]), 4),
            "ny": round(float(pt["ny"]), 4),
            "interest": round(float(pt["interest"]), 4),
            "is_dark": bool(pt.get("is_dark", False)),
            "is_bright": bool(pt.get("is_bright", False)),
            "color": [int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])],
            "w_bright": round(float(w_adj["brightness_adj"]), 4),
            "w_op": round(float(w_adj["opacity_mult"]), 4),
        })

    serializable_spots = []
    for sp in all_spots:
        if not sp.get("_accepted", False):
            continue
        c_rgb = _sample_area_color(subject_crop, sp["x"], sp["y"], 15)
        if c_rgb is None:
            continue
        w_adj = weight_map.get_adjustment(sp["x"] + offset[0], sp["y"] + offset[1])
        serializable_spots.append({
            "x": int(sp["x"] + offset[0]),
            "y": int(sp["y"] + offset[1]),
            "nx": round(float(sp["nx"]), 4),
            "ny": round(float(sp["ny"]), 4),
            "interest": round(float(sp["interest"]), 4),
            "patch_area": int(sp.get("patch_area", 500)),
            "color": [int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])],
            "w_bright": round(float(w_adj["brightness_adj"]), 4),
            "w_op": round(float(w_adj["opacity_mult"]), 4),
        })

    return {
        "combined": strokes,
        "main": main_layer,
        "dark": dark_layer,
        "bright": bright_layer,
        "spot": spot_layer,
        "candidates": serializable_candidates,
        "spots": serializable_spots,
    }


if __name__ == "__main__":
    main()