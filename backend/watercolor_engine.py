"""
🎨 Watercolor Studio Engine (100% Pixel-Identical to HF + Un-Flattened Layers)
"""

import os
import io
import math
import time
import base64
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from scipy import ndimage
from rembg import remove, new_session

from presets import get_preset_copy, PRESETS
from edge_curve_editor import apply_edge_curve_effect
from accent_strokes import create_accent_strokes_detailed

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "assets")


def asset_path(filename: str) -> str:
    p = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(p):
        return p
    return os.path.join(BACKEND_DIR, filename)


# ============================================
# 🚀 GPU REMBG SESSION
# ============================================

REMBG_SESSION = None


def get_rembg_session():
    """Load u2net on GPU (CUDA) if available, with CPU fallback."""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        print("🚀 Loading u2net background removal model...")
        try:
            REMBG_SESSION = new_session(
                "u2net",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception:
            REMBG_SESSION = new_session("u2net")
        print("✅ Model loaded!")
    return REMBG_SESSION


# ============================================
# 🚀 FAST RANK-FILTER & IMAGE HELPERS
# ============================================

def _fast_min_filter(image: Image.Image, size: int) -> Image.Image:
    if size <= 5:
        return image.filter(ImageFilter.MinFilter(size))
    arr = np.asarray(image)
    if arr.ndim == 2:
        out = ndimage.minimum_filter(arr, size=size, mode="nearest")
    else:
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[:, :, c] = ndimage.minimum_filter(arr[:, :, c], size=size, mode="nearest")
    return Image.fromarray(out, image.mode)


def _fast_max_filter(image: Image.Image, size: int) -> Image.Image:
    if size <= 5:
        return image.filter(ImageFilter.MaxFilter(size))
    arr = np.asarray(image)
    if arr.ndim == 2:
        out = ndimage.maximum_filter(arr, size=size, mode="nearest")
    else:
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[:, :, c] = ndimage.maximum_filter(arr[:, :, c], size=size, mode="nearest")
    return Image.fromarray(out, image.mode)


_BRUSH_CACHE: Dict[str, Image.Image] = {}


def load_brush(filename: str) -> Image.Image:
    full_path = asset_path(os.path.basename(filename))
    if full_path in _BRUSH_CACHE:
        return _BRUSH_CACHE[full_path].copy()
    try:
        brush = Image.open(full_path).convert("RGBA")
        _BRUSH_CACHE[full_path] = brush
        return brush.copy()
    except Exception:
        size = 50
        brush = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(brush).ellipse([5, 5, size - 5, size - 5], fill=(0, 0, 0, 255))
        return brush


def pil_to_base64(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", compress_level=3)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ============================================
# 🚀 THREAD-SAFE EDGE CACHE
# ============================================

class EdgeCache:
    def __init__(self, mask: Image.Image):
        mask_array = np.array(mask)
        binary = (mask_array > 20).astype(np.uint8) * 255
        binary_img = Image.fromarray(binary)
        eroded = binary_img.filter(ImageFilter.MinFilter(3))
        eroded_array = np.array(eroded)

        edge = binary - eroded_array
        ys, xs = np.where(edge > 0)

        self.edge_pixels: List[Tuple[int, int]] = list(zip(xs.tolist(), ys.tolist()))
        self.mask_array = mask_array
        self.mask_size = mask.size
        self.sampled_points: Dict[int, List[Tuple[int, int]]] = {}
        self._edge_array_cache: Optional[np.ndarray] = None

        # Precompute outward normals via C-level uniform filter
        binary_f = (self.mask_array > 128).astype(np.float32)
        kernel_size = 21  # radius=10 -> 2*10+1
        count = ndimage.uniform_filter(binary_f, size=kernel_size, mode="constant", cval=0)
        h, w = self.mask_array.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        avg_x = ndimage.uniform_filter(binary_f * xx, size=kernel_size, mode="constant", cval=0)
        avg_y = ndimage.uniform_filter(binary_f * yy, size=kernel_size, mode="constant", cval=0)

        safe_count = np.maximum(count, 1e-10)
        avg_x /= safe_count
        avg_y /= safe_count

        nx = -(avg_x - xx)
        ny = -(avg_y - yy)
        mag = np.sqrt(nx ** 2 + ny ** 2) + 1e-10
        self._normal_x = (nx / mag).astype(np.float32)
        self._normal_y = (ny / mag).astype(np.float32)

    def get_sampled_points(self, spacing: int) -> List[Tuple[int, int]]:
        if spacing in self.sampled_points:
            return self.sampled_points[spacing]
        if not self.edge_pixels:
            self.sampled_points[spacing] = []
            return []

        spacing_sq = spacing * spacing
        cell_size = spacing / math.sqrt(2)
        grid = {}
        sampled = []

        for px, py in self.edge_pixels:
            gx = int(px / cell_size)
            gy = int(py / cell_size)
            too_close = False
            for dx in range(-2, 3):
                if too_close:
                    break
                for dy in range(-2, 3):
                    neighbor = grid.get((gx + dx, gy + dy))
                    if neighbor is not None:
                        if (px - neighbor[0]) ** 2 + (py - neighbor[1]) ** 2 < spacing_sq:
                            too_close = True
                            break
            if not too_close:
                sampled.append((px, py))
                grid[(gx, gy)] = (px, py)

        self.sampled_points[spacing] = sampled
        return sampled

    def get_normal(self, x: int, y: int) -> Tuple[float, float]:
        h, w = self.mask_array.shape
        cx = max(0, min(w - 1, x))
        cy = max(0, min(h - 1, y))
        return (float(self._normal_x[cy, cx]), float(self._normal_y[cy, cx]))

    def get_edge_array(self) -> np.ndarray:
        if self._edge_array_cache is None:
            if self.edge_pixels:
                self._edge_array_cache = np.array(self.edge_pixels, dtype=np.float32)
            else:
                self._edge_array_cache = np.empty((0, 2), dtype=np.float32)
        return self._edge_array_cache

    def get_middle_of_edge_position(self, x: int, y: int, edge_width: int = 20) -> Tuple[int, int]:
        nx, ny = self.get_normal(x, y)
        mx = int(x + (-nx) * (edge_width // 2))
        my = int(y + (-ny) * (edge_width // 2))
        w, h = self.mask_size
        return max(0, min(w - 1, mx)), max(0, min(h - 1, my))


# ============================================
# 🎨 COLOR & BRUSH MATH (100% IDENTICAL)
# ============================================

def fast_colorize_brush(brush: Image.Image, color: tuple) -> Image.Image:
    brush_array = np.array(brush)
    result = np.empty_like(brush_array)
    result[:, :, 0] = color[0]
    result[:, :, 1] = color[1]
    result[:, :, 2] = color[2]
    result[:, :, 3] = brush_array[:, :, 3]
    return Image.fromarray(result, "RGBA")


def fast_apply_opacity(image: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return image
    img_array = np.array(image)
    img_array[:, :, 3] = (img_array[:, :, 3] * opacity).astype(np.uint8)
    return Image.fromarray(img_array, "RGBA")


def lighten_color(color: tuple, amount: float) -> tuple:
    if amount <= 0:
        return color
    r = int(color[0] + (255 - color[0]) * amount)
    g = int(color[1] + (255 - color[1]) * amount)
    b = int(color[2] + (255 - color[2]) * amount)
    return (min(255, r), min(255, g), min(255, b))


def warm_color(color: tuple, amount: float) -> tuple:
    if amount <= 0:
        return color
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    base_warm = amount * 0.3
    coolness = max(0, (b - r) / 255.0)
    cyan_factor = max(0, (b + g - 2 * r) / 510.0)
    cool_factor = max(coolness, cyan_factor * 0.7)
    warm_strength = base_warm + (cool_factor * amount * 0.7)
    r_new = r + int(warm_strength * 35)
    g_new = g + int(warm_strength * 10)
    b_new = b - int(warm_strength * 20)
    return (min(255, max(0, r_new)), min(255, max(0, g_new)), min(255, max(0, b_new)))


def apply_brush_color_adjustments(color: tuple, cfg: dict) -> tuple:
    warmth = cfg.get("BRUSH_WARMTH", 0.0)
    if warmth > 0:
        color = warm_color(color, warmth)
    if cfg.get("DARK_BACKGROUND_ENABLED", False):
        lighten = cfg.get("BRUSH_LIGHTEN_AMOUNT", 0.0)
        if lighten > 0:
            color = lighten_color(color, lighten)
    return color


def adjust_color_brightness_saturation(color: tuple, brightness: float, saturation: float) -> tuple:
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    r *= brightness
    g *= brightness
    b *= brightness
    gray = (r + g + b) / 3
    r = gray + (r - gray) * saturation
    g = gray + (g - gray) * saturation
    b = gray + (b - gray) * saturation
    return (
        int(max(0, min(255, r * 255))),
        int(max(0, min(255, g * 255))),
        int(max(0, min(255, b * 255))),
    )


def mix_colors(color1: tuple, color2: tuple, amount: float) -> tuple:
    return (
        int(color1[0] * (1 - amount) + color2[0] * amount),
        int(color1[1] * (1 - amount) + color2[1] * amount),
        int(color1[2] * (1 - amount) + color2[2] * amount),
    )


def composite_brush_at_position(layer: Image.Image, brush: Image.Image, px: int, py: int):
    brush_w, brush_h = brush.size
    layer_w, layer_h = layer.size
    src_x1 = max(0, -px)
    src_y1 = max(0, -py)
    src_x2 = min(brush_w, layer_w - px)
    src_y2 = min(brush_h, layer_h - py)
    dst_x = max(0, px)
    dst_y = max(0, py)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return
    cropped_brush = brush.crop((src_x1, src_y1, src_x2, src_y2))
    layer.alpha_composite(cropped_brush, (dst_x, dst_y))


def get_brush_rotation_angle(normal_x: float, normal_y: float, jiggle: float = 0.0, rng=None) -> float:
    angle_rad = math.atan2(-normal_y, normal_x)
    angle_deg = math.degrees(angle_rad)
    rotation = angle_deg + 270
    if jiggle > 0:
        r = rng if rng else random
        rotation += r.uniform(-180 * jiggle, 180 * jiggle)
    return rotation


def get_ellipse_brush_rotation(
    x: int, y: int, center_x: int, center_y: int,
    rotation_offset: float = 0, jiggle: float = 0.0, rng=None
) -> float:
    dx = center_x - x
    dy = -(center_y - y)
    angle_to_center = math.atan2(dy, dx)
    rotation = math.degrees(angle_to_center) + 90 + rotation_offset
    if jiggle > 0:
        r = rng if rng else random
        rotation += r.uniform(-180 * jiggle, 180 * jiggle)
    return rotation


def get_subject_average_color(subject: Image.Image) -> tuple:
    img_array = np.array(subject)
    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    visible_mask = alpha > 128
    if not np.any(visible_mask):
        return (128, 128, 128)
    return (
        int(np.mean(rgb[:, :, 0][visible_mask])),
        int(np.mean(rgb[:, :, 1][visible_mask])),
        int(np.mean(rgb[:, :, 2][visible_mask])),
    )


def get_subject_dominant_color(subject: Image.Image) -> Optional[tuple]:
    """Vectorized dominant color calculation (100% identical result to original dict loop)."""
    img_array = np.array(subject)
    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    visible_mask = alpha > 128
    if not np.any(visible_mask):
        return None
    visible_pixels = rgb[visible_mask]
    if len(visible_pixels) == 0:
        return None

    bin_size = 24
    quantized = (visible_pixels // bin_size) * bin_size + (bin_size // 2)
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)

    best_color = None
    best_count = -1
    for c, count in zip(unique_colors, counts):
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance < 40 or luminance > 220:
            continue
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        saturation = (max_c - min_c) / (max_c + 1) if max_c > 0 else 0
        if saturation > 0.20 and count > best_count:
            best_count = int(count)
            best_color = (r, g, b)

    return best_color


# ============================================
# ✂️ CUTOUT & COLOR CORRECTION
# ============================================

def trim_subject_edges(image: Image.Image, pixels: int) -> Image.Image:
    a = image.split()[3]
    a_eroded = _fast_min_filter(a, pixels * 2 + 1)
    result = image.copy()
    result.putalpha(a_eroded)
    return result


def remove_background_raw(image: Image.Image) -> Image.Image:
    """Run AI background removal and return raw RGBA before edge trim."""
    session = get_rembg_session()
    return remove(image.convert("RGBA"), session=session)


def apply_color_correction(image: Image.Image, cfg: dict) -> Image.Image:
    if not cfg.get("COLOR_CORRECTION_ENABLED", True):
        return image

    has_alpha = image.mode == "RGBA"
    if has_alpha:
        alpha = image.split()[3]
        rgb = image.convert("RGB")
    else:
        rgb = image.convert("RGB")

    # 1. Shadow / Highlight
    img = np.array(rgb).astype(np.float32)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    lum_img = Image.fromarray(lum.astype(np.uint8))
    small_lum = lum_img.resize(( max(1, lum_img.width // 4), max(1, lum_img.height // 4) ), Image.Resampling.NEAREST)
    small_blur = small_lum.filter(ImageFilter.GaussianBlur(radius=30 // 4))
    lum_blur = np.array(small_blur.resize(lum_img.size, Image.Resampling.BILINEAR)).astype(np.float32)

    shadow_mask = np.power(1.0 - (lum_blur / 255.0), 0.7)
    highlight_mask = np.clip((np.power(lum_blur / 255.0, 0.7) - 0.5) * 2, 0, 1)

    shadow_lift = cfg.get("CC_SHADOW_LIFT", 10)
    highlight_reduce = cfg.get("CC_HIGHLIGHT_REDUCE", 30)
    for c in range(3):
        img[:, :, c] = img[:, :, c] + (shadow_mask * shadow_lift) - (highlight_mask * highlight_reduce)
    result = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # 2. Vibrance
    img_v = np.array(result).astype(np.float32) / 255.0
    max_rgb = np.max(img_v, axis=2)
    min_rgb = np.min(img_v, axis=2)
    current_sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
    vibrance_mask = np.power(1.0 - current_sat, 0.5)
    gray = np.mean(img_v, axis=2, keepdims=True)
    vibrance = cfg.get("CC_VIBRANCE", 1.25)
    boost = 1.0 + (vibrance - 1.0) * vibrance_mask[:, :, np.newaxis]
    result = Image.fromarray(np.clip((gray + (img_v - gray) * boost) * 255, 0, 255).astype(np.uint8))

    # 3. Saturation, Brightness, Contrast
    result = ImageEnhance.Color(result).enhance(cfg.get("CC_SATURATION", 1.10))
    result = ImageEnhance.Brightness(result).enhance(cfg.get("CC_BRIGHTNESS", 1.08))
    result = ImageEnhance.Contrast(result).enhance(cfg.get("CC_CONTRAST", 1.02))

    # 4. Warm filter
    warm_filter = cfg.get("CC_WARM_FILTER", 0.02)
    if warm_filter > 0:
        img_wf = np.array(result).astype(np.float32)
        warm_c = np.array([255, 197, 143])
        for c in range(3):
            img_wf[:, :, c] = img_wf[:, :, c] * (1 - warm_filter) + warm_c[c] * warm_filter
        result = Image.fromarray(np.clip(img_wf, 0, 255).astype(np.uint8))

    # 5. Selective warm boost
    warm_boost = cfg.get("CC_WARM_BOOST", 2.6)
    if warm_boost > 1.0:
        img_wb = np.array(result).astype(np.float32)
        R, G, B = img_wb[:, :, 0], img_wb[:, :, 1], img_wb[:, :, 2]
        warmth = np.clip((R - B) / 255.0, 0, 1) * (1.0 - np.clip((G - R) / 255.0, 0, 1))
        b_amt = warm_boost - 1.0
        img_wb[:, :, 0] = np.clip(R + warmth * b_amt * 20, 0, 255)
        img_wb[:, :, 2] = np.clip(B - warmth * b_amt * 10, 0, 255)
        result = Image.fromarray(img_wb.astype(np.uint8))

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


# ============================================
# 🎨 OUTLINE & TEXTURE BUILDERS
# ============================================

def create_color_outline(subject: Image.Image, cfg: dict) -> Image.Image:
    edge_width = cfg.get("EDGE_SAMPLE_WIDTH", 20)
    boost = cfg.get("DARK_BRIGHT_BOOST", 2.0)

    mask = subject.split()[3]
    eroded = _fast_min_filter(mask, edge_width * 2 + 1)
    edge_array = np.clip(np.array(mask, dtype=np.int16) - np.array(eroded, dtype=np.int16), 0, 255).astype(np.uint8)

    edge_colors = subject.copy()
    r, g, b, a = edge_colors.split()
    edge_colors.putalpha(Image.fromarray(np.minimum(np.array(a), edge_array).astype(np.uint8)))

    img_array = np.array(edge_colors).astype(np.float32)
    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    lum_min, lum_max = luminance.min(), luminance.max()
    lum_normalized = (luminance - lum_min) / (lum_max - lum_min) if lum_max > lum_min else luminance * 0

    dark_mask = np.clip((0.3 - lum_normalized) * boost, 0, 1)
    bright_mask = np.clip((lum_normalized - 0.7) * boost, 0, 1)
    boost_mask = np.maximum(dark_mask, bright_mask)

    boost_mask_img = Image.fromarray((boost_mask * 255).astype(np.uint8))
    boost_mask_dilated = boost_mask_img.filter(ImageFilter.MaxFilter(5))
    boost_array = np.array(boost_mask_dilated).astype(np.float32) / 255.0
    new_alpha = np.clip(alpha + (boost_array * 200), 0, 255)

    rgb_img = edge_colors.convert("RGB")
    rgb_saturated = ImageEnhance.Color(rgb_img).enhance(1.5)
    rgb_contrasted = ImageEnhance.Contrast(rgb_saturated).enhance(1.3)
    res = rgb_contrasted.convert("RGBA")
    res.putalpha(Image.fromarray(new_alpha.astype(np.uint8)))
    return res.filter(ImageFilter.GaussianBlur(radius=3))


def get_color_from_outline_at_position(
    color_outline_array: np.ndarray,
    subject_array: np.ndarray,
    edge_cache: EdgeCache,
    canvas_x: int,
    canvas_y: int,
    offset: tuple,
    cfg: dict,
) -> tuple:
    subject_x = canvas_x - offset[0]
    subject_y = canvas_y - offset[1]
    if not edge_cache.edge_pixels:
        return (100, 80, 60)

    edge_array = edge_cache.get_edge_array()
    dx = edge_array[:, 0] - subject_x
    dy = edge_array[:, 1] - subject_y
    nearest_idx = int(np.argmin(dx * dx + dy * dy))
    nearest_x, nearest_y = edge_cache.edge_pixels[nearest_idx]

    edge_width = cfg.get("EDGE_SAMPLE_WIDTH", 20)
    sample_x, sample_y = edge_cache.get_middle_of_edge_position(nearest_x, nearest_y, edge_width)
    h, w = color_outline_array.shape[:2]
    sample_x = min(max(0, sample_x), w - 1)
    sample_y = min(max(0, sample_y), h - 1)

    color = tuple(color_outline_array[sample_y, sample_x])
    if len(color) == 4 and color[3] < 50:
        sh, sw = subject_array.shape[:2]
        fx = min(max(0, nearest_x), sw - 1)
        fy = min(max(0, nearest_y), sh - 1)
        color = tuple(subject_array[fy, fx])
        if len(color) == 4 and color[3] < 50:
            return (100, 80, 60)
    return (int(color[0]), int(color[1]), int(color[2]))


def apply_watercolor_texture_overlay(image: Image.Image, size_mult: float, opacity: float, contrast: float) -> Image.Image:
    try:
        texture = Image.open(asset_path("watercolortexture.png")).convert("L")
    except Exception:
        return image

    img_w, img_h = image.size
    tex_w = int(texture.width * size_mult)
    tex_h = int(texture.height * size_mult)
    if tex_w <= 0 or tex_h <= 0:
        return image

    texture = texture.resize((tex_w, tex_h), Image.Resampling.BILINEAR)
    tiled_texture = Image.new("L", (img_w, img_h), 128)
    for y in range(0, img_h, tex_h):
        for x in range(0, img_w, tex_w):
            pw, ph = min(tex_w, img_w - x), min(tex_h, img_h - y)
            if pw < tex_w or ph < tex_h:
                tiled_texture.paste(texture.crop((0, 0, pw, ph)), (x, y))
            else:
                tiled_texture.paste(texture, (x, y))

    if contrast != 1.0:
        tiled_texture = ImageEnhance.Contrast(tiled_texture).enhance(contrast)

    alpha = image.split()[3] if image.mode == "RGBA" else None
    rgb = image.convert("RGB")
    base_array = np.array(rgb).astype(np.float32) / 255.0
    overlay_3ch = np.stack([np.array(tiled_texture).astype(np.float32) / 255.0] * 3, axis=-1)

    res_arr = np.where(base_array < 0.5, 2 * base_array * overlay_3ch, 1 - 2 * (1 - base_array) * (1 - overlay_3ch))
    res_arr = base_array * (1 - opacity) + res_arr * opacity
    res = Image.fromarray((np.clip(res_arr, 0, 1) * 255).astype(np.uint8), "RGB")
    if alpha is not None:
        res = res.convert("RGBA")
        res.putalpha(alpha)
    return res


def create_paper_texture_layer(canvas_size: tuple, cfg: dict) -> Image.Image:
    if cfg.get("DARK_BACKGROUND_ENABLED", False):
        bg_color = cfg.get("DARK_BACKGROUND_COLOR", (5, 5, 5))
        return Image.new("RGBA", canvas_size, (*bg_color, 255))

    layer = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    if not cfg.get("PAPER_TEXTURE_ENABLED", True):
        return layer

    try:
        tile = Image.open(asset_path("papertexture.png")).convert("RGBA")
    except Exception:
        return layer

    tile_w, tile_h = tile.size
    canvas_w, canvas_h = canvas_size
    texture_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for y in range(0, canvas_h, tile_h):
        for x in range(0, canvas_w, tile_w):
            pw, ph = min(tile_w, canvas_w - x), min(tile_h, canvas_h - y)
            if pw < tile_w or ph < tile_h:
                c_tile = tile.crop((0, 0, pw, ph))
                texture_layer.paste(c_tile, (x, y), c_tile)
            else:
                texture_layer.paste(tile, (x, y), tile)

    warmth = cfg.get("PAPER_TEXTURE_WARMTH", 0.2)
    if warmth > 0:
        img = np.array(texture_layer).astype(np.float32)
        warm_color = np.array([255, 240, 220])
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - warmth * 0.3) + warm_color[c] * (warmth * 0.3)
        texture_layer = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    texture_layer = fast_apply_opacity(texture_layer, cfg.get("PAPER_TEXTURE_OPACITY", 0.4))
    layer.alpha_composite(texture_layer)
    return layer


# ============================================
# 🖌️ OIL PAINT ENGINE
# ============================================

def apply_oil_paint_effect(image: Image.Image, cfg: dict) -> Image.Image:
    original_size = image.size
    scale = cfg.get("OIL_DETAIL_SCALE", 0.85)
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)

    # 1. Shadow / Highlight
    img = np.array(image).astype(np.float32)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    lum_blur = np.array(Image.fromarray(lum.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=25)))
    shadow_mask = np.power(1.0 - lum_blur / 255.0, 0.5)
    highlight_mask = np.clip((lum_blur / 255.0 - 0.5) * 2, 0, 1)
    shadow_lift = shadow_mask * (cfg.get("OIL_SHADOW_AMOUNT", 60) / 100.0) * 100
    highlight_reduce = highlight_mask * (cfg.get("OIL_HIGHLIGHT_AMOUNT", 180) / 100.0) * 80
    for c in range(3):
        img[:, :, c] = img[:, :, c] + shadow_lift - highlight_reduce
    res = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # 2. Levels
    img_l = np.array(res).astype(np.float32)
    black = cfg.get("OIL_LEVELS_BLACK", 40)
    white = cfg.get("OIL_LEVELS_WHITE", 220)
    gamma = cfg.get("OIL_LEVELS_GAMMA", 0.91)
    img_l = np.clip((img_l - black) / max(1, white - black) * 255, 0, 255)
    img_l = 255 * np.power(img_l / 255, 1 / gamma)
    res = Image.fromarray(np.clip(img_l, 0, 255).astype(np.uint8))

    # 3. Oil Flow
    img_f = np.array(res).astype(np.float32)
    h, w = img_f.shape[:2]
    channels = img_f.shape[2]
    blur_sigma = 1.5
    length = 10
    blurred = res.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    gray = np.array(blurred.convert("L")).astype(np.float32)
    Iy, Ix = np.gradient(gray)
    Ixx = ndimage.gaussian_filter(Ix * Ix, blur_sigma)
    Iyy = ndimage.gaussian_filter(Iy * Iy, blur_sigma)
    Ixy = ndimage.gaussian_filter(Ix * Iy, blur_sigma)
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy + 1e-10)
    flow_x = np.cos(theta + np.pi / 2)
    flow_y = np.sin(theta + np.pi / 2)
    mag = np.sqrt(flow_x ** 2 + flow_y ** 2) + 1e-10
    flow_x = ndimage.gaussian_filter(flow_x / mag, 2)
    flow_y = ndimage.gaussian_filter(flow_y / mag, 2)

    out = np.zeros((h, w, channels), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    steps = 4
    step_mult = 2.0
    for direction in [1, -1]:
        x = x_coords.copy()
        y = y_coords.copy()
        for step in range(steps):
            wt = 1.0 - step / (steps * 2)
            xi = np.clip(x.astype(np.int32), 0, w - 1)
            yi = np.clip(y.astype(np.int32), 0, h - 1)
            out += img_f[yi, xi] * wt
            wsum += wt
            x += flow_x[yi, xi] * direction * (length / steps) * step_mult
            y += flow_y[yi, xi] * direction * (length / steps) * step_mult
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
    for c in range(channels):
        out[:, :, c] /= (wsum + 1e-10)
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    # 4. Oil Enhance (High-pass + Unsharp)
    img_e = np.array(res).astype(np.float32)
    blurred_e = np.array(res.filter(ImageFilter.GaussianBlur(radius=3))).astype(np.float32)
    high_pass = np.clip(img_e - blurred_e + 128, 0, 255)
    base, overlay = img_e / 255.0, high_pass / 255.0
    blended = np.where(base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
    hp_strength = cfg.get("OIL_HIGH_PASS_STRENGTH", 0.5)
    img_e = (base + (blended - base) * hp_strength) * 255
    blurred_u = np.array(
        Image.fromarray(np.clip(img_e, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.5))
    ).astype(np.float32)
    img_e = img_e + (img_e - blurred_u) * cfg.get("OIL_UNSHARP_AMOUNT", 1.1)
    res = Image.fromarray(np.clip(img_e, 0, 255).astype(np.uint8))

    contrast = cfg.get("OIL_FINAL_CONTRAST", 1.0)
    if contrast > 1.0:
        res = ImageEnhance.Contrast(res).enhance(contrast)
    if scale < 1.0:
        res = res.resize(original_size, Image.Resampling.BILINEAR)
    return res


def create_oil_paint_layer(input_image: Image.Image, subject_mask: Image.Image, canvas_size: tuple, offset: tuple, cfg: dict) -> Tuple[Image.Image, Image.Image]:
    """Returns (oil_layer_with_preset_opacity, oil_layer_full_100_opacity_for_live_slider)."""
    oil_painted = apply_oil_paint_effect(input_image.convert("RGB"), cfg).convert("RGBA")
    trim = cfg.get("SUBJECT_EDGE_TRIM", 8)
    if trim > 0:
        trimmed_mask = _fast_min_filter(subject_mask, trim * 2 + 1)
        oil_painted.putalpha(trimmed_mask)
    else:
        oil_painted.putalpha(subject_mask)

    canvas_full = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas_full.paste(oil_painted, offset, oil_painted)
    canvas_preset = fast_apply_opacity(canvas_full.copy(), cfg.get("OIL_PAINT_OPACITY", 0.40))
    return canvas_preset, canvas_full


# ============================================
# 🖌️ INDIVIDUAL BRUSH LAYER GENERATORS
# ============================================

def get_ellipse_points(canvas_size: tuple, spacing: int, bottom_only: bool = False, bottom_percent: float = 0.25, inset: int = 0) -> list:
    width, height = canvas_size
    center_x, center_y = width // 2, height // 2
    radius_x = max(10, (width // 2) - inset)
    radius_y = max(10, (height // 2) - inset)
    circumference = 2 * math.pi * max(radius_x, radius_y)
    num_points = max(1, int(circumference / spacing))
    bottom_threshold = height * (1 - bottom_percent)
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        x = int(center_x + radius_x * math.cos(angle))
        y = int(center_y + radius_y * math.sin(angle))
        if not bottom_only or y >= bottom_threshold:
            points.append((x, y, angle))
    return points


def blend_ellipse_with_outline(ellipse_points: list, edge_pixels: list, blend_amount: float = 0.25) -> list:
    if not edge_pixels or not ellipse_points or blend_amount <= 0:
        return ellipse_points
    edge_array = np.array(edge_pixels, dtype=np.float32)
    result = []
    for ex, ey, angle in ellipse_points:
        dx = edge_array[:, 0] - ex
        dy = edge_array[:, 1] - ey
        nearest_idx = int(np.argmin(dx * dx + dy * dy))
        nx, ny = edge_pixels[nearest_idx]
        result.append((int(ex + (nx - ex) * blend_amount), int(ey + (ny - ey) * blend_amount), angle))
    return result


def get_bottom_outline_points(edge_cache: EdgeCache, spacing: int, left_margin: int = 50, right_margin: int = 50, inward_offset: int = 0) -> list:
    if not edge_cache.edge_pixels:
        return []
    all_x = [p[0] for p in edge_cache.edge_pixels]
    min_x, max_x = min(all_x), max(all_x)
    bottom_by_x = {}
    for x, y in edge_cache.edge_pixels:
        if x not in bottom_by_x or y > bottom_by_x[x]:
            bottom_by_x[x] = y
    effective_min_x = min_x + left_margin
    effective_max_x = max_x - right_margin
    bottom_pixels = [(x, y) for x, y in bottom_by_x.items() if effective_min_x <= x <= effective_max_x]
    if not bottom_pixels:
        return []
    bottom_pixels.sort(key=lambda p: p[0])
    sampled = [bottom_pixels[0]]
    for pixel in bottom_pixels[1:]:
        if not any((pixel[0] - sp[0]) ** 2 + (pixel[1] - sp[1]) ** 2 < spacing * spacing for sp in sampled):
            sampled.append(pixel)
    result = []
    for x, y in sampled:
        nx, ny = edge_cache.get_normal(x, y)
        result.append((x + int(-nx * inward_offset), y + int(-ny * inward_offset), nx, ny))
    return result


def build_bg_brush_layer(subject, color_outline, edge_cache: EdgeCache, canvas_size, offset, average_color, dominant_color, cfg) -> Image.Image:
    if not cfg.get("BG_BRUSH_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 200)
    points = edge_cache.get_sampled_points(cfg.get("BG_BRUSH_SPACING", 220))
    brush = load_brush(cfg.get("BG_BRUSH_FILE", "bgbrush.png"))
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    co_arr = np.array(color_outline)
    sub_arr = np.array(subject)

    for x, y in points:
        nx, ny = edge_cache.get_normal(x, y)
        sx, sy = edge_cache.get_middle_of_edge_position(x, y, cfg.get("EDGE_SAMPLE_WIDTH", 20))
        color = tuple(co_arr[sy, sx])
        if len(color) == 4 and color[3] < 50:
            color = tuple(sub_arr[sy, sx])
            if len(color) == 4 and color[3] < 50:
                continue
        outline_color = (int(color[0]), int(color[1]), int(color[2]))
        if cfg.get("BG_BRUSH_MIX_AVERAGE", True) and average_color is not None:
            outline_color = mix_colors(outline_color, average_color, cfg.get("BG_BRUSH_MIX_AMOUNT", 0.2))
        if cfg.get("BG_BRUSH_MIX_DOMINANT", False) and dominant_color is not None:
            outline_color = mix_colors(outline_color, dominant_color, cfg.get("BG_BRUSH_MIX_DOMINANT_AMOUNT", 0.3))
        outline_color = apply_brush_color_adjustments(outline_color, cfg)
        adjusted_color = adjust_color_brightness_saturation(
            outline_color, cfg.get("BG_BRUSH_BRIGHTNESS", 0.6), cfg.get("BG_BRUSH_SATURATION", 1.0)
        )
        rot_angle = get_brush_rotation_angle(nx, ny, cfg.get("BRUSH_DIRECTION_JIGGLE", 0.8), rng)
        varied_size = max(20, int(cfg.get("BG_BRUSH_SIZE", 1100) * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(rotated, adjusted_color)
        composite_brush_at_position(layer, colored, x + offset[0] - rotated.width // 2, y + offset[1] - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get("BG_BRUSH_OPACITY", 0.6))


def build_sharp_brush_layer(subject, color_outline, edge_cache: EdgeCache, canvas_size, offset, cfg) -> Image.Image:
    if not cfg.get("SHARP_BRUSH_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 150)
    points = edge_cache.get_sampled_points(cfg.get("SHARP_BRUSH_SPACING", 500))
    brush = load_brush(cfg.get("SHARP_BRUSH_FILE", "sharpbrush.png"))
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    co_arr = np.array(color_outline)
    sub_arr = np.array(subject)

    for x, y in points:
        sx, sy = edge_cache.get_middle_of_edge_position(x, y, cfg.get("EDGE_SAMPLE_WIDTH", 20))
        color = tuple(co_arr[sy, sx])
        if len(color) == 4 and color[3] < 50:
            color = tuple(sub_arr[sy, sx])
            if len(color) == 4 and color[3] < 50:
                continue
        brush_color = apply_brush_color_adjustments((int(color[0]), int(color[1]), int(color[2])), cfg)
        rot_angle = rng.uniform(0, 360)
        varied_size = max(20, int(cfg.get("SHARP_BRUSH_SIZE", 500) * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(rotated, brush_color)
        composite_brush_at_position(layer, colored, x + offset[0] - rotated.width // 2, y + offset[1] - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get("SHARP_BRUSH_OPACITY", 0.1))


def build_blur_and_mask_layers(subject: Image.Image, edge_cache: EdgeCache, canvas_size: tuple, offset: tuple, cfg: dict) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Returns (composited_masked_blur_layer, unmasked_blur_bg, mask_alpha_layer)."""
    if not cfg.get("MASK_BRUSH_ENABLED", True):
        empty = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        return empty, empty, Image.new("L", canvas_size, 0)

    # 1. Layered Blur Background
    scale_down = 4
    small_canvas = (canvas_size[0] // scale_down, canvas_size[1] // scale_down)
    layers = Image.new("RGBA", small_canvas, (0, 0, 0, 0))
    small_subject = subject.resize((max(1, subject.width // scale_down), max(1, subject.height // scale_down)), Image.Resampling.BILINEAR)
    w, h = small_subject.size
    cx, cy = small_canvas[0] // 2, small_canvas[1] // 2
    num_layers = cfg.get("NUM_BLUR_LAYERS", 7)
    scale_step = cfg.get("LAYER_SCALE_STEP", 0.3)
    for i in range(num_layers, 0, -1):
        sc = 1.0 + (i * scale_step)
        nw, nh = int(w * sc), int(h * sc)
        lyr = small_subject.resize((nw, nh), Image.Resampling.BILINEAR)
        layers.paste(lyr, (cx - nw // 2, cy - nh // 2), lyr)

    blur_amount = max(1, cfg.get("BLUR_AMOUNT", 25) // scale_down)
    blurred = layers.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    rgb = blurred.convert("RGB")
    alpha = blurred.split()[3]
    rgb = ImageEnhance.Color(rgb).enhance(cfg.get("SATURATION_BOOST", 1.1))
    rgb = ImageEnhance.Contrast(rgb).enhance(cfg.get("CONTRAST_BOOST", 1.4))
    small_result = rgb.convert("RGBA")
    small_result.putalpha(alpha)
    blur_bg = small_result.resize(canvas_size, Image.Resampling.BILINEAR)

    if cfg.get("BLUR_BG_TEXTURE_ENABLED", False):
        blur_bg = apply_watercolor_texture_overlay(
            blur_bg,
            cfg.get("BLUR_BG_TEXTURE_SIZE", 1.0),
            cfg.get("BLUR_BG_TEXTURE_OPACITY", 0.8),
            cfg.get("BLUR_BG_TEXTURE_CONTRAST", 1.2),
        )

    # 2. Mask Brush Layer (fast slice maximum — 100% identical to full-canvas maximum)
    rng = random.Random(cfg.get("RANDOM_SEED", 42))
    points = edge_cache.get_sampled_points(cfg.get("MASK_BRUSH_SPACING", 200))
    brush = load_brush(cfg.get("MASK_BRUSH_FILE", "square.png"))
    mask_arr = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    base_size = cfg.get("MASK_BRUSH_SIZE", 900)
    jiggle = cfg.get("BRUSH_DIRECTION_JIGGLE", 0.8)

    for x, y in points:
        nx, ny = edge_cache.get_normal(x, y)
        rot_angle = get_brush_rotation_angle(nx, ny, jiggle, rng)
        varied_size = max(20, int(base_size * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        brush_alpha = np.array(rotated.split()[3])
        bh, bw = brush_alpha.shape
        px = x + offset[0] - bw // 2
        py = y + offset[1] - bh // 2

        src_x1, src_y1 = max(0, -px), max(0, -py)
        src_x2, src_y2 = min(bw, canvas_size[0] - px), min(bh, canvas_size[1] - py)
        dst_x, dst_y = max(0, px), max(0, py)
        if src_x1 < src_x2 and src_y1 < src_y2:
            sub_slice = mask_arr[dst_y:dst_y + (src_y2 - src_y1), dst_x:dst_x + (src_x2 - src_x1)]
            np.maximum(sub_slice, brush_alpha[src_y1:src_y2, src_x1:src_x2], out=sub_slice)

    opacity = cfg.get("MASK_BRUSH_OPACITY", 0.5)
    if opacity < 1.0:
        mask_arr = (mask_arr.astype(np.float32) * opacity).astype(np.uint8)
    mask_canvas = Image.fromarray(mask_arr, "L")

    # Combine into a standalone RGBA layer that composites identically to result.paste(blur_bg, (0,0), mask_canvas)
    combined = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    combined.paste(blur_bg, (0, 0), mask_canvas)
    return combined, blur_bg, mask_canvas


def build_color_brush_layer(subject, color_outline, edge_cache: EdgeCache, canvas_size, offset, cfg) -> Image.Image:
    if not cfg.get("COLOR_BRUSH_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 100)
    points = edge_cache.get_sampled_points(cfg.get("COLOR_BRUSH_SPACING", 80))
    brush = load_brush(cfg.get("COLOR_BRUSH_FILE", "brush2.png"))
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    co_arr = np.array(color_outline)
    sub_arr = np.array(subject)

    for x, y in points:
        nx, ny = edge_cache.get_normal(x, y)
        sx, sy = edge_cache.get_middle_of_edge_position(x, y, cfg.get("EDGE_SAMPLE_WIDTH", 20))
        color = tuple(co_arr[sy, sx])
        if len(color) == 4 and color[3] < 50:
            color = tuple(sub_arr[sy, sx])
            if len(color) == 4 and color[3] < 50:
                continue
        brush_color = apply_brush_color_adjustments((int(color[0]), int(color[1]), int(color[2])), cfg)
        rot_angle = get_brush_rotation_angle(nx, ny, cfg.get("BRUSH_DIRECTION_JIGGLE", 0.8), rng)
        varied_size = max(20, int(cfg.get("COLOR_BRUSH_SIZE", 800) * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(rotated, brush_color)
        composite_brush_at_position(layer, colored, x + offset[0] - rotated.width // 2, y + offset[1] - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get("COLOR_BRUSH_OPACITY", 0.3))


def build_dots_layer(edge_cache: EdgeCache, canvas_size, offset, brush_file, size, spacing, opacity, seed_offset, cfg, color=None) -> Image.Image:
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + seed_offset)
    points = edge_cache.get_sampled_points(spacing)
    brush = load_brush(brush_file)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for x, y in points:
        rot_angle = rng.uniform(0, 360)
        varied_size = max(20, int(size * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        if color is not None:
            rotated = fast_colorize_brush(rotated, color)
        composite_brush_at_position(layer, rotated, x + offset[0] - rotated.width // 2, y + offset[1] - rotated.height // 2)
    return fast_apply_opacity(layer, opacity)


def build_glow_layer(subject: Image.Image, canvas_size: tuple, offset: tuple, cfg: dict) -> Image.Image:
    if not cfg.get("GLOW_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    mask = subject.split()[3]
    glow_width = cfg.get("GLOW_WIDTH", 7)
    glow_blur = cfg.get("GLOW_BLUR", 12)
    glow_opacity = cfg.get("GLOW_OPACITY", 0.4)
    glow_color = cfg.get("GLOW_COLOR", (255, 255, 255))

    dilated = _fast_max_filter(mask, glow_width * 2 + 1)
    glow_array = np.clip(np.array(dilated, dtype=np.int16) - np.array(mask, dtype=np.int16), 0, 255).astype(np.uint8)
    glow = Image.new("RGBA", subject.size, (*glow_color, 0))
    glow.putalpha(Image.fromarray(glow_array))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=glow_blur))
    glow = fast_apply_opacity(glow, glow_opacity)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(glow, offset)
    return canvas


def build_shadow_brush_layer(edge_cache: EdgeCache, canvas_size: tuple, offset: tuple, cfg: dict) -> Image.Image:
    if not cfg.get("SHADOW_BRUSH_ENABLED", False):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 800)
    points = get_bottom_outline_points(
        edge_cache,
        cfg.get("SHADOW_BRUSH_SPACING", 100),
        cfg.get("SHADOW_BRUSH_LEFT_MARGIN", 50),
        cfg.get("SHADOW_BRUSH_RIGHT_MARGIN", 50),
        cfg.get("SHADOW_BRUSH_INWARD_OFFSET", 20),
    )
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if not points:
        return layer
    brush = load_brush("bgbrush.png")
    base_size = cfg.get("SHADOW_BRUSH_SIZE", 300)
    size_jitter = cfg.get("SHADOW_BRUSH_SIZE_JITTER", 0.2)
    rot_offset = cfg.get("SHADOW_BRUSH_ROTATION_OFFSET", 0)
    rot_jitter = cfg.get("SHADOW_BRUSH_ROTATION_JITTER", 0.1)
    gray_val = int(255 * cfg.get("SHADOW_BRUSH_LIGHTNESS", 0.0))
    shadow_color = (gray_val, gray_val, gray_val)
    end_turn = cfg.get("SHADOW_BRUSH_END_TURN_ANGLE", 45)

    all_x = [p[0] for p in points]
    min_x, max_x = min(all_x), max(all_x)
    x_range = max(1, max_x - min_x)
    left_thresh = min_x + x_range * 0.15
    right_thresh = max_x - x_range * 0.15

    for x, y, nx, ny in points:
        rot_angle = rot_offset + rng.uniform(-180 * rot_jitter, 180 * rot_jitter)
        if x <= left_thresh:
            t = 1.0 - (x - min_x) / max(1e-5, left_thresh - min_x)
            rot_angle += end_turn * t
        elif x >= right_thresh:
            t = (x - right_thresh) / max(1e-5, max_x - right_thresh)
            rot_angle -= end_turn * t
        varied_size = max(20, int(base_size * rng.uniform(1 - size_jitter, 1 + size_jitter)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(rotated, shadow_color)
        composite_brush_at_position(layer, colored, x + offset[0] - rotated.width // 2, y + offset[1] - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get("SHADOW_BRUSH_OPACITY", 0.5))


def build_colordrop_layer(edge_cache: EdgeCache, canvas_size: tuple, offset: tuple, cfg: dict) -> Image.Image:
    if not cfg.get("COLORDROP_BRUSH_ENABLED", False):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 850)
    points = get_bottom_outline_points(
        edge_cache,
        cfg.get("COLORDROP_BRUSH_SPACING", 150),
        cfg.get("COLORDROP_BRUSH_LEFT_MARGIN", 80),
        cfg.get("COLORDROP_BRUSH_RIGHT_MARGIN", 80),
        cfg.get("COLORDROP_BRUSH_INWARD_OFFSET", 0),
    )
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    brush = load_brush("colordrop.png")
    base_size = cfg.get("COLORDROP_BRUSH_SIZE", 200)
    size_jitter = cfg.get("COLORDROP_BRUSH_SIZE_JITTER", 0.3)
    gray_val = int(255 * cfg.get("COLORDROP_BRUSH_LIGHTNESS", 0.0))
    drop_color = (gray_val, gray_val, gray_val)

    for x, y, nx, ny in points:
        varied_size = max(20, int(base_size * rng.uniform(1 - size_jitter, 1 + size_jitter)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(sized, drop_color)
        composite_brush_at_position(layer, colored, x + offset[0] - sized.width // 2, y + offset[1])

    return fast_apply_opacity(layer, cfg.get("COLORDROP_BRUSH_OPACITY", 0.4))


def build_edge_curve_layer(input_image: Image.Image, subject_mask: Image.Image, canvas_size: tuple, offset: tuple, cfg: dict) -> Image.Image:
    if not cfg.get("EDGE_CURVE_ENABLED", False):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    ec_settings = {
        "dark_threshold": 60, "light_threshold": 170,
        "mask_smooth_radius": 6, "mask_open_close_size": 0.2, "mask_final_threshold": 0.3,
        "detail_enabled": True, "detail_min_contrast": 1.2, "detail_neighborhood": 0.2, "detail_erode_smooth": 1,
        "dark_blur_radius": 1, "dark_blur_passes": 4,
        "mid_blur_radius": 12, "mid_blur_passes": 4,
        "light_blur_radius": 6, "light_blur_passes": 4,
        "dark_darken_amount": 0.3, "light_lighten_amount": 0.2, "mid_adjustment": 0.0,
        "texture_enabled": True,
        "texture_file": asset_path("watercolortexture.png"),
        "texture_opacity": cfg.get("EDGE_CURVE_TEXTURE_OPACITY", 0.5),
        "texture_scale": cfg.get("EDGE_CURVE_TEXTURE_SCALE", 0.8),
        "brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
    }
    outputs = apply_edge_curve_effect(input_image.convert("RGB"), ec_settings)
    effected = outputs["result"].convert("RGBA")
    trim = cfg.get("SUBJECT_EDGE_TRIM", 8)
    if trim > 0:
        effected.putalpha(_fast_min_filter(subject_mask, trim * 2 + 1))
    else:
        effected.putalpha(subject_mask)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.paste(effected, offset, effected)
    return fast_apply_opacity(canvas, cfg.get("EDGE_CURVE_OPACITY", 0.5))


def build_dark_ellipse_layer(subject, color_outline, edge_cache: EdgeCache, canvas_size, offset, cfg) -> Image.Image:
    if not cfg.get("DARK_ELLIPSE_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 600)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    points = get_ellipse_points(
        canvas_size, cfg.get("DARK_ELLIPSE_SPACING", 200),
        bottom_only=True, bottom_percent=cfg.get("DARK_ELLIPSE_BOTTOM_PERCENT", 0.15)
    )
    brush = load_brush("bgbrush.png")
    cx, cy = canvas_size[0] // 2, canvas_size[1] // 2
    base_size = cfg.get("DARK_ELLIPSE_SIZE", 400)
    rot_offset = cfg.get("DARK_ELLIPSE_ROTATION_OFFSET", 180)
    jiggle = cfg.get("DARK_ELLIPSE_JIGGLE", 0.2)
    co_arr = np.array(color_outline)
    sub_arr = np.array(subject)

    for x, y, angle in points:
        color = get_color_from_outline_at_position(co_arr, sub_arr, edge_cache, x, y, offset, cfg)
        rot_angle = get_ellipse_brush_rotation(x, y, cx, cy, rot_offset, jiggle, rng)
        varied_size = max(20, int(base_size * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        colored = fast_colorize_brush(rotated, color)
        composite_brush_at_position(layer, colored, x - rotated.width // 2, y - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get("DARK_ELLIPSE_OPACITY", 0.2))


def build_ellipse_brush_layer(subject, color_outline, edge_cache: EdgeCache, canvas_size, offset, cfg, is_outer: bool = False) -> Image.Image:
    prefix = "OUTER_WHITE_BRUSH" if is_outer else "WHITE_BRUSH"
    if not cfg.get(f"{prefix}_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    seed_offset = 550 if is_outer else 500
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + seed_offset)
    ellipse_mode = cfg.get("ELLIPSE_BRUSH_MODE", "white")
    brush_file = cfg.get("ELLIPSE_BRUSH_FILE", "bgbrush.png") if ellipse_mode == "dark" else "whitebrush.png"
    brush = load_brush(brush_file)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    spacing = cfg.get(f"{prefix}_SPACING", 250 if is_outer else 200)
    inset = cfg.get(f"{prefix}_ELLIPSE_INSET", -200 if is_outer else -40)
    points = get_ellipse_points(canvas_size, spacing, inset=inset)

    if edge_cache.edge_pixels and offset is not None:
        canvas_edges = [(x + offset[0], y + offset[1]) for x, y in edge_cache.edge_pixels]
        blend = cfg.get(f"{prefix}_OUTLINE_BLEND", 0.0 if is_outer else 0.25)
        points = blend_ellipse_with_outline(points, canvas_edges, blend)

    cx, cy = canvas_size[0] // 2, canvas_size[1] // 2
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    co_arr = np.array(color_outline) if color_outline is not None else None
    sub_arr = np.array(subject) if subject is not None else None

    base_size = cfg.get(f"{prefix}_SIZE", 500)
    rot_offset = cfg.get(f"{prefix}_ROTATION_OFFSET", 180)
    jiggle = cfg.get(f"{prefix}_JIGGLE", 0.2 if is_outer else 0.5)
    falloff = cfg.get(f"{prefix}_OPACITY_FALLOFF", 0.0 if is_outer else 0.8)

    for x, y, angle in points:
        rot_angle = get_ellipse_brush_rotation(x, y, cx, cy, rot_offset, jiggle, rng)
        varied_size = max(20, int(base_size * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)

        if ellipse_mode == "dark":
            rotated = fast_colorize_brush(rotated, (0, 0, 0))
        elif ellipse_mode == "colored" and sub_arr is not None and co_arr is not None:
            c = get_color_from_outline_at_position(co_arr, sub_arr, edge_cache, x, y, offset, cfg)
            rotated = fast_colorize_brush(rotated, lighten_color(c, 0.5))

        dist_ratio = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
        pt_op = (1.0 - falloff) + (falloff * dist_ratio)
        b_op = fast_apply_opacity(rotated, pt_op)
        composite_brush_at_position(layer, b_op, x - rotated.width // 2, y - rotated.height // 2)

    return fast_apply_opacity(layer, cfg.get(f"{prefix}_OPACITY", 0.6))


def build_small_white_dots_layer(edge_cache: EdgeCache, canvas_size: tuple, offset: tuple, cfg: dict) -> Image.Image:
    if not cfg.get("SMALL_WHITE_DOTS_ENABLED", True):
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    rng = random.Random(cfg.get("RANDOM_SEED", 42) + 700)
    points = edge_cache.get_sampled_points(cfg.get("SMALL_WHITE_DOTS_SPACING", 500))
    brush = load_brush("whitedots.png")
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    base_size = cfg.get("SMALL_WHITE_DOTS_SIZE", 150)
    jitter = cfg.get("SMALL_WHITE_DOTS_DISTANCE_JITTER", -50)

    avoid_bottom = cfg.get("SMALL_WHITE_DOTS_AVOID_BOTTOM", False)
    bottom_coords = []
    bottom_margin = cfg.get("SMALL_WHITE_DOTS_BOTTOM_MARGIN", 50)
    if avoid_bottom:
        b_pts = get_bottom_outline_points(edge_cache, spacing=20, left_margin=0, right_margin=0, inward_offset=0)
        bottom_coords = [(p[0], p[1]) for p in b_pts]

    for x, y in points:
        if avoid_bottom and bottom_coords:
            if any((x - bx) ** 2 + (y - by) ** 2 < bottom_margin ** 2 for bx, by in bottom_coords):
                continue
        nx, ny = edge_cache.get_normal(x, y)
        j_amt = rng.uniform(-jitter, jitter)
        jx = x + nx * j_amt
        jy = y + ny * j_amt
        rot_angle = rng.uniform(0, 360)
        varied_size = max(10, int(base_size * rng.uniform(0.7, 1.3)))
        sized = brush.resize((varied_size, varied_size), Image.Resampling.BILINEAR)
        rotated = sized.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)
        composite_brush_at_position(
            layer, rotated,
            int(jx + offset[0] - rotated.width // 2),
            int(jy + offset[1] - rotated.height // 2),
        )

    return fast_apply_opacity(layer, cfg.get("SMALL_WHITE_DOTS_OPACITY", 0.4))


# ============================================
# 🚀 MASTER STUDIO PIPELINE
# ============================================

def generate_full_studio_payload(
    original_image: Image.Image,
    preset_key: str = "animals",
    custom_mask: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Runs the complete watercolor pipeline and returns:
      1. 100% pixel-identical final composite image
      2. Every individual layer (under-subject, subject, and over-subject) as RGBA PNGs
      3. Complete edge/normal/color stroke geometry so the browser can re-render any brush layer at 60 FPS
    """
    t0 = time.time()
    cfg = get_preset_copy(preset_key)

    output_width = cfg.get("OUTPUT_WIDTH", 1260)
    output_height = cfg.get("OUTPUT_HEIGHT", 1782)
    canvas_size = (output_width, output_height)

    # 1. Cutout & Mask
    original_rgba = original_image.convert("RGBA")
    if custom_mask is not None:
        mask_clean = custom_mask.convert("L").resize(original_rgba.size, Image.Resampling.BILINEAR)
        mask_clean = mask_clean.filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.MedianFilter(size=3))
        subject_uncorr = original_rgba.copy()
        subject_uncorr.putalpha(mask_clean)
    else:
        raw_cutout = remove_background_raw(original_rgba)
        trim = cfg.get("SUBJECT_EDGE_TRIM", 8)
        subject_uncorr = trim_subject_edges(raw_cutout, trim) if trim > 0 else raw_cutout
        mask_clean = subject_uncorr.split()[3]

    offset_x = (output_width - subject_uncorr.width) // 2
    offset_y = (output_height - subject_uncorr.height) // 2
    offset = (offset_x, offset_y)

    # 2. Color Correction
    if cfg.get("COLOR_CORRECTION_ENABLED", True):
        subject = apply_color_correction(subject_uncorr, cfg)
        corrected_original = apply_color_correction(original_image.convert("RGB"), cfg)
    else:
        subject = subject_uncorr.copy()
        corrected_original = original_image.convert("RGB")

    # 3. Edge Cache & Color Palette
    edge_cache = EdgeCache(subject.split()[3])
    average_color = get_subject_average_color(subject) if cfg.get("BG_BRUSH_MIX_AVERAGE", True) else (128, 128, 128)
    dominant_color = get_subject_dominant_color(subject)

    bg_paper = create_paper_texture_layer(canvas_size, cfg)
    color_outline = create_color_outline(subject, cfg)

    subject_to_paste = subject
    if cfg.get("WATERCOLOR_TEXTURE_ENABLED", False):
        subject_to_paste = apply_watercolor_texture_overlay(
            subject,
            cfg.get("WATERCOLOR_TEXTURE_SIZE", 1.0),
            cfg.get("WATERCOLOR_TEXTURE_OPACITY", 0.6),
            cfg.get("WATERCOLOR_TEXTURE_CONTRAST", 1.2),
        )

    # Dark dots color calculation
    dark_dots_color = None
    if cfg.get("DARK_DOTS_USE_DOMINANT_COLOR", False) and dominant_color is not None:
        darken = cfg.get("DARK_DOTS_COLOR_DARKEN", 0.3)
        dark_dots_color = (
            int(dominant_color[0] * (1 - darken)),
            int(dominant_color[1] * (1 - darken)),
            int(dominant_color[2] * (1 - darken)),
        )

    # 4. Run All Independent Layer Builders in Parallel
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            "bg_brush": pool.submit(build_bg_brush_layer, subject, color_outline, edge_cache, canvas_size, offset, average_color, dominant_color, cfg),
            "sharp_brush": pool.submit(build_sharp_brush_layer, subject, color_outline, edge_cache, canvas_size, offset, cfg),
            "mask_pack": pool.submit(build_blur_and_mask_layers, subject, edge_cache, canvas_size, offset, cfg),
            "color_brush": pool.submit(build_color_brush_layer, subject, color_outline, edge_cache, canvas_size, offset, cfg),
            "white_dots": pool.submit(
                build_dots_layer, edge_cache, canvas_size, offset, "whitedots.png",
                cfg.get("WHITE_DOTS_SIZE", 600), cfg.get("WHITE_DOTS_SPACING", 400),
                cfg.get("WHITE_DOTS_OPACITY", 0.5), 300, cfg, None
            ),
            "dark_dots": pool.submit(
                build_dots_layer, edge_cache, canvas_size, offset, "darkdots.png",
                cfg.get("DARK_DOTS_SIZE", 1200), cfg.get("DARK_DOTS_SPACING", 800),
                cfg.get("DARK_DOTS_OPACITY", 0.2), 400, cfg, dark_dots_color
            ),
            "glow": pool.submit(build_glow_layer, subject, canvas_size, offset, cfg),
            "colordrop": pool.submit(build_colordrop_layer, edge_cache, canvas_size, offset, cfg),
            "shadow": pool.submit(build_shadow_brush_layer, edge_cache, canvas_size, offset, cfg),
            "oil_pack": pool.submit(create_oil_paint_layer, corrected_original, subject.split()[3], canvas_size, offset, cfg),
            "edge_curve": pool.submit(build_edge_curve_layer, corrected_original, subject.split()[3], canvas_size, offset, cfg),
            "dark_ellipse": pool.submit(build_dark_ellipse_layer, subject, color_outline, edge_cache, canvas_size, offset, cfg),
            "white_ellipse": pool.submit(build_ellipse_brush_layer, subject, color_outline, edge_cache, canvas_size, offset, cfg, False),
            "outer_white": pool.submit(build_ellipse_brush_layer, subject, color_outline, edge_cache, canvas_size, offset, cfg, True),
            "small_dots": pool.submit(build_small_white_dots_layer, edge_cache, canvas_size, offset, cfg),
        }
        parts = {k: f.result() for k, f in futures.items()}

    masked_blur_layer, blur_bg_layer, mask_alpha_layer = parts["mask_pack"]
    oil_preset_layer, oil_full_layer = parts["oil_pack"]

    # 5. Composite Bottom Pre-Accent Stack (to feed exact weight balance into accent_strokes)
    result = bg_paper.copy()
    result.alpha_composite(parts["bg_brush"])
    result.alpha_composite(parts["sharp_brush"])
    result.paste(blur_bg_layer, (0, 0), mask_alpha_layer)
    result.alpha_composite(parts["color_brush"])

    # Subject canvas layers (raw uncorrected + preset corrected)
    subject_cc_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    subject_cc_canvas.alpha_composite(subject_to_paste, offset)

    subject_raw_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    subject_raw_canvas.alpha_composite(subject_uncorr, offset)

    # 6. Accent strokes — PASS 1 (under the dots), same as HF
    def _accent(base):
        return create_accent_strokes_detailed(
            base, subject_cc_canvas, subject_cc_canvas.copy(),
            cfg={"random_seed": cfg.get("RANDOM_SEED", 42)},
            assets_dir=ASSETS_DIR,
        )

    empty = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    accent_on = cfg.get("ACCENT_STROKES_ENABLED", False)

    accent_detail = _accent(result) if accent_on else None
    if accent_on:
        result.alpha_composite(accent_detail["combined"])

    # 7. Dots + glow
    result.alpha_composite(parts["white_dots"])
    result.alpha_composite(parts["dark_dots"])
    result.alpha_composite(parts["glow"])

    # 7b. Accent strokes — PASS 2 (HF export path runs it again on top)
    accent_detail_2 = _accent(result) if accent_on else None
    if accent_on:
        result.alpha_composite(accent_detail_2["combined"])

    accent_layer = accent_detail["combined"] if accent_on else empty
    accent_layer_2 = accent_detail_2["combined"] if accent_on else empty
    if not accent_on:
        accent_detail = {"candidates": [], "spots": []}

    if cfg.get("COLORDROP_BRUSH_ENABLED", False):
        result.alpha_composite(parts["colordrop"])
    if cfg.get("SHADOW_BRUSH_ENABLED", False):
        result.alpha_composite(parts["shadow"])

    result.alpha_composite(subject_to_paste, offset)

    if cfg.get("OIL_PAINT_ENABLED", True):
        result.alpha_composite(oil_preset_layer)
    if cfg.get("EDGE_CURVE_ENABLED", False):
        result.alpha_composite(parts["edge_curve"])
    if cfg.get("DARK_ELLIPSE_ENABLED", True):
        result.alpha_composite(parts["dark_ellipse"])
    if cfg.get("WHITE_BRUSH_ENABLED", True):
        result.alpha_composite(parts["white_ellipse"])
    if cfg.get("OUTER_WHITE_BRUSH_ENABLED", True):
        result.alpha_composite(parts["outer_white"])
    if cfg.get("SMALL_WHITE_DOTS_ENABLED", True):
        result.alpha_composite(parts["small_dots"])

    # 8. Export Edge Stroke Geometry for 60 FPS Browser Re-Rendering
    co_arr = np.array(color_outline)
    sub_arr = np.array(subject)
    edge_width = cfg.get("EDGE_SAMPLE_WIDTH", 20)
    edge_Metadata = []
    for ex, ey in edge_cache.edge_pixels:
        nx, ny = edge_cache.get_normal(ex, ey)
        sx, sy = edge_cache.get_middle_of_edge_position(ex, ey, edge_width)
        c = tuple(co_arr[sy, sx])
        if len(c) == 4 and c[3] < 50:
            c = tuple(sub_arr[sy, sx])
        alpha_ok = 1 if (len(c) < 4 or c[3] >= 50) else 0
        edge_Metadata.append([
            int(ex), int(ey),
            round(nx, 4), round(ny, 4),
            int(c[0]), int(c[1]), int(c[2]), alpha_ok
        ])

    elapsed = time.time() - t0
    print(f"✅ Full studio generation completed in {elapsed:.2f}s")

    return {
        "preset_key": preset_key,
        "preset_settings": cfg,
        "canvas_size": [output_width, output_height],
        "subject_size": [subject.width, subject.height],
        "offset": [offset_x, offset_y],
        "average_color": list(average_color) if average_color else [128, 128, 128],
        "dominant_color": list(dominant_color) if dominant_color else None,
        "elapsed_sec": round(elapsed, 2),
        "composite_jpg": pil_to_base64(result, "JPEG", quality=95),
        "mask_png": pil_to_base64(mask_clean, "PNG"),
        "layers": {
            "bg_paper": pil_to_base64(bg_paper, "PNG"),
            "bg_brush": pil_to_base64(parts["bg_brush"], "PNG"),
            "sharp_brush": pil_to_base64(parts["sharp_brush"], "PNG"),
            "blur_bg": pil_to_base64(blur_bg_layer, "PNG"),
            "mask_brush": pil_to_base64(mask_alpha_layer, "PNG"),
            "color_brush": pil_to_base64(parts["color_brush"], "PNG"),
            "accent_strokes": pil_to_base64(accent_layer, "PNG"),
            "accent_strokes_2": pil_to_base64(accent_layer_2, "PNG"),
            "white_dots": pil_to_base64(parts["white_dots"], "PNG"),
            "dark_dots": pil_to_base64(parts["dark_dots"], "PNG"),
            "glow": pil_to_base64(parts["glow"], "PNG"),
            "colordrop": pil_to_base64(parts["colordrop"], "PNG"),
            "shadow": pil_to_base64(parts["shadow"], "PNG"),
            "subject_raw": pil_to_base64(subject_raw_canvas, "PNG"),
            "subject_cc": pil_to_base64(subject_cc_canvas, "PNG"),
            "oil_paint_full": pil_to_base64(oil_full_layer, "PNG"),
            "edge_curve": pil_to_base64(parts["edge_curve"], "PNG"),
            "dark_ellipse": pil_to_base64(parts["dark_ellipse"], "PNG"),
            "white_ellipse": pil_to_base64(parts["white_ellipse"], "PNG"),
            "outer_white": pil_to_base64(parts["outer_white"], "PNG"),
            "small_dots": pil_to_base64(parts["small_dots"], "PNG"),
        },
        "stroke_data": {
            # Compact array of [x, y, nx, ny, r, g, b, valid] for every edge pixel
            "edges": edge_Metadata,
            "accent_candidates": accent_detail["candidates"],
            "accent_spots": accent_detail["spots"],
        },
    }