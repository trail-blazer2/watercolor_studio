"""
🎨 Watercolor Edge Curve Effect - Tri-Tone Mask Approach (OPTIMIZED)

Usage:
    Place input.jpg and watercolor_texture.png in the same folder and run:
    python edge_curve_editor.py
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from scipy import ndimage
import os
import time


# ============================================
# SETTINGS
# ============================================

SETTINGS = {
    "dark_threshold": 60,
    "light_threshold": 170,

    "mask_smooth_radius": 12,
    "mask_open_close_size": 1,
    "mask_final_threshold": 0.5,

    "detail_enabled": True,
    "detail_min_contrast": 1.2,
    "detail_neighborhood": 0.2,
    "detail_erode_smooth": 3,

    "dark_blur_radius": 1,
    "dark_blur_passes": 4,
    "mid_blur_radius": 12,
    "mid_blur_passes": 4,
    "light_blur_radius": 6,
    "light_blur_passes": 4,
    "final_edge_smooth": 1.0,

    "dark_darken_amount": 0.4,
    "light_lighten_amount": 0.2,
    "mid_adjustment": 0.0,

    "texture_enabled": True,
    "texture_file": "watercolor_texture.png",
    "texture_opacity": 0.5,
    "texture_scale": 0.8,

    "brightness": 1.02,
    "contrast": 1.02,
    "saturation": 1.0,
}


# ============================================
# TEXTURE CACHE (load once, reuse)
# ============================================

_TEXTURE_CACHE = {}


def _get_tiled_texture(texture_path: str, scale: float,
                        img_w: int, img_h: int) -> np.ndarray:
    """Load texture once, tile to image size, cache the result."""
    cache_key = (texture_path, scale, img_w, img_h)
    if cache_key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[cache_key]

    if not os.path.exists(texture_path):
        return None

    texture = Image.open(texture_path).convert("L")
    if scale != 1.0:
        tw, th = texture.size
        texture = texture.resize((max(1, int(tw * scale)), max(1, int(th * scale))),
                                  Image.Resampling.BILINEAR)

    tex_array = np.array(texture).astype(np.float32) / 255.0
    tex_h, tex_w = tex_array.shape

    # Tile using np.tile (no Python loops)
    reps_y = (img_h + tex_h - 1) // tex_h
    reps_x = (img_w + tex_w - 1) // tex_w
    tiled = np.tile(tex_array, (reps_y, reps_x))[:img_h, :img_w]

    _TEXTURE_CACHE[cache_key] = tiled
    return tiled


# ============================================
# CORE FUNCTIONS (OPTIMIZED)
# ============================================

def compute_luminosity(image: Image.Image) -> np.ndarray:
    img_array = np.array(image, dtype=np.float32)
    return img_array[:, :, 0] * 0.299 + img_array[:, :, 1] * 0.587 + img_array[:, :, 2] * 0.114


def create_zone_masks_raw(luminosity: np.ndarray,
                           dark_thresh: int, light_thresh: int) -> tuple:
    dark = luminosity < dark_thresh
    light = luminosity > light_thresh
    mid = ~dark & ~light
    return dark.astype(np.float32), mid.astype(np.float32), light.astype(np.float32)


def smooth_mask_shape(mask: np.ndarray,
                       smooth_radius: float,
                       morph_size: float,
                       threshold: float) -> np.ndarray:
    morph_int = int(morph_size)
    if morph_int > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        binary = mask > 0.5
        binary = ndimage.binary_closing(binary, structure=struct, iterations=morph_int)
        binary = ndimage.binary_opening(binary, structure=struct, iterations=morph_int)
        opened = binary.astype(np.float32)
    else:
        opened = mask

    if smooth_radius > 0:
        blurred = ndimage.gaussian_filter(opened, sigma=smooth_radius * 0.7)
    else:
        blurred = opened

    return (blurred >= threshold).astype(np.float32)


def restore_fine_detail(smoothed_mask: np.ndarray,
                         raw_mask: np.ndarray,
                         luminosity: np.ndarray,
                         zone_type: str,
                         dark_thresh: int,
                         light_thresh: int,
                         min_contrast: float,
                         neighborhood: float,
                         erode_smooth: int) -> np.ndarray:
    """Optimized fine detail restoration."""
    raw_bool = raw_mask > 0.5
    smooth_bool = smoothed_mask > 0.5
    lost = raw_bool & ~smooth_bool

    if not np.any(lost):
        return smoothed_mask

    # Local contrast (use max of int, 1 for filter size)
    filter_size = max(1, int(neighborhood * 2 + 1))
    local_mean = ndimage.uniform_filter(luminosity, size=filter_size)
    local_diff = np.abs(luminosity - local_mean)

    local_min = ndimage.minimum_filter(luminosity, size=filter_size)
    local_max = ndimage.maximum_filter(luminosity, size=filter_size)
    local_range = local_max - local_min

    has_contrast = (local_diff >= min_contrast) | (local_range >= min_contrast * 3)

    # Zone confidence (vectorized)
    if zone_type == "dark":
        conf_a = np.clip((local_mean - luminosity) / 40.0, 0, 1)
        conf_b = np.clip((dark_thresh - luminosity) / max(dark_thresh, 1), 0, 1)
        zone_confidence = np.maximum(conf_a, conf_b)
    elif zone_type == "light":
        conf_a = np.clip((luminosity - local_mean) / 40.0, 0, 1)
        conf_b = np.clip((luminosity - light_thresh) / max(255 - light_thresh, 1), 0, 1)
        zone_confidence = np.maximum(conf_a, conf_b)
    else:
        mid_center = (dark_thresh + light_thresh) / 2.0
        mid_range = (light_thresh - dark_thresh) / 2.0
        zone_confidence = np.clip(1.0 - np.abs(luminosity - mid_center) / (mid_range + 1), 0, 1)

    restore = lost & has_contrast & (zone_confidence > 0.15)

    # Connected component rescue — only label the lost region, not the whole image
    labeled_lost, num_features = ndimage.label(lost)
    if num_features > 0:
        # Use bincount for O(n) component size calculation instead of loop
        component_sizes = np.bincount(labeled_lost.ravel())
        # component_sizes[0] is background, skip it
        large_enough = np.zeros(num_features + 1, dtype=bool)
        large_enough[1:] = component_sizes[1:] >= 20
        restore |= large_enough[labeled_lost]

    # Cleanup jaggies
    if erode_smooth > 0 and np.any(restore):
        restore_float = ndimage.gaussian_filter(restore.astype(np.float32),
                                                 sigma=erode_smooth * 0.5)
        restore = restore_float > 0.4

    result = smoothed_mask.copy()
    result[restore] = 1.0
    return result


def equivalent_single_blur_sigma(radius: int, passes: int) -> float:
    """
    N passes of Gaussian blur with radius r is equivalent to
    a single Gaussian blur with sigma = sqrt(N) * sigma_single.
    PIL GaussianBlur radius ≈ 2*sigma, so sigma ≈ radius/2.
    """
    single_sigma = radius / 2.0
    return single_sigma * np.sqrt(passes)


def fast_multi_blur(image: Image.Image, radius: int, passes: int) -> Image.Image:
    """
    Replace N blur passes with a single equivalent blur.
    Mathematically equivalent for Gaussian blur (convolution is associative).
    """
    if radius <= 0 or passes <= 0:
        return image
    equiv_sigma = equivalent_single_blur_sigma(radius, passes)
    equiv_radius = max(1, int(equiv_sigma * 2 + 0.5))
    return image.filter(ImageFilter.GaussianBlur(radius=equiv_radius))


def apply_zone_tonal_push_inplace(image_array: np.ndarray,
                                    darken: float = 0.0,
                                    lighten: float = 0.0) -> np.ndarray:
    """Apply tonal push to entire array (no mask needed since we push before compositing)."""
    if darken > 0:
        image_array *= (1.0 - darken)
    if lighten > 0:
        image_array[:] = 255.0 - (255.0 - image_array) * (1.0 - lighten)
    return image_array


def apply_texture_overlay_inplace(image_array: np.ndarray,
                                    tiled_texture: np.ndarray,
                                    opacity: float) -> np.ndarray:
    """Apply overlay blend with pre-tiled texture. Operates in-place."""
    if tiled_texture is None or opacity <= 0:
        return image_array

    base = image_array / 255.0
    tex_3ch = tiled_texture[:, :, np.newaxis]  # broadcast to 3 channels

    # Overlay blend: vectorized
    blended = np.where(base < 0.5,
                       2.0 * base * tex_3ch,
                       1.0 - 2.0 * (1.0 - base) * (1.0 - tex_3ch))

    # Mix with opacity
    np.multiply(blended - base, opacity, out=blended)
    np.add(base, blended, out=base)
    np.clip(base, 0, 1, out=base)
    np.multiply(base, 255.0, out=image_array)

    return image_array


def create_debug_mask_image(dark_mask: np.ndarray,
                             mid_mask: np.ndarray,
                             light_mask: np.ndarray) -> Image.Image:
    h, w = dark_mask.shape
    debug = np.zeros((h, w, 3), dtype=np.uint8)
    dark_bool = dark_mask > 0.5
    mid_bool = mid_mask > 0.5
    light_bool = light_mask > 0.5
    debug[dark_bool] = [30, 30, 60]
    debug[mid_bool] = [140, 130, 120]
    debug[light_bool] = [240, 235, 230]
    return Image.fromarray(debug)


# ============================================
# MAIN EFFECT
# ============================================

def apply_edge_curve_effect(image: Image.Image, settings: dict = None) -> dict:
    cfg = SETTINGS.copy()
    if settings:
        cfg.update(settings)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    has_alpha = image.mode == "RGBA"
    if has_alpha:
        alpha = image.split()[3]
        rgb = image.convert("RGB")
    else:
        rgb = image.convert("RGB")

    img_w, img_h = rgb.size

    # ---- Step 1: Luminosity + raw masks ----
    t = time.time()
    luminosity = compute_luminosity(rgb)
    dark_raw, mid_raw, light_raw = create_zone_masks_raw(
        luminosity, cfg["dark_threshold"], cfg["light_threshold"])
    print(f"   [1/7] Zones: dark={np.mean(dark_raw)*100:.1f}% mid={np.mean(mid_raw)*100:.1f}% "
          f"light={np.mean(light_raw)*100:.1f}%  ({time.time()-t:.2f}s)")

    # ---- Step 2: Smooth masks ----
    t = time.time()
    dark_smooth = smooth_mask_shape(dark_raw, cfg["mask_smooth_radius"],
                                     cfg["mask_open_close_size"], cfg["mask_final_threshold"])
    mid_smooth = smooth_mask_shape(mid_raw, cfg["mask_smooth_radius"],
                                    cfg["mask_open_close_size"], cfg["mask_final_threshold"])
    light_smooth = smooth_mask_shape(light_raw, cfg["mask_smooth_radius"],
                                      cfg["mask_open_close_size"], cfg["mask_final_threshold"])
    print(f"   [2/7] Smoothed masks  ({time.time()-t:.2f}s)")

    # ---- Step 3: Fine detail restoration ----
    t = time.time()
    if cfg.get("detail_enabled", True):
        dark_mask = restore_fine_detail(
            dark_smooth, dark_raw, luminosity, "dark",
            cfg["dark_threshold"], cfg["light_threshold"],
            cfg["detail_min_contrast"], cfg["detail_neighborhood"], cfg["detail_erode_smooth"])
        mid_mask = restore_fine_detail(
            mid_smooth, mid_raw, luminosity, "mid",
            cfg["dark_threshold"], cfg["light_threshold"],
            cfg["detail_min_contrast"], cfg["detail_neighborhood"], cfg["detail_erode_smooth"])
        light_mask = restore_fine_detail(
            light_smooth, light_raw, luminosity, "light",
            cfg["dark_threshold"], cfg["light_threshold"],
            cfg["detail_min_contrast"], cfg["detail_neighborhood"], cfg["detail_erode_smooth"])
    else:
        dark_mask, mid_mask, light_mask = dark_smooth, mid_smooth, light_smooth
    print(f"   [3/7] Detail restored  ({time.time()-t:.2f}s)")
    
    # ---- Step 3b: Feather mask edges for smooth transitions ----
    feather = cfg.get("final_edge_smooth", 2.0)
    if feather > 0:
        # Blur the binary masks — this turns the hard 0/1 stair-step edges
        # into a smooth gradient (e.g., 0 → 0.2 → 0.5 → 0.8 → 1.0)
        # Do NOT re-threshold after this — the soft edges are the point.
        dark_mask = ndimage.gaussian_filter(dark_mask, sigma=feather)
        mid_mask = ndimage.gaussian_filter(mid_mask, sigma=feather)
        light_mask = ndimage.gaussian_filter(light_mask, sigma=feather)
        print(f"   [3b]  Edge feather σ={feather}  ({time.time()-t:.2f}s)")

    # ---- Step 4: Blurred image layers (single equivalent blur) ----
    t = time.time()
    dark_blurred = fast_multi_blur(rgb, cfg["dark_blur_radius"], cfg["dark_blur_passes"])
    mid_blurred = fast_multi_blur(rgb, cfg["mid_blur_radius"], cfg["mid_blur_passes"])
    light_blurred = fast_multi_blur(rgb, cfg["light_blur_radius"], cfg["light_blur_passes"])

    dark_array = np.array(dark_blurred, dtype=np.float32)
    mid_array = np.array(mid_blurred, dtype=np.float32)
    light_array = np.array(light_blurred, dtype=np.float32)
    print(f"   [4/7] Blurred layers  ({time.time()-t:.2f}s)")

    # ---- Step 5: Tonal push (in-place, no mask needed) ----
    t = time.time()
    if cfg["dark_darken_amount"] > 0:
        apply_zone_tonal_push_inplace(dark_array, darken=cfg["dark_darken_amount"])
    if cfg["light_lighten_amount"] > 0:
        apply_zone_tonal_push_inplace(light_array, lighten=cfg["light_lighten_amount"])
    if cfg["mid_adjustment"] > 0:
        apply_zone_tonal_push_inplace(mid_array, lighten=cfg["mid_adjustment"])
    elif cfg["mid_adjustment"] < 0:
        apply_zone_tonal_push_inplace(mid_array, darken=-cfg["mid_adjustment"])
    print(f"   [5/7] Tonal push  ({time.time()-t:.2f}s)")

    # ---- Step 6: Watercolor texture (cached, in-place) ----
    t = time.time()
    if cfg["texture_enabled"]:
        assets_dir = os.path.join(os.path.dirname(script_dir), "assets")
        texture_path = os.path.join(assets_dir, os.path.basename(cfg["texture_file"]))
        if not os.path.exists(texture_path):
            texture_path = os.path.join(script_dir, cfg["texture_file"])
        tiled = _get_tiled_texture(texture_path, cfg["texture_scale"], img_w, img_h)
        if tiled is not None:
            apply_texture_overlay_inplace(dark_array, tiled, cfg["texture_opacity"])
            apply_texture_overlay_inplace(mid_array, tiled, cfg["texture_opacity"])
            apply_texture_overlay_inplace(light_array, tiled, cfg["texture_opacity"])
    print(f"   [6/7] Texture overlay  ({time.time()-t:.2f}s)")

    # ---- Step 7: Composite ----
    t = time.time()
    result_array = np.array(rgb, dtype=np.float32)

    # Vectorized compositing with pre-expanded masks
    light_m = light_mask[:, :, np.newaxis]
    mid_m = mid_mask[:, :, np.newaxis]
    dark_m = dark_mask[:, :, np.newaxis]

    # Light (bottom), mid (middle), dark (top)
    result_array += (light_array - result_array) * light_m
    result_array += (mid_array - result_array) * mid_m
    result_array += (dark_array - result_array) * dark_m

    result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))

    # Tonal adjustments (only if != 1.0)
    if cfg["saturation"] != 1.0:
        result = ImageEnhance.Color(result).enhance(cfg["saturation"])
    if cfg["brightness"] != 1.0:
        result = ImageEnhance.Brightness(result).enhance(cfg["brightness"])
    if cfg["contrast"] != 1.0:
        result = ImageEnhance.Contrast(result).enhance(cfg["contrast"])

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    print(f"   [7/7] Composited  ({time.time()-t:.2f}s)")

    # Debug outputs
    debug_masks = create_debug_mask_image(dark_mask, mid_mask, light_mask)
    dark_mask_img = Image.fromarray((dark_mask * 255).astype(np.uint8), "L")
    mid_mask_img = Image.fromarray((mid_mask * 255).astype(np.uint8), "L")
    light_mask_img = Image.fromarray((light_mask * 255).astype(np.uint8), "L")

    return {
        "result": result,
        "debug_masks": debug_masks,
        "dark_mask": dark_mask_img,
        "mid_mask": mid_mask_img,
        "light_mask": light_mask_img,
    }


# ============================================
# LAYER FUNCTION (for watercolor.py integration)
# ============================================

def create_edge_curve_layer(subject: Image.Image, canvas_size: tuple,
                            offset: tuple, settings: dict = None) -> Image.Image:
    outputs = apply_edge_curve_effect(subject, settings)
    effected = outputs["result"]
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if effected.mode == "RGBA":
        layer.paste(effected, offset, effected)
    else:
        layer.paste(effected, offset)
    return layer


# ============================================
# STANDALONE ENTRY POINT
# ============================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.jpg")

    if not os.path.exists(input_path):
        print(f"❌ Place 'input.jpg' in: {script_dir}")
        exit(1)

    print("🎨 Watercolor Edge Curve Effect (Optimized)")
    print("=" * 60)

    input_image = Image.open(input_path).convert("RGB")
    print(f"   Input: {input_image.size[0]}×{input_image.size[1]}")

    t0 = time.time()
    outputs = apply_edge_curve_effect(input_image)
    elapsed = time.time() - t0

    result_path = os.path.join(script_dir, "edge_curve_result.png")
    outputs["result"].save(result_path, quality=95)
    print(f"\n   ✅ Result       → {result_path}")

    outputs["debug_masks"].save(os.path.join(script_dir, "debug_masks.png"))
    outputs["dark_mask"].save(os.path.join(script_dir, "debug_dark_mask.png"))
    outputs["mid_mask"].save(os.path.join(script_dir, "debug_mid_mask.png"))
    outputs["light_mask"].save(os.path.join(script_dir, "debug_light_mask.png"))
    print(f"   🔍 Debug        → debug_masks.png, debug_*_mask.png")

    print(f"\n   ⏱️  Total: {elapsed:.2f}s")
