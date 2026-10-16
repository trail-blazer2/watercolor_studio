"""
☁️ Modal GPU Backend for Watercolor Studio
Deploy with: modal deploy backend/modal_app.py
"""

import os
import io
import base64
from typing import Optional
import modal

# Paths on your local computer
LOCAL_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ASSETS_DIR = os.path.join(os.path.dirname(LOCAL_BACKEND_DIR), "assets")

# Build the container image with GPU ONNX Runtime & pre-downloaded u2net model
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "ffmpeg")
    .pip_install(
        "pillow==10.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "rembg[gpu]==2.0.57",
        "onnxruntime-gpu==1.18.1",
        "fastapi[standard]==0.115.0",
        "pydantic==2.7.4",
        "python-multipart==0.0.9",
    )
    .run_commands(
        # Pre-download u2net weights into the container image so cold starts never wait on network
        "python -c 'from rembg import new_session; new_session(\"u2net\")'"
    )
    .add_local_dir(LOCAL_ASSETS_DIR, remote_path="/root/assets")
    .add_local_dir(LOCAL_BACKEND_DIR, remote_path="/root/backend")
)

app = modal.App("watercolor-studio-gpu", image=image)


@app.cls(
    gpu="T4",
    cpu=4.0,
    memory=8192,
    scaledown_window=15,
    enable_memory_snapshot=True,
)
class WatercolorService:
    @modal.enter(snap=True)
    def warmup(self):
        """Pre-load u2net and brush textures into memory snapshot for <4s startup."""
        import sys
        sys.path.insert(0, "/root/backend")
        import watercolor_engine as we
        we.get_rembg_session()
        for b in [
            "bgbrush.png", "brush2.png", "sharpbrush.png", "square.png",
            "whitebrush.png", "whitedots.png", "darkdots.png", "colordrop.png"
        ]:
            we.load_brush(b)
        print("⚡ WatercolorService warmed up and snapshotted!")

    @modal.asgi_app()
    def api(self):
        import sys
        sys.path.insert(0, "/root/backend")
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.middleware.cors import CORSMiddleware
        from PIL import Image
        import watercolor_engine as we
        from presets import PRESETS, auto_detect_preset

        web_app = FastAPI(title="Watercolor Studio GPU API")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "presets": {
                    k: {"name": v["name"], "category": v["category"], "description": v["description"]}
                    for k, v in PRESETS.items()
                },
            }

        @web_app.post("/cutout")
        async def cutout_only(
            image_file: UploadFile = File(...),
            preset_key: str = Form("animals"),
        ):
            """Fast AI cutout for manual mask editing before or during studio session."""
            raw_bytes = await image_file.read()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
            if preset_key == "auto":
                preset_key = auto_detect_preset(pil_img)
            cfg = we.get_preset_copy(preset_key)
            cutout = we.remove_background_raw(pil_img)
            trim = cfg.get("SUBJECT_EDGE_TRIM", 6)
            if trim > 0:
                cutout = we.trim_subject_edges(cutout, trim)
            mask = cutout.split()[3]
            return {
                "preset_key": preset_key,
                "mask_png": we.pil_to_base64(mask, "PNG"),
                "cutout_png": we.pil_to_base64(cutout, "PNG"),
            }

        @web_app.post("/generate")
        async def generate_studio(
            image_file: UploadFile = File(...),
            preset_key: str = Form("animals"),
            mask_base64: Optional[str] = Form(None),
        ):
            """Full multi-layer watercolor generation + stroke data for live editing."""
            raw_bytes = await image_file.read()
            pil_img = Image.open(io.BytesIO(raw_bytes))

            if preset_key == "auto":
                preset_key = auto_detect_preset(pil_img)

            custom_mask = None
            if mask_base64:
                b64_clean = mask_base64.split(",")[-1]
                custom_mask = Image.open(io.BytesIO(base64.b64decode(b64_clean))).convert("L")

            payload = we.generate_full_studio_payload(
                original_image=pil_img,
                preset_key=preset_key,
                custom_mask=custom_mask,
            )
            return payload

        @web_app.get("/brushes")
        def get_brush_textures():
            """Returns base64 PNGs of all brush textures so the browser WebGL/Canvas can stamp them live."""
            brush_files = [
                "bgbrush.png", "brush2.png", "sharpbrush.png", "square.png",
                "whitebrush.png", "whitedots.png", "darkdots.png", "colordrop.png"
            ]
            out = {}
            for bf in brush_files:
                img = we.load_brush(bf)
                out[bf] = we.pil_to_base64(img, "PNG")
            return out

        return web_app