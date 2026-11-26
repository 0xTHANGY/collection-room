#!/usr/bin/env python3
"""2x upscale of layer cutouts with Real-ESRGAN x4plus (ONNX, CPU), downsampled x4->x2 with Lanczos.
RGB from the model (premultiplied on a mid-grey matte to avoid edge halos), alpha resized with Lanczos.
Originals are copied to the report folder before replacement."""
import sys, shutil, json
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image

MODEL = "/workspace/sr-models/realesrgan-x4plus.onnx"
ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT.parent / "collection-room-impl-report/uniform-size-2/upscale/originals"
_sess = None

def sr(rgb):
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    y = _sess.run(None, {"input": x})[0][0].transpose(1, 2, 0)
    return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)

def upscale(src: Path, dst: Path):
    im = Image.open(src).convert("RGBA")
    a = np.asarray(im)
    alpha = a[:, :, 3:4].astype(np.float32) / 255
    # bleed colours into transparent pixels so the model does not invent dark fringes
    rgb = a[:, :, :3].astype(np.float32)
    matte = (rgb * alpha).sum((0, 1)) / max(alpha.sum(), 1)
    flat = (rgb * alpha + matte * (1 - alpha)).astype(np.uint8)
    up4 = Image.fromarray(sr(flat))
    w2, h2 = im.width * 2, im.height * 2
    rgb2 = up4.resize((w2, h2), Image.Resampling.LANCZOS)
    a2 = Image.fromarray(a[:, :, 3]).resize((w2, h2), Image.Resampling.LANCZOS)
    out = rgb2.convert("RGBA"); out.putalpha(a2)
    out.save(dst, "WEBP", quality=92, method=6)

if __name__ == "__main__":
    ORIG.mkdir(parents=True, exist_ok=True)
    for path in sys.argv[1:]:
        p = ROOT / path
        keep = ORIG / p.name
        if not keep.exists():
            shutil.copy2(p, keep)
        upscale(keep, p)
        print("upscaled", path, Image.open(keep).size, "->", Image.open(p).size)
