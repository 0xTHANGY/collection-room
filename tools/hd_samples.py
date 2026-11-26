#!/usr/bin/env python3
"""HD candidate previews for a few layer cutouts (no live replacement).

Methods (all output 2x the native cutout):
  lanczos         Lanczos 2x (reference)
  lanczos_usm     Lanczos 2x + unsharp mask (r 1.4, 70 %, thr 2)
  x4plus_b35      Real-ESRGAN x4plus -> 2x, blended 35 % over lanczos
  general_v3      realesr-general-x4v3 (compact model) -> 2x
  general_v3_b50  general_v3 blended 50 % over lanczos_usm
  detail_xfer     lanczos_usm base; x4plus only inside high-contrast logo/print edges (soft mask)
Edge handling for every method: RGB of semi-transparent / transparent pixels is replaced with the nearest
opaque interior colour before resampling (colour decontamination, kills light fringes), and alpha is the
Lanczos-resized original alpha (never model output).
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from PIL import Image, ImageFilter, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "collection-room-impl-report/uniform-size-3/hd-samples"
MODELS = {"x4plus": "/workspace/sr-models/realesrgan-x4plus.onnx", "general": "/workspace/sr-models/realesr-general-x4v3.onnx"}
WALL = (197, 197, 199)
_s = {}

def run(model, rgb):
    if model not in _s:
        _s[model] = ort.InferenceSession(MODELS[model], providers=["CPUExecutionProvider"])
    x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    y = _s[model].run(None, {"input": x})[0][0].transpose(1, 2, 0)
    return np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)

def decontaminate(rgba):
    a = rgba[:, :, 3]
    interior = (a >= 250).astype(np.uint8)
    if interior.sum() == 0:
        return rgba[:, :, :3].copy()
    # labels of nearest interior pixel
    _, labels = cv2.distanceTransformWithLabels(1 - interior, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(interior == 0 if False else interior > 0)
    # map label -> coordinate of that zero pixel (interior)
    zero = np.zeros(labels.max() + 1, dtype=np.int64)
    lab_int = labels[interior > 0]
    idx = np.flatnonzero(interior.ravel() > 0)
    zero[lab_int] = idx
    src = zero[labels.ravel()]
    rgb = rgba[:, :, :3].reshape(-1, 3)[src].reshape(rgba.shape[0], rgba.shape[1], 3)
    out = rgba[:, :, :3].copy()
    out[a < 250] = rgb[a < 250]
    return out

def up2(img_u8):
    return np.asarray(Image.fromarray(img_u8).resize((img_u8.shape[1] * 2, img_u8.shape[0] * 2), Image.Resampling.LANCZOS))

def from4(img4, w2, h2):
    return np.asarray(Image.fromarray(img4).resize((w2, h2), Image.Resampling.LANCZOS))

def usm(img):
    return np.asarray(Image.fromarray(img).filter(ImageFilter.UnsharpMask(radius=1.4, percent=70, threshold=2)))

def candidates(rgba):
    rgb = decontaminate(rgba)
    h, w = rgb.shape[:2]
    lan = up2(rgb)
    lan_usm = usm(lan)
    x4 = from4(run("x4plus", rgb), w * 2, h * 2)
    gv = from4(run("general", rgb), w * 2, h * 2)
    blend = lambda a, b, t: np.clip(a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t + 0.5, 0, 255).astype(np.uint8)
    # logo/print mask: strong local contrast in the original, dilated + feathered
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = (grad > 90).astype(np.uint8) * (rgba[:, :, 3] > 250)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(cv2.resize(m.astype(np.float32), (w * 2, h * 2)), (0, 0), 1.5)[:, :, None]
    xfer = np.clip(lan_usm * (1 - m) + x4 * m + 0.5, 0, 255).astype(np.uint8)
    return {
        "lanczos": lan, "lanczos_usm": lan_usm, "x4plus_b35": blend(lan, x4, 0.35),
        "general_v3": gv, "general_v3_b50": blend(lan_usm, gv, 0.5), "detail_xfer": xfer,
        "x4plus_full": x4,
    }

def metrics(orig_rgba, cand_rgb, alpha2):
    o = orig_rgba[:, :, :3].astype(np.float32); a = orig_rgba[:, :, 3] > 250
    down = np.asarray(Image.fromarray(cand_rgb).resize((o.shape[1], o.shape[0]), Image.Resampling.LANCZOS)).astype(np.float32)
    mse = ((down - o) ** 2)[a].mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-6))
    # halo: luminance excess of the 2-px edge band over the 3–6 px band just inside, composited on the wall
    al = alpha2.astype(np.float32) / 255
    comp = cand_rgb.astype(np.float32) * al[:, :, None] + np.array(WALL, np.float32) * (1 - al[:, :, None])
    L = comp.mean(2)
    solid = (alpha2 > 128).astype(np.uint8)
    d = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
    edge = (d > 0) & (d <= 2); inner = (d > 3) & (d <= 6)
    halo = float(L[edge].mean() - L[inner].mean()) if edge.any() and inner.any() else 0.0
    gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharp = float(lap[alpha2 > 250].std())
    return dict(psnr_vs_native=round(float(psnr), 2), halo_luma=round(halo, 2), sharpness=round(sharp, 2))

def on_wall(rgb, alpha):
    al = alpha.astype(np.float32)[:, :, None] / 255
    return Image.fromarray((rgb * al + np.array(WALL) * (1 - al) + 0.5).astype(np.uint8))

def logo_box(rgba, size):
    g = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) * (rgba[:, :, 3] > 250)
    k = cv2.boxFilter(e, -1, (size, size), normalize=False)
    y, x = np.unravel_index(np.argmax(k), k.shape)
    x0 = int(np.clip(x - size // 2, 0, rgba.shape[1] - size)); y0 = int(np.clip(y - size // 2, 0, rgba.shape[0] - size))
    return x0, y0, size

def main(ids):
    cat = json.loads((ROOT / "data/catalog.json").read_text())
    items = {i["id"]: i for i in cat["items"]}
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    report = {}
    for iid in ids:
        it = items[iid]
        src = ROOT / it["layer"]["image"]
        rgba = np.asarray(Image.open(src).convert("RGBA"))
        h, w = rgba.shape[:2]
        alpha2 = np.asarray(Image.fromarray(rgba[:, :, 3]).resize((w * 2, h * 2), Image.Resampling.LANCZOS))
        cands = candidates(rgba)
        d = OUT / iid; d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(d / "original.png")
        report[iid] = {"name": it["name"], "native": [w, h], "methods": {}}
        for name, rgb in cands.items():
            out = Image.fromarray(rgb).convert("RGBA"); out.putalpha(Image.fromarray(alpha2))
            out.save(d / f"{name}.png")
            report[iid]["methods"][name] = metrics(rgba, rgb, alpha2)
        # compare sheets: 100 % (2x pixels shown 1:1) and 200 % crop of the logo/texture window
        names = ["original_nearest"] + list(cands)
        tiles100, tiles200 = [], []
        x0, y0, s = logo_box(rgba, max(32, min(w, h) // 3))
        for name in names:
            if name == "original_nearest":
                rgb = np.asarray(Image.fromarray(rgba[:, :, :3]).resize((w * 2, h * 2), Image.Resampling.NEAREST))
            else:
                rgb = cands[name]
            t = on_wall(rgb, alpha2)
            tiles100.append((name, t))
            crop = t.crop((x0 * 2, y0 * 2, (x0 + s) * 2, (y0 + s) * 2)).resize((s * 4, s * 4), Image.Resampling.NEAREST)
            tiles200.append((name, crop))
        for label, tiles in (("100", tiles100), ("200", tiles200)):
            tw, th = tiles[0][1].size
            cols = 4
            rows = (len(tiles) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * (tw + 10), rows * (th + 34)), "white")
            dr = ImageDraw.Draw(sheet)
            for k, (name, t) in enumerate(tiles):
                cx, cy = (k % cols) * (tw + 10), (k // cols) * (th + 34)
                sheet.paste(t, (cx, cy + 30))
                m = report[iid]["methods"].get(name)
                dr.text((cx + 4, cy + 4), name + (f"  psnr {m['psnr_vs_native']} halo {m['halo_luma']}" if m and label == "100" else ""), fill="black", font=font)
            sheet.save(OUT / f"{iid}-compare-{label}.png")
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for iid, r in report.items():
        print(iid, r["name"], r["native"])
        for n, m in r["methods"].items():
            print("   ", f"{n:15}", m)

if __name__ == "__main__":
    main(sys.argv[1:])
