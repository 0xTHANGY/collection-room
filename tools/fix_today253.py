#!/usr/bin/env python3
"""P2 (2026-11-24): today-253 (Nahmias MARTY SUPREME windbreaker) lost its black upper sleeves/yoke
when it was cut from the jackets-f painting in 9303dc6 (dark-on-dark neighbours), leaving two wedge
shapes. Re-cut from that same painting (git 9303dc6, box 993,144,216x269 == current layer pixels):
non-wall pixels inside the box (+-2px), the component under the garment centre, small holes filled;
hanger kept from the old mask. No redraw: the painting already contains the full garment."""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cutout_matte import rgba_from_mask
import repack_three_rails as R

def main(write=False):
    paint = np.array(Image.open(io.BytesIO(subprocess.run(
        ["git", "-C", str(ROOT), "show", "9303dc6:assets/zones/jackets-f-1600.webp"], capture_output=True, check=True).stdout)).convert("RGB")).astype(np.int16)
    x0, y0, w, h, P = 993, 144, 216, 269, 40
    cr = paint[y0:y0 + h, x0 - P:x0 + w + P]
    lay = np.array(Image.open(io.BytesIO(subprocess.run(["git", "-C", str(ROOT), "show", "326fc1d:assets/zones/layers/today-253.webp"],
                                                       capture_output=True, check=True).stdout)).convert("RGBA"))
    cur = np.zeros(cr.shape[:2], np.uint8); cur[:, P:P + w] = lay[..., 3] > 128
    wall = np.array([182, 176, 176])
    L = cr.mean(2); sat = cr.max(2) - cr.min(2)
    nw = ((np.abs(cr - wall).max(2) > 22) & ~((sat < 22) & (L > 95) & (L < 215))).astype(np.uint8)
    nw[:, :P - 2] = 0; nw[:, P + w + 2:] = 0
    nw = cv2.morphologyEx(nw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hem = np.where(cur.any(1))[0].max()
    nw[hem - 6:] &= cur[hem - 6:]
    n, lab, st, _ = cv2.connectedComponentsWithStats(nw, 8)
    m = (lab == lab[h // 2, P + w // 2]).astype(np.uint8)
    m |= cur & cv2.dilate(m, np.ones((3, 3), np.uint8))
    m[:40] |= cur[:40]
    ff = m.copy(); cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1); m |= 1 - ff
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = rgba_from_mask(paint[y0:y0 + h, x0:x0 + w].astype(np.uint8), m[:, P:P + w], shrink=0.5, wall=wall)
    print("old px", int(cur.sum()), "new px", int((out[..., 3] > 128).sum()))
    if write:
        p = ROOT / "assets/zones/layers/today-253.webp"
        Image.fromarray(out).save(p, "WEBP", quality=90, method=6)
        cat = json.loads((ROOT / "data/catalog.json").read_text())
        it = next(i for i in cat["items"] if i["id"] == "today-253")
        z = next(z for z in cat["zones"] if z["id"] == it["zone"]); W, H = z["natural_size"]
        Lr = it["layer"]; dw, dh = round(Lr["width"] / 100 * W), round(Lr["height"] / 100 * H)
        alpha = np.array(Image.fromarray(out).resize((dw, dh), Image.Resampling.LANCZOS))[..., 3]
        it["hotspot"]["clip"] = R.clip_polygon(alpha)
        (ROOT / "data/catalog.json").write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n")
    return out

if __name__ == "__main__":
    main("--write" in sys.argv)
