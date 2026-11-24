#!/usr/bin/env python3
"""P3 (2026-11-24): re-segment layers from the original AI painting they were cut from.

The painting is read from git history (tools/cutout_locate.py finds commit + pixel box, verified
by RGB agreement with the current layer). GrabCut is seeded by the current alpha: the eroded core is
sure-foreground, everything > `grow` px outside the old alpha is sure-background, and wall-coloured
pixels near the old edge become probable-background. The new mask is matted with
cutout_matte.rgba_from_mask (smooth alpha + interior colour on the rim). Layer size/rect unchanged.
"""
import argparse, io, json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import cutout_locate as LOC
from cutout_matte import rgba_from_mask

def recut(iid, core=6, pad=14, wall_tol=26):
    loc = LOC.locate(iid)
    if loc is None:
        raise RuntimeError("no source painting found")
    diff, zc, src, (x, y, w, h), zid, lc = loc
    paint = np.array(Image.open(io.BytesIO(LOC.show(zc, src))).convert("RGB"))
    H, W = paint.shape[:2]
    lay = np.array(Image.open(ROOT / f"assets/zones/layers/{iid}.webp").convert("RGBA"))
    a = lay[..., 3] > 128
    x0, y0 = max(0, x - pad), max(0, y - pad); x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = paint[y0:y1, x0:x1].copy()
    # inside the old alpha use the layer's own pixels (v2.3 layers carry reconstructed regions where a
    # neighbour overlapped in the painting); the painting only supplies real background context
    sub_c = crop[y - y0:y - y0 + h, x - x0:x - x0 + w]
    sub_c[a] = lay[..., :3][a]
    old = np.zeros(crop.shape[:2], np.uint8); old[y - y0:y - y0 + h, x - x0:x - x0 + w] = a
    # wall colour = median of crop pixels well outside the old alpha
    far = cv2.dilate(old, np.ones((3, 3), np.uint8), iterations=8) == 0
    wall = np.median(crop[far], 0) if far.sum() > 50 else np.array([200, 200, 200])
    wallish = np.abs(crop.astype(int) - wall).max(2) < wall_tol
    gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    gc[old > 0] = cv2.GC_PR_FGD  # never grow past the old alpha
    edge_ring = (old > 0) & (cv2.erode(old, np.ones((3, 3), np.uint8), iterations=core) == 0)
    gc[edge_ring & wallish] = cv2.GC_PR_BGD
    gc[cv2.erode(old, np.ones((3, 3), np.uint8), iterations=core) > 0] = cv2.GC_FGD
    # anything outside the original box stays background (layer size is fixed)
    box = np.zeros_like(old); box[y - y0:y - y0 + h, x - x0:x - x0 + w] = 1
    gc[box == 0] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # fill only small holes (keep real gaps such as arm/body)
    inv = 1 - m
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for j in range(1, n):
        if st[j, 4] < 40 and not (st[j, 0] == 0 or st[j, 1] == 0 or st[j, 0] + st[j, 2] >= m.shape[1] or st[j, 1] + st[j, 3] >= m.shape[0]):
            m[lab == j] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sub = m[y - y0:y - y0 + h, x - x0:x - x0 + w]
    rgb = lay[..., :3].copy()
    out = rgba_from_mask(rgb, sub, shrink=0.5, wall=wall)
    stats = dict(source_commit=zc, source=src, box=[x, y, w, h], rgb_diff=float(diff),
                 old_px=int(a.sum()), new_px=int((out[..., 3] > 128).sum()))
    return out, stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ids", nargs="+"); ap.add_argument("--write", action="store_true"); ap.add_argument("--preview")
    args = ap.parse_args()
    res = {}
    for iid in args.ids:
        out, st = recut(iid)
        res[iid] = st
        if args.preview:
            Path(args.preview).mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(f"{args.preview}/{iid}.png")
        if args.write:
            p = ROOT / f"assets/zones/layers/{iid}.webp"
            Image.fromarray(out).save(p.with_suffix(".webp.tmp"), "WEBP", lossless=False, quality=90, method=6, exact=False)
            p.with_suffix(".webp.tmp").replace(p)
        print(iid, st)
    if args.write:
        f = ROOT / "data/source/display/recut-from-painting.json"
        prev = json.loads(f.read_text()) if f.exists() else {}
        prev.update(res); f.write_text(json.dumps(prev, indent=1) + "\n")

if __name__ == "__main__":
    main()
