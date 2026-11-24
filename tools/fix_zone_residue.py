#!/usr/bin/env python3
"""P1 (cutout audit 2026-11-24): clean rail-end / bracket / bench-corner residue that
repack_three_rails.py left on cloned plates. Base 1600 images are edited in place,
then 1024/640 variants and the cover composite are regenerated.

Wall fill = per-row linear interpolation between clean wall columns on each side of the
region (robust to vertical light gradients), smoothed vertically, plus matched grain.
"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repack_three_rails as R

def wall_like(rgb):
    f = rgb.astype(np.int16)
    sat = f.max(2) - f.min(2)
    return (sat < 18) & (f.mean(2) > 140)

def row_lerp_fill(img, x0, x1, y0, y1, lref, rref, rng):
    """Fill img[y0:y1, x0:x1] with wall interpolated from columns lref=(a,b) and rref=(a,b)."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    def ref(a, b):
        a, b = max(0, a), min(w, b)
        if b - a < 3:
            return None
        blk = f[y0:y1, a:b]
        ok = wall_like(img[y0:y1, a:b])
        out = np.zeros((y1 - y0, 3), np.float32)
        valid = np.zeros(y1 - y0, bool)
        for r in range(y1 - y0):
            m = ok[r]
            if m.sum() >= 3:
                out[r] = np.median(blk[r][m], 0); valid[r] = True
        if valid.sum() < 3:
            return None
        idx = np.arange(y1 - y0)
        for c in range(3):
            out[:, c] = np.interp(idx, idx[valid], out[valid, c])
        out = cv2.GaussianBlur(out[:, None, :], (1, 0), sigmaX=0.1, sigmaY=6)[:, 0, :]
        return out, blk[ok].std(0) if ok.sum() > 10 else np.ones(3)
    L = ref(*lref); Rr = ref(*rref)
    if L is None and Rr is None:
        raise RuntimeError("no clean wall reference")
    if L is None: L = Rr
    if Rr is None: Rr = L
    t = np.linspace(0, 1, x1 - x0)[None, :, None]
    fill = L[0][:, None, :] * (1 - t) + Rr[0][:, None, :] * t
    grain = np.minimum((L[1] + Rr[1]) / 2, 2.0)
    fill = fill + rng.normal(0, 1, fill.shape).astype(np.float32) * grain * 0.6
    return fill

def col_lerp_fill(img, x0, x1, y0, y1, rng):
    """Fill by interpolating each column between clean wall rows just above and below."""
    T = np.transpose(img, (1, 0, 2))
    fill = row_lerp_fill(T, y0, y1, x0, x1, (y0 - 14, y0 - 3), (y1 + 3, y1 + 14), rng)
    return np.transpose(fill, (1, 0, 2))

def apply(img, mask, fill, x0, y0):
    sub = img[y0:y0 + fill.shape[0], x0:x0 + fill.shape[1]].astype(np.float32)
    m = mask[y0:y0 + fill.shape[0], x0:x0 + fill.shape[1]].astype(np.float32)
    m = cv2.GaussianBlur(m, (5, 5), 1.0)
    m = np.maximum(m, mask[y0:y0 + fill.shape[0], x0:x0 + fill.shape[1]].astype(np.float32) * 0.0)
    out = sub * (1 - m[..., None]) + fill * m[..., None]
    img[y0:y0 + fill.shape[0], x0:x0 + fill.shape[1]] = np.clip(out, 0, 255).astype(np.uint8)

def rail_keep(img, band):
    """Pixels that belong to the rail body (inside its band, not wall-coloured)."""
    y0, y1, x0, x1 = band
    h, w = img.shape[:2]
    keep = np.zeros((h, w), bool)
    sub = img[max(0, y0 - 2):y1 + 3, max(0, x0 - 8):x1 + 9]
    k = ~wall_like(sub)
    keep[max(0, y0 - 2):y1 + 3, max(0, x0 - 8):x1 + 9] = k
    return keep

def clean_ends(img, bands, spec, rng, log):
    h, w = img.shape[:2]
    for side in spec.get("sides", ["left", "right"]):
        xs = [b[2] for b in bands] if side == "left" else [b[3] for b in bands]
        xe = int(np.median(xs))
        a, b = (xe - spec.get("out", 70), xe + spec.get("in", 40)) if side == "left" else (xe - spec.get("in", 40), xe + spec.get("out", 70))
        a, b = max(0, a), min(w, b)
        ytop = max(0, min(bb[0] for bb in bands) - spec.get("above", 30))
        ybot = min(h, max(bb[1] for bb in bands) + spec.get("below", 110))
        keep = np.zeros((h, w), bool)
        for bb in bands:
            keep |= rail_keep(img, bb)
        mask = np.zeros((h, w), bool)
        mask[ytop:ybot, a:b] = True
        mask &= ~keep
        f = img.astype(np.int16)
        other = R.wood_mask(img) | ((f[..., 0] - f[..., 2]) > 18)
        other[max(0, ytop):max(bb[1] for bb in bands) + 3] = False  # rails handled by keep
        mask &= ~(cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        fill = row_lerp_fill(img, a, b, ytop, ybot, (a - 40, a - 6), (b + 6, b + 40), rng)
        apply(img, mask, fill, a, ytop)
        log.append(dict(kind=f"rail_{side}", box=[a, ytop, b - a, ybot - ytop]))

def trim_fade(img, bands, spec, rng, log):
    """Cut faded (semi-transparent looking) rail ends back to the solid wood."""
    for (y0, y1, x0, x1) in bands:
        f = img[y0:y1 + 1].astype(np.int16)
        sat = (f.max(2) - f.min(2)).mean(0)
        solid = np.median(sat[x0 + 40:x1 - 40])
        for side in spec["sides"]:
            if side == "left":
                xs = range(max(0, x0 - 10), x0 + 80)
                cut = next(x for x in xs if sat[x] >= 0.85 * solid and sat[x + 1] >= 0.85 * solid and sat[x + 2] >= 0.85 * solid)
                a, b = max(0, x0 - 14), cut
            else:
                xs = range(min(img.shape[1] - 3, x1 + 10), x1 - 80, -1)
                cut = next(x for x in xs if sat[x] >= 0.85 * solid and sat[x - 1] >= 0.85 * solid and sat[x - 2] >= 0.85 * solid)
                a, b = cut + 1, min(img.shape[1], x1 + 14)
            if b - a < 2:
                continue
            ya, yb = y0 - 6, y1 + 8
            mask = np.zeros(img.shape[:2], bool); mask[ya:yb, a:b] = True
            fill = col_lerp_fill(img, a, b, ya, yb, rng)
            apply(img, mask, fill, a, ya)
            log.append(dict(kind=f"fade_{side}", box=[a, ya, b - a, yb - ya], cut=int(cut)))

def clean_box(img, spec, rng, log):
    """Generic box: replace wall-ish non-wood pixels with interpolated wall."""
    x, y, bw, bh = spec["box"]
    f = img.astype(np.int16)
    keep = R.wood_mask(img) | ((f[..., 0] - f[..., 2]) > 18)
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = np.zeros(img.shape[:2], bool); mask[y:y + bh, x:x + bw] = True
    mask &= ~keep
    fill = row_lerp_fill(img, x, x + bw, y, y + bh, (x - 30, x - 3), tuple(spec.get("rref", (x + bw + 3, x + bw + 30))), rng)
    apply(img, mask, fill, x, y)
    log.append(dict(kind="box", box=[x, y, bw, bh]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--proof")
    args = ap.parse_args()
    spec_all = json.loads((ROOT / "data/source/display/zone-residue-fix.json").read_text())
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    items = catalog["items"]; zones = {z["id"]: z for z in catalog["zones"]}
    summary = {}
    for zid, spec in spec_all["zones"].items():
        rng = np.random.default_rng(zlib.crc32(zid.encode()))
        src = ROOT / f"assets/zones/{zid}-1600.webp"
        img = np.array(Image.open(src).convert("RGB"))
        before = img.copy()
        bands = R.detect_rail_bands(img)
        log = []
        for op in spec["ops"]:
            if op["op"] == "ends": clean_ends(img, bands, op, rng, log)
            elif op["op"] == "fade": trim_fade(img, bands, op, rng, log)
            elif op["op"] == "box": clean_box(img, op, rng, log)
        summary[zid] = dict(bands=[list(map(int, b)) for b in bands], ops=log,
                            changed_px=int((np.abs(img.astype(int) - before).sum(2) > 6).sum()))
        if args.proof:
            Path(args.proof).mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate([before, img], 1)).save(f"{args.proof}/{zid}-before-after.png")
        if args.write:
            im = Image.fromarray(img)
            R.variants_from(im, f"assets/zones/{zid}", [1600, 1024, 640])
    print(json.dumps(summary, indent=1))
    if args.write:
        (ROOT / "data/source/display/zone-residue-fix-summary.json").write_text(json.dumps(summary, indent=1) + "\n")

if __name__ == "__main__":
    main()
