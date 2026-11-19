#!/usr/bin/env python3
"""v3: remove light fringe / magenta spill on dark magenta-keyed cutouts (site webps only)."""
import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/source/today-2026-11-18/cutout-log.json"

def lum(rgb): return rgb[..., 0]*.2126 + rgb[..., 1]*.7152 + rgb[..., 2]*.0722

def analyse(arr):
    a = arr[..., 3]
    rgb = arr[..., :3].astype(np.float32)
    solid = (a >= 250).astype(np.uint8)
    core = cv2.erode(solid, np.ones((3, 3), np.uint8), iterations=4)  # 4px inside
    vis = (a > 0).astype(np.uint8)
    # ring: visible pixels within 3px of the outside
    inner = cv2.erode(vis, np.ones((3, 3), np.uint8), iterations=3)
    ring = (vis > 0) & (inner == 0)
    L = lum(rgb)
    med = float(np.median(L[core > 0])) if core.any() else float(np.median(L[vis > 0]))
    ringL = float(np.mean(L[ring])) if ring.any() else 0.0
    # spill: magenta-ness in ring
    spill = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    spillr = float(np.mean(spill[ring] > 40)) if ring.any() else 0.0
    return dict(core_median=med, ring_mean=ringL, ring_spill_frac=spillr), ring, core, vis

def fix(arr, strength_choke=True):
    a = arr[..., 3].astype(np.float32)
    rgb = arr[..., :3].astype(np.float32)
    info, ring, core, vis = analyse(arr)
    # 1) choke alpha ~0.7px: shrink the soft edge so light fringe pixels fall away
    af = cv2.GaussianBlur(a, (0, 0), 0.6)
    a2 = np.clip((a - 255 * 0.25) / 0.75, 0, 255)
    a2 = np.minimum(a2, a)
    # 2) recolour all non-core pixels from nearest core colour
    src = (core == 0).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(core > 0)
    lab_of_core = np.zeros(labels.max() + 1, np.int64)
    lab_of_core[labels[ys, xs]] = np.arange(len(ys))
    idx = lab_of_core[labels]
    near = rgb[ys[idx], xs[idx]]
    ringd = cv2.dilate(ring.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sel = (core == 0) & (vis > 0)
    out = rgb.copy()
    # blend toward core colour but keep dark target luminance: take the darker of own and core
    new = np.minimum(out, near + 6)
    out[sel] = new[sel]
    # 3) despill any remaining magenta
    sp = np.maximum(0, np.minimum(out[..., 0], out[..., 2]) - out[..., 1])
    m = sel & (sp > 15)
    out[..., 0][m] -= sp[m] * 0.9
    out[..., 2][m] -= sp[m] * 0.9
    out = np.clip(out, 0, 255)
    res = np.dstack([out, a2]).astype(np.uint8)
    return res

def main(dry=False):
    log = json.loads(LOG.read_text())
    ids = [k for k, v in log.items() if v.get("method") == "codex-magenta-key"]
    rep = {}
    for iid in ids:
        for size in (400, 600):
            p = ROOT / f"assets/items/v2/{iid}-{size}.webp"
            if not p.exists(): continue
            arr = np.array(Image.open(p).convert("RGBA"))
            info, *_ = analyse(arr)
            dark = info["core_median"] < 95
            rep.setdefault(iid, {})[size] = dict(info, dark=dark)
            if dark and not dry:
                res = fix(arr)
                after, *_ = analyse(res)
                Image.fromarray(res, "RGBA").save(p, "WEBP", quality=88, method=4, exact=True)
                rep[iid][size]["after"] = after
    return rep

if __name__ == "__main__":
    dry = "--dry" in sys.argv
    rep = main(dry)
    darks = [i for i, v in rep.items() if v.get(600, {}).get("dark")]
    print("magenta-keyed:", len(rep), "dark:", len(darks))
    for i in darks:
        v = rep[i][600]
        print(i, "core %.0f ring %.0f spill %.2f" % (v["core_median"], v["ring_mean"], v["ring_spill_frac"]),
              ("-> ring %.0f spill %.2f" % (v["after"]["ring_mean"], v["after"]["ring_spill_frac"])) if "after" in v else "")
