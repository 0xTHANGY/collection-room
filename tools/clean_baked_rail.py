#!/usr/bin/env python3
"""Erase a baked wooden rail from an item layer, keeping / rebuilding the hook.

Does not generate a new painting. Transparent pixels have RGB zeroed so
ghost rails do not composite. Used by the three-rod rollout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

def wood_mask(arr, alpha_th=8):
    r, g, b, a = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int), arr[:, :, 3]
    return (a > alpha_th) & (r > b + 12) & (r > g - 5) & (r > 50) & (r < 235)

def detect_rail(arr):
    h, w = arr.shape[:2]
    wood = wood_mask(arr, alpha_th=8)
    top_h = min(h, max(32, h // 4))
    frac = wood[:top_h].mean(1)
    bands = []
    i = 0
    while i < top_h:
        if frac[i] >= 0.40 or (i <= 8 and frac[i] >= 0.20):
            j = i
            while j < top_h and frac[j] >= 0.20:
                j += 1
            height = j - i
            cols = np.where(wood[i:j].mean(0) > 0.2)[0]
            wf = ((cols[-1] - cols[0] + 1) / w) if len(cols) else 0
            if height <= 40 and wf >= 0.70 and frac[i:j].max() >= 0.55 and i <= 24:
                yb = min(h, j + 16)
                wood_below = float(wood[j:yb].mean()) if yb > j else 1
                if wood_below <= 0.28:
                    bands.append((i, j - 1, float(frac[i:j].max()), wf, wood_below))
            i = max(j, i + 1)
        else:
            i += 1
    if not bands:
        return None
    y0, y1, mf, wf, wb = bands[0]
    y = y1 + 1
    while y < min(h, y1 + 8) and wood[y].mean() >= 0.25:
        y1 = y
        y += 1
    return (int(y0), int(y1), mf, wf, wb)

def find_stem(arr, y0, y1):
    h, w = arr.shape[:2]
    wood = wood_mask(arr)
    cx_hanger = None
    for y in range(y1 + 4, min(h, y1 + 90)):
        xs = np.where(wood[y])[0]
        if len(xs) >= 10:
            span = xs[-1] - xs[0] + 1
            if 10 <= span <= int(w * 0.55):
                cx_hanger = int(xs.mean())
                break
    y_lo, y_hi = min(h, y1 + 1), min(h, y1 + 24)
    a = arr[y_lo:y_hi, :, 3]
    col = (a > 80).mean(0)
    thin = np.zeros(w)
    i = 0
    while i < w:
        if col[i] > 0.2:
            j = i
            while j < w and col[j] > 0.15:
                j += 1
            if 1 <= (j - i) <= 8:
                thin[i:j] = col[i:j].max()
            i = j
        else:
            i += 1
    center = cx_hanger if cx_hanger is not None else w // 2
    win = max(20, w // 6)
    xa, xb = max(0, center - win), min(w, center + win + 1)
    if thin[xa:xb].max() > 0:
        cx = int(xa + thin[xa:xb].argmax())
    else:
        cx = center
    widths, colors = [], []
    for yy in range(y_lo, min(y_hi, y_lo + 16)):
        xs = np.where(arr[yy, max(0, cx - 8) : min(w, cx + 9), 3] > 80)[0]
        if 1 <= len(xs) <= 8:
            widths.append(len(xs))
            colors.append(arr[yy, max(0, cx - 8) + xs, :])
    width = int(np.median(widths)) if widths else 3
    width = max(2, min(5, width))
    if colors:
        colv = np.median(np.concatenate(colors, 0), 0).astype(np.uint8)
    else:
        colv = np.array([150, 150, 150, 255], dtype=np.uint8)
    return cx, width, colv

def clean_array(arr):
    """Return cleaned RGBA array, or None if no rail found."""
    band = detect_rail(arr)
    if not band:
        return None, None
    out = arr.copy()
    trans = out[:, :, 3] == 0
    out[trans, :3] = 0
    y0, y1 = band[0], band[1]
    h, w = out.shape[:2]
    cx, sw, scol = find_stem(arr, y0, y1)
    ey0, ey1 = max(0, y0 - 1), min(h - 1, y1 + 3)
    out[ey0 : ey1 + 1] = 0
    xa, xb = max(0, cx - sw // 2), min(w, cx - sw // 2 + sw)
    has_cap = (y0 > 2) and (arr[max(0, y0 - 12) : y0, max(0, xa - 6) : min(w, xb + 7), 3] > 80).any()
    top_y = max(0, y0 - 1) if has_cap else max(0, y0 - 2)
    for y in range(top_y, min(h, y1 + 8)):
        if y > y1 and out[y, max(0, cx - 4) : min(w, cx + 5), 3].max() > 0:
            break
        for x in range(xa, xb):
            if out[y, x, 3] == 0:
                out[y, x] = scol
    if not has_cap:
        hy = max(0, top_y)
        for dx in range(0, 6):
            x = cx - dx
            if 0 <= x < w and out[hy, x, 3] == 0:
                out[hy, x] = scol
            if hy + 1 < h and 0 <= x - 1 < w and dx >= 3 and out[hy + 1, x - 1, 3] == 0:
                out[hy + 1, x - 1] = scol
    trans = out[:, :, 3] == 0
    out[trans, :3] = 0
    return out, dict(band=band, cx=cx, sw=sw)

def save_webp_rgba(arr, path: Path, quality=80):
    path = Path(path)
    im = Image.fromarray(arr)
    tmp = path.with_suffix(path.suffix + ".tmp")
    im.save(tmp, "WEBP", quality=quality, method=4)
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for p in args.images:
        path = Path(p)
        arr = np.array(Image.open(path).convert("RGBA"))
        cleaned, meta = clean_array(arr)
        if cleaned is None:
            print(f"NO_RAIL {path}")
            continue
        print(f"CLEAN {path} {meta}")
        if not args.dry_run:
            save_webp_rgba(cleaned, path)

if __name__ == "__main__":
    main()
