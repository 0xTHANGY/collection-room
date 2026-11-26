#!/usr/bin/env python3
"""Displayed visible-size metrics per item (full alpha bbox and 'body' bbox without thin cords/hooks)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W, H = 1536, 1024
_cache = {}

def metrics(item):
    """px in the 1536x1024 plate: (full_w, full_h, body_w, body_h)."""
    layer = item["layer"]
    key = (layer["image"], round(layer["width"], 4), round(layer["height"], 4))
    if key in _cache:
        return _cache[key]
    with Image.open(ROOT / layer["image"]) as im:
        a = np.asarray(im.convert("RGBA").getchannel("A")) > 24
    bw, bh = layer["width"] * W / 100, layer["height"] * H / 100
    sx, sy = bw / a.shape[1], bh / a.shape[0]
    def box(m):
        ys, xs = np.where(m)
        return (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy
    fw, fh = box(a)
    # thin-structure removal: opening with a kernel ~2.5% of the shorter displayed side
    r = max(3, int(round(0.025 * min(a.shape))))
    body = cv2.morphologyEx(a.astype("uint8"), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if body.sum() < 0.3 * a.sum():
        body = a.astype("uint8")
    bwid, bhei = box(body > 0)
    _cache[key] = (fw, fh, bwid, bhei)
    return _cache[key]

def cv(x):
    x = np.asarray(x, float)
    return float(x.std() / x.mean())

def room_stats(catalog, zone):
    rows = [metrics(next(i for i in catalog["items"] if i["id"] == iid)) for iid in zone["items"]]
    a = np.array(rows)
    return {
        "n": len(rows),
        "cv_full_h": cv(a[:, 1]), "cv_body_h": cv(a[:, 3]), "cv_body_w": cv(a[:, 2]),
        "cv_body_sqrtarea": cv(np.sqrt(a[:, 2] * a[:, 3])),
        "mean_body_h": float(a[:, 3].mean()), "min_body_h": float(a[:, 3].min()), "max_body_h": float(a[:, 3].max()),
    }

if __name__ == "__main__":
    cat = json.load(open(ROOT / "data/catalog.json"))
    for z in cat["zones"]:
        if z["id"].split("-")[0] in ("hats", "totes"):
            s = room_stats(cat, z)
            print(z["id"], " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
