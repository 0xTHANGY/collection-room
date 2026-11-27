#!/usr/bin/env python3
"""Move the wooden rods (with their wall brackets and soft shadow) of a jackets plate to new heights.

Each rod strip (band top-12 .. band bottom+16, rod span +-40 px) is lifted off the wall as a layer:
the wall under it is rebuilt by per-column linear interpolation between the clean rows above and
below, and the layer's alpha is the strip's difference from that wall. Rods are then laid back at the
new heights on the cleaned wall. Plate size, rod length, rug and floor are unchanged.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repack_three_rails import detect_rail_bands, save_webp

UP, DOWN, SIDE = 12, 16, 40


def wall_fill(img, y0, y1, x0, x1):
    """Replace rows y0..y1 (exclusive) by linear blend of the 3-row means just outside."""
    a = img[y0 - 3:y0, x0:x1].mean(0); b = img[y1:y1 + 3, x0:x1].mean(0)
    t = (np.arange(y1 - y0)[:, None, None] + 0.5) / (y1 - y0)
    img[y0:y1, x0:x1] = a[None] * (1 - t) + b[None] * t


def move(plate: Image.Image, new_tops):
    src = np.asarray(plate.convert("RGB")).astype(np.float32)
    bands = detect_rail_bands(src.astype(np.uint8))[:len(new_tops)]
    wall = src.copy(); strips = []
    for (y0, y1, x0, x1) in bands:
        ya, yb, xa, xb = y0 - UP, y1 + DOWN + 1, max(0, x0 - SIDE), min(src.shape[1], x1 + SIDE + 1)
        wall_fill(wall, ya, yb, xa, xb)
        strip = src[ya:yb, xa:xb]; bg = wall[ya:yb, xa:xb]
        d = np.abs(strip - bg).max(-1)
        alpha = np.clip((d - 3) / 18, 0, 1)
        strips.append((y0, ya, xa, strip, alpha))
    out = wall.copy()
    for (y0, ya, xa, strip, alpha), ny in zip(strips, new_tops):
        nya = ny - (y0 - ya)
        reg = out[nya:nya + strip.shape[0], xa:xa + strip.shape[1]]
        # the strip's own wall tone is replaced by the destination wall: layer = wall + (strip - old wall)
        reg[:] = reg * (1 - alpha[..., None]) + strip * alpha[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), bands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("tops", type=int, nargs="+")
    a = ap.parse_args()
    im, bands = move(Image.open(a.src), a.tops)
    im.save(a.out)
    print("old", bands, "->", detect_rail_bands(np.asarray(im)))


if __name__ == "__main__":
    main()
