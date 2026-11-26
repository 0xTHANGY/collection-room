#!/usr/bin/env python3
"""Check that every shoe rests on a shelf, stool or the rug, and does not overlap a rack post."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from repack_shoes import rug_mask  # noqa: E402
from repack_three_rails import wood_mask  # noqa: E402

W, H = 1536, 1024
TOL_ABOVE = 6  # cannot hover more than this above the board top
TOL_BELOW = 32  # board thickness plus a little perspective (front of the stool reads lower)
POST_OVERLAP_MAX = 8  # opaque pixels allowed to graze a post


def load_catalog(path=None):
    return json.loads((path or (ROOT / "data/catalog.json")).read_text())


def plate_rgb(zone):
    src = max(zone["images"], key=lambda i: i["width"])["src"]
    return np.asarray(Image.open(ROOT / src).convert("RGB"))


def rug_top_y(rgb):
    """First sustained rug band in the lower half — ignore false hits near the rail."""
    row = rug_mask(rgb).mean(1)
    for y in range(300, 900):
        if row[y] > 0.30 and row[y:y + 50].mean() > 0.30:
            return y
    ys = np.where(row[300:] > 0.30)[0]
    return int(ys[0] + 300) if len(ys) else 700


def furniture(rgb, rug_y):
    """Shelf/stool sitting surfaces and vertical rack posts, above the rug.

    The sitting surface is the first wood pixel below the hanging-rail band
    (y >= 150). Contiguous x-runs with a stable first-wood y are one board.
    """
    wood = wood_mask(rgb).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    h, w = wood.shape
    first = np.full(w, -1, dtype=int)
    for x in range(w):
        ys = np.where(wood[150:rug_y, x] > 0)[0]
        if len(ys):
            first[x] = 150 + int(ys[0])
    shelves = []
    x = 0
    while x < w:
        if first[x] < 0:
            x += 1
            continue
        x0, ys = x, [int(first[x])]
        yref = first[x]
        while x + 1 < w and first[x + 1] > 0 and abs(int(first[x + 1]) - yref) <= 14:
            x += 1
            ys.append(int(first[x]))
            yref = int(np.median(ys))
        x1 = x
        if x1 - x0 + 1 >= 80:
            shelves.append((int(np.median(ys)), int(x0), int(x1)))
        x = x1 + 1

    posts = []
    col = wood[:rug_y].sum(0)
    x = 0
    while x < w:
        if col[x] < 80:
            x += 1
            continue
        x1 = x
        while x1 + 1 < w and col[x1 + 1] >= 80:
            x1 += 1
        width = x1 - x + 1
        if 6 <= width <= 45:
            ys = np.where(wood[:rug_y, x:x1 + 1].max(1) > 0)[0]
            if len(ys) and ys[-1] - ys[0] >= 140:
                posts.append((int(x), int(x1), int(ys[0]), int(ys[-1])))
        x = x1 + 1
    return shelves, posts, rug_y


def placed_alpha(item):
    layer = item["layer"]
    im = Image.open(ROOT / layer["image"]).convert("RGBA")
    x = round(layer["left"] / 100 * W)
    y = round(layer["top"] / 100 * H)
    w = max(1, round(layer["width"] / 100 * W))
    h = max(1, round(layer["height"] / 100 * H))
    placed = im.resize((w, h), Image.Resampling.LANCZOS)
    a = np.array(placed.split()[-1])
    return x, y, a


def alpha_bottom(x, y, a, threshold=24):
    rows = np.where(a.max(1) > threshold)[0]
    cols = np.where(a.max(0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return y + a.shape[0] - 1, x, x + a.shape[1] - 1
    return y + int(rows[-1]), x + int(cols[0]), x + int(cols[-1])


def nearest_shelf(bottom, x0, x1, shelves):
    best = None
    for y, sx0, sx1 in shelves:
        # require some x overlap with the shoe footprint
        if min(x1, sx1) - max(x0, sx0) <= 4:
            continue
        dist = abs(bottom - y)
        if best is None or dist < best[0]:
            best = (dist, y, sx0, sx1)
    return best


def post_overlap(x, y, a, posts, threshold=24):
    total = 0
    h, w = a.shape
    ys, xs = np.where(a > threshold)
    for py, px in zip(ys, xs):
        gx, gy = x + int(px), y + int(py)
        for x0, x1, y0, y1 in posts:
            if x0 <= gx <= x1 and y0 <= gy <= y1:
                total += 1
                break
    return total


def check(catalog) -> list[str]:
    zones = {z["id"]: z for z in catalog["zones"]}
    errors = []
    details = []
    for zid in ("shoes-a", "shoes-b", "shoes-c", "shoes-d", "shoes-e"):
        rgb = plate_rgb(zones[zid])
        shelves, posts, rug_y = furniture(rgb, rug_top_y(rgb))
        items = [it for it in catalog["items"] if it["zone"] == zid]
        # Rug-row baselines from CSS box bottoms of items that sit on the rug
        # Furniture vs rug by where the box bottom sits: rug rows sit well below the
        # rug's back edge, furniture surfaces sit above it. (Top-% fails once
        # back-row rug shoes are enlarged.)
        def on_rug(it):
            return (it["layer"]["top"] + it["layer"]["height"]) / 100 * H > rug_y
        rug_items = [it for it in items if on_rug(it)]
        baselines = sorted({round((it["layer"]["top"] + it["layer"]["height"]) / 100 * H)
                            for it in rug_items})
        for it in items:
            x, y, a = placed_alpha(it)
            bottom, bx0, bx1 = alpha_bottom(x, y, a)
            overlap = post_overlap(x, y, a, posts)
            on_furniture = not on_rug(it)
            if on_furniture:
                hit = nearest_shelf(bottom, bx0, bx1, shelves)
                if hit is None:
                    ok_support = False
                    support = "NO SHELF"
                else:
                    dist, sy, *_ = hit
                    # signed: positive = sole below the detected top edge (into the board)
                    signed = bottom - sy
                    ok_support = -TOL_ABOVE <= signed <= TOL_BELOW
                    support = f"shelf y={sy} signed={signed}"
            else:
                # must sit on a rug-row baseline (or the rug surface itself)
                d_row = min((abs(bottom - b) for b in baselines), default=999)
                d_rug = bottom - rug_y  # should be well below rug top; contact is the row baseline
                ok_support = d_row <= 8
                support = f"row-baseline d={d_row} rug_top={rug_y}"
            ok_post = overlap <= POST_OVERLAP_MAX
            status = "OK" if ok_support and ok_post else "FAIL"
            details.append(f"{status} {zid}/{it['id']} bottom={bottom} {support} post_px={overlap}")
            if not ok_support:
                errors.append(f"{zid}/{it['id']}: bottom y={bottom} is not on a support ({support})")
            if not ok_post:
                errors.append(f"{zid}/{it['id']}: overlaps a rack post by {overlap} opaque px")
    return errors, details, {zid: furniture(plate_rgb(zones[zid]), rug_top_y(plate_rgb(zones[zid])))
                             for zid in ("shoes-a", "shoes-b", "shoes-c", "shoes-d", "shoes-e")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    errors, details, _ = check(catalog)
    print("\n".join(details))
    if errors:
        print(f"\n{len(errors)} failure(s):")
        print("\n".join(errors))
        sys.exit(1)
    print("\nall shoes sit on a support surface; no rack-post overlaps")


if __name__ == "__main__":
    main()
