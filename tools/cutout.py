#!/usr/bin/env python3
"""Cut per-item transparent WebP layers from a furnished room vs an empty room.

Given a pixel-aligned pair of 1280x720 illustrations — one with collection
items and one without — build a difference mask for each item in items.json,
clean it, crop a tight RGBA sprite, and (optionally) write layer metadata
back into the JSON.

The originals are only read. Output files are written under --out.

Typical usage (from the collection-room directory):

    python3 site/tools/cutout.py \\
        --full incoming/style-v2-03-pinkchair.png \\
        --empty incoming/style-v2-05-empty-pinkchair.png \\
        --items site/data/items.json \\
        --out site/assets/layers \\
        --update-json \\
        --report /tmp/cutout/report.json

Hats expand downward so the peg under the cap is kept. Hanging clothes
expand upward so the hanger is kept. Adjacent items are assigned
exclusively (a pixel belongs to at most one sprite). Edges are feathered
1–2 px.

Requires: Python 3, Pillow, numpy, OpenCV (cv2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HANG_CATEGORIES = {"hat", "top", "knitwear", "dress"}
REST_CATEGORIES = {"bottoms", "bag"}

# Search-box padding around each hotspot, in pixels at 1280x720.
PADDING = {
    "hat": dict(left=16, right=16, top=8, bottom=10),
    "top": dict(left=40, right=40, top=96, bottom=28),
    "knitwear": dict(left=40, right=40, top=96, bottom=28),
    "dress": dict(left=36, right=36, top=96, bottom=24),
    "bottoms": dict(left=16, right=16, top=12, bottom=14),
    "bag": dict(left=18, right=18, top=12, bottom=16),
}

# Difference threshold (mean absolute RGB, 0–255) to seed the mask.
THRESHOLD = 22
# Weaker pixels attached by closing / flood can stay if >= this.
KEEP_MIN = 10
FEATHER_PX = 1.6
# Morph kernels (odd integers).
OPEN_K = 3
CLOSE_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", required=True, type=Path, help="Furnished room PNG (read-only).")
    p.add_argument("--empty", required=True, type=Path, help="Empty room PNG, pixel-aligned (read-only).")
    p.add_argument("--items", required=True, type=Path, help="items.json with hotspot rectangles.")
    p.add_argument("--out", required=True, type=Path, help="Directory for layer WebP files.")
    p.add_argument("--update-json", action="store_true", help="Write layer metadata back into --items.")
    p.add_argument("--report", type=Path, help="Optional JSON quality report path.")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--feather", type=float, default=FEATHER_PX)
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def hotspot_px(hotspot: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round(hotspot["left"] / 100 * width))
    y0 = int(round(hotspot["top"] / 100 * height))
    x1 = int(round((hotspot["left"] + hotspot["width"]) / 100 * width))
    y1 = int(round((hotspot["top"] + hotspot["height"]) / 100 * height))
    return x0, y0, x1, y1


def clamp_box(x0, y0, x1, y1, w, h):
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def expand_box(hot, category, w, h):
    pad = PADDING.get(category, PADDING["bag"])
    x0, y0, x1, y1 = hotspot_px(hot, w, h)
    return clamp_box(x0 - pad["left"], y0 - pad["top"], x1 + pad["right"], y1 + pad["bottom"], w, h)


def motion_for(category: str) -> str:
    if category in HANG_CATEGORIES:
        return "hang"
    return "rest"



def build_item_masks(full: np.ndarray, diff: np.ndarray, items: list, threshold: float) -> list[np.ndarray]:
    """Seeded watershed: each hotspot core grows through changed pixels,
    following colour boundaries so neighbouring garments stay separate."""
    h, w = diff.shape[:2]
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    openk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    fg = (diff >= threshold).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, openk, iterations=1)
    # Allow slightly weaker pixels next to the foreground (shadows, anti-alias).
    weak = (diff >= KEEP_MIN).astype(np.uint8) * 255
    fg = np.where(cv2.dilate(fg, close, iterations=1) & weak, 255, fg).astype(np.uint8)

    markers = np.zeros((h, w), np.int32)
    cores = []
    for i, item in enumerate(items):
        x0, y0, x1, y1 = hotspot_px(item["hotspot"], w, h)
        # Inset left/right only so overlapping neighbour sleeves are not
        # used as seeds. Do not inset the top: the hanger hook must stay
        # closer to this seed than to the rail (background).
        inset = 4 if item.get("category") == "hat" else 16
        sx0, sy0, sx1, sy1 = clamp_box(x0 + inset, y0, x1 - inset, y1, w, h)
        if sx1 - sx0 < 8:
            sx0, sy0, sx1, sy1 = x0, y0, x1, y1
        seed = np.zeros((h, w), np.uint8)
        seed[sy0:sy1, sx0:sx1] = fg[sy0:sy1, sx0:sx1]
        eroded = cv2.erode(seed, openk, iterations=1)
        if eroded.sum() == 0:
            eroded = seed
        cores.append(eroded)
        markers[eroded > 0] = i + 1

    bg = (diff < KEEP_MIN).astype(np.uint8) * 255
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, openk, iterations=1)
    # Don't mark as background anything inside a core.
    for core in cores:
        bg[core > 0] = 0
    markers[bg > 0] = 255

    # Restrict watershed to a bounding area around all search boxes so it
    # cannot wander across the whole wall.
    union = np.zeros((h, w), np.uint8)
    for item in items:
        x0, y0, x1, y1 = expand_box(item["hotspot"], item.get("category", ""), w, h)
        union[y0:y1, x0:x1] = 255
    # Pixels outside the union and not already marked become background.
    markers[(union == 0) & (markers == 0)] = 255

    ws = cv2.watershed(full, markers)
    masks = []
    for i, item in enumerate(items):
        keep = np.zeros((h, w), np.uint8)
        keep[ws == i + 1] = 255
        # Watershed leaves a 1px boundary (-1); attach those if they touch us and are in fg.
        boundary = (ws == -1) & (fg > 0)
        dil = cv2.dilate(keep, openk, iterations=1)
        keep = np.where(boundary & (dil > 0), 255, keep).astype(np.uint8)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, close, iterations=1)

        if item.get("category") == "hat":
            hx0, hy0, hx1, hy1 = hotspot_px(item["hotspot"], w, h)
            cx = int(round((hx0 + hx1) / 2))
            peg = np.zeros((h, w), np.uint8)
            cv2.circle(peg, (cx, 108), 8, 255, -1)
            keep = np.where((peg > 0) & (diff >= 4), 255, keep).astype(np.uint8)

        # Final strip of truly unchanged furniture, except the hat peg.
        protect = np.zeros((h, w), np.uint8)
        if item.get("category") == "hat":
            hx0, hy0, hx1, hy1 = hotspot_px(item["hotspot"], w, h)
            cv2.circle(protect, (int(round((hx0 + hx1) / 2)), 108), 8, 255, -1)
        keep = np.where((diff < KEEP_MIN) & (protect == 0), 0, keep).astype(np.uint8)
        # Drop tiny specks
        num, labels, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
        cleaned = np.zeros_like(keep)
        for lab in range(1, num):
            if stats[lab, cv2.CC_STAT_AREA] >= 30:
                cleaned[labels == lab] = 255
        masks.append(cleaned)
    return masks



def feather_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Return float alpha 0..1 with a 1–2 px soft edge."""
    if radius <= 0:
        return mask.astype(np.float32) / 255.0
    k = max(3, int(round(radius * 2) * 2 + 1))
    soft = cv2.GaussianBlur(mask, (k, k), radius)
    return np.clip(soft.astype(np.float32) / 255.0, 0, 1)


def crop_layer(full: np.ndarray, alpha: np.ndarray, pad: int = 2):
    ys, xs = np.nonzero(alpha > 0.02)
    if len(xs) == 0:
        raise RuntimeError("empty mask")
    h, w = alpha.shape
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    rgb = full[y0:y1, x0:x1]
    a = (alpha[y0:y1, x0:x1] * 255.0).round().astype(np.uint8)
    rgba = np.dstack([rgb, a])
    return rgba, (x0, y0, x1, y1)


def pivot_for(item: dict, box, width: int, height: int) -> tuple[float, float]:
    """Pivot as a percentage of the cropped layer box. Hang = top-center hang point."""
    x0, y0, x1, y1 = box
    lw, lh = max(1, x1 - x0), max(1, y1 - y0)
    motion = motion_for(item.get("category", ""))
    if motion != "hang":
        return 50.0, 50.0
    # Top-center of the opaque region: for hats this is near the peg/crown,
    # for clothes near the hanger hook.
    hx0, hy0, hx1, hy1 = hotspot_px(item["hotspot"], width, height)
    # Horizontal: center of the hotspot (the hang point is mid-item).
    px = ((hx0 + hx1) / 2 - x0) / lw * 100
    if item.get("category") == "hat":
        # Peg sits near y=108, which should be close to the top-center hang.
        py = (min(max(108, y0), y1) - y0) / lh * 100
        # If the crown is above the peg, pivot at the peg; otherwise at 8% from top.
        py = min(py, 18.0)
    else:
        # Hanger hook is near the top of the cutout.
        py = 6.0
    return float(np.clip(px, 0, 100)), float(np.clip(py, 0, 100))


def save_webp(rgba: np.ndarray, path: Path) -> None:
    im = Image.fromarray(rgba, "RGBA")
    # Pillow WebP saves without EXIF unless info is passed.
    im.save(path, "WEBP", quality=88, method=6, lossless=False)
    # Also try to avoid any metadata keys.
    out = Image.open(path)
    if out.info.get("exif"):
        raise RuntimeError(f"EXIF present in {path}")


def composite_and_score(empty: np.ndarray, full: np.ndarray, layers: list[tuple[np.ndarray, tuple[int, int, int, int]]]):
    """Alpha-over layers onto empty; return metrics vs the furnished room."""
    comp = empty.astype(np.float32)
    for rgba, (x0, y0, x1, y1) in layers:
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        dest = comp[y0:y1, x0:x1]
        comp[y0:y1, x0:x1] = rgb * a + dest * (1.0 - a)
    comp_u = np.clip(comp, 0, 255).astype(np.uint8)
    err = np.abs(comp.astype(np.int16) - full.astype(np.int16)).mean(axis=2)
    mae = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mse = float(((comp - full.astype(np.float32)) ** 2).mean())
    psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else 99.0
    return comp_u, {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "psnr": round(psnr, 3),
        "pct_diff_gt_10": round(float((err > 10).mean()) * 100, 3),
        "pct_diff_gt_25": round(float((err > 25).mean()) * 100, 3),
    }


def main() -> int:
    args = parse_args()
    full = load_rgb(args.full)
    empty = load_rgb(args.empty)
    if full.shape != empty.shape:
        print(f"size mismatch: full {full.shape} vs empty {empty.shape}", file=sys.stderr)
        return 2
    h, w = full.shape[:2]
    items = json.loads(args.items.read_text())
    if not isinstance(items, list):
        print("items.json must be a list", file=sys.stderr)
        return 2

    diff = np.abs(full.astype(np.int16) - empty.astype(np.int16)).mean(axis=2)
    masks = build_item_masks(full, diff, items, args.threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    layer_records = []
    cropped = []
    per_item = []

    for item, mask in zip(items, masks):
        alpha = feather_mask(mask, args.feather)
        coverage = float((alpha > 0.05).sum())
        if coverage < 20:
            print(f"warning: almost empty mask for {item.get('id')}", file=sys.stderr)
        rgba, box = crop_layer(full, alpha)
        x0, y0, x1, y1 = box
        stem = Path(item["image"]).stem if "image" in item else item["id"]
        # Keep a stable, short layer filename.
        name = f"{item['id']}-layer.webp"
        dest = args.out / name
        save_webp(rgba, dest)
        px, py = pivot_for(item, box, w, h)
        rel = f"assets/layers/{name}"
        rec = {
            "image": rel,
            "left": round(x0 / w * 100, 3),
            "top": round(y0 / h * 100, 3),
            "width": round((x1 - x0) / w * 100, 3),
            "height": round((y1 - y0) / h * 100, 3),
            "pivotX": round(px, 2),
            "pivotY": round(py, 2),
            "motion": motion_for(item.get("category", "")),
        }
        layer_records.append(rec)
        cropped.append((rgba, box))
        opaque = int((rgba[..., 3] > 8).sum())
        per_item.append({
            "id": item["id"],
            "file": name,
            "bytes": dest.stat().st_size,
            "size_px": [int(x1 - x0), int(y1 - y0)],
            "opaque_pixels": opaque,
            "box_pct": rec,
        })
        print(f"{item['id']:7} {name:22} {x1-x0}x{y1-y0}  {dest.stat().st_size:6d}B  motion={rec['motion']}  pivot={px:.1f},{py:.1f}")

    comp, scores = composite_and_score(empty, full, cropped)
    print("composite vs furnished:", scores)
    report = {"scores": scores, "items": per_item, "full": str(args.full), "empty": str(args.empty)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        Image.fromarray(comp).save(args.report.with_suffix(".png"))
        err = np.abs(comp.astype(np.int16) - full.astype(np.int16)).mean(axis=2)
        heat = np.clip(err * 6, 0, 255).astype(np.uint8)
        Image.fromarray(heat).save(args.report.with_name(args.report.stem + "-residual.png"))

    if args.update_json:
        for item, rec in zip(items, layer_records):
            item["layer"] = rec
        args.items.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {args.items}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
