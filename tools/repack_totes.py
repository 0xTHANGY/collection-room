#!/usr/bin/env python3
"""Repack the totes category onto one empty plate with three cloned rails (display coords only).

Does not change extraction layouts or card images. No model image generation:
the empty plate is built with the existing v21 wall reconstruct (bags path),
surgical hook cleanup, then a third rail cloned from the existing rod pixels.
Existing layer webps are reused.

Safe to re-run: the original totes-a painting is frozen at
data/source/display/totes-a-painting.webp on the first run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT / "data/source/display"
PAINTING = DISPLAY / "totes-a-painting.webp"
LAYOUT_A = ROOT / "data/source/layout/totes-a.json"
CATALOG = ROOT / "data/catalog.json"
OUT_JSON = DISPLAY / "totes-repack.json"

SCALE = 0.62
# hook tip sits slightly below the top of the rail (matches original totes-a placements)
RAIL_X0, RAIL_X1 = 230, 1335  # usable hang span inside the end caps
HOOK_XS_TOP = (357, 640, 920, 1207)
HOOK_XS_BOT = (358, 639, 927, 1205)
MIN_GAP = 10
# 3-rail layout: wood tops evenly spaced; hook_y = wood_top + 9
WOOD_Y0 = 58
WOOD_SPACING = 260
HOOK_FROM_WOOD = 9
RAIL_WOOD_H = 34  # original wood y=63..97


def save_webp(im: Image.Image, path: Path, quality=80):
    """Write a metadata-free webp (same approach as tools/build_v2.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Image.fromarray(np.asarray(im.convert("RGB") if im.mode != "RGBA" else im))
    temporary = path.with_suffix(path.suffix + ".tmp")
    if clean.mode == "RGBA":
        clean.save(temporary, "WEBP", quality=quality, method=4)
    else:
        clean.save(temporary, "WEBP", quality=quality, method=4)
    temporary.replace(path)


def variants_from(im: Image.Image, stem: str, sizes):
    """Write stem-{size}.webp like build_v2.variants (scale by width)."""
    result = []
    for size in sizes:
        scale = min(1, size / im.width)
        out = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
            Image.Resampling.LANCZOS,
        )
        path = ROOT / f"{stem}-{size}.webp"
        save_webp(out, path)
        result.append({"src": f"{stem}-{size}.webp", "width": out.width, "height": out.height})
    return result


def freeze_painting():
    DISPLAY.mkdir(parents=True, exist_ok=True)
    src = ROOT / "assets/zones/totes-a-1600.webp"
    if not PAINTING.exists():
        im = Image.open(src).convert("RGB")
        save_webp(im, PAINTING, quality=90)
        print("froze painting", PAINTING)
    else:
        print("using frozen painting", PAINTING)
    return np.array(Image.open(PAINTING).convert("RGB"))


def reconstruct_plate(rgb):
    sys.path.insert(0, str(ROOT / "tools"))
    from v21_images import reconstruct

    layout = json.loads(LAYOUT_A.read_text())
    h, w = rgb.shape[:2]
    lw, lh = layout["image_size"]
    boxes = {}
    for iid, entry in layout["items"].items():
        x0, y0, x1, y1 = entry["bbox"]
        boxes[iid] = [
            round(x0 * w / lw),
            round(y0 * h / lh),
            round(x1 * w / lw),
            round(y1 * h / lh),
        ]
    # Use the bags wall path (totes is the same hook-rail furniture). Official
    # zid "totes-a" fails a bench-boundary gate on 3 items and leaves bags behind.
    plate, masks, reasons = reconstruct(rgb, boxes, "bags-totes-a")
    failed = {k: v for k, v in reasons.items() if v}
    if failed:
        raise SystemExit(f"plate reconstruct failed: {failed}")
    return plate


def surgical_remove_hooks(im, hooks, y0, y1, x_rail0, x_rail1, below=24, halfw=18):
    out = im.copy()
    h, w = out.shape[:2]
    for hx in hooks:
        x0 = max(x_rail0 + 30, hx - halfw)
        x1 = min(x_rail1 - 30, hx + halfw)
        for y in range(y0, y1 + 1):
            left = out[y, max(x_rail0, x0 - 18) : x0]
            right = out[y, x1 + 1 : min(x_rail1, x1 + 19)]
            if len(left) == 0 and len(right) == 0:
                continue
            if len(left) and len(right):
                el, er = left[-1].astype(float), right[0].astype(float)
                t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
                out[y, x0 : x1 + 1] = np.clip(el * (1 - t) + er * t, 0, 255).astype(np.uint8)
            else:
                donors = left if len(left) else right
                out[y, x0 : x1 + 1] = np.median(donors, 0)
        for y in range(y1 + 1, min(h, y1 + 1 + below)):
            left = out[y, max(0, x0 - 40) : x0]
            right = out[y, x1 + 1 : min(w, x1 + 41)]
            if len(left) == 0 and len(right) == 0:
                continue
            if len(left) and len(right):
                el, er = left[-1].astype(float), right[0].astype(float)
                t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
                fill = np.clip(el * (1 - t) + er * t, 0, 255).astype(np.uint8)
            else:
                donors = left if len(left) else right
                fill = np.median(donors, 0)
                if fill.ndim == 1:
                    fill = np.repeat(fill[None, :], x1 - x0 + 1, 0)
            region = out[y, x0 : x1 + 1]
            neigh = np.concatenate(
                [left[-5:] if len(left) else region, right[:5] if len(right) else region]
            )
            nmed = np.median(neigh, 0)
            diff = np.linalg.norm(region.astype(float) - nmed, axis=1)
            target = fill if fill.ndim > 1 else np.repeat(np.asarray(fill)[None, :], x1 - x0 + 1, 0)
            region[diff > 8] = target[diff > 8]
            out[y, x0 : x1 + 1] = region
    return out


def fill_under_rail_patches(im, hooks, y1, x_rail0, x_rail1, depth=28, halfw=36):
    """Remove leftover mid-grey reconstruct patches just under the rail at old hook xs."""
    out = im.copy()
    h, w = out.shape[:2]
    for hx in hooks:
        x0 = max(x_rail0 + 20, hx - halfw)
        x1 = min(x_rail1 - 20, hx + halfw)
        for y in range(y1 + 1, min(h, y1 + 1 + depth)):
            left = out[y, max(0, x0 - 40) : x0]
            right = out[y, x1 + 1 : min(w, x1 + 41)]
            if len(left) == 0 and len(right) == 0:
                continue
            if len(left) and len(right):
                el, er = left[-1].astype(float), right[0].astype(float)
                t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
                fill = np.clip(el * (1 - t) + er * t, 0, 255).astype(np.uint8)
            else:
                donors = left if len(left) else right
                fill = np.repeat(np.median(donors, 0)[None, :], x1 - x0 + 1, 0)
            region = out[y, x0 : x1 + 1]
            neigh = np.concatenate(
                [left[-8:] if len(left) else region, right[:8] if len(right) else region]
            )
            nmed = np.median(neigh, 0)
            diff = np.linalg.norm(region.astype(float) - nmed, axis=1)
            region[diff > 10] = fill[diff > 10]
            out[y, x0 : x1 + 1] = region
    return out



def even_under_rail(im, y_shadow_end, x0, x1, wall_row, n=15):
    """After the short natural contact shadow, replace blotchy reconstruct
    patches with a copy of a clean wall row (keeps left-right illumination)."""
    out = im.copy()
    wall = out[wall_row, x0:x1+1].copy()
    for dy in range(n):
        y = y_shadow_end + 1 + dy
        t = min(1.0, dy / 3.0)  # 3-row fade out of the mid-shadow
        fade = (1 - t) * out[y_shadow_end, x0:x1+1].astype(float) + t * wall.astype(float)
        out[y, x0:x1+1] = np.clip(fade, 0, 255).astype(np.uint8)
    return out



def extract_rail_sprite(plate):
    """Cut the top rail (wood, end caps, contact shadow) as an RGBA sprite."""
    y0, y1 = 63, 97
    sy0, sy1 = 58, 108
    sx0, sx1 = 193, 1372
    crop = plate[sy0 : sy1 + 1, sx0 : sx1 + 1].copy()
    ch, cw = crop.shape[:2]
    r, g, b = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    chroma = crop.max(2).astype(int) - crop.min(2)
    lum = crop.mean(2)
    wood = (r > b + 18) & (r > 55) & (r < 210) & (g > 35) & (g < 170)
    wood_mask = cv2.dilate(np.uint8(wood) * 255, np.ones((3, 3), np.uint8))
    shadow = np.zeros_like(wood_mask)
    wood_bottom_local = y1 - sy0
    for yy in range(wood_bottom_local + 1, ch):
        dark = (lum[yy] < 175) | (chroma[yy] > 18)
        near = cv2.dilate(wood_mask, np.ones((25, 5), np.uint8))[yy] > 0
        shadow[yy] = np.uint8(dark & near) * 255
    above = np.zeros_like(wood_mask)
    for yy in range(0, max(0, y0 - sy0)):
        near = cv2.dilate(wood_mask, np.ones((7, 3), np.uint8))[yy] > 0
        odd = (np.abs(lum[yy] - 198) > 8) | (chroma[yy] > 16)
        above[yy] = np.uint8(odd & near) * 255
    alpha = np.maximum.reduce([wood_mask, shadow, above])
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha_f = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    sprite = Image.fromarray(np.dstack([crop, alpha_f]))
    return sprite, sx0, sy0, y0 - sy0


def erase_rail_band(img, ey0, ey1, ex0, ex1, donor_above=12, donor_below=16):
    """Fill a horizontal band with a vertical lerp of wall rows outside the rail."""
    out = img.copy()
    h = out.shape[0]
    y_top = max(0, ey0 - donor_above)
    y_bot = min(h - 1, ey1 + donor_below)
    top = np.median(out[max(0, y_top - 5) : y_top, ex0 : ex1 + 1], axis=0)
    bot = np.median(out[y_bot + 1 : min(h, y_bot + 6), ex0 : ex1 + 1], axis=0)
    span = max(1, y_bot - y_top)
    for y in range(y_top, y_bot + 1):
        t = (y - y_top) / span
        out[y, ex0 : ex1 + 1] = np.clip((1 - t) * top + t * bot, 0, 255).astype(np.uint8)
    return out


def add_three_rails(two_rail_plate):
    """Erase the original two rods and paste three clones of the top rod."""
    sprite, sx0, sy0, wood_off = extract_rail_sprite(two_rail_plate)
    wall = two_rail_plate.copy()
    wall = erase_rail_band(wall, 58, 108, 188, 1380, donor_above=12, donor_below=16)
    wall = erase_rail_band(wall, 462, 518, 188, 1380, donor_above=16, donor_below=16)
    wood_ys = [WOOD_Y0, WOOD_Y0 + WOOD_SPACING, WOOD_Y0 + 2 * WOOD_SPACING]
    canvas = Image.fromarray(wall).convert("RGBA")
    rails = []
    for yt in wood_ys:
        canvas.alpha_composite(sprite, (sx0, yt - wood_off))
        rails.append(
            dict(
                hook_y=yt + HOOK_FROM_WOOD,
                rail_bottom=yt + RAIL_WOOD_H,
                x0=RAIL_X0,
                x1=RAIL_X1,
                wood_top=yt,
            )
        )
    return np.array(canvas.convert("RGB")), rails


def build_two_rail_plate():
    rgb = freeze_painting()
    plate = reconstruct_plate(rgb)
    plate = surgical_remove_hooks(plate, HOOK_XS_TOP, 63, 96, 203, 1361, below=20, halfw=18)
    plate = surgical_remove_hooks(plate, HOOK_XS_BOT, 469, 501, 201, 1338, below=28, halfw=18)
    plate = fill_under_rail_patches(plate, HOOK_XS_BOT, 501, 201, 1338)
    plate = even_under_rail(plate, y_shadow_end=504, x0=230, x1=1318, wall_row=520, n=15)
    return plate


def build_plate():
    two = build_two_rail_plate()
    return add_three_rails(two)


def load_layers(catalog):
    recs = []
    for item in catalog["items"]:
        if item["category"] != "totes":
            continue
        arr = np.array(Image.open(ROOT / item["layer"]["image"]).convert("RGBA"))
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 80)
        ax = float(xs[ys <= ys.min() + 8].mean()) if len(xs) else arr.shape[1] / 2
        ay = float(ys.min()) if len(ys) else 0
        top_width = int((alpha[1:9] > 80).sum(1).max()) if alpha.shape[0] > 8 else 0
        kind = "handle" if top_width > arr.shape[1] * 0.4 else "hook"
        recs.append(
            dict(
                id=item["id"],
                item=item,
                arr=arr,
                w=arr.shape[1],
                h=arr.shape[0],
                ax=ax,
                ay=ay,
                kind=kind,
                src_zone=item["zone"],
            )
        )
    recs.sort(key=lambda r: r["id"])
    return recs


def pack_row(row, rail, scale):
    widths = [it["w"] * scale for it in row]
    usable = rail["x1"] - rail["x0"]
    remain = usable - sum(widths)
    n = len(row)
    gap = remain / (n + 1) if n else 0
    x = rail["x0"] + gap
    placed = []
    for it, ww in zip(row, widths):
        if it["kind"] == "handle":
            top = rail["rail_bottom"] + 4 - it["ay"] * scale
        else:
            top = rail["hook_y"] - it["ay"] * scale
        placed.append(
            dict(
                rec=it,
                left=x,
                top=top,
                scale=scale,
                ww=ww,
                hh=it["h"] * scale,
                hook_x=x + it["ax"] * scale,
                gap=gap,
                kind=it["kind"],
            )
        )
        x += ww + gap
    return placed, gap


def clip_polygon(alpha):
    mask = np.uint8(alpha > 40)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [[0, 0], [100, 0], [100, 100], [0, 100]]
    cnt = max(contours, key=cv2.contourArea)
    # aim for a modest polygon
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, max(1.5, 0.008 * peri), True)
    if len(approx) < 3:
        x, y, w, h = cv2.boundingRect(cnt)
        return [
            [100 * x / alpha.shape[1], 100 * y / alpha.shape[0]],
            [100 * (x + w) / alpha.shape[1], 100 * y / alpha.shape[0]],
            [100 * (x + w) / alpha.shape[1], 100 * (y + h) / alpha.shape[0]],
            [100 * x / alpha.shape[1], 100 * (y + h) / alpha.shape[0]],
        ]
    h, w = alpha.shape
    pts = []
    for x, y in approx.reshape(-1, 2):
        pts.append(
            [
                float(np.clip(100.0 * x / w, 0, 100)),
                float(np.clip(100.0 * y / h, 0, 100)),
            ]
        )
    return pts


def composite(plate, placed):
    canvas = Image.fromarray(plate).convert("RGBA")
    for i, p in enumerate(placed):
        im = Image.fromarray(p["rec"]["arr"])
        nw, nh = max(1, round(p["ww"])), max(1, round(p["hh"]))
        rs = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.alpha_composite(rs.convert("RGBA"), (round(p["left"]), round(p["top"])))
        p["nw"], p["nh"] = nw, nh
        p["x"], p["y"] = round(p["left"]), round(p["top"])
        p["clip"] = clip_polygon(np.array(rs)[:, :, 3])
        p["zIndex"] = i + 1
    return canvas.convert("RGB")


def pct(box, w, h):
    x, y, bw, bh = box
    return dict(left=x / w * 100, top=y / h * 100, width=bw / w * 100, height=bh / h * 100)


def apply(catalog=None):
    catalog = catalog or json.loads(CATALOG.read_text())
    plate, rails = build_plate()
    recs = load_layers(catalog)
    if len(recs) != 15:
        raise SystemExit(f"expected 15 totes, found {len(recs)}")
    rows = [recs[0:5], recs[5:10], recs[10:15]]
    packed, gaps = [], []
    for row, rail in zip(rows, rails):
        pl, g = pack_row(row, rail, SCALE)
        packed.extend(pl)
        gaps.append(g)
    if any(g < MIN_GAP for g in gaps):
        raise SystemExit(f"gaps too tight: {gaps}")
    placed = packed
    composited = composite(plate, placed)
    g1, g2, g3 = gaps
    h, w = plate.shape[:2]

    # Zone images are the EMPTY plate (layers sit on top in the browser).
    plate_im = Image.fromarray(plate)
    images = variants_from(plate_im, "assets/zones/totes-a", [1600, 1024, 640])
    # Cover is plate + packed layers, metadata-free.
    cover_im = composited.resize(
        (640, max(1, round(640 * composited.height / composited.width))),
        Image.Resampling.LANCZOS,
    )
    cover_path = ROOT / "assets/zones/totes-a-cover-640.webp"
    save_webp(cover_im, cover_path)
    cover = {"src": "assets/zones/totes-a-cover-640.webp", "width": cover_im.width, "height": cover_im.height}

    # Update catalog: merge b into a, display coords, outline clips.
    by_id = {i["id"]: i for i in catalog["items"]}
    display_items = {}
    for p in placed:
        item = by_id[p["rec"]["id"]]
        box = pct((p["x"], p["y"], p["nw"], p["nh"]), w, h)
        layer = dict(item["layer"])
        layer.update(box)
        layer["rect"] = dict(box)
        layer["fallback"] = False
        layer["motion"] = "hang"
        layer["pivot"] = {"x": 50, "y": 0}
        layer["pivotX"] = 50
        layer["pivotY"] = 0
        layer["zIndex"] = p["zIndex"]
        item["zone"] = "totes-a"
        item["layer"] = layer
        item["hotspot"] = dict(box)
        item["hotspot"]["clip"] = p["clip"]
        display_items[item["id"]] = {
            "source_zone": p["rec"]["src_zone"],
            "kind": p["kind"],
            "left": box["left"],
            "top": box["top"],
            "width": box["width"],
            "height": box["height"],
            "zIndex": p["zIndex"],
            "scale": SCALE,
        }

    tote_ids = [r["id"] for r in recs]
    zones = []
    for z in catalog["zones"]:
        if z["id"] == "totes-a":
            z = dict(z)
            z["images"] = images
            z["cover"] = cover
            z["items"] = tote_ids
            z["natural_size"] = [w, h]
            z["aspect"] = w / h
            z["has_layout"] = True
            zones.append(z)
        elif z["id"] == "totes-b":
            continue
        else:
            zones.append(z)
    catalog["zones"] = zones

    for cat in catalog["categories"]:
        if cat["id"] == "totes":
            cat["count"] = 15
            cat["cover_zone"] = "totes-a"

    catalog["redirects"] = dict(catalog.get("redirects") or {})
    catalog["redirects"]["totes/b"] = "#/totes/a"
    catalog["redirects"]["totes/totes-b"] = "#/totes/a"

    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")

    record = {
        "scale": SCALE,
        "target_zone": "totes-a",
        "merged": ["totes-b"],
        "gaps_px": {"row0": g1, "row1": g2, "row2": g3},
        "pack": "5/5/5 on three cloned rails, number order today-455..today-469",
        "rails": rails,
        "handle_special": [r["id"] for r in recs if r["kind"] == "handle"],
        "redirects": catalog["redirects"],
        "items": display_items,
        "note": "Display coordinates only. Extraction bboxes in data/source/layout/ are unchanged.",
    }
    OUT_JSON.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "totes_zones": 1,
                "items": 15,
                "row_gaps_px": [round(g1, 2), round(g2, 2), round(g3, 2)],
                "scale": SCALE,
                "rails": [{"wood_top": r["wood_top"], "hook_y": r["hook_y"]} for r in rails],
                "handle_special": record["handle_special"],
                "cover": cover,
                "images": images,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return catalog


if __name__ == "__main__":
    apply()
