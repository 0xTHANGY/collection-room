#!/usr/bin/env python3
"""Repack the totes category onto one empty plate (display coords only).

Does not change extraction layouts or card images. No model image generation:
the empty plate is built with the existing v21 wall reconstruct (bags path)
plus a surgical rail cleanup. Existing layer webps are reused.

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

SCALE = 0.54
# hook tip sits slightly below the top of the rail (matches original totes-a placements)
RAIL_TOP = dict(hook_y=72, rail_bottom=96, x0=230, x1=1335)
RAIL_BOT = dict(hook_y=478, rail_bottom=501, x0=230, x1=1315)
HOOK_XS_TOP = (357, 640, 920, 1207)
HOOK_XS_BOT = (358, 639, 927, 1205)
MIN_GAP = 10


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


def build_plate():
    rgb = freeze_painting()
    plate = reconstruct_plate(rgb)
    plate = surgical_remove_hooks(plate, HOOK_XS_TOP, 63, 96, 203, 1361, below=20, halfw=18)
    plate = surgical_remove_hooks(plate, HOOK_XS_BOT, 469, 501, 201, 1338, below=28, halfw=18)
    plate = fill_under_rail_patches(plate, HOOK_XS_BOT, 501, 201, 1338)
    plate = even_under_rail(plate, y_shadow_end=504, x0=230, x1=1318, wall_row=520, n=15)
    return plate


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
    plate = build_plate()
    recs = load_layers(catalog)
    if len(recs) != 15:
        raise SystemExit(f"expected 15 totes, found {len(recs)}")
    top, bot = recs[:8], recs[8:]
    p1, g1 = pack_row(top, RAIL_TOP, SCALE)
    p2, g2 = pack_row(bot, RAIL_BOT, SCALE)
    if g1 < MIN_GAP or g2 < MIN_GAP:
        raise SystemExit(f"gaps too tight: top={g1:.1f} bottom={g2:.1f}")
    placed = p1 + p2
    composited = composite(plate, placed)
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
        "gaps_px": {"top": g1, "bottom": g2},
        "pack": "8 top + 7 bottom, number order today-455..today-469",
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
                "top_gap_px": round(g1, 2),
                "bottom_gap_px": round(g2, 2),
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
