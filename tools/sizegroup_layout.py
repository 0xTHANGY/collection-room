#!/usr/bin/env python3
"""Scale Collection Room items to sourced real-world proportions, apply a
small middle-rod visual bump if needed, then cluster each rod by displayed
size. Layout/transform only: no generation, no pixel redraws.

Usage:
  python3 tools/sizegroup_layout.py --zones jackets-a
  python3 tools/sizegroup_layout.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("/workspace/collection-room-impl-report/sizegroup")
CANVAS_W, CANVAS_H = 1536.0, 1024.0
MARGIN_RAIL_PX = 10.0  # keep hems off the next rail's wood
MARGIN_FLOOR_PX = 8.0
WALL_MIN, WALL_MAX = 6.0, 94.0
GAP_MIN_PCT = 0.35
BUMP_THRESHOLD = 0.85  # "noticeably smaller" vs median neighbor height
BUMP_CAP = 0.15        # at most +15% over the real-size scale
MOVE_SCALE_RATIO = 0.70  # move to a lower rod if current rod caps scale below this fraction of the room's unconstrained median

# ---------------------------------------------------------------------------
# Sourced size-M (or published equivalent) garment / object lengths in cm.
# Every figure is taken from the cited URL. Variants are chosen from the item
# name only when the source chart itself lists that variant (mini/midi/…).
# ---------------------------------------------------------------------------

SOURCES = {
    "asos_blazer": "https://www.asos.com/us/discover/size-charts/men/blazers/",
    "doppiaa_jacket": "https://doppiaa.com/pages/doppiaa-size-guide-coat-jacket",
    "elganso_coat": "https://www2.elganso.com/en/guia-de-tallas",
    "uniqlo_vest": "https://www.uniqlo.com/eu-pl/en/products/E478270-000/00",
    "asos_tee": "https://www.asos.com/discover/size-charts/men/tshirts-polo-shirts/",
    "asos_hoodie": "https://www.asos.com/us/discover/size-charts/men/hoodies-sweatshirts/",
    "uniqlo_shirt": "https://www.uniqlo.com/jp/ja/size/418998_size.html",
    "asos_women": "https://www.asos.com/discover/size-charts/women/",
    "uniqlo_pants": "https://www.uniqlo.com/jp/ja/size/437872_size.html",
    "uniqlo_cropped": "https://www.uniqlo.com/jp/ja/size/432466_size.html",
    "uniqlo_shorts": "https://www.uniqlo.com/jp/ja/size/433109_size.html",
    "llbean_tote": "https://www.llbean.com/llb/shop/33381?page=boat-and-tote-open-top",
    "ikea_frakta": "https://www.ikea.com/gb/en/p/frakta-carrier-bag-medium-blue-60611111/",
}

# type -> default (length_cm, width_cm or None, source_key, note)
TYPE_DEFAULTS = {
    "vest": (66.5, 60.0, "uniqlo_vest", "Uniqlo PUFFERTECH vest size M: body length 66.5 cm, body width 60 cm"),
    "knit-vest": (66.5, 60.0, "uniqlo_vest", "Uniqlo PUFFERTECH vest size M length 66.5 cm (vest type)"),
    "sweater-vest": (66.5, 60.0, "uniqlo_vest", "Uniqlo PUFFERTECH vest size M length 66.5 cm (vest type)"),
    "blazer": (72.0, None, "asos_blazer", "ASOS regular size 40 in (M) blazer centre-back length 72 cm"),
    "jacket": (72.8, None, "doppiaa_jacket", "Doppiaa coat & jacket standard length, size 48 (M / UK 38) 72.8 cm"),
    "windbreaker": (72.8, None, "doppiaa_jacket", "Doppiaa standard jacket length size 48 (M) 72.8 cm"),
    "bomber": (72.8, None, "doppiaa_jacket", "Doppiaa standard jacket length size 48 (M) 72.8 cm"),
    "varsity": (72.8, None, "doppiaa_jacket", "Doppiaa standard jacket length size 48 (M) 72.8 cm"),
    "puffer": (72.8, None, "doppiaa_jacket", "Doppiaa standard jacket length size 48 (M) 72.8 cm"),
    "coat": (91.0, None, "elganso_coat", "El Ganso men's coat size M total length 91 cm"),
    "tee": (70.0, None, "asos_tee", "ASOS size M T-shirt / polo average centre-back length 70 cm"),
    "jersey": (70.0, None, "asos_tee", "ASOS size M T-shirt / polo average centre-back length 70 cm"),
    "ringer-tee": (70.0, None, "asos_tee", "ASOS size M T-shirt / polo average centre-back length 70 cm"),
    "polo": (70.0, None, "asos_tee", "ASOS size M T-shirt / polo average centre-back length 70 cm"),
    "long-sleeve-tee": (70.0, None, "asos_tee", "ASOS size M long & short sleeve T-shirt average length 70 cm"),
    "waffle-long-sleeve": (70.0, None, "asos_tee", "ASOS size M long & short sleeve T-shirt average length 70 cm"),
    "top": (70.0, None, "asos_tee", "ASOS size M top average centre-back length 70 cm"),
    "tank": (70.0, None, "asos_tee", "ASOS size M top average centre-back length 70 cm"),
    "sleeveless-top": (70.0, None, "asos_tee", "ASOS size M top average centre-back length 70 cm"),
    "shirt": (78.0, 56.5, "uniqlo_shirt", "Uniqlo shirt size M 着丈 (body length) 78 cm, 身幅 56.5 cm"),
    "blouse": (78.0, 56.5, "uniqlo_shirt", "Uniqlo shirt size M 着丈 78 cm (woven top)"),
    "hoodie": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average centre-back length 70 cm"),
    "zip-hoodie": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "sweatshirt": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "pullover": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "quarter-zip": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "fleece-pullover": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "technical-hoodie": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "polo-sweatshirt": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "short-sleeve-hoodie": (70.0, None, "asos_hoodie", "ASOS size M hoodie / sweatshirt average length 70 cm"),
    "sweater": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "cardigan": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "turtleneck": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "knit-polo": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "knit-tank": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "draped-knit-top": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "layered-knit": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "half-zip-sweater": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "short-sleeve-knit": (70.0, None, "asos_hoodie", "ASOS size M sweat / knit top average length 70 cm"),
    "pants": (99.5, None, "uniqlo_pants", "Uniqlo size M 総丈 (total length) 99.5 cm"),
    "jeans": (99.5, None, "uniqlo_pants", "Uniqlo size M 総丈 99.5 cm"),
    "cargo-pants": (99.5, None, "uniqlo_pants", "Uniqlo size M 総丈 99.5 cm"),
    "cropped-pants": (69.0, None, "uniqlo_cropped", "Uniqlo size M 総丈 69 cm (cropped / shorter pant)"),
    "shorts": (51.5, None, "uniqlo_shorts", "Uniqlo size M 総丈 51.5 cm"),
    "dress": (95.0, None, "asos_women", "ASOS main-range dress length, knee, size 10: 95 cm (published length table)"),
    "skirt": (55.0, None, "asos_women", "ASOS main-range skirt length, knee, size 10: 55 cm"),
    "canvas-tote": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium: 12 in H x 13 in W (30.48 x 33.02 cm), body only"),
    "tote": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium 12 in H (30.48 cm)"),
    "technical-tote": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium 12 in H (30.48 cm)"),
    "woven-shopper": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium 12 in H (30.48 cm)"),
    "patchwork-tote": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium 12 in H (30.48 cm)"),
    "organza-tote": (30.48, 33.02, "llbean_tote", "L.L.Bean Boat and Tote medium 12 in H (30.48 cm)"),
    "mini-tote": (26.67, 24.13, "llbean_tote", "L.L.Bean Boat and Tote small: 10.5 in H x 9.5 in W (26.67 x 24.13 cm)"),
    "organza-baguette": (26.67, 24.13, "llbean_tote", "L.L.Bean Boat and Tote small 10.5 in H (26.67 cm), used for mini bag body"),
    "shopping-bag": (45.0, 45.0, "ikea_frakta", "IKEA FRAKTA carrier bag medium: 45 x 18 x 45 cm (height 45 cm)"),
    "reusable-bag": (45.0, 45.0, "ikea_frakta", "IKEA FRAKTA carrier bag medium height 45 cm"),
}

# Types we do not rescale: no reliable published dimension that matches
# the displayed measurement (crown height, shoe last vs. standing height,
# coiled belt length, etc.). They are still clustered by current display size.
SKIP_RESCALE_TYPES = {
    "bag", "bucket-bag", "pouch", "dry-bag", "crossbody-pouch", "crossbody-bag",
    "sneakers", "flats", "shoes", "heels", "boots", "ankle boots", "loafers",
    "slides", "sandals", "mules", "slip-ons",
    "cap", "beanie", "bucket", "earflap", "newsboy", "balaclava", "beret",
    "brim", "boater", "other",
    "belt", "gloves", "tie", "scarf", "neck-warmer", "silk-scarf", "twilly",
    "headband", "card-holder", "passport-cover",
    "overalls", "jumpsuit",
    "cropped-tee", "crop-top", "knit-crop-top", "cropped-sweater",
    "cropped-sweatshirt", "short-sleeve-cropped-hoodie",
}

DRESS_LENGTHS = {
    "mini": (85.0, "ASOS main-range mini dress, size 10: 85 cm"),
    "midi": (90.0, "ASOS main-range midi dress, size 10: 90 cm"),
    "knee": (95.0, "ASOS main-range knee dress, size 10: 95 cm"),
    "calf": (100.0, "ASOS main-range calf dress, size 10: 100 cm"),
    "maxi": (142.0, "ASOS main-range maxi dress, size 10: 142 cm"),
}
SKIRT_LENGTHS = {
    "mini": (35.0, "ASOS main-range mini skirt, size 10: 35 cm"),
    "midi": (45.0, "ASOS main-range midi skirt, size 10: 45 cm"),
    "knee": (55.0, "ASOS main-range knee skirt, size 10: 55 cm"),
    "calf": (75.0, "ASOS main-range calf skirt, size 10: 75 cm"),
    "maxi": (95.0, "ASOS main-range maxi skirt, size 10: 95 cm"),
}


def name_lower(item):
    return " ".join([
        str(item.get("name") or ""),
        str(item.get("display_name") or ""),
        str(item.get("type") or ""),
    ]).lower()


def dress_or_skirt_variant(name: str, kind: str) -> str:
    n = name.lower()
    if "maxi" in n or "floor-length" in n or "floor length" in n:
        return "maxi"
    if "mini" in n:
        return "mini"
    if "midi" in n:
        return "midi"
    if "calf" in n or "ankle" in n:
        return "calf"
    if "knee" in n:
        return "knee"
    if "long" in n:
        return "maxi" if kind == "dress" else "calf"
    return "knee"


def measure_item(item):
    """Return (length_cm, width_cm, source_url, note) or None if skipped."""
    typ = item.get("type") or ""
    if typ in SKIP_RESCALE_TYPES:
        return None
    name = name_lower(item)
    if typ == "dress":
        var = dress_or_skirt_variant(name, "dress")
        length, note = DRESS_LENGTHS[var]
        return length, None, SOURCES["asos_women"], note
    if typ == "skirt":
        var = dress_or_skirt_variant(name, "skirt")
        length, note = SKIRT_LENGTHS[var]
        return length, None, SOURCES["asos_women"], note
    if typ in ("jacket", "windbreaker", "bomber", "varsity", "puffer") and (
        "long coat" in name or "long jacket" in name or "duster" in name
    ):
        return 91.5, None, SOURCES["doppiaa_jacket"], "Doppiaa long jacket/coat length, size 48 (M) 91.5 cm"
    if typ == "coat" and ("long" in name or "duster" in name or "maxi" in name):
        return 91.5, None, SOURCES["doppiaa_jacket"], "Doppiaa long jacket/coat length, size 48 (M) 91.5 cm"
    if typ in TYPE_DEFAULTS:
        length, width, key, note = TYPE_DEFAULTS[typ]
        return length, width, SOURCES[key], note
    return None


def hook_of(layer):
    left, top, w, h = layer["left"], layer["top"], layer["width"], layer["height"]
    px = layer.get("pivotX", 50.0)
    py = layer.get("pivotY", 0.0)
    return left + w * px / 100.0, top + h * py / 100.0, px, py


def is_shelf_rest(layer):
    """Shoes and other rest items (pivot near center) sit on a shelf.
    Accessories with motion=lift but pivotY~0 are still hooked at the top."""
    return (layer.get("motion") or "hang") == "lift" and float(layer.get("pivotY", 0.0)) >= 25.0


def load_rails(catalog):
    """Rail hook_y / wood_top in canvas pixels for every zone that has hang items."""
    rails = {}
    for p in (ROOT / "data/source/display").glob("*-repack.json"):
        data = json.loads(p.read_text())
        raw = data.get("rails")
        if isinstance(raw, dict):
            for zid, entries in raw.items():
                rails[zid] = [
                    {"wood_top": float(e["wood_top"]), "hook_y": float(e["hook_y"])}
                    for e in entries
                ]
        elif isinstance(raw, list) and "totes" in p.name:
            parsed = [
                {"wood_top": float(e["wood_top"]), "hook_y": float(e["hook_y"]),
                 "x0": e.get("x0"), "x1": e.get("x1")}
                for e in raw
            ]
            for letter in "abc":
                rails[f"totes-{letter}"] = parsed
    # Infer any catalog zone still missing rails from current hook clusters.
    items = {i["id"]: i for i in catalog["items"]}
    for zone in catalog["zones"]:
        zid = zone["id"]
        if zid in rails:
            continue
        hooks = []
        for iid in zone["items"]:
            it = items[iid]
            if it["layer"].get("motion") == "lift":
                # use bottom as baseline for rest items
                L = it["layer"]
                hooks.append(L["top"] + L["height"])
            else:
                hooks.append(hook_of(it["layer"])[1])
        if not hooks:
            continue
        hooks_px = sorted(h / 100.0 * CANVAS_H for h in hooks)
        clusters = []
        for y in hooks_px:
            if not clusters or y - clusters[-1][-1] > 0.08 * CANVAS_H:
                clusters.append([y])
            else:
                clusters[-1].append(y)
        inferred = []
        for c in clusters:
            hy = sum(c) / len(c)
            inferred.append({"wood_top": hy - 7.0, "hook_y": hy})
        rails[zid] = inferred
    return rails


def detect_floor_pct(zone):
    """Return a floor/rug limit as a % of canvas height, default 93."""
    images = zone.get("images") or []
    # Prefer the natural-size painting (usually the 1600/1536 file).
    path = None
    if images:
        cand = images[0]["src"] if images else None
        if cand and (ROOT / cand).exists():
            path = ROOT / cand
    if path is None:
        return 93.0
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    h, w = arr.shape[:2]
    # Look at the lower 25% for a wide, stable horizontal band (rug / cabinet
    # top). Use a simple row-to-row change detector near the bottom.
    start = int(h * 0.70)
    row_mean = arr[start:].mean(axis=1).mean(axis=1)
    # First strong change from the bottom upward often marks rug top.
    diffs = np.abs(np.diff(row_mean.astype(float)))
    # Smooth
    if len(diffs) >= 5:
        kernel = np.ones(5) / 5.0
        diffs = np.convolve(diffs, kernel, mode="same")
    # threshold relative to typical
    thr = max(4.0, float(np.percentile(diffs, 80)))
    for i in range(len(diffs) - 1, -1, -1):
        if diffs[i] > thr:
            y = start + i
            pct = y / h * 100.0
            # keep in a sane range
            return float(min(94.5, max(82.0, pct)))
    return 93.0


_LAYER_META = {}


def layer_meta(path: str):
    if path in _LAYER_META:
        return _LAYER_META[path]
    im = Image.open(ROOT / path).convert("RGBA")
    alpha = np.array(im)[:, :, 3]
    rows = (alpha > 10).any(axis=1)
    ys = np.where(rows)[0]
    if len(ys) == 0:
        meta = {"image_h": im.size[1], "image_w": im.size[0], "garment_h": im.size[1],
                "opaque_top": 0, "opaque_bot": im.size[1] - 1, "hook_px": 0}
        _LAYER_META[path] = meta
        return meta
    y0, y1 = int(ys[0]), int(ys[-1])
    widths = []
    for y in range(y0, y1 + 1):
        xs = np.where(alpha[y] > 10)[0]
        widths.append(int(xs[-1] - xs[0] + 1) if len(xs) else 0)
    maxw = max(widths) or 1
    garment_start = y0
    run = 0
    for i, w in enumerate(widths):
        if w >= 0.40 * maxw:
            run += 1
            if run >= 6:
                garment_start = y0 + i - run + 1
                break
        else:
            run = 0
    meta = {
        "image_h": im.size[1],
        "image_w": im.size[0],
        "opaque_top": y0,
        "opaque_bot": y1,
        "garment_h": max(1, y1 - garment_start + 1),
        "hook_px": garment_start - y0,
    }
    _LAYER_META[path] = meta
    return meta


def visual_frac(layer):
    """Fraction of the CSS box height that is the opaque garment (ex-hanger)."""
    meta = layer_meta(layer["image"])
    return meta["garment_h"] / meta["image_h"]


def css_h_px(layer):
    return layer["height"] / 100.0 * CANVAS_H


def visual_h_px(layer):
    return css_h_px(layer) * visual_frac(layer)


def assign_rod(hook_y_pct, rails, motion):
    if not rails:
        return 0, None
    if motion == "lift":
        # compare item bottom in px to shelf baselines (stored as hook_y)
        y = hook_y_pct / 100.0 * CANVAS_H
        dists = [abs(y - r["hook_y"]) for r in rails]
        idx = int(np.argmin(dists))
        return idx, rails[idx]
    y = hook_y_pct / 100.0 * CANVAS_H
    dists = [abs(y - r["hook_y"]) for r in rails]
    idx = int(np.argmin(dists))
    return idx, rails[idx]


def clearance_px(rod_idx, rails, floor_pct, motion):
    floor_y = floor_pct / 100.0 * CANVAS_H
    if motion == "lift":
        # rest items grow upward from the shelf; clearance is to the shelf/rail above or ceiling
        shelf = rails[rod_idx]["hook_y"] if rails else floor_y
        if rod_idx == 0:
            return max(20.0, shelf - 20.0)
        above = rails[rod_idx - 1]["hook_y"]
        return max(20.0, shelf - above - MARGIN_RAIL_PX)
    hook = rails[rod_idx]["hook_y"] if rails else 50.0
    if rod_idx + 1 < len(rails):
        limit = rails[rod_idx + 1]["wood_top"] - MARGIN_RAIL_PX
    else:
        limit = floor_y - MARGIN_FLOOR_PX
    return max(16.0, limit - hook)


def set_box_keep_anchor(layer, new_w, new_h, motion, hook_x, hook_y):
    """Resize CSS box, keeping hang hook or rest bottom in place. Aspect from new_w/new_h."""
    px = layer.get("pivotX", 50.0)
    py = layer.get("pivotY", 0.0)
    if motion == "lift":
        # keep the bottom-center on the shelf (more stable than the 50/50 pivot)
        bottom = hook_y  # for lift we pass the old bottom as hook_y
        cx = hook_x
        layer["width"] = new_w
        layer["height"] = new_h
        layer["left"] = cx - new_w / 2.0
        layer["top"] = bottom - new_h
    else:
        layer["width"] = new_w
        layer["height"] = new_h
        layer["left"] = hook_x - new_w * px / 100.0
        layer["top"] = hook_y - new_h * py / 100.0
    layer["rect"] = {
        "left": layer["left"],
        "top": layer["top"],
        "width": layer["width"],
        "height": layer["height"],
    }


def sync_hotspot(item):
    L = item["layer"]
    hs = item["hotspot"]
    hs["left"] = L["left"]
    hs["top"] = L["top"]
    hs["width"] = L["width"]
    hs["height"] = L["height"]
    # clip stays in box-relative percentages


def pack_rod(entries, x_min, x_max):
    """Place entries left-to-right in the given span, equal gaps, no overlaps."""
    if not entries:
        return
    widths = [e["item"]["layer"]["width"] for e in entries]
    total = sum(widths)
    span = x_max - x_min
    n = len(entries)
    if n == 1:
        gaps = [(span - total) / 2.0]
        # center the single item
        left = x_min + max(GAP_MIN_PCT, gaps[0])
        e = entries[0]
        L = e["item"]["layer"]
        shelf = is_shelf_rest(L)
        if shelf:
            hook_x = left + L["width"] / 2.0
            hook_y = e["hook_y"]
        else:
            hook_x = left + L["width"] * L.get("pivotX", 50.0) / 100.0
            hook_y = e["hook_y"]
        set_box_keep_anchor(L, L["width"], L["height"], "lift" if shelf else "hang", hook_x, hook_y)
        sync_hotspot(e["item"])
        return
    gap = (span - total) / (n + 1)
    if gap < GAP_MIN_PCT:
        # widen toward walls if possible
        extra = (GAP_MIN_PCT - gap) * (n + 1)
        x_min = max(WALL_MIN, x_min - extra / 2.0)
        x_max = min(WALL_MAX, x_max + extra / 2.0)
        span = x_max - x_min
        gap = (span - total) / (n + 1)
    if gap < GAP_MIN_PCT:
        # last resort: shrink all items on this rod uniformly so they fit
        # with minimum gaps, keeping hook_y and aspect.
        avail = span - GAP_MIN_PCT * (n + 1)
        if avail > 1 and total > 0:
            shrink = avail / total
            for e in entries:
                L = e["item"]["layer"]
                shelf = is_shelf_rest(L)
                if shelf:
                    hx, hy = L["left"] + L["width"] / 2.0, e["hook_y"]
                else:
                    hx, hy = hook_of(L)[0], e["hook_y"]
                set_box_keep_anchor(L, L["width"] * shrink, L["height"] * shrink, "lift" if shelf else "hang", hx, hy)
                e["shrunk"] = shrink
            widths = [e["item"]["layer"]["width"] for e in entries]
            total = sum(widths)
            gap = (span - total) / (n + 1)
        else:
            gap = GAP_MIN_PCT
    x = x_min + gap
    for e in entries:
        L = e["item"]["layer"]
        shelf = is_shelf_rest(L)
        left = x
        if shelf:
            hook_x = left + L["width"] / 2.0
        else:
            hook_x = left + L["width"] * L.get("pivotX", 50.0) / 100.0
        hook_y = e["hook_y"]
        set_box_keep_anchor(L, L["width"], L["height"], "lift" if shelf else "hang", hook_x, hook_y)
        sync_hotspot(e["item"])
        x += L["width"] + gap


def resolve_overlaps(records, by_zone):
    """Separate any leftover hotspot AABB overlaps without generating art.

    Same-rod pairs are re-packed. Cross-rod pairs: first try shifting the
    upper item horizontally into a gap; if that fails, shrink the upper
    item's CSS height (hook stays put) until the boxes no longer overlap.
    A long unscaled hanger that still cannot be separated is moved to the
    lower rod (necessary to keep hotspots from crossing).
    """
    def aabb(it):
        L = it["layer"]
        return L["left"], L["top"], L["left"] + L["width"], L["top"] + L["height"]

    def overlap(a, b):
        ax0, ay0, ax1, ay1 = aabb(a)
        bx0, by0, bx1, by1 = aabb(b)
        return min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)

    rec_of = {r["item"]["id"]: r for r in records}
    for zid, recs in by_zone.items():
        # several passes
        for _ in range(8):
            items = [r["item"] for r in recs]
            pairs = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    ox, oy = overlap(items[i], items[j])
                    if ox > 0.1 and oy > 0.1:
                        pairs.append((items[i], items[j], ox, oy))
            if not pairs:
                break
            a, b, ox, oy = max(pairs, key=lambda p: p[2] * p[3])
            ra, rb = rec_of[a["id"]], rec_of[b["id"]]
            # put the visually higher item first
            if a["layer"]["top"] > b["layer"]["top"]:
                a, b, ra, rb = b, a, rb, ra
            if ra["rod_idx"] == rb["rod_idx"]:
                # re-pack this rod with a slightly tighter wall if needed
                rod = [r for r in recs if r["rod_idx"] == ra["rod_idx"]]
                rod.sort(key=lambda r: -visual_h_px(r["item"]["layer"]))
                pack_rod(rod, WALL_MIN, WALL_MAX)
                continue
            # cross-rod: shrink the upper item just enough (keep hook)
            La = a["layer"]
            needed = oy + 0.25  # percent
            new_h = La["height"] - needed
            if new_h > 4.0:
                s = new_h / La["height"]
                shelf = is_shelf_rest(La)
                hx = (La["left"] + La["width"] / 2.0) if shelf else hook_of(La)[0]
                set_box_keep_anchor(La, La["width"] * s, new_h, "lift" if shelf else "hang", hx, ra["hook_y"])
                sync_hotspot(a)
                ra["scale_final"] = ra["scale_final"] * s
                continue
            # last resort: move the upper item down to the lower rod
            ra["rod_idx"] = rb["rod_idx"]
            ra["hook_y"] = rb["hook_y"]
            ra["moved_rod"] = True
            La = a["layer"]
            shelf = is_shelf_rest(La)
            hx = (La["left"] + La["width"] / 2.0) if shelf else hook_of(La)[0]
            set_box_keep_anchor(La, La["width"], La["height"], "lift" if shelf else "hang", hx, ra["hook_y"])
            sync_hotspot(a)
            rod = [r for r in recs if r["rod_idx"] == rb["rod_idx"]]
            rod.sort(key=lambda r: -visual_h_px(r["item"]["layer"]))
            pack_rod(rod, WALL_MIN, WALL_MAX)



def cluster_label(h, edges):
    """short / medium / long from tertiles of a room-rod or zone."""
    if h <= edges[0]:
        return "short"
    if h <= edges[1]:
        return "medium"
    return "long"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="*", help="Limit to these zone ids (prototype).")
    ap.add_argument("--write-catalog", action="store_true", help="Write data/catalog.json")
    ap.add_argument("--report-dir", default=str(REPORT))
    args = ap.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    items = {i["id"]: i for i in catalog["items"]}
    rails = load_rails(catalog)
    zone_floor = {}
    for z in catalog["zones"]:
        zone_floor[z["id"]] = detect_floor_pct(z)

    target_zones = args.zones or [z["id"] for z in catalog["zones"]]

    # Measure and book-keep every item in target zones.
    records = []  # dicts used throughout
    for z in catalog["zones"]:
        if z["id"] not in target_zones:
            continue
        zrails = rails.get(z["id"]) or []
        floor = zone_floor[z["id"]]
        for iid in z["items"]:
            it = items[iid]
            L = it["layer"]
            motion = L.get("motion") or "hang"
            shelf = is_shelf_rest(L)
            if shelf:
                hook_x = L["left"] + L["width"] / 2.0
                hook_y = L["top"] + L["height"]  # bottom sits on the shelf
            else:
                hook_x, hook_y, _, _ = hook_of(L)
            rod_idx, rail = assign_rod(hook_y, zrails, "lift" if shelf else "hang")
            if rail and not shelf:
                hook_y = rail["hook_y"] / CANVAS_H * 100.0
            meas = measure_item(it)
            old_css = css_h_px(L)
            old_vis = visual_h_px(L)
            rec = {
                "item": it,
                "zone": z["id"],
                "category": it.get("category"),
                "type": it.get("type"),
                "name": it.get("name") or it.get("display_name") or it["id"],
                "motion": motion,
                "rod_idx": rod_idx,
                "n_rods": len(zrails),
                "hook_x": hook_x,
                "hook_y": hook_y,
                "old_left": L["left"],
                "old_css_h_px": old_css,
                "old_vis_h_px": old_vis,
                "old_w": L["width"],
                "old_h": L["height"],
                "meas": meas,
                "scale_real": 1.0,
                "scale_final": 1.0,
                "px_per_cm": None,
                "moved_rod": False,
            }
            records.append(rec)

    # ---- 1. Choose a px-per-cm scale per zone and scale items ----
    by_zone = defaultdict(list)
    for r in records:
        by_zone[r["zone"]].append(r)

    zone_scales = {}
    for zid, recs in by_zone.items():
        zrails = rails.get(zid) or []
        floor = zone_floor[zid]
        # First, compute the max px/cm each scalable item can take on its rod.
        def max_ppc(rec, rod_idx):
            if not rec["meas"]:
                return None
            length_cm = rec["meas"][0]
            frac = visual_frac(rec["item"]["layer"])
            # Clearance is the max CSS box hang (hook to next rail / floor).
            # Visual garment height = CSS height * frac, so
            #   ppc <= clear_css * frac / length_cm
            # Using clearance/length (as if the CSS box were all garment)
            # lets the box overlap the row below.
            clear = clearance_px(rod_idx, zrails, floor, "lift" if is_shelf_rest(rec["item"]["layer"]) else "hang")
            return (clear * frac) / length_cm

        scalable = [r for r in recs if r["meas"]]
        # Room scale = min of max_ppc, after moving items that would collapse the scale.
        def room_min(recs_local):
            vals = [max_ppc(r, r["rod_idx"]) for r in recs_local if r["meas"]]
            vals = [v for v in vals if v and v > 0]
            return min(vals) if vals else None

        def ppc_of(subset):
            vals = [max_ppc(r, r["rod_idx"]) for r in subset if r["meas"]]
            vals = [v for v in vals if v and v > 0]
            return min(vals) if vals else None

        ppc = ppc_of(recs)
        # Move an item off its rod only when it is a true bottleneck: without
        # it the room scale would be substantially higher, and a lower rod can
        # fit it at that higher scale. Same-length items on a tighter top rod
        # are not bottlenecks and stay put.
        if scalable and ppc:
            for r in list(scalable):
                others = [o for o in scalable if o is not r]
                ppc_rest = ppc_of(others)
                cur = max_ppc(r, r["rod_idx"])
                if not ppc_rest or not cur:
                    continue
                if cur > MOVE_SCALE_RATIO * ppc_rest:
                    continue
                # try a lower (more clearance) rod first, then any rod that fits ppc_rest
                best_i = None
                for i in range(r["n_rods"]):
                    v = max_ppc(r, i)
                    if v and v >= ppc_rest * 0.98:
                        # prefer lower rods (higher index) when several fit
                        if best_i is None or i > best_i:
                            best_i = i
                if best_i is not None and best_i != r["rod_idx"]:
                    r["rod_idx"] = best_i
                    r["moved_rod"] = True
                    if zrails:
                        r["hook_y"] = zrails[best_i]["hook_y"] / CANVAS_H * 100.0
            ppc = ppc_of(recs)

        # Don't blow short rooms up past a comfortable fill: cap so the longest
        # scalable item uses at most 92% of its clearance at this scale.
        zone_scales[zid] = ppc
        for r in recs:
            if not r["meas"] or not ppc:
                r["scale_real"] = 1.0
                r["scale_final"] = 1.0
                r["px_per_cm"] = None
                continue
            length_cm, width_cm, url, note = r["meas"]
            r["px_per_cm"] = ppc
            target_vis = length_cm * ppc
            old_vis = r["old_vis_h_px"]
            if old_vis <= 0:
                r["scale_real"] = 1.0
                r["scale_final"] = 1.0
                continue
            scale = target_vis / old_vis
            r["scale_real"] = scale
            r["scale_final"] = scale

        # Apply real-size scales now (before bump and cluster).
        for r in recs:
            L = r["item"]["layer"]
            anchor = "lift" if is_shelf_rest(L) else "hang"
            if not r["meas"] or abs(r["scale_real"] - 1.0) < 1e-6:
                set_box_keep_anchor(L, L["width"], L["height"], anchor, r["hook_x"], r["hook_y"])
                sync_hotspot(r["item"])
                continue
            s = r["scale_real"]
            set_box_keep_anchor(L, r["old_w"] * s, r["old_h"] * s, anchor, r["hook_x"], r["hook_y"])
            sync_hotspot(r["item"])

    # ---- 2. Visual bump on middle rods ----
    bumps = []
    for zid, recs in by_zone.items():
        n_rods = max((r["n_rods"] for r in recs), default=0)
        if n_rods < 3:
            continue
        # middle rod indices: all except first and last
        middle = set(range(1, n_rods - 1))
        for rod in sorted(middle):
            rod_recs = [r for r in recs if r["rod_idx"] == rod]
            if not rod_recs:
                continue
            heights = [visual_h_px(r["item"]["layer"]) for r in rod_recs]
            others_by_id = {}
            for r in rod_recs:
                others = [visual_h_px(o["item"]["layer"]) for o in rod_recs if o is not r]
                if len(others) >= 2:
                    others_by_id[r["item"]["id"]] = float(np.median(others))
                else:
                    adj = [visual_h_px(o["item"]["layer"]) for o in recs if o["rod_idx"] in (rod - 1, rod + 1)]
                    others_by_id[r["item"]["id"]] = float(np.median(adj)) if adj else heights[0]
            for r in rod_recs:
                if not r["meas"]:
                    continue  # only bump items we already scaled to a real size
                vis = visual_h_px(r["item"]["layer"])
                median_nb = others_by_id[r["item"]["id"]]
                if vis >= BUMP_THRESHOLD * median_nb:
                    continue
                # bump just enough to reach 85% of the median, capped at +15%
                needed = (BUMP_THRESHOLD * median_nb) / vis if vis > 0 else 1.0
                cap = 1.0 + BUMP_CAP
                factor = min(needed, cap)
                if factor <= 1.001:
                    continue
                new_final = r["scale_real"] * factor
                L = r["item"]["layer"]
                s_rel = factor  # additional over current (which is scale_real)
                shelf = is_shelf_rest(L)
                set_box_keep_anchor(
                    L, L["width"] * s_rel, L["height"] * s_rel,
                    "lift" if shelf else "hang",
                    (L["left"] + L["width"] / 2.0) if shelf else hook_of(L)[0],
                    r["hook_y"],
                )
                sync_hotspot(r["item"])
                r["scale_final"] = new_final
                bump_pct = (factor - 1.0) * 100.0
                reason = (
                    f"Middle rod {rod}: displayed height {vis:.1f}px was "
                    f"{vis / median_nb * 100:.1f}% of neighbor median {median_nb:.1f}px "
                    f"(threshold {BUMP_THRESHOLD * 100:.0f}%). "
                    f"Bumped by {bump_pct:.1f}% (cap {BUMP_CAP * 100:.0f}%)."
                )
                bumps.append({
                    "item_id": r["item"]["id"],
                    "room": zid,
                    "real_size_scale": r["scale_real"],
                    "final_scale": new_final,
                    "pct_bump": bump_pct,
                    "reason": reason,
                })

    # ---- 3. Cluster each rod by displayed size (long → short, left to right) ----
    layouts = {}  # zid -> list of rod descriptions
    for zid, recs in by_zone.items():
        zrails = rails.get(zid) or []
        # horizontal span: current (post-scale) union, clamped to walls, per zone
        lefts, rights = [], []
        for r in recs:
            L = r["item"]["layer"]
            lefts.append(L["left"])
            rights.append(L["left"] + L["width"])
        zone_xmin = max(WALL_MIN, min(lefts) if lefts else WALL_MIN)
        zone_xmax = min(WALL_MAX, max(rights) if rights else WALL_MAX)
        n_rods = max((r["n_rods"] for r in recs), default=1)
        rod_layout = []
        for rod in range(n_rods):
            rod_recs = [r for r in recs if r["rod_idx"] == rod]
            if not rod_recs:
                continue
            # sort by displayed opaque height descending so long sit together
            rod_recs.sort(key=lambda r: -visual_h_px(r["item"]["layer"]))
            heights = [visual_h_px(r["item"]["layer"]) for r in rod_recs]
            # Tertile edges on unique heights so equal sizes share a label.
            uniq = sorted({round(h, 1) for h in heights})
            if len(uniq) <= 1:
                edges = (1e12, 1e12)  # all "short" is wrong; cluster_label: treat as one bucket
                # override: mark all as the same band using a sentinel handled below
                same_band = True
            elif len(uniq) == 2:
                edges = (uniq[0] + 0.05, 1e12)  # lower=short, higher=medium (cluster_label uses two cuts)
                same_band = False
            else:
                e1 = uniq[max(0, len(uniq)//3 - 1)]
                e2 = uniq[max(0, (2*len(uniq)//3) - 1)]
                edges = (e1, e2)
                same_band = False
            pack_rod(rod_recs, zone_xmin, zone_xmax)
            entries = []
            for r in rod_recs:
                vis = visual_h_px(r["item"]["layer"])
                label = "same-size" if same_band else cluster_label(vis, edges)
                # 2 unique heights: shorter=short, taller=long
                if (not same_band) and len({round(visual_h_px(x["item"]["layer"]), 1) for x in rod_recs}) == 2:
                    label = "short" if vis <= edges[0] else "long"
                entries.append({
                    "id": r["item"]["id"],
                    "name": r["name"],
                    "type": r["type"],
                    "cluster": label,
                    "display_h_px": round(vis, 2),
                    "scale_final": round(r["scale_final"], 4),
                })
            rod_layout.append({
                "rod": rod,
                "hook_y_pct": round(rod_recs[0]["hook_y"], 2),
                "order_left_to_right": entries,
            })
            # z-index left to right
            for i, r in enumerate(rod_recs, start=1):
                r["item"]["layer"]["zIndex"] = rod * 20 + i
        layouts[zid] = rod_layout

    resolve_overlaps(records, by_zone)

    # ---- 4. Write CSVs and optional catalog ----
    sizes_path = report_dir / "sizes.csv"
    with sizes_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "item_id", "room", "type", "real_length_cm", "real_width_cm",
            "source_url", "old_px_height", "new_px_height", "scale_ratio",
            "note",
        ])
        w.writeheader()
        for r in records:
            L = r["item"]["layer"]
            new_vis = visual_h_px(L)
            if r["meas"]:
                length, width, url, note = r["meas"]
            else:
                length, width, url, note = "", "", "", "not rescaled (no reliable published size for the displayed dimension)"
            w.writerow({
                "item_id": r["item"]["id"],
                "room": r["zone"],
                "type": r["type"],
                "real_length_cm": length,
                "real_width_cm": width if width is not None else "",
                "source_url": url,
                "old_px_height": round(r["old_vis_h_px"], 2),
                "new_px_height": round(new_vis, 2),
                "scale_ratio": round(new_vis / r["old_vis_h_px"], 4) if r["old_vis_h_px"] else "",
                "note": note + (f"; moved to rod {r['rod_idx']}" if r["moved_rod"] else ""),
            })

    bumps_path = report_dir / "visual-bumps.csv"
    with bumps_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "item_id", "room", "real_size_scale", "final_scale", "pct_bump", "reason",
        ])
        w.writeheader()
        for b in bumps:
            w.writerow(b)

    meta = {
        "zone_scales_px_per_cm": {k: (round(v, 4) if v else None) for k, v in zone_scales.items()},
        "zone_floor_pct": zone_floor,
        "bump_threshold": BUMP_THRESHOLD,
        "bump_cap": BUMP_CAP,
        "layouts": layouts,
        "bumps": bumps,
        "n_records": len(records),
        "n_scaled": sum(1 for r in records if r["meas"]),
        "n_unscaled": sum(1 for r in records if not r["meas"]),
        "n_moved_rods": sum(1 for r in records if r["moved_rod"]),
    }
    (report_dir / "layout-meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if args.write_catalog:
        (ROOT / "data/catalog.json").write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
        )
        print("wrote data/catalog.json")

    print(f"zones {len(by_zone)} items {len(records)} scaled {meta['n_scaled']} "
          f"unscaled {meta['n_unscaled']} bumps {len(bumps)} moved {meta['n_moved_rods']}")
    for zid, ppc in zone_scales.items():
        print(f"  {zid}: px/cm={ppc:.3f}" if ppc else f"  {zid}: px/cm=None (nothing scaled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
