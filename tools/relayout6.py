#!/usr/bin/env python3
"""Plan and render the seven hanging categories from the 48f8268 baseline."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import string
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from rebuild_covers import composite as composite_catalog_room
from repack_three_rails import (
    detect_hook_xs, detect_obstacle_y, detect_rail_bands, erase_rail_band,
    extract_rail_sprite, heal_under_rail, save_webp, surgical_remove_hooks,
    variants_from, wood_mask,
)
from sizegroup_layout import (
    DRESS_LENGTHS, ROOT, SKIRT_LENGTHS, SOURCES, TYPE_DEFAULTS,
    hook_of, layer_meta, measure_item, visual_h_px,
)

BASELINE = "48f8268"
CATALOG = ROOT / "data/catalog.json"
REPORT = Path("/workspace/collection-room-impl-report/relayout-6")
W, H = 1536, 1024
SHORT_TOPS = (48, 314, 580)
SHORT_GAP, LONG_GAP, FLOOR_GAP = 8, 14, 10
ITEM_GAP, END_MARGIN = 28, 40
MAX_SHORT_ROW, MAX_LONG_ROW = 7, 6
MIN_ROOM_ITEMS = 8
LONG_LENGTH, MIN_SHORT_RATIO = 85, .92
EPS = 1e-6

# Each category has one sprite, including rooms with its other backgrounds.
# The jacket sprite is the calibration rod; the others represent the
# predominant shape in their existing plates.
RAIL_STYLE = {
    "jackets": ("jackets-b", "round pipe with circular ends"),
    "shirts": ("shirts-c", "flat wood rail"),
    "tshirts": ("tshirts-a", "flat wood rail"),
    "bottoms": ("bottoms-a", "flat wood rail"),
    "dresses": ("dresses-c", "flat wood rail"),
    "sweaters": ("sweaters-a", "round wood rail with mounts"),
    "hoodies": ("hoodies-b", "round pipe with slim end caps"),
}

# Families appear in reading order. A tuple contains the catalog types in a family.
FAMILIES = {
    "jackets": {
        "blazers": ("blazer",),
        "jackets": ("jacket", "windbreaker", "bomber", "varsity", "puffer", "hoodie"),
        "vests": ("vest",),
        "coats": ("coat",),
    },
    "shirts": {
        "shirts": ("shirt",),
        "blouses": ("blouse",),
        "polos and tops": ("polo", "top", "tank"),
    },
    "tshirts": {
        "tees and jerseys": ("tee", "jersey", "ringer-tee"),
        "long sleeves": ("long-sleeve-tee", "waffle-long-sleeve"),
        "tops and tanks": ("top", "tank", "sleeveless-top", "cropped-tee", "crop-top"),
    },
    "bottoms": {
        "shorts": ("shorts",),
        "skirts": ("skirt",),
        "cropped trousers": ("cropped-pants",),
        "full-length bottoms": ("pants", "jeans", "cargo-pants", "overalls"),
    },
    "dresses": {
        "skirts": ("skirt",),
        "short dresses": (),
        "midi and long dresses": ("dress",),
        "jumpsuits": ("jumpsuit",),
    },
    "sweaters": {
        "sweaters": ("sweater", "turtleneck", "layered-knit", "half-zip-sweater", "cropped-sweater"),
        "cardigans": ("cardigan",),
        "vests and knit tops": (
            "knit-vest", "sweater-vest", "knit-polo", "knit-tank",
            "draped-knit-top", "short-sleeve-knit", "knit-crop-top",
        ),
    },
    "hoodies": {
        "hoodies": ("hoodie", "zip-hoodie", "technical-hoodie", "short-sleeve-hoodie"),
        "sweatshirts": ("sweatshirt", "pullover", "quarter-zip", "fleece-pullover", "polo-sweatshirt"),
        "cropped": ("cropped-sweatshirt", "short-sleeve-cropped-hoodie"),
    },
}
NEAREST_TYPE = {
    "cropped-tee": "tee",
    "crop-top": "top",
    "knit-crop-top": "knit-tank",
    "cropped-sweater": "sweater",
    "cropped-sweatshirt": "sweatshirt",
    "short-sleeve-cropped-hoodie": "short-sleeve-hoodie",
    "overalls": "pants",
    "jumpsuit": "dress",
}
# Leading edge of the furniture where the broad-band detector sees the face,
# returns a rail, or misses a piece at the edge. Values are in normalized coords.
FURNITURE = {
    "jackets-b": ((0, W, 804),),
    "jackets-c": ((1174, 1490, 782),),
    "jackets-e": ((0, W, 771),),
    "shirts-a": ((0, W, 814),),
    "shirts-b": ((346, 1196, 830),),
    "shirts-d": ((0, W, 775),),
    "tshirts-b": ((0, W, 804),),
    "tshirts-c": ((1174, 1490, 782),),
    "dresses-a": ((107, 259, 858), (1380, W, 858)),
    "dresses-b": ((98, 273, 804), (1310, 1490, 804)),
    "dresses-c": ((107, 259, 858), (1380, W, 858)),
    "sweaters-a": ((193, 1461, 769),),
    "sweaters-b": ((0, W, 771),),
    "hoodies-a": ((0, W, 804),),
    "hoodies-b": ((0, W, 814),),
}
# These plates have light floors, so the detector returns its generic fallback
# before reaching the baseboard. Side furniture still has its own x/y limit.
WALL_LIMITS = {"dresses-a": 916, "dresses-b": 901, "dresses-c": 916}


@dataclass(frozen=True)
class Photo:
    width: int
    height: int
    opaque: tuple
    garment_height: int
    aspect: float
    lightness: float


@dataclass
class Garment:
    item: dict
    photo: Photo
    family: str
    length: float
    source: str
    note: str
    scale: float
    long_hang: bool

    @property
    def anchor_y(self):
        return self.photo.height * self.item["layer"]["pivotY"] / 100

    @property
    def hem(self):
        return self.photo.opaque[3] - self.anchor_y


@dataclass
class Background:
    id: str
    rgb: np.ndarray
    bands: tuple
    span: tuple
    wood_height: int
    obstacle: int
    furniture: tuple
    detected_obstacle: int

    def limit(self, left, right):
        furniture_y = min((y for x0, x1, y in self.furniture if left < x1 and right > x0),
                          default=H)
        return min(self.obstacle, furniture_y)


@dataclass(frozen=True)
class Placement:
    garment: Garment
    opaque_left: float
    scale: float

    @property
    def opaque_right(self):
        p = self.garment.photo
        return self.opaque_left + (p.opaque[2] - p.opaque[0]) * self.scale

    @property
    def box_left(self):
        return self.opaque_left - self.garment.photo.opaque[0] * self.scale


@dataclass(frozen=True)
class Row:
    items: tuple
    gap: float | None

    @property
    def hem(self):
        return max((p.garment.hem * p.scale for p in self.items), default=0)

    @property
    def box_bottom(self):
        return max(((p.garment.photo.height - p.garment.anchor_y) * p.scale for p in self.items), default=0)

    @property
    def box_top(self):
        return max((p.garment.anchor_y * p.scale for p in self.items), default=0)

    def clearance(self, bg):
        return min((bg.limit(p.opaque_left, p.opaque_right) - p.garment.hem * p.scale
                    for p in self.items), default=bg.obstacle)


@dataclass(frozen=True)
class Room:
    id: str
    background: Background
    wood_tops: tuple
    rows: tuple


def short_hem_limit(hook):
    # The hook offset already measures from the wood top. Clearance is from
    # the hem to the next wood top, so the rest of the rail depth is outside
    # this measurement.
    return SHORT_TOPS[1] - SHORT_TOPS[0] - hook - SHORT_GAP


@cache
def photo_metrics(path):
    with Image.open(ROOT / path) as image:
        rgba = np.array(image.convert("RGBA"))
    meta = layer_meta(path)
    alpha = rgba[:, :, 3]
    mask = alpha > 10
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError(f"Empty item layer {path}")
    start = meta["opaque_top"] + meta["hook_px"]
    height = meta["garment_h"]
    # Just below the neckline, before sleeves flare. For bottoms this is the waist band.
    band = mask[start + round(.06 * height):start + max(2, round(.16 * height))]
    widths = [x[-1] - x[0] + 1 for row in band if len(x := np.flatnonzero(row))]
    shoulder = median(widths) if widths else xs.max() - xs.min() + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    body = rgba[start + round(.16 * height):start + round(.65 * height),
                x0 + round(.30 * (x1 - x0)):x0 + round(.70 * (x1 - x0))]
    pixels = body[body[:, :, 3] > 200, :3]
    if not len(pixels):
        pixels = rgba[start:, :, :3][alpha[start:] > 80]
    return Photo(rgba.shape[1], rgba.shape[0], (x0, y0, x1, y1), height,
                 height / shoulder, float(np.median(pixels @ np.array([.2126, .7152, .0722]))))


def baseline_plate(zone):
    path = max(zone["images"], key=lambda image: image["width"])["src"]
    data = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
    with Image.open(BytesIO(data)) as image:
        return np.array(image.convert("RGB").resize((W, H), Image.Resampling.LANCZOS))


def background(zone):
    zid = zone["id"]
    rgb = baseline_plate(zone)
    bands = detect_rail_bands(rgb)
    wood = wood_mask(rgb)
    if len(bands) < 3:
        # The detector drops thick rails below 62% of the image. Recover the
        # lower clone by its width and thickness, before the floor begins.
        candidates = np.flatnonzero(wood[:int(.75 * H), W // 8:7 * W // 8].mean(1) > .45)
        for run in np.split(candidates, np.flatnonzero(np.diff(candidates) > 3) + 1):
            if not 8 <= len(run) <= 48 or any(lo - 4 <= run[0] <= hi + 4 for lo, hi, _, _ in bands):
                continue
            x = np.flatnonzero(wood[run].mean(0) > .2)
            if len(x) and x[-1] - x[0] > W / 2:
                bands.append((int(run[0]), int(run[-1]), int(x[0]), int(x[-1])))
    bands.sort()
    if len(bands) != 3:
        raise ValueError(f"{zid}: expected three source rail bands, found {bands}")
    y0, y1, x0, x1 = bands[0]
    detected_obstacle = detect_obstacle_y(rgb)
    obstacle = WALL_LIMITS.get(zid, detected_obstacle)
    if any(lo - 3 <= detected_obstacle <= hi + 3 for lo, hi, _, _ in bands):
        # shirts-b's wooden bench and third rail trigger the same detector branch.
        wall = rgb.copy()
        for lo, hi, left, right in bands:
            wall = erase_rail_band(wall, lo, hi, left - 24, right + 24)
        obstacle = detect_obstacle_y(wall)
    furniture = FURNITURE.get(zid, ())
    # A narrow bench is allowed below an otherwise empty portion of the row.
    if zid == "shirts-b":
        obstacle = 933
    return Background(zid, rgb, tuple(bands), (x0, x1 + 1), y1 - y0 + 1, obstacle, furniture,
                      detected_obstacle)


def calibration(items, ref):
    wood_tops = [b[0] for b in ref.bands[:2]]
    ratios, offsets = [], []
    for item in items:
        if item["zone"] != ref.id:
            continue
        hook = hook_of(item["layer"])[1] * H / 100
        for top in wood_tops:
            if abs(hook - top) < 24:
                offsets.append(hook - top)
                # The vest from the follow-up was set to 65 cm. Its catalog
                # type default is 66.5, so use the regular jacket witnesses.
                if item["type"] in {"jacket", "windbreaker", "bomber", "varsity", "puffer"}:
                    ratios.append(visual_h_px(item["layer"]) / measure_item(item)[0])
    if not ratios or not offsets:
        raise ValueError("No jackets-b calibration items")
    return median(ratios), round(median(offsets), 6)


def estimate_length(item, photo, aspects):
    typ = item["type"]
    nearest = NEAREST_TYPE.get(typ, typ)
    default, _, key, _ = TYPE_DEFAULTS[nearest]
    if item["id"] == "batch2-151":
        # Narrow tank straps distort the comparison with sleeved jerseys.
        return default, SOURCES[key], "Basketball tank jersey at the 70 cm jersey type default"
    name = item["name"].lower()
    words = set(re.findall(r"\b[a-z]+\b", name))
    if typ in {"dress", "skirt"}:
        # Sleeve length is independent of the garment's hem length.
        hem_name = re.sub(r"\b(?:long|short)[ -]+sleeve\w*\b", "", name)
        hem_words = set(re.findall(r"\b[a-z]+\b", hem_name))
        variant = next((v for v in ("maxi", "mini", "midi", "calf", "knee") if v in hem_words), None)
        if "floor-length" in hem_name or "floor length" in hem_name:
            variant = "maxi"
        if not variant and "ankle" in hem_words:
            variant = "calf"
        if not variant and "long" in hem_words:
            variant = "maxi" if typ == "dress" else "calf"
        if not variant and f"short {typ}" in hem_name:
            variant = "mini"
        table = DRESS_LENGTHS if typ == "dress" else SKIRT_LENGTHS
        if variant:
            return table[variant][0], SOURCES[key], f"Published {variant} {typ} length"
    if typ in FAMILIES["jackets"]["jackets"] + ("coat", "blazer") and (
        words & {"duster", "trench", "overcoat", "maxi"} or re.search(r"\blong[- ]+(?:coat|jacket|blazer)\b", name)
    ):
        return 91.5, SOURCES["doppiaa_jacket"], "Published Doppiaa long jacket/coat length"
    if typ in {"pants", "jeans", "cargo-pants"} and words & {"cropped", "crop"}:
        return TYPE_DEFAULTS["cropped-pants"][0], SOURCES["uniqlo_cropped"], "Published cropped trouser length"

    # Unsourced cropped types use the regular type's median so that the
    # naturally shorter crop remains reflected in its estimate.
    median_type = nearest if nearest in aspects else typ
    aspect_ratio = photo.aspect / aspects[median_type]
    multiplier = min(1.45, max(.8, aspect_ratio))
    note = f"{nearest} default {default:g} cm × photo aspect factor {multiplier:.4f}"
    if multiplier != aspect_ratio:
        note += f" (raw {aspect_ratio:.4f}, clamped)"
    return default * multiplier, SOURCES[key], note


def garments(catalog, categories, px_per_cm, hook_offset):
    selected = [item for item in catalog["items"] if item["category"] in categories]
    median_types = {NEAREST_TYPE.get(item["type"], item["type"]) for item in selected}
    samples = defaultdict(list)
    for item in catalog["items"]:
        if item["type"] in median_types or item["category"] in categories:
            samples[item["type"]].append(photo_metrics(item["layer"]["image"]).aspect)
    aspects = {typ: median(values) for typ, values in samples.items()}
    out = defaultdict(list)
    for item in selected:
        category, typ = item["category"], item["type"]
        families = FAMILIES[category]
        photo = photo_metrics(item["layer"]["image"])
        length, source, note = estimate_length(item, photo, aspects)
        name = item["name"].lower()
        family = next(f for f, types in families.items() if typ in types)
        if typ == "dress" and length <= DRESS_LENGTHS["mini"][0]:
            family = "short dresses"
        if category == "jackets" and family in {"blazers", "jackets", "coats"} and (
            re.search(r"\b(?:trench|overcoat|duster|long[- ]coat)\b", name)
        ):
            family = "coats"
        if category == "bottoms" and typ in {"pants", "jeans", "cargo-pants"} and re.search(r"\bcrop(?:ped)?\b", name):
            family = "cropped trousers"
        if category == "hoodies" and re.search(r"\bcrop(?:ped)?\b", name):
            family = "cropped"
        anchor = photo.height * item["layer"]["pivotY"] / 100
        scale = px_per_cm * length / photo.garment_height
        if category == "jackets":
            long_hang = family == "coats" or length >= LONG_LENGTH
        else:
            hem = (photo.opaque[3] - anchor) * scale
            named_midi = typ == "dress" and bool(re.search(r"\b(?:midi|maxi)\b", name))
            long_hang = (length >= LONG_LENGTH or family in {"full-length bottoms", "jumpsuits"}
                         or named_midi or hem * MIN_SHORT_RATIO > short_hem_limit(hook_offset) + EPS)
        out[category].append(Garment(item, photo, family, length, source, note, scale, long_hang))
    for category, records in out.items():
        families = list(FAMILIES[category])
        records.sort(key=lambda g: (families.index(g.family), int(g.length // 5),
                                    -g.photo.lightness, g.item["id"]))
    return out


def pack_row(garments, span, max_hem=None):
    if not garments:
        return Row((), None)
    scales = [min(g.scale, max_hem / g.hem) if max_hem is not None else g.scale for g in garments]
    widths = [(g.photo.opaque[2] - g.photo.opaque[0]) * s for g, s in zip(garments, scales)]
    remaining = span[1] - span[0] - sum(widths)
    n = len(widths)
    if n == 1:
        gap, margin = None, remaining / 2
    else:
        gap = min(remaining / (n + 1), (remaining - 2 * END_MARGIN) / (n - 1))
        padding = max((a.photo.width - a.photo.opaque[2]) * s + b.photo.opaque[0] * t
                      for a, s, b, t in zip(garments, scales, garments[1:], scales[1:]))
        if gap + EPS < max(ITEM_GAP, padding):
            return None
        margin = (remaining - gap * (n - 1)) / 2
    if margin + EPS < END_MARGIN:
        return None
    x = span[0] + margin
    placements = []
    for garment, scale, width in zip(garments, scales, widths):
        placements.append(Placement(garment, x, scale))
        x += width + (gap or 0)
    return Row(tuple(placements), gap)


def long_tops(rows, bg, hook):
    first, second = rows
    lower = max(
        math.ceil(SHORT_TOPS[0] + hook + first.hem + LONG_GAP),
        math.ceil(SHORT_TOPS[0] + first.box_bottom + second.box_top),
    )
    upper = math.floor(second.clearance(bg) - hook - FLOOR_GAP)
    if lower > upper:
        return None
    # Equalize the physical air below each hem, within the clearance bounds.
    midpoint = (SHORT_TOPS[0] + hook + first.hem + second.clearance(bg) - hook) / 2
    return (SHORT_TOPS[0], min(upper, max(lower, round(midpoint))))


@cache
def row_splits(count, rods, max_row):
    if rods == 1:
        return ((count,),) if 1 <= count <= max_row else ()
    return tuple((first, *rest) for first in range(1, min(max_row, count - rods + 1) + 1)
                 for rest in row_splits(count - first, rods - 1, max_row))


def short_tops(rows, bg, hook):
    for top, row in zip(SHORT_TOPS, rows):
        if top != SHORT_TOPS[-1] and row.hem > short_hem_limit(hook) + EPS:
            return None
        if top + hook + FLOOR_GAP > row.clearance(bg) + EPS:
            return None
    for upper, lower in zip(rows, rows[1:]):
        if upper.box_bottom + lower.box_top > SHORT_TOPS[1] - SHORT_TOPS[0] + EPS:
            return None
    return SHORT_TOPS


def plan_group(records, own, siblings, category, offset, rods, hook):
    """Minimum-cost partition of an ordered short- or long-hang run."""
    if not records:
        return []
    count = len(records)
    max_row = MAX_SHORT_ROW if rods == 3 else MAX_LONG_ROW
    max_room = 9 if category == "jackets" and rods == 2 else min(rods * max_row, 17)
    target = {2: 9, 3: 13.4}[rods] if category == "jackets" else {2: 10, 3: 14}[rods]
    balance_weight = 1.4 if category == "jackets" else .22
    if category == "tshirts" and rods == 3:
        max_room = target = 13

    @cache
    def packed(start, stop, span, cap=None):
        return pack_row(records[start:stop], span, cap)

    @cache
    def fits(boundaries, source_id, allow_cap):
        bg = backgrounds[source_id]
        indices = tuple(zip(boundaries, boundaries[1:]))

        def pack(limits):
            rows = tuple(packed(a, b, bg.span, limit) for (a, b), limit in zip(indices, limits))
            if any(row is None for row in rows):
                return None
            tops = short_tops(rows, bg, hook) if rods == 3 else long_tops(rows, bg, hook)
            return (rows, tops) if tops else None

        if rods == 3:
            limit = short_hem_limit(hook)
            result = pack((limit, limit, None))
        else:
            result = pack((None, None))
            if result or not allow_cap or any(packed(a, b, bg.span) is None for a, b in indices):
                return result, 0
            hi = max(g.hem * g.scale for g in records[boundaries[0]:boundaries[-1]])
            lo = hi * .5
            if pack((lo, lo)) is None:
                return None, 0
            for _ in range(15):
                mid = (lo + hi) / 2
                if pack((mid, mid)):
                    lo = mid
                else:
                    hi = mid
            result = pack((lo, lo))
        if not result:
            return None, 0
        ratios = [p.scale / p.garment.scale for row in result[0] for p in row.items]
        cap_loss = sum(1 - ratio + 8 * max(0, MIN_SHORT_RATIO - ratio) for ratio in ratios)
        return result, cap_loss

    def choose(allow_cap):
        @cache
        def solve(start, index):
            if start == count:
                return 0, ()
            if index >= len(string.ascii_lowercase):
                return math.inf, ()
            zid = f"{category}-{string.ascii_lowercase[index]}"
            best = math.inf, ()
            for size in range(MIN_ROOM_ITEMS, min(max_room, count - start) + 1):
                end = start + size
                if 0 < count - end < MIN_ROOM_ITEMS:
                    continue
                rest_score, rest = solve(end, index + 1)
                size_score = 8 + (size - target) ** 2 * balance_weight + max(0, size - 16) * 3
                if rest_score + size_score >= best[0]:
                    continue
                candidates = []
                for splits in row_splits(size, rods, max_row):
                    boundaries = [start]
                    for n in splits:
                        boundaries.append(boundaries[-1] + n)
                    boundaries = tuple(boundaries)
                    row_score = sum(.80 * max(0, 4 - n)**2 + .40 * (n - size / rods)**2
                                    for n in splits)
                    row_score += 2 * sum(
                        records[k - 1].family != records[k].family
                        for a, b in zip(boundaries, boundaries[1:]) for k in range(a + 1, b)
                    )
                    candidates.append((row_score, boundaries))
                # Same-category plates win whenever an uncapped placement fits.
                for tier, sources in enumerate((own, siblings)):
                    any_fit = False
                    for row_score, boundaries in sorted(candidates):
                        score = rest_score + size_score + row_score + tier * 6
                        if score >= best[0]:
                            break
                        for bg in sources:
                            result, cap_loss = fits(boundaries, bg.id, False)
                            if result:
                                any_fit = True
                                plate_score = background_cost(zid, bg, rods) + cap_loss * 42
                                if score + plate_score < best[0]:
                                    rows, tops = result
                                    best = score + plate_score, (Room(zid, bg, tops, rows), *rest)
                    if any_fit:
                        break
                # If needed, compare capped options across all usable plates.
                if allow_cap and best[0] > rest_score + size_score + 80:
                    for row_score, boundaries in sorted(candidates):
                        for bg in (*own, *siblings):
                            result, cap_loss = fits(boundaries, bg.id, True)
                            if not result or not cap_loss:
                                continue
                            score = (rest_score + size_score + row_score + background_cost(zid, bg, rods)
                                     + 80 + cap_loss * 100 + (6 if bg in siblings else 0))
                            if score < best[0]:
                                rows, tops = result
                                best = score, (Room(zid, bg, tops, rows), *rest)
            return best

        return solve(0, offset)[1]

    backgrounds = {bg.id: bg for bg in (*own, *siblings)}
    result = choose(False)
    if not result and rods == 2:
        result = choose(True)
    if not result:
        raise ValueError(f"Could not fit {count} {category} items in {rods}-rod rooms at the required scale")
    return list(result)


def background_cost(zid, bg, rods):
    if zid == bg.id:
        return 0
    # If b's original cabinet forces a replacement, favor the neighboring
    # jacket plate with the same window-lit wall and no full-width furniture.
    if zid == "jackets-b" and rods == 3 and bg.id == "jackets-a":
        return .1
    # Two sibling plates have the same split-bench style as the hoodie room
    # and leave more wall below the last rail when its own benches are too high.
    if zid.startswith("hoodies-") and bg.id in {"bottoms-b", "shirts-c"}:
        return .5
    return 1 + abs(ord(zid[-1]) - ord(bg.id[-1])) * .03


def plan_category(records, own, siblings, category, hook):
    short = [g for g in records if not g.long_hang]
    long = [g for g in records if g.long_hang]
    if category == "dresses":
        # Keep the jumpsuits together before the dress run so the deepest
        # dress hems occupy the bottom rod of the final room.
        long.sort(key=lambda g: g.family != "jumpsuits")
    rooms = plan_group(short, own, siblings, category, 0, 3, hook)
    rooms.extend(plan_group(long, own, siblings, category, len(rooms), 2, hook))
    return rooms


def clean_wall(bg):
    """Replace old rails with wall donors and feather the edges of each repair."""
    wall = bg.rgb.copy()
    for index, (y0, y1, x0, x1) in enumerate(bg.bands):
        # The second dresses-b plate rail has a light paint patch above its
        # right bracket that extends beyond its regular contact shadow.
        above = 34 if bg.id == "dresses-b" and index == 1 else 20
        top = max(0, y0 - above)
        bottom = min(H - 1, y1 + 43)
        left, right = max(0, x0 - 29), min(W, x1 + 30)
        fill = erase_rail_band(wall, top, bottom, left, right - 1, donor_above=0, donor_below=0)
        y, x = np.ogrid[top:bottom + 1, :right - left]
        fade_x = np.clip(np.minimum(x, right - left - 1 - x) / 13, 0, 1)
        fade_y = np.clip(np.minimum(y - top, bottom - y) / 9, 0, 1)
        if top == 0:
            # Ceiling-hugging source rails leave no clean upper donor.
            fill[:bottom + 1, left:right] = np.median(wall[bottom:bottom + 7, left:right], axis=0)
            fade_y = np.clip((bottom - y) / 9, 0, 1)
        opacity = (fade_x * fade_y)[:, :, None]
        patch = wall[top:bottom + 1, left:right]
        patch[:] = np.rint(opacity * fill[top:bottom + 1, left:right] + (1 - opacity) * patch)
    return wall


def rail_art(source, wall):
    """Extract a solid wood sprite and a translucent, wall-independent shadow."""
    top, bottom, left, right = source.bands[0]
    hooks = detect_hook_xs(source.rgb, source.bands[0])
    cleaned = surgical_remove_hooks(source.rgb, hooks, top, bottom, left, right, below=28, above=12, halfw=19)
    sprite, sx, sy, wood_offset, _ = extract_rail_sprite(cleaned, source.bands[0])
    rgba = np.array(sprite)
    # A painted mounting bracket can extend farther than the default sprite
    # crop. Keep the brown hardware at the ends, using the wall only as alpha.
    sw = sprite.width
    sh = min(H, bottom + 33) - sy
    raw = cleaned[sy:sy + sh, sx:sx + sw]
    row, col = np.ogrid[:sh, :sw]
    colors = raw.copy()
    colors[:sprite.height] = rgba[:, :, :3]
    on_wood = (row >= top - sy - 1) & (row <= bottom - sy + 1)
    near_end = (col < left - sx + 56) | (col > right - sx - 56)
    hardware = np.uint8(wood_mask(raw) & near_end) * 255
    hardware = cv2.morphologyEx(hardware, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hardware = cv2.GaussianBlur(hardware, (3, 3), .55)
    wood_alpha = hardware
    wood_alpha[:sprite.height] = np.maximum(
        hardware[:sprite.height], np.where(on_wood[:sprite.height], rgba[:, :, 3], 0)
    )
    # The existing extraction can include pieces of the donor wall under
    # a rod. Use their darkness as opacity so the new wall color shows through.
    expected = wall[sy:sy + sh, sx:sx + sw].mean(2)
    observed = raw.mean(2)
    shadow_alpha = np.clip((expected - observed - 2.5) / np.maximum(expected, 1), 0, .44)
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (5, 5), .8)
    shadow_alpha *= np.clip((sh - 1 - row) / 4, 0, 1)
    shadow_alpha *= np.clip(np.minimum(col, sw - 1 - col) / 4, 0, 1)
    shadow_alpha *= (row >= top - sy - 3)
    shadow = Image.fromarray(np.dstack((np.zeros((sh, sw, 3), dtype=np.uint8),
                                      np.uint8(np.rint(255 * shadow_alpha)))))
    wood = Image.fromarray(np.dstack((colors, wood_alpha)).astype(np.uint8))
    shadow.alpha_composite(wood)
    return shadow, sx - left, wood_offset


def fit_sprite(sprite, width):
    """Fit the shaft to the target span without distorting mounting hardware."""
    if sprite.width == width:
        return sprite
    cap = 64
    image = Image.new("RGBA", (width, sprite.height))
    image.paste(sprite.crop((0, 0, cap, sprite.height)), (0, 0))
    shaft = sprite.crop((cap, 0, sprite.width - cap, sprite.height))
    image.paste(shaft.resize((width - 2 * cap, sprite.height), Image.Resampling.LANCZOS), (cap, 0))
    image.paste(sprite.crop((sprite.width - cap, 0, sprite.width, sprite.height)), (width - cap, 0))
    return image


def render_plate(room, source, get_wall, get_art):
    bg = room.background
    sprite, inset, wood_offset = get_art(source.id)
    width = bg.span[1] - bg.span[0] + sprite.width - (source.span[1] - source.span[0])
    sprite = fit_sprite(sprite, width)
    canvas = Image.fromarray(get_wall(bg.id)).convert("RGBA")
    rails = []
    for top in room.wood_tops:
        canvas.alpha_composite(sprite, (bg.span[0] + inset, top - wood_offset))
        rails.append({
            "rail_bottom": top + source.wood_height - 1,
            "x0": bg.span[0],
            "x1": bg.span[1] - 1,
        })
    return Image.fromarray(heal_under_rail(np.array(canvas.convert("RGB")), rails))


def pct_box(x, y, w, h):
    return {"left": 100 * x / W, "top": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H}


def position_items(room, hook):
    placed = []
    b = room.background
    rail_gap = SHORT_GAP if len(room.rows) == 3 else LONG_GAP
    for i, (top, row) in enumerate(zip(room.wood_tops, room.rows)):
        for p in row.items:
            g = p.garment
            photo = g.photo
            top_px = top + hook - g.anchor_y * p.scale
            box = pct_box(p.box_left, top_px, photo.width * p.scale, photo.height * p.scale)
            length_ratio = float(p.scale / g.scale)
            px_height = float(photo.garment_height * p.scale)
            old = g.item
            layer = {**old["layer"], **box, "rect": dict(box), "zIndex": len(placed) + 1}
            layer["pivot"] = {"x": layer["pivotX"], "y": layer["pivotY"]}
            item = {**old, "zone": room.id, "layer": layer, "hotspot": {**old["hotspot"], **box}}
            note = g.note
            if length_ratio < 1 - EPS:
                note += f"; fit cap {length_ratio:.6f}×"
            measurement = {
                "item_id": old["id"], "old_zone": old["zone"], "new_zone": room.id,
                "rod": i + 1, "type": old["type"], "family": g.family, "length_cm": round(g.length, 4),
                "length_source": g.source, "px_height": round(px_height, 4),
                "scale_ratio_vs_48f8268": round(px_height / visual_h_px(old["layer"]), 6),
                "cap_ratio": round(length_ratio, 6), "note": note,
            }
            if not (0 <= box["left"] and box["left"] + box["width"] <= 100 and
                    0 <= box["top"] and box["top"] + box["height"] <= 100):
                raise ValueError(f"Layer out of bounds: {old['id']}")
            if top + hook + g.hem * p.scale + FLOOR_GAP > b.limit(p.opaque_left, p.opaque_right) + EPS:
                raise ValueError(f"Layer crosses the obstacle clearance: {old['id']}")
            if i + 1 < len(room.wood_tops):
                if top + hook + g.hem * p.scale + rail_gap > room.wood_tops[i + 1] + EPS:
                    raise ValueError(f"Layer crosses the next rail clearance: {old['id']}")
            placed.append((item, measurement, length_ratio))
    for a, (first, _, _) in enumerate(placed):
        l = first["hotspot"]
        for second, _, _ in placed[a + 1:]:
            r = second["hotspot"]
            if (min(l["left"] + l["width"], r["left"] + r["width"]) > max(l["left"], r["left"]) + EPS
                    and min(l["top"] + l["height"], r["top"] + r["height"]) > max(l["top"], r["top"]) + EPS):
                raise ValueError(f"Overlapping hotspots: {first['id']} and {second['id']}")
    return placed


def composite(plate, placements):
    """Use rebuild_covers' z-order, percent rects, and Lanczos compositing."""
    base = plate.convert("RGBA")
    for item, _, _ in sorted(placements, key=lambda p: p[0]["layer"]["zIndex"]):
        layer = item["layer"]
        with Image.open(ROOT / layer["image"]) as source:
            size = (max(1, round(layer["width"] / 100 * base.width)),
                    max(1, round(layer["height"] / 100 * base.height)))
            position = (round(layer["left"] / 100 * base.width), round(layer["top"] / 100 * base.height))
            base.alpha_composite(source.convert("RGBA").resize(size, Image.Resampling.LANCZOS), position)
    return base.convert("RGB")


def room_info(room, placed, hook, source, style):
    bg = room.background
    rails = []
    for index, (y, row) in enumerate(zip(room.wood_tops, room.rows), 1):
        items = []
        for item, measurement, ratio in placed:
            if measurement["rod"] != index:
                continue
            items.append(dict(measurement, name=item["display_name"], cap_ratio=round(ratio, 6)))
        after = (room.wood_tops[index] - y - hook - row.hem if index < len(room.rows)
                 else row.clearance(bg) - y - hook)
        rails.append({
            "wood_top": y, "hook_y": y + hook, "opaque_gap_px": round(row.gap, 3) if row.gap else None,
            "minimum_clearance_below_px": round(after, 3), "items": items,
        })
    return {
        "id": room.id, "background": bg.id, "rail_sprite": source.id, "rail_style": style,
        "rod_wood_height_px": source.wood_height,
        "rail_span": list(bg.span), "detected_obstacle_y": bg.detected_obstacle, "wall_limit_y": bg.obstacle,
        "furniture": [{"x0": x0, "x1": x1, "y": y} for x0, x1, y in bg.furniture],
        "rods": rails,
    }


def update_catalog(catalog, category, room_files, members):
    existing = [z for z in catalog["zones"] if z["category"] == category]
    if not existing:
        raise ValueError(f"Catalog missing category {category}")
    templates = {z["id"]: z for z in existing}
    zones = []
    for zid, (images, cover, placed) in room_files.items():
        zones.append({
            **templates.get(zid, existing[0]), "id": zid, "letter": zid[-1], "category": category,
            "images": images, "cover": cover, "natural_size": [W, H], "aspect": W / H,
            "has_layout": True, "items": [item["id"] for item, _, _ in placed],
        })
    first = catalog["zones"].index(existing[0])
    catalog["zones"] = [z for z in catalog["zones"] if z["category"] != category]
    catalog["zones"][first:first] = zones
    catalog["items"] = [members.get(i["id"], i) for i in catalog["items"]]
    cat = next(c for c in catalog["categories"] if c["id"] == category)
    cat.update(count=len(members), cover_zone=zones[0]["id"])

    catalog["redirects"] = room_redirects(
        catalog.get("redirects", {}), category, [z["id"] for z in existing], list(room_files)
    )


def room_redirects(previous, category, old_rooms, new_rooms):
    redirects = dict(previous)
    letters = [z[-1] for z in new_rooms]
    removed = {z[-1] for z in old_rooms if z[-1] not in letters}
    for letter in letters:
        # Reintroducing a room also revives both of its older links.
        redirects.pop(f"{category}/{letter}", None)
        redirects.pop(f"{category}/{category}-{letter}", None)
    for key, target in list(redirects.items()):
        prefix = f"#/{category}/"
        if target.startswith(prefix):
            dest = target[len(prefix):].removeprefix(f"{category}-")
            if dest in removed:
                nearest = min(letters, key=lambda letter: abs(ord(letter) - ord(dest)))
                redirects[key] = f"#/{category}/{nearest}"
    for old in sorted(removed):
        nearest = min(letters, key=lambda letter: abs(ord(letter) - ord(old)))
        redirects[f"{category}/{old}"] = f"#/{category}/{nearest}"
        redirects[f"{category}/{category}-{old}"] = f"#/{category}/{nearest}"
    return redirects


def comparison(zid, after, destinations=()):
    before_path = REPORT / "before" / f"{zid}.jpg"
    w, h = W, H
    image = Image.new("RGB", (2 * w, h + 64), (33, 36, 41))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    draw.text((24, 16), "BEFORE" if before_path.exists() else "BEFORE  (new room)", font=font, fill="white")
    draw.text((w + 24, 16), "AFTER" if after else "AFTER  (room removed)", font=font, fill="white")
    if before_path.exists():
        with Image.open(before_path) as before:
            fitted = ImageOps.contain(before.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        image.paste(fitted, ((w - fitted.width) // 2, 64 + (h - fitted.height) // 2))
    if after:
        image.paste(after, (w, 64))
    elif destinations:
        draw.text((w + 24, 112), "Items moved to " + ", ".join(destinations), font=font, fill="white")
    return image


def write_reports(plan):
    rows, lines = [], []
    for category, info in plan["categories"].items():
        lines.append(f"{category}  |  {len(info['rooms_before'])} → {len(info['rooms'])} rooms  |  {info['item_count']} items")
        lines.append(f"  Rail style {info['rail_style']} ({info['rail_sprite']})")
        for key, target in info["redirect_changes"].items():
            lines.append(f"  Redirect {key} → {target}" if target else f"  Revive direct link {key}")
        for room in info["rooms"]:
            lines.append(f"  {room['id']}  |  {len(room['rods'])} rods  |  background {room['background']}")
            for number, rod in enumerate(room["rods"], 1):
                lines.append(f"    Rod {number} at y={rod['wood_top']}  |  {len(rod['items'])} items, left → right")
                for item in rod["items"]:
                    suffix = f" (cap {item['cap_ratio']:.6f}×)" if item["cap_ratio"] < 1 else ""
                    lines.append(
                        f"      {item['item_id']}  {item['name']}  "
                        f"[{item['type']}, {item['length_cm']:.1f} cm, {item['px_height']:.1f} px]{suffix}"
                    )
                    rows.append(item)
        lines.append("")
    (REPORT / "grouping.txt").write_text("\n".join(lines), encoding="utf-8")
    (REPORT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REPORT / "sizes.csv").open("w", newline="", encoding="utf-8") as dest:
        columns = [
            "item_id", "old_zone", "new_zone", "rod", "type", "family", "length_cm", "length_source",
            "px_height", "scale_ratio_vs_48f8268", "note", "cap_ratio",
        ]
        writer = csv.DictWriter(dest, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_room(stage, room, plate, placed):
    stem = stage / "assets/zones" / room.id
    images = variants_from(plate, str(stem), (1600, 1024, 640))
    with Image.open(images[0]["src"]) as encoded_plate:
        after = composite(encoded_plate, placed)
    for image in images:
        image["src"] = str(Path(image["src"]).relative_to(stage))
    cover = after.resize((640, round(640 * H / W)), Image.Resampling.LANCZOS)
    path = stem.with_name(f"{room.id}-cover-640.webp")
    save_webp(cover, path)
    return images, {"src": str(path.relative_to(stage)), "width": cover.width, "height": cover.height}, placed


def publish(stage):
    """Publish the prepared catalog last and restore existing files on failure."""
    assets = sorted((stage / "assets/zones").glob("*.webp"))
    paths = [p.relative_to(stage) for p in (*assets, stage / "data/catalog.json")]
    backup = stage / "rollback"
    replaced = []
    try:
        for path in paths:
            dest = ROOT / path
            previous = backup / path
            if dest.exists():
                previous.parent.mkdir(parents=True, exist_ok=True)
                os.link(dest, previous)
            (stage / path).replace(dest)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            previous = backup / path
            if previous.exists():
                previous.replace(ROOT / path)
            else:
                (ROOT / path).unlink()
        raise


def run(categories, stage=None):
    baseline = json.loads((REPORT / f"catalog-before-{BASELINE}.json").read_text())
    live = json.loads(CATALOG.read_text()) if stage else None
    source_zones = {z["id"]: z for z in baseline["zones"] if z["category"] in FAMILIES}

    @cache
    def get_background(zid):
        return background(source_zones[zid])

    @cache
    def get_wall(zid):
        return clean_wall(get_background(zid))

    @cache
    def get_art(zid):
        return rail_art(get_background(zid), get_wall(zid))

    ref = get_background("jackets-b")
    px_per_cm, hook = calibration(baseline["items"], ref)
    records = garments(baseline, categories, px_per_cm, hook)
    plan = {
        "baseline": BASELINE, "plate_size": [W, H], "px_per_cm": px_per_cm, "hook_offset_px": hook,
        "short_hang_max_hook_to_hem_px": short_hem_limit(hook),
        "short_hang_rail_clearance_px": SHORT_GAP, "long_hang_rail_clearance_px": LONG_GAP,
        "long_hang_length_cm": LONG_LENGTH, "categories": {},
    }
    image_dir = REPORT / ("after" if stage else "preview")
    comparison_dir = REPORT if stage else image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    expected_previews = set()
    for category in categories:
        before_ids = [zid for zid, z in source_zones.items() if z["category"] == category]
        own = [get_background(zid) for zid in before_ids if zid != "jackets-f"]
        sprite_id, style = RAIL_STYLE[category]
        sprite = get_background(sprite_id)
        try:
            rooms = plan_category(records[category], own, (), category, hook)
        except ValueError:
            if category == "jackets":
                raise
            rooms = []
        if category != "jackets" and (not rooms or any(
            p.scale / p.garment.scale < 1 - EPS
            for room in rooms if len(room.rows) == 2 for row in room.rows for p in row.items
        )):
            siblings = [get_background(zid) for zid, z in source_zones.items() if z["category"] != category]
            rooms = plan_category(records[category], own, siblings, category, hook)

        members, files, info = {}, {}, []
        for room in rooms:
            placed = position_items(room, hook)
            members.update((item["id"], item) for item, _, _ in placed)
            plate = render_plate(room, sprite, get_wall, get_art)
            if stage:
                files[room.id] = stage_room(stage, room, plate, placed)
                after_path = image_dir / f"{room.id}.jpg"
            else:
                plate_path = image_dir / f"{room.id}-plate.png"
                plate.save(plate_path)
                after = composite(plate, placed)
                after_path = image_dir / f"{room.id}-after.jpg"
                after.save(after_path, quality=93, subsampling=0)
                expected_previews.update((plate_path, after_path))
            comparisons.append((room.id, after_path, ()))
            info.append(room_info(room, placed, hook, sprite, style))

        if len(members) != len(records[category]):
            raise ValueError(f"{category}: expected {len(records[category])} unique items, placed {len(members)}")
        after_ids = [room.id for room in rooms]
        removed = {}
        for zid in before_ids:
            if zid not in after_ids:
                destinations = sorted({members[i["id"]]["zone"] for i in baseline["items"] if i["zone"] == zid})
                removed[zid] = destinations
                comparisons.append((zid, None, destinations))
        if stage:
            update_catalog(live, category, files, members)
        previous = baseline.get("redirects", {})
        redirects = room_redirects(previous, category, before_ids, after_ids)
        plan["categories"][category] = {
            "item_count": len(members), "rooms_before": before_ids,
            "new_rooms": [z for z in after_ids if z not in before_ids],
            "removed_rooms": removed, "rooms": info,
            "rail_style": style, "rail_sprite": sprite_id,
            "redirect_changes": {
                key: redirects.get(key) for key in sorted(previous.keys() | redirects.keys())
                if previous.get(key) != redirects.get(key)
            },
            "capped_items": [
                {"item_id": item["item_id"], "ratio": item["cap_ratio"]}
                for room in info for rail in room["rods"] for item in rail["items"] if item["cap_ratio"] < 1
            ],
        }
        print(f"{category}: {len(members)} items in {len(rooms)} rooms")

    if stage:
        (stage / "data").mkdir()
        (stage / "data/catalog.json").write_text(json.dumps(live, ensure_ascii=False, indent=2) + "\n")
        publish(stage)
        written = json.loads(CATALOG.read_text())
        for zid, path, _ in comparisons:
            if path:
                after, _ = composite_catalog_room(written, zid)
                after.save(path, quality=93, subsampling=0)

    for zid, path, destinations in comparisons:
        if path:
            with Image.open(path) as after:
                combined = comparison(zid, after, destinations)
        else:
            combined = comparison(zid, None, destinations)
        comparison_path = comparison_dir / f"{zid}-before-after.jpg"
        combined.save(comparison_path, quality=91, subsampling=0)
        expected_previews.add(comparison_path)
    if not stage:
        category_pattern = "|".join(re.escape(c) for c in categories)
        generated = re.compile(rf"(?:{category_pattern})-[a-z]-(?:plate\.png|after\.jpg|before-after\.jpg)")
        for path in image_dir.iterdir():
            if generated.fullmatch(path.name) and path not in expected_previews:
                path.unlink()
    write_reports(plan)
    print(f"Used {px_per_cm:.8f} px/cm with a {hook:g} px hook offset. Reports at {REPORT}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", required=True, help="comma-separated hanging category ids")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="write preview images and planning reports only")
    mode.add_argument("--write", action="store_true", help="write plates, covers, catalog and reports")
    args = parser.parse_args()
    categories = list(dict.fromkeys(part.strip() for part in args.categories.split(",")))
    if unknown := set(categories) - FAMILIES.keys():
        parser.error(f"unknown hanging categories: {', '.join(map(repr, sorted(unknown)))}")
    if args.dry_run:
        run(categories)
    else:
        # Serialize writers without leaving a lock file in the worktree.
        lock = os.open(ROOT, os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with tempfile.TemporaryDirectory(prefix=".relayout6-", dir=ROOT) as temp:
                run(categories, Path(temp))
        finally:
            os.close(lock)


if __name__ == "__main__":
    main()
