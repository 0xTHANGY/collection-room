#!/usr/bin/env python3
"""Option-B fill for tote rooms. Default: dry-run totes-a only (no catalog write)."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import dataclass
import fcntl
from io import BytesIO
from itertools import combinations
import json
from math import ceil, floor
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True

from rebuild_covers import composite
import relayout6 as rails
from repack_three_rails import detect_rail_bands, heal_under_rail, save_webp, variants_from, wood_mask

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parent / "collection-room-impl-report"
OUT = REPORT / "fill-all"
CATALOG = Path("data/catalog.json")
BASELINE = "88d42ef"
ZONES = ("totes-a", "totes-b", "totes-c")
RAIL_SOURCE = "totes-a"
W, H = rails.W, rails.H
EDGE, BODY_GAP, RAIL_CLEAR, FLOOR_CLEAR = 24, 28, 12, 14
HOOK_OFFSET = 10
MAX_BAGS_ON_TWO_RODS = 8
KIND_ORDER = ("mini", "canvas", "organza", "market", "woven", "reusable", "technical")
JUDGMENTS = {
    "batch2-326": "The Topologie Cloud Tote stays technical, grouped with the other performance/packable totes on the lower rod when they share a room.",
    "batch2-322": "ApeFest thank-you bag is a packable reusable, not a merch canvas tote.",
    "batch2-358": "Echigo-Tsumari bag is a packable reusable, same family as BAGGU-style totes.",
}

@dataclass(frozen=True)
class Bag:
    id: str
    brand: str
    model: str
    category: str
    old_zone: str
    height_cm: float
    found: bool
    ratio: float
    width: float
    height: float
    pivot_x: float
    pivot_y: float
    alpha: tuple[float, float, float, float]


@dataclass(frozen=True)
class Footprint:
    bag: Bag
    width: float
    height: float
    # Visible offsets from the box's left and from the rail's wood top.
    left: float
    right: float
    drop: float
    above: float
    box_right: float
    box_drop: float


@dataclass(frozen=True)
class Placement:
    item: Footprint
    x: int
    top: int

    @property
    def rect(self):
        item = self.item
        return rails.pct_box(self.x, self.top + HOOK_OFFSET - item.height * item.bag.pivot_y,
                             item.width, item.height)


@dataclass(frozen=True)
class Room:
    bg: rails.Background
    benches: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class Fit:
    tops: list[int]
    rows: list[list[Placement]]
    slack: int


def baseline(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)


def read_rows(path, key):
    with path.open(encoding="utf-8-sig", newline="") as infile:
        return {row[key]: row for row in csv.DictReader(infile)}



def classify(item):
    typ = item["type"]
    brand = (item.get("brand") or "").lower()
    name = (item.get("name") or item.get("display_name") or "").lower()
    if typ == "mini-tote":
        return "mini"
    if typ == "woven-shopper":
        return "woven"
    if typ in ("reusable-bag", "shopping-bag"):
        return "reusable"
    if typ == "technical-tote":
        return "technical"
    if typ in ("organza-tote", "organza-baguette") or "organza" in typ or "organza" in name:
        return "organza"
    if typ in ("tote", "bag") and any(tag in brand for tag in ("bode", "baziszt")):
        return "market"
    return "canvas"


def load_bags(catalog):
    previous = read_rows(REPORT / "sizegroup/bags.csv", "item_id")
    specs = read_rows(REPORT / "fill-all/specs.csv", "id")
    bags = []
    for item in catalog["items"]:
        if item["category"] != "totes":
            continue
        iid = item["id"]
        spec, old = specs[iid], previous[iid]
        found = spec["confidence"] != "未查到"
        old_height = float(old["height_cm"])
        height = float(spec["height_cm"]) if found and spec.get("height_cm") else old_height
        if not (old_height > 0 and height > 0):
            raise ValueError(f"{iid}: invalid source height")
        ratio = height / old_height
        layer = item["layer"]
        with Image.open(ROOT / layer["image"]) as image:
            alpha = image.convert("RGBA").getchannel("A")
            x0, y0, x1, y1 = alpha.point(lambda p: 255 if p > 24 else 0).getbbox()
            bounds = (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)
        bags.append(Bag(
            iid, spec["brand"] or old["brand"] or item["brand"],
            spec["model"] or old["model"] or item["name"],
            classify(item), item["zone"], height, found, ratio,
            layer["width"] * W / 100 * ratio, layer["height"] * H / 100 * ratio,
            layer["pivotX"] / 100, layer["pivotY"] / 100, bounds,
        ))
    if len({bag.id for bag in bags}) != 42 or len(bags) != 42:
        raise ValueError("Expected 42 distinct totes")
    return bags


def assign_rooms(bags, keep_rooms=True):
    """First pass keeps current room membership and sorts by kind then size."""
    if keep_rooms:
        rooms = {zid: [] for zid in ZONES}
        for bag in bags:
            rooms[bag.old_zone].append(bag)
        for zid in rooms:
            rooms[zid] = sorted(rooms[zid], key=lambda bag: (
                KIND_ORDER.index(bag.category) if bag.category in KIND_ORDER else 99,
                bag.height_cm, bag.height, bag.id))
        return rooms
    groups = {kind: [bag for bag in bags if bag.category == kind] for kind in KIND_ORDER}
    # Later passes may reassign across totes-a..c.
    return {zid: sorted((bag for bag in bags if bag.old_zone == zid),
                        key=lambda bag: (KIND_ORDER.index(bag.category) if bag.category in KIND_ORDER else 99,
                                         bag.height_cm, bag.height, bag.id))
            for zid in ZONES}


def floor_y(image):
    """Find the lower baseboard edge using the clear middle of the wall."""
    level = np.median(np.asarray(image.convert("L"), dtype=float)[:, 300:1040], axis=1)
    # Both baseboard edges can be strong. Take the last sustained downward
    # step before the rug, then locate the sharpest pixel transition in it.
    edges = [y for y in range(835, 920)
             if np.median(level[y - 5:y]) - np.median(level[y:y + 5]) >= 18]
    if not edges:
        raise ValueError("Could not locate the floor line")
    last_edge = [edges[-1]]
    for y in reversed(edges[:-1]):
        if y != last_edge[-1] - 1:
            break
        last_edge.append(y)
    return max(last_edge, key=lambda y: (level[y - 1] - level[y], -y))


def load_room(zone):
    zid = zone["id"]
    src = max(zone["images"], key=lambda img: img["width"])["src"]
    with Image.open(BytesIO(baseline(src))) as source:
        image = source.convert("RGB")
    if image.size != (W, H):
        raise ValueError(f"{zid}: expected a {W}×{H} plate")
    rgb = np.asarray(image)
    bands = tuple(detect_rail_bands(rgb))
    if len(bands) != 3:
        raise ValueError(f"{zid}: expected three original rails, found {bands}")
    top, bottom, left, right = bands[0]
    floor_line = floor_y(image)
    benches = ()
    if zid == "bags-b":
        # Look below the third rod and above the rug for the wooden bench.
        wood = wood_mask(rgb[700:floor_line + 28, W // 2:]).astype("uint8")
        mask = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, _, components, _ = cv2.connectedComponentsWithStats(mask)
        candidates = [(x + W // 2, y + 700, x + w + W // 2, y + h + 700)
                      for x, y, w, h, area in components[1:].tolist()
                      if w > 220 and h > 40 and area > 5000 and y + 700 < floor_line - 35]
        if len(candidates) != 1:
            raise ValueError(f"{zid}: could not isolate the bench: {candidates}")
        benches = tuple(candidates)
    bg = rails.Background(zid, rgb, bands, (left, right + 1), bottom - top + 1, floor_line,
                          tuple((x0, x1, y0) for x0, y0, x1, _ in benches), floor_line)
    return Room(bg, benches)


def footprint(bag, k):
    width, height = bag.width * k, bag.height * k
    # Take the outer bound of the CSS geometry and the integer render used
    # in the covers. The alpha window is relative to the original cutout.
    ax0, _, ax1, ay1 = bag.alpha
    rw, rh = max(1, round(width)), max(1, round(height))
    top = HOOK_OFFSET - height * bag.pivot_y
    return Footprint(
        bag, width, height, min(ax0 * width, floor(ax0 * rw)),
        max(ax1 * width, ceil(ax1 * rw)),
        max(top + ay1 * height, round(top) + ceil(ay1 * rh)),
        min(top, round(top)), max(width, rw), max(top + height, round(top) + rh),
    )


def pack_row(row, span, bench_left=None, left_count=0):
    left, right = span[0] + EDGE, span[1] - EDGE
    steps = [ceil(max(a.box_right, a.right + BODY_GAP - b.left)) for a, b in zip(row, row[1:])]
    first = [max(0, ceil(left - row[0].left))]
    for step in steps:
        first.append(first[-1] + step)
    last = [floor(min((bench_left if i < left_count else right) - item.right,
                      W - item.box_right))
            for i, item in enumerate(row)]
    for i in range(len(steps) - 1, -1, -1):
        last[i] = min(last[i], last[i + 1] - steps[i])
    if first[0] > last[0]:
        return None
    spare = last[-1] - first[-1]
    positions = [min(last[0], first[0] + spare // (len(row) + 1))]
    for i, step in enumerate(steps, 1):
        ideal = first[i] + spare * (i + 1) // (len(row) + 1)
        positions.append(min(last[i], max(ideal, positions[-1] + step)))
    return positions


def bottom_row(row, room):
    floor_limit = floor(room.bg.obstacle - FLOOR_CLEAR - max(item.drop for item in row))
    if not room.benches:
        positions = pack_row(row, room.bg.span)
        return (row, positions, floor_limit) if positions else None

    bench_left, bench_top, bench_right, _ = room.benches[0]
    if bench_right < room.bg.span[1] - EDGE:
        raise ValueError("Expected the bags-b bench to extend past the usable rail")
    # At any given rod height, bags reaching the bench must fit entirely on
    # its left. Try each height at which that set changes.
    for limit in sorted({floor_limit} | {floor(bench_top - FLOOR_CLEAR - item.drop)
                                         for item in row}, reverse=True):
        if limit > floor_limit:
            continue
        left = [item for item in row if limit + item.drop > bench_top - FLOOR_CLEAR]
        right = [item for item in row if item not in left]
        ordered = left + right
        positions = pack_row(ordered, room.bg.span, bench_left, len(left))
        if positions is not None:
            return ordered, positions, limit
    return None


def fit_rows(rows, room, cents):
    rows = [[footprint(bag, cents / 100) for bag in row] for row in rows]
    bottom = bottom_row(rows[-1], room)
    if bottom is None:
        return None
    rows[-1], bottom_positions, limit = bottom
    positions = [pack_row(row, room.bg.span) for row in rows[:-1]] + [bottom_positions]
    if any(pos is None for pos in positions):
        return None
    # Honor both visible clearance to the next rail and full hotspot
    # separation between rows, including any pixels above a cutout's pivot.
    steps = [ceil(max(max(item.drop for item in row) + RAIL_CLEAR,
                      max(item.box_drop for item in row) - min(item.above for item in below)))
             for row, below in zip(rows, rows[1:])]
    first = room.bg.bands[0][0]
    if first + min(item.above for item in rows[0]) < 0:
        return None
    slack = limit - first - sum(steps)
    if slack < 0:
        return None
    tops = [first]
    for i, step in enumerate(steps):
        tops.append(tops[-1] + step + slack * (i + 1) // len(rows) - slack * i // len(rows))
    placed = [[Placement(item, x, top) for item, x in zip(row, xs)]
              for top, row, xs in zip(tops, rows, positions)]
    return Fit(tops, placed, slack)


def candidate_rows(bags, room, rod_count):
    # Prefer splits that keep kinds together and avoid singleton rods.
    for cuts in combinations(range(1, len(bags)), rod_count - 1):
        bounds = (0, *cuts, len(bags))
        rows = [bags[lo:hi] for lo, hi in zip(bounds, bounds[1:])]
        if any(len(row) < 3 for row in rows):
            continue
        if rod_count == 3 or all(len(row) <= MAX_BAGS_ON_TWO_RODS for row in rows):
            yield rows


def best_rows(bags, room, rod_count):
    best = None
    for rows in candidate_rows(bags, room, rod_count):
        mixed = sum(len({bag.category for bag in row}) > 1 for row in rows)
        if best and mixed > best[0][0]:
            continue
        if fit_rows(rows, room, 1) is None:
            continue
        low, high = 1, 100
        while fit_rows(rows, room, high) is not None:
            low, high = high, high * 2
        while high - low > 1:
            mid = (low + high) // 2
            if fit_rows(rows, room, mid) is None:
                high = mid
            else:
                low = mid
        fit = fit_rows(rows, room, low)
        rank = (mixed, -low, max(map(len, rows)) - min(map(len, rows)), -fit.slack)
        if best is None or rank < best[0]:
            best = rank, low, fit
    return best


def plan_room(bags, room):
    if len(bags) < 8:
        raise ValueError(f"{room.bg.id}: expected at least eight totes")
    three_rods = best_rows(bags, room, 3)
    two_rods = best_rows(bags, room, 2)
    alternatives = {"3": None, "2": None}
    if three_rods is None and two_rods is None:
        raise ValueError(f"{room.bg.id}: no feasible 2- or 3-rod layout")
    if three_rods is not None:
        rank3, cents3, fit3 = three_rods
        alternatives["3"] = cents3 / 100
    if two_rods is not None:
        rank2, cents2, fit2 = two_rods
        alternatives["2"] = cents2 / 100
        if three_rods is None or (rank2[0] <= rank3[0] and cents2 > cents3 + 5):
            return cents2 / 100, fit2, alternatives
    return cents3 / 100, fit3, alternatives


def apply_plan(catalog, plans):
    items = {item["id"]: item for item in catalog["items"]}
    zones = {zone["id"]: zone for zone in catalog["zones"]}
    for zid, (_, fit, _) in plans.items():
        placed = [p for row in fit.rows for p in row]
        zones[zid]["items"] = [p.item.bag.id for p in placed]
        for z_index, p in enumerate(placed, 1):
            bag, box = p.item.bag, p.rect
            item = items[bag.id]
            item["zone"] = zid
            item["layer"].update(box, rect=box, zIndex=z_index, pivotX=bag.pivot_x * 100,
                                 pivotY=bag.pivot_y * 100,
                                 pivot={"x": bag.pivot_x * 100, "y": bag.pivot_y * 100})
            item["hotspot"].update(box)


def validate_catalog(before, after, active):
    original = {item["id"]: item for item in before["items"]}
    items = {item["id"]: item for item in after["items"]}
    if list(original) != list(items) or before["redirects"] != after["redirects"]:
        raise ValueError("Item IDs or redirects changed")
    category_counts = Counter(item["category"] for item in items.values())
    if (category_counts["bags"], category_counts["totes"]) != (53, 42):
        raise ValueError("Expected 53 bags and 42 totes")
    zones = {zone["id"]: zone for zone in after["zones"]}
    tote_zones = [zid for zid in ZONES if zid in zones]
    if tote_zones != list(ZONES):
        raise ValueError("Expected totes-a through totes-c")
    membership = []
    for zid in ZONES:
        zone = zones[zid]
        if zone["letter"] != zid[-1] or f"totes/{zone['letter']}" in after["redirects"]:
            raise ValueError(f"{zid}: the existing room route would change")
        membership.extend(zone["items"])
        if any(iid not in items or items[iid]["zone"] != zid for iid in zone["items"]):
            raise ValueError(f"{zid}: item membership mismatch")
        if zid not in active:
            continue
        boxes = []
        for iid in zone["items"]:
            item = items[iid]
            box = item["layer"]["rect"]
            if any(item["layer"][k] != v or item["hotspot"][k] != v for k, v in box.items()):
                raise ValueError(f"{iid}: layer and hotspot boxes differ")
            x, y, width, height = (box[k] for k in ("left", "top", "width", "height"))
            if not (0 <= x < x + width <= 100 and 0 <= y < y + height <= 100):
                raise ValueError(f"{iid}: box outside the room")
            for ax, ay, bx, by in boxes:
                if min(bx, x + width) - max(ax, x) > 1e-9 and min(by, y + height) - max(ay, y) > 1e-9:
                    raise ValueError(f"{iid}: overlapping hotspots")
            boxes.append((x, y, x + width, y + height))
    if Counter(membership) != Counter(iid for iid, item in items.items() if item["category"] == "totes"):
        raise ValueError("Every tote must occur in exactly one room")
    for iid, item in items.items():
        old = original[iid]
        moved = item["category"] == "totes" and item["zone"] in active
        mutable = {"zone", "layer", "hotspot"} if moved else set()
        if any(item[key] != old[key] for key in old.keys() - mutable):
            raise ValueError(f"{iid}: protected item fields changed")
        if moved and (item["layer"]["image"] != old["layer"]["image"]
                      or item["hotspot"].get("clip") != old["hotspot"].get("clip")):
            raise ValueError(f"{iid}: layer image or hotspot clip changed")
    for key, target in after["redirects"].items():
        if key.startswith("totes/") and target.removeprefix("#/") not in {
            f"totes/{zones[zid]['letter']}" for zid in ZONES
        }:
            raise ValueError(f"{key}: unresolved totes redirect {target}")


def validate_paths(catalog, stage=None):
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("src", "image") and isinstance(child, str):
                    if not ((stage and (stage / child).is_file()) or (ROOT / child).is_file()):
                        raise FileNotFoundError(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(catalog)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def comparison(zid, image, factor, path):
    with Image.open(OUT / "before" / f"{zid}.jpg") as before:
        before = before.resize(image.size, Image.Resampling.LANCZOS)
    w, h = image.size
    label_h = 76
    canvas = Image.new("RGB", (2 * w + 24, h + label_h), "white")
    canvas.paste(before, (0, label_h))
    canvas.paste(image, (w + 24, label_h))
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f"BEFORE  {zid}", font=font, fill="black")
    draw.text((w + 40, 10), f"AFTER  {zid}  ×{factor:.2f}", font=font, fill="black")
    canvas.resize((canvas.width // 2, canvas.height // 2), Image.Resampling.LANCZOS).save(path, quality=93)


def render_plates(rooms, plans, selected):
    source = rooms[RAIL_SOURCE].bg
    donor_wall = rails.clean_wall(source)
    sprite, inset, offset = rails.rail_art(source, donor_wall)
    extras = sprite.width - (source.span[1] - source.span[0])
    plates = {}
    for zid in selected:
        bg = rooms[zid].bg
        _, fit, _ = plans[zid]
        wall = donor_wall if zid == RAIL_SOURCE else rails.clean_wall(bg)
        left, right = bg.span
        fitted = rails.fit_sprite(sprite, right - left + extras)
        plate = Image.fromarray(wall).convert("RGBA")
        for y in fit.tops:
            plate.alpha_composite(fitted, (left + inset, y - offset))
        rgb = np.asarray(plate.convert("RGB"))
        healed = heal_under_rail(rgb, [
            {"rail_bottom": y + source.wood_height - 1, "x0": left, "x1": right - 1}
            for y in fit.tops
        ])
        # Feather out the repair before the mounting caps, where the
        # horizontal healing window would otherwise leave a hard notch.
        x = np.arange(W)
        fade = np.clip(np.minimum(x - left - 16, right - 17 - x) / 36, 0, 1)[None, :, None]
        plates[zid] = Image.fromarray(np.rint(rgb * (1 - fade) + healed * fade).astype(np.uint8))
    return plates


def preview(catalog, plans, plates):
    out = OUT / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for zid, plate in plates.items():
        path = out / f"{zid}-plate.png"
        plate.save(path)
        zone = next(zone for zone in catalog["zones"] if zone["id"] == zid)
        preview_zone = {**zone, "images": [{"src": str(path), "width": W, "height": H}]}
        image, _ = composite({"zones": [preview_zone], "items": catalog["items"]}, zid)
        image.save(out / f"{zid}-after.jpg", quality=95)
        comparison(zid, image, plans[zid][0], out / f"{zid}-before-after.jpg")


def write_reports(rooms, plans, selected):
    records, grouping, summary, factors = [], [], {}, {}
    for zid, (factor, fit, alternatives) in plans.items():
        room = rooms[zid]
        counts = Counter(p.item.bag.category for row in fit.rows for p in row)
        factors[zid] = {"factor": factor, "alternatives_by_rod_count": alternatives}
        summary[zid] = {
            **factors[zid], "category_counts": dict(counts), "floor_y": room.bg.obstacle,
            "bench_boxes": room.benches, "rail_span": room.bg.span,
            "original_rail_bands": room.bg.bands, "rod_wood_tops": fit.tops,
            "unallocated_height_px": fit.slack, "rows": [],
        }
        grouping.append(f"{zid} — ×{factor:.2f} — " + ", ".join(f"{n} {c}" for c, n in counts.items()))
        for i, row in enumerate(fit.rows, 1):
            grouping.append(f"  Rod {i} (y={fit.tops[i - 1]}, {len(row)} bags)")
            members = []
            for p in row:
                bag = p.item.bag
                category = dict(id=bag.id, brand=bag.brand, model=bag.model, category=bag.category,
                                old_zone=bag.old_zone, new_zone=zid, height_cm=bag.height_cm,
                                height_source="spec" if bag.found else "7268bed",
                                found="yes" if bag.found else "no")
                records.append(category)
                members.append({**category, "spec_to_7268bed_ratio": bag.ratio, "rect_pct": p.rect})
                grouping.append(f"    {bag.id}  {bag.category}  {bag.height_cm:g} cm  "
                                f"{bag.brand} {bag.model}  ({category['height_source']})")
            summary[zid]["rows"].append(members)
        grouping.append("")
    grouping.append("Category decisions")
    grouping.extend(f"  {iid}  {note}" for iid, note in JUDGMENTS.items())
    with (OUT / "totes-preview-categories.csv").open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "totes-preview-grouping.txt").write_text("\n".join(grouping) + "\n", encoding="utf-8")
    (OUT / "totes-preview-not-found.txt").write_text(
        "Totes using current/sizegroup heights (未查到)\n"
        + "".join(f"{r['new_zone']}  {r['id']}  {r['height_cm']:g} cm  {r['brand']} {r['model']}\n"
                  for r in records if r["found"] == "no"), encoding="utf-8")
    dump(OUT / "totes-preview-plan.json", {"baseline": BASELINE, "rail_source": RAIL_SOURCE,
                            "rendered_zones": selected, "rooms": summary})
    dump(OUT / "totes-preview-factors.json", factors)


def stage_cache(stage):
    changed = []
    for name, pattern, count in (
        ("app.js", r"(const ASSET_V = ')2026112[56][a-z](';)", 1),
        ("index.html", r'((?:styles\.css|app\.js)\?v=)2026112[56][a-z](")', 2),
    ):
        previous = (ROOT / name).read_text(encoding="utf-8")
        contents, replacements = re.subn(pattern, r"\g<1>20261126a\2", previous)
        if replacements != count:
            raise ValueError(f"{name}: expected {count} cache version references")
        if contents != previous:
            (stage / name).write_text(contents, encoding="utf-8")
            changed.append(Path(name))
    return changed


def publish(stage, paths):
    """Replace prepared files and restore originals if a replacement fails."""
    replaced = []
    try:
        for path in paths:
            backup = stage / "rollback" / path
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / path, backup)
            (stage / path).replace(ROOT / path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            (stage / "rollback" / path).replace(ROOT / path)
        raise


def write(plans, plates):
    # The directory lock stays valid when the catalog is atomically replaced.
    root_fd = os.open(ROOT, os.O_RDONLY)
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        current = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
        target_zones = {p.item.bag.id: zid for zid, (_, fit, _) in plans.items()
                        for row in fit.rows for p in row}
        if any((item["zone"] in plates) != (target_zones[item["id"]] in plates)
               for item in current["items"] if item["id"] in target_zones):
            raise ValueError("The write must include both rooms for each remaining bag move")
        updated = deepcopy(current)
        selected_plans = {zid: plans[zid] for zid in plates}
        apply_plan(updated, selected_plans)
        validate_catalog(current, updated, plates)
        with TemporaryDirectory(prefix=".fill-totes-", dir=ROOT) as tmp:
            stage = Path(tmp)
            paths = []
            for zid, plate in plates.items():
                variants = variants_from(plate, str(stage / "assets/zones" / zid), (1600, 1024, 640))
                zone = next(zone for zone in updated["zones"] if zone["id"] == zid)
                staged_zone = {**zone, "images": variants}
                image, _ = composite({"zones": [staged_zone], "items": updated["items"]}, zid)
                for variant in variants:
                    variant["src"] = str(Path(variant["src"]).relative_to(stage))
                    paths.append(Path(variant["src"]))
                zone["images"] = variants
                cover = Path(f"assets/zones/{zid}-cover-640.webp")
                height = round(640 * image.height / image.width)
                save_webp(image.resize((640, height), Image.Resampling.LANCZOS), stage / cover)
                zone["cover"] = {"src": str(cover), "width": 640, "height": height}
                paths.append(cover)
            cache_paths = stage_cache(stage)
            validate_paths(updated, stage)
            (stage / CATALOG.parent).mkdir()
            dump(stage / CATALOG, updated)
            publish(stage, [*paths, CATALOG, *cache_paths])
        written = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
        validate_catalog(current, written, plates)
        validate_paths(written)
        (OUT / "after").mkdir(exist_ok=True)
        for zid in plates:
            image, _ = composite(written, zid)
            image.save(OUT / "after" / f"{zid}.jpg", quality=95)
            comparison(zid, image, plans[zid][0], OUT / f"{zid}-before-after.jpg")
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="write the reports and requested preview images")
    mode.add_argument("--write", action="store_true", help="write the plates, covers, catalog, and cache version")
    parser.add_argument("--zones", default=",".join(ZONES),
                        help="comma-separated rooms to render; writes must include both ends of each move")
    args = parser.parse_args()
    selected = tuple(zone.strip() for zone in args.zones.split(","))
    if set(selected) - set(ZONES) or len(selected) != len(set(selected)):
        parser.error("--zones must contain distinct room IDs from totes-a through totes-c")

    original = json.loads(baseline(CATALOG))
    assignment = assign_rooms(load_bags(original), keep_rooms=True)
    rooms = {zone["id"]: load_room(zone) for zone in original["zones"] if zone["id"] in selected}
    plans = {zid: plan_room(assignment[zid], rooms[zid]) for zid in selected}
    projected = deepcopy(original)
    apply_plan(projected, plans)
    validate_catalog(original, projected, selected)
    validate_paths(projected)
    plates = render_plates(rooms, plans, selected)
    if args.dry_run:
        preview(projected, plans, plates)
    else:
        write(plans, plates)
    write_reports(rooms, plans, selected)
    for zid, (factor, fit, _) in plans.items():
        counts = " / ".join(str(len(row)) for row in fit.rows)
        print(f"{zid}  ×{factor:.2f}  {counts}")


if __name__ == "__main__":
    main()
