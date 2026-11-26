#!/usr/bin/env python3
"""Option-B fill for hats and accessories rooms (spec scale, one k per room, regroup by kind)."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True

import fill_totes as ft
from fill_totes import (
    BODY_GAP, CATALOG, EDGE, FLOOR_CLEAR, OUT, REPORT, ROOT, W, H,
    Bag, Placement, Room, apply_plan, baseline, comparison, dump,
    fit_rows, footprint, pack_row, publish, read_rows,
    render_plates, stage_cache, validate_paths,
)
from rebuild_covers import composite
import relayout6 as rails
from repack_three_rails import heal_under_rail, save_webp, variants_from, wood_mask

RATIO_CLAMP = (0.7, 1.5)
RAW_RATIO = {}

CONFIGS = {
    "hats": {
        "zones": tuple(f"hats-{c}" for c in "abcdefg"),
        "rail_source": "hats-a",
        "catalog_category": "hats",
        "kind_order": (
            "cap", "beanie", "bucket", "earflap",
            "brim", "newsboy", "balaclava", "beret", "boater", "other",
        ),
        "expected_n": 98,
        "keep_rooms": False,
        # hats-d's painted rail is much shorter (~756 px) than the other hat rooms.
        "capacities": {"hats-a": 16, "hats-b": 14, "hats-c": 14, "hats-d": 12,
                       "hats-e": 14, "hats-f": 14, "hats-g": 14},
        "judgments": {},
    },
    "accessories": {
        "zones": ("accessories-a", "accessories-b", "accessories-c"),
        "rail_source": "accessories-a",
        "catalog_category": "accessories",
        "kind_order": (
            "belt", "tie", "gloves",
            "scarf", "silk-scarf", "twilly", "neck-warmer",
            "headband", "card-holder", "passport-cover",
        ),
        "expected_n": 35,
        "keep_rooms": False,
        "judgments": {},
    },
}



from itertools import combinations


def candidate_rows(bags, room, rod_count):
    span = room.bg.span[1] - room.bg.span[0]
    max_on_rod = 5 if span < 900 else 6
    min_on_rod = 2
    for cuts in combinations(range(1, len(bags)), rod_count - 1):
        bounds = (0, *cuts, len(bags))
        rows = [bags[lo:hi] for lo, hi in zip(bounds, bounds[1:])]
        if any(len(row) < min_on_rod or len(row) > max_on_rod for row in rows):
            continue
        yield rows


def best_rows(bags, room, rod_count):
    best = None
    for rows in candidate_rows(bags, room, rod_count):
        mixed = sum(len({bag.category for bag in row}) > 1 for row in rows)
        if fit_rows(rows, room, 1) is None:
            continue
        low, high = 1, 100
        while fit_rows(rows, room, high) is not None:
            low, high = high, high * 2
            if high > 400:
                break
        while high - low > 1:
            mid = (low + high) // 2
            if fit_rows(rows, room, mid) is None:
                high = mid
            else:
                low = mid
        fit = fit_rows(rows, room, low)
        # Prefer k>=1, then kind-pure rods, then no sparse (≤2) rods, then evenness, then larger k.
        n_short = sum(len(row) <= 2 for row in rows)
        rank = (0 if low >= 100 else 1, mixed, n_short, max(map(len, rows)) - min(map(len, rows)), -low, -fit.slack)
        if best is None or rank < best[0]:
            best = rank, low, fit
    return best


def plan_room(bags, room):
    if len(bags) < 6:
        raise ValueError(f"{room.bg.id}: expected at least six items")
    three = best_rows(bags, room, 3)
    two = best_rows(bags, room, 2)
    alternatives = {"3": None, "2": None}
    if three is None and two is None:
        raise ValueError(f"{room.bg.id}: no feasible 2- or 3-rod layout")
    if three is not None:
        rank3, cents3, fit3 = three
        alternatives["3"] = cents3 / 100
    if two is not None:
        rank2, cents2, fit2 = two
        alternatives["2"] = cents2 / 100
        if three is None or (rank2[0] < rank3[0]) or (rank2[0] == rank3[0] and rank2[1] <= rank3[1] and cents2 > cents3 + 5):
            return cents2 / 100, fit2, alternatives
    return cents3 / 100, fit3, alternatives


def classify(item, kind_order):
    typ = (item.get("type") or "other").lower()
    return typ if typ in kind_order else "other"


def floor_y(image):
    """Lower baseboard edge; search a wider band than tote rooms."""
    level = np.median(np.asarray(image.convert("L"), dtype=float)[:, 300:1040], axis=1)
    edges = [y for y in range(780, 940)
             if np.median(level[y - 5:y]) - np.median(level[y:y + 5]) >= 18]
    if not edges:
        raise ValueError("Could not locate the floor line")
    last_edge = [edges[-1]]
    for y in reversed(edges[:-1]):
        if y != last_edge[-1] - 1:
            break
        last_edge.append(y)
    return max(last_edge, key=lambda y: (level[y - 1] - level[y], -y))


def detect_rails(rgb):
    """Three hanging rails, including a slightly lower third rod (accessories-b)."""
    from repack_three_rails import detect_rail_bands
    bands = detect_rail_bands(rgb)
    if len(bands) >= 3:
        return tuple(bands[:3])
    h, w = rgb.shape[:2]
    wood = wood_mask(rgb)
    xs0, xs1 = int(w * 0.12), int(w * 0.88)
    ys = [y for y in range(0, int(h * 0.85)) if wood[y, xs0:xs1].mean() > 0.18]
    raw, a, prev = [], ys[0], ys[0]
    for y in ys[1:] + [10 ** 9]:
        if y > prev + 3:
            if prev - a >= 8:
                cols = np.where(wood[a:prev + 1, :].mean(0) > 0.2)[0]
                x0, x1 = (int(cols[0]), int(cols[-1])) if len(cols) else (xs0, xs1)
                raw.append((a, prev, x0, x1))
            a = y
        prev = y
    rails_found = []
    for y0, y1, x0, x1 in raw:
        if not (8 <= (y1 - y0) <= 70) or y0 > h * 0.75:
            continue
        below = wood[y1 + 1:min(h, y1 + 50), max(0, x0):min(w, x1 + 1)].mean() if y1 + 1 < h else 0
        if below > 0.25:
            continue
        rails_found.append((y0, y1, x0, x1))
    rails_found = sorted(rails_found, key=lambda b: b[0])[:3]
    if len(rails_found) not in (2, 3):
        raise ValueError(f"expected 2 or 3 rails, found {rails_found}")
    return tuple(rails_found)


def find_benches(rgb, bands, floor_line):
    last = max(b[1] for b in bands)
    wood = wood_mask(rgb[last + 10:floor_line + 28, :]).astype("uint8")
    mask = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    benches = []
    for x, y, w, h, area in components[1:].tolist():
        if w < 180 or w > 1000 or h < 35 or area < 4000:
            continue
        box = (int(x), int(y + last + 10), int(x + w), int(y + h + last + 10))
        # Legs may reach the floor; reject only if the top is already at/below it.
        if box[1] >= floor_line - 20:
            continue
        benches.append(box)
    return tuple(sorted(benches, key=lambda b: b[0]))


def load_room(zone):
    zid = zone["id"]
    src = max(zone["images"], key=lambda img: img["width"])["src"]
    from io import BytesIO
    with Image.open(BytesIO(baseline(src))) as source:
        image = source.convert("RGB")
    if image.size != (W, H):
        raise ValueError(f"{zid}: expected a {W}×{H} plate")
    rgb = np.asarray(image)
    bands = detect_rails(rgb)
    top, bottom, left, right = bands[0]
    # Prefer the widest rail span so a slightly shorter bottom rod still packs to the same ends.
    left = min(b[2] for b in bands)
    right = max(b[3] for b in bands)
    floor_line = floor_y(image)
    benches = find_benches(rgb, bands, floor_line)
    # A long low bench under the hanging area is a floor obstacle, not a right-side block.
    side_benches = []
    obstacle = floor_line
    for x0, y0, x1, y1 in benches:
        if (x1 - x0) >= 0.45 * W:
            obstacle = min(obstacle, y0)
        else:
            side_benches.append((x0, y0, x1, y1))
    bg = rails.Background(
        zid, rgb, bands, (left, right + 1), bottom - top + 1, obstacle,
        tuple((x0, x1, y0) for x0, y0, x1, _ in side_benches), floor_line,
    )
    return Room(bg, tuple(side_benches))


def load_items(catalog, cfg):
    specs = read_rows(REPORT / "fill-all/specs.csv", "id")
    bags, found_pairs = [], []
    for item in catalog["items"]:
        if item["category"] != cfg["catalog_category"]:
            continue
        iid = item["id"]
        if iid not in specs:
            raise ValueError(f"{iid}: missing specs.csv row")
        spec = specs[iid]
        found = spec["confidence"] != "未查到" and bool(spec.get("height_cm"))
        layer = item["layer"]
        width = layer["width"] * W / 100
        height = layer["height"] * H / 100
        height_cm = float(spec["height_cm"]) if found else height  # placeholder; ratio applied below
        with Image.open(ROOT / layer["image"]) as image:
            alpha = image.convert("RGBA").getchannel("A")
            x0, y0, x1, y1 = alpha.point(lambda p: 255 if p > 24 else 0).getbbox()
            bounds = (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)
        bag = Bag(
            iid, spec["brand"] or item.get("brand") or "",
            spec["model"] or item.get("name") or "",
            classify(item, cfg["kind_order"]), item["zone"],
            float(spec["height_cm"]) if found else 0.0, found, 1.0,
            width, height,
            layer["pivotX"] / 100, layer["pivotY"] / 100, bounds,
        )
        bags.append(bag)
        if found:
            found_pairs.append((bag, float(spec["height_cm"]), height))
    if len(bags) != cfg["expected_n"] or len({b.id for b in bags}) != cfg["expected_n"]:
        raise ValueError(f"Expected {cfg['expected_n']} distinct {cfg['catalog_category']}")
    # Looked-up sizes: share the median px/cm of found items so relative cm order is honored
    # and 未查到 / identify-but-no-height rows keep their current display size.
    if len(found_pairs) >= 2:
        px_per_cm = float(np.median([px / cm for _, cm, px in found_pairs]))
        rescaled = []
        for bag in bags:
            if not bag.found:
                rescaled.append(bag)
                continue
            target = bag.height_cm * px_per_cm
            raw = target / bag.height if bag.height else 1.0
            # Spec heights mix bases (cap crown depth vs. flat overall length of a ski mask),
            # so clamp per-item ratios like fill_shoes does and keep the raw value for the report.
            lo, hi = cfg.get("ratio_clamp", RATIO_CLAMP)
            ratio = min(hi, max(lo, raw))
            RAW_RATIO[bag.id] = raw
            rescaled.append(Bag(
                bag.id, bag.brand, bag.model, bag.category, bag.old_zone,
                bag.height_cm, bag.found, ratio,
                bag.width * ratio, bag.height * ratio,
                bag.pivot_x, bag.pivot_y, bag.alpha,
            ))
        bags = rescaled
        print(f"{cfg['catalog_category']}: {len(found_pairs)} found heights, median {px_per_cm:.3f} px/cm")
    else:
        print(f"{cfg['catalog_category']}: fewer than 2 found heights; all ratios stay 1.00 before k")
    return bags


def assign_rooms(bags, cfg, catalog):
    zones = cfg["zones"]
    if cfg["keep_rooms"]:
        rooms = {zid: [] for zid in zones}
        for bag in bags:
            rooms[bag.old_zone].append(bag)
        for zid in rooms:
            rooms[zid] = sorted(rooms[zid], key=lambda bag: (
                cfg["kind_order"].index(bag.category) if bag.category in cfg["kind_order"] else 99,
                bag.height, bag.id))
        return rooms
    ordered = []
    for kind in cfg["kind_order"]:
        group = [bag for bag in bags if bag.category == kind]
        group.sort(key=lambda bag: (bag.height, bag.id))
        ordered.extend(group)
    leftover = [bag for bag in bags if bag.category not in cfg["kind_order"]]
    leftover.sort(key=lambda bag: (bag.height, bag.id))
    ordered.extend(leftover)
    if cfg.get("capacities"):
        caps = [cfg["capacities"][zid] for zid in zones]
    else:
        caps = [sum(1 for item in catalog["items"] if item["zone"] == zid) for zid in zones]
    if sum(caps) != len(ordered):
        raise ValueError("Room capacities do not cover every item")
    rooms, i = {}, 0
    for zid, n in zip(zones, caps):
        rooms[zid] = ordered[i:i + n]
        i += n
    return rooms


def validate_catalog(before, after, active, cfg):
    original = {item["id"]: item for item in before["items"]}
    items = {item["id"]: item for item in after["items"]}
    if list(original) != list(items) or before["redirects"] != after["redirects"]:
        raise ValueError("Item IDs or redirects changed")
    cat = cfg["catalog_category"]
    if Counter(item["category"] for item in items.values())[cat] != cfg["expected_n"]:
        raise ValueError(f"Expected {cfg['expected_n']} {cat}")
    zones = {zone["id"]: zone for zone in after["zones"]}
    if [zid for zid in cfg["zones"] if zid in zones] != list(cfg["zones"]):
        raise ValueError("Expected every configured room")
    membership = []
    for zid in cfg["zones"]:
        zone = zones[zid]
        if zone["letter"] != zid.split("-")[-1] or f"{cat}/{zone['letter']}" in after["redirects"]:
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
    if Counter(membership) != Counter(iid for iid, item in items.items() if item["category"] == cat):
        raise ValueError("Every item must occur in exactly one room")
    for iid, item in items.items():
        old = original[iid]
        moved = item["category"] == cat and item["zone"] in active
        mutable = {"zone", "layer", "hotspot"} if moved else set()
        if any(item[key] != old[key] for key in old.keys() - mutable):
            raise ValueError(f"{iid}: protected item fields changed")
        if moved and (item["layer"]["image"] != old["layer"]["image"]
                      or item["hotspot"].get("clip") != old["hotspot"].get("clip")):
            raise ValueError(f"{iid}: layer image or hotspot clip changed")
    for key, target in after["redirects"].items():
        if key.startswith(f"{cat}/") and target.removeprefix("#/") not in {
            f"{cat}/{zones[zid]['letter']}" for zid in cfg["zones"]
        }:
            raise ValueError(f"{key}: unresolved redirect {target}")


def write_reports(cfg, rooms, plans, selected):
    prefix = cfg["catalog_category"]
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
            grouping.append(f"  Rod {i} (y={fit.tops[i - 1]}, {len(row)} items)")
            members = []
            for p in row:
                bag = p.item.bag
                category = dict(id=bag.id, brand=bag.brand, model=bag.model, category=bag.category,
                                old_zone=bag.old_zone, new_zone=zid, height_cm=bag.height_cm,
                                height_source="spec" if bag.found else "current",
                                found="yes" if bag.found else "no", ratio=bag.ratio,
                                raw_ratio=round(RAW_RATIO.get(bag.id, bag.ratio), 3))
                records.append(category)
                members.append({**category, "rect_pct": p.rect})
                grouping.append(f"    {bag.id}  {bag.category}  {bag.height_cm or 0:g} cm  "
                                f"{bag.brand} {bag.model}  ({category['height_source']}, ×{bag.ratio:.3f})")
            summary[zid]["rows"].append(members)
        grouping.append("")
    if cfg["judgments"]:
        grouping.append("Category decisions")
        grouping.extend(f"  {iid}  {note}" for iid, note in cfg["judgments"].items())
    with (OUT / f"{prefix}-preview-categories.csv").open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / f"{prefix}-preview-grouping.txt").write_text("\n".join(grouping) + "\n", encoding="utf-8")
    (OUT / f"{prefix}-preview-not-found.txt").write_text(
        f"{prefix} using current display size (未查到 or no published height)\n"
        + "".join(f"{r['new_zone']}  {r['id']}  {r['brand']} {r['model']}\n"
                  for r in records if r["found"] == "no"), encoding="utf-8")
    dump(OUT / f"{prefix}-preview-plan.json", {"baseline": ft.BASELINE, "rail_source": cfg["rail_source"],
                                               "rendered_zones": selected, "rooms": summary})
    dump(OUT / f"{prefix}-preview-factors.json", factors)


def preview(catalog, plans, plates, cfg):
    out = OUT / "preview"
    out.mkdir(parents=True, exist_ok=True)
    # Temporarily point selected zones at the preview plates.
    projected = deepcopy(catalog)
    apply_plan(projected, plans)
    for zid, plate in plates.items():
        path = out / f"{zid}-plate.png"
        plate.save(path)
        zone = next(zone for zone in projected["zones"] if zone["id"] == zid)
        preview_zone = {**zone, "images": [{"src": str(path), "width": W, "height": H}]}
        image, _ = composite({"zones": [preview_zone], "items": projected["items"]}, zid)
        image.save(out / f"{zid}-after.jpg", quality=95)
        comparison(zid, image, plans[zid][0], out / f"{zid}-before-after.jpg")


def write(cfg, plans, plates):
    root_fd = os.open(ROOT, os.O_RDONLY)
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        current = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
        target_zones = {p.item.bag.id: zid for zid, (_, fit, _) in plans.items()
                        for row in fit.rows for p in row}
        if any((item["zone"] in plates) != (target_zones[item["id"]] in plates)
               for item in current["items"] if item["id"] in target_zones):
            raise ValueError("The write must include both rooms for each remaining item move")
        updated = deepcopy(current)
        selected_plans = {zid: plans[zid] for zid in plates}
        apply_plan(updated, selected_plans)
        validate_catalog(current, updated, plates, cfg)
        with TemporaryDirectory(prefix=".fill-hanging-", dir=ROOT) as tmp:
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
        validate_catalog(current, written, plates, cfg)
        validate_paths(written)
        (OUT / "after").mkdir(exist_ok=True)
        for zid in plates:
            image, _ = composite(written, zid)
            image.save(OUT / "after" / f"{zid}.jpg", quality=95)
            comparison(zid, image, plans[zid][0], OUT / f"{zid}-before-after.jpg")
    finally:
        os.close(root_fd)


def render_for(cfg, rooms, plans, selected):
    # Temporarily set RAIL_SOURCE used by fill_totes.render_plates
    previous = ft.RAIL_SOURCE
    try:
        ft.RAIL_SOURCE = cfg["rail_source"]
        return render_plates(rooms, plans, selected)
    finally:
        ft.RAIL_SOURCE = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--category", required=True, choices=CONFIGS)
    parser.add_argument("--zones", default="", help="comma-separated rooms; default is all in the category")
    parser.add_argument("--keep-rooms", action="store_true", help="do not reassign items across rooms")
    args = parser.parse_args()
    cfg = dict(CONFIGS[args.category])
    if args.keep_rooms:
        cfg["keep_rooms"] = True
    selected = tuple(z.strip() for z in args.zones.split(",")) if args.zones else cfg["zones"]
    if set(selected) - set(cfg["zones"]) or len(selected) != len(set(selected)):
        parser.error("--zones must be rooms in this category")
    # Plan from the live catalog so later fills see earlier writes; plates come from 88d42ef.
    current = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
    assignment = assign_rooms(load_items(current, cfg), cfg, current)
    rooms = {zone["id"]: load_room(zone) for zone in current["zones"] if zone["id"] in selected}
    plans = {zid: plan_room(assignment[zid], rooms[zid]) for zid in selected}
    projected = deepcopy(current)
    apply_plan(projected, plans)
    validate_catalog(current, projected, selected, cfg)
    validate_paths(projected)
    plates = render_for(cfg, rooms, plans, selected)
    if args.dry_run:
        preview(projected, plans, plates, cfg)
    else:
        write(cfg, plans, plates)
    write_reports(cfg, rooms, plans, selected)
    for zid, (factor, fit, _) in plans.items():
        counts = " / ".join(str(len(row)) for row in fit.rows)
        kinds = ",".join(f"{n} {c}" for c, n in Counter(p.item.bag.category for row in fit.rows for p in row).items())
        print(f"{zid}  ×{factor:.2f}  {counts}  ({kinds})")


if __name__ == "__main__":
    main()
