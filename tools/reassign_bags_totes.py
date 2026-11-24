#!/usr/bin/env python3
"""Reassign bags and totes from a membership manifest and re-hang them in display order.

Reuse the existing empty v3 three-rail plates and layer WebPs. Fit each
category at a uniform scale using the v3 row packer, compositor and hotspot
geometry. Run with --write to update the catalog, source membership, images
and summary. Use --proof-dir to export labelled room and card images.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import repack_three_rails as repack

ROOT = repack.ROOT
DISPLAY = repack.DISPLAY
MEMBERSHIP = DISPLAY / "bags-totes-membership.json"
NAME_CLEANUP = DISPLAY / "name-cleanup.json"
SUMMARY = DISPLAY / "bags-totes-reassign.json"
SOURCE_ZONES = ROOT / "data/source/zones.json"
SOURCE_ITEMS = ROOT / "data/source/today-2026-11-18/items.json"
PLATES = {
    "bags": ["bags-a", "bags-b", "bags-c", "bags-d"],
    "totes": ["totes-a"] * 3,
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_plate(zone_id, rail_records):
    path = ROOT / f"assets/zones/{zone_id}-1600.webp"
    plate = np.array(Image.open(path).convert("RGB"))
    bands = repack.detect_rail_bands(plate)
    if len(bands) != len(rail_records):
        raise ValueError(f"{zone_id}: expected three existing rails, found {bands}")
    wall = plate.copy()
    rails = []
    for (top, bottom, left, right), record in zip(bands, rail_records):
        # Exclude rail pixels from floor/furniture detection.
        wall[max(0, top - 2):bottom + 3] = 190
        rails.append({
            "wood_top": top,
            "rail_bottom": record.get("rail_bottom", bottom),
            "hook_y": top + record["hook_y"] - record["wood_top"],
            "x0": record.get("x0", max(left + 24, 200)),
            "x1": record.get("x1", min(right - 24, plate.shape[1] - 200)),
        })
    return path, plate, rails, repack.detect_obstacle_y(wall)


def plan_category(catalog, category):
    recs = sorted(repack.load_layers(catalog, category),
                  key=lambda rec: (int(rec["item"]["number"]), rec["id"]))
    plates = PLATES[category]
    base, extra = divmod(len(recs), len(plates))
    counts = [base + (index < extra) for index in range(len(plates))]
    previous = read_json(DISPLAY / f"{category}-repack.json")
    rail_records = previous["rails"]
    if category == "totes":
        rail_records = {"totes-a": rail_records}
    loaded = {plate: load_plate(plate, rail_records[plate]) for plate in dict.fromkeys(plates)}
    plans = []
    limits = []
    cursor = 0
    for index, (count, source) in enumerate(zip(counts, plates)):
        zone_id = f"{category}-{repack.letters(len(plates))[index]}"
        path, plate, rails, obstacle = loaded[source]
        rows = []
        for row_index, count_on_rail in enumerate(repack.rod_counts(count)):
            row = recs[cursor:cursor + count_on_rail]
            cursor += count_on_rail
            rows.append(row)
            rail = rails[row_index]
            room_bottom = rails[row_index + 1]["wood_top"] if row_index < 2 else obstacle
            limits.append((
                (rail["x1"] - rail["x0"] - repack.MIN_GAP * (len(row) + 1))
                / sum(rec["w"] for rec in row), zone_id, f"rail {row_index + 1} width",
            ))
            for rec in row:
                anchor = rail["rail_bottom"] + 4 if rec["kind"] == "handle" else rail["hook_y"]
                limits.append(((room_bottom - 8 - anchor) / (rec["h"] - rec["ay"]),
                               zone_id, rec["id"]))
        plans.append((zone_id, path, plate, rails, rows))
    limit, limiting_zone, limiting_item = min(limits)
    scale = min(previous["scale"], math.floor(limit * 100) / 100)
    if scale <= 0:
        raise ValueError(f"No usable hanging space for {limiting_item} in {limiting_zone}")
    scaled_for = {"zone": limiting_zone, "constraint": limiting_item} if scale < previous["scale"] else None
    return plans, scale, scaled_for


def pack_category(catalog, category, plans, scale):
    existing = {zone["id"]: zone for zone in catalog["zones"]}
    output = {}
    for zone_id, source, plate, rails, rows in plans:
        placed = []
        for row, rail in zip(rows, rails):
            positions, gap = repack.pack_row(row, rail, scale)
            if gap < repack.MIN_GAP:
                raise ValueError(f"{zone_id}: row gap too small ({gap})")
            placed.extend(positions)
        cover = repack.composite(plate, placed)
        height, width = plate.shape[:2]
        for position in placed:
            item = position["rec"]["item"]
            box = repack.pct((position["x"], position["y"], position["nw"], position["nh"]),
                             width, height)
            item["zone"] = zone_id
            item["layer"].update(
                box, rect=dict(box), fallback=False, motion=position["rec"]["motion"],
                pivot={"x": 50, "y": 0}, pivotX=50, pivotY=0, zIndex=position["zIndex"],
            )
            item["hotspot"] = dict(box, clip=position["clip"])
        stem = f"assets/zones/{zone_id}"
        images = []
        for size in (1600, 1024, 640):
            ratio = min(1, size / width)
            images.append(dict(src=f"{stem}-{size}.webp", width=round(width * ratio),
                               height=round(height * ratio)))
        zone = dict(existing.get(zone_id, existing[plans[0][0]]))
        zone.update(id=zone_id, letter=zone_id.rsplit("-", 1)[1], category=category,
                    images=images, cover=dict(src=f"{stem}-cover-640.webp", width=640,
                                             height=round(height * 640 / width)),
                    items=[position["rec"]["id"] for position in placed],
                    natural_size=[width, height], aspect=width / height, has_layout=True)
        output[zone_id] = (zone, source, plate, cover)
    return output


def write_images(output):
    for zone, source, plate, cover in output.values():
        stem = f"assets/zones/{zone['id']}"
        # The full-size empty plate is the canonical input. Copy its encoded
        # bytes to cloned rooms so reruns cannot accumulate WebP losses.
        destination = ROOT / zone["images"][0]["src"]
        temporary = destination.with_suffix(".webp.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        repack.variants_from(Image.fromarray(plate), stem, (1024, 640))
        cover_info = zone["cover"]
        repack.save_webp(cover.resize((cover_info["width"], cover_info["height"]),
                                     Image.Resampling.LANCZOS), ROOT / cover_info["src"])


def validate(catalog, membership, expected_ids):
    items = {item["id"]: item for item in catalog["items"]}
    zones = {zone["id"]: zone for zone in catalog["zones"]}
    if Counter(item["id"] for item in catalog["items"]) != expected_ids:
        raise ValueError("Catalog item IDs were lost or duplicated")
    if any(item["zone"] not in zones for item in items.values()):
        raise ValueError("An item references a missing zone")
    routes = {f"#/{zone['category']}/{zone['letter']}" for zone in zones.values()}
    routes.update(f"#/item/{item_id}" for item_id in items)
    for route, target in catalog["redirects"].items():
        if target not in routes:
            raise ValueError(f"Dangling redirect {route}: {target}")
    for category, ids in membership.items():
        actual = {item["id"] for item in items.values() if item["category"] == category}
        zone_items = []
        for zone in zones.values():
            if zone["category"] != category:
                continue
            zone_items.extend(zone["items"])
            members = [items[item_id] for item_id in zone["items"]]
            if any(item["zone"] != zone["id"] for item in members):
                raise ValueError(f"Catalog membership differs in {zone['id']}")
            numbers = [int(item["number"]) for item in members]
            if numbers != sorted(numbers):
                raise ValueError(f"Display numbers out of order in {zone['id']}")
            boxes = []
            for item in members:
                box = item["hotspot"]
                x, y = box["left"], box["top"]
                right, bottom = x + box["width"], y + box["height"]
                if not (0 <= x < right <= 100 and 0 <= y < bottom <= 100
                        and all(0 <= coord <= 100 for point in box["clip"] for coord in point)):
                    raise ValueError(f"Hotspot outside bounds for {item['id']}")
                boxes.append((x, y, right, bottom))
            # Disjoint bounding rectangles also guarantee disjoint clip polygons.
            for a, b in combinations(boxes, 2):
                if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                    raise ValueError(f"Hotspot overlap in {zone['id']}: {a}, {b}")
        if actual != set(ids) or Counter(zone_items) != Counter(ids):
            raise ValueError(f"Incorrect or duplicated membership in {category}")


def write_proofs(directory, catalog):
    directory.mkdir(parents=True, exist_ok=True)
    items = {item["id"]: item for item in catalog["items"]}
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    heading = ImageFont.truetype("DejaVuSans.ttf", 24)
    cell_w, cell_h, columns = 220, 198, 7
    for category in PLATES:
        zones = [zone for zone in catalog["zones"] if zone["category"] == category]
        height = sum(42 + cell_h * math.ceil(len(zone["items"]) / columns) for zone in zones)
        sheet = Image.new("RGB", (cell_w * columns, height), "#ede9e2")
        sheet_draw = ImageDraw.Draw(sheet)
        offset = 0
        for zone in zones:
            members = [items[item_id] for item_id in zone["items"]]
            base = Image.open(ROOT / zone["images"][0]["src"]).convert("RGBA")
            width, height = base.size
            labels = []
            for item in sorted(members, key=lambda item: item["layer"]["zIndex"]):
                layer = item["layer"]
                x, y, w, h = (round(layer[key] * dim / 100) for key, dim in
                              zip(("left", "top", "width", "height"), (width, height, width, height)))
                sprite = Image.open(ROOT / layer["image"]).convert("RGBA")
                base.alpha_composite(sprite.resize((w, h), Image.Resampling.LANCZOS), (x, y))
                labels.append((x + w // 2, y + h - 5, item["id"]))
            draw = ImageDraw.Draw(base)
            for x, y, label in labels:
                bounds = draw.textbbox((x, y), label, font=font, anchor="mt")
                draw.rectangle((bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3),
                               fill="#202020")
                draw.text((x, y), label, font=font, anchor="mt", fill="white")
            base.convert("RGB").save(directory / f"zone-{zone['id']}.jpg", quality=94)
            sheet_draw.text((10, offset + 6), zone["id"], font=heading, fill="#292421")
            offset += 42
            for index, item in enumerate(members):
                col, row = index % columns, index // columns
                x, y = col * cell_w, offset + row * cell_h
                image = Image.open(ROOT / item["card_images"][0]["src"]).convert("RGBA")
                image.thumbnail((cell_w - 16, 108))
                sheet.paste(image, (x + (cell_w - image.width) // 2, y), image)
                sheet_draw.text((x + 8, y + 110), item["id"], font=font, fill="#292421")
                name = "\n".join(textwrap.wrap(item["display_name"], width=23))
                sheet_draw.multiline_text((x + 8, y + 128), name, font=font, fill="#292421", spacing=0)
            offset += cell_h * math.ceil(len(members) / columns)
        sheet.save(directory / f"{category}-room.jpg", quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--proof-dir", type=Path, help="output directory for JPEG proofs with --write")
    args = parser.parse_args()
    if args.proof_dir and not args.write:
        parser.error("--proof-dir requires --write")

    catalog = read_json(repack.CATALOG)
    membership = read_json(args.membership)
    removed = membership.pop("removed_duplicates", {})
    original_ids = Counter(item["id"] for item in catalog["items"])
    current = {item["id"]: item["category"] for item in catalog["items"]
               if item["category"] in PLATES}
    target = {item_id: category for category, ids in membership.items() for item_id in ids}
    if (set(membership) != set(PLATES) or set(target) != current.keys() - removed.keys()
            or sum(map(len, membership.values())) != len(target)
            or any(count != 1 for count in original_ids.values())):
        raise ValueError("Membership must partition the original bags and totes IDs exactly once")
    if (removed.keys() & (original_ids.keys() - current.keys())
            or set(removed.values()) - target.keys()):
        raise ValueError("Duplicate removals must point from removed bags or totes to retained items")

    names = read_json(NAME_CLEANUP)
    source_items = read_json(SOURCE_ITEMS)
    source_by_id = {item["id"]: item for item in source_items["items"]}
    if names.keys() - target.keys() or names.keys() - source_by_id.keys():
        raise ValueError("Name cleanup IDs must be retained bags or totes with source records")
    for item_id, name in names.items():
        source_by_id[item_id]["name"] = name

    before = {zone["id"]: len(zone["items"]) for zone in catalog["zones"]
              if zone["category"] in PLATES}
    moves = {item_id: {"from": current[item_id], "to": category}
             for item_id, category in target.items() if current[item_id] != category}
    catalog["items"] = [item for item in catalog["items"] if item["id"] not in removed]
    for item in catalog["items"]:
        item_id = item["id"]
        if item_id in target:
            item["category"] = target[item_id]
        if item_id in names:
            item.update(name=names[item_id], display_name=names[item_id])

    source_zones = read_json(SOURCE_ZONES)
    assets, scales, limits = {}, {}, {}
    for category in PLATES:
        plans, scales[category], limits[category] = plan_category(catalog, category)
        output = pack_category(catalog, category, plans, scales[category])
        assets.update(output)
        first_zone = next(iter(output))
        zones = []
        for zone in catalog["zones"]:
            if zone["id"] == first_zone:
                zones.extend(entry[0] for entry in output.values())
            elif zone["category"] != category:
                zones.append(zone)
        catalog["zones"] = zones
        source_zones[category] = {zone["letter"]: zone["items"] for zone, *_ in output.values()}
        category_entry = next(entry for entry in catalog["categories"] if entry["id"] == category)
        category_entry.update(count=len(membership[category]), cover_zone=first_zone)

    old_redirects = catalog["redirects"].copy()
    for letter in "efghij":
        for key in (f"bags/{letter}", f"bags/bags-{letter}"):
            catalog["redirects"][key] = "#/bags/d"
    for key in ("totes/b", "totes/totes-b"):
        catalog["redirects"].pop(key, None)
    for duplicate, retained in removed.items():
        catalog["redirects"][f"item/{duplicate}"] = f"#/item/{retained}"
    after = {zone_id: len(zone["items"]) for zone_id, (zone, *_) in assets.items()}
    summary = dict(before=before, after=after, moved_ids=moves, removed_duplicates=removed, scale=scales,
                   scale_constraints=limits, deleted_files=[], redirects_changed={
                       key: {"from": old_redirects.get(key), "to": catalog["redirects"].get(key)}
                       for key in sorted(old_redirects.keys() | catalog["redirects"].keys())
                       if old_redirects.get(key) != catalog["redirects"].get(key)
                   })
    validate(catalog, membership, original_ids - Counter(removed.keys()))
    if args.write:
        write_images(assets)
        if catalog != read_json(repack.CATALOG) or not SUMMARY.exists():
            write_json(SUMMARY, summary)
        write_json(repack.CATALOG, catalog)
        write_json(SOURCE_ZONES, source_zones)
        write_json(SOURCE_ITEMS, source_items)
        if args.proof_dir:
            write_proofs(args.proof_dir, catalog)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PASS: membership, unique IDs, display order, hotspot bounds and separation, zone references, and redirects.")


if __name__ == "__main__":
    main()
