#!/usr/bin/env python3
"""Recover 14 shoe pairs from their source paintings using local GrabCut.

Run with --write to install the layers, update the shoe placements and covers,
and write the change summary. With --proof-dir, also render labelled JPEGs
outside the checkout. Geometry and old layer sizes come from commit 7fc1bc7,
so a repeated run never uses its own output as input.
"""

import argparse
import io
import json
import math
import subprocess
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from repack_shoes import clean_ghost, detect_rug
from repack_three_rails import clip_polygon, pct, save_webp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source"
BASELINE = "7fc1bc7"
IDS = tuple(f"shoes-n{i:03}" for i in (1, 2, 6, 14, 15, 18, 25, 26, 35, 44, 49, 50, 51, 56))
ZONES = tuple(f"shoes-{letter}" for letter in "abcde")
WIDTH, HEIGHT = 1536, 1024
GAP = 16
SHELF_ITEM = "shoes-n056"
# Bbox-relative outlines repair missing toes and sidewalls in the prepared masks.
# GrabCut refines the band around each outline using the original painting.
OUTLINES = {
    "shoes-n006": (
        [(31, 58), (41, 28), (57, 15), (82, 13), (103, 23), (115, 43),
         (118, 77), (124, 97), (125, 134), (130, 171), (132, 215),
         (127, 244), (115, 262), (95, 270), (69, 270), (38, 260),
         (17, 243), (4, 218), (0, 195), (5, 155), (12, 119), (18, 97)],
        [(150, 72), (157, 38), (172, 18), (194, 13), (215, 17), (231, 31),
         (239, 52), (244, 81), (256, 98), (264, 126), (268, 149),
         (275, 184), (273, 214), (260, 238), (240, 256), (207, 269),
         (174, 270), (155, 260), (146, 243), (142, 211), (143, 172),
         (148, 134), (145, 101)],
    ),
    "shoes-n018": (
        [(13, 103), (17, 77), (27, 53), (30, 34), (38, 21), (57, 14),
         (78, 13), (89, 21), (96, 44), (105, 65), (110, 92), (111, 144),
         (118, 177), (118, 207), (112, 233), (102, 250), (86, 260),
         (62, 261), (43, 255), (29, 241), (18, 222), (12, 198), (10, 175),
         (10, 141), (1, 135), (13, 124)],
        [(131, 94), (133, 63), (137, 41), (143, 23), (154, 15), (177, 13),
         (196, 19), (207, 33), (213, 53), (222, 73), (227, 93),
         (231, 131), (249, 151), (263, 169), (255, 166), (239, 150),
         (244, 175), (246, 199), (241, 220), (231, 243), (218, 256),
         (201, 261), (174, 260), (155, 251), (143, 233), (137, 209),
         (133, 183), (135, 152)],
    ),
    "shoes-n025": (
        [(19, 97), (29, 54), (43, 24), (62, 8), (85, 6), (110, 13),
         (131, 36), (145, 68), (158, 105), (159, 135), (154, 161),
         (151, 189), (152, 231), (156, 269), (154, 310), (145, 343),
         (130, 369), (108, 385), (86, 391), (64, 388), (43, 375),
         (27, 354), (17, 321), (12, 290), (13, 255), (18, 215),
         (17, 188), (5, 192), (2, 181), (16, 167), (17, 138)],
        [(171, 107), (171, 76), (177, 48), (188, 26), (208, 9), (231, 7),
         (253, 13), (274, 32), (292, 62), (305, 100), (311, 130),
         (307, 164), (307, 174), (321, 188), (324, 204), (317, 213),
         (309, 211), (316, 240), (320, 281), (318, 320), (310, 352),
         (292, 375), (270, 388), (245, 391), (223, 385), (203, 368),
         (190, 343), (180, 309), (176, 275), (178, 229), (182, 204),
         (170, 217), (160, 209), (163, 194), (176, 178), (172, 156)],
    ),
    "shoes-n026": (
        [(17, 102), (24, 67), (36, 33), (49, 14), (68, 6), (90, 5),
         (111, 12), (129, 29), (141, 53), (149, 82), (153, 119),
         (152, 159), (153, 203), (154, 243), (155, 278), (149, 318),
         (138, 351), (119, 376), (96, 391), (67, 395), (47, 388),
         (30, 370), (17, 342), (9, 310), (5, 282), (5, 247),
         (8, 207), (9, 163)],
        [(163, 111), (169, 69), (179, 37), (194, 16), (215, 8),
         (238, 8), (257, 19), (271, 36), (284, 62), (293, 99),
         (297, 136), (299, 167), (305, 199), (312, 228), (319, 267),
         (320, 300), (311, 338), (297, 366), (278, 388), (257, 398),
         (233, 397), (208, 386), (186, 364), (172, 335), (163, 301),
         (159, 266), (159, 219), (157, 181), (159, 145)],
    ),
}


def ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def baseline_file(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)


def pair_layers():
    paintings = SOURCE / "newshoes/zones"
    pairs = {}
    for path in sorted(paintings.glob("shoes-*.webp")):
        layout = json.loads((SOURCE / "layout" / f"{path.stem}.json").read_text())
        for iid in IDS:
            if iid in layout["items"]:
                assert iid not in pairs, iid
                rgb = np.array(Image.open(path).convert("RGB"))
                pairs[iid] = cut_pair(iid, rgb, layout["items"][iid]["bbox"])
    assert pairs.keys() == set(IDS)
    return pairs


def cut_pair(iid, rgb, bbox):
    x0, y0, x1, y1 = bbox
    pad = max(12, round(max(x1 - x0, y1 - y0) * .12))
    left, top = max(0, x0 - pad), max(0, y0 - pad)
    roi = np.s_[top:y1 + pad, left:x1 + pad]
    rgb = rgb[roi]
    allowed = np.array(Image.open(SOURCE / "newshoes/masks" / f"{iid}.png").convert("L"))[roi] > 127
    distance = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5)
    seeds = np.full(allowed.shape, cv2.GC_BGD, np.uint8)
    seeds[allowed] = cv2.GC_PR_BGD
    seeds[distance > 12] = cv2.GC_PR_FGD
    seeds[distance > 24] = cv2.GC_FGD
    if iid in OUTLINES:
        outline = np.zeros(allowed.shape, np.uint8)
        for points in OUTLINES[iid]:
            points = np.array(points, np.int32) + (x0 - left, y0 - top)
            cv2.fillPoly(outline, [points.astype(np.int32)], 1)
        margin = cv2.dilate(outline, ellipse(8)) > 0
        allowed |= margin
        seeds[:] = cv2.GC_BGD
        seeds[margin] = cv2.GC_PR_BGD
        seeds[outline > 0] = cv2.GC_PR_FGD
        seeds[cv2.erode(outline, ellipse(5)) > 0] = cv2.GC_FGD
    cv2.setRNGSeed(42)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), seeds, None,
                np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    fg = np.uint8((seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    areas = stats[1:, cv2.CC_STAT_AREA]
    mask = np.isin(labels, 1 + np.flatnonzero(areas >= .20 * areas.max())) & allowed
    _, holes, stats, _ = cv2.connectedComponentsWithStats(np.uint8(~mask), connectivity=4)
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if x > 0 and y > 0 and x + w < mask.shape[1] and y + h < mask.shape[0] and area <= 250:
            mask |= (holes == label) & allowed

    alpha = cv2.GaussianBlur(np.uint8(mask) * 255, (3, 3), .55)
    ys, xs = np.nonzero(alpha)
    crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(clean_ghost(np.dstack((rgb, alpha))[crop]))


def pixel_rect(rect):
    return tuple(round(value) for value in exact_pixel_rect(rect))


def exact_pixel_rect(rect):
    return tuple(rect[key] * size / 100 for key, size in
                 zip(("left", "top", "width", "height"), (WIDTH, HEIGHT, WIDTH, HEIGHT)))


def bottom(rect):
    return rect[1] + rect[3]


def right(rect):
    return rect[0] + rect[2]


def anchored(rect, width, height):
    return round((rect[0] + right(rect) - width) / 2), bottom(rect) - height, width, height


def place_pairs(members, plate, old, new, before, sizes, old_layers):
    """Expand about the frozen anchors, then space only rows with tight gaps."""
    rug_top, span_at = detect_rug(np.array(plate))
    fixes = [iid for iid in members if iid in sizes]
    factors = {iid: 1.0 for iid in fixes}

    def resize(iid, factor):
        w, h = sizes[iid]
        ow, oh = old_layers[iid].size
        rect = old[iid]
        new[iid] = anchored(rect, round(w * rect[2] / ow * factor),
                            round(h * rect[3] / oh * factor))
        factors[iid] = factor

    for iid in fixes:
        resize(iid, 1)
    if SHELF_ITEM in fixes:
        # Leave clearance under the rod above the hand-seated shelf in shoes-e.
        width, height = new[SHELF_ITEM][2:]
        usable_width = old[SHELF_ITEM][2] + 24
        usable_height = bottom(old[SHELF_ITEM]) - 108
        resize(SHELF_ITEM, min(1, usable_width / width, usable_height / height))

    # Keep taller replacement layers clear of the preceding row at the same anchors.
    for iid in sorted(fixes, key=lambda key: bottom(old[key])):
        if bottom(old[iid]) < rug_top:
            continue
        rect = new[iid]
        for key in members:
            if bottom(old[key]) >= bottom(old[iid]) - 40:
                continue
            obstacle = new[key]
            if key not in fixes:
                spot = exact_pixel_rect(before[key]["hotspot"])
                if min(right(spot), right(rect)) > max(spot[0], rect[0]):
                    obstacle = spot if bottom(spot) > bottom(obstacle) else obstacle
            if min(right(obstacle), right(rect)) <= max(obstacle[0], rect[0]):
                continue
            available = bottom(rect) - math.ceil(bottom(obstacle))
            if available < rect[3]:
                resize(iid, factors[iid] * available / rect[3])
                rect = new[iid]

    handled = set()
    for iid in fixes:
        if bottom(old[iid]) < rug_top or iid in handled:
            continue
        row = sorted(
            (key for key in members if bottom(old[key]) >= rug_top
             and abs(bottom(old[key]) - bottom(old[iid])) <= 40),
            key=lambda key: old[key][0],
        )
        handled.update(row)
        if all(new[b][0] - right(new[a]) >= GAP for a, b in zip(row, row[1:])):
            continue
        lo, hi = min(old[key][0] for key in row), max(right(old[key]) for key in row)
        required = sum(new[key][2] for key in row) + GAP * (len(row) - 1)
        if required > hi - lo:
            rug_lo, rug_hi = span_at(bottom(old[iid]) - max(new[key][3] for key in row) * .35)
            lo, hi = min(lo, rug_lo), max(hi, rug_hi)
        while required > hi - lo:
            # Reduce only repaired layers, a pixel at a time to preserve the minimum scale.
            adjustable = [key for key in row if key in factors
                          and new[key][2] - 1 >= .42 * sizes[key][0]
                          and round((new[key][2] - 1) / sizes[key][0] * sizes[key][1]) >= .42 * sizes[key][1]]
            if not adjustable:
                raise ValueError(f"Row cannot fit at minimum scale: {row}")
            key = max(adjustable, key=lambda key: new[key][2])
            factor = (new[key][2] - 1) * old_layers[key].width / (sizes[key][0] * old[key][2])
            resize(key, factor)
            required -= 1
        gap = (hi - lo - sum(new[key][2] for key in row)) / (len(row) - 1)
        x = lo
        for key in row:
            _, y, w, h = new[key]
            new[key] = round(x), y, w, h
            x += w + gap
    for iid in factors:
        if factors[iid] < 1 and min(a / b for a, b in zip(new[iid][2:], sizes[iid])) < .42:
            raise ValueError(f"{iid} needs a placement below the minimum scale")
    return factors


def placed_layer(image, rect):
    return image.resize(rect[2:], Image.Resampling.LANCZOS)


def pair_clip(alpha):
    # clip_polygon traces one contour. A hull lets its single CSS polygon
    # reach both shoes when the placed alpha has two substantial components.
    foreground = np.uint8(alpha > 40)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(map(cv2.contourArea, contours))
        main = [contour for contour in contours if cv2.contourArea(contour) >= .20 * largest]
        if len(main) > 1:
            alpha = np.zeros_like(alpha)
            cv2.fillConvexPoly(alpha, cv2.convexHull(np.concatenate(main)), 255)
    return clip_polygon(alpha)


def composite(plate, members, rects, layers):
    canvas = plate.convert("RGBA")
    for iid in members:
        rect = rects[iid]
        canvas.alpha_composite(placed_layer(layers[iid], rect), rect[:2])
    return canvas.convert("RGB")


def validate(items, members):
    for iid in members:
        x, y, w, h = exact_pixel_rect(items[iid]["layer"])
        if not (0 <= x < x + w <= WIDTH and 0 <= y < y + h <= HEIGHT):
            raise ValueError(f"Layer outside zone: {iid}")
    for a, b in combinations(members, 2):
        ra, rb = (items[iid]["hotspot"] for iid in (a, b))
        overlap = [min(ra[p] + ra[d], rb[p] + rb[d]) - max(ra[p], rb[p])
                   for p, d in (("left", "width"), ("top", "height"))]
        if min(overlap) > 1e-8:
            raise ValueError(f"Hotspot overlap: {a}, {b} ({overlap})")


def write_proofs(directory, rooms, old_layers, layers, old, new, items):
    directory.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 248, 290
    sheet = Image.new("RGB", (4 * tile_w, 7 * tile_h), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    for index, iid in enumerate(IDS):
        plate, _, _ = rooms[items[iid]["zone"]]
        rug = plate.crop((600, HEIGHT - tile_h, 600 + tile_w, HEIGHT))
        # Use the same magnification on both sides to make the recovered shoe visible.
        scale = min(1, (tile_w - 12) / max(old[iid][2], new[iid][2]),
                    (tile_h - 38) / max(old[iid][3], new[iid][3]))
        for offset, (label, image, rect) in enumerate(
            (("before", old_layers[iid], old[iid]), ("after", layers[iid], new[iid]))
        ):
            tile = rug.convert("RGBA")
            shoe = image.resize(tuple(round(value * scale) for value in rect[2:]),
                                Image.Resampling.LANCZOS)
            tile.alpha_composite(shoe, ((tile_w - shoe.width) // 2, tile_h - 35 - shoe.height))
            x, y = (index % 2 * 2 + offset) * tile_w, index // 2 * tile_h
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y + tile_h - 26, x + tile_w - 1, y + tile_h - 1), fill="#ffffff")
            draw.text((x + 8, y + tile_h - 21), f"{iid}  {label}", fill="#1a1a1a")
    sheet.save(directory / "pairs-before-after.jpg", quality=92)

    for zid, (_, image, members) in rooms.items():
        labelled = image.copy()
        draw = ImageDraw.Draw(labelled)
        for iid in members:
            x, y, w, h = new[iid]
            label_y = min(y + h + 2, HEIGHT - 13)
            draw.text((x + w / 2, label_y), iid, anchor="mt", fill="white",
                      stroke_width=2, stroke_fill="#201a15")
        labelled.save(directory / f"zone-{zid}-after.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--proof-dir", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    catalog_path = ROOT / "data/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    original = json.loads(baseline_file("data/catalog.json"))
    items = {item["id"]: item for item in catalog["items"] if item["zone"] in ZONES}
    before = {item["id"]: item for item in original["items"] if item["zone"] in ZONES}
    if {iid: item["zone"] for iid, item in items.items()} != {
        iid: item["zone"] for iid, item in before.items()
    }:
        raise ValueError("The shoe zone membership differs from the placement baseline")

    old = {iid: pixel_rect(item["layer"]) for iid, item in before.items()}
    new = old.copy()
    old_layers = {iid: Image.open(io.BytesIO(baseline_file(items[iid]["layer"]["image"]))).convert("RGBA")
                  for iid in IDS}
    cutouts = pair_layers()
    sizes = {iid: image.size for iid, image in cutouts.items()}
    layers = {iid: Image.open(ROOT / item["layer"]["image"]).convert("RGBA")
              for iid, item in items.items() if iid not in cutouts}
    encoded = {}
    for iid, cutout in cutouts.items():
        stream = io.BytesIO()
        cutout.save(stream, "WEBP", quality=82, method=4)
        encoded[iid] = stream.getvalue()
        layers[iid] = Image.open(io.BytesIO(encoded[iid])).convert("RGBA")

    rooms, factors = {}, {}
    for zone in catalog["zones"]:
        zid = zone["id"]
        if zid not in ZONES:
            continue
        src = max(zone["images"], key=lambda image: image["width"])["src"]
        plate = Image.open(ROOT / src).convert("RGB")
        assert plate.size == (WIDTH, HEIGHT), (zid, plate.size)
        members = sorted(zone["items"], key=lambda iid: items[iid]["layer"].get("zIndex", 0))
        factors.update(place_pairs(members, plate, old, new, before, sizes, old_layers))
        for iid in members:
            if iid not in cutouts and new[iid] == old[iid]:
                continue
            box = pct(new[iid], WIDTH, HEIGHT)
            alpha = np.array(placed_layer(layers[iid], new[iid]).getchannel("A"))
            items[iid]["layer"].update(box, rect=box.copy())
            items[iid]["hotspot"] = dict(box, clip=pair_clip(alpha))
        if zid == "shoes-c":
            # The baseline padded hotspots for n029 and n027 already overlap.
            # Trim n029's hotspot to the edge of n027, leaving both layers seated.
            iid = "shoes-n029"
            x, y, w, h = new[iid]
            limit = math.floor(exact_pixel_rect(items["shoes-n027"]["hotspot"])[1])
            box = (x, y, w, min(h, limit - y))
            alpha = np.array(placed_layer(layers[iid], new[iid]).getchannel("A"))[:box[3]]
            items[iid]["hotspot"] = dict(pct(box, WIDTH, HEIGHT), clip=pair_clip(alpha))
        validate(items, members)
        rooms[zid] = plate, composite(plate, members, new, layers), members

    summary = {
        "baseline": BASELINE,
        "rect_units": "zone pixels (left, top, width, height)",
        "pairs": {
            iid: {"zone": items[iid]["zone"], "old_size": old_layers[iid].size, "new_size": sizes[iid],
                  "old_rect": old[iid], "new_rect": new[iid], "scale_factor": factors[iid]}
            for iid in IDS
        },
        "hotspot_only_adjustments": ["shoes-n029"],
        "moved_neighbours": {
            iid: {"zone": items[iid]["zone"], "old_rect": old[iid], "new_rect": new[iid]}
            for iid in items if iid not in cutouts and old[iid] != new[iid]
        },
    }
    if args.write:
        for iid, data in encoded.items():
            path = ROOT / items[iid]["layer"]["image"]
            temp = path.with_suffix(".webp.tmp")
            temp.write_bytes(data)
            temp.replace(path)
        for zid, (_, image, _) in rooms.items():
            cover = image.resize((640, round(640 * HEIGHT / WIDTH)), Image.Resampling.LANCZOS)
            save_webp(cover, ROOT / f"assets/zones/{zid}-cover-640.webp")
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        (SOURCE / "display/shoe-pairs-fix.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.proof_dir:
        write_proofs(args.proof_dir, rooms, old_layers, layers, old, new, items)
    for iid in IDS:
        print(f"{iid}: {old_layers[iid].size} -> {sizes[iid]}, scale factor {factors[iid]:.3f}")
    print(f"{len(rooms)} zones, {len(items)} layers in bounds, no hotspot rectangle overlaps, "
          f"{len(summary['moved_neighbours'])} neighbours moved.")


if __name__ == "__main__":
    main()
