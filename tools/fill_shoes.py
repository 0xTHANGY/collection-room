#!/usr/bin/env python3
"""Option-B fill for shoe rooms: spec scale, one k per room, regroup by kind.

Shoes sit on the existing rug-row baselines or on detected rack/stool
surfaces. Plates are not redrawn. After a write, check_shoe_supports must pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True

from fill_totes import CATALOG, OUT, REPORT, ROOT, W, H, dump, publish, read_rows, stage_cache, validate_paths
from rebuild_covers import composite
from repack_three_rails import save_webp
from check_shoe_supports import plate_rgb, rug_top_y, furniture, alpha_bottom, load_catalog

ZONES = tuple(f"shoes-{c}" for c in "abcde")
KIND_ORDER = (
    "sneakers", "shoes", "slip-ons", "loafers", "flats", "heels",
    "sandals", "slides", "mules", "ankle boots", "boots",
)
EXPECTED_N = 61
MIN_GAP = 28
X_INSET = 36
FLOOR_BOX = 1010  # typical alpha/box bottom on the lowest rug row
TOL_ABOVE, TOL_BELOW = 6, 32
RATIO_MIN, RATIO_MAX = 0.85, 1.0


def classify(item):
    typ = (item.get("type") or "shoes").lower()
    return typ if typ in KIND_ORDER else "shoes"


def load_specs():
    return read_rows(REPORT / "fill-all/specs.csv", "id")


class Shoe:
    def __init__(self, item, spec, found, height_cm, ratio, width, height, alpha):
        self.id = item["id"]
        self.brand = spec["brand"] or item.get("brand") or ""
        self.model = spec["model"] or item.get("name") or ""
        self.category = classify(item)
        self.old_zone = item["zone"]
        self.found = found
        self.height_cm = height_cm
        self.ratio = ratio
        self.raw_ratio = ratio
        self.width = width * ratio
        self.height = height * ratio
        self.alpha = alpha  # (ax0, ay0, ax1, ay1) relative
        self.item = item

    @property
    def vis_w(self):
        ax0, _, ax1, _ = self.alpha
        return max(1.0, (ax1 - ax0) * self.width)

    @property
    def vis_h(self):
        _, ay0, _, ay1 = self.alpha
        return max(1.0, (ay1 - ay0) * self.height)

    def box_at(self, left, top, k):
        return {
            "left": left / W * 100,
            "top": top / H * 100,
            "width": self.width * k / W * 100,
            "height": self.height * k / H * 100,
        }


def alpha_bounds(path):
    with Image.open(path) as image:
        alpha = image.convert("RGBA").getchannel("A")
        box = alpha.point(lambda p: 255 if p > 24 else 0).getbbox()
        if not box:
            return (0, 0, 1, 1)
        x0, y0, x1, y1 = box
        return (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)


def load_shoes(catalog):
    specs = load_specs()
    shoes, found_pairs = [], []
    for item in catalog["items"]:
        if item["category"] != "shoes":
            continue
        iid = item["id"]
        if iid not in specs:
            raise ValueError(f"{iid}: missing specs.csv row")
        spec = specs[iid]
        found = spec["confidence"] != "未查到" and bool(spec.get("height_cm"))
        layer = item["layer"]
        width = layer["width"] / 100 * W
        height = layer["height"] / 100 * H
        height_cm = float(spec["height_cm"]) if found else 0.0
        bounds = alpha_bounds(ROOT / layer["image"])
        shoe = Shoe(item, spec, found, height_cm, 1.0, width, height, bounds)
        shoes.append(shoe)
        if found:
            found_pairs.append((height_cm, height))
    if len(shoes) != EXPECTED_N:
        raise ValueError(f"expected {EXPECTED_N} shoes, got {len(shoes)}")
    if len(found_pairs) >= 2:
        px_per_cm = float(np.median([px / cm for cm, px in found_pairs]))
        print(f"shoes: {len(found_pairs)} found heights, median {px_per_cm:.3f} px/cm")
        for shoe in shoes:
            if not shoe.found:
                continue
            target = shoe.height_cm * px_per_cm
            shoe.ratio = target / (shoe.height / shoe.ratio) if shoe.height else 1.0
            # Rug rows have a fixed ~210 px pitch, so a full real-cm ratio would
            # shrink a whole room for one tall boot. Clamp the spec ratio and
            # record the raw value for the report.
            shoe.raw_ratio = shoe.ratio
            shoe.ratio = min(max(shoe.ratio, RATIO_MIN), RATIO_MAX)
            shoe.width *= shoe.ratio
            shoe.height *= shoe.ratio
    else:
        print("shoes: fewer than 2 found heights; all ratios stay 1.00 before k")
    return shoes


def is_furniture(item):
    # Box bottom above ~45% of the plate = rack/stool, not a rug row.
    return item["layer"]["top"] + item["layer"]["height"] < 45.0


def assign_rooms(shoes, catalog):
    """Regroup rug shoes by kind across shoes-a..e. Furniture shoes stay put (QA fix)."""
    rug = [s for s in shoes if not is_furniture(s.item)]
    ordered = []
    for kind in KIND_ORDER:
        group = [s for s in rug if s.category == kind]
        group.sort(key=lambda s: (s.height, s.id))
        ordered.extend(group)
    caps = [sum(1 for item in catalog["items"] if item["zone"] == zid and not is_furniture(item))
            for zid in ZONES]
    if sum(caps) != len(ordered):
        raise ValueError("shoe room capacities do not cover every rug shoe")
    rooms, i = {}, 0
    for zid, n in zip(ZONES, caps):
        rooms[zid] = ordered[i:i + n]
        i += n
    fixed = {zid: [s for s in shoes if is_furniture(s.item) and s.old_zone == zid] for zid in ZONES}
    return rooms, fixed


def room_geometry(catalog, zid, zones):
    rgb = plate_rgb(zones[zid])
    rug_y = rug_top_y(rgb)
    shelves, posts, _ = furniture(rgb, rug_y)
    # Rug-row baselines from the current (already-correct) layout.
    bottoms = []
    for item in catalog["items"]:
        if item["zone"] != zid:
            continue
        L = item["layer"]
        top = L["top"] / 100 * H
        if top < 0.28 * H:
            continue
        bottoms.append(round((L["top"] / 100 * H) + (L["height"] / 100 * H)))
    row_bottoms = sorted(set(bottoms))
    if len(row_bottoms) != 3:
        # fall back to even splits between rug and the typical floor line
        lo, hi = rug_y + 40, FLOOR_BOX
        row_bottoms = [round(lo + (hi - lo) * (i + 1) / 3) for i in range(3)]
    # usable x at each row: rug span with inset; approximate from plate width
    from repack_shoes import detect_rug
    _, span_at = detect_rug(rgb)
    span = (X_INSET, W - X_INSET)
    return {
        "span_at": span_at,
        "rug_y": rug_y,
        "shelves": shelves,  # (y, x0, x1)
        "posts": posts,
        "row_bottoms": row_bottoms,
        "span": span,
        "rgb": rgb,
    }


def pack_row(items, k, bottom, span):
    """Left-to-right pack, bottoms aligned at `bottom`. Returns xs or None."""
    left, right = span
    widths = [s.width * k for s in items]
    # visible left/right insets
    lefts = [s.alpha[0] * s.width * k for s in items]
    rights = [s.alpha[2] * s.width * k for s in items]
    # first item's box left so visible left >= span left
    pos = []
    x = left - items[0].alpha[0] * items[0].width * k if items else left
    x = max(0, x)
    for i, s in enumerate(items):
        if i:
            prev = items[i - 1]
            # body gap between visible edges
            x = pos[-1] + prev.alpha[2] * prev.width * k + MIN_GAP - s.alpha[0] * s.width * k
        if x < 0 or x + s.width * k > W:
            return None
        if x + s.alpha[2] * s.width * k > right + 2:
            return None
        pos.append(x)
    # distribute leftover slack
    last_vis = pos[-1] + items[-1].alpha[2] * items[-1].width * k
    slack = right - last_vis
    if slack < 0:
        return None
    if slack and len(items) > 1:
        for i in range(len(items)):
            pos[i] += slack * (i + 1) / (len(items) + 1)
    return pos


def place_room(shoes, geom, k, max_imb=3):
    """Place shoes at scale k. Smallest items go to furniture slots first."""
    shelves = []  # furniture shoes are fixed; see assign_rooms
    row_bottoms = geom["row_bottoms"]
    span = geom["span"]
    ordered = sorted(shoes, key=lambda s: (s.height * k, s.width * k, s.id))
    furniture_places = []
    remaining = list(ordered)
    for y, x0, x1 in shelves:
        slot_w = x1 - x0
        # pick the smallest remaining shoe that fits the slot at this k
        chosen = None
        for i, s in enumerate(remaining):
            w, h = s.width * k, s.height * k
            vis_w = (s.alpha[2] - s.alpha[0]) * w
            if vis_w > slot_w + 8:
                continue
            # sit the alpha bottom on the board (a few px of sole may go below)
            # box top so alpha bottom ≈ y + 12 (within TOL_BELOW)
            ay0, ay1 = s.alpha[1], s.alpha[3]
            # we want top + ay1*h ≈ y + 12
            top = (y + 12) - ay1 * h
            if top < 8:
                continue
            if top + ay0 * h < 4:
                continue
            chosen = (i, s, (x0 + x1) / 2 - w / 2, top)
            break
        if chosen is None:
            return None
        i, s, left, top = chosen
        # clamp left into the shelf
        left = min(max(left, x0 - s.alpha[0] * s.width * k), x1 - s.alpha[2] * s.width * k)
        remaining.pop(i)
        furniture_places.append((s, left, top, "shelf", y))

    # remaining on 3 rug rows, smallest on the top row
    remaining.sort(key=lambda s: (s.height * k, s.width * k, s.id))
    # try splits that keep rows from being empty and prefer even counts
    n = len(remaining)
    if n < 3:
        return None
    best = None
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            # Tallest third goes on the back row, which has the most headroom
            # (its top may rise over the rug's back edge); then descending.
            rows = [remaining[b:], remaining[a:b], remaining[:a]]
            if any(len(r) == 0 for r in rows):
                continue
            # skip very unbalanced rows
            if max(map(len, rows)) - min(map(len, rows)) > max_imb:
                continue
            placed_rows = []
            ok = True
            for ri, (row, bottom) in enumerate(zip(rows, row_bottoms)):
                # Box tops must clear the row behind (or the back edge of the rug).
                limit_top = (geom["rug_y"] - 40) if ri == 0 else row_bottoms[ri - 1] + 4
                if any(bottom - s.height * k < limit_top for s in row):
                    ok = False
                    break
                typ_h = max(s.height * k for s in row)
                x0, x1 = geom["span_at"](bottom - typ_h * 0.35)
                xs = pack_row(row, k, bottom, (x0, x1))
                if xs is None:
                    ok = False
                    break
                # back row must not touch fixed furniture shoes
                for s_, x_ in zip(row, xs):
                    t_ = bottom - s_.height * k
                    for fx0, fy0, fx1, fy1 in geom.get("fixed_boxes", ()):
                        if min(fx1, x_ + s_.width * k) - max(fx0, x_) > 0 and min(fy1, bottom) - max(fy0, t_) > 0:
                            ok = False
                if not ok:
                    break
                placed_rows.append([(s, x, bottom - s.height * k, "row", bottom) for s, x in zip(row, xs)])
            if not ok:
                continue
            rank = (max(map(len, rows)) - min(map(len, rows)), -a)
            if best is None or rank < best[0]:
                best = (rank, furniture_places + [p for row in placed_rows for p in row])
    if best is None:
        return None
    return best[1]


def best_k(shoes, geom):
    """Largest k (hundredths) per imbalance level; prefer even rows when k >= 1."""
    results = []
    for imb in (1, 2, 3):
        def fits(c, imb=imb):
            return place_room(shoes, geom, c / 100, imb) is not None
        if not fits(50):
            continue
        low, high = 50, 300
        while high - low > 1:
            mid = (low + high) // 2
            if fits(mid):
                low = mid
            else:
                high = mid
        results.append((imb, low))
        if low >= 100:
            break
    if not results:
        return None
    imb, low = results[-1] if results[-1][1] >= 100 or len(results) == 1 else max(results, key=lambda r: r[1])
    return low / 100, place_room(shoes, geom, low / 100, imb)


def apply_plan(catalog, plans):
    items = {item["id"]: item for item in catalog["items"]}
    zones = {z["id"]: z for z in catalog["zones"]}
    for zid, (factor, placed) in plans.items():
        zones[zid]["items"] = [s.id for s, *_ in placed]
        for z_index, (s, left, top, kind, _) in enumerate(placed, 1):
            item = items[s.id]
            item["zone"] = zid
            if kind == "fixed":
                item["layer"]["zIndex"] = z_index
                continue
            box = s.box_at(left, top, factor)
            item["layer"].update(box, rect=box, zIndex=z_index)
            item["hotspot"].update(box)


def validate(before, after, active):
    original = {i["id"]: i for i in before["items"]}
    items = {i["id"]: i for i in after["items"]}
    if list(original) != list(items) or before["redirects"] != after["redirects"]:
        raise ValueError("Item IDs or redirects changed")
    if Counter(i["category"] for i in items.values())["shoes"] != EXPECTED_N:
        raise ValueError("expected 61 shoes")
    membership = []
    zones = {z["id"]: z for z in after["zones"]}
    for zid in ZONES:
        zone = zones[zid]
        membership.extend(zone["items"])
        if any(items[iid]["zone"] != zid for iid in zone["items"]):
            raise ValueError(f"{zid}: membership mismatch")
        if zid not in active:
            continue
        boxes = []
        for iid in zone["items"]:
            item = items[iid]
            box = item["layer"]["rect"]
            if any(item["layer"][k] != v or item["hotspot"][k] != v for k, v in box.items()):
                raise ValueError(f"{iid}: layer/hotspot mismatch")
            x, y, w, h = (box[k] for k in ("left", "top", "width", "height"))
            if not (0 <= x < x + w <= 100 and 0 <= y < y + h <= 100):
                raise ValueError(f"{iid}: box outside the room")
            for ax, ay, bx, by in boxes:
                if min(bx, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9:
                    raise ValueError(f"{iid}: overlapping hotspots")
            boxes.append((x, y, x + w, y + h))
    if Counter(membership) != Counter(i["id"] for i in items.values() if i["category"] == "shoes"):
        raise ValueError("every shoe must occur in exactly one room")
    for iid, item in items.items():
        old = original[iid]
        moved = item["category"] == "shoes" and item["zone"] in active
        mutable = {"zone", "layer", "hotspot"} if moved else set()
        if any(item[k] != old[k] for k in old.keys() - mutable):
            raise ValueError(f"{iid}: protected fields changed")


def write_reports(rooms, geoms, plans):
    records, grouping, factors = [], [], {}
    for zid, (factor, placed) in plans.items():
        counts = Counter(s.category for s, *_ in placed)
        factors[zid] = {"factor": factor, "category_counts": dict(counts),
                        "shelves": geoms[zid]["shelves"], "row_bottoms": geoms[zid]["row_bottoms"]}
        grouping.append(f"{zid} — ×{factor:.2f} — " + ", ".join(f"{n} {c}" for c, n in counts.items()))
        for s, left, top, kind, support in placed:
            rec = dict(id=s.id, brand=s.brand, model=s.model, category=s.category,
                       old_zone=s.old_zone, new_zone=zid, height_cm=s.height_cm,
                       found="yes" if s.found else "no", ratio=s.ratio, raw_ratio=s.raw_ratio, support=kind,
                       support_y=support, box_left=left, box_top=top)
            records.append(rec)
            grouping.append(f"    {s.id}  {s.category}  {s.height_cm or 0:g} cm  "
                            f"{kind}@{support}  ({'spec' if s.found else 'current'}, ×{s.ratio:.3f})")
        grouping.append("")
    with (OUT / "shoes-preview-categories.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    (OUT / "shoes-preview-grouping.txt").write_text("\n".join(grouping) + "\n")
    dump(OUT / "shoes-preview-factors.json", factors)


def comparison(zid, image, factor, path):
    from fill_totes import comparison as cmp
    cmp(zid, image, factor, path)


def preview(catalog, plans):
    out = OUT / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for zid, (factor, _) in plans.items():
        image, _ = composite(catalog, zid)
        image.save(out / f"{zid}-after.jpg", quality=95)
        comparison(zid, image, factor, out / f"{zid}-before-after.jpg")


def write(plans):
    root_fd = os.open(ROOT, os.O_RDONLY)
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        current = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
        updated = deepcopy(current)
        apply_plan(updated, plans)
        validate(current, updated, set(plans))
        with TemporaryDirectory(prefix=".fill-shoes-", dir=ROOT) as tmp:
            stage = Path(tmp)
            paths = []
            for zid in plans:
                zone = next(z for z in updated["zones"] if z["id"] == zid)
                image, _ = composite(updated, zid)
                cover = Path(f"assets/zones/{zid}-cover-640.webp")
                height = round(640 * image.height / image.width)
                (stage / cover.parent).mkdir(parents=True, exist_ok=True)
                save_webp(image.resize((640, height), Image.Resampling.LANCZOS), stage / cover)
                zone["cover"] = {"src": str(cover), "width": 640, "height": height}
                paths.append(cover)
            cache_paths = stage_cache(stage)
            validate_paths(updated, stage)
            (stage / CATALOG.parent).mkdir()
            dump(stage / CATALOG, updated)
            publish(stage, [*paths, CATALOG, *cache_paths])
        written = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
        validate(current, written, set(plans))
        (OUT / "after").mkdir(exist_ok=True)
        for zid, (factor, _) in plans.items():
            image, _ = composite(written, zid)
            image.save(OUT / "after" / f"{zid}.jpg", quality=95)
            comparison(zid, image, factor, OUT / f"{zid}-before-after.jpg")
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--zones", default=",".join(ZONES))
    args = parser.parse_args()
    selected = tuple(z.strip() for z in args.zones.split(","))
    current = json.loads((ROOT / CATALOG).read_text(encoding="utf-8"))
    shoes = load_shoes(current)
    assignment, fixed = assign_rooms(shoes, current)
    zones = {z["id"]: z for z in current["zones"]}
    geoms = {zid: room_geometry(current, zid, zones) for zid in selected}
    plans = {}
    for zid in selected:
        geoms[zid]["fixed_boxes"] = [
            (f.item["layer"]["left"] / 100 * W, f.item["layer"]["top"] / 100 * H,
             (f.item["layer"]["left"] + f.item["layer"]["width"]) / 100 * W,
             (f.item["layer"]["top"] + f.item["layer"]["height"]) / 100 * H)
            for f in fixed[zid]]
        result = best_k(assignment[zid], geoms[zid])
        if result is None:
            raise SystemExit(f"{zid}: no feasible layout")
        factor, placed = result
        for f in fixed[zid]:
            L = f.item["layer"]
            placed = [(f, L["left"] / 100 * W, L["top"] / 100 * H, "fixed", "furniture")] + placed
        plans[zid] = (factor, placed)
        print(f"{zid}  ×{result[0]:.2f}  n={len(assignment[zid])}  "
              f"shelves={len(geoms[zid]['shelves'])}  rows={geoms[zid]['row_bottoms']}")
    projected = deepcopy(current)
    apply_plan(projected, plans)
    validate(current, projected, set(plans))
    write_reports(assignment, geoms, plans)
    if args.dry_run:
        preview(projected, plans)
    else:
        write(plans)


if __name__ == "__main__":
    main()
