#!/usr/bin/env python3
"""Enlarge compact minis and small pouches in bags-a..d so they read next to the
big bags, without changing the room fill factor or reordering items.

Rule (recorded in fill-all.md):
  Scope = bags in the `minis` category, plus small pouches (pouch/sacoche with
  body height_cm <= 12; currently only batch2-362).
  Shared display scale S is chosen so the *tallest compact* item in scope
  (the ones currently below the visual floor of the already-legible chain-minis)
  sits 8 px shorter than the shortest unboosted bag in the same room, and
  relative body-cm order among the boosted items is preserved.
  Items already at or above their target size stay at factor 1.00.
  Room k, rod Ys, and all non-scope bags are unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parent / "collection-room-impl-report" / "fill-all" / "bags-minis"
W, H = 1536, 1024
EDGE, BODY_GAP = 24, 28
# From bags_fill.py for bags-a (the only affected room today).
RAIL_SPAN = {"bags-a": (104, 1434), "bags-b": (171, 1427), "bags-c": (106, 1429), "bags-d": (136, 1398)}
MINIS = {"today-427", "today-401", "today-399", "today-425", "today-412", "today-391"}
MARGIN = 8  # px shorter than the shortest unboosted bag in the room


def heights():
    spec = {r["id"]: r for r in csv.DictReader(
        (ROOT.parent / "collection-room-impl-report" / "bags-spec" / "bags-spec.csv").open(encoding="utf-8-sig"))}
    old = {r["item_id"]: r for r in csv.DictReader(
        (ROOT.parent / "collection-room-impl-report" / "sizegroup" / "bags.csv").open())}
    out = {}
    for iid, row in old.items():
        s = spec.get(iid, {})
        found = s.get("confidence") not in (None, "", "未查到") and s.get("height_cm")
        out[iid] = float(s["height_cm"]) if found else float(row["height_cm"])
    return out


def is_scope(item, cm):
    typ = (item.get("type") or "").lower()
    name = (item.get("name") or "").lower()
    if item["id"] in MINIS:
        return True
    if cm <= 12 and ("pouch" in typ or "sacoche" in typ or "pouch" in name or "sacoche" in name):
        return True
    if item["id"] == "batch2-362":
        return True
    return False


def box_h(item):
    return item["layer"]["height"] / 100 * H


def scale_around_hook(item, factor):
    layer = item["layer"]
    hook_x = layer["left"] + layer["width"] * layer["pivotX"] / 100
    hook_y = layer["top"] + layer["height"] * layer["pivotY"] / 100
    layer["width"] *= factor
    layer["height"] *= factor
    layer["left"] = hook_x - layer["width"] * layer["pivotX"] / 100
    layer["top"] = hook_y - layer["height"] * layer["pivotY"] / 100
    box = {k: layer[k] for k in ("left", "top", "width", "height")}
    layer["rect"] = dict(box)
    item["hotspot"].update(box)


def pack_row(items, span):
    """Keep hook Y, redistribute X with EDGE/BODY_GAP. Order (current left) is kept."""
    items = sorted(items, key=lambda it: it["layer"]["left"])
    left, right = span[0] + EDGE, span[1] - EDGE
    widths = []
    for it in items:
        w = it["layer"]["width"] / 100 * W
        widths.append(w)
    total = sum(widths) + BODY_GAP * (len(items) - 1)
    # If they don't fit, this caller should have chosen a smaller boost.
    if total > right - left + 1e-6:
        raise ValueError(f"row overflow by {total - (right - left):.1f}px")
    slack = (right - left) - total
    gap = BODY_GAP
    extra = slack / (len(items) + 1)  # share leftover on both sides and between
    x = left + extra
    for it, w in zip(items, widths):
        px = it["layer"]["pivotX"] / 100
        # x is the box left
        new_left = x / W * 100
        # shift so we place by box left (same as bags_fill even packing)
        hook_y = it["layer"]["top"] + it["layer"]["height"] * it["layer"]["pivotY"] / 100
        it["layer"]["left"] = new_left
        it["layer"]["top"] = hook_y - it["layer"]["height"] * it["layer"]["pivotY"] / 100
        box = {k: it["layer"][k] for k in ("left", "top", "width", "height")}
        it["layer"]["rect"] = dict(box)
        it["hotspot"].update(box)
        x += w + gap + extra


def main():
    path = ROOT / "data/catalog.json"
    catalog = json.loads(path.read_text())
    cm = heights()
    bags = [it for it in catalog["items"] if it["category"] == "bags"]
    # Plan per room
    records = []
    affected_zones = set()
    for zid in ("bags-a", "bags-b", "bags-c", "bags-d"):
        room = [it for it in bags if it["zone"] == zid]
        scope = [it for it in room if is_scope(it, cm[it["id"]])]
        others = [it for it in room if it not in scope]
        if not scope:
            continue
        # Shortest unboosted display height in this room
        if not others:
            cap = 200
        else:
            cap = min(box_h(it) for it in others) - MARGIN
        # Compact = currently shorter than cap (the ones that actually need a lift)
        compact = [it for it in scope if box_h(it) < cap]
        if not compact:
            for it in scope:
                records.append(dict(id=it["id"], zone=zid, height_cm=cm[it["id"]],
                                    before_px=round(box_h(it), 1), after_px=round(box_h(it), 1),
                                    factor=1.0, note="already at or above the cap; no boost"))
            continue
        # Shared px/cm so the tallest compact body-cm reaches `cap`, smaller ones scale in proportion.
        max_cm = max(cm[it["id"]] for it in compact)
        px_per_cm = cap / max_cm
        factors = {}
        for it in scope:
            target = px_per_cm * cm[it["id"]]
            before = box_h(it)
            f = max(1.0, target / before) if it in compact else 1.0
            # never shrink; cap only applies to a boost
            if f > 1 and before * f > cap + 1e-6:
                f = max(1.0, cap / before)
            factors[it["id"]] = f
        # Apply, then repack each rod that changed
        rods = {}
        for it in room:
            rods.setdefault(round(it["layer"]["top"], 2), []).append(it)
        for it in scope:
            f = factors[it["id"]]
            before = box_h(it)
            if f > 1.0001:
                scale_around_hook(it, f)
                affected_zones.add(zid)
            records.append(dict(id=it["id"], zone=zid, height_cm=cm[it["id"]],
                                before_px=round(before, 1), after_px=round(box_h(it), 1),
                                factor=round(f, 3),
                                note="boosted" if f > 1.0001 else "already at or above target; factor 1.00"))
        if zid in affected_zones:
            for row in rods.values():
                pack_row(row, RAIL_SPAN[zid])

    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "factors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "zone", "height_cm", "before_px", "after_px", "factor", "note"])
        w.writeheader(); w.writerows(records)
    (REPORT / "rule.json").write_text(json.dumps({
        "rule": "minis + small pouches (pouch/sacoche, height_cm<=12) share a px/cm so the tallest compact item is 8px shorter than the shortest unboosted bag in the room; relative cm order is kept; items already larger stay at x1.00",
        "margin_px": MARGIN,
        "records": records,
        "affected_zones": sorted(affected_zones),
    }, indent=2) + "\n")
    for r in records:
        print(f"{r['id']:14} {r['zone']:8} {r['height_cm']:5.1f}cm  {r['before_px']:6.1f} -> {r['after_px']:6.1f} px  x{r['factor']:.3f}  {r['note']}")
    print("affected", sorted(affected_zones))


if __name__ == "__main__":
    main()
