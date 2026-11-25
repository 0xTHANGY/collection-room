#!/usr/bin/env python3
"""Per-bag sourced rescaling + clustering, cabinet clearance caps, rod-spacing report.

Local layout only: no generation, no artwork redraw, hooks stay on existing rods.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sizegroup_layout as sg  # noqa: E402

REPORT = Path("/workspace/collection-room-impl-report/sizegroup")
RESEARCH = (ROOT / "tools/bags-research.json") if (ROOT / "tools/bags-research.json").exists() else (REPORT / "bags-research.json")
CABINET_MARGIN_PCT = 0.80  # ~8 px on the 1024-tall canvas
# Confirmed furniture tops (not rails) from background row analysis + 1440 inspection.
CABINET_TOP_PCT = {
    "jackets-b": 78.515625,  # long low cabinet
    "bottoms-b": 84.1796875,  # split benches
}


def load_all():
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    research = {r["item_id"]: r for r in json.loads(RESEARCH.read_text())}
    items_by_id = {it["id"]: it for it in catalog["items"]}
    return catalog, research, items_by_id


def rod_span(recs):
    lefts, rights = [], []
    for r in recs:
        L = r["item"]["layer"]
        lefts.append(L["left"])
        rights.append(L["left"] + L["width"])
    xmin = max(sg.WALL_MIN, min(lefts) if lefts else sg.WALL_MIN)
    xmax = min(sg.WALL_MAX, max(rights) if rights else sg.WALL_MAX)
    return xmin, xmax


def scale_and_cluster_bags(catalog, research):
    rails = sg.load_rails(catalog)
    zone_by_id = {z["id"]: z for z in catalog["zones"]}
    bags = []
    for it in catalog["items"]:
        if it["id"] in research:
            bags.append(it)

    by_zone = defaultdict(list)
    records = []
    for it in bags:
        L = it["layer"]
        meta = research[it["id"]]
        hx, hy, _, _ = sg.hook_of(L)
        zrails = rails.get(it["zone"]) or []
        rod_idx, rail = sg.assign_rod(hy, zrails, "hang")
        if zrails:
            hy = rail["hook_y"] / sg.CANVAS_H * 100.0
        floor = sg.detect_floor_pct(zone_by_id[it["zone"]])
        # If this room has a confirmed cabinet, use it as the bottom limit.
        if it["zone"] in CABINET_TOP_PCT:
            floor = min(floor, CABINET_TOP_PCT[it["zone"]])
        clear = sg.clearance_px(rod_idx, zrails, floor, "hang")
        frac = sg.visual_frac(L)
        h_cm = float(meta["height_cm"])
        old_css = sg.css_h_px(L)
        max_ppc = (clear * frac / h_cm) if h_cm > 0 and frac > 0 else None
        rec = {
            "item": it,
            "meta": meta,
            "rod_idx": rod_idx,
            "hook_x": hx,
            "hook_y": hy,
            "frac": frac,
            "clear_px": clear,
            "old_css_px": old_css,
            "old_visual_px": old_css * frac,
            "max_ppc": max_ppc,
            "h_cm": h_cm,
        }
        records.append(rec)
        by_zone[it["zone"]].append(rec)

    zone_ppc = {}
    for zid, recs in by_zone.items():
        ppcs = [r["max_ppc"] for r in recs if r["max_ppc"] and r["max_ppc"] > 0]
        zone_ppc[zid] = min(ppcs) if ppcs else None

    # Scale around hooks.
    for r in records:
        it = r["item"]
        L = it["layer"]
        ppc = zone_ppc[it["zone"]]
        if not ppc:
            r["scale"] = 1.0
            r["new_css_px"] = r["old_css_px"]
            continue
        target_css = r["h_cm"] * ppc / max(r["frac"], 1e-6)
        # never exceed clearance
        target_css = min(target_css, r["clear_px"])
        scale = target_css / r["old_css_px"] if r["old_css_px"] else 1.0
        new_w = L["width"] * scale
        new_h = L["height"] * scale
        sg.set_box_keep_anchor(L, new_w, new_h, "hang", r["hook_x"], r["hook_y"])
        sg.sync_hotspot(it)
        r["scale"] = scale
        r["ppc"] = ppc
        r["new_css_px"] = sg.css_h_px(L)
        r["new_visual_px"] = r["new_css_px"] * r["frac"]

    # Cluster each rod by displayed opaque height, long → short left to right.
    for zid, recs in by_zone.items():
        xmin, xmax = rod_span(recs)
        rods = defaultdict(list)
        for r in recs:
            rods[r["rod_idx"]].append(r)
        for idx, rod in rods.items():
            rod.sort(key=lambda r: -sg.visual_h_px(r["item"]["layer"]))
            sg.pack_rod(rod, xmin, xmax)
            for r in rod:
                r["new_css_px"] = sg.css_h_px(r["item"]["layer"])
                r["new_visual_px"] = r["new_css_px"] * r["frac"]
                r["scale"] = r["new_css_px"] / r["old_css_px"] if r["old_css_px"] else 1.0

    sg.resolve_overlaps(records, by_zone)
    for r in records:
        r["new_css_px"] = sg.css_h_px(r["item"]["layer"])
        r["new_visual_px"] = r["new_css_px"] * r["frac"]
        r["scale"] = r["new_css_px"] / r["old_css_px"] if r["old_css_px"] else 1.0
        r["moved_rod"] = bool(r.get("moved_rod"))

    return records, zone_ppc, by_zone


def cap_furniture_items(catalog, rails):
    """Minimal scale cap so opaque hems stay above a confirmed cabinet/bench."""
    items_by_zone = defaultdict(list)
    for it in catalog["items"]:
        items_by_zone[it["zone"]].append(it)
    adjustments = []
    for zid, cabinet in CABINET_TOP_PCT.items():
        limit = cabinet - CABINET_MARGIN_PCT
        zrails = rails.get(zid) or []
        for it in items_by_zone.get(zid, []):
            L = it["layer"]
            if sg.is_shelf_rest(L):
                continue
            meta = sg.layer_meta(L["image"])
            frac_bot = (meta["opaque_bot"] + 1) / meta["image_h"]
            opaque_bot = L["top"] + L["height"] * frac_bot
            if opaque_bot <= limit:
                continue
            hx, hy, _, _ = sg.hook_of(L)
            old_h = L["height"]
            old_hang = opaque_bot - hy
            need_hang = max(4.0, limit - hy)
            if old_hang <= 0:
                continue
            s = need_hang / old_hang
            # cap only; do not enlarge
            if s >= 0.999:
                continue
            sg.set_box_keep_anchor(L, L["width"] * s, L["height"] * s, "hang", hx, hy)
            sg.sync_hotspot(it)
            adjustments.append({
                "item_id": it["id"],
                "room": zid,
                "name": it.get("name"),
                "cabinet_top_pct": cabinet,
                "old_opaque_bot_pct": opaque_bot,
                "new_opaque_bot_pct": hy + (L["top"] + L["height"] * frac_bot - hy) if False else (L["top"] + L["height"] * frac_bot),
                "old_css_h_pct": old_h,
                "new_css_h_pct": L["height"],
                "scale_cap": s,
                "pct_cut": (1.0 - s) * 100.0,
                "reason": (
                    f"Opaque hem at {opaque_bot:.2f}% overlapped {zid} furniture top "
                    f"{cabinet:.2f}%. Capped scale by {((1-s)*100):.1f}% so the hem stays "
                    f"{CABINET_MARGIN_PCT:.2f}% above the furniture (limit {limit:.2f}%). "
                    "Rods are baked into the background and could not be moved up enough "
                    "without colliding with the row above."
                ),
            })
        # re-pack rods in this zone after caps (same sizes otherwise, tighter items)
        # Keep existing x order by displayed height after cap.
        recs = []
        for it in items_by_zone[zid]:
            L = it["layer"]
            hx, hy, _, _ = sg.hook_of(L)
            rod_idx, rail = sg.assign_rod(hy, zrails, "hang")
            if zrails:
                hy = rail["hook_y"] / sg.CANVAS_H * 100.0
            recs.append({"item": it, "rod_idx": rod_idx, "hook_y": hy})
        xmin, xmax = rod_span(recs)
        rods = defaultdict(list)
        for r in recs:
            rods[r["rod_idx"]].append(r)
        for idx, rod in rods.items():
            rod.sort(key=lambda r: -sg.visual_h_px(r["item"]["layer"]))
            sg.pack_rod(rod, xmin, xmax)
    return adjustments


def write_bags_csv(records, path):
    fields = [
        "item_id", "room", "brand", "model", "width_cm", "height_cm", "depth_cm",
        "source_url", "confidence", "old_px", "new_px", "ratio", "note",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["item"]["zone"], x["item"]["id"])):
            it, m = r["item"], r["meta"]
            w.writerow({
                "item_id": it["id"],
                "room": it["zone"],
                "brand": m["brand"],
                "model": m["model"],
                "width_cm": m["width_cm"] if m["width_cm"] is not None else "",
                "height_cm": m["height_cm"] if m["height_cm"] is not None else "",
                "depth_cm": m["depth_cm"] if m["depth_cm"] is not None else "",
                "source_url": m["source_url"],
                "confidence": m["confidence"],
                "old_px": round(r["old_css_px"], 2),
                "new_px": round(r["new_css_px"], 2),
                "ratio": round(r["scale"], 4),
                "note": m.get("note") or "",
            })


def write_rod_spacing(catalog, path):
    """Report old vs new rod Y. Rods are baked into backgrounds; none were moved."""
    rails = sg.load_rails(catalog)
    meta = json.loads((REPORT / "layout-meta.json").read_text())
    floors = meta.get("zone_floor_pct") or {}
    fields = [
        "room", "rod_index", "old_hook_y_px", "new_hook_y_px",
        "old_hook_y_pct", "new_hook_y_pct", "old_gap_below_px", "new_gap_below_px",
        "action", "reason",
    ]
    reason = (
        "Rods are painted into the 1536x1024 zone plates (wood rail + wall mounts + "
        "soft shadow). A non-generative strip move would cut a horizontal band, slide it, "
        "and fill the gap with cloned wall pixels. At 1440 those fills are not seamless: "
        "walls have a vertical window-light gradient and the rails have end-caps/mounts, "
        "so a clone fill would show a seam. Per the brief, rods were left in place. "
        "Hooks and items stay on the existing rails. Vertical gaps were already even "
        "in almost every hanging room (equal spacing between the three rails)."
    )
    rows = []
    for z in catalog["zones"]:
        zid = z["id"]
        zrails = rails.get(zid) or []
        hooks = [r["hook_y"] for r in zrails]
        for i, hy in enumerate(hooks):
            gap = (hooks[i + 1] - hy) if i + 1 < len(hooks) else None
            rows.append({
                "room": zid,
                "rod_index": i,
                "old_hook_y_px": round(hy, 2),
                "new_hook_y_px": round(hy, 2),
                "old_hook_y_pct": round(hy / sg.CANVAS_H * 100, 3),
                "new_hook_y_pct": round(hy / sg.CANVAS_H * 100, 3),
                "old_gap_below_px": "" if gap is None else round(gap, 2),
                "new_gap_below_px": "" if gap is None else round(gap, 2),
                "action": "unchanged",
                "reason": reason,
            })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows, floors


def main():
    catalog, research, _ = load_all()
    print(f"research bags: {len(research)}")
    records, zone_ppc, by_zone = scale_and_cluster_bags(catalog, research)
    rails = sg.load_rails(catalog)
    caps = cap_furniture_items(catalog, rails)

    # write catalog
    out_cat = ROOT / "data/catalog.json"
    out_cat.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print("wrote catalog")

    write_bags_csv(records, REPORT / "bags.csv")
    print("wrote bags.csv", len(records))
    rod_rows, floors = write_rod_spacing(catalog, REPORT / "rod-spacing.csv")
    print("wrote rod-spacing.csv", len(rod_rows))
    (REPORT / "cabinet-caps.json").write_text(json.dumps(caps, indent=2, ensure_ascii=False) + "\n")
    print("cabinet caps:", json.dumps(caps, indent=2, ensure_ascii=False)[:2000])

    print("\nPer-room bag px/cm:")
    for zid, ppc in sorted(zone_ppc.items()):
        recs = by_zone[zid]
        ratios = [r["scale"] for r in recs]
        print(f"  {zid:12} ppc={ppc:.3f} n={len(recs)} scale {min(ratios):.2f}–{max(ratios):.2f}  "
              f"model={sum(1 for r in recs if r['meta']['confidence']=='model')} "
              f"type-est={sum(1 for r in recs if r['meta']['confidence']=='type-estimate')}")

    moved = [r for r in records if r.get("moved_rod")]
    print("bags moved between rods:", len(moved), [r["item"]["id"] for r in moved])


if __name__ == "__main__":
    main()
