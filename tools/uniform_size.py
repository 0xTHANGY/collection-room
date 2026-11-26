#!/usr/bin/env python3
"""Uniform displayed size per room for hats-a..g and totes-a..c.

Every item's visible *body* bbox (alpha bbox with thin cords/hooks opened away) is normalised to the
same sqrt(width*height) target, then the largest uniform factor that fits the rods is chosen (same
packer as fill_hanging / fill_totes: no overlap, rod contact or floor/bench clash). Rooms keep their
current membership and kind-sorted order. Output goes to ../collection-room-impl-report/uniform-size/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fill_hanging as fh
import fill_totes as ft
import uniform_metrics as um
from fill_hanging import Bag, W, H
from rebuild_covers import composite

REPORT = ft.REPORT
OUTDIR = REPORT / "uniform-size"
WORK = OUTDIR / "work"
CACHE = "20261126c"

CONFIGS = {
    "hats": dict(fh.CONFIGS["hats"], keep_rooms=True, expected_n=98),
    "totes": {
        "zones": ("totes-a", "totes-b", "totes-c"), "rail_source": "totes-a", "catalog_category": "totes",
        "kind_order": ft.KIND_ORDER, "expected_n": 42, "keep_rooms": True, "judgments": {},
    },
}


def stage_cache(stage):
    changed = []
    for name, pattern, count in (
        ("app.js", r"(const ASSET_V = ')2026112[56][a-z](';)", 1),
        ("index.html", r'((?:styles\.css|app\.js)\?v=)2026112[56][a-z](")', 2),
    ):
        previous = (ft.ROOT / name).read_text(encoding="utf-8")
        contents, n = re.subn(pattern, rf"\g<1>{CACHE}\2", previous)
        if n != count:
            raise ValueError(f"{name}: expected {count} cache references")
        if contents != previous:
            (stage / name).write_text(contents, encoding="utf-8")
            changed.append(Path(name))
    return changed


def load_items(catalog, cfg):
    """Bags with ratio = room-median body sqrt-area / own body sqrt-area (absolute vs current layer box)."""
    cat = cfg["catalog_category"]
    by_room, raw = {}, []
    for item in catalog["items"]:
        if item["category"] != cat:
            continue
        layer = item["layer"]
        fw, fhh, bw, bh = um.metrics(item)
        s = float(np.sqrt(bw * bh))
        from_kind = ft.classify(item) if cat == "totes" else fh.classify(item, cfg["kind_order"])
        raw.append((item, s, from_kind))
        by_room.setdefault(item["zone"], []).append(s)
    target = {z: float(np.median(v)) for z, v in by_room.items()}
    bags = []
    for item, s, kind in raw:
        layer = item["layer"]
        ratio = target[item["zone"]] / s
        width, height = layer["width"] * W / 100, layer["height"] * H / 100
        with Image.open(ft.ROOT / layer["image"]) as image:
            alpha = image.convert("RGBA").getchannel("A")
            x0, y0, x1, y1 = alpha.point(lambda p: 255 if p > 24 else 0).getbbox()
            bounds = (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)
        bags.append(Bag(item["id"], item.get("brand") or "", item.get("name") or "", kind, item["zone"],
                        0.0, False, ratio, width * ratio, height * ratio,
                        layer["pivotX"] / 100, layer["pivotY"] / 100, bounds))
        fh.RAW_RATIO[item["id"]] = ratio
    if len(bags) != cfg["expected_n"]:
        raise ValueError("unexpected item count")
    return bags, target


def best_rows(bags, room, rod_count):
    """fill_hanging.best_rows with sparse rods (<=2 items) ranked ahead of kind purity."""
    best = None
    for rows in fh.candidate_rows(bags, room, rod_count):
        if fh.fit_rows(rows, room, 1) is None:
            continue
        low, high = 1, 100
        while fh.fit_rows(rows, room, high) is not None:
            low, high = high, high * 2
            if high > 400:
                break
        while high - low > 1:
            mid = (low + high) // 2
            if fh.fit_rows(rows, room, mid) is None:
                high = mid
            else:
                low = mid
        fit = fh.fit_rows(rows, room, low)
        mixed = sum(len({bag.category for bag in row}) > 1 for row in rows)
        n_short = sum(len(row) <= 3 for row in rows)
        rank = (0 if low >= 100 else 1, n_short, mixed, max(map(len, rows)) - min(map(len, rows)), -low, -fit.slack)
        if best is None or rank < best[0]:
            best = rank, low, fit
    return best


def make_comparison(before_dir, before_label="b09e153", after_label="uniform"):
    def comparison(zid, image, factor, path):
        with Image.open(before_dir / f"{zid}.jpg") as before:
            before = before.resize(image.size, Image.Resampling.LANCZOS)
        w, h = image.size
        label_h = 76
        canvas = Image.new("RGB", (2 * w + 24, h + label_h), "white")
        canvas.paste(before, (0, label_h))
        canvas.paste(image, (w + 24, label_h))
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"BEFORE  {zid}  ({before_label})", font=font, fill="black")
        draw.text((w + 40, 10), f"AFTER  {zid}  {after_label}  ×{factor:.2f}", font=font, fill="black")
        canvas.resize((canvas.width // 2, canvas.height // 2), Image.Resampling.LANCZOS).save(path, quality=93)
    return comparison


def spread_rows(catalog, zones):
    out = {}
    for z in catalog["zones"]:
        if z["id"] in zones:
            out[z["id"]] = um.room_stats(catalog, z)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    ap.add_argument("--zones", default="", help="comma-separated rooms (default: all 10)")
    args = ap.parse_args()
    selected_all = tuple(args.zones.split(",")) if args.zones else CONFIGS["hats"]["zones"] + CONFIGS["totes"]["zones"]
    OUTDIR.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    (WORK / "before").mkdir(exist_ok=True)
    fh.OUT = WORK
    fh.best_rows = best_rows
    fh.stage_cache = stage_cache
    fh.comparison = make_comparison(WORK / "before")
    current = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    for zid in selected_all:                        # frozen "before" composites (b09e153 state)
        p = WORK / "before" / f"{zid}.jpg"
        if not p.exists():
            composite(current, zid)[0].save(p, quality=95)
    summary = {}
    for cat, cfg in CONFIGS.items():
        selected = tuple(z for z in cfg["zones"] if z in selected_all)
        if not selected:
            continue
        bags, target = load_items(current, cfg)
        assignment = fh.assign_rooms(bags, cfg, current)
        loader = ft.load_room if cat == "totes" else fh.load_room
        rooms = {z["id"]: loader(z) for z in current["zones"] if z["id"] in set(selected) | {cfg["rail_source"]}}
        plans = {zid: fh.plan_room(assignment[zid], rooms[zid]) for zid in selected}
        projected = deepcopy(current)
        ft.apply_plan(projected, plans)
        fh.validate_catalog(current, projected, selected, cfg)
        fh.validate_paths(projected)
        plates = fh.render_for(cfg, rooms, plans, selected)
        if args.dry_run:
            fh.preview(projected, plans, plates, cfg)
            after_cat = projected
        else:
            fh.write(cfg, plans, plates)
            after_cat = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
        fh.write_reports(cfg, rooms, plans, selected)
        before_stats, after_stats = spread_rows(current, selected), spread_rows(after_cat, selected)
        for zid, (factor, fit, alts) in plans.items():
            counts = " / ".join(str(len(r)) for r in fit.rows)
            kinds = ",".join(f"{n} {c}" for c, n in Counter(p.item.bag.category for r in fit.rows for p in r).items())
            summary[zid] = dict(k=factor, rows=counts, kinds=kinds, ref_sqrt_area_px=target[zid],
                                target_sqrt_area_px=factor * target[zid], before=before_stats[zid], after=after_stats[zid],
                                alternatives=alts)
            b, a = before_stats[zid], after_stats[zid]
            print(f"{zid} k={factor:.2f} target√A={factor*target[zid]:.0f}px rows {counts} ({kinds}) "
                  f"CVh {b['cv_body_h']:.3f}->{a['cv_body_h']:.3f} CV√A {b['cv_body_sqrtarea']:.3f}->{a['cv_body_sqrtarea']:.3f}")
    tag = "dry" if args.dry_run else "write"
    (WORK / f"summary-{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
