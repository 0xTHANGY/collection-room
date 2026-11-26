#!/usr/bin/env python3
"""Uniform-size pass 2 for hats-a..g and totes-a..c (owner-approved suggestions).

1. Shape rods (hats-f, hats-g, totes-c): items ordered by aspect (wide-flat first, tall-narrow last) and
   cut into rods, normalised with a height-leaning metric h^0.75·w^0.25 so heights align.
2. Size-only rods (totes-a, totes-c): no category grouping; rods chosen to maximise the uniform factor.
3. Real-size nuance: target × clamp(exp(0.3·ln(h_cm / ref_cm)), 0.88, 1.12) for items with a found
   height (ref = median found height in the room if ≥2, else in the category); 未查到 → 1.00.
4. hats-a/b/c must keep k ≥ 1.00 (try 3 and 4 rods; rods are re-painted from the rail sprite).
k is relative to the b09e153 room median of the metric (same reference as uniform-size pass 1).
"""
from __future__ import annotations

import argparse, csv, json, math, random, subprocess, sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_hanging as fh
import fill_totes as ft
import uniform_metrics as um
import uniform_size as us
from fill_hanging import Bag, W, H
from rebuild_covers import composite

OUTDIR = ft.REPORT / "uniform-size-2"
WORK = OUTDIR / "work"
us.CACHE = "20261126d"
REF_COMMIT = "b09e153"
GAMMA, NUANCE = 0.3, 0.12

ROOMS = {
    # zid: (category, mode, alpha, min_k, rod_counts)
    "hats-a": ("hats", "kind", 0.5, 1.00, (3, 4)),
    "hats-b": ("hats", "kind", 0.5, 1.00, (3, 4)),
    "hats-c": ("hats", "kind", 0.5, 1.00, (3, 4)),
    "hats-d": ("hats", "kind", 0.5, 0.0, (3,)),
    "hats-e": ("hats", "kind", 0.5, 0.0, (3,)),
    "hats-f": ("hats", "shape", 0.75, 0.0, (3,)),
    "hats-g": ("hats", "shape", 0.75, 0.0, (3,)),
    "totes-a": ("totes", "size", 0.5, 0.0, (3,)),
    "totes-b": ("totes", "kind", 0.5, 0.0, (3,)),
    "totes-c": ("totes", "shape", 0.75, 0.0, (3,)),
}
# Item 4: widen the painted rails (re-painted from the rail sprite on clean wall) where the wall allows.
SPAN = {"hats-a": (140, 1490), "hats-b": (255, 1500), "hats-c": (110, 1490)}
# ...and allow the top rod a little higher on clean wall (pseudo band first; originals still erased).
TOP = {"hats-b": 70, "hats-c": 58}
CFG = {"hats": us.CONFIGS["hats"], "totes": us.CONFIGS["totes"]}


def metric(item, alpha):
    _, _, w, h = um.metrics(item)
    return h ** alpha * w ** (1 - alpha), w, h


def nuance_factors(catalog):
    specs = ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")
    found = {i["id"]: float(specs[i["id"]]["height_cm"]) for i in catalog["items"]
             if i["category"] in CFG and specs[i["id"]]["confidence"] != "未查到" and specs[i["id"]]["height_cm"]}
    cat_med = {c: float(np.median([v for k, v in found.items()
                                   if next(i for i in catalog["items"] if i["id"] == k)["category"] == c]))
               for c in CFG}
    out, refs = {}, {}
    for z in catalog["zones"]:
        if z["id"] not in ROOMS:
            continue
        here = [found[i] for i in z["items"] if i in found]
        ref = float(np.median(here)) if len(here) >= 2 else cat_med[z["category"]]
        refs[z["id"]] = (ref, "room" if len(here) >= 2 else "category")
        for iid in z["items"]:
            if iid in found:
                raw = math.exp(GAMMA * math.log(found[iid] / ref))
                out[iid] = (min(1 + NUANCE, max(1 - NUANCE, raw)), raw, found[iid])
            else:
                out[iid] = (1.0, 1.0, None)
    return out, refs


def make_bags(catalog, ref_catalog, zid, nuance):
    cat, mode, alpha, _, _ = ROOMS[zid]
    zone = next(z for z in catalog["zones"] if z["id"] == zid)
    rzone = next(z for z in ref_catalog["zones"] if z["id"] == zid)
    ritems = {i["id"]: i for i in ref_catalog["items"]}
    ref = float(np.median([metric(ritems[i], alpha)[0] for i in rzone["items"]]))
    items = {i["id"]: i for i in catalog["items"]}
    bags = []
    for iid in zone["items"]:
        item = items[iid]
        m, w, h = metric(item, alpha)
        ratio = ref * nuance[iid][0] / m
        layer = item["layer"]
        with Image.open(ft.ROOT / layer["image"]) as image:
            a = image.convert("RGBA").getchannel("A")
            x0, y0, x1, y1 = a.point(lambda p: 255 if p > 24 else 0).getbbox()
            bounds = (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)
        kind = ft.classify(item) if cat == "totes" else fh.classify(item, CFG[cat]["kind_order"])
        bw, bh = layer["width"] * W / 100, layer["height"] * H / 100
        bags.append(Bag(iid, item.get("brand") or "", item.get("name") or "", kind, zid, w / h, False,
                        ratio, bw * ratio, bh * ratio, layer["pivotX"] / 100, layer["pivotY"] / 100, bounds))
        fh.RAW_RATIO[iid] = ratio
    ko = CFG[cat]["kind_order"]
    if mode == "kind":
        bags.sort(key=lambda b: (ko.index(b.category) if b.category in ko else 99, b.height, b.id))
    else:  # shape / size start from aspect order: wide-flat first, tall-narrow last
        bags.sort(key=lambda b: (-b.height_cm, b.id))
    return bags, ref


def search(rows_fn, room):
    """Largest k (cents) for fixed rows."""
    if fh.fit_rows(rows_fn, room, 1) is None:
        return None
    low, high = 1, 100
    while fh.fit_rows(rows_fn, room, high) is not None:
        low, high = high, high * 2
        if high > 400:
            break
    while high - low > 1:
        mid = (low + high) // 2
        if fh.fit_rows(rows_fn, room, mid) is None:
            high = mid
        else:
            low = mid
    return low


def aspect_spread(rows):
    return sum(max(b.height_cm for b in r) - min(b.height_cm for b in r) for r in rows)


def best_for(bags, room, n_rods, mode):
    best = None
    for rows in fh.candidate_rows(bags, room, n_rods):
        k = search(rows, room)
        if k is None:
            continue
        fit = fh.fit_rows(rows, room, k)
        n_short = sum(len(r) <= 2 for r in rows)
        imbalance = max(map(len, rows)) - min(map(len, rows))
        if mode == "kind":
            mixed = sum(len({b.category for b in r}) > 1 for r in rows)
            rank = (n_short, mixed, imbalance > 2, -k)
        elif mode == "shape":
            # keep wide and tall on different rods; among near-equal k prefer purer shape rods
            rank = (n_short, imbalance > 2, -(k // 3), aspect_spread(rows), -k)
        else:
            rank = (n_short, imbalance > 2, -k)
        if best is None or rank < best[0]:
            best = (rank, k, fit, rows)
    return best


def size_search(bags, room, n_rods, iters=400, seed=7):
    """Free partition: local search over orders, contiguous cuts, maximise k."""
    rnd = random.Random(seed)
    order = list(bags)
    cur = best_for(order, room, n_rods, "size")
    for _ in range(iters):
        a, b = rnd.sample(range(len(order)), 2)
        cand = order[:]
        cand[a], cand[b] = cand[b], cand[a]
        res = best_for(cand, room, n_rods, "size")
        if res and (cur is None or res[1] >= cur[1]):
            order, cur = cand, res
    # final: within each rod, order wide→tall for a tidy look, recompute fit
    rows = [sorted(r, key=lambda b: (-b.height_cm, b.id)) for r in cur[3]]
    k = search(rows, room)
    return (cur[0], k, fh.fit_rows(rows, room, k), rows)


def plan(zid, bags, room):
    _, mode, _, min_k, rods = ROOMS[zid]
    options = {}
    for n in rods:
        res = size_search(bags, room, n) if mode == "size" else best_for(bags, room, n, mode)
        if res:
            options[n] = res
    if not options:
        raise ValueError(f"{zid}: no feasible layout")
    ok = {n: r for n, r in options.items() if r[1] >= min_k * 100}
    pool = ok or options
    n = max(pool, key=lambda n: (pool[n][1], -n))  # biggest k; fewer rods on ties
    rank, k, fit, rows = pool[n]
    return k / 100, fit, {str(m): r[1] / 100 for m, r in options.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--zones", default="")
    args = ap.parse_args()
    selected_all = tuple(args.zones.split(",")) if args.zones else tuple(ROOMS)
    for d in (OUTDIR, WORK, WORK / "before"):
        d.mkdir(exist_ok=True)
    fh.OUT = WORK
    fh.stage_cache = us.stage_cache
    fh.comparison = us.make_comparison(WORK / "before", "43bd6e2", "uniform-2")
    current = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    ref_catalog = json.loads(subprocess.check_output(["git", "show", f"{REF_COMMIT}:data/catalog.json"], cwd=ft.ROOT))
    for zid in selected_all:
        p = WORK / "before" / f"{zid}.jpg"
        if not p.exists():
            composite(current, zid)[0].save(p, quality=95)
    nuance, refs = nuance_factors(current)
    summary = {}
    for cat in ("hats", "totes"):
        cfg = CFG[cat]
        selected = tuple(z for z in cfg["zones"] if z in selected_all)
        if not selected:
            continue
        loader = ft.load_room if cat == "totes" else fh.load_room
        rooms = {z["id"]: loader(z) for z in current["zones"] if z["id"] in set(selected) | {cfg["rail_source"]}}
        import dataclasses
        source_room = rooms[cfg["rail_source"]]  # untouched span: the rail sprite is cut from it
        for zid in rooms:
            if zid in SPAN and zid in selected:
                r = rooms[zid]
                bands = r.bg.bands
                if zid in TOP:
                    y0, y1, x0, x1 = bands[0]
                    d = y0 - TOP[zid]
                    bands = ((y0 - d, y1 - d, x0, x1), *bands)
                rooms[zid] = ft.Room(dataclasses.replace(r.bg, span=SPAN[zid], bands=bands), r.benches)
        plans, refm = {}, {}
        for zid in selected:
            bags, refm[zid] = make_bags(current, ref_catalog, zid, nuance)
            plans[zid] = plan(zid, bags, rooms[zid])
        projected = deepcopy(current)
        ft.apply_plan(projected, plans)
        fh.validate_catalog(current, projected, selected, cfg)
        fh.validate_paths(projected)
        plates = fh.render_for(dict(cfg, rail_source="__src__"), {**rooms, "__src__": source_room}, plans, selected)
        if args.dry_run:
            fh.preview(projected, plans, plates, cfg)
            after = projected
        else:
            fh.write(cfg, plans, plates)
            after = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
        fh.write_reports(cfg, rooms, plans, selected)
        for zid, (k, fit, alts) in plans.items():
            zb = next(z for z in current["zones"] if z["id"] == zid)
            za = next(z for z in after["zones"] if z["id"] == zid)
            b, a = um.room_stats(current, zb), um.room_stats(after, za)
            rows = [[p.item.bag.id for p in r] for r in fit.rows]
            rod_cv = [um.cv([um.metrics(next(i for i in after["items"] if i["id"] == x))[3] for x in r]) for r in rows]
            summary[zid] = dict(mode=ROOMS[zid][1], alpha=ROOMS[zid][2], k=k, ref_metric_px=refm[zid],
                                alternatives_by_rods=alts, rods=rows, before=b, after=a,
                                mean_within_rod_cv_h=float(np.mean(rod_cv)),
                                nuance={x: nuance[x] for x in za["items"] if nuance[x][2] is not None},
                                nuance_ref=refs[zid])
            print(f"{zid} {ROOMS[zid][1]:5} k={k:.2f} rods {'/'.join(str(len(r)) for r in rows)} alts={alts} "
                  f"CVh {b['cv_body_h']:.3f}->{a['cv_body_h']:.3f} rodCVh {np.mean(rod_cv):.3f} "
                  f"CV√A {b['cv_body_sqrtarea']:.3f}->{a['cv_body_sqrtarea']:.3f}")
    (WORK / f"summary-{'dry' if args.dry_run else 'write'}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
