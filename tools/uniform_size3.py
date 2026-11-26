#!/usr/bin/env python3
"""Uniform-size pass 3: move one cap out of hats-c (headroom), recompute the ±12 % real-size nudge with
the second-pass specs (hat height convention), and re-layout the hat/tote rooms whose inputs changed.
Same rules as uniform_size2 (rods by kind / shape / size, k vs b09e153 room median, wider rails for hats-a..c).
"""
from __future__ import annotations
import argparse, dataclasses, json, subprocess, sys
from copy import deepcopy
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_hanging as fh
import fill_totes as ft
import uniform_metrics as um
import uniform_size as us
import uniform_size2 as u2
from rebuild_covers import composite

OUTDIR = ft.REPORT / "uniform-size-3"
WORK = OUTDIR / "work"
us.CACHE = "20261126e"
BEFORE = "206e635"


def room_obj(current, zid, loader):
    z = next(z for z in current["zones"] if z["id"] == zid)
    r = loader(z)
    if zid in u2.SPAN:
        bands = r.bg.bands
        if zid in u2.TOP:
            y0, y1, x0, x1 = bands[0]
            d = y0 - u2.TOP[zid]
            bands = ((y0 - d, y1 - d, x0, x1), *bands)
        r = ft.Room(dataclasses.replace(r.bg, span=u2.SPAN[zid], bands=bands), r.benches)
    return r


def with_move(catalog, iid, dest):
    c = deepcopy(catalog)
    item = next(i for i in c["items"] if i["id"] == iid)
    src = item["zone"]
    item["zone"] = dest
    for z in c["zones"]:
        if z["id"] == src:
            z["items"] = [x for x in z["items"] if x != iid]
        if z["id"] == dest:
            z["items"] = z["items"] + [iid]
    return c


def k_of(catalog, ref, nuance, zid, room):
    bags, _ = u2.make_bags(catalog, ref, zid, nuance)
    return u2.plan(zid, bags, room)[0]


def choose_move(current, ref, nuance, rooms):
    """Pick destination among cap rooms with spare capacity, then the cap that leaves hats-c largest."""
    zc = next(z for z in current["zones"] if z["id"] == "hats-c")["items"]
    base = {z: k_of(current, ref, nuance, z, rooms[z]) for z in ("hats-a", "hats-b", "hats-c", "hats-d")}
    trials = []
    for iid in zc:
        moved = None
        for dest in ("hats-a", "hats-b", "hats-d"):
            cat = with_move(current, iid, dest)
            kc = k_of(cat, ref, nuance, "hats-c", rooms["hats-c"])
            kd = k_of(cat, ref, nuance, dest, rooms[dest])
            trials.append(dict(item=iid, dest=dest, k_c=kc, k_dest=kd, k_dest_before=base[dest],
                               dest_loss=round(base[dest] - kd, 2)))
    # most suitable: smallest loss in the receiving room (spare capacity), then largest hats-c gain, then id
    best = min(trials, key=lambda t: (t["dest_loss"], -t["k_c"], -t["k_dest"], t["item"]))
    return best, base, trials


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--move", default="auto", help="item:dest, 'auto', or 'none'")
    args = ap.parse_args()
    selected_all = tuple(args.zones.split(","))
    for d in (OUTDIR, WORK, WORK / "before"):
        d.mkdir(exist_ok=True)
    fh.OUT = WORK
    fh.stage_cache = us.stage_cache
    fh.comparison = us.make_comparison(WORK / "before", BEFORE, "uniform-3")
    fh.best_rows = us.best_rows
    current = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    ref = json.loads(subprocess.check_output(["git", "show", f"{u2.REF_COMMIT}:data/catalog.json"], cwd=ft.ROOT))
    for zid in selected_all:
        p = WORK / "before" / f"{zid}.jpg"
        if not p.exists():
            composite(current, zid)[0].save(p, quality=95)
    nuance, refs = u2.nuance_factors(current)
    hat_rooms = {z: room_obj(current, z, fh.load_room) for z in u2.CFG["hats"]["zones"]}
    move = None
    if args.move == "auto":
        best, base, trials = choose_move(current, ref, nuance, hat_rooms)
        (WORK / "move-trials.json").write_text(json.dumps(dict(best=best, base=base, trials=trials), indent=1))
        move = (best["item"], best["dest"])
        print("move:", best, "base:", base)
    elif args.move != "none":
        move = tuple(args.move.split(":"))
    planning = with_move(current, *move) if move else current
    nuance, refs = u2.nuance_factors(planning)  # refs follow the post-move membership
    summary = {}
    for cat in ("hats", "totes"):
        cfg = u2.CFG[cat]
        selected = tuple(z for z in cfg["zones"] if z in selected_all)
        if not selected:
            continue
        loader = ft.load_room if cat == "totes" else fh.load_room
        rooms = hat_rooms if cat == "hats" else {z: room_obj(current, z, loader) for z in cfg["zones"]}
        source_room = loader(next(z for z in current["zones"] if z["id"] == cfg["rail_source"]))
        plans, refm = {}, {}
        for zid in selected:
            bags, refm[zid] = u2.make_bags(planning, ref, zid, nuance)
            plans[zid] = u2.plan(zid, bags, rooms[zid])
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
            items_after = {i["id"]: i for i in after["items"]}
            rod_cv = [um.cv([um.metrics(items_after[x])[3] for x in r]) for r in rows]
            summary[zid] = dict(mode=u2.ROOMS[zid][1], alpha=u2.ROOMS[zid][2], k=k, alternatives_by_rods=alts,
                                rods=rows, before=b, after=a, mean_within_rod_cv_h=float(np.mean(rod_cv)),
                                nuance={x: nuance[x] for x in za["items"] if nuance[x][2] is not None},
                                nuance_ref=refs.get(zid))
            print(f"{zid} {u2.ROOMS[zid][1]:5} k={k:.2f} n={sum(map(len, rows))} rods {'/'.join(str(len(r)) for r in rows)} "
                  f"CVh {b['cv_body_h']:.3f}->{a['cv_body_h']:.3f} CV√A {b['cv_body_sqrtarea']:.3f}->{a['cv_body_sqrtarea']:.3f} "
                  f"nudged {len(summary[zid]['nuance'])}")
    (WORK / f"summary-{'dry' if args.dry_run else 'write'}.json").write_text(
        json.dumps(dict(move=move, rooms=summary), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
