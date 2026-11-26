#!/usr/bin/env python3
"""Owner circles 2026-11-26: hat rooms with 6 hats per rod, smaller overall.

* 98 hats in 6 rooms (hats-a..f); hats-g is retired and redirected (#/hats/g, #/hats/l → its main room).
* Backgrounds: a,b,c keep theirs; the narrow hats-d wall is dropped, d/e/f use the e/f/g walls (88d42ef plates).
* Rods hold 6 hats (last rod takes the remainder); rails re-painted from the rail sprite on clean wall.
* Size: one common target for every hat room (uniform rule) × the ±12 % real-height nudge (H convention).
  The body box ignores cords, chin straps, braids, tassels and pompoms (tools/hat_body.py), the same parts
  the H convention excludes, so strap/pompom hats are no longer shrunk.
"""
from __future__ import annotations
import argparse, dataclasses, json, math, os, fcntl, sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_hanging as fh
import fill_totes as ft
import uniform_size as us
import hat_body as hb
from fill_totes import Bag, W, H
from rebuild_covers import composite
from repack_three_rails import save_webp, variants_from

OUTDIR = ft.REPORT / "owner-circles"
WORK = OUTDIR / "work" / "hats6"
us.CACHE = "20261126f"
CFG = us.CONFIGS["hats"]
ZONES = ("hats-a", "hats-b", "hats-c", "hats-d", "hats-e", "hats-f")
GONE = "hats-g"
PLATE = {"hats-a": "hats-a", "hats-b": "hats-b", "hats-c": "hats-c",
         "hats-d": "hats-e", "hats-e": "hats-f", "hats-f": "hats-g"}
SPAN = {"hats-a": (140, 1490), "hats-b": (255, 1500), "hats-c": (110, 1490),
        "hats-d": (200, 1480), "hats-e": (130, 1440), "hats-f": (190, 1480)}
TOP = {"hats-b": 70, "hats-c": 58, "hats-e": 60}
ALPHA = {"hats-e": 0.75, "hats-f": 0.75}
PER_ROD = 6
GAMMA, NUANCE = 0.3, 0.12
EXTRA_BEANIES = ("batch2-137", "batch2-063", "batch2-148")  # pompom / tall beanies join the odd-shape room


def kind(item):
    return fh.classify(item, CFG["kind_order"])


def assign(catalog):
    order = [iid for z in catalog["zones"] if z["category"] == "hats" for iid in z["items"]]
    items = {i["id"]: i for i in catalog["items"]}
    by = {}
    for iid in order:
        by.setdefault(kind(items[iid]), []).append(iid)
    caps = by.pop("cap")
    beanies = [b for b in by.pop("beanie") if b not in EXTRA_BEANIES]
    odd = list(EXTRA_BEANIES) + by.pop("bucket") + by.pop("earflap") + by.pop("balaclava")
    misc = [i for k in ("brim", "newsboy", "beret", "boater", "other") for i in by.pop(k, [])]
    if by:
        raise ValueError(f"unassigned kinds {by}")
    n = len(caps)
    a, b = math.ceil(n / 3), math.ceil((n - math.ceil(n / 3)) / 2)
    rooms = {"hats-a": caps[:a], "hats-b": caps[a:a + b], "hats-c": caps[a + b:],
             "hats-d": beanies, "hats-e": odd, "hats-f": misc}
    assert sum(map(len, rooms.values())) == 98, Counter({k: len(v) for k, v in rooms.items()})
    return rooms


def nuance(catalog, rooms):
    specs = ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")
    found = {i: float(specs[i]["height_cm"]) for ids in rooms.values() for i in ids
             if specs[i]["confidence"] != "未查到" and specs[i]["height_cm"]}
    cat_med = float(np.median(list(found.values())))
    out, refs = {}, {}
    for zid, ids in rooms.items():
        here = [found[i] for i in ids if i in found]
        ref = float(np.median(here)) if len(here) >= 2 else cat_med
        refs[zid] = (ref, "room" if len(here) >= 2 else "category")
        for i in ids:
            if i in found:
                raw = math.exp(GAMMA * math.log(found[i] / ref))
                out[i] = (min(1 + NUANCE, max(1 - NUANCE, raw)), raw, found[i])
            else:
                out[i] = (1.0, 1.0, None)
    return out, refs


def bags_for(catalog, zid, ids, nu, target=100.0):
    items = {i["id"]: i for i in catalog["items"]}
    alpha = ALPHA.get(zid, 0.5)
    bags = []
    for iid in ids:
        item = items[iid]
        w, h = hb.body_px(item)
        ratio = target * nu[iid][0] / (h ** alpha * w ** (1 - alpha))
        L = item["layer"]
        with Image.open(ft.ROOT / L["image"]) as image:
            a = image.convert("RGBA").getchannel("A")
            x0, y0, x1, y1 = a.point(lambda p: 255 if p > 24 else 0).getbbox()
            bounds = (x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)
        bw, bh = L["width"] * W / 100, L["height"] * H / 100
        bags.append(Bag(iid, item.get("brand") or "", item.get("name") or "", kind(item), item["zone"],
                        w / h, False, ratio, bw * ratio, bh * ratio, L["pivotX"] / 100, L["pivotY"] / 100, bounds))
    if zid in ALPHA:   # shape rooms: long cords/braids/straps on the lowest rod, then wide-flat → tall
        dangle = {b.id: 1 - hb.body_frac(items[b.id])[3] for b in bags}
        lows = set(sorted(bags, key=lambda b: (-dangle[b.id], b.id))[: len(bags) - 2 * PER_ROD]) if len(bags) > 12 else set()
        bags.sort(key=lambda b: (b in lows, -b.height_cm, b.id))
    else:
        ko = CFG["kind_order"]
        bags.sort(key=lambda b: (ko.index(b.category), -b.height_cm, b.id))
    return bags


def rows_of(bags):
    n = len(bags)
    if n <= 12:  # misc room: even 4/4/4
        k = math.ceil(n / 3)
        return [bags[:k], bags[k:2 * k], bags[2 * k:]]
    return [bags[:PER_ROD], bags[PER_ROD:2 * PER_ROD], bags[2 * PER_ROD:]]


def load_rooms(catalog):
    zones = {z["id"]: z for z in catalog["zones"]}
    rooms = {}
    for zid in ZONES:
        r = fh.load_room(zones[PLATE[zid]])
        bands = r.bg.bands
        if zid in TOP:
            y0, y1, x0, x1 = bands[0]
            d = y0 - TOP[zid]
            bands = ((y0 - d, y1 - d, x0, x1), *bands)
        rooms[zid] = ft.Room(dataclasses.replace(r.bg, span=SPAN[zid], bands=bands), r.benches)
    rooms["__src__"] = fh.load_room(zones[CFG["rail_source"]])
    return rooms


def max_k(rows, room):
    lo, hi = 1, 400
    if ft.fit_rows(rows, room, lo) is None:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ft.fit_rows(rows, room, mid) is None:
            hi = mid
        else:
            lo = mid
    return lo


def redirect_target(catalog, rooms):
    old = next(z for z in catalog["zones"] if z["id"] == GONE)["items"]
    best = max(ZONES, key=lambda z: (sum(i in rooms[z] for i in old), z))
    return f"#/hats/{best.split('-')[1]}"


def build(catalog, plans, rooms_ids):
    updated = deepcopy(catalog)
    ft.apply_plan(updated, plans)
    target = redirect_target(catalog, rooms_ids)
    updated["zones"] = [z for z in updated["zones"] if z["id"] != GONE]
    red = updated["redirects"]
    for k, v in list(red.items()):
        if v == "#/hats/g":
            red[k] = target
    red["hats/g"] = target
    red["hats/hats-g"] = target
    return updated, target


def validate(before, after):
    bi = {i["id"]: i for i in before["items"]}
    ai = {i["id"]: i for i in after["items"]}
    assert list(bi) == list(ai), "item ids changed"
    zones = {z["id"]: z for z in after["zones"]}
    assert GONE not in zones and all(z in zones for z in ZONES)
    hats = [i for i in ai if ai[i]["category"] == "hats"]
    member = [i for z in ZONES for i in zones[z]["items"]]
    assert Counter(member) == Counter(hats) and len(hats) == 98, "hat membership"
    for zid in ZONES:
        boxes = []
        for iid in zones[zid]["items"]:
            it = ai[iid]
            assert it["zone"] == zid
            box = it["layer"]["rect"]
            assert all(it["layer"][k] == v and it["hotspot"][k] == v for k, v in box.items()), iid
            x, y, w, h = (box[k] for k in ("left", "top", "width", "height"))
            assert 0 <= x < x + w <= 100 and 0 <= y < y + h <= 100, f"{iid} outside"
            for ax, ay, bx, by in boxes:
                assert not (min(bx, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{iid} overlap"
            boxes.append((x, y, x + w, y + h))
    for iid, it in ai.items():
        old = bi[iid]
        mutable = {"zone", "layer", "hotspot"} if it["category"] == "hats" else set()
        assert all(it[k] == old[k] for k in old.keys() - mutable), f"{iid} protected field changed"
        if it["category"] == "hats":
            assert it["layer"]["image"] == old["layer"]["image"] and it["hotspot"].get("clip") == old["hotspot"].get("clip")
    zl = {f"{z['category']}/{z['letter']}" for z in after["zones"]}
    for k, v in after["redirects"].items():
        assert v.removeprefix("#/") in zl or v.startswith("#/item/"), f"{k}: unresolved {v}"
    for z in before["zones"]:
        if z["category"] != "hats":
            assert z == zones[z["id"]], f"{z['id']} changed"


def write(updated, plates):
    root_fd = os.open(ft.ROOT, os.O_RDONLY)
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        with TemporaryDirectory(prefix=".hats6-", dir=ft.ROOT) as tmp:
            stage = Path(tmp)
            paths = []
            for zid, plate in plates.items():
                variants = variants_from(plate, str(stage / "assets/zones" / zid), (1600, 1024, 640))
                zone = next(z for z in updated["zones"] if z["id"] == zid)
                image, _ = composite({"zones": [{**zone, "images": variants}], "items": updated["items"]}, zid)
                for v in variants:
                    v["src"] = str(Path(v["src"]).relative_to(stage))
                    paths.append(Path(v["src"]))
                zone["images"] = variants
                cover = Path(f"assets/zones/{zid}-cover-640.webp")
                height = round(640 * image.height / image.width)
                save_webp(image.resize((640, height), Image.Resampling.LANCZOS), stage / cover)
                zone["cover"] = {"src": str(cover), "width": 640, "height": height}
                paths.append(cover)
            cache_paths = us.stage_cache(stage)
            ft.validate_paths(updated, stage)
            (stage / ft.CATALOG.parent).mkdir()
            ft.dump(stage / ft.CATALOG, updated)
            ft.publish(stage, [*paths, ft.CATALOG, *cache_paths])
    finally:
        os.close(root_fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--cap", type=int, default=175, help="common size ceiling (cents); a room that cannot reach it uses its own max")
    args = ap.parse_args()
    (WORK / "before").mkdir(parents=True, exist_ok=True)
    (WORK / "preview").mkdir(exist_ok=True)
    current = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    for z in current["zones"]:
        if z["category"] == "hats":
            p = WORK / "before" / f"{z['id']}.jpg"
            if not p.exists():
                composite(current, z["id"])[0].save(p, quality=95)
    ids = assign(current)
    nu, refs = nuance(current, ids)
    rooms = load_rooms(current)
    rows = {zid: rows_of(bags_for(current, zid, ids[zid], nu)) for zid in ZONES}
    ks = {zid: max_k(rows[zid], rooms[zid]) for zid in ZONES}
    print("max k per room", ks)
    if any(v is None for v in ks.values()):
        raise SystemExit("infeasible room")
    K = {zid: min(args.cap, ks[zid]) for zid in ZONES}
    plans = {zid: (K[zid] / 100, ft.fit_rows(rows[zid], rooms[zid], K[zid]), {}) for zid in ZONES}
    updated, target = build(current, plans, ids)
    validate(current, updated)
    cfg = dict(CFG, rail_source="__src__")
    plates = fh.render_for(cfg, rooms, plans, ZONES)
    summary = {"K": K, "max_k": ks, "redirect_hats_g": target, "refs": refs,
               "rooms": {z: [[p.item.bag.id for p in r] for r in plans[z][1].rows] for z in ZONES},
               "nuance": {i: v for i, v in nu.items() if v[2] is not None}}
    dest = WORK / ("preview" if args.dry_run else "after")
    dest.mkdir(exist_ok=True)
    if args.write:
        write(updated, plates)
        updated = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
        validate(current, updated)
    for zid in ZONES:
        zone = next(z for z in updated["zones"] if z["id"] == zid)
        if args.dry_run:
            plates[zid].save(dest / f"{zid}-plate.png")
            zone = {**zone, "images": [{"src": str(dest / f"{zid}-plate.png"), "width": W, "height": H}]}
        img, _ = composite({"zones": [zone], "items": updated["items"]}, zid)
        img.save(dest / f"{zid}.jpg", quality=92)
        us.make_comparison(WORK / "before", "992fcfd", "6-per-rod")(zid, img, K[zid] / 100, dest / f"{zid}-before-after.jpg")
        print(zid, "rods", "/".join(str(len(r)) for r in summary["rooms"][zid]), "plate", PLATE[zid])
    (WORK / f"summary-{'dry' if args.dry_run else 'write'}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("K", K, "redirect hats/g ->", target)


if __name__ == "__main__":
    main()
