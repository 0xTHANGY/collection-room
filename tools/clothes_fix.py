#!/usr/bin/env python3
"""Owner circles 2026-11-26: jackets-f / jackets-g at real proportions.

One px/cm scale S per room (garment body length 着丈 from specs.csv): jackets-g gets the largest S its two
rods and the bench allow; jackets-f uses min(S_CAP, its own max) so its jackets stay at the scale of
jackets-a..e (~3.2 px/cm for ~70 cm jackets) and stay visibly shorter than the coats next door. Items without a published
length (未查到) keep their size relative to the room's found items (scaled by the median found change).
Rods and plates stay as painted; hooks keep the room's hanger offset (rail top + 5). Long coats go on the
rod with more drop. S = the largest value that keeps ≥12 px to the next rail, ≥14 px above the bench,
and ≥28 px between garments on every rod.
"""
from __future__ import annotations
import argparse, json, math, sys, os, fcntl
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_totes as ft
import fill_hanging as fh
import uniform_metrics as um
import uniform_size as us
from rebuild_covers import composite
from repack_three_rails import save_webp
W, H = 1536, 1024
ROOMS = ("jackets-f", "jackets-g")
BENCH_TOP = 771
S_CAP = {"jackets-f": 3.2, "jackets-g": 9.9}
HANG = 5
EDGE, GAP, RAIL_CLEAR, FLOOR_CLEAR = 24, 28, 12, 14
us.CACHE = "20261126f"
OUT = ft.REPORT / "owner-circles" / "work" / "clothes"


def alpha_box(item):
    with Image.open(ft.ROOT / item["layer"]["image"]) as im:
        a = np.asarray(im.convert("RGBA").getchannel("A")) > 24
    ys, xs = np.where(a)
    h, w = a.shape
    return xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h


def rails(zid):
    rgb = np.asarray(Image.open(ft.ROOT / f"assets/zones/{zid}-1600.webp").convert("RGB"))
    return fh.detect_rails(rgb)


def plan(catalog, S, rooms=ROOMS):
    specs = ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")
    items = {i["id"]: i for i in catalog["items"]}
    out = {}
    for zid in rooms:
        z = next(z for z in catalog["zones"] if z["id"] == zid)
        bands = rails(zid)
        recs = []
        for iid in z["items"]:
            it = items[iid]
            _, _, bw, bh = um.metrics(it)
            s = specs.get(iid)
            L = float(s["height_cm"]) if s and s["confidence"] != "未查到" and s["height_cm"] else None
            recs.append(dict(id=iid, bh=bh, cm=L, item=it))
        found = [r for r in recs if r["cm"]]
        Sz = S[zid] if isinstance(S, dict) else S
        med = float(np.median([Sz * r["cm"] / r["bh"] for r in found]))
        for r in recs:
            r["f"] = Sz * r["cm"] / r["bh"] if r["cm"] else med
        # rod assignment: keep counts, longest garments on the rod with the most drop
        counts = [sum(1 for iid in z["items"] if abs(items[iid]["layer"]["top"] * H / 100 - (b[0] + HANG)) < 25) for b in bands]
        limits = [bands[i + 1][0] - RAIL_CLEAR if i + 1 < len(bands) else BENCH_TOP - FLOOR_CLEAR for i in range(len(bands))]
        drops = [limits[i] - bands[i][0] for i in range(len(bands))]
        rows = {i: [r for r in recs if abs(r["item"]["layer"]["top"] * H / 100 - (bands[i][0] + HANG)) < 25]
                for i in range(len(bands))}
        assert sum(map(len, rows.values())) == len(recs)
        placed = []
        for i, band in enumerate(bands):
            row = sorted(rows[i], key=lambda r: z["items"].index(r["id"]))
            top = band[0] + HANG
            geo = []
            for r in row:
                L = r["item"]["layer"]
                w, h = L["width"] * W / 100 * r["f"], L["height"] * H / 100 * r["f"]
                ax0, ay0, ax1, ay1 = alpha_box(r["item"])
                if top + ay1 * h > limits[i]:
                    return None
                geo.append((r, w, h, ax0, ax1))
            left, right = band[2] + EDGE, band[3] - EDGE
            vis = sum((ax1 - ax0) * w for _, w, _, ax0, ax1 in geo)
            spare = right - left - vis - GAP * (len(geo) - 1)
            if spare < 0:
                return None
            gap = GAP + spare / (len(geo) + 1)
            x = left + spare / (len(geo) + 1)
            for r, w, h, ax0, ax1 in geo:
                bx = x - ax0 * w
                placed.append((r["id"], bx, top, w, h, r))
                x += (ax1 - ax0) * w + GAP + spare / (len(geo) + 1)
        out[zid] = placed
    return out


def pct(x, y, w, h):
    return {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}


def apply(catalog, plans):
    c = deepcopy(catalog)
    items = {i["id"]: i for i in c["items"]}
    for zid, placed in plans.items():
        z = next(z for z in c["zones"] if z["id"] == zid)
        z["items"] = [p[0] for p in placed]
        for zi, (iid, x, y, w, h, _) in enumerate(placed, 1):
            box = pct(x, y, w, h)
            it = items[iid]
            it["layer"].update(box, rect=box, zIndex=zi)
            it["hotspot"].update(box)
    return c


def validate(before, after):
    bi = {i["id"]: i for i in before["items"]}
    for it in after["items"]:
        o = bi[it["id"]]
        mutable = {"layer", "hotspot"} if it["zone"] in ROOMS else set()
        assert all(it[k] == o[k] for k in o.keys() - mutable), it["id"]
    for zid in ROOMS:
        z = next(z for z in after["zones"] if z["id"] == zid)
        boxes = []
        for iid in z["items"]:
            it = next(i for i in after["items"] if i["id"] == iid)
            b = it["layer"]["rect"]
            assert all(it["hotspot"][k] == v for k, v in b.items())
            x, y, w, h = b["left"], b["top"], b["width"], b["height"]
            assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100, iid
            for ax, ay, bx, by in boxes:
                assert not (min(bx, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{iid} overlap"
            boxes.append((x, y, x + w, y + h))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "before").mkdir(exist_ok=True)
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    for zid in ROOMS:
        p = OUT / "before" / f"{zid}.jpg"
        if not p.exists():
            composite(cur, zid)[0].save(p, quality=95)
    S = {}
    for zid in ROOMS:
        lo, hi = 1.0, 6.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if plan(cur, mid, (zid,)):
                lo = mid
            else:
                hi = mid
        S[zid] = min(S_CAP[zid], math.floor(lo * 100) / 100)
    plans = plan(cur, S)
    new = apply(cur, plans)
    validate(cur, new)
    rep = {zid: [dict(id=iid, cm=r["cm"], factor=round(r["f"], 3), body_h_before=round(r["bh"]),
                      body_h_after=round(r["bh"] * r["f"]), top=round(y)) for iid, x, y, w, h, r in placed]
           for zid, placed in plans.items()}
    print("S px/cm", S)
    for zid in rep:
        for r in rep[zid]:
            print(zid, r)
    (OUT / f"summary-{'write' if args.write else 'dry'}.json").write_text(json.dumps({"S_px_per_cm": S, "rooms": rep}, indent=1))
    if args.write:
        fd = os.open(ft.ROOT, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with TemporaryDirectory(prefix=".clothes-", dir=ft.ROOT) as tmp:
                stage = Path(tmp); paths = []
                for zid in ROOMS:
                    z = next(z for z in new["zones"] if z["id"] == zid)
                    img, _ = composite(new, zid)
                    cover = Path(f"assets/zones/{zid}-cover-640.webp")
                    hh = round(640 * img.height / img.width)
                    save_webp(img.resize((640, hh), Image.Resampling.LANCZOS), stage / cover)
                    z["cover"] = {"src": str(cover), "width": 640, "height": hh}
                    paths.append(cover)
                cache = us.stage_cache(stage)
                ft.validate_paths(new, stage)
                (stage / ft.CATALOG.parent).mkdir()
                ft.dump(stage / ft.CATALOG, new)
                ft.publish(stage, [*paths, ft.CATALOG, *cache])
        finally:
            os.close(fd)
    for zid in ROOMS:
        img, _ = composite(new, zid)
        img.save(OUT / f"{zid}-{'after' if args.write else 'preview'}.jpg", quality=92)
        us.make_comparison(OUT / "before", "992fcfd", f"{S[zid]:.2f} px/cm ×")(zid, img, 1.0, OUT / f"{zid}-before-after.jpg")


if __name__ == "__main__":
    main()
