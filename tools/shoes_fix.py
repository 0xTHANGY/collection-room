#!/usr/bin/env python3
"""Owner-circles shoe pass (2026-11-26): real proportions + firm support in shoes-a..e.

Rug pairs: every pair is scaled to the same *visible pair width* RUG_W (a pair of shoes is about
equally wide whatever its style), so height then follows the real shape - boots come out taller
than loafers/sneakers/sandals. A pair taller than its row pitch is capped (box must not reach into
the row behind). Each rug row is re-spread evenly over its old span, bottoms stay on the baseline.
Bench / shelf pairs: visible width BENCH_W = 0.8 * RUG_W (perspective: they stand further back),
alpha bottom seated BENCH_SINK px into the shelf top detected by check_shoe_supports, centred on
the free part of the shelf (clear of rack posts).
Box sizes come from each layer image's own aspect, so re-cut layers (shoes-n035) are not stretched.
Hotspot clips are recomputed from the resized alpha.
"""
from __future__ import annotations
import argparse, json, os, fcntl, sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fill_totes as ft
import uniform_size as us
us.CACHE = "20261126f"
from rebuild_covers import composite
from repack_three_rails import save_webp, clip_polygon
import check_shoe_supports as cs

W, H = 1536, 1024
ROOMS = ("shoes-a", "shoes-b", "shoes-c", "shoes-d", "shoes-e")
RUG_W = 165
BENCH_W = round(0.8 * RUG_W)
BENCH_SINK = 3
SHELF_PAD = 12
ROW_CLEAR = 2
RAIL_CLEAR = 12
OUT = Path("/workspace/collection-room-impl-report/owner-circles/work/shoes")


def native(it):
    im = Image.open(ROOT / it["layer"]["image"]).convert("RGBA")
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > 24)
    return im.size, (xs.min(), ys.min(), xs.max(), ys.max())


def box_px(it):
    L = it["layer"]
    return L["left"] / 100 * W, L["top"] / 100 * H, L["width"] / 100 * W, L["height"] / 100 * H


def plan(cat):
    zones = {z["id"]: z for z in cat["zones"]}
    out = {}
    for zid in ROOMS:
        rgb = cs.plate_rgb(zones[zid])
        rug_y = cs.rug_top_y(rgb)
        shelves, posts, _ = cs.furniture(rgb, rug_y)
        items = [it for it in cat["items"] if it["zone"] == zid]
        rug = [it for it in items if (it["layer"]["top"] + it["layer"]["height"]) / 100 * H > rug_y]
        bench = [it for it in items if it not in rug]
        placed = {}
        # bench first
        by_shelf = {}
        for it in bench:
            x, y, w, h = box_px(it)
            px, py, a = cs.placed_alpha(it)
            bottom, bx0, bx1 = cs.alpha_bottom(px, py, a)
            sh = cs.nearest_shelf(bottom, bx0, bx1, shelves)
            by_shelf.setdefault(sh[1:], []).append(it)
        for (sy, sx0, sx1), its in by_shelf.items():
            lo, hi = sx0 + SHELF_PAD, sx1 - SHELF_PAD
            for p0, p1, py0, py1 in posts:
                if py0 < sy < py1 and p0 < hi and p1 > lo:
                    if p0 > (lo + hi) / 2:
                        hi = p0 - SHELF_PAD
                    else:
                        lo = p1 + SHELF_PAD
            wood = cs.wood_mask(rgb)
            frac = wood[:, int(lo):int(hi)].mean(1)
            rail_rows = [y for y in range(0, sy - 15) if frac[y] > 0.6]
            ceiling = (max(rail_rows) + RAIL_CLEAR) if rail_rows else 40
            its.sort(key=lambda i: box_px(i)[0])
            n = len(its)
            for k, it in enumerate(its):
                (iw, ih), (ax0, ay0, ax1, ay1) = native(it)
                s = BENCH_W / (ax1 - ax0 + 1)
                s = min(s, (hi - lo) / n * 0.9 / (ax1 - ax0 + 1),
                        (sy + BENCH_SINK - ceiling) / (ay1 - ay0 + 1))
                cx = lo + (hi - lo) * (k + 0.5) / n
                bw, bh = iw * s, ih * s
                left = cx - (ax0 + ax1 + 1) / 2 * s
                top = sy + BENCH_SINK - (ay1 + 1) * s
                placed[it["id"]] = (left, top, bw, bh, dict(kind="bench", shelf=sy, ceiling=ceiling, s=s))
        # rug rows
        rows = {}
        for it in rug:
            x, y, w, h = box_px(it)
            rows.setdefault(round(y + h), []).append(it)
        bases = sorted(rows)
        merged = []
        for b in bases:
            if merged and b - merged[-1][0] <= 8:
                merged[-1][1].extend(rows[b])
            else:
                merged.append([b, list(rows[b])])
        prev_bottom = None
        for ri, (base, its) in enumerate(merged):
            its.sort(key=lambda i: box_px(i)[0])
            if prev_bottom is None:
                # back row: stay below any bench box above it
                cap_top = 0
            else:
                cap_top = prev_bottom + ROW_CLEAR
            span0 = min(box_px(i)[0] for i in its)
            span1 = max(box_px(i)[0] + box_px(i)[2] for i in its)
            sizes = []
            for it in its:
                (iw, ih), (ax0, ay0, ax1, ay1) = native(it)
                s = RUG_W / (ax1 - ax0 + 1)
                lim_top = cap_top
                if prev_bottom is None:
                    x, y, w, h = box_px(it)
                    for bid, (bl, bt, bw_, bh_, _) in placed.items():
                        if bl < x + w and bl + bw_ > x:
                            lim_top = max(lim_top, bt + bh_ + ROW_CLEAR)
                    lim_top = max(lim_top, min(y, rug_y - 80))
                s = min(s, (base - lim_top) / ih)
                sizes.append((it, iw * s, ih * s, s, (ax0, ax1)))
            tot = sum(z[1] for z in sizes)
            gap = (span1 - span0 - tot) / max(1, len(sizes) - 1) if len(sizes) > 1 else 0
            x = span0 if len(sizes) > 1 else span0 + (span1 - span0 - tot) / 2
            for it, bw, bh, s, _ in sizes:
                placed[it["id"]] = (x, base - bh, bw, bh, dict(kind=f"rug{ri}", s=s))
                x += bw + gap
            prev_bottom = base
        out[zid] = placed
    return out


def _inside(poly, x, y):
    import cv2
    return cv2.pointPolygonTest(np.array(poly, np.float32), (x, y), False) > 0


def centre_safe_clip(alpha):
    """clip_polygon, but the box centre (where a plain click lands) must be inside the clip.
    Pairs with a gap between the two shoes otherwise leave the centre on the plate image."""
    poly = clip_polygon(alpha)
    if _inside(poly, 50, 50):
        return poly
    a = alpha.copy()
    h, w = a.shape
    ys, xs = np.where(a > 80)
    yc = h // 2
    band = a[max(0, yc - h // 12): yc + h // 12]
    cols = np.where(band.max(0) > 80)[0]
    x0, x1 = (cols.min(), cols.max()) if len(cols) else (w // 3, 2 * w // 3)
    a[max(0, yc - h // 12): yc + h // 12, x0:x1 + 1] = np.maximum(a[max(0, yc - h // 12): yc + h // 12, x0:x1 + 1], 120)
    poly = clip_polygon(a)
    assert _inside(poly, 50, 50)
    return poly


def apply(cat, plans):
    c = deepcopy(cat)
    items = {i["id"]: i for i in c["items"]}
    for zid, placed in plans.items():
        for iid, (x, y, w, h, _) in placed.items():
            box = {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}
            it = items[iid]
            it["layer"].update(box, rect=dict(box))
            it["hotspot"].update(box)
            im = Image.open(ROOT / it["layer"]["image"]).convert("RGBA")
            rs = im.resize((max(1, round(w)), max(1, round(h))), Image.Resampling.LANCZOS)
            it["hotspot"]["clip"] = centre_safe_clip(np.array(rs)[:, :, 3])
    return c


def validate(before, after):
    bi = {i["id"]: i for i in before["items"]}
    for it in after["items"]:
        o = bi[it["id"]]
        mutable = {"layer", "hotspot"} if it["zone"] in ROOMS else set()
        assert all(it[k] == o[k] for k in o.keys() - mutable), it["id"]
    for zid in ROOMS:
        boxes = []
        for it in [i for i in after["items"] if i["zone"] == zid]:
            b = it["layer"]["rect"]
            x, y, w, h = b["left"], b["top"], b["width"], b["height"]
            assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100.0001, it["id"]
            for jid, ax, ay, bx, by in boxes:
                assert not (min(bx, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{it['id']} overlaps {jid}"
            boxes.append((it["id"], x, y, x + w, y + h))
    errs, _, _ = cs.check(after)
    assert not errs, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--rooms", nargs="*")
    args = ap.parse_args()
    global ROOMS
    if args.rooms:
        ROOMS = tuple(args.rooms)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "before").mkdir(exist_ok=True)
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    for zid in ROOMS:
        p = OUT / "before" / f"{zid}.jpg"
        if not p.exists():
            composite(cur, zid)[0].save(p, quality=95)
    plans = plan(cur)
    new = apply(cur, plans)
    rep = {}
    for zid, placed in plans.items():
        rep[zid] = []
        for iid, (x, y, w, h, m) in placed.items():
            o = next(i for i in cur["items"] if i["id"] == iid)["layer"]
            rep[zid].append(dict(id=iid, **m, box_before=[round(o["width"] / 100 * W), round(o["height"] / 100 * H)],
                                 box_after=[round(w), round(h)], top=round(y), bottom=round(y + h)))
            print(zid, rep[zid][-1])
    (OUT / f"summary-{'write' if args.write else 'dry'}.json").write_text(json.dumps(rep, indent=1, default=float))
    try:
        validate(cur, new)
        print("VALID")
    except AssertionError as e:
        print("INVALID", e)
        if args.write:
            raise
    if args.write:
        fd = os.open(ft.ROOT, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with TemporaryDirectory(prefix=".shoes-", dir=ft.ROOT) as tmp:
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
        us.make_comparison(OUT / "before", "992fcfd", "real proportions")(zid, img, 1.0, OUT / f"{zid}-before-after.jpg")


if __name__ == "__main__":
    main()
