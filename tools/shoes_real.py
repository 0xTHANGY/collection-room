#!/usr/bin/env python3
"""Owner circles round 2 (2026-11-26): rug and bench shoes at real size from specs.csv.

Every pair is photographed from the same front-above angle, so a pair's visible width is proportional
to the shoe's real length L (published outsole length; median of the known lengths where 未查到).
Pair width = K_CM * L px. Height then follows each photo, which is where the real shaft/heel height shows:
boots come out taller, loafers/sandals lower. K_CM is fixed so the median low shoe keeps its 992fcfd size
(nothing is shrunk to make boots stand out).
Tall pairs need more room than the old equal rows gave them, so each room re-rows its rug pairs: tallest
pairs go to the back row, rows keep their counts, and each row baseline is set from the row in front of it
(box tops of the front row + ROW_CLEAR); the front baseline stays where it was. The back row must still stand
on the rug (baseline >= rug top + RUG_MIN) and below any bench box above it.
Bench/shelf pairs: width 0.8 * K_CM * L (further back), seated BENCH_SINK px into the shelf, kept
RAIL_CLEAR below any rail and clear of posts (same rules as shoes_fix.py).
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
from rebuild_covers import composite
from repack_three_rails import save_webp
import check_shoe_supports as cs
from shoes_fix import native, box_px, centre_safe_clip, SHELF_PAD, RAIL_CLEAR, BENCH_SINK
us.CACHE = "20261126g"   # after shoes_fix import (it sets its own)

W, H = 1536, 1024
ROOMS = ("shoes-a", "shoes-b", "shoes-c", "shoes-d", "shoes-e")
BENCH_P = 0.8
ROW_CLEAR = 3
RUG_MIN = 40
OUT = ft.REPORT / "owner-circles-2" / "work" / "shoes"
BASE_REF = "992fcfd"


def lengths(cat):
    sp = ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")
    out = {}
    for it in cat["items"]:
        if it["zone"].startswith("shoes"):
            try:
                out[it["id"]] = float((sp.get(it["id"]) or {}).get("length_cm") or "")
            except ValueError:
                out[it["id"]] = None
    return out


def vis_w_native(it):
    (_, _), (ax0, ay0, ax1, ay1) = native(it)
    return ax1 - ax0 + 1


def k_cm(cat, base, Ls, Lmed):
    """Pick K so the median low (non-boot) rug pair keeps its 992fcfd visible width."""
    import subprocess
    widths = []
    for it in base["items"]:
        if not it["zone"].startswith("shoes"):
            continue
        L_ = it["layer"]
        rug_like = (L_["top"] + L_["height"]) / 100 * H > 400
        if not rug_like:
            continue
        x, y, a = cs.placed_alpha(it)
        ys, xs = np.where(a > 24)
        vw, vh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if vh / vw < 1.15:  # low shoes only
            widths.append(vw / (Ls.get(it["id"]) or Lmed))
    return float(np.median(widths))


def plan(cat, K, Ls, Lmed):
    zones = {z["id"]: z for z in cat["zones"]}
    out, notes = {}, {}
    for zid in ROOMS:
        rgb = cs.plate_rgb(zones[zid])
        rug_y = cs.rug_top_y(rgb)
        shelves, posts, _ = cs.furniture(rgb, rug_y)
        wood = cs.wood_mask(rgb)
        items = [it for it in cat["items"] if it["zone"] == zid]
        rug = [it for it in items if (it["layer"]["top"] + it["layer"]["height"]) / 100 * H > rug_y]
        bench = [it for it in items if it not in rug]
        placed = {}
        by_shelf = {}
        for it in bench:
            px, py, a = cs.placed_alpha(it)
            bottom, bx0, bx1 = cs.alpha_bottom(px, py, a)
            by_shelf.setdefault(cs.nearest_shelf(bottom, bx0, bx1, shelves)[1:], []).append(it)
        for (sy, sx0, sx1), its in by_shelf.items():
            lo, hi = sx0 + SHELF_PAD, sx1 - SHELF_PAD
            for p0, p1, py0, py1 in posts:
                if py0 < sy < py1 and p0 < hi and p1 > lo:
                    if p0 > (lo + hi) / 2:
                        hi = p0 - SHELF_PAD
                    else:
                        lo = p1 + SHELF_PAD
            frac = wood[:, int(lo):int(hi)].mean(1)
            rail_rows = [y for y in range(0, sy - 15) if frac[y] > 0.6]
            ceiling = (max(rail_rows) + RAIL_CLEAR) if rail_rows else 40
            its.sort(key=lambda i: box_px(i)[0])
            n = len(its)
            for k, it in enumerate(its):
                (iw, ih), (ax0, ay0, ax1, ay1) = native(it)
                L = Ls.get(it["id"]) or Lmed
                want = BENCH_P * K * L
                s = want / (ax1 - ax0 + 1)
                cap = min((hi - lo) / n * 0.9 / (ax1 - ax0 + 1), (sy + BENCH_SINK - ceiling) / (ay1 - ay0 + 1))
                capped = s > cap
                s = min(s, cap)
                cx = lo + (hi - lo) * (k + 0.5) / n
                placed[it["id"]] = ((cx - (ax0 + ax1 + 1) / 2 * s), sy + BENCH_SINK - (ay1 + 1) * s, iw * s, ih * s,
                                    dict(kind="bench", L=L, L_known=bool(Ls.get(it["id"])), want_w=round(want), capped=capped))
        # rug rows (back -> front) with their spans and counts
        rows = {}
        for it in rug:
            x, y, w, h = box_px(it)
            rows.setdefault(round(y + h), []).append(it)
        merged = []
        for b in sorted(rows):
            if merged and b - merged[-1][0] <= 8:
                merged[-1][1].extend(rows[b])
            else:
                merged.append([b, list(rows[b])])
        counts = [len(r) for _, r in merged]
        spans = [(min(box_px(i)[0] for i in r), max(box_px(i)[0] + box_px(i)[2] for i in r)) for _, r in merged]
        front_base = merged[-1][0]
        sized, padb = {}, {}
        for it in rug:
            (iw, ih), (ax0, ay0, ax1, ay1) = native(it)
            L = Ls.get(it["id"]) or Lmed
            s = K * L / (ax1 - ax0 + 1)
            sized[it["id"]] = (iw * s, ih * s, s, L)
            padb[it["id"]] = (ih - ay1 - 1) * s   # transparent rows under the sole
        class Infeasible(Exception):
            pass

        def attempt(order):
            got = {}
            assign, k0 = [], 0
            for c in counts:
                assign.append(order[k0:k0 + c]); k0 += c
            # baselines from the front
            bases = [None] * len(assign)
            bases[-1] = front_base
            bases[-1] = front_base - max(padb[i["id"]] for i in assign[-1])
            for r in range(len(assign) - 1, 0, -1):
                top_r = min(bases[r] + padb[i["id"]] - sized[i["id"]][1] for i in assign[r])
                bases[r - 1] = top_r - ROW_CLEAR - max(padb[i["id"]] for i in assign[r - 1])
            note = dict(rug_y=rug_y, bases=[round(b) for b in bases], counts=counts)
            if bases[0] < rug_y + RUG_MIN:
                raise Infeasible(f"{zid}: back row baseline {bases[0]:.0f} is not on the rug (rug top {rug_y})")
            benches = [(bx, by, bw, bh) for bx, by, bw, bh, bm in placed.values()]
            import itertools

            def lay(its, r, s0, s1):
                tot = sum(sized[i["id"]][0] for i in its)
                gap = (s1 - s0 - tot) / max(1, len(its) - 1)
                x = s0 if len(its) > 1 else s0 + (s1 - s0 - tot) / 2
                if gap < 12:
                    raise Infeasible(f"{zid} row {r}: pairs do not fit the row span (gap {gap:.1f})")
                res = {}
                for it in its:
                    w, h, s, L = sized[it["id"]]
                    res[it["id"]] = (x, bases[r] + padb[it["id"]] - h, w, h, dict(kind=f"rug{r}", L=L, L_known=bool(Ls.get(it["id"]))))
                    x += w + gap
                return res

            def dodge(its, r, s0, s1):
                """Greedy: a pair tall enough to reach a bench pair skips past that bench column."""
                cols = sorted((bx - 6, bx + bw + 6, by + bh) for bx, by, bw, bh in benches)
                x, res = s0, {}
                for it in its:
                    w, h, s, L = sized[it["id"]]
                    top = bases[r] + padb[it["id"]] - h
                    for c0, c1, cb in cols:
                        if top < cb and x < c1 and x + w > c0:
                            x = c1
                    res[it["id"]] = (x, top, w, h, dict(kind=f"rug{r}", L=L, L_known=bool(Ls.get(it["id"]))))
                    x += w + 12
                if x - 12 > s1:
                    return None
                # spread the leftover slack evenly over the gaps that are not bench-skips
                slack = s1 - (x - 12)
                ids = list(res)
                out, shift = {}, 0.0
                for k, iid in enumerate(ids):
                    x0, top, w, h, m = res[iid]
                    cand = x0 + shift
                    out[iid] = (cand, top, w, h, m)
                    if k < len(ids) - 1:
                        nxt = res[ids[k + 1]][0]
                        tight = abs(nxt - (x0 + w + 12)) < 1e-6
                        if tight:
                            shift += slack / max(1, len(ids) - 1)
                if hits_bench(out):
                    return None
                return out

            def hits_bench(res):
                return any(bx < x + w and bx + bw > x and by + bh > y
                           for x, y, w, h, _ in res.values() for bx, by, bw, bh in benches)

            for r, (its, (s0, s1)) in enumerate(zip(assign, spans)):
                its = sorted(its, key=lambda i: box_px(i)[0])
                res = lay(its, r, s0, s1)
                if r == 0 and hits_bench(res):
                    # reorder the back row so tall pairs stand clear of the bench / shelf pairs above
                    for perm in sorted(itertools.permutations(its),
                                       key=lambda p: sum(abs(a["id"] != b["id"]) for a, b in zip(p, its))):
                        res = lay(list(perm), r, s0, s1)
                        if not hits_bench(res):
                            break
                    else:
                        res = None
                        for perm in itertools.permutations(its):
                            res = dodge(list(perm), r, s0, s1)
                            if res:
                                break
                        if not res:
                            raise Infeasible(f"{zid}: back row cannot stand clear of the bench pairs")
                got.update(res)
            # back row must stay clear of bench boxes
            for iid, (x, y, w, h, m) in got.items():
                if m["kind"] != "rug0":
                    continue
                for bid, (bx, by, bw, bh, bm) in placed.items():
                    if bm["kind"] == "bench" and bx < x + w and bx + bw > x and by + bh > y:
                        raise Infeasible(f"{zid}: back-row {iid} reaches bench pair {bid}")
            return got, note

        strategies = [
            ("tallest pairs at the back", sorted(rug, key=lambda i: -sized[i["id"]][1])),
            ("tallest pairs at the front", sorted(rug, key=lambda i: sized[i["id"]][1])),
            ("rows as before", [i for _, r in merged for i in sorted(r, key=lambda i: box_px(i)[0])]),
        ]
        why = []
        for name, order in strategies:
            try:
                got, note = attempt(order)
                note["strategy"] = name
                notes[zid] = note
                placed.update(got)
                break
            except Infeasible as e:
                why.append(f"{name}: {e}")
        else:
            raise SystemExit(f"{zid}: no row order works -> " + " | ".join(why))
        out[zid] = placed
    return out, notes


def apply(cat, plans):
    c = deepcopy(cat)
    items = {i["id"]: i for i in c["items"]}
    for zid, placed in plans.items():
        z = next(z for z in c["zones"] if z["id"] == zid)
        for iid, (x, y, w, h, _) in placed.items():
            bx = {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}
            it = items[iid]
            it["layer"].update(bx, rect=dict(bx))
            it["hotspot"].update(bx)
            im = Image.open(ROOT / it["layer"]["image"]).convert("RGBA")
            rs = im.resize((max(1, round(w)), max(1, round(h))), Image.Resampling.LANCZOS)
            it["hotspot"]["clip"] = centre_safe_clip(np.array(rs)[:, :, 3])
        # paint order: back to front
        ids = sorted(z["items"], key=lambda i: items[i]["layer"]["top"] + items[i]["layer"]["height"])
        for zi, iid in enumerate(ids, 1):
            items[iid]["layer"]["zIndex"] = zi
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
    import subprocess
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    (OUT / "before").mkdir(parents=True, exist_ok=True)
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    base = json.loads(subprocess.check_output(["git", "show", f"{BASE_REF}:data/catalog.json"], cwd=ROOT))
    Ls = lengths(cur)
    Lmed = float(np.median([v for v in Ls.values() if v]))
    K = k_cm(cur, base, Ls, Lmed)
    print(f"K = {K:.3f} px per cm of shoe length; median L = {Lmed}")
    plans, notes = plan(cur, K, Ls, Lmed)
    new = apply(cur, plans)
    validate(cur, new)
    rep = {"K_px_per_cm": K, "L_median": Lmed, "rooms": notes, "items": {}}
    for zid, placed in plans.items():
        for iid, (x, y, w, h, m) in placed.items():
            rep["items"][iid] = dict(zone=zid, **m, box=[round(w), round(h)], top=round(y), bottom=round(y + h))
    (OUT / f"summary-{'write' if args.write else 'dry'}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(notes))
    if args.write:
        fd = os.open(ft.ROOT, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with TemporaryDirectory(prefix=".shoes2-", dir=ft.ROOT) as tmp:
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


if __name__ == "__main__":
    main()
