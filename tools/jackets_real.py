#!/usr/bin/env python3
"""Owner circles round 2 (2026-11-26): jackets-f / jackets-g at one real px/cm.

Scale is measured on visible garment landmarks (tools/garment_landmarks.py): high shoulder point to
hem, i.e. what a published body length (着丈, hood excluded) measures. Hanger, hood and collar above the
shoulder are not counted, so a hooded jacket is no longer shrunk for its hood.
S_L = 2.5 px/cm in both rooms: the scale jackets-f's five measured garments already read at (2.42-2.56,
median 2.50), so the circled two are corrected to their neighbours; jackets-a..e's uniform ~186 px
body reads as a ~74 cm jacket at this scale.

jackets-g: two rows of 117-121 cm coats need ~850 px of hanging height at 2.6 px/cm; the wall between
the ceiling line and the bench has ~720 px, so a redrawn two-rod room cannot hold them without the hems
reaching the bench/floor. It becomes a single long-hang rod (lower rod erased, top rod re-painted from
the room's own rod sprite). A single rod holds 8 coats with >=28 px gaps; today-240 (the shortest,
unknown-length black frog-button coat) moves to jackets-f, whose two rods have room for it.
"""
from __future__ import annotations
import argparse, json, os, fcntl, sys, dataclasses
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_totes as ft
import fill_hanging as fh
import uniform_size as us
import garment_landmarks as gl
from rebuild_covers import composite
from repack_three_rails import save_webp
from repack_three_rails import variants_from

us.CACHE = "20261126g"
W, H = 1536, 1024
S_L = 2.5
G_SPAN = (170, 1490)
HANG = 5
EDGE, GAP, RAIL_CLEAR = 24, 28, 12
BENCH_TOP = 771
FLOOR_CLEAR = 14
G_ROD_TOP = 180
MOVE = {"today-240": "jackets-f"}
OUT = ft.REPORT / "owner-circles-2" / "work" / "jackets"


def specs():
    return ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")


def cm_of(sp, iid):
    r = sp.get(iid) or {}
    try:
        return float(r.get("height_cm") or "") if r.get("height_basis") == "body_length" else None
    except ValueError:
        return None


def box(it):
    L = it["layer"]
    return L["left"] / 100 * W, L["top"] / 100 * H, L["width"] / 100 * W, L["height"] / 100 * H


def factors(cat):
    sp = specs()
    items = {i["id"]: i for i in cat["items"]}
    out = {}
    for zid in ("jackets-f", "jackets-g"):
        z = next(z for z in cat["zones"] if z["id"] == zid)
        found = {}
        for iid in z["items"]:
            cm = cm_of(sp, iid)
            if cm:
                found[iid] = (S_L * cm / gl.body_px(items[iid]), cm)
        med = float(np.median([f for f, _ in found.values()]))
        for iid in z["items"]:
            out[iid] = dict(f=found[iid][0] if iid in found else med, cm=found.get(iid, (None, None))[1],
                            body_before=gl.body_px(items[iid]), from_room=zid)
    return out


def rails_of(zid):
    rgb = np.asarray(Image.open(ft.ROOT / f"assets/zones/{zid}-1600.webp").convert("RGB").resize((W, H)))
    return fh.detect_rails(rgb)


def spread(row, x0, x1):
    """Even spacing with margins at both ends; if that is tighter than GAP, use the full span (ends at EDGE)."""
    tot = sum(w for _, w, _ in row)
    gap = (x1 - x0 - tot) / (len(row) + 1)
    start = x0 + gap
    if gap < GAP:
        gap = (x1 - x0 - tot) / (len(row) - 1)
        start = x0
    assert gap >= GAP, f"row too wide: gap {gap:.1f}"
    x, out = start, []
    for iid, w, h in row:
        out.append((iid, x, w, h))
        x += w + gap
    return out


def plan(cat):
    fs = factors(cat)
    items = {i["id"]: i for i in cat["items"]}
    sized = {iid: (box(items[iid])[2] * v["f"], box(items[iid])[3] * v["f"]) for iid, v in fs.items()}
    order = {iid: (box(items[iid])[1] > 300, box(items[iid])[0]) for iid in fs}
    placed = {}
    # jackets-f: two painted rods, 9 garments -> longest 4 on the lower rod
    r = rails_of("jackets-f")
    (t0, _, a0, b0), (t1, _, a1, b1) = r[0], r[1]
    members = [i for i in fs if fs[i]["from_room"] == "jackets-f" or MOVE.get(i) == "jackets-f"]
    by_h = sorted(members, key=lambda i: -sized[i][1])
    low, top = by_h[:4], by_h[4:]
    for rod_top, ids, x0, x1, limit in ((t0, top, a0, b0, t1 - RAIL_CLEAR), (t1, low, a1, b1, BENCH_TOP - FLOOR_CLEAR)):
        ids = sorted(ids, key=lambda i: order[i])
        for iid, x, w, h in spread([(i, *sized[i]) for i in ids], x0 + EDGE, x1 - EDGE):
            y = rod_top + HANG
            assert y + h <= limit, f"{iid} too long for rod at {rod_top}: {y + h:.0f} > {limit}"
            placed[iid] = ("jackets-f", x, y, w, h)
    # jackets-g: single rod
    g = [i for i in fs if fs[i]["from_room"] == "jackets-g" and i not in MOVE]
    rg = rails_of("jackets-g")
    a, b = G_SPAN
    for iid, x, w, h in spread([(i, *sized[i]) for i in sorted(g, key=lambda i: order[i])], a + EDGE, b - EDGE):
        y = G_ROD_TOP + HANG
        assert y + h <= BENCH_TOP - FLOOR_CLEAR, f"{iid} reaches the bench"
        placed[iid] = ("jackets-g", x, y, w, h)
    return placed, fs


def apply(cat, placed):
    c = deepcopy(cat)
    items = {i["id"]: i for i in c["items"]}
    for zid in ("jackets-f", "jackets-g"):
        z = next(z for z in c["zones"] if z["id"] == zid)
        ids = [i for i, p in placed.items() if p[0] == zid]
        ids.sort(key=lambda i: (placed[i][2], placed[i][1]))
        z["items"] = ids
        for zi, iid in enumerate(ids, 1):
            _, x, y, w, h = placed[iid]
            bx = {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}
            it = items[iid]
            it["zone"] = zid
            it["layer"].update(bx, rect=dict(bx), zIndex=zi)
            it["hotspot"].update(bx)
    return c


def validate(before, after):
    bi = {i["id"]: i for i in before["items"]}
    assert len(before["items"]) == len(after["items"])
    for it in after["items"]:
        o = bi[it["id"]]
        mutable = {"layer", "hotspot", "zone"} if o["zone"] in ("jackets-f", "jackets-g") else set()
        assert all(it[k] == o[k] for k in o.keys() - mutable), it["id"]
    for zid in ("jackets-f", "jackets-g"):
        z = next(z for z in after["zones"] if z["id"] == zid)
        bx = []
        for iid in z["items"]:
            it = next(i for i in after["items"] if i["id"] == iid)
            assert it["zone"] == zid
            r = it["layer"]["rect"]
            x, y, w, h = r["left"], r["top"], r["width"], r["height"]
            assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100, iid
            for j, ax, ay, bx_, by in bx:
                assert not (min(bx_, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{iid} overlaps {j}"
            bx.append((iid, x, y, x + w, y + h))


def g_plate(cat):
    zones = {z["id"]: z for z in cat["zones"]}
    src = fh.load_room(zones["jackets-g"])          # rod sprite cut from the room's own painted rod
    room = ft.Room(dataclasses.replace(src.bg, span=G_SPAN), src.benches)
    rooms = {"jackets-g": room, "__src__": src}
    plans = {"jackets-g": (1.0, SimpleNamespace(tops=[G_ROD_TOP]), {})}
    return fh.render_for({"rail_source": "__src__"}, rooms, plans, ["jackets-g"])["jackets-g"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    (OUT / "before").mkdir(parents=True, exist_ok=True)
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    for zid in ("jackets-f", "jackets-g"):
        p = OUT / "before" / f"{zid}.png"
        if not p.exists():
            composite(cur, zid)[0].save(p)
    placed, fs = plan(cur)
    new = apply(cur, placed)
    validate(cur, new)
    plate = g_plate(cur)
    items = {i["id"]: i for i in new["items"]}
    rep = []
    for iid, (zid, x, y, w, h) in placed.items():
        v = fs[iid]
        rep.append(dict(id=iid, room=zid, from_room=v["from_room"], cm=v["cm"], factor=round(v["f"], 3),
                        body_px_before=round(v["body_before"]), body_px_after=round(gl.body_px(items[iid])),
                        px_per_cm=round(gl.body_px(items[iid]) / v["cm"], 2) if v["cm"] else None,
                        top=round(y), bottom=round(y + h), x=round(x), w=round(w)))
        print(rep[-1])
    (OUT / f"summary-{'write' if args.write else 'dry'}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if args.write:
        fd = os.open(ft.ROOT, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with TemporaryDirectory(prefix=".jackets-", dir=ft.ROOT) as tmp:
                stage = Path(tmp); paths = []
                zg = next(z for z in new["zones"] if z["id"] == "jackets-g")
                variants = variants_from(plate, str(stage / "assets/zones/jackets-g"), (1600, 1024, 640))
                for v in variants:
                    v["src"] = str(Path(v["src"]).relative_to(stage)); paths.append(Path(v["src"]))
                zg["images"] = variants
                for zid in ("jackets-f", "jackets-g"):
                    z = next(z for z in new["zones"] if z["id"] == zid)
                    zz = {**z, "images": [{**vv, "src": str(stage / vv["src"]) if (stage / vv["src"]).exists() else vv["src"]} for vv in z["images"]]}
                    img, _ = composite({"zones": [zz], "items": new["items"]}, zid)
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
    plate_path = OUT / "jackets-g-plate.png"
    plate.save(plate_path)
    for zid in ("jackets-f", "jackets-g"):
        z = next(z for z in new["zones"] if z["id"] == zid)
        if zid == "jackets-g" and not args.write:
            z = {**z, "images": [{"src": str(plate_path), "width": W, "height": H}]}
        img, _ = composite({"zones": [z], "items": new["items"]}, zid)
        img.save(OUT / f"{zid}-{'after' if args.write else 'preview'}.png")


if __name__ == "__main__":
    main()
