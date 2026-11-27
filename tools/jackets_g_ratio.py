#!/usr/bin/env python3
"""jackets-g length/width fix (owner feedback on 9d86c9e, 2026-11-26).

Root cause: jackets-g's 8 coat layers were cut (b03fdaf -> c88d89c/3b588e5) out of the generated room
painting, which redrew every coat longer and narrower than the item (and 079 in the wrong colour).
Their displayed shoulder->hem / chest-width ratio reads 2.6-3.1; the same measure on the item's own
card cutout (assets/items/v2/<id>-600.webp, already RGBA) reads 2.0-2.3. Layer boxes keep the native
aspect (no CSS/object-fit stretch) and no sleeve/edge is cropped.
Fix: each layer is rebuilt from its card cutout (uniform scale only, hanger kept), and the room goes
back to the c0abd72 two-rod plate. One room scale S (px per cm of body length) is the largest that fits
both rows; unknown lengths use the median known length.
"""
from __future__ import annotations
import argparse, json, os, fcntl, sys, shutil, subprocess
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_totes as ft
import uniform_size as us
from repack_three_rails import save_webp
from shoes_fix import centre_safe_clip
import coat_ratio as R

us.CACHE = "20261126h"
W, H = 1536, 1024
ZID = "jackets-g"
BASE = "c0abd72"
RODS = ((48, 61), (392, 404))       # c0abd72 plate rails (fill_hanging.detect_rails)
SPAN = (202, 1404)
HANG, EDGE, GAP, RAIL_CLEAR = 5, 24, 28, 12
BENCH_TOP, FLOOR_CLEAR = 771, 14
OUT = ft.REPORT / "jackets-g-ratio" / "work"


def specs():
    return ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")


def cm_of(sp, iid):
    r = sp.get(iid) or {}
    try:
        return float(r.get("height_cm") or "") if r.get("height_basis") == "body_length" else None
    except ValueError:
        return None


def card_layer(item):
    im = Image.open(ft.ROOT / item["card_images"][0]["src"]).convert("RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > 8)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def plan(cat):
    items = {i["id"]: i for i in cat["items"]}
    z = next(z for z in cat["zones"] if z["id"] == ZID)
    sp = specs()
    imgs, info = {}, {}
    tmp = OUT / "cards"; tmp.mkdir(exist_ok=True)
    for iid in z["items"]:
        im = card_layer(items[iid]); imgs[iid] = im
        p = tmp / f"{iid}.png"; im.save(p)
        m = R.measure(str(p)); cm = cm_of(sp, iid)
        info[iid] = dict(cm=cm, m=m, aspect=im.width / im.height, frac=m["L"] / im.height)
    known = [v["cm"] for v in info.values() if v["cm"]]
    med = float(np.median(known))
    for v in info.values():
        v["cm_used"] = v["cm"] or med
        v["h_per_S"] = v["cm_used"] / v["frac"]          # layer px height per (px/cm)
    order = sorted(info, key=lambda i: -info[i]["h_per_S"])
    low, top = order[:4], order[4:]
    lim_top = RODS[1][0] - RAIL_CLEAR - (RODS[0][0] + HANG)
    lim_low = BENCH_TOP - FLOOR_CLEAR - (RODS[1][0] + HANG)
    S = min(lim_top / max(info[i]["h_per_S"] for i in top), lim_low / max(info[i]["h_per_S"] for i in low))
    S = float(np.floor(S * 100) / 100)
    placed = {}
    cur_x = {i: items[i]["layer"]["left"] for i in z["items"]}
    for rod, ids in ((RODS[0], top), (RODS[1], low)):
        ids = sorted(ids, key=lambda i: cur_x[i])
        row = [(i, info[i]["h_per_S"] * S * info[i]["aspect"], info[i]["h_per_S"] * S) for i in ids]
        x0, x1 = SPAN[0] + EDGE, SPAN[1] - EDGE
        tot = sum(w for _, w, _ in row); gap = (x1 - x0 - tot) / (len(row) + 1)
        assert gap >= GAP, gap
        x = x0 + gap
        for iid, w, h in row:
            placed[iid] = (x, rod[0] + HANG, w, h); x += w + gap
    return placed, imgs, info, S, med


def apply(cat, placed, imgs):
    c = deepcopy(cat)
    items = {i["id"]: i for i in c["items"]}
    z = next(z for z in c["zones"] if z["id"] == ZID)
    base = json.loads(subprocess.check_output(["git", "show", f"{BASE}:data/catalog.json"], cwd=ft.ROOT))
    z["images"] = deepcopy(next(zz for zz in base["zones"] if zz["id"] == ZID)["images"])
    ids = sorted(placed, key=lambda i: (placed[i][1], placed[i][0]))
    z["items"] = ids
    for zi, iid in enumerate(ids, 1):
        x, y, w, h = placed[iid]
        bx = {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}
        a = np.asarray(imgs[iid])[..., 3]
        hook_x = float(np.where(a[: max(2, a.shape[0] // 100)] > 60)[1].mean()) / a.shape[1] * 100
        it = items[iid]
        it["layer"].update(bx, rect=dict(bx), zIndex=zi, pivot={"x": round(hook_x, 4), "y": 0}, pivotX=round(hook_x, 4), pivotY=0)
        it["hotspot"].update(bx)
        rs = imgs[iid].resize((max(1, round(w)), max(1, round(h))), Image.Resampling.LANCZOS)
        it["hotspot"]["clip"] = centre_safe_clip(np.array(rs)[:, :, 3])
    return c


def validate(before, after):
    bi = {i["id"]: i for i in before["items"]}
    assert len(before["items"]) == len(after["items"])
    for it in after["items"]:
        o = bi[it["id"]]
        mutable = {"layer", "hotspot"} if o["zone"] == ZID else set()
        assert all(it[k] == o[k] for k in o.keys() - mutable), it["id"]
    for z0, z1 in zip(before["zones"], after["zones"]):
        if z0["id"] != ZID:
            assert z0 == z1, z0["id"]
    z = next(z for z in after["zones"] if z["id"] == ZID)
    its = {i["id"]: i for i in after["items"]}
    bx = []
    for iid in z["items"]:
        r = its[iid]["layer"]["rect"]; x, y, w, h = r["left"], r["top"], r["width"], r["height"]
        assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100, iid
        for j, ax, ay, bx_, by in bx:
            assert not (min(bx_, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{iid} overlaps {j}"
        bx.append((iid, x, y, x + w, y + h))


def compose(plate, cat, imgs):
    base = plate.convert("RGBA")
    its = sorted([i for i in cat["items"] if i["zone"] == ZID], key=lambda i: i["layer"]["zIndex"])
    for it in its:
        L = it["layer"]; w, h = max(1, round(L["width"] / 100 * W)), max(1, round(L["height"] / 100 * H))
        base.alpha_composite(imgs[it["id"]].resize((w, h), Image.Resampling.LANCZOS), (round(L["left"] / 100 * W), round(L["top"] / 100 * H)))
    return base.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    placed, imgs, info, S, med = plan(cur)
    new = apply(cur, placed, imgs)
    validate(cur, new)
    plates = {s: subprocess.check_output(["git", "show", f"{BASE}:assets/zones/{ZID}-{s}.webp"], cwd=ft.ROOT) for s in (1600, 1024, 640)}
    tmpd = Path(OUT / "plate"); tmpd.mkdir(exist_ok=True)
    (tmpd / "1600.webp").write_bytes(plates[1600])
    plate = Image.open(tmpd / "1600.webp").convert("RGB").resize((W, H))
    img = compose(plate, new, imgs)
    rep = dict(S_px_per_cm=S, median_cm=med, items={})
    for iid, (x, y, w, h) in placed.items():
        v = info[iid]
        rep["items"][iid] = dict(cm=v["cm"], cm_used=v["cm_used"], ratio_card=v["m"]["ratio"], x=round(x), top=round(y), bottom=round(y + h), w=round(w), h=round(h))
        print(iid, rep["items"][iid])
    print("S", S, "median cm", med)
    (OUT / f"summary-{'write' if args.write else 'dry'}.json").write_text(json.dumps(rep, indent=1))
    img.save(OUT / f"jackets-g-{'after' if args.write else 'preview'}.png")
    if not args.write:
        return
    fd = os.open(ft.ROOT, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with TemporaryDirectory(prefix=".jgr-", dir=ft.ROOT) as tmp:
            stage = Path(tmp); paths = []
            for s, b in plates.items():
                p = Path(f"assets/zones/{ZID}-{s}.webp"); (stage / p).parent.mkdir(parents=True, exist_ok=True)
                (stage / p).write_bytes(b); paths.append(p)
            for iid, im in imgs.items():
                p = Path(f"assets/zones/layers/{iid}.webp"); save_webp(im, stage / p, quality=92); paths.append(p)
            z = next(z for z in new["zones"] if z["id"] == ZID)
            cover = Path(z["cover"]["src"]); hh = round(640 * H / W)
            save_webp(img.resize((640, hh), Image.Resampling.LANCZOS), stage / cover)
            z["cover"] = {"src": str(cover), "width": 640, "height": hh}; paths.append(cover)
            cache = us.stage_cache(stage)
            (stage / ft.CATALOG.parent).mkdir(exist_ok=True)
            ft.dump(stage / ft.CATALOG, new)
            ft.publish(stage, [*paths, ft.CATALOG, *cache])
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
