#!/usr/bin/env python3
"""jackets-a..f: rebuild garment layers from their item-card cutouts (owner pick, 2026-11-26).

Same treatment as jackets-g (f5aec6f): each layer = the item's card cutout (assets/items/v2/<id>-600.webp,
RGBA), alpha-trimmed, uniform scale only. Scale stays the jacket-room 2.5 px/cm of shoulder->hem body
length (garment_landmarks): researched body length -> 2.5 x cm; otherwise the item keeps its current
displayed body length (the rooms already read at ~2.5). Cards drawn without a hanger get the standard
wooden hanger + hook used in the jackets-f/g cards (sprite cut from batch2-015's card), placed behind the
garment at the neckline, at one constant real size. Plates/rods unchanged; items re-spread on their
current rods. Flagged cards (tools: report jackets-cards/flags.json) keep their current layer.
"""
from __future__ import annotations
import argparse, json, os, fcntl, sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_totes as ft
import fill_hanging as fh
import uniform_size as us
import garment_landmarks as gl
from repack_three_rails import save_webp
from shoes_fix import centre_safe_clip
from rebuild_covers import composite

us.CACHE = "20261126j"
W, H = 1536, 1024
S_L = 2.5
# round 2 (owner, 9:24 PM): jackets-c/d/e at 2.4 px/cm so the card versions fit their rods
S_ROOM = {"jackets-c": 2.4, "jackets-d": 2.4, "jackets-e": 2.4}
HANG, EDGE, GAP, RAIL_CLEAR, FLOOR_CLEAR = 5, 24, 28, 8, 14   # 8 px = the clearance jackets-a..e already use
BENCH = {"jackets-f": 771}
ROOMS = ["jackets-a", "jackets-b", "jackets-c", "jackets-d", "jackets-e", "jackets-f"]
REP = ft.REPORT / os.environ.get("JC_REPORT", "jackets-cards-2")
OUT = REP / "work"
HANGER_SRC = "batch2-015"
HANGERLESS_TOP = 0.15
CARD_OVERRIDE = REP / "card-override"   # <id>-600.png: re-matted / redrawn cards (also written as the item cards)         # top 6 % of the cutout wider than this fraction of the card = no hook


def specs():
    return ft.read_rows(ft.REPORT / "fill-all/specs.csv", "id")


def cm_of(sp, iid):
    r = sp.get(iid) or {}
    try:
        return float(r.get("height_cm") or "") if r.get("height_basis") == "body_length" else None
    except ValueError:
        return None


def trim(im):
    a = np.asarray(im)[..., 3]
    ys, xs = np.where(a > 8)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def card(item):
    ov = CARD_OVERRIDE / f"{item['id']}-600.png"
    if ov.exists():
        return Image.open(ov).convert("RGBA")
    src = max(item["card_images"], key=lambda c: c["height"])["src"]
    return Image.open(ft.ROOT / src).convert("RGBA")


def overrides():
    return sorted(p.name[:-8] for p in CARD_OVERRIDE.glob("*-600.png")) if CARD_OVERRIDE.exists() else []


def has_hook(im):
    a = np.asarray(im)[..., 3] > 60
    ys = np.where(a.any(1))[0]
    top = a[ys[0]: ys[0] + max(2, int(0.06 * (ys[-1] - ys[0])))]
    return top.any(0).sum() / a.shape[1] < HANGERLESS_TOP


def hanger_sprite():
    a = np.asarray(Image.open(ft.ROOT / f"assets/items/v2/{HANGER_SRC}-600.webp").convert("RGBA")).copy()
    s = a[:90].copy()
    r, g, b = (s[..., k].astype(int) for k in range(3))
    yy = np.arange(90)[:, None]
    keep = (yy < 52) | ((r > 150) & (g > 120) & (r > b + 30))
    s[..., 3] = np.where(keep, s[..., 3], 0)
    return trim(Image.fromarray(s))


def body_card(im):
    s, e = gl.landmarks_alpha(np.asarray(im)[..., 3] > 60)
    return (e - s) * im.height


def add_hanger(im, sprite, sprite_h):
    """Put the sprite (scaled to sprite_h card px) behind the garment, hook centred on the neckline."""
    sp = sprite.resize((max(1, round(sprite.width * sprite_h / sprite.height)), max(1, round(sprite_h))), Image.Resampling.LANCZOS)
    a = np.asarray(im)[..., 3] > 60
    ys = np.where(a.any(1))[0]
    band = a[ys[0]: ys[0] + max(2, int(0.03 * (ys[-1] - ys[0])))]
    xs = np.where(band.any(0))[0]
    cx = (xs.min() + xs.max()) / 2
    overlap = round(0.14 * sp.height)
    up = sp.height - overlap
    W2 = max(im.width, int(cx + sp.width / 2) + 1) - min(0, int(cx - sp.width / 2))
    ox = -min(0, int(cx - sp.width / 2))
    canvas = Image.new("RGBA", (W2, im.height + up), (0, 0, 0, 0))
    canvas.alpha_composite(sp, (round(cx - sp.width / 2) + ox, 0))
    canvas.alpha_composite(im, (ox, up))
    return trim(canvas)


def alpha_bottom(im):
    a = np.asarray(im)[..., 3] > 60
    return (np.where(a.any(1))[0][-1] + 1) / a.shape[0]


def layout(zid, bands, floor, entries, counts):
    """entries: iid -> dict(w, h(alpha height), order). Tallest go to the lowest rod (it has the most
    room); the rest keep their old rod order. Returns (rows, None) or (None, iid that does not fit)."""
    cap = [((bands[k + 1][0] - RAIL_CLEAR) if k + 1 < len(bands) else floor) - (bands[k][0] + HANG) for k in range(len(bands))]
    def check(rows):
        for k, row in enumerate(rows):
            for iid in sorted(row, key=lambda i: -entries[i]["h"]):
                if entries[iid]["h"] > cap[k] + 0.5:
                    return iid
            x0, x1 = bands[k][2] + EDGE, bands[k][3] - EDGE
            tot = sum(entries[i]["w"] for i in row)
            if len(row) > 1 and (x1 - x0 - tot) / (len(row) - 1) < GAP:
                return max(row, key=lambda i: entries[i]["w"])
        return None
    # A: every item stays on its rod
    a = [sorted([i for i in entries if entries[i]["order"][0] == k], key=lambda i: entries[i]["order"][1]) for k in range(len(bands))]
    bad_a = check(a)
    if not bad_a:
        return a, None
    # B: the tallest go to the lowest rod (most room), the rest keep their old rod order
    ids = sorted(entries, key=lambda i: -entries[i]["h"])
    nb = counts[-1]
    rest = sorted(ids[nb:], key=lambda i: entries[i]["order"])
    b, pos = [], 0
    for k in range(len(bands) - 1):
        b.append(rest[pos: pos + counts[k]]); pos += counts[k]
    b.append(sorted(ids[:nb], key=lambda i: entries[i]["order"][1]))
    bad_b = check(b)
    if not bad_b:
        return b, None
    return None, bad_b if entries[bad_b]["new"] else bad_a


def search_layout(bands, floor, entries, counts, olddim, max_rev=6):
    """Exhaustive: fewest reverted card layers, then fewest items moved off their rod. Rows keep old x order."""
    from itertools import combinations
    nb = len(bands)
    cap = [((bands[k + 1][0] - RAIL_CLEAR) if k + 1 < nb else floor) - (bands[k][0] + HANG) for k in range(nb)]
    span = [bands[k][3] - bands[k][2] - 2 * EDGE for k in range(nb)]
    def row_ok(row, dims, k):
        if any(dims[i]["h"] > cap[k] + 0.5 for i in row):
            return False
        tot = sum(dims[i]["w"] for i in row)
        return len(row) < 2 or (span[k] - tot) / (len(row) - 1) >= GAP
    ids = list(entries)
    news = [i for i in ids if i in olddim]
    for r in range(0, min(max_rev, len(news)) + 1):
        best = None
        for rev in combinations(news, r):
            dims = {i: (olddim[i] if i in rev else entries[i]) for i in ids}
            def rec(k, left, acc, moves):
                nonlocal best
                if k == nb - 1:
                    row = sorted(left, key=lambda i: entries[i]["order"])
                    if row_ok(row, dims, k):
                        mv = moves + sum(1 for i in row if entries[i]["order"][0] != k)
                        if best is None or mv < best[0]:
                            best = (mv, acc + [row], list(rev))
                    return
                for row in combinations(sorted(left, key=lambda i: entries[i]["order"]), counts[k]):
                    if row_ok(row, dims, k):
                        mv = moves + sum(1 for i in row if entries[i]["order"][0] != k)
                        if best is not None and mv >= best[0]:
                            continue
                        rec(k + 1, [i for i in left if i not in row], acc + [list(row)], mv)
            rec(0, ids, [], 0)
        if best:
            return best[1], best[2]
    return None


def plan(cat, flagged, rooms):
    items = {i["id"]: i for i in cat["items"]}
    sp = specs()
    sprite = hanger_sprite()
    # constant real hanger size: batch2-015's own card hanger at its room scale
    i015 = items[HANGER_SRC]; c015 = trim(card(i015))
    cm = cm_of(sp, HANGER_SRC)
    tgt015 = S_L * cm if cm else gl.body_px(i015)
    hanger_room_px = sprite.height * tgt015 / body_card(c015)
    new_imgs, info, placed = {}, {}, {}
    for zid in rooms:
        S = S_ROOM.get(zid, S_L); k_s = S / S_L
        z = next(z for z in cat["zones"] if z["id"] == zid)
        room = fh.load_room(z)
        bands = room.bg.bands
        floor = BENCH.get(zid, room.bg.obstacle) - FLOOR_CLEAR
        entries, cand, counts = {}, {}, [0] * len(bands)
        for iid in z["items"]:
            it = items[iid]; L = it["layer"]
            top = L["top"] / 100 * H
            k = min(range(len(bands)), key=lambda j: abs(bands[j][0] + HANG - top))
            counts[k] += 1
            old = Image.open(ft.ROOT / L["image"]).convert("RGBA")
            ow, oh = L["width"] / 100 * W * k_s, L["height"] / 100 * H * k_s   # kept layers: uniform S/2.5
            entries[iid] = dict(w=ow, h=oh * alpha_bottom(old), box_h=oh, order=(k, L["left"]), new=False)
            cmv = cm_of(sp, iid)
            tgt = S * cmv if cmv else gl.body_px(it) * k_s
            info[iid] = dict(zone=zid, scale=S, cm=cmv, body_target=round(tgt, 1), old_row=k, card_src="override" if (CARD_OVERRIDE / f"{iid}-600.png").exists() else "card")
            if iid in flagged:
                info[iid].update(kept=True, reason=flagged[iid]); continue
            im = trim(card(it))
            f = tgt / body_card(im)
            hook = has_hook(im)
            if not hook:
                im = add_hanger(im, sprite, hanger_room_px * k_s / f)
            cand[iid] = im
            entries[iid].update(w=im.width * f, h=im.height * f, box_h=im.height * f, new=True)
            info[iid].update(kept=False, hanger_added=not hook, factor=round(f, 4))
        rows0, _ = layout(zid, bands, floor, entries, counts)
        if not rows0:
            # round 2: search for the fewest card layers to keep at their current size (then the fewest rod moves)
            olddim = {}
            for iid in entries:
                if entries[iid]["new"]:
                    L = items[iid]["layer"]; old = Image.open(ft.ROOT / L["image"]).convert("RGBA")
                    oh = L["height"] / 100 * H * k_s
                    olddim[iid] = dict(w=L["width"] / 100 * W * k_s, h=oh * alpha_bottom(old), box_h=oh)
            res = search_layout(bands, floor, entries, counts, olddim)
            if res:
                rows_s, reverts = res
                for bad in reverts:
                    entries[bad].update(new=False, **olddim[bad]); cand.pop(bad)
                    info[bad].update(kept=True, reason=f"card version does not fit the room's rods at {S} px/cm (taller hanger/collar than the current layer)", card_h=info[bad].pop("factor", None))
        while True:
            rows, bad = layout(zid, bands, floor, entries, counts)
            if not rows and rows0 is None and res:
                rows = rows_s
            if rows:
                break
            if not entries[bad]["new"]:
                # a kept layer is in the way: give way by reverting the tallest card layer in the room
                news = [i for i in entries if entries[i]["new"]]
                assert news, f"{zid}: {bad} does not fit even with current layers"
                bad = max(news, key=lambda i: entries[i]["h"])
            # the card version is too long/wide for the rods at 2.5 px/cm: keep the current layer
            L = items[bad]["layer"]; old = Image.open(ft.ROOT / L["image"]).convert("RGBA")
            oh = L["height"] / 100 * H * k_s
            entries[bad].update(w=L["width"] / 100 * W * k_s, h=oh * alpha_bottom(old), box_h=oh, new=False)
            cand.pop(bad)
            info[bad].update(kept=True, reason=f"card version does not fit the room's rods at {S} px/cm (taller hanger/collar than the current layer)", card_h=info[bad].pop("factor", None))
        new_imgs.update(cand)
        for k, row in enumerate(rows):
            y0 = bands[k][0] + HANG
            x0, x1 = bands[k][2] + EDGE, bands[k][3] - EDGE
            tot = sum(entries[i]["w"] for i in row)
            gap = (x1 - x0 - tot) / (len(row) + 1); start = x0 + gap
            if gap < GAP:
                gap = (x1 - x0 - tot) / (len(row) - 1); start = x0
            x = start
            for iid in row:
                e = entries[iid]
                placed[iid] = (zid, x, y0, e["w"], e["box_h"])
                info[iid].update(row=k, gap=round(gap, 1), bottom_alpha=round(y0 + e["h"]))
                x += e["w"] + gap
    return placed, new_imgs, info


def apply(cat, placed, imgs):
    c = deepcopy(cat)
    items = {i["id"]: i for i in c["items"]}
    for iid, (zid, x, y, w, h) in placed.items():
        bx = {"left": x / W * 100, "top": y / H * 100, "width": w / W * 100, "height": h / H * 100}
        it = items[iid]
        it["layer"].update(bx, rect=dict(bx))
        it["hotspot"].update(bx)
        im = imgs.get(iid) or Image.open(ft.ROOT / it["layer"]["image"]).convert("RGBA")
        if iid in imgs:
            a = np.asarray(im)[..., 3]
            hx = float(np.where(a[: max(2, a.shape[0] // 100)] > 60)[1].mean()) / a.shape[1] * 100
            it["layer"].update(pivot={"x": round(hx, 4), "y": 0}, pivotX=round(hx, 4), pivotY=0)
        rs = im.resize((max(1, round(w)), max(1, round(h))), Image.Resampling.LANCZOS)
        it["hotspot"]["clip"] = centre_safe_clip(np.array(rs)[:, :, 3])
    return c


def validate(before, after, rooms):
    bi = {i["id"]: i for i in before["items"]}
    for it in after["items"]:
        o = bi[it["id"]]
        mutable = {"layer", "hotspot"} if o["zone"] in rooms else set()
        assert all(it[k] == o[k] for k in o.keys() - mutable), it["id"]
    for z0, z1 in zip(before["zones"], after["zones"]):
        assert z0["items"] == z1["items"] and z0["images"] == z1["images"], z0["id"]
        if z0["id"] not in rooms:
            assert z0 == z1, z0["id"]
    its = {i["id"]: i for i in after["items"]}
    for zid in rooms:
        z = next(z for z in after["zones"] if z["id"] == zid)
        bx = []
        for iid in z["items"]:
            r = its[iid]["layer"]["rect"]; x, y, w, h = r["left"], r["top"], r["width"], r["height"]
            assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100, iid
            for j, ax, ay, bx_, by in bx:
                assert not (min(bx_, x + w) - max(ax, x) > 1e-9 and min(by, y + h) - max(ay, y) > 1e-9), f"{iid} overlaps {j}"
            bx.append((iid, x, y, x + w, y + h))


def compose(cat, zid, imgs):
    z = next(z for z in cat["zones"] if z["id"] == zid)
    base = Image.open(ft.ROOT / max(z["images"], key=lambda i: i["width"])["src"]).convert("RGBA").resize((W, H))
    for it in sorted([i for i in cat["items"] if i["zone"] == zid], key=lambda i: i["layer"].get("zIndex", 0)):
        L = it["layer"]; w, h = max(1, round(L["width"] / 100 * W)), max(1, round(L["height"] / 100 * H))
        im = imgs.get(it["id"]) or Image.open(ft.ROOT / L["image"]).convert("RGBA")
        base.alpha_composite(im.resize((w, h), Image.Resampling.LANCZOS), (round(L["left"] / 100 * W), round(L["top"] / 100 * H)))
    return base.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--write", action="store_true")
    ap.add_argument("rooms", nargs="*", default=ROOMS)
    args = ap.parse_args()
    rooms = args.rooms
    flagged = json.loads((REP / "flags.json").read_text()) if (REP / "flags.json").exists() else {}
    cur = json.loads((ft.ROOT / ft.CATALOG).read_text(encoding="utf-8"))
    placed, imgs, info = plan(cur, flagged, rooms)
    new = apply(cur, placed, imgs)
    validate(cur, new, rooms)
    tag = "write" if args.write else "dry"
    (OUT / f"summary-{tag}-{'_'.join(r[-1] for r in rooms)}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    for zid in rooms:
        compose(new, zid, imgs).save(OUT / f"{zid}-{'after' if args.write else 'preview'}.png")
    print({z: (sum(1 for v in info.values() if v["zone"] == z and not v["kept"]), sum(1 for v in info.values() if v["zone"] == z and v["kept"])) for z in rooms})
    if not args.write:
        return
    fd = os.open(ft.ROOT, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with TemporaryDirectory(prefix=".jcards-", dir=ft.ROOT) as tmp:
            stage = Path(tmp); paths = []
            for iid, im in imgs.items():
                p = Path(f"assets/zones/layers/{iid}.webp"); save_webp(im, stage / p, quality=92); paths.append(p)
            its = {i["id"]: i for i in new["items"]}
            for iid in overrides():
                if its[iid]["zone"] not in rooms:
                    continue
                src = Image.open(CARD_OVERRIDE / f"{iid}-600.png").convert("RGBA")
                for c in its[iid]["card_images"]:
                    assert c["height"] in (600, 400), (iid, c)
                    im = src if c["height"] == 600 else src.resize((c["width"], c["height"]), Image.Resampling.LANCZOS)
                    assert im.size == (c["width"], c["height"]), (iid, im.size, c)
                    p = Path(c["src"]); save_webp(im, stage / p, quality=92); paths.append(p)
            for zid in rooms:
                z = next(z for z in new["zones"] if z["id"] == zid)
                img = compose(new, zid, imgs)
                cover = Path(z["cover"]["src"])
                save_webp(img.resize((z["cover"]["width"], z["cover"]["height"]), Image.Resampling.LANCZOS), stage / cover); paths.append(cover)
            cache = us.stage_cache(stage)
            (stage / ft.CATALOG.parent).mkdir(exist_ok=True)
            ft.dump(stage / ft.CATALOG, new)
            ft.publish(stage, [*paths, ft.CATALOG, *cache])
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
