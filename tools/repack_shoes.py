#!/usr/bin/env python3
"""Pack shoes onto empty-rug plates, three rows, ~0.6 scale.

Does not generate new paintings. Cleans leftover rug-fringe pixels on
existing cutouts (edge-connected, low-alpha, rug-colored only), then
places items on finished empty rugs. Motion stays lift; fallback is
cleared once items sit on an empty plate.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repack_three_rails as R  # noqa: E402

CATALOG = ROOT / "data/catalog.json"
DISPLAY = ROOT / "data/source/display"
GEN = ROOT / "data/source/generated"
LAYER_DIR = ROOT / "assets/zones/layers"

DEFAULT_SCALE = 0.60
MIN_GAP = 16
X_INSET = 36
BOTTOM_PAD = 14
TOP_PAD = 12
TARGET = 13

RUG_FILES = {L: next(GEN.glob(f"shoes-zone-{L}-rug-*.webp")) for L in string.ascii_lowercase if list(GEN.glob(f"shoes-zone-{L}-rug-*.webp"))}

# Warm beige of the generated rugs (not leather, not white wall).
def rug_mask(rgb):
    r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    return (
        (r > 150)
        & (r < 230)
        & (g > 130)
        & (g < 215)
        & (b > 100)
        & (b < 190)
        & (np.abs(r - g) < 45)
        & ((r - b) > 8)
    )


def detect_rug(plate):
    """Return rug_top and a function y -> (x0, x1) usable span."""
    h, w = plate.shape[:2]
    m = rug_mask(plate)
    row = m.mean(1)
    y_top = None
    for y in range(200, h):
        if row[y] > 0.40 and row[min(h - 1, y + 8) : min(h, y + 24)].mean() > 0.40:
            y_top = y
            break
    if y_top is None:
        ys = np.where(row > 0.35)[0]
        y_top = int(ys[0]) if len(ys) else int(h * 0.4)

    spans = []
    for y in range(h):
        xs = np.where(m[y])[0]
        if len(xs) > 20:
            spans.append((int(xs[0]), int(xs[-1])))
        else:
            spans.append(None)
    # fill gaps downward from first known, then upward
    last = (X_INSET, w - 1 - X_INSET)
    filled = [None] * h
    for y in range(h):
        if spans[y]:
            last = spans[y]
        filled[y] = last
    last = filled[-1]
    for y in range(h - 1, -1, -1):
        if spans[y]:
            last = spans[y]
        if filled[y] is None or (spans[y] is None and y < y_top):
            filled[y] = last

    def span_at(y):
        y = int(np.clip(y, 0, h - 1))
        x0, x1 = filled[y]
        return max(0, x0 + X_INSET), min(w - 1, x1 - X_INSET)

    return y_top, span_at


def is_rug_colored(px):
    r, g, b, a = int(px[0]), int(px[1]), int(px[2]), int(px[3])
    if a <= 0:
        return False
    return (
        r > 145
        and r < 235
        and g > 120
        and g < 220
        and b > 95
        and b < 195
        and abs(r - g) < 48
        and (r - b) > 6
        and (r + g + b) / 3 > 140
    )


def clean_ghost(arr):
    """Erase edge-connected low-alpha rug fringe; zero RGB of transparent pixels.

    Leaves opaque shoe pixels (including tan leather) alone. Only pixels
    with alpha < 96 that match rug color and touch transparency / the
    image border are cleared, plus a 1px almost-transparent dust ring.
    """
    out = arr.copy()
    h, w = out.shape[:2]
    a = out[:, :, 3]
    # dust
    out[a < 12] = 0
    a = out[:, :, 3]
    trans = a < 12
    # candidates: low-alpha rug-colored, or mid-alpha but only if almost at the edge
    cand = np.zeros((h, w), dtype=bool)
    for y in range(h):
        row = out[y]
        for x in range(w):
            if trans[y, x]:
                continue
            if row[x, 3] < 96 and is_rug_colored(row[x]):
                cand[y, x] = True
    # flood from transparent-adjacent candidates and image border
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    # any candidate next to transparent or at border
    ys, xs = np.where(cand)
    for y, x in zip(ys, xs):
        if y == 0 or x == 0 or y == h - 1 or x == w - 1:
            q.append((y, x))
            seen[y, x] = True
            continue
        if trans[y - 1, x] or trans[y + 1, x] or trans[y, x - 1] or trans[y, x + 1]:
            q.append((y, x))
            seen[y, x] = True
    while q:
        y, x = q.popleft()
        out[y, x] = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    trans = out[:, :, 3] == 0
    out[trans, :3] = 0
    return out


def pack_row(row, x0, x1, baseline, scale):
    widths = [it["w"] * scale for it in row]
    heights = [it["h"] * scale for it in row]
    usable = x1 - x0
    remain = usable - sum(widths)
    n = len(row)
    gap = remain / (n + 1) if n else 0
    x = x0 + gap
    placed = []
    for it, ww, hh in zip(row, widths, heights):
        placed.append(
            dict(
                rec=it,
                left=x,
                top=baseline - hh,
                scale=scale,
                ww=ww,
                hh=hh,
                gap=gap,
            )
        )
        x += ww + gap
    return placed, gap


def apply(scale=DEFAULT_SCALE, plates=None, preview_dir=None, write=False, clean=True):
    catalog = json.loads(CATALOG.read_text())
    items = [i for i in catalog["items"] if i["category"] == "shoes"]
    old_zones = []
    seen = set()
    for i in items:
        if i["zone"] not in seen:
            old_zones.append(i["zone"])
            seen.add(i["zone"])

    recs = []
    cleaned_ids = []
    for item in items:
        path = ROOT / item["layer"]["image"]
        arr = np.array(Image.open(path).convert("RGBA"))
        if clean:
            cleaned = clean_ghost(arr)
            if not np.array_equal(cleaned, arr):
                cleaned_ids.append(item["id"])
                arr = cleaned
                if write:
                    R.save_webp(Image.fromarray(arr), path)
        recs.append(
            dict(
                id=item["id"],
                item=item,
                arr=arr,
                w=arr.shape[1],
                h=arr.shape[0],
                src_zone=item["zone"],
                motion=item.get("layer", {}).get("motion") or "lift",
            )
        )

    counts = R.zone_counts(len(recs), target=TARGET, max_per=15, min_per=10)
    n_zones = len(counts)
    new_zone_ids = [f"shoes-{L}" for L in string.ascii_lowercase[:n_zones]]

    if plates:
        plate_letters = [p.strip()[-1] if "-" in p.strip() else p.strip() for p in plates]
    else:
        # largest usable rugs, skip short/narrow a and k
        plate_letters = list("bcdeig"[:n_zones])
    if len(plate_letters) < n_zones:
        raise SystemExit(f"need {n_zones} plates, got {plate_letters}")

    plate_imgs = []
    rug_meta = []
    for L, zid in zip(plate_letters, new_zone_ids):
        src = RUG_FILES[L]
        im = np.array(Image.open(src).convert("RGB"))
        y_top, span_at = detect_rug(im)
        plate_imgs.append(im)
        rug_meta.append(dict(letter=L, path=str(src.relative_to(ROOT)), y_top=y_top, span_at=span_at, h=im.shape[0], w=im.shape[1], src_id=f"shoes-{L}-rug"))

    zone_packs = []
    idx = 0
    for zi, k in enumerate(counts):
        chunk = recs[idx : idx + k]
        idx += k
        rc = R.rod_counts(k, rods=3)  # 5/4/4 or 4/4/4
        meta = rug_meta[zi]
        h, w = meta["h"], meta["w"]
        y_top = meta["y_top"]
        front_b = h - BOTTOM_PAD
        # first-pass pitch assuming median height
        med_h = float(np.median([r["h"] * scale for r in chunk]))
        back_b = y_top + TOP_PAD + med_h
        # three baselines equally spaced, but back not above back_b
        # Use equal spacing from a fitted back to front.
        # Ensure back row sits on the rug: back_b >= y_top + max_h_back (adjusted below).
        span = front_b - back_b
        if span < 80:
            raise SystemExit(f"{new_zone_ids[zi]} vertical room too tight: y_top={y_top} back={back_b} front={front_b}")
        mid_b = back_b + span / 2
        baselines = [back_b, mid_b, front_b]

        # Assign rows, moving oversized items to the front row.
        rows = []
        cursor = 0
        for n in rc:
            rows.append(chunk[cursor : cursor + n])
            cursor += n

        def pitch(i):
            if i == 0:
                return baselines[0] - y_top - TOP_PAD
            return baselines[i] - baselines[i - 1]

        # swap tall items forward
        for ri in (0, 1):
            room = pitch(ri) * 1.05
            for j, it in list(enumerate(rows[ri])):
                if it["h"] * scale > room:
                    # swap with shortest item on a later row that fits here
                    swapped = False
                    for rj in range(2, ri, -1):
                        short = min(rows[rj], key=lambda t: t["h"])
                        if short["h"] * scale <= room:
                            si = rows[rj].index(short)
                            rows[ri][j], rows[rj][si] = rows[rj][si], rows[ri][j]
                            swapped = True
                            break
                    if not swapped:
                        # keep on this row but shrink this item later
                        it["fit_scale"] = max(0.42, room / it["h"])

        # Recompute back baseline from actual back-row heights
        max_back = max((it.get("fit_scale", scale) * it["h"] for it in rows[0]), default=med_h)
        back_b = y_top + TOP_PAD + max_back
        if back_b + 40 >= front_b:
            raise SystemExit(f"{new_zone_ids[zi]} cannot fit back row (back_b={back_b})")
        mid_b = back_b + (front_b - back_b) / 2
        baselines = [back_b, mid_b, front_b]

        # Cap any item that would overflow into the row in front of it
        # (the two shoes-a cutouts are framed much larger than the rest).
        def row_pitch(i):
            if i == 0:
                return baselines[0] - y_top - 4
            return baselines[i] - baselines[i - 1] - 8
        for ri, row in enumerate(rows):
            room = max(120, row_pitch(ri))
            for it in row:
                cap = min(scale, room / it["h"])
                if cap < it.get("fit_scale", scale):
                    it["fit_scale"] = cap

        packed = []
        gaps = []
        for ri, row in enumerate(rows):
            if not row:
                gaps.append(None)
                continue
            baseline = baselines[ri]
            # usable x at mid-height of a typical item in this row
            typ_h = max(it.get("fit_scale", scale) * it["h"] for it in row)
            x0, x1 = meta["span_at"](baseline - typ_h * 0.35)
            # per-item scales
            # pack with mixed scales: place left-to-right using each item's scale
            widths, heights, scs = [], [], []
            for it in row:
                sc = it.get("fit_scale", scale)
                scs.append(sc)
                widths.append(it["w"] * sc)
                heights.append(it["h"] * sc)
            usable = x1 - x0
            remain = usable - sum(widths)
            n = len(row)
            gap = remain / (n + 1) if n else 0
            x = x0 + gap
            for it, ww, hh, sc in zip(row, widths, heights, scs):
                packed.append(
                    dict(
                        rec=it,
                        left=x,
                        top=baseline - hh,
                        scale=sc,
                        ww=ww,
                        hh=hh,
                        gap=gap,
                    )
                )
                x += ww + gap
            gaps.append(gap)

        if any(g is not None and g < MIN_GAP for g in gaps):
            raise SystemExit(f"{new_zone_ids[zi]} gaps too tight: {gaps}")

        # no item may start above the rug (except a few px)
        for p in packed:
            if p["top"] < y_top - 20:
                raise SystemExit(
                    f"{new_zone_ids[zi]} {p['rec']['id']} top {p['top']:.1f} above rug {y_top}"
                )

        composited = R.composite(plate_imgs[zi], packed)
        zone_packs.append(
            dict(
                zone_id=new_zone_ids[zi],
                src_plate=f"shoes-{plate_letters[zi]}-rug",
                plate=plate_imgs[zi],
                rug_top=y_top,
                baselines=baselines,
                placed=packed,
                composited=composited,
                gaps=gaps,
                row_counts=rc,
                item_ids=[p["rec"]["id"] for p in packed],
            )
        )

    # redirects
    item_to_new = {}
    for zp in zone_packs:
        for iid in zp["item_ids"]:
            item_to_new[iid] = zp["zone_id"]
    redirects = dict(catalog.get("redirects") or {})
    old_map = {}
    new_set = set(new_zone_ids)
    for old in old_zones:
        members = [r for r in recs if r["src_zone"] == old]
        dest_counts = {}
        for r in members:
            dest_counts[item_to_new[r["id"]]] = dest_counts.get(item_to_new[r["id"]], 0) + 1
        majority = max(dest_counts, key=dest_counts.get)
        letter = old.split("-")[-1]
        if old not in new_set:
            redirects[f"shoes/{letter}"] = f"#/shoes/{majority.split('-')[-1]}"
            redirects[f"shoes/{old}"] = f"#/shoes/{majority.split('-')[-1]}"
        old_map[old] = dict(majority=majority, dest_counts=dest_counts, split=len(dest_counts) > 1)

    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for zp in zone_packs:
            Image.fromarray(zp["plate"]).save(preview_dir / f"{zp['zone_id']}-plate.png")
            zp["composited"].save(preview_dir / f"{zp['zone_id']}-pack.png")

    per_item_scale = {
        zp["zone_id"]: sorted({round(p["scale"], 3) for p in zp["placed"]}) for zp in zone_packs
    }
    summary = {
        "category": "shoes",
        "items": len(recs),
        "scale": scale,
        "zones_before": old_zones,
        "zones_after": new_zone_ids,
        "counts": counts,
        "per_zone_rows": {zp["zone_id"]: zp["row_counts"] for zp in zone_packs},
        "gaps": {zp["zone_id"]: zp["gaps"] for zp in zone_packs},
        "src_plates": {zp["zone_id"]: zp["src_plate"] for zp in zone_packs},
        "rug_tops": {zp["zone_id"]: zp["rug_top"] for zp in zone_packs},
        "baselines": {zp["zone_id"]: zp["baselines"] for zp in zone_packs},
        "per_item_scale": per_item_scale,
        "redirects_added": {k: redirects[k] for k in redirects if k.startswith("shoes/")},
        "old_zone_mapping": old_map,
        "cleaned_layers": cleaned_ids,
    }

    if not write:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return summary, zone_packs, catalog

    by_id = {i["id"]: i for i in catalog["items"]}
    zone_entries_new = {}
    for zp in zone_packs:
        plate_im = Image.fromarray(zp["plate"])
        h, w = zp["plate"].shape[:2]
        images = R.variants_from(plate_im, f"assets/zones/{zp['zone_id']}", [1600, 1024, 640])
        cover_im = zp["composited"].resize(
            (640, max(1, round(640 * zp["composited"].height / zp["composited"].width))),
            Image.Resampling.LANCZOS,
        )
        cover_path = ROOT / f"assets/zones/{zp['zone_id']}-cover-640.webp"
        R.save_webp(cover_im, cover_path)
        cover = {"src": f"assets/zones/{zp['zone_id']}-cover-640.webp", "width": cover_im.width, "height": cover_im.height}
        letter = zp["zone_id"].split("-")[-1]
        for p in zp["placed"]:
            item = by_id[p["rec"]["id"]]
            box = R.pct((p["x"], p["y"], p["nw"], p["nh"]), w, h)
            layer = dict(item["layer"])
            layer.update(box)
            layer["rect"] = dict(box)
            layer["fallback"] = False
            layer["motion"] = "lift"
            layer["pivot"] = {"x": 50, "y": 50}
            layer["pivotX"] = 50
            layer["pivotY"] = 50
            layer["zIndex"] = p["zIndex"]
            item["zone"] = zp["zone_id"]
            item["layer"] = layer
            item["hotspot"] = dict(box)
            item["hotspot"]["clip"] = p["clip"]
        zone_entries_new[zp["zone_id"]] = dict(
            letter=letter,
            images=images,
            cover=cover,
            items=zp["item_ids"],
            natural_size=[w, h],
            aspect=w / h,
            has_layout=True,
        )

    zones = []
    existing = {z["id"]: z for z in catalog["zones"]}
    inserted = False
    for z in catalog["zones"]:
        if z["id"].startswith("shoes-") or z["id"] == "shoes":
            if not inserted:
                for zid in new_zone_ids:
                    base = dict(existing.get(zid) or existing[old_zones[0]])
                    base["id"] = zid
                    base.update(zone_entries_new[zid])
                    zones.append(base)
                inserted = True
            continue
        zones.append(z)
    catalog["zones"] = zones
    for cat in catalog["categories"]:
        if cat["id"] == "shoes":
            cat["count"] = sum(1 for i in catalog["items"] if i["category"] == "shoes")
            cat["cover_zone"] = new_zone_ids[0]
    catalog["redirects"] = redirects
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    DISPLAY.mkdir(parents=True, exist_ok=True)
    (DISPLAY / "shoes-repack.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary, zone_packs, catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--plates", default=None, help="comma-separated rug letters or shoes-X ids")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()
    plates = [p.strip() for p in args.plates.split(",")] if args.plates else None
    apply(scale=args.scale, plates=plates, preview_dir=args.preview, write=args.write, clean=not args.no_clean)


if __name__ == "__main__":
    main()
