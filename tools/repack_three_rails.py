#!/usr/bin/env python3
"""Repack a hanging category onto three cloned wall rails (display coords only).

No new paintings. Clones an existing wooden rod (wood, end caps, contact
shadow); hooks baked into the plate are erased so layer hooks can sit anywhere.
Existing layer webps are reused. Extraction layouts are not changed.
"""
from __future__ import annotations

import argparse
import json
import math
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/catalog.json"
DISPLAY = ROOT / "data/source/display"

MIN_GAP = 10
TARGET_PER_ZONE = 15
MAX_PER_ZONE = 17
MIN_PER_ZONE = 8
DEFAULT_SCALE = 0.62


def save_webp(im: Image.Image, path: Path, quality=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Image.fromarray(np.asarray(im.convert("RGB") if im.mode != "RGBA" else im))
    temporary = path.with_suffix(path.suffix + ".tmp")
    clean.save(temporary, "WEBP", quality=quality, method=4)
    temporary.replace(path)


def variants_from(im: Image.Image, stem: str, sizes):
    result = []
    for size in sizes:
        scale = min(1, size / im.width)
        out = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
            Image.Resampling.LANCZOS,
        )
        path = ROOT / f"{stem}-{size}.webp"
        save_webp(out, path)
        result.append({"src": f"{stem}-{size}.webp", "width": out.width, "height": out.height})
    return result


def wood_mask(rgb):
    r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    return (r > b + 18) & (r > 55) & (r < 220) & (g > 30) & (g < 180) & (r > g)


def detect_rail_bands(plate, min_frac=0.18):
    """Return wood rail bands as (y0, y1, x0, x1), ignoring floor rug / furniture."""
    h, w = plate.shape[:2]
    wood = wood_mask(plate)
    xs0, xs1 = int(w * 0.12), int(w * 0.88)
    # ignore bottom 18% (rug / baseboard); keep the first few rows so a
    # ceiling-hugging rail (e.g. dresses-a at y≈11) is not dropped.
    y_lo, y_hi = 0, int(h * 0.82)
    ys = [y for y in range(y_lo, y_hi) if wood[y, xs0:xs1].mean() > min_frac]
    bands = []
    if not ys:
        return bands
    a = prev = ys[0]
    for y in ys[1:] + [10**9]:
        if y > prev + 3:
            if prev - a >= 8:
                # x extent
                rows = wood[a : prev + 1, :]
                cols = np.where(rows.mean(0) > 0.2)[0]
                if len(cols):
                    x0, x1 = int(cols[0]), int(cols[-1])
                else:
                    x0, x1 = xs0, xs1
                bands.append((a, prev, x0, x1))
            a = y
        prev = y
    # Rails are thin/medium horizontal bars. Cabinets/benches are lower and continue downward.
    h = plate.shape[0]
    wood = wood_mask(plate)
    rails = []
    for b in bands:
        y0, y1, x0, x1 = b
        if not (8 <= (y1 - y0) <= 70):
            continue
        below = wood[y1 + 1 : min(h, y1 + 50), max(0, x0) : min(plate.shape[1], x1 + 1)].mean() if y1 + 1 < h else 0
        if below > 0.25:
            continue  # furniture face continues under this band
        # Cabinets sit low; thin isolated rails may sit just past 0.62*h
        # (dresses-a bottom rail is at 0.631*h on a 900px plate).
        thick = (y1 - y0) > 28
        if y0 > h * 0.75 or (y0 > h * 0.62 and thick):
            continue
        rails.append(b)
    rails.sort(key=lambda b: (b[1] - b[0], -b[0]), reverse=True)
    rails = sorted(rails[:3], key=lambda b: b[0])
    return rails


def detect_obstacle_y(plate):
    """First y of furniture/rug in the lower wall (usable wall ends above this)."""
    h, w = plate.shape[:2]
    wood = wood_mask(plate)
    # cabinet / bench: dark wood block in lower third, wide.
    # Skip thin isolated wood rows — those are rails, not furniture.
    y = int(h * 0.55)
    mid0, mid1 = int(w * 0.15), int(w * 0.85)
    while y < int(h * 0.92):
        frac = wood[y, mid0:mid1].mean()
        if frac > 0.35:
            y1 = y
            while y1 < h - 1 and wood[y1, mid0:mid1].mean() > 0.15:
                y1 += 1
            if y1 - y <= 28:
                y = y1 + 1
                continue
            return y
        y += 1
    # rug / baseboard fallback: sudden darkening / texture in bottom
    lum = plate.mean(2)
    for y in range(int(h * 0.70), h - 8):
        row = lum[y, int(w * 0.2) : int(w * 0.8)]
        below = lum[min(h - 1, y + 8), int(w * 0.2) : int(w * 0.8)]
        if row.mean() < 160 and below.mean() < 155 and row.std() > 18:
            return y
    return int(h * 0.86)



def detect_hook_xs(plate, band):
    """X centers of metal hooks, using the top of the wood (not the grey wall)."""
    y0, y1, x0, x1 = band
    # Score metal only on the wood (not the grey wall above/below, which matches "bright low-chroma").
    y_a, y_b = y0, min(plate.shape[0], y0 + min(12, max(8, y1 - y0)))
    crop = plate[y_a:y_b, x0:x1 + 1]
    lum = crop.mean(2)
    chroma = crop.max(2).astype(int) - crop.min(2)
    r, b = crop[:, :, 0].astype(int), crop[:, :, 2].astype(int)
    metal = (lum > 140) & (chroma < 50) & (np.abs(r - b) < 50) & (r < b + 40)
    score = metal.mean(0)
    # Peak-pick metal on the wood so thin 1-column hooks are not dropped.
    kernel = np.ones(5) / 5.0
    sm = np.convolve(score, kernel, mode="same")
    xs = []
    for i in range(8, len(sm) - 8):
        if sm[i] < 0.16:
            continue
        if sm[i] >= sm[i - 2 : i + 3].max() - 1e-9:
            hx = int(x0 + i)
            if hx - x0 > 40 and x1 - hx > 40:
                xs.append(hx)
    # Also find hooks that hang *below* the rod (dark thin silhouette vs the wall).
    y_lo, y_hi = y1 + 1, min(plate.shape[0], y1 + 24)
    if y_hi - y_lo >= 8:
        below = plate[y_lo:y_hi, x0:x1 + 1].astype(np.float32).mean(2)
        col = below.mean(0)
        wall = float(np.median(col))
        dark = col < (wall - 12)
        i = 0
        n = len(dark)
        while i < n:
            if dark[i]:
                j = i
                while j < n and dark[j]:
                    j += 1
                if 1 <= j - i <= 18:
                    hx = int(x0 + i + int(col[i:j].argmin()))
                    if hx - x0 > 40 and x1 - hx > 40:
                        xs.append(hx)
                i = j
            else:
                i += 1
    # Local color-anomaly hooks: small rings that hug the wood and are too
    # light to show up as a dark silhouette against the wall (dresses-a).
    wood = plate[y0 : y1 + 1, x0 : x1 + 1].astype(float)
    col = wood.mean(0)
    k, pad = 41, 20
    padded = np.pad(col, ((pad, pad), (0, 0)), mode="edge")
    mm = np.empty_like(col)
    for i in range(len(col)):
        mm[i] = np.median(padded[i : i + k], axis=0)
    dev = np.linalg.norm(col - mm, axis=1)
    extra = np.zeros(len(col))
    y_lo2, y_hi2 = y1 + 1, min(plate.shape[0], y1 + 10)
    y_a2, y_b2 = max(0, y0 - 8), y0
    if y_hi2 > y_lo2:
        below2 = plate[y_lo2:y_hi2, x0 : x1 + 1].astype(float).mean(2)
        extra = np.maximum(extra, np.abs(below2 - float(np.median(below2))).max(0))
    if y_b2 > y_a2:
        above2 = plate[y_a2:y_b2, x0 : x1 + 1].astype(float).mean(2)
        extra = np.maximum(extra, np.abs(above2 - float(np.median(above2))).max(0))
    anom = np.convolve(dev + 0.35 * extra, np.ones(5) / 5.0, mode="same")
    mad = float(np.median(np.abs(anom - np.median(anom))))
    ath = max(12.0, float(np.median(anom)) + 4 * mad)
    anomaly_xs = []
    for i in range(20, len(anom) - 20):
        if anom[i] < ath:
            continue
        if anom[i] >= anom[max(0, i - 6) : i + 7].max() - 1e-9:
            hx = int(x0 + i)
            if hx - x0 > 40 and x1 - hx > 40:
                anomaly_xs.append(hx)
    # Drop weak wood-grain peaks; keep strong rings.
    if anomaly_xs:
        ascores = [float(anom[hx - x0]) for hx in anomaly_xs]
        keep_thr = max(30.0, float(np.median(ascores)) * 0.7)
        anomaly_xs = [hx for hx, s in zip(anomaly_xs, ascores) if s >= keep_thr]
    xs.extend(anomaly_xs)

    xs = sorted(xs)
    merged = []
    for hx in xs:
        if not merged or hx - merged[-1] > 70:
            merged.append(hx)
        else:
            merged[-1] = (merged[-1] + hx) // 2
    return merged


def surgical_remove_hooks(im, hooks, y0, y1, x_rail0, x_rail1, below=24, above=16, halfw=16):
    out = im.copy()
    h, w = out.shape[:2]
    for hx in hooks:
        xa = max(x_rail0 + 8, hx - halfw)
        xb = min(x_rail1 - 8, hx + halfw)
        if xb <= xa:
            xa = max(0, hx - halfw)
            xb = min(im.shape[1] - 1, hx + halfw)
        if xb <= xa:
            continue
        for y in range(max(0, y0 - above), y1 + 1):
            left = out[y, max(x_rail0, xa - 18) : xa]
            right = out[y, xb + 1 : min(x_rail1, xb + 19)]
            if len(left) == 0 and len(right) == 0:
                continue
            if len(left) and len(right):
                el, er = left[-1].astype(float), right[0].astype(float)
                tt = np.linspace(0, 1, xb - xa + 1)[:, None]
                out[y, xa : xb + 1] = np.clip(el * (1 - tt) + er * tt, 0, 255).astype(np.uint8)
            else:
                donors = left if len(left) else right
                out[y, xa : xb + 1] = np.median(donors, 0)
        for y in range(y1 + 1, min(h, y1 + 1 + below)):
            left = out[y, max(0, xa - 40) : xa]
            right = out[y, xb + 1 : min(w, xb + 41)]
            if len(left) == 0 and len(right) == 0:
                continue
            if len(left) and len(right):
                el, er = left[-1].astype(float), right[0].astype(float)
                tt = np.linspace(0, 1, xb - xa + 1)[:, None]
                fill = np.clip(el * (1 - tt) + er * tt, 0, 255).astype(np.uint8)
            else:
                donors = left if len(left) else right
                fill = np.repeat(np.median(donors, 0)[None, :], xb - xa + 1, 0)
            region = out[y, xa : xb + 1]
            neigh = np.concatenate(
                [left[-5:] if len(left) else region, right[:5] if len(right) else region]
            )
            nmed = np.median(neigh, 0)
            diff = np.linalg.norm(region.astype(float) - nmed, axis=1)
            region[diff > 8] = fill[diff > 8]
            out[y, xa : xb + 1] = region
    return out


def extract_rail_sprite(plate, band):
    """Cut one rail (wood, end caps, contact shadow) as RGBA; drop metal hooks."""
    y0, y1, x0, x1 = band
    pad_x, pad_top, pad_bot = 18, 10, 16
    sy0, sy1 = max(0, y0 - pad_top), min(plate.shape[0] - 1, y1 + pad_bot)
    sx0, sx1 = max(0, x0 - pad_x), min(plate.shape[1] - 1, x1 + pad_x)
    crop = plate[sy0 : sy1 + 1, sx0 : sx1 + 1].copy()
    ch, cw = crop.shape[:2]
    r, g, b = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    chroma = crop.max(2).astype(int) - crop.min(2)
    lum = crop.mean(2)
    wood = wood_mask(crop)
    # metal hooks: bright, low-chroma, often slightly above the wood
    metal = (lum > 140) & (chroma < 28) & (np.abs(r.astype(int) - b) < 35)
    wood_mask_u8 = cv2.dilate(np.uint8(wood) * 255, np.ones((3, 3), np.uint8))
    # do not treat metal as wood
    wood_mask_u8[metal] = 0
    # fill tiny hook holes in the wood so the bar is solid
    wood_mask_u8 = cv2.morphologyEx(wood_mask_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    shadow = np.zeros_like(wood_mask_u8)
    wood_bottom_local = y1 - sy0
    for yy in range(wood_bottom_local + 1, ch):
        dark = (lum[yy] < 175) | (chroma[yy] > 18)
        near = cv2.dilate(wood_mask_u8, np.ones((25, 5), np.uint8))[yy] > 0
        shadow[yy] = np.uint8(dark & near & (~metal[yy])) * 255
    above = np.zeros_like(wood_mask_u8)
    for yy in range(0, max(0, y0 - sy0)):
        near = cv2.dilate(wood_mask_u8, np.ones((7, 3), np.uint8))[yy] > 0
        odd = ((np.abs(lum[yy] - float(np.median(lum[yy]))) > 10) | (chroma[yy] > 16)) & (~metal[yy])
        # don't keep hooks above the rail
        above[yy] = np.uint8(odd & near) * 255
    alpha = np.maximum.reduce([wood_mask_u8, shadow, above])
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # inpaint hook pixels on the bar so we don't paste silver rings
    hook_on_bar = metal & (wood_mask_u8 > 0)
    if hook_on_bar.any():
        fill = cv2.inpaint(crop, np.uint8(hook_on_bar) * 255, 3, cv2.INPAINT_TELEA)
        # only replace hook pixels, keep end caps
        crop = np.where(hook_on_bar[:, :, None], fill, crop)
    alpha_f = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    sprite = Image.fromarray(np.dstack([crop, alpha_f]))
    return sprite, sx0, sy0, y0 - sy0, (y1 - y0)


def erase_rail_band(img, ey0, ey1, ex0, ex1, donor_above=14, donor_below=18):
    out = img.copy()
    h, w = out.shape[:2]
    y_top = max(0, ey0 - donor_above)
    y_bot = min(h - 1, ey1 + donor_below)
    x0, x1 = max(0, ex0), min(w - 1, ex1)
    top = np.median(out[max(0, y_top - 6) : y_top + 1, x0 : x1 + 1], axis=0)
    bot = np.median(out[y_bot : min(h, y_bot + 7), x0 : x1 + 1], axis=0)
    span = max(1, y_bot - y_top)
    for y in range(y_top, y_bot + 1):
        t = (y - y_top) / span
        out[y, x0 : x1 + 1] = np.clip((1 - t) * top + t * bot, 0, 255).astype(np.uint8)
    return out



def heal_under_rail(img, rails, depth=16):
    """Smooth leftover hook-slot ticks just under each cloned bar."""
    out = img.copy()
    h, w = out.shape[:2]
    for rail in rails:
        y0 = rail["rail_bottom"] - 4
        x0 = max(0, rail.get("wood_x0", rail["x0"]) + 16)
        x1 = min(w - 1, rail.get("wood_x1", rail["x1"]) - 16)
        for dy in range(depth):
            y = y0 + dy
            if y >= h:
                break
            row = out[y, x0 : x1 + 1].astype(float)
            med = np.median(row, 0)
            lum = row.mean(1)
            med_lum = float(np.median(lum))
            # ticks are much brighter than the contact shadow
            bad = lum > med_lum + 18
            if not bad.any():
                continue
            # expand a little
            bad = np.convolve(bad.astype(np.uint8), np.ones(5), mode="same") > 0
            xs = np.where(bad)[0]
            i = 0
            while i < len(xs):
                j = i
                while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
                    j += 1
                a, b = xs[i], xs[j]
                left = row[max(0, a - 8) : a]
                right = row[b + 1 : min(len(row), b + 9)]
                if len(left) and len(right):
                    el, er = left[-1], right[0]
                    tt = np.linspace(0, 1, b - a + 1)[:, None]
                    row[a : b + 1] = el * (1 - tt) + er * tt
                elif len(left) or len(right):
                    donors = left if len(left) else right
                    row[a : b + 1] = np.median(donors, 0)
                i = j + 1
            out[y, x0 : x1 + 1] = np.clip(row, 0, 255).astype(np.uint8)
    return out


def add_three_rails(plate, first_wood=None, spacing=None, max_item_below=260, keep_spacing=False):
    bands = detect_rail_bands(plate)
    if len(bands) < 1:
        raise SystemExit("no rail bands detected")
    # strip metal hooks from every source rail first, then clone a clean bar
    cleaned = plate.copy()
    for band in bands:
        y0, y1, x0, x1 = band
        hooks = detect_hook_xs(cleaned, band)
        if hooks:
            cleaned = surgical_remove_hooks(cleaned, hooks, y0, y1, x0, x1, below=64, above=18, halfw=32)
    src = bands[0]
    sprite, sx0, sy0, wood_off, wood_h = extract_rail_sprite(cleaned, src)
    # erase all detected rails (and a little extra for hooks sticking above)
    wall = cleaned.copy()
    for y0, y1, x0, x1 in bands:
        wall = erase_rail_band(wall, y0 - 10, y1 + 6, x0 - 24, x1 + 24)
    obstacle = detect_obstacle_y(plate)
    bands_by_y = sorted(bands, key=lambda b: b[0])
    if keep_spacing and len(bands_by_y) >= 3:
        wood_ys = [b[0] for b in bands_by_y[:3]]
        first_wood = wood_ys[0]
        spacing = int(round((wood_ys[1] - wood_ys[0] + wood_ys[2] - wood_ys[1]) / 2))
    else:
        if first_wood is None:
            # lift the first rail a bit if the original sat low, to fit three rows
            first_wood = min(src[0], max(48, src[0] - 20))
        if spacing is None:
            # Prefer enough room *between* rails so a typical tall item does not cover the rail below.
            gap = 16
            want = int(max_item_below + gap)
            # Even split of leftover wall under the last rail (so bottom items can hang).
            usable = obstacle - 16 - first_wood - wood_h - max_item_below
            even = usable // 2 if usable > 0 else 240
            spacing = max(wood_h + 40, want, even)
            last = first_wood + 2 * spacing
            if last + wood_h + max_item_below > obstacle - 8:
                # shrink: keep as close to `want` as the wall allows
                spacing = max(wood_h + 36, (obstacle - 8 - max_item_below - wood_h - first_wood) // 2)
        wood_ys = [first_wood, first_wood + spacing, first_wood + 2 * spacing]
    canvas = Image.fromarray(wall).convert("RGBA")
    rails = []
    hook_from_wood = max(6, min(12, wood_h // 3))
    for yt in wood_ys:
        canvas.alpha_composite(sprite, (sx0, yt - wood_off))
        rails.append(
            dict(
                hook_y=yt + hook_from_wood,
                rail_bottom=yt + wood_h,
                x0=max(src[2] + 24, 200),
                x1=min(src[3] - 24, plate.shape[1] - 200),
                wood_top=yt,
                wood_h=wood_h,
                wood_x0=sx0,
                wood_x1=sx0 + sprite.size[0] - 1,
            )
        )
    rgb = np.array(canvas.convert("RGB"))
    # second pass: strip any hooks that were cloned because detection missed them first time
    for rail in rails:
        band = (rail["wood_top"], rail["rail_bottom"], rail.get("wood_x0", rail["x0"]), rail.get("wood_x1", rail["x1"]))
        hooks = detect_hook_xs(rgb, band)
        if hooks:
            rgb = surgical_remove_hooks(rgb, hooks, rail["wood_top"], rail["rail_bottom"], band[2], band[3], below=64, above=8, halfw=28)
    rgb = heal_under_rail(rgb, rails)
    return rgb, rails, dict(
        first_wood=first_wood, spacing=spacing, wood_h=wood_h, obstacle=obstacle, src_bands=bands
    )


def zone_counts(n, target=TARGET_PER_ZONE, max_per=MAX_PER_ZONE, min_per=MIN_PER_ZONE):
    if n <= max_per:
        return [n]
    nz = max(1, int(round(n / target)))
    while n / nz > max_per:
        nz += 1
    # avoid a tiny last room
    while nz > 1:
        base, rem = divmod(n, nz)
        counts = [base + 1] * rem + [base] * (nz - rem)
        if min(counts) >= min_per or n / (nz - 1) > max_per:
            return counts
        nz -= 1
    base, rem = divmod(n, nz)
    return [base + 1] * rem + [base] * (nz - rem)


def rod_counts(k, rods=3):
    base, rem = divmod(k, rods)
    return [base + 1] * rem + [base] * (rods - rem)


def load_layers(catalog, category):
    recs = []
    for item in catalog["items"]:
        if item["category"] != category:
            continue
        arr = np.array(Image.open(ROOT / item["layer"]["image"]).convert("RGBA"))
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 80)
        ax = float(xs[ys <= ys.min() + 8].mean()) if len(xs) else arr.shape[1] / 2
        ay = float(ys.min()) if len(ys) else 0
        top_width = int((alpha[1:9] > 80).sum(1).max()) if alpha.shape[0] > 8 else 0
        kind = "handle" if top_width > arr.shape[1] * 0.4 else "hook"
        recs.append(
            dict(
                id=item["id"],
                item=item,
                arr=arr,
                w=arr.shape[1],
                h=arr.shape[0],
                ax=ax,
                ay=ay,
                kind=kind,
                src_zone=item["zone"],
                motion=item.get("layer", {}).get("motion") or "hang",
            )
        )
    # catalog order (already the collection walk / type groups)
    return recs


def pack_row(row, rail, scale):
    widths = [it["w"] * scale for it in row]
    usable = rail["x1"] - rail["x0"]
    remain = usable - sum(widths)
    n = len(row)
    gap = remain / (n + 1) if n else 0
    x = rail["x0"] + gap
    placed = []
    for it, ww in zip(row, widths):
        if it["kind"] == "handle":
            top = rail["rail_bottom"] + 4 - it["ay"] * scale
        else:
            top = rail["hook_y"] - it["ay"] * scale
        placed.append(
            dict(
                rec=it,
                left=x,
                top=top,
                scale=scale,
                ww=ww,
                hh=it["h"] * scale,
                hook_x=x + it["ax"] * scale,
                gap=gap,
                kind=it["kind"],
            )
        )
        x += ww + gap
    return placed, gap


def _poly_from_cnt(cnt, w, h):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, max(1.5, 0.008 * peri), True)
    if len(approx) < 3:
        x, y, bw, bh = cv2.boundingRect(cnt)
        return [
            [100 * x / w, 100 * y / h],
            [100 * (x + bw) / w, 100 * y / h],
            [100 * (x + bw) / w, 100 * (y + bh) / h],
            [100 * x / w, 100 * (y + bh) / h],
        ]
    return [
        [float(np.clip(100.0 * x / w, 0, 100)), float(np.clip(100.0 * y / h, 0, 100))]
        for x, y in approx.reshape(-1, 2)
    ]


def _pip(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside


def clip_polygon(alpha):
    """Tight outline, pulled just far enough to contain the opaque centroid (crotch gap)."""
    mask = np.uint8(alpha > 40)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = alpha.shape
    if not contours:
        return [[0, 0], [100, 0], [100, 100], [0, 100]]
    cnt = max(contours, key=cv2.contourArea)
    poly = _poly_from_cnt(cnt, w, h)
    ys, xs = np.where(alpha > 80)
    if len(xs) == 0:
        return poly
    cx, cy = float(xs.mean() / w * 100), float(ys.mean() / h * 100)
    # A vertex exactly on the centroid is treated as outside by CSS clip-path.
    # Always pull a small diamond around the centroid into the polygon.
    diamond = np.array(
        [
            [cx, cy - 4],
            [cx + 4, cy],
            [cx, cy + 4],
            [cx - 4, cy],
        ],
        dtype=np.float32,
    )
    diamond[:, 0] = np.clip(diamond[:, 0] / 100.0 * w, 0, w - 1)
    diamond[:, 1] = np.clip(diamond[:, 1] / 100.0 * h, 0, h - 1)
    if _pip(cx, cy, poly) and all(
        _pip(float(px / w * 100), float(py / h * 100), poly) for px, py in diamond
    ):
        return poly
    # Fatten only the existing tight outline around the centroid, not the whole silhouette.
    verts = np.array([[x / 100.0 * w, y / 100.0 * h] for x, y in poly], dtype=np.float32)
    hull = cv2.convexHull(np.vstack([verts, diamond]))
    return _poly_from_cnt(hull, w, h)


def composite(plate, placed):
    canvas = Image.fromarray(plate).convert("RGBA")
    for i, p in enumerate(placed):
        im = Image.fromarray(p["rec"]["arr"])
        nw, nh = max(1, round(p["ww"])), max(1, round(p["hh"]))
        rs = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.alpha_composite(rs.convert("RGBA"), (max(0, round(p["left"])), max(0, round(p["top"]))))
        p["nw"], p["nh"] = nw, nh
        p["x"], p["y"] = round(p["left"]), round(p["top"])
        p["clip"] = clip_polygon(np.array(rs)[:, :, 3])
        p["zIndex"] = i + 1
    return canvas.convert("RGB")


def pct(box, w, h):
    x, y, bw, bh = box
    return dict(left=x / w * 100, top=y / h * 100, width=bw / w * 100, height=bh / h * 100)


def letters(n):
    return list(string.ascii_lowercase[:n])


def layout_height(recs):
    """Height used to space rails. Ignore a single extreme outlier (e.g. overalls)."""
    hs = sorted(r["h"] for r in recs)
    if not hs:
        return 1
    p95 = hs[min(len(hs) - 1, int(round(0.95 * (len(hs) - 1))))]
    mx = hs[-1]
    if mx > p95 * 1.2 and len(hs) >= 8:
        return max(1, p95)
    return max(1, mx)


def choose_scale(recs, counts, plates_meta, target=DEFAULT_SCALE):
    """Fit a consistent scale: prefer ~0.62, shrink if a row would collide."""
    max_h = layout_height(recs)
    # horizontal: busiest row (max items on one rail) vs usable width
    max_row = max(max(rod_counts(c)) for c in counts)
    max_w_sum = 0
    # worst-case: the widest `max_row` items in sequence
    widths = [r["w"] for r in recs]
    for i in range(0, max(1, len(widths) - max_row + 1)):
        max_w_sum = max(max_w_sum, sum(widths[i : i + max_row]))
    usable = min(m["rails"][0]["x1"] - m["rails"][0]["x0"] for m in plates_meta) if plates_meta else 1000
    s_h = (usable - MIN_GAP * (max_row + 1)) / max(1, max_w_sum)
    # vertical: between rails and under the last rail
    s_v = 1.0
    for m in plates_meta:
        rails = m["rails"]
        obstacle = m["meta"]["obstacle"]
        spacing = rails[1]["wood_top"] - rails[0]["wood_top"]
        if spacing > 24:
            s_v = min(s_v, (spacing - 16) / max_h)
        room = obstacle - 8 - rails[2]["rail_bottom"]
        if room > 40:
            s_v = min(s_v, room / max_h)
        # also: could we have spaced rails farther? bound by full wall
        first = m["meta"].get("first_wood", rails[0]["wood_top"])
        wood_h = m["meta"].get("wood_h", rails[0].get("wood_h", 16))
        wall = obstacle - 8 - first - wood_h
        # 2 gaps + last hang = 3 * (scale*h + slack)
        if wall > 80:
            s_v = min(s_v, (wall - 48) / (3 * max_h))
    s = min(target, s_h, s_v)
    return max(0.42, float(round(s, 2)))


def apply_category(category, scale=None, preview_dir=None, write=False, plate_ids=None, target=None, only_zones=None, keep_spacing=False):
    catalog = json.loads(CATALOG.read_text())
    recs = load_layers(catalog, category)
    if only_zones:
        allow = set(only_zones)
        recs = [r for r in recs if r["src_zone"] in allow]
    if not recs:
        raise SystemExit(f"no items for {category}")
    old_zones = []
    seen = set()
    for r in recs:
        if r["src_zone"] not in seen:
            old_zones.append(r["src_zone"])
            seen.add(r["src_zone"])
    if only_zones:
        # Keep the named rooms; do not collapse the rest of the category.
        new_zone_ids = list(only_zones)
        counts = zone_counts(len(recs), target=target or TARGET_PER_ZONE)
        if len(counts) != len(new_zone_ids):
            if len(new_zone_ids) == 1:
                counts = [len(recs)]
            else:
                raise SystemExit(f"only_zones {new_zone_ids} does not match counts {counts}")
        n_new = len(counts)
    else:
        counts = zone_counts(len(recs), target=target or TARGET_PER_ZONE)
        n_new = len(counts)
        new_letters = letters(n_new)
        new_zone_ids = [f"{category}-{L}" for L in new_letters]

    # choose source empty plates: prefer first n existing zones (unique rooms)
    if plate_ids is None:
        plate_ids = old_zones[:n_new]
        if len(plate_ids) < n_new:
            # reuse the first plate
            while len(plate_ids) < n_new:
                plate_ids.append(old_zones[0])

    # preview max item height at target scale to place rails
    target = scale or DEFAULT_SCALE
    max_h_guess = layout_height(recs) * target

    plates_meta = []
    built = []
    for zid in plate_ids:
        src = ROOT / "assets/zones" / f"{zid}-1600.webp"
        if not src.exists():
            raise SystemExit(f"missing plate {src}")
        plate = np.array(Image.open(src).convert("RGB"))
        three, rails, meta = add_three_rails(plate, max_item_below=int(max_h_guess) + 8, keep_spacing=keep_spacing)
        built.append((three, rails, meta, zid))
        plates_meta.append({"rails": rails, "meta": meta})

    if scale is None:
        scale = choose_scale(recs, counts, plates_meta, target=DEFAULT_SCALE)
        # if auto scale changed a lot, rebuild rails with the real max height
        max_h_real = layout_height(recs) * scale
        if abs(max_h_real - max_h_guess) > 20:
            built, plates_meta = [], []
            for zid in plate_ids:
                src = ROOT / "assets/zones" / f"{zid}-1600.webp"
                plate = np.array(Image.open(src).convert("RGB"))
                three, rails, meta = add_three_rails(plate, max_item_below=int(max_h_real) + 8, keep_spacing=keep_spacing)
                built.append((three, rails, meta, zid))
                plates_meta.append({"rails": rails, "meta": meta})

    # pack
    zone_packs = []
    idx = 0
    for zi, count in enumerate(counts):
        plate, rails, meta, src_zid = built[zi]
        chunk = recs[idx : idx + count]
        idx += count
        rc = rod_counts(count)
        rows, cursor = [], 0
        for n in rc:
            rows.append(chunk[cursor : cursor + n])
            cursor += n
        # Keep items taller than the inter-rail gap on the bottom rail.
        if len(rows) == 3 and len(rails) >= 2:
            spacing = rails[1]["wood_top"] - rails[0]["wood_top"]
            limit = spacing - 12
            for ri in (0, 1):
                kept, moved = [], []
                for it in rows[ri]:
                    if it["h"] * scale > limit:
                        moved.append(it)
                    else:
                        kept.append(it)
                if not moved:
                    continue
                # swap with the shortest items already on the bottom rail
                bottom = list(rows[2])
                bottom.sort(key=lambda it: it["h"])
                take = bottom[: len(moved)]
                bottom = bottom[len(moved) :] + moved
                rows[ri] = kept + take
                rows[2] = bottom
        packed, gaps = [], []
        for row, rail in zip(rows, rails):
            if not row:
                gaps.append(None)
                continue
            pl, g = pack_row(row, rail, scale)
            packed.extend(pl)
            gaps.append(g)
        if any(g is not None and g < MIN_GAP for g in gaps):
            raise SystemExit(f"{new_zone_ids[zi]} gaps too tight: {gaps}")
        composited = composite(plate, packed)
        zone_packs.append(
            dict(
                zone_id=new_zone_ids[zi],
                src_plate=src_zid,
                plate=plate,
                rails=rails,
                meta=meta,
                placed=packed,
                composited=composited,
                gaps=gaps,
                rod_counts=rc,
                item_ids=[p["rec"]["id"] for p in packed],
            )
        )

    # redirects: removed letters, and any old zone whose items all moved
    old_letter = {z: z.split("-")[-1] for z in old_zones}
    new_set = set(new_zone_ids)
    # map each old zone -> majority new zone
    redirects = dict(catalog.get("redirects") or {})
    for old in old_zones:
        dests = {}
        for zp, rec in zip(
            # item -> new zone
            [],
            [],
        ):
            pass
        # compute from packed
    item_to_new = {}
    for zp in zone_packs:
        for iid in zp["item_ids"]:
            item_to_new[iid] = zp["zone_id"]
    for old in old_zones:
        members = [r for r in recs if r["src_zone"] == old]
        dest_counts = {}
        for r in members:
            dest_counts[item_to_new[r["id"]]] = dest_counts.get(item_to_new[r["id"]], 0) + 1
        majority = max(dest_counts, key=dest_counts.get)
        letter = old_letter[old]
        if old not in new_set:
            redirects[f"{category}/{letter}"] = f"#/{category}/{majority.split('-')[-1]}"
            redirects[f"{category}/{old}"] = f"#/{category}/{majority.split('-')[-1]}"
        split = len(dest_counts) > 1
        old_letter[old] = dict(majority=majority, dest_counts=dest_counts, split=split)

    # preview images
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for zp in zone_packs:
            Image.fromarray(zp["plate"]).save(preview_dir / f"{zp['zone_id']}-plate.png")
            zp["composited"].save(preview_dir / f"{zp['zone_id']}-pack.png")

    summary = {
        "category": category,
        "items": len(recs),
        "scale": scale,
        "zones_before": old_zones,
        "zones_after": new_zone_ids,
        "counts": counts,
        "per_zone_rods": {zp["zone_id"]: zp["rod_counts"] for zp in zone_packs},
        "gaps": {zp["zone_id"]: zp["gaps"] for zp in zone_packs},
        "src_plates": {zp["zone_id"]: zp["src_plate"] for zp in zone_packs},
        "rails": {
            zp["zone_id"]: [{"wood_top": r["wood_top"], "hook_y": r["hook_y"]} for r in zp["rails"]]
            for zp in zone_packs
        },
        "redirects_added": {k: redirects[k] for k in redirects if k.startswith(category + "/")},
        "old_zone_mapping": {z: old_letter[z] for z in old_zones if isinstance(old_letter[z], dict)},
        "handle_special": [r["id"] for r in recs if r["kind"] == "handle"],
    }

    if not write:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return summary, zone_packs, catalog

    # write plates, covers, catalog
    by_id = {i["id"]: i for i in catalog["items"]}
    zone_entries_new = {}
    for zp in zone_packs:
        plate_im = Image.fromarray(zp["plate"])
        h, w = zp["plate"].shape[:2]
        images = variants_from(plate_im, f"assets/zones/{zp['zone_id']}", [1600, 1024, 640])
        cover_im = zp["composited"].resize(
            (640, max(1, round(640 * zp["composited"].height / zp["composited"].width))),
            Image.Resampling.LANCZOS,
        )
        cover_path = ROOT / f"assets/zones/{zp['zone_id']}-cover-640.webp"
        save_webp(cover_im, cover_path)
        cover = {"src": f"assets/zones/{zp['zone_id']}-cover-640.webp", "width": cover_im.width, "height": cover_im.height}
        for p in zp["placed"]:
            item = by_id[p["rec"]["id"]]
            box = pct((p["x"], p["y"], p["nw"], p["nh"]), w, h)
            layer = dict(item["layer"])
            layer.update(box)
            layer["rect"] = dict(box)
            layer["fallback"] = False
            layer["motion"] = p["rec"]["motion"]
            layer["pivot"] = {"x": 50, "y": 0}
            layer["pivotX"] = 50
            layer["pivotY"] = 0
            layer["zIndex"] = p["zIndex"]
            item["zone"] = zp["zone_id"]
            item["layer"] = layer
            item["hotspot"] = dict(box)
            item["hotspot"]["clip"] = p["clip"]
            # do not touch name/story/card_images/image
        zone_entries_new[zp["zone_id"]] = dict(
            images=images, cover=cover, items=zp["item_ids"], natural_size=[w, h], aspect=w / h, has_layout=True
        )

    zones = []
    existing = {z["id"]: z for z in catalog["zones"]}
    if only_zones:
        # Leave sibling rooms in this category untouched.
        for z in catalog["zones"]:
            if z["id"] in zone_entries_new:
                base = dict(z)
                base.update(zone_entries_new[z["id"]])
                zones.append(base)
            else:
                zones.append(z)
    else:
        # keep non-category zones in original order; replace category zones with the new set in letter order
        inserted = False
        for z in catalog["zones"]:
            if z["id"].startswith(category + "-") or z["id"] == category:
                if not inserted:
                    for zid in new_zone_ids:
                        base = dict(existing.get(zid) or existing[old_zones[0]])
                        base = dict(base)
                        base["id"] = zid
                        base.update(zone_entries_new[zid])
                        zones.append(base)
                    inserted = True
                continue
            zones.append(z)
        if not inserted:
            for zid in new_zone_ids:
                base = dict(existing[old_zones[0]])
                base["id"] = zid
                base.update(zone_entries_new[zid])
                zones.append(base)
    catalog["zones"] = zones

    for cat in catalog["categories"]:
        if cat["id"] == category:
            cat["count"] = sum(1 for i in catalog["items"] if i["category"] == category)
            if not only_zones:
                cat["cover_zone"] = new_zone_ids[0]

    catalog["redirects"] = redirects
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")

    rec_path = DISPLAY / f"{category}-repack.json"
    DISPLAY.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary, zone_packs, catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--preview", default=None, help="directory for plate/pack PNGs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--plates", default=None, help="comma-separated source plate zone ids")
    ap.add_argument("--target", type=int, default=None, help="target items per zone")
    ap.add_argument("--only-zones", default=None, help="comma-separated zone ids to convert; leave siblings alone")
    ap.add_argument("--keep-spacing", action="store_true", help="clone rails at the detected y positions")
    args = ap.parse_args()
    plate_ids = args.plates.split(",") if args.plates else None
    only_zones = args.only_zones.split(",") if args.only_zones else None
    apply_category(args.category, scale=args.scale, preview_dir=args.preview, write=args.write, plate_ids=plate_ids, target=args.target, only_zones=only_zones, keep_spacing=args.keep_spacing)


if __name__ == "__main__":
    main()
