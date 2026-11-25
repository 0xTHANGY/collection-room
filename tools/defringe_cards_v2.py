#!/usr/bin/env python3
"""Cutout-fix-2 card cleaner: remove residual white rims, enclosed backdrop inside
strap/chain loops, white chain specks, and whitish bag-bottom shadow blobs.

Uses QA's whitescan.py definition of a white halo (light pixels within 2px of
transparency whose 4–7px interior is not light) plus a peel of thicker
backdrop-coloured bands that sit between transparency and a darker object
(today-400 inner chain, bag shadows). Enclosed flat studio-white holes are
keyed even when the surrounding strap is also light (today-427 / today-413).

White / cream garments, thin white straps and piped trims, sheer items, and
thin chain/hook parts are protected. Image size is unchanged.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cutout_matte import decontaminate, soft_matte, thin_parts  # noqa: E402

_CAT = json.loads((ROOT / "data/catalog.json").read_text())
CAT_OF = {it["id"]: it.get("category", "") for it in _CAT["items"]}
HOLE_CATS = {"bags", "totes", "accessories"}
BAG_CATS = {"bags", "totes", "accessories"}
# Clothing keeps designed white trim (collars, hems, stripes, piping). Aggressive
# leftover-flood / 8-pass halo is for bags only (studio shelves, chain gaps).
CLOTHING_CATS = {
    "jackets", "shirts", "tshirts", "tops", "bottoms", "dresses", "shoes",
    "knitwear", "coats", "skirts", "pants", "shorts", "sweaters", "hoodies",
    "hats",
}


def qa_halo(A, near_iter=2, inner_iter=4, mn_thr=215, chroma=25, ratio_thr=0.2, a_near=40):
    """Exact QA whitescan halo mask (default args) plus optional slightly looser pass."""
    a = A[..., 3]
    rgb = A[..., :3].astype(np.int16)
    mn = rgb.min(axis=2)
    mx = rgb.max(axis=2)
    op = a > 128
    tr = a < 16
    light = (mn > mn_thr) & ((mx - mn) < chroma)
    near = ndimage.binary_dilation(tr, iterations=near_iter) & (a > a_near)
    inner = op & ~ndimage.binary_dilation(tr, iterations=inner_iter)
    innerlight = ndimage.uniform_filter((light & inner).astype(float), 9)
    innercount = ndimage.uniform_filter(inner.astype(float), 9)
    ratio = np.where(innercount > 0.05, innerlight / np.maximum(innercount, 1e-6), 1.0)
    halo = near & light & (ratio < ratio_thr)
    return halo, dict(ring=int(near.sum()), halo=int(halo.sum()),
                      halo_frac=round(int(halo.sum()) / max(1, int(near.sum())), 4))


def qa_scan(A):
    """Full QA whitescan.scan on an RGBA array (no file I/O)."""
    A16 = A.astype(np.int16)
    a = A16[..., 3]
    rgb = A16[..., :3]
    mn = rgb.min(axis=2)
    mx = rgb.max(axis=2)
    op = a > 128
    tr = a < 16
    light = (mn > 215) & ((mx - mn) < 25)
    near = ndimage.binary_dilation(tr, iterations=2) & (a > 40)
    inner = op & ~ndimage.binary_dilation(tr, iterations=4)
    innerlight = ndimage.uniform_filter((light & inner).astype(float), 9)
    innercount = ndimage.uniform_filter(inner.astype(float), 9)
    ratio = np.where(innercount > 0.05, innerlight / np.maximum(innercount, 1e-6), 1)
    halo = near & light & (ratio < 0.2)
    g = rgb.mean(axis=2)
    loc_std = np.sqrt(np.maximum(ndimage.uniform_filter(g ** 2, 5) - ndimage.uniform_filter(g, 5) ** 2, 0))
    flat = op & (mn > 200) & ((mx - mn) < 20) & (loc_std < 2.5)
    lab, n = ndimage.label(flat)
    holes = []
    if n:
        sizes = ndimage.sum(flat, lab, range(1, n + 1))
        for k, s in enumerate(sizes, 1):
            if s < 60:
                continue
            comp = lab == k
            ys, xs = np.where(comp)
            ring = ndimage.binary_dilation(comp, iterations=3) & ~comp
            if ring.sum() == 0:
                continue
            dark = (ring & op & ~light).sum() / ring.sum()
            touch_tr = (ring & tr).sum() / ring.sum()
            holes.append({"area": int(s), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                          "ring_dark": round(float(dark), 2), "ring_transparent": round(float(touch_tr), 2)})
    garment_light = float((light & op).sum() / max(1, op.sum()))
    return {"ring": int(near.sum()), "halo": int(halo.sum()),
            "halo_frac": round(int(halo.sum()) / max(1, int(near.sum())), 4),
            "holes": holes, "light_frac": round(garment_light, 3),
            "size": [int(A.shape[1]), int(A.shape[0])]}


def _local_std(L, k=5):
    Lf = L.astype(np.float32)
    return np.sqrt(np.maximum(cv2.blur(Lf ** 2, (k, k)) - cv2.blur(Lf, (k, k)) ** 2, 0))


def designed_light(alpha, L, sat, min_thick=3.0, bag_shelf=False):
    """Protect the interior of thick cream/white structures (handles, collars,
    soles, piping) plus a 2px collar so we do not nibble their edge. A connected
    paper rim that merely *touches* a thicker blob is not fully protected — only
    the thick core is. For bags, also drop protection on pixels that sit strictly
    below the bag body (studio shelves)."""
    light = (alpha > 0.5) & (L >= 188) & (sat <= 28)
    prot = np.zeros(alpha.shape, bool)
    if not light.any():
        return prot
    dt = cv2.distanceTransform(light.astype(np.uint8), cv2.DIST_L2, 3)
    core = dt >= min_thick
    if core.any():
        k = np.ones((5, 5), np.uint8)
        prot = (cv2.dilate(core.astype(np.uint8), k, iterations=1) > 0) & light
    if bag_shelf:
        # Bags: protect the ENTIRE upper cream handle component (thin mid-strap
        # sections included). Dead-white studio mattes (sat ~ 0) stay peelable.
        ys, _ = np.where(alpha > 0.5)
        prot[:] = False
        if ys.size and light.any():
            y0, y1 = int(ys.min()), int(ys.max())
            n, lab, st, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8), 8)
            dt = cv2.distanceTransform(light.astype(np.uint8), cv2.DIST_L2, 3)
            for j in range(1, n):
                cj = lab == j
                cy = st[j, 1] + st[j, 3] / 2.0
                cyrel = (cy - y0) / max(1, y1 - y0)
                if cyrel > 0.55:
                    continue
                if float(np.median(sat[cj])) < 10:
                    continue
                if float(dt[cj].max()) < 2.5 and st[j, 4] < 400:
                    continue
                prot |= cj
    return prot


def paper_fringe(alpha, L, sat, protect, max_thick=2.5, ring=3, min_near=0.85):
    """Thin light components that live almost entirely in the outer transparent
    ring — leftover studio paper. Stripes, collars and handles fail this test
    (they extend inside the item, or they are thicker than a 2px rim)."""
    light = (alpha > 0.5) & (L >= 200) & (sat <= 25) & ~protect
    fringe = np.zeros(alpha.shape, bool)
    if not light.any():
        return fringe
    tr = alpha < 16 / 255.0
    k = np.ones((3, 3), np.uint8)
    near = cv2.dilate(tr.astype(np.uint8), k, iterations=ring) > 0
    dt = cv2.distanceTransform(light.astype(np.uint8), cv2.DIST_L2, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8), 8)
    for j in range(1, n):
        cj = lab == j
        if st[j, 4] < 4:
            continue
        mx = float(dt[cj].max()) if cj.any() else 0
        if mx > max_thick:
            continue
        frac = float((cj & near).sum()) / float(cj.sum())
        if frac >= min_near:
            fringe |= cj
    return fringe




def fix_card(A, iid=""):
    """Return cleaned RGBA and a stats dict. Never changes HxW."""
    A = A.copy()
    rgb = A[..., :3].copy()
    a0 = A[..., 3].astype(np.float32) / 255.0
    alpha = a0.copy()
    H, W = alpha.shape
    if (a0 > 0.5).sum() < 100:
        return A, {"skipped": "too few opaque pixels"}
    cat = CAT_OF.get(iid, "")
    vis = a0 > 0.05
    sheer = vis.sum() > 0 and ((a0 > 0.05) & (a0 < 0.95)).sum() > 0.30 * vis.sum()
    if sheer:
        # organza / clear bags: do not peel or key holes (would punch the film)
        return A.copy(), {"skipped": "sheer/clear material", "cover_loss": 0.0,
                          "hole_px": 0, "peel_px": 0, "band_px": 0, "dropped_px": 0,
                          "opaque0": int((a0 > 0.5).sum()), "opaque1": int((a0 > 0.5).sum()),
                          "body_L": 0, "mode": "sheer"}

    f = rgb.astype(np.int16)
    L = f.mean(2)
    sat = (f.max(2) - f.min(2)).astype(np.int16)
    k3 = np.ones((3, 3), np.uint8)

    def pack():
        return np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])

    # --- 1) Enclosed studio-white holes (strap/chain loops), including when the
    # strap itself is white (today-427). Never drop a component that is the
    # garment body (light-wash panels, white shirts).
    op = alpha > 0.5
    opaque0 = int(op.sum())
    allow_holes = cat in HOLE_CATS
    mn = f.min(2)
    mx = f.max(2)
    locstd = _local_std(L, 5)
    # QA-style flat studio backdrop (locstd is low in a true hole interior;
    # leaving it out merges today-427's hole with the white strap so the
    # component touches transparency and gets skipped)
    flat_white = op & (mn > 200) & ((mx - mn) < 20) & (locstd < 2.5)
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(flat_white.astype(np.uint8), 4)
    holes = np.zeros(alpha.shape, bool)
    hole_areas = []
    for j in range(1, nh):
        area = int(sh[j, 4])
        # small "holes" are often a charm's face or a highlight (today-406 Hello Kitty),
        # not a strap-loop backdrop. Only key large enclosed fills.
        if area < 3000:
            continue
        if not allow_holes:
            continue
        cj = lh == j
        x, y, w, h, _ = sh[j]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        ring = (cv2.dilate(cj.astype(np.uint8), k3, iterations=3) > 0) & ~cj
        if ring.sum() == 0:
            continue
        # handle-loop holes sit in the upper half of the item (today-427 / today-413).
        # tote-print panels and shoe insoles sit mid/low — skip those.
        ys, _xs = np.where(op)
        y0, y1 = int(ys.min()), int(ys.max())
        cy = y + h / 2.0
        if y1 > y0 and (cy - y0) / (y1 - y0) > 0.55:
            continue
        tr = alpha < 16 / 255.0
        touch_tr = float((ring & tr).sum()) / float(ring.sum())
        if touch_tr > 0.08:
            continue
        remain = op & ~cj & ~holes
        darker = remain & ((L < 200) | (sat > 25))
        # no substantial darker body → this is a light garment panel, not a loop hole
        if darker.sum() < 0.10 * opaque0:
            continue
        holes |= cj
        hole_areas.append(area)
    hole_color = None
    if holes.any():
        grow = cv2.dilate(holes.astype(np.uint8), k3, iterations=2) > 0
        holes |= grow & op & (L >= 215) & (sat <= 20)
        remain = op & ~holes
        darker = remain & ((L < 200) | (sat > 25))
        if remain.sum() < 0.25 * max(1, opaque0) and darker.sum() < 0.08 * max(1, opaque0):
            holes[:] = False
            hole_areas = []
        else:
            hole_color = (float(np.median(L[holes])),
                          tuple(np.median(f[holes], 0).tolist()))
            alpha[holes] = 0.0
    large_hole = int(sum(hole_areas)) >= 3000 if hole_areas else False

    # classify the remaining body: light garments get a conservative halo-only
    # peel so cream sweaters / light-wash jeans are not dissolved.
    body = (alpha > 0.5)
    core = cv2.erode(body.astype(np.uint8), k3, iterations=5) > 0
    if core.sum() < 80:
        core = cv2.erode(body.astype(np.uint8), k3, iterations=2) > 0
    body_L = float(np.median(L[core])) if core.sum() else float(np.median(L[body])) if body.any() else 0
    # light fraction of the *remaining* body (after any loop-hole keying) so a
    # white fill inside a strap does not classify a dark bag as a light garment
    light_frac0 = float(((L >= 200) & (sat <= 25) & (alpha > 0.5)).sum()) / max(1, int((alpha > 0.5).sum()))
    is_clothing = cat in CLOTHING_CATS or cat == "shoes"
    is_bag = cat in BAG_CATS
    is_light_body = body_L >= 185 or (light_frac0 >= 0.15 and not is_bag)
    is_dark_body = (body_L <= 170) and (light_frac0 < 0.12) and not is_light_body
    protect = designed_light(alpha, L, sat, bag_shelf=is_bag)

    peel_total = np.zeros(alpha.shape, bool)
    band_total = np.zeros(alpha.shape, bool)

    # spine of thin parts (white straps, chains, hooks) — never eat the core
    def thin_spine(al):
        th = thin_parts(np.maximum(al, a0), rgb)
        return cv2.erode(th.astype(np.uint8), k3, iterations=1) > 0

    # --- 2) Thin / exact halo peel
    # large_hole (427/413): 1 exact QA pass like the proven test16 path (the 2px
    # grow already happened; 8-pass+flood would eat a white strap).
    # clothing: peel only achromatic paper (sat<=4) on the QA halo / fringe so
    # a ringer collar (sat~6–15) is kept and a dead-white matte rim (sat=0)
    # around a black jacket comes off.
    # bags: paper_fringe here; thicker studio shelves are handled in 3b/3c.
    if large_hole:
        Aa = pack()
        halo, _ = qa_halo(Aa)
        halo &= ~thin_spine(alpha)
        if halo.any():
            peel_total |= halo
            alpha[halo] = 0.0
    elif is_clothing and cat != "shoes" and not is_light_body:
        Aa = pack()
        halo, _ = qa_halo(Aa)
        fringe = paper_fringe(alpha, L, sat, protect)
        peel = (halo | fringe) & (sat <= 4) & ~protect & ~thin_spine(alpha)
        if peel.any():
            peel_total |= peel
            alpha[peel] = 0.0
    elif is_bag:
        fringe = paper_fringe(alpha, L, sat, protect)
        if fringe.any():
            peel_total |= fringe
            alpha[fringe] = 0.0
    # light garments and shoes: no alpha peel here (recolor at the end for light)

    # --- 3) Extra peel for leftover bands / bag shadows
    tr = alpha < 16 / 255.0
    leftover = (alpha > 0.12) & (L >= 188) & (sat <= 28)

    # 3a. If we keyed a hole, flood ONLY into pixels matching that backdrop colour
    # (cleans the leftover blobs that were attached to the loop, without eating
    # cream sherpa / white straps).
    if False and hole_color is not None:  # flood ate 427 strap/piping; 2px grow is enough
        hL, hrgb = hole_color
        hrgb = np.array(hrgb)
        distc = np.abs(f.astype(np.float32) - hrgb).sum(2)
        # tight colour + only a short grow so a white strap (today-427) is not eaten
        match = (alpha > 0.12) & (distc <= 24) & (np.abs(L - hL) <= 10) & (sat <= 16)
        seed = tr | holes
        reached = seed.astype(np.uint8)
        for _ in range(4):
            nxt = cv2.dilate(reached, k3, iterations=1)
            nxt = ((nxt > 0) & (match | (reached > 0) | tr | holes)).astype(np.uint8)
            if np.array_equal(nxt, reached):
                break
            reached = nxt
        band = (reached > 0) & match & ~tr & ~holes
        band &= ~thin_spine(alpha)
        # cap: this matching flood should stay small relative to the body
        if band.sum() > 0.25 * max(1, int((alpha > 0.5).sum())):
            band[:] = False
        band_total |= band
        peel_total |= band
        alpha[band] = 0.0

    # 3a2. After a large loop hole is opened, drop leftover paper-white scraps
    # that now touch transparency (today-413 strap-junction blobs) — small only.
    if large_hole:
        op_ = alpha > 0.5
        tr = alpha < 16 / 255.0
        locstd = _local_std(L, 5)
        paper = op_ & (L >= 225) & (sat <= 16) & (locstd < 2.8)
        nh, lh, sh, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8), 4)
        scraps = np.zeros(alpha.shape, bool)
        for j in range(1, nh):
            ar = int(sh[j, 4])
            if ar < 40 or ar > 2500:
                continue
            cj = lh == j
            ring = (cv2.dilate(cj.astype(np.uint8), k3, iterations=2) > 0) & ~cj
            touch = float((ring & tr).sum()) / float(ring.sum()) if ring.sum() else 0
            # leftover blobs sit mostly in the open (high touch_tr); piping/strap do not
            if touch >= 0.35:
                scraps |= cj
        if scraps.any():
            band_total |= scraps
            peel_total |= scraps
            alpha[scraps] = 0.0

    # 3a3. Dark clothing: flood only dead-white (sat<=4) leftover from
    # transparency toward the dark garment, so a thick studio matte around a
    # black jacket (today-237) is removed without walking into a ringer collar.
    if is_clothing and cat != "shoes" and is_dark_body and not large_hole:
        tr = alpha < 16 / 255.0
        leftover = (alpha > 0.12) & (L >= 200) & (sat <= 4)
        item = (alpha > 0.5) & ((L < 185) | (sat > 8))
        near2 = cv2.dilate(tr.astype(np.uint8), k3, iterations=2) > 0
        seed = tr | (leftover & near2)
        floodable = leftover  # sat<=4 already excludes ringer/cream trim
        if item.sum() > 50:
            dist_item = cv2.distanceTransform((~item).astype(np.uint8), cv2.DIST_L2, 5)
            floodable = floodable & (dist_item <= 24)
        reached = seed.astype(np.uint8)
        for _ in range(20):
            nxt = cv2.dilate(reached, k3, iterations=1)
            nxt = ((nxt > 0) & (floodable | (reached > 0) | tr)).astype(np.uint8)
            if np.array_equal(nxt, reached):
                break
            reached = nxt
        band = (reached > 0) & floodable & ~tr
        band &= ~thin_spine(alpha)
        if band.any() and band.sum() <= 0.20 * max(1, int((alpha > 0.5).sum())):
            band_total |= band
            peel_total |= band
            alpha[band] = 0.0

    # 3b. Dark bags: flood leftover from the rim toward a dark core, and strip
    # a much-lighter low-sat rim up to 32px (bag-bottom shadows, inner chain).
    if is_bag and is_dark_body and not large_hole:
        tr = alpha < 16 / 255.0
        leftover = (alpha > 0.12) & (L >= 200) & (sat <= 12)
        item = (alpha > 0.5) & ((L < 185) | (sat > 14))
        near2 = cv2.dilate(tr.astype(np.uint8), k3, iterations=2) > 0
        seed = tr | (leftover & near2) | peel_total
        sample = leftover & (near2 | peel_total)
        if sample.sum() >= 20:
            sampL = float(np.median(L[sample]))
            floodable = leftover & (np.abs(L - sampL) <= 20) & (L >= sampL - 6)
        else:
            floodable = leftover & (L >= 205)
        if item.sum() > 50:
            dist_item = cv2.distanceTransform((~item).astype(np.uint8), cv2.DIST_L2, 5)
            floodable = floodable & ((dist_item <= 26) | near2)
        reached = seed.astype(np.uint8)
        for _ in range(36):
            nxt = cv2.dilate(reached, k3, iterations=1)
            nxt = ((nxt > 0) & (floodable | (reached > 0) | tr)).astype(np.uint8)
            if np.array_equal(nxt, reached):
                break
            reached = nxt
        band = (reached > 0) & floodable & ~tr
        # dark-core rim strip
        core_d = (cv2.erode((alpha > 0.5).astype(np.uint8), k3, iterations=4) > 0) & (L < 185) & (sat > 8)
        if not core_d.any():
            core_d = (cv2.erode((alpha > 0.5).astype(np.uint8), k3, iterations=4) > 0) & (L < 175)
        if core_d.any():
            cL = float(np.median(L[core_d]))
            dist_tr = cv2.distanceTransform((~tr).astype(np.uint8), cv2.DIST_L2, 5)
            paper = (alpha > 0.12) & (sat <= 12) & (L >= 195) & (L > cL + 50) & ~core_d
            dt_p = cv2.distanceTransform(paper.astype(np.uint8), cv2.DIST_L2, 3) if paper.any() else np.zeros(alpha.shape, np.float32)
            thick = paper & (dt_p >= 4) & (dist_tr <= 32)
            if thick.any():
                rim = (cv2.dilate(thick.astype(np.uint8), k3, iterations=4) > 0) & paper & (dist_tr <= 32)
                band |= rim
        band &= ~thin_spine(alpha)
        band &= ~protect
        band_total |= band
        peel_total |= band
        alpha[band] = 0.0

    if is_bag and not is_light_body:
        # 3c. Column-wise drop-shadow: light low-sat pixels that sit strictly below
        # the lowest non-paper object pixel in that column (bag-bottom white shelves).
        # Charms / Hello Kitty are "object" (colour or texture) so they are not shaved.
        # White piping / straps (protect) count as object so 427's bottom welt stays.
        op_ = alpha > 0.5
        object_px = op_ & (~((L >= 200) & (sat <= 22)) | protect)
        shelf = np.zeros(alpha.shape, bool)
        # vectorised: lowest object y per column
        obj = object_px
        if obj.any():
            rows = np.arange(H)[:, None]
            has = obj.any(0)
            # max y of object in each column (0 if none)
            maxy = np.where(has, (obj * rows).max(0), -1)
            below = (rows > maxy[None, :]) & (maxy[None, :] >= 0)
            shelf = below & (alpha > 0.12) & (L >= 190) & (sat <= 26)
            # don't climb more than 40px above the object's foot (keeps this local)
            shelf &= (rows <= maxy[None, :] + 0) | True
            shelf &= (rows - maxy[None, :] <= 40)
            shelf &= ~protect
            if shelf.any():
                band_total |= shelf
                peel_total |= shelf
                alpha[shelf] = 0.0

    # 3d. Bags: leftover achromatic paper components that still touch
    # transparency (side mattes that sit next to, not strictly under, the bag).
    if is_bag and is_dark_body and not large_hole:
        op_ = alpha > 0.5
        tr = alpha < 16 / 255.0
        paper = op_ & (L >= 200) & (sat <= 8) & ~protect
        if paper.any():
            ys, _ = np.where(op_)
            y0, y1 = int(ys.min()), int(ys.max()) if ys.size else (0, 1)
            n, lab, st, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8), 8)
            extra = np.zeros(alpha.shape, bool)
            for j in range(1, n):
                ar = int(st[j, 4])
                if ar < 40 or ar > 8000:
                    continue
                cj = lab == j
                cy = st[j, 1] + st[j, 3] / 2.0
                cyrel = (cy - y0) / max(1, y1 - y0)
                ring = (cv2.dilate(cj.astype(np.uint8), k3, iterations=2) > 0) & ~cj
                touch = float((ring & tr).sum()) / float(ring.sum()) if ring.sum() else 0
                if touch >= 0.15 and cyrel >= 0.35:
                    extra |= cj
            extra &= ~protect
            if extra.any():
                band_total |= extra
                peel_total |= extra
                alpha[extra] = 0.0

    # safety net: if we ate more than 12% of the non-hole body, revert the peel
    # and keep only holes + exact QA halo (something went wrong on a mid-tone item)
    body0 = opaque0 - int(sum(hole_areas) if hole_areas else 0)
    body1 = int((alpha > 0.5).sum())
    if body0 > 0 and (body0 - body1) / body0 > 0.12 and not large_hole:
        alpha = a0.copy()
        if holes.any():
            alpha[holes] = 0.0
        peel_total[:] = False
        band_total[:] = False
        fringe = paper_fringe(alpha, L, sat, protect)
        if fringe.any():
            peel_total |= fringe
            alpha[fringe] = 0.0

    # --- 4) Tiny leftover specks
    vis = (alpha > 0.1).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(vis, 8)
    dropped = 0
    if n > 2:
        big = int(st[1:, 4].max())
        for j in range(1, n):
            ar = int(st[j, 4])
            if ar >= 0.15 * big:
                continue
            cj = lab == j
            med = np.median(f[cj], 0)
            mL = float(med.mean())
            ms = float(med.max() - med.min())
            if ar < 20 or min(st[j, 2], st[j, 3]) <= 2 or (mL > 198 and ms < 24 and ar < 0.05 * big):
                alpha[cj] = 0.0
                dropped += ar

    # --- 5) Smooth the new edge slightly (never grow); keep thin real parts
    # Skip on light garments: shrink/recolor at the inseam mottles striped pants (today-327).
    m2 = (alpha > 0.5).astype(np.uint8)
    if m2.sum() >= 50 and is_bag and not is_light_body:
        soft = soft_matte(m2, shrink=0.7)
        band_e = (m2 > 0) & (cv2.erode(m2, k3, iterations=2) == 0)
        band_e |= (alpha > 0) & (m2 == 0)
        sm = np.minimum(alpha, soft)
        thin2 = thin_parts(alpha, rgb)
        sm = np.where(thin2, alpha, sm)
        alpha = np.where(band_e, sm, alpha)

    # --- 6) never detach a hook/strap tip that belonged to the main body
    mo = (a0 > 0.5).astype(np.uint8)
    mn_ = (alpha > 0.5).astype(np.uint8)
    nn, ln, sn, _ = cv2.connectedComponentsWithStats(mn_, 8)
    if nn > 2:
        no, lo, so, _ = cv2.connectedComponentsWithStats(mo, 8)
        main_o = lo == 1 + int(so[1:, 4].argmax())
        big = sn[1:, 4].max()
        removed = ((mo > 0) & (mn_ == 0) & ~holes).astype(np.uint8)
        nr, lr, _, _ = cv2.connectedComponentsWithStats(removed, 8)
        give = np.zeros_like(main_o)
        for j in range(1, nn):
            if sn[j, 4] >= 0.2 * big:
                continue
            cj = ln == j
            if not (cj & main_o).any():
                continue
            touch = np.unique(lr[(cv2.dilate(cj.astype(np.uint8), k3, iterations=1) > 0) & (lr > 0)])
            give |= np.isin(lr, touch[touch > 0])
        give &= ~((L >= 188) & (sat <= 28))
        alpha = np.where(give, a0, alpha)

    # --- 7) decontaminate remaining rim colour
    if is_bag and not is_light_body:
        rgb = decontaminate(rgb, alpha, depth=3, band=4, wall=None)
    # Recolor leftover paper-fringe colour on bags / dark clothing, but never
    # recolor designed white trim (collars, handles, soles) — that would dye them.
    if True:
        # Light garments: recolour remaining QA-halo from the (also light) core
        # so a cream edge is no longer flagged, without shrinking stripes.
        # Dark clothing / shoes: only recolour paper_fringe, never designed trim.
        tmp = np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
        if is_light_body:
            rem, _ = qa_halo(tmp)
        else:
            rem = paper_fringe(alpha, L, sat, protect)
        rem = rem & ~protect
        if rem.any():
            solid = (alpha >= 0.98).astype(np.uint8)
            core = cv2.erode(solid, k3, iterations=4)
            if core.sum() >= 30:
                filled = cv2.inpaint(rgb, (core == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
                # recolour every remaining QA-halo pixel from the garment core
                # (neckline, hem, armholes included). Alpha is unchanged, so
                # patterned inseams are not eaten.
                rgb = rgb.copy()
                rgb[rem] = filled[rem]
    out = np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
    out[out[..., 3] == 0, :3] = 0

    opaque1 = int((out[..., 3] > 128).sum())
    stt = dict(
        hole_px=int(sum(hole_areas)) if hole_areas else int(holes.sum()),
        peel_px=int(peel_total.sum()),
        band_px=int(band_total.sum()),
        dropped_px=int(dropped),
        cover_loss=round(1 - opaque1 / max(1, opaque0), 4),
        opaque0=opaque0,
        opaque1=opaque1,
        body_L=round(body_L, 1),
        mode="light" if is_light_body else ("dark" if is_dark_body else "mid"),
    )
    return out, stt



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="*")
    ap.add_argument("--ids-json", help="fix2/ids-108.json; process all + 400 variants")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--preview", help="directory for PNG previews")
    ap.add_argument("--metrics", help="write before/after QA scan JSON")
    ap.add_argument("--root", default=str(ROOT))
    o = ap.parse_args()
    root = Path(o.root)
    srcs = list(o.srcs)
    if o.ids_json:
        ids = json.loads(Path(o.ids_json).read_text())["all"]
        for iid in ids:
            for suf in ("-600.webp", "-400.webp"):
                p = Path("assets/items/v2") / f"{iid}{suf}"
                if (root / p).exists():
                    srcs.append(str(p))
    if not srcs:
        sys.exit("no sources")
    log = {}
    for src in srcs:
        A = np.array(Image.open(root / src).convert("RGBA"))
        before = qa_scan(A)
        out, st = fix_card(A, Path(src).stem.rsplit("-", 1)[0])
        after = qa_scan(out)
        rec = dict(before=before, after=after, **st)
        log[src] = rec
        if o.preview:
            Path(o.preview).mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(f"{o.preview}/{Path(src).stem}.png")
        if o.write:
            Image.fromarray(out).save(root / src, "WEBP", quality=90, method=6)
    if o.metrics:
        Path(o.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(o.metrics).write_text(json.dumps(log, indent=1) + "\n")
    # summary
    n = len(log)
    h0 = sum(v["before"]["halo"] for v in log.values())
    h1 = sum(v["after"]["halo"] for v in log.values())
    still = [k for k, v in log.items() if k.endswith("-600.webp") and v["after"]["halo"] >= 100]
    holes1 = [k for k, v in log.items() if any(h.get("ring_transparent", 1) <= 0.08 and h["area"] >= 200 for h in v["after"]["holes"])]
    print(f"{n} files; halo {h0} -> {h1}; 600 still halo>=100: {len(still)}; enclosed holes remain: {len(holes1)}")
    if still:
        for k in still[:20]:
            print("  still", k, log[k]["before"]["halo"], "->", log[k]["after"]["halo"], "loss", log[k]["cover_loss"])
    if holes1:
        for k in holes1[:10]:
            print("  hole", k, log[k]["after"]["holes"][:3])


if __name__ == "__main__":
    main()
