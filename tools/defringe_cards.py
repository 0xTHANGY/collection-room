#!/usr/bin/env python3
"""P6 (2026-11-24 cutout fix): clean card / detail-page images (assets/items/v2/<id>-600|400.webp).
Product-photo cutouts keep a light white-backdrop halo, faint detached shadow/backdrop specks and
semi-transparent light ghosts. Per image:
  * edge band only (<=3px from the solid edge): alpha = min(old, smooth matte) -> no stair steps, never grows
  * thin parts (chains, hooks, cords) keep their old alpha
  * detached components < 20% of the main part are dropped only if light & low-saturation (backdrop/shadow)
    or < 15 px; darker/colourful detached parts (chain pieces, second shoe, charms) are kept
  * semi-transparent (<0.5) light low-sat pixels clearly lighter than the item behind them -> transparent
  * light halo pixels in the edge band recoloured from the interior (cutout_matte.decontaminate)
Image size unchanged. Reads originals from --base (default 326fc1d) so re-runs are idempotent.
Usage: defringe_cards.py [--write] [--preview DIR] src [src ...]   (src = assets/items/v2/xxx-600.webp)"""
import argparse, io, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cutout_matte import soft_matte, decontaminate, thin_parts

# opaque garments whose card was keyed against magenta so the whole body came out semi-transparent
# (pink/red key spill, eye-confirmed): interior made solid, edge kept. Clear/organza items are left alone.
KEY_SPILL_SOLID = {"cr-008", "batch2-331", "today-235", "today-189"}

def solidify(A):
    a = A[..., 3]
    m = cv2.morphologyEx((a > 20).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, l, s, _ = cv2.connectedComponentsWithStats(m, 8)
    main = (l == 1 + s[1:, 4].argmax()).astype(np.uint8)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(1 - main, 4)
    for j in range(1, n2):
        x, y, w, h, _ = s2[j]
        if not (x == 0 or y == 0 or x + w >= a.shape[1] or y + h >= a.shape[0]):
            main[l2 == j] = 1
    core = cv2.erode(main, np.ones((3, 3), np.uint8), iterations=2) > 0
    B = A.copy(); B[..., 3] = np.where(core, 255, a)
    return B

def fix_card(A, iid=""):
    solid = False
    if iid in KEY_SPILL_SOLID:
        A = solidify(A); solid = True
    rgb = A[..., :3].copy(); a = A[..., 3].astype(np.float32) / 255
    m = (a > 0.5).astype(np.uint8)
    if m.sum() < 100:
        return A.copy(), {}
    vis = a > 0.05
    if ((a > 0.05) & (a < 0.95)).sum() > 0.3 * max(1, vis.sum()):
        return A.copy(), dict(skipped="mostly semi-transparent (clear/sheer item or pre-existing see-through matte)")
    f = rgb.astype(np.int16); L = f.mean(2); sat = f.max(2) - f.min(2)
    core = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=3)
    filled = cv2.inpaint(rgb, (core == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.int16).mean(2) if core.sum() > 50 else L
    lightish = (L > 185) & (sat < 30)
    alpha = a.copy()
    # 0) enclosed / edge backdrop: flat near-white photo backdrop (e.g. inside chain-strap loops) -> transparent
    std = cv2.blur(L.astype(np.float32) ** 2, (5, 5)) - cv2.blur(L.astype(np.float32), (5, 5)) ** 2
    flat_white = (a > 0.5) & (L >= 243) & (sat <= 8) & (std <= 4)
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(flat_white.astype(np.uint8), 4)
    holes = np.zeros_like(flat_white)
    for j in range(1, nh):
        if sh[j, 4] >= 60:
            cj = lh == j
            # only a real backdrop hole: what surrounds it is mostly strap/item edge or transparency,
            # not more light fabric (a flat white tee or bag panel is never keyed out)
            c8 = cj.astype(np.uint8)
            ring = (cv2.dilate(c8, np.ones((3, 3), np.uint8), iterations=8) > 0) & ~(cv2.dilate(c8, np.ones((3, 3), np.uint8), iterations=3) > 0)
            notlight = ring & ((L < 200) | (a < 0.5))
            if notlight.sum() >= 0.5 * max(1, ring.sum()):
                holes |= cj
    holes_px = int(holes.sum())
    if holes_px:
        grow = cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        holes |= grow & (L >= 225) & (sat <= 14)
        alpha[holes] = 0
    # bright halo ring (up to 4px) around a clearly darker item
    for _ in range(4):
        mm = (alpha > 0.5).astype(np.uint8)
        edge = (mm > 0) & (cv2.erode(mm, np.ones((3, 3), np.uint8)) == 0)
        ring = edge & (L >= 228) & (sat <= 14) & (L - filled > 60)
        if not ring.any():
            break
        alpha[ring] = 0
    # 1) semi-transparent light ghosts
    ghost = (a > 0) & (a < 0.5) & lightish & (L - filled > 20)
    if ghost.sum() > 0.03 * m.sum():  # lots of light translucent pixels = sheer/clear material, keep it
        ghost[:] = False
    alpha[ghost] = 0
    # 2) detached light specks
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 0.1).astype(np.uint8), 8)
    dropped = 0
    if n > 2:
        big = st[1:, 4].max()
        for j in range(1, n):
            ar = st[j, 4]
            if ar >= 0.2 * big:
                continue
            cj = lab == j
            med = np.median(f[cj], 0); mL = med.mean(); ms = med.max() - med.min()
            if ar < 15 or min(st[j, 2], st[j, 3]) <= 1 or (mL > 185 and ms < 30):
                alpha[cj] = 0; dropped += int(ar)
    # 3) smooth the edge band, never grow; keep thin parts
    m2 = (alpha > 0.5).astype(np.uint8)
    soft = soft_matte(m2, shrink=0.5)
    band = (m2 > 0) & (cv2.erode(m2, np.ones((3, 3), np.uint8), iterations=3) == 0)
    band |= (alpha > 0) & (m2 == 0)
    sm = np.minimum(alpha, soft)
    thin = thin_parts(a, rgb)
    sm = np.where(thin, alpha, sm)
    alpha = np.where(band, sm, alpha)
    # 3b) never break a part off: if a cleaned piece became detached from the main body (hanger hooks,
    #     strap tips), give back every removed pixel run that touches it
    mo = (a > 0.5).astype(np.uint8)
    mn = (alpha > 0.5).astype(np.uint8)
    nn, ln, sn, _ = cv2.connectedComponentsWithStats(mn, 8)
    if nn > 2:
        no, lo, so, _ = cv2.connectedComponentsWithStats(mo, 8)
        main_o = lo == 1 + int(so[1:, 4].argmax())
        big = sn[1:, 4].max()
        removed = ((mo > 0) & (mn == 0) & ~holes).astype(np.uint8)
        nr, lr, _, _ = cv2.connectedComponentsWithStats(removed, 8)
        give = np.zeros_like(main_o)
        for j in range(1, nn):
            if sn[j, 4] >= 0.2 * big:
                continue
            cj = ln == j
            if not (cj & main_o).any():
                continue  # it was already separate in the original
            touch = np.unique(lr[(cv2.dilate(cj.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0) & (lr > 0)])
            give |= np.isin(lr, touch[touch > 0])
        alpha = np.where(give, a, alpha)
    # 4) halo colour
    rgb = decontaminate(rgb, alpha, depth=2, band=2, wall=None)
    out = np.dstack([rgb, np.clip(alpha * 255 + .5, 0, 255).astype(np.uint8)])
    out[out[..., 3] == 0, :3] = 0
    st = dict(ghost_px=int(ghost.sum()), dropped_px=dropped, backdrop_px=holes_px)
    if solid:
        st["solidified"] = True
    return out, st

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("srcs", nargs="+"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--preview"); ap.add_argument("--base", default="326fc1d")
    o = ap.parse_args(); log = {}
    for src in o.srcs:
        b = subprocess.run(["git", "-C", str(ROOT), "show", f"{o.base}:{src}"], capture_output=True, check=True).stdout
        A = np.array(Image.open(io.BytesIO(b)).convert("RGBA"))
        out, st = fix_card(A, Path(src).stem.rsplit('-', 1)[0])
        log[src] = st
        if o.preview:
            Path(o.preview).mkdir(parents=True, exist_ok=True); Image.fromarray(out).save(f"{o.preview}/{Path(src).stem}.png")
        if o.write:
            Image.fromarray(out).save(ROOT / src, "WEBP", quality=90, method=6)
    if o.write:
        p = ROOT / "data/source/display/defringe-cards.json"
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update(log); p.write_text(json.dumps(prev, indent=1, sort_keys=True) + "\n")
    print(len(log), "cards;", sum(1 for v in log.values() if v.get("dropped_px")), "with dropped specks;",
          sum(v.get("ghost_px", 0) for v in log.values()), "ghost px")

if __name__ == "__main__":
    main()
