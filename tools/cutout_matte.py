#!/usr/bin/env python3
"""Shared matting helpers for the 2026-11-24 cutout fixes (P2-P5).

soft_matte   : binary/rough mask -> smooth anti-aliased alpha (kills stair steps), optional inward shrink
drop_bits    : remove small detached alpha components (keeps every component >= keep_frac of the largest)
decontaminate: replace RGB in the edge band with colour pulled from >= `depth` px inside the item,
               so old-painting wall / backdrop colour cannot show as a light rim
defringe_rgba: full pass used by P4 on existing layers
"""
import numpy as np, cv2

def soft_matte(mask, shrink=0.0, up=4, sigma=1.1):
    m = (mask > 0).astype(np.float32)
    h, w = m.shape
    big = cv2.resize(m, (w * up, h * up), interpolation=cv2.INTER_NEAREST)
    big = cv2.GaussianBlur(big, (0, 0), sigma * up)
    thr = 0.5 + 0.12 * shrink
    big = np.clip((big - thr) / 0.18 + 0.5, 0, 1)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)

def drop_bits(alpha, keep_frac=0.2, min_px=12):
    m = (alpha > 0.25).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return alpha, 0
    areas = st[1:, 4]; big = areas.max()
    keep = np.zeros(n, bool); keep[0] = False
    for j in range(1, n):
        keep[j] = areas[j - 1] >= max(min_px, keep_frac * big)
    # pixels with low alpha not in any component stay as they are only if adjacent to kept comps
    kept = keep[lab]
    kept = cv2.dilate(kept.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dropped = int(((~kept) & (alpha > 0.02)).sum())
    return np.where(kept, alpha, 0).astype(np.float32), dropped

def decontaminate(rgb, alpha, depth=2, band=3, wall=None):
    """Edge-band pixels (alpha<0.98 or within `band` px of the edge) that look like backdrop/wall
    (light, low saturation, clearly lighter than the interior colour behind them, or close to `wall`)
    get colour inpainted from the interior. Dark painted outlines are kept."""
    solid = (alpha >= 0.98).astype(np.uint8)
    core = cv2.erode(solid, np.ones((3, 3), np.uint8), iterations=depth)
    if core.sum() < 30:
        return rgb
    edge = (alpha > 0.0) & ~(cv2.erode(solid, np.ones((3, 3), np.uint8), iterations=band) > 0)
    src = rgb.copy()
    hole = ((core == 0)).astype(np.uint8)
    filled = cv2.inpaint(src, hole, 4, cv2.INPAINT_TELEA)
    f = rgb.astype(np.int16); g = filled.astype(np.int16)
    L = f.mean(2); sat = f.max(2) - f.min(2); Lf = g.mean(2)
    bad = (sat < 32) & (L > 125) & (L - Lf > 14)
    if wall is not None:
        bad |= np.abs(f - np.asarray(wall, np.int16)).max(2) < 22
    sel = edge & bad
    out = rgb.copy()
    out[sel] = filled[sel]
    return out

def defringe_rgba(rgba, shrink=1.0, keep_frac=0.2, depth=2, band=2, wall=None):
    """Existing layer -> smoother alpha, no detached bits, edge colour from the interior."""
    rgb = rgba[..., :3].copy(); a = rgba[..., 3].astype(np.float32) / 255
    alpha = soft_matte(a > 0.5, shrink=shrink)
    alpha = np.minimum(alpha, cv2.dilate(a, np.ones((3, 3), np.uint8)))  # never grow past old coverage
    alpha, dropped = drop_bits(alpha, keep_frac=keep_frac)
    rgb = decontaminate(rgb, alpha, depth=depth, band=band, wall=wall)
    out = np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
    out[out[..., 3] == 0, :3] = 0
    return out, dropped

def rgba_from_mask(rgb, mask, shrink=0.5, wall=None):
    alpha = soft_matte(mask, shrink=shrink)
    alpha, _ = drop_bits(alpha)
    rgb = decontaminate(rgb.copy(), alpha, depth=2, band=2, wall=wall)
    out = np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
    out[out[..., 3] == 0, :3] = 0
    return out
