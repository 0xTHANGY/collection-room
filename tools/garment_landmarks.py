#!/usr/bin/env python3
"""Visible garment landmarks on a layer image (fractions of image height).

shoulder: first row (below hanger/hood/collar) whose opaque span reaches SHOULDER_FRAC of the
          chest span (row at 35-45 % of the garment). Hood and hanger rows are narrower.
hem:      lowest row whose opaque span is >= HEM_FRAC of chest span (ignores dangling straps/cords).
Body length in px = (hem - shoulder) * layer height; this is what the published 着丈/body length
(measured from the high shoulder point, hood excluded) corresponds to.
OVERRIDE holds eye-checked values where the profile is fooled (e.g. dropped shoulders).
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SHOULDER_FRAC = 0.80
HEM_FRAC = 0.35
OVERRIDE: dict[str, tuple[float, float]] = {}


def spans(a):
    out = []
    for row in a:
        xs = np.where(row)[0]
        out.append((xs[-1] - xs[0] + 1) if len(xs) else 0)
    return np.array(out)


def landmarks(item):
    if item["id"] in OVERRIDE:
        return OVERRIDE[item["id"]]
    a = np.asarray(Image.open(ROOT / item["layer"]["image"]).convert("RGBA"))[..., 3] > 60
    return landmarks_alpha(a)


def landmarks_alpha(a):
    """landmarks() on a boolean opaque mask (any image, e.g. a card cutout)."""
    h = a.shape[0]
    ys = np.where(a.any(1))[0]
    top, bot = ys[0], ys[-1]
    sp = spans(a)
    chest = np.median(sp[top + int(0.35 * (bot - top)): top + int(0.45 * (bot - top)) + 1])
    sh = next(y for y in range(top, bot) if sp[y] >= SHOULDER_FRAC * chest)
    hem = max(y for y in range(top, bot + 1) if sp[y] >= HEM_FRAC * chest)
    return sh / h, (hem + 1) / h


def body_px(item, H=1024):
    s, e = landmarks(item)
    return (e - s) * item["layer"]["height"] / 100 * H
