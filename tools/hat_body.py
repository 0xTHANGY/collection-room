"""Hat body box for size normalisation: excludes cords, chin straps, braids/tassels (narrow rows) and
pompoms / top nubs (manual overrides), matching the H convention used for real heights."""
import numpy as np
from PIL import Image
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# fractions of layer-image height kept as hat body (pompom on top / straps below), checked by eye
OVERRIDE = {"batch2-137": (0.30, 1.0), "batch2-063": (0.23, 1.0)}
# earflap hats: the flaps belong to H (narrow, so the width rule would drop them); braids/top nub do not
FIXED = {"batch2-135": (0.02, 0.54), "batch2-139": (0.09, 0.80)}
_c = {}

def body_frac(item, min_w=0.28):
    """(x0,y0,x1,y1) fractions of the layer image covering the hat body."""
    p = item["layer"]["image"]
    if p in _c:
        return _c[p]
    with Image.open(ROOT / p) as im:
        a = np.asarray(im.convert("RGBA").getchannel("A")) > 24
    h, w = a.shape
    rows = a.sum(1)
    if item["id"] in FIXED:
        fy0, fy1 = FIXED[item["id"]]
        ys, ye = int(fy0 * h), int(np.ceil(fy1 * h)) - 1
        colsum = a[ys:ye + 1].sum(0)
        cols = np.where(colsum >= 3)[0]
        _c[p] = (cols.min() / w, ys / h, (cols.max() + 1) / w, (ye + 1) / h)
        return _c[p]
    y0o, y1o = OVERRIDE.get(item["id"], (0.0, 1.0))
    rows[: int(y0o * h)] = 0
    rows[int(np.ceil(y1o * h)):] = 0
    good = rows >= min_w * rows.max()
    # largest contiguous block of wide rows, allowing 3-px gaps
    best, cur, start, gap = (0, 0, 0), 0, None, 0
    for y in range(h):
        if good[y]:
            if start is None:
                start = y
            gap = 0
            cur = y
        elif start is not None:
            gap += 1
            if gap > 3:
                if cur - start > best[2] - best[1]:
                    best = (0, start, cur)
                start, gap = None, 0
    if start is not None and cur - start > best[2] - best[1]:
        best = (0, start, cur)
    _, ys, ye = best
    cols = np.where(a[ys:ye + 1].any(0))[0]
    # width: columns that are body in at least 3 rows of the block
    colsum = a[ys:ye + 1].sum(0)
    cols = np.where(colsum >= 3)[0]
    _c[p] = (cols.min() / w, ys / h, (cols.max() + 1) / w, (ye + 1) / h)
    return _c[p]

def body_px(item, W=1536, H=1024):
    L = item["layer"]
    x0, y0, x1, y1 = body_frac(item)
    return (x1 - x0) * L["width"] / 100 * W, (y1 - y0) * L["height"] / 100 * H
