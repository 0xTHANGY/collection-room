#!/usr/bin/env python3
"""Scan every catalog layer for a 1px full-width rod remnant and clear it,
keeping hook/handle pixels that cross the line."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parent / "collection-room-impl-report" / "fill-all" / "qa-fix" / "rodlines"


def detect_line(alpha: np.ndarray):
    h, w = alpha.shape
    op = (alpha >= 80).sum(1)
    hits = []
    for y in range(h):
        if op[y] < max(40, 0.75 * w):
            continue
        neigh = 0
        if y:
            neigh = max(neigh, int(op[y - 1]))
        if y + 1 < h:
            neigh = max(neigh, int(op[y + 1]))
        if op[y] >= 0.85 * w and op[y] >= neigh * 4:
            hits.append(y)
        elif op[y] >= 0.75 * w and op[y] >= neigh * 6:
            hits.append(y)
    return hits


def hook_columns(alpha: np.ndarray, y: int) -> set[int]:
    """Columns that are part of a hook or handle crossing the rod line.

    Require opaque pixels both above and below the line so a wider handle
    below the line does not preserve a horizontal rod stub beside it.
    Dilate 1px so antialiased hook edges stay.
    """
    h, w = alpha.shape
    cols: set[int] = set()
    for x in range(w):
        up = max(alpha[y - 2, x] if y >= 2 else 0,
                 alpha[y - 3, x] if y >= 3 else 0)
        dn = max(alpha[y + 2, x] if y + 2 < h else 0,
                 alpha[y + 3, x] if y + 3 < h else 0)
        if up >= 40 and dn >= 40:
            cols.add(x)
    extra: set[int] = set()
    for x in cols:
        for dx in (-1, 1):
            if 0 <= x + dx < w:
                extra.add(x + dx)
    return cols | extra


def clear_line(arr: np.ndarray, y: int) -> int:
    """Zero the rod line and a +/-2px halo; keep hook/handle columns."""
    a0 = arr[:, :, 3].copy()
    hooks = hook_columns(a0, y)
    h, w = a0.shape
    cleared = 0
    for row in range(max(0, y - 2), min(h, y + 3)):
        for x in range(w):
            if x in hooks:
                continue
            if arr[row, x, 3] >= 8:
                arr[row, x, 3] = 0
                cleared += 1
    return cleared


def main():
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    REPORT.mkdir(parents=True, exist_ok=True)
    seen, found, fixed = set(), [], []
    for item in catalog["items"]:
        rel = item["layer"]["image"]
        if rel in seen:
            continue
        seen.add(rel)
        path = ROOT / rel
        im = Image.open(path).convert("RGBA")
        arr = np.array(im)
        hits = detect_line(arr[:, :, 3])
        if not hits:
            continue
        found.append((item["id"], rel, hits, im.size))
        backup = REPORT / f"{item['id']}-original.webp"
        if not backup.exists():
            im.save(backup, lossless=True)
        cleared = sum(clear_line(arr, y) for y in hits)
        out = Image.fromarray(arr)
        # Keep webp lossless-ish
        out.save(path, lossless=True)
        # residual check
        leftover = detect_line(arr[:, :, 3])
        fixed.append((item["id"], rel, hits, cleared, leftover))
        print(f"fixed {item['id']} {rel} rows={hits} cleared={cleared} leftover={leftover}")
    (REPORT / "scan.json").write_text(json.dumps({
        "layers_scanned": len(seen),
        "hits": [{"id": i, "path": p, "rows": h, "size": s} for i, p, h, s in found],
        "fixed": [{"id": i, "path": p, "rows": h, "cleared": c, "leftover": L}
                  for i, p, h, c, L in fixed],
    }, indent=2) + "\n")
    if any(L for *_, L in fixed):
        raise SystemExit("rod line still detected after the fix")
    print(f"scanned {len(seen)} layers, found {len(found)}, fixed {len(fixed)}")


if __name__ == "__main__":
    main()
