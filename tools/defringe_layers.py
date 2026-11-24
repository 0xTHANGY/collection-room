#!/usr/bin/env python3
"""P4 (2026-11-24): batch defringe of existing room layers (no painting re-segmentation):
smooth anti-aliased alpha pulled in ~0.5px (never grows), detached bits < 20% of the main part removed,
light wall-coloured rim pixels recoloured from the interior (dark painted outlines kept).
If the source painting can be located in git (tools/cutout_locate.py) its wall colour is used too."""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cutout_matte import defringe_rgba

def wall_of(iid, ref="HEAD"):
    try:
        import cutout_locate as LOC, io
        loc = LOC.locate(iid, ref)
        if not loc or loc[0] > 8:
            return None
        diff, zc, src, (x, y, w, h), zid, lc = loc
        p = np.array(Image.open(io.BytesIO(LOC.show(zc, src))).convert("RGB"))
        ring = p[max(0, y - 10):y + h + 10, max(0, x - 10):x + w + 10].reshape(-1, 3).astype(int)
        L = ring.mean(1); sat = ring.max(1) - ring.min(1)
        sel = ring[(sat < 25) & (L > 140)]
        return np.median(sel, 0) if len(sel) > 50 else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ids", nargs="+"); ap.add_argument("--write", action="store_true"); ap.add_argument("--preview")
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--base", help="read the original layer from this git ref (re-runs stay idempotent)")
    a = ap.parse_args()
    log = {}
    for iid in a.ids:
        p = ROOT / f"assets/zones/layers/{iid}.webp"
        if a.base:
            import subprocess, io
            rgba = np.array(Image.open(io.BytesIO(subprocess.run(["git", "-C", str(ROOT), "show", f"{a.base}:assets/zones/layers/{iid}.webp"], capture_output=True, check=True).stdout)).convert("RGBA"))
        else:
            rgba = np.array(Image.open(p).convert("RGBA"))
        wall = wall_of(iid, a.base or "HEAD") if iid.startswith("batch2") else None
        out, dropped = defringe_rgba(rgba, shrink=a.shrink, wall=wall)
        log[iid] = dict(old_px=int((rgba[..., 3] > 128).sum()), new_px=int((out[..., 3] > 128).sum()), dropped_px=dropped,
                        wall=None if wall is None else [int(v) for v in wall])
        if a.preview:
            Path(a.preview).mkdir(parents=True, exist_ok=True); Image.fromarray(out).save(f"{a.preview}/{iid}.png")
        if a.write:
            Image.fromarray(out).save(p.with_suffix(".webp.tmp"), "WEBP", quality=90, method=6); p.with_suffix(".webp.tmp").replace(p)
        print(iid, log[iid])
    if a.write:
        f = ROOT / "data/source/display/defringe-layers.json"
        prev = json.loads(f.read_text()) if f.exists() else {}
        prev.update(log); f.write_text(json.dumps(prev, indent=1) + "\n")

if __name__ == "__main__":
    main()
