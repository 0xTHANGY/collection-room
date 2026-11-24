#!/usr/bin/env python3
"""Rebuild <zone>-cover-640.webp = zone 1600 plate + item layers (catalog rects, zIndex order).
Usage: rebuild_covers.py zone [zone ...] | --all"""
import json, sys
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from repack_three_rails import save_webp

def composite(catalog, zid):
    z = next(z for z in catalog["zones"] if z["id"] == zid)
    src = max(z["images"], key=lambda i: i["width"])["src"]
    base = Image.open(ROOT / src).convert("RGBA")
    W, H = base.size
    its = sorted([i for i in catalog["items"] if i["zone"] == zid], key=lambda i: i["layer"].get("zIndex", 0))
    for it in its:
        L = it["layer"]
        im = Image.open(ROOT / L["image"]).convert("RGBA")
        w, h = max(1, round(L["width"] / 100 * W)), max(1, round(L["height"] / 100 * H))
        base.alpha_composite(im.resize((w, h), Image.Resampling.LANCZOS), (round(L["left"] / 100 * W), round(L["top"] / 100 * H)))
    return base.convert("RGB"), z

def main():
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    zids = [z["id"] for z in catalog["zones"]] if sys.argv[1:] == ["--all"] else sys.argv[1:]
    for zid in zids:
        comp, z = composite(catalog, zid)
        cw, ch = z["cover"]["width"], z["cover"]["height"]
        save_webp(comp.resize((cw, ch), Image.Resampling.LANCZOS), ROOT / z["cover"]["src"])
        print("cover", zid, cw, ch)

if __name__ == "__main__":
    main()
