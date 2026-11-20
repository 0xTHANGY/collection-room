#!/usr/bin/env python3
"""v3 review fixes: enlarge hotspots for charms/straps; refresh build-v2-summary.json with real counts."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data/catalog.json"
# left, top, right, bottom in % of the zone image (include hanging charm / side strap / tassels)
HOTSPOTS = {
    "today-455": (15.6, 7.0, 31.4, 41.0),   # totes-a: blue plush charm
    "today-462": (70.8, 47.0, 86.4, 79.2),  # totes-a: red bow charm + straps
    "today-422": (54.3, 8.4, 71.9, 42.6),   # bags-j: rope side strap + tassels
}

def main():
    c = json.loads(CAT.read_text())
    by = {i["id"]: i for i in c["items"]}
    for iid, (l, t, r, b) in HOTSPOTS.items():
        by[iid]["hotspot"] = dict(left=l, top=t, width=r - l, height=b - t)
    CAT.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
    sp = ROOT / "data/build-v2-summary.json"
    s = json.loads(sp.read_text())
    s["categories"] = len(c["categories"])
    s["zones"] = len(c["zones"])
    s["items"] = len(c["items"])
    s["zones_with_layout"] = sum(1 for z in c["zones"] if z["has_layout"])
    s["items_with_hotspot"] = sum(1 for i in c["items"] if i.get("hotspot"))
    s["list_only_items"] = s["items"] - s["items_with_hotspot"]
    s["fallback_ids"] = sorted(i["id"] for i in c["items"] if (i.get("layer") or {}).get("fallback"))
    s["zones_lacking_layout"] = [z["id"] for z in c["zones"] if not z["has_layout"]]
    files = [p for p in (ROOT / "assets").rglob("*") if p.is_file()]
    s["assets_total_bytes"] = sum(p.stat().st_size for p in files)
    s["card_images_bytes"] = sum(p.stat().st_size for p in (ROOT / "assets/items/v2").glob("*") if p.is_file())
    s["asset_files_total"] = len(files)
    for k in ("asset_bytes_written", "asset_files_written"):
        s.pop(k, None)
    sp.write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in s.items() if k not in ("fallback_ids", "first_load_files", "overlap_pairs")}, len(s["fallback_ids"]))
if __name__ == "__main__":
    main()
