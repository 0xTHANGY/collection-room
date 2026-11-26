#!/usr/bin/env python3
"""Put furniture-band shoes back on their 7ceb60f shelves/stools (restore left;
tops were already correct). Then run the support check."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_shoe_supports import check  # noqa: E402

# 7ceb60f left values for every shoe whose top sits in the furniture band.
# tops, widths and heights are left unchanged.
RESTORE_LEFT = {
    "batch2-055": 26.24777450042517,   # shoes-a left 2-tier shelf
    "shoes-n023": 34.17364413721379,   # shoes-b left 2-tier shelf
    "shoes-n017": 64.63378215158752,   # shoes-b right low stool
    "shoes-n024": 68.63606770833334,   # shoes-c left 2-tier shelf (small drift)
    "shoes-n034": 87.5331512014315,    # shoes-c right low stool
    "shoes-n041": 48.71265156060935,   # shoes-d left 2-tier shelf
    "shoes-n040": 75.33196223498234,   # shoes-d right low stool
    "shoes-n056": 46.09375,            # shoes-e left 2-tier shelf (was on the post)
    "shoes-n053": 84.95034761615788,   # shoes-e right low stool
}


def apply(catalog):
    items = {it["id"]: it for it in catalog["items"]}
    changed = []
    for iid, left in RESTORE_LEFT.items():
        it = items[iid]
        old = it["layer"]["left"]
        for box in (it["layer"], it["hotspot"], it["layer"].get("rect")):
            if isinstance(box, dict) and "left" in box:
                box["left"] = left
        changed.append((iid, it["zone"], old, left))
    return changed


def main():
    path = ROOT / "data/catalog.json"
    catalog = json.loads(path.read_text())
    before_err, before_details, _ = check(catalog)
    print("BEFORE:")
    print("\n".join(d for d in before_details if d.startswith("FAIL") or any(i in d for i in RESTORE_LEFT)))
    print(f"failures before: {len(before_err)}")
    catalog = json.loads(path.read_text())
    changed = apply(catalog)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print("\nRestored left:")
    for iid, zone, old, new in changed:
        print(f"  {zone}/{iid}: {old:.4f} -> {new:.4f}")
    after = json.loads(path.read_text())
    after_err, after_details, _ = check(after)
    print("\nAFTER:")
    print("\n".join(d for d in after_details if d.startswith("FAIL") or any(i in d for i in RESTORE_LEFT)))
    if after_err:
        print(f"\n{len(after_err)} failure(s) remain:")
        print("\n".join(after_err))
        sys.exit(1)
    print(f"\nall {sum(1 for it in after['items'] if it['category']=='shoes')} shoes pass the support check")


if __name__ == "__main__":
    main()
