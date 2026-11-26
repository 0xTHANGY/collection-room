#!/usr/bin/env python3
"""Assert every requested room cover was rebuilt: mtime is new, hash differs from
the 88d42ef blob, and the file matches a fresh composite of the current catalog."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from rebuild_covers import composite  # noqa: E402
from repack_three_rails import save_webp  # noqa: E402

BASELINE = "88d42ef"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def git_sha(rel: str) -> str:
    data = subprocess.check_output(["git", "show", f"{BASELINE}:{rel}"], cwd=ROOT)
    return sha(data)


def main(zids):
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    zones = {z["id"]: z for z in catalog["zones"]}
    # baseline commit time (unix) — cover mtimes must be newer
    ctime = int(subprocess.check_output(
        ["git", "log", "-1", "--format=%ct", BASELINE], cwd=ROOT).decode())
    errors = []
    print(f"{'zone':16} {'mtime_new':8} {'hash_changed':12} {'matches_room':12}  sha256")
    for zid in zids:
        z = zones[zid]
        rel = z["cover"]["src"]
        path = ROOT / rel
        now = path.stat().st_mtime
        mtime_new = now > ctime
        new_hash = file_sha(path)
        try:
            old_hash = git_sha(rel)
            hash_changed = new_hash != old_hash
        except subprocess.CalledProcessError:
            old_hash = None
            hash_changed = True
        # Rebuild a cover in memory and compare pixels (webp bytes may differ)
        image, _ = composite(catalog, zid)
        cw, ch = z["cover"]["width"], z["cover"]["height"]
        rebuilt = image.resize((cw, ch), Image.Resampling.LANCZOS)
        on_disk = Image.open(path).convert("RGB")
        a, b = np.asarray(rebuilt.convert("RGB"), dtype=int), np.asarray(on_disk, dtype=int)
        if a.shape != b.shape:
            matches = False
            mad = None
        else:
            mad = float(np.mean(np.abs(a - b)))
            matches = mad <= 3.4  # same tolerance band as the QA updated covers
        print(f"{zid:16} {str(mtime_new):8} {str(hash_changed):12} {str(matches)+(' mad=%.2f'%mad if mad is not None else ''):12}  {new_hash[:12]}")
        if not mtime_new:
            errors.append(f"{zid}: cover mtime {now} is not newer than {BASELINE} ({ctime})")
        if not hash_changed:
            errors.append(f"{zid}: cover hash still matches {BASELINE}")
        if not matches:
            errors.append(f"{zid}: cover does not match a fresh room composite (mad={mad})")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"\n{len(zids)} covers: mtime new, hash changed vs {BASELINE}, match current room")


if __name__ == "__main__":
    zids = sys.argv[1:] or [
        "shoes-a","shoes-b","shoes-c","shoes-d","shoes-e",
        "totes-a","totes-b","totes-c",
        "accessories-a","accessories-b","accessories-c",
        "hats-a","hats-b","hats-c","hats-d","hats-e","hats-f","hats-g",
    ]
    main(zids)
