# Collection Room

A buildless static HTML/CSS/JavaScript website. No dependencies, API keys, or build step.

From this directory, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/ (opening index.html directly cannot fetch JSON).

`data/items.json` is the single source for item names, categories, stories, images, zones, and hotspot rectangles. Add a sanitized image and JSON entry to add an item. Hotspot percentages are relative to the full 1280 × 720 room image. Use the supplied WebP assets without regenerating or recompressing them. Each item has an explicit three-digit `number`; image paths must be same-site relative paths.

No remote, hosting, analytics, or external assets are used.

Version 1.1 adds responsive room WebP sources, image-aligned non-overlapping hotspots, stronger muted text contrast, per-item validation, state-aware controls, English language tags, and a self-only Content Security Policy. Review items 3 and 10 are intentionally excluded.

Version 1.2 swaps in the pink-chair room illustration (`style-v2-03` WebP srcset), keeps hotspot rectangles after visual re-check (no coordinate shift), and adds a same-origin SVG/ICO favicon.

Version 1.3 uses an empty-room base (`style-v2-05` WebP srcset) plus per-item transparent layers. Hats and hanging clothes swing from their hang point on pointer pass; shorts and the tote lift slightly on hover. `prefers-reduced-motion: reduce` disables the motion and keeps the static highlight. Layers are built with `tools/cutout.py` from the furnished/empty room pair (originals are never modified).

## Cutout tool

From the collection-room directory:

```sh
python3 site/tools/cutout.py \
  --full incoming/style-v2-03-pinkchair.png \
  --empty incoming/style-v2-05-empty-pinkchair.png \
  --items site/data/items.json \
  --out site/assets/layers \
  --update-json
```

Requires Pillow, numpy, and OpenCV. The script only reads the incoming originals and writes layer WebPs plus optional JSON metadata (`layer` fields). See the module docstring for padding, threshold, and feather defaults.

Version 1.4 opens collection cards with a FLIP scale/translate from the item (or list row) and adds a drop-shadow silhouette glow on hover and keyboard focus. Reduced motion uses a plain fade and a static glow.
