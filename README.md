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

## V2.0 — category rooms

The original ten-object hub still uses `data/items.json` and its original room/layers. V2 adds 11 category rooms, 43 zone paintings and 384 source items. Open `#/`, `#/hats`, `#/hats/a`, or `#/item/batch2-006`. Browser Back/Forward follows room navigation. The site remains buildless with `default-src 'self'` and no page dependencies.

Generate the static catalog and metadata-free WebP assets from this directory:

```sh
python3 tools/build_v2.py
python3 tools/verify_v2.py
```

The generator requires Pillow, numpy and OpenCV. The verifier and the generator's first-load measurement additionally use Python Playwright with `/usr/bin/google-chrome --no-sandbox`. Keep the existing preview running at `http://127.0.0.1:8765/`; the scripts do not start or stop it. Build output and diagnostics are in `data/catalog.json` and `data/build-v2-summary.json`. Verification writes `screenshots/v2.0-verify.json` and desktop screenshots. First-load bytes count successful unique page requests at 1440×900, including responsive hub images and any cover images requested by Chromium's lazy-load threshold, excluding HTTP headers and browser cache compression.

To add or update content, the coordinator drops final source paintings/cutouts and category `docs/<category>-items.json` into the read-only `../final/` source tree, updates `manifest.csv` / `zone-finals.csv`, and updates `data/source/zones.json` (authoritative membership). Add a new category's Chinese/English label to `LABELS` in the generator. Add or replace `data/source/layout/<category>-<letter>.json` with the zone's `image_size` and item `bbox` coordinates. Optional item `motion: "hang"` or `"lift"` overrides the category/vertical-position rule. Then rerun the generator and verifier. The builder never writes to `final/`, `incoming/` or `imported/`.

Missing layouts are reported and remain fully browsable through the zone item list; no fictitious painting coordinates are generated. With a layout, GrabCut extracts the largest connected painted component; a feathered source layer covers the inpainted clean plate. Sparse/solid extractions, overlapping source bboxes, or an inpaint hole with an interior radius above 24 pixels at the generated zone scale conservatively trigger original-painting fallback for that zone's layers. The radius gate follows visual review of Telea wedge artifacts in large filled regions. Those layers appear on hover/focus with a smaller swing, and all fallback IDs appear in the build summary. Hotspots are inset and split at neighbour midlines to eliminate overlap. Review screenshots after adding layouts: automated masks cannot establish semantic item boundaries or judge inpaint quality.

Only source `name`, `type`, and eligible `brand` are displayed. Uncertain/unknown brands are suppressed; every story is `故事待补充`. Source `visual`/`notes` never enter the catalog. Card images are loaded only when opening the dialog, with 400/800 responsive sources; zones and their layers are created only on navigation. Generated image metadata is stripped by constructing fresh pixel-only images before WebP encoding.
