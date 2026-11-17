# Collection Room

Current version: V2.4.0 (the footer version is stamped at build time from `data/source/site.json`).

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

The generator requires Pillow, numpy and OpenCV. The verifier and the generator's first-load measurement additionally use Python Playwright with `/usr/bin/google-chrome --no-sandbox`. Keep the existing preview running at `http://127.0.0.1:8765/`; the scripts do not start or stop it. Build output and diagnostics are in `data/catalog.json` and `data/build-v2-summary.json`. Verification writes `screenshots/v2.1-verify.json` and desktop screenshots. First-load bytes count successful unique page requests at 1440×900, including responsive hub images and any cover images requested by Chromium's lazy-load threshold, excluding HTTP headers and browser cache compression.

To add or update content, the coordinator drops final source paintings/cutouts and category `docs/<category>-items.json` into the read-only `../final/` source tree, updates `manifest.csv` / `zone-finals.csv`, and updates `data/source/zones.json` (authoritative membership). Add a new category's Chinese/English label to `LABELS` in the generator. Add or replace `data/source/layout/<category>-<letter>.json` with the zone's `image_size` and item `bbox` coordinates. Optional item `motion: "hang"` or `"lift"` overrides the category/vertical-position rule. Then rerun the generator and verifier. The builder never writes to `final/`, `incoming/` or `imported/`.

Missing layouts are reported and remain browsable through zone lists. V2.1 uses per-item gates and fitted-wall reconstruction for hats, bags and accessories; see the V2.1 notes below. Other backgrounds retain original-painting fallback. Hotspots are inset and split at neighbour midlines to eliminate overlap. Review images remain essential: automated masks cannot prove semantic boundaries or artistic quality.

Only source `name`, `type`, and eligible `brand` are displayed. Uncertain/unknown brands display Unknown; every story is `故事待补充`. Source `visual`/`notes` never enter the catalog. Card images are loaded only when opening the dialog, with 400/600 responsive sources; zones and their layers are created only on navigation. Generated image metadata is stripped by constructing fresh pixel-only images before WebP encoding.

## V2.1

Card variants are metadata-free 600/400 WebP q80. The three documented low-resolution dresses are deterministically upscaled; other cards are never upscaled. `data/source/cutout-fixes.json` records deterministic site-only pink silhouette and edge-despill overrides; source originals remain untouched. Hub display names override catalog names. Uncertain brands display Unknown.

`tools/v21_images.py` fits a robust quadratic neutral-wall model outside item bboxes for hats, bags and accessories. It excludes horizontal rails, clips neighbour cores, selects a central component, and gates each item independently by coverage, mask collision and unsupported boundary/floor regions. Passing masks get a reconstructed plate and visible rest layers. Failed masks retain original plate pixels and hover-only fallback. Other categories retain conservative fallback. This is an approximation, not a claim of recovered hidden detail. Gate reasons and labelled checkerboard reviews are under `../work/v2.1/review/`; `gate-table.md` covers all 43 zones. QA dispositions are in `data/source/qa-v21.json`.

### V2.1 part B

`data/source/overrides_v21.py` and `cutout-fixes.json` are reproducible site-only pixel recipes, applied by `tools/build_v2.py`; part A overrides remain. Dresses 039/102/234 use separate Lanczos RGB/alpha scaling to a 1200px working image and mild unsharp masking, then the normal 600/400 variants. They look smoother and display at the same card size as neighbours, but remain visibly softer: original 124–186px widths cannot supply new weave, lace or edge detail.

Shirts A uses a smooth background colour mask protected by garment extraction masks. Wall/floor warmth is neutralised consistently in the base, all zone layers and cover; the sampled upper wall is RGB 205/205/205. Wood, plants and garment colours are retained; very narrow warm contact edges can remain. The cr-009 card cuff lettering is replaced by nearby striped cuff pixels with a feathered clone. Its zone painting/layer has no corresponding lettering and receives only the room background correction.

Accessory card hooks 116/119/278/361 are desaturated only in measured above-item hook regions. Cards 123/117/128/133/240/242 receive magenta despill limited to semi-transparent pixels and a narrow alpha edge band. Pink garment interiors are protected. Zone paintings/layers were visually checked and do not show the corresponding keying casts, so receive no accessory colour overrides. No AI generation or new text; no source/archive/git writes. No generation is required for these repairs (0 images); truly recovered high-resolution detail for the three dresses would require three replacement images or better originals.

Recreate the six matching after images with `python3 tools/capture_v21_part_b.py` while the preview serves port 8765. Browser evidence is 1440×900. Byte weights and complete verification results are in `data/build-v2-summary.json` and `screenshots/v2.1-verify.json`.

Part B final build: card images 24,237,402 bytes; assets total 38,782,508 bytes; hub first load 869,801 bytes at 1440×900. Verifier PASS: 384/384 zone items, 10/10 hub items, 142 visible passing rest layers; zero overlaps, console/page errors, failed requests or HTTP errors/404. Lazy-load, FLIP, history, keyboard/focus and mobile checks passed. Full results: screenshots/v2.1-verify.json.

## V2.3.2

Collection cards close immediately on Esc or ✕, including while a card is still opening: the dialog is closed synchronously and a non-interactive top-layer copy plays the same 650 ms FLIP (or the reduced-motion fade), so the real dialog and its ✕ never move. The open card keeps the compositor-layer state the animated dialog had in V2.3.1, so its pixels match V2.3.1. Footer version V2.3.2.

## V2.4.0 — new shoes, pilot of zone-01

Pilot of the "new shoes" delivery (Bruce, 2026-11-16, 59 pairs). Only zone-01 is integrated: the 鞋履 / Shoes room gets zone **C** (`#/shoes/c`) with six pairs, `shoes-n007`…`shoes-n012` (display numbers 394–399). The remaining nine zones (test image = B, zone-02…09 = D…K) are not in this build; ids `shoes-n001…n059` and numbers 388–446 are reserved in delivery order so the later rollout stays consistent. Total: 390 zone items + 10 hub items = 400; the existing 394 are unchanged.

- Names are the catalogue-label text of the original photos (Chinese parentheticals of the delivery list dropped); every story is `故事待补充`; no brand or story is invented.
- Second source directory `data/source/newshoes/`: `items.json`, the zone painting (`zones/`), card cutouts (`cutouts/`, lossless WebP) and prepared silhouettes (`masks/`). `tools/build_v2.py` reads it through three small changes (extra item lookup, cutouts/zones already listed in `generated-v22.json`, optional `number` field); `tools/v23_surfaces.py` accepts `external_masks` for zones that list them, shoes-a is untouched.
- Card cutouts (`tools/newshoes_prep.py cutouts`) are deterministic and local: the original photo pixels are kept; the background is the region reachable from the photo border without crossing an edge, soft shadows are peeled with a narrow GrabCut band, the catalogue label (vertical rule + text, lower right) is excluded before anything else. No AI redrawing. Light grey soft-shadow remnants can remain between/under a pair.
- Layered motion: the empty-rug plate `data/source/generated/shoes-zone-c-rug-v240.webp` was generated once with Codex (ChatGPT plan) from the zone-01 painting and rug crops, metadata stripped, used only in a thin collar around the pairs and their shadows. Pair silhouettes come from painting-vs-plate differencing (`tools/newshoes_prep.py zone`). The per-item gate of `v23_surfaces.py` ran unchanged; with external masks the removal check is judged on the mask core (margin removed).
- Footer and `data/source/site.json` read V2.4.0.

