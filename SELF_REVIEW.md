# Self review — Collection Room v1.1

Built a local-only static HTML/CSS/JS collection room, with the supplied room illustration, ten original item photographs, a toggle to reveal hotspots, and an accessible expandable collection index. One JSON file (`data/items.json`) generates all hotspots, list entries, and detail content. Names and categories come directly from the manifest; every story is exactly `故事待补充`.

## Hotspot mapping

Inspected the actual 1280 × 720 room image. Values below are left/top/width/height percentages; the authoritative values live in JSON.

| Item | Visible object | Rectangle (%) |
|---|---|---|
| cr-001 | Leftmost cream cap | 42.2 / 6 / 7.8 / 16 |
| cr-002 | Black cap | 50.6 / 5.5 / 8.3 / 17 |
| cr-003 | Cream and green cap | 59.8 / 6 / 8.1 / 17 |
| cr-004 | Blue cap | 68.9 / 6.2 / 8.2 / 17 |
| cr-005 | Navy shorts in crate | 49.4 / 65.5 / 12 / 16.5 |
| cr-006 | Navy sweatshirt | 49.7 / 31 / 12.5 / 33 |
| cr-007 | Checkered tote | 65.2 / 64 / 13.7 / 27 |
| cr-008 | Pink cardigan | 62.8 / 32 / 9.6 / 32 |
| cr-009 | Blue blouse | 73 / 32 / 8.7 / 31.4 |
| cr-010 | Cream dress | 82.3 / 34 / 10.8 / 44 |

## Original v1 validation (historical)

- Served successfully through Python's local static server.
- Headless Chrome checked all 10 hotspots and all 10 index buttons, exact names/categories/story text, successful photos, Enter/Space opening, Escape closing and focus return, missing-photo fallback, and no JavaScript errors.
- A 375px mobile viewport had no horizontal overflow. Inspected desktop and mobile screenshots.
- `node --check app.js` passed.
- Pillow saved all 11 images into fresh PNGs and reopened each: `getexif()` empty, no EXIF entry in image info. Original inputs were never changed.
- Native modal dialog provides focus containment. CSS includes reduced-motion support; buttons provide hover and keyboard focus feedback. Room and detail photographs include loading/error states.

## Limits / unimplemented

- Optional day/night toggle skipped.
- Hotspots are approximate rectangles; the item index provides larger mobile targets.
- JavaScript and an HTTP server are required for the JSON-driven interface. No deployment or remote Git was created.
- The environment mounts `site/.git` read-only, so normal `git init` failed. Local history instead lives at `/workspace/collection-room/site-history.git`, with this directory as its work tree. Use `git --git-dir=/workspace/collection-room/site-history.git --work-tree=/workspace/collection-room/site log` to inspect the commit.

## v1.1 changes and verification

- Item photos were resized from imported originals to 800×800 WebP (EXIF stripped); room has 640w/800w/1280w WebP with responsive `srcset`/`sizes` (intrinsic 1280×720). Originals under `imported/` and `incoming/` were not modified. Site no longer ships the large PNGs.
- Room has aspect-ratio 1280/720; the block image uses width 100% and height auto. The absolute hotspot layer covers that image box without object-fit letterboxing.
- Muted text is `#5f6156` against paper `#f5f1e9`: WCAG sRGB relative-luminance contrast **5.5966:1**, exceeding 4.5:1.
- All 45 unordered hotspot pairs were checked programmatically with AABB intersection: **zero overlaps**. Garment gaps are 0.6 percentage points horizontally; blouse/tote gap is 0.6 vertically. JSON table above reflects current rectangles.
- Each entry is independently validated and caught/logged by ID on failure. Hotspot coordinates must be finite numbers within 0–100, dimensions positive; explicit numbers 001–010 drive list labels.
- Reveal text follows aria-pressed; collection summary text follows native details open state. English names/categories and static English text have lang="en"; page remains zh-CN.
- CSP meta is `default-src 'self'`. Static audit confirms local scripts/styles/images/JSON, no inline scripts or HTML style attributes. Hotspot positions use individual element.style properties. No unsafe-inline exception is needed. Browser CSP/network verification remains with the coordinator.
- Node syntax and DOM-stub behavior checks passed, including invalid paths/coordinates, skipping a bad entry while rendering subsequent entries, toggle labels, language tags, and explicit item numbering. Asset existence and unchanged names/categories/stories checked against original tracked JSON.
- Review items 3 and 10 intentionally skipped. No server lifecycle actions, deployment, push, or remote changes performed.

## v1.2 room swap

- Room assets: `assets/room/style-v2-03.webp` (1280×720), `-800.webp` (800×450), `-640.webp` (640×360), regenerated from `incoming/style-v2-03-pinkchair.png` without modifying the original. Old `style-v2-01` site WebPs removed.
- Hotspots: phase-correlation shift on hats/wardrobe crops was (0,0); per-item crops on the new image still cover each object. Coordinates unchanged; AABB overlap check remains zero across all pairs.
- Favicon: hand-written `favicon.svg` plus matching `favicon.ico` (Pillow), both same-origin under CSP `default-src 'self'`.

## v1.3 layered motion

- Empty room `assets/room/style-v2-05.webp` (1280×720, 800, 640) generated from `incoming/style-v2-05-empty-pinkchair.png`. Furnished `style-v2-03` site WebPs removed. Incoming originals were not modified.
- Ten transparent WebP layers in `assets/layers/`, cut with `tools/cutout.py` (difference mask, morphological clean-up, seeded watershed, 1.6px feather). Hats include the peg under the cap; hanging clothes include their hanger.
- Composite of layers over the empty room vs the furnished original: global MAE 3.77, PSNR 28.67. Much of the leftover is illustration variance outside the items (empty-vs-full already differs on the left wall/plant). Item-union MAE 3.62. Adjacent hanging garments share painted overlaps in the source illustration, so a neighbour sleeve sliver can remain on cr-008 and cr-010; cr-006 keeps a small pink sliver. Shorts/tote are clean aside from a faint contact shadow or tiny speck.
- Motion: hang items use a requestAnimationFrame spring (stiffness 8, damping 1.1, max ±10deg, skewX ≤2deg) driven by pointer passes on the room, with `pointer-events: none` on sprites. Rest items use CSS `translateY(-4px) scale(1.03)` (300ms) and press `scale(0.98)` (100ms). Cards fade in and move up 8px. `prefers-reduced-motion: reduce` disables all of this.
- CSP remains `default-src 'self'`. Hotspots, keyboard, and ARIA are unchanged.

## v1.4 FLIP cards and silhouette glow

- Opening a card plays a 650ms FLIP (`cubic-bezier(0.22, 1, 0.36, 1)`) from the sprite's on-screen rect, or from the list row when opened from the index. Only `transform` and `opacity` are animated. Close reverses to the same origin, then the dialog closes. Escape is intercepted via `cancel` (`preventDefault`) so the reverse can run. Rapid clicks and double-closes are ignored while a flip is in progress; a close requested mid-open is queued.
- Native `<dialog>` still uses `showModal()` for focus trap; focus returns to the opener on `close`. Reduced motion replaces the FLIP with a 650ms fade.
- Hover and `:focus-visible` add a warm-white / light-pink `filter: drop-shadow` on a `layer-wrap` around each sprite (~200ms fade). Transforms for swing and lift stay on the inner image so they can run with the glow. The old hotspot box fill/outline is removed; the reveal-button plus marks remain. Reduced motion keeps a static glow (no filter transition).
- Glow approach: CSS `filter: drop-shadow` on `.layer-wrap` (not pre-generated outlines). rAF timing during hover + swing (86 steady frames): mean 16.67ms, p95 16.7ms, max 16.8ms, 0 frames over 20ms. No jank, so cutout glow layers were not needed.

## V2.0

The existing hub illustration, ten original hotspots and original `data/items.json` remain intact. Category navigation is generated from the final documents and authoritative zone membership. The page shares the existing spring, glow, FLIP dialog, reduced-motion handling and focus return across hub and zone views. Cards expose `data-item-id`, display source names/eligible brands, and always use `故事待补充`.

The builder writes pixel-only WebP zone sizes, alpha card sizes and painted layers, plus the catalog and machine-readable build summary. It never modifies the read-only originals. It runs GrabCut in a padded bbox crop and inpaints the valid mask union. Overlapping source bboxes, implausible masks, or an inpaint interior radius above 24 pixels trigger conservative whole-zone original-painting fallback; fallback layers remain hidden until hover/focus and receive one quarter of the normal swing impulse. This is a deliberate quality limitation: source paintings alone do not contain the occluded garment edges or wall texture, so inpainting is an approximation and masks still require visual review. The catalog records the actual fallback IDs, rather than implying all layers are cleanly separated.

Missing layout zones retain the painting and complete item lists. The verifier explicitly reports list-only checks separately from painted hotspot checks, rather than claiming unavailable coordinates were tested. Responsive/lazy first-load bytes are measured in Chromium at 1440×900 and recorded with the exact file list. Browser verification and final coverage counts are recorded in `screenshots/v2.0-verify.json`; build coverage, fallback IDs and byte totals are in `data/build-v2-summary.json`.

Final v2 build coverage: all 43 layouts arrived during implementation; all 384 items now have painted hotspots, and no zones lack layout. All 43 generated overlap counts are zero. The final image decode/metadata audit checked 1,324 WebPs with zero failures (no EXIF/XMP/ICC).

**Known quality gap:** all 384 current items trigger the conservative fallback gate (large inpaint holes and/or overlapping bboxes). Their zones therefore retain the original painting, and extracted layers appear only on hover/focus with reduced motion. No visibly broken clean plates are presented. The clean-plate code path exists for suitable masks, but this source set does not currently pass its quality gate. Mask extraction can retain rail, handle or neighbour fragments; future refinement should replace or review masks and background reconstruction before relaxing the gate. Automated tests establish correct item identity, accessibility and geometry, not artistic segmentation quality. The exact fallback IDs are in `data/build-v2-summary.json`.

Generated v2 assets: 53,126,612 bytes across 1,324 files. Whole assets directory: 53,676,030 bytes. The first-load measurement and browser results are recorded in the final JSON reports; the verifier refreshes first-load bytes after any frontend changes. Screenshot coverage includes hats A, jackets A and jackets B, plus the hub and an open card. No git add, commit, publication or original-source mutation was performed.

Final Playwright result: PASS. Checked 384/384 painted hotspots and all ten hub hotspots; zero list-only items, rendered overlaps, console errors, page errors, failed requests or HTTP errors. Lazy initial card/zone-layer checks, normal-motion FLIP, reduced-motion fades, history/deep links, keyboard activation/Escape, focus return and 375px horizontal-overflow checks passed. Measured hub first load at 1440×900: 869,901 bytes. All requested desktop screenshots are saved, including jackets B as the third layout zone.
