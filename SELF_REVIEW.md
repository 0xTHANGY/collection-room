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
