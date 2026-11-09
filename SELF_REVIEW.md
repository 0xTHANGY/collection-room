# Self review — Collection Room v1

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
| cr-006 | Navy sweatshirt | 49.7 / 31 / 14.5 / 33 |
| cr-007 | Checkered tote | 65.2 / 64 / 13.7 / 27 |
| cr-008 | Pink cardigan | 62.8 / 32 / 11.3 / 32 |
| cr-009 | Blue blouse | 73 / 32 / 11.1 / 33 |
| cr-010 | Cream dress | 82.3 / 34 / 10.8 / 44 |

## Validation

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
