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
