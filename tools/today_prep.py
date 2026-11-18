#!/usr/bin/env python3
"""Prepare original-photo cards and register the 2026-11-18 batch locally."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from PIL import Image, ImageOps
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/source/today-2026-11-18'
INCOMING = Path('/workspace/collection-room/incoming/new-items')
PENDING = ROOT.parent / 'pending-doubao'
CARDS = Path('assets/items/v2')
NO_PHOTO = {'today-109', 'today-111', 'today-117'}
SIZES = (600, 400)


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            temp.write(payload)
            temp.close()
            temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, data, indent=None, newline=True):
    options = {'indent': indent} if indent else {'separators': (',', ':')}
    payload = (json.dumps(data, ensure_ascii=False, **options) + ('\n' if newline else '')).encode()
    if not path.exists() or path.read_bytes() != payload:
        atomic_write(path, payload)


def original(item):
    filename = item['source_file']
    if Path(filename).name != filename:
        raise ValueError(f'Expected a filename for {item["id"]}: {filename}')
    return INCOMING / filename


def dimensions(size, longest):
    scale = longest / max(size)
    return tuple(max(1, round(edge * scale)) for edge in size)


def card_specs(item, size):
    return [
        dict(src=str(CARDS / f'{item["id"]}-{longest}.webp'), width=width, height=height)
        for longest in SIZES
        for width, height in [dimensions(size, longest)]
    ]


def cutout(image):
    import cv2
    import numpy as np
    from newshoes_prep import find_label, lab_of

    rgb = np.array(image.convert('RGB'))
    height, width = rgb.shape[:2]
    rule = find_label(rgb)
    if rule:
        x, y, _, _ = map(int, rule)
        x, y = max(1, x - 8), max(1, y - round(height * .025))
    else:
        # The label occupies the lower-right margin on these catalogue photos.
        x, y = round(width * .74), round(height * .82)

    lab = cv2.GaussianBlur(lab_of(rgb), (0, 0), 1.3)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1)
    edge = np.uint8(np.hypot(gx, gy).max(axis=2) > 12)
    kernel = np.ones((3, 3), np.uint8)
    free = 1 - cv2.dilate(edge, kernel)
    free[y:, x:] = 0
    _, regions = cv2.connectedComponents(free, connectivity=4)
    border = np.unique(np.concatenate((regions[0], regions[-1], regions[:, 0], regions[:, -1])))
    mask = np.uint8(~np.isin(regions, border[border != 0]))
    mask[y:, x:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask_area = float(mask.mean())
    area = mask_area
    reasons = []
    if not .025 < area < .72:
        reasons.append('foreground area outside expected range')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        reasons.append('foreground touches frame')
    if count < 2 or stats[1:, cv2.CC_STAT_AREA].max() < mask.sum() * .80:
        reasons.append('foreground lacks a dominant component')
    if mask[y - 1, x:].any() or mask[y:, x - 1].any():
        reasons.append('foreground touches label exclusion')
    if not rule:
        reasons.append('catalogue label rule not detected')

    if reasons:
        # Paint typography in the lower-right only. Never fill a large rectangle
        # through the garment.
        rx, ry = round(width * .55), round(height * .72)
        if rule:
            rx = min(rx, max(0, int(rule[0]) - 24))
            ry = min(ry, max(0, int(rule[1]) - round(height * .04)))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sub = gray[ry:, rx:]
        dark = np.uint8(sub < 160)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
        typo = np.zeros(sub.shape, np.uint8)
        corner_area = sub.size
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w_i, h_i = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Skip only obviously huge masses (the garment).
            if a > corner_area * .20:
                continue
            typo[labels == i] = 255
        typo = cv2.dilate(typo, np.ones((11, 11), np.uint8))
        patch = rgb[max(0, ry - max(4, height // 30)):max(1, ry), min(width-8, rx + (width-rx)//2):]
        if patch.size == 0:
            patch = rgb[:max(8, height // 40), rx:]
        fill = np.median(patch, axis=(0, 1)).astype(np.uint8)
        corner = rgb[ry:, rx:].copy()
        corner[typo > 0] = fill
        rgb[ry:, rx:] = corner
        alpha = np.full((height, width), 255, np.uint8)
        method = 'full-frame-label-fill'
        method = 'full-frame-label-fill'
    else:
        alpha = cv2.GaussianBlur(mask * 255, (0, 0), .8)
        alpha[y:, x:] = 0
        method = 'border-flood'

    return Image.fromarray(np.dstack((rgb, alpha))), dict(
        method=method, label_box=[x, y, width, height], foreground_fraction=round(mask_area, 4),
        gate_reasons=reasons,
    )


def has_metadata(path):
    with Image.open(path) as image:
        if any(key.lower() in {'exif', 'icc_profile', 'xmp', 'gps'} for key in image.info):
            return True
        if image.getexif():
            return True
    # Check container chunks as well, including metadata Pillow may ignore.
    with path.open('rb') as stream:
        stream.seek(12)
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError(f'Truncated WebP chunk: {path}')
            if header[:4] in {b'EXIF', b'ICCP', b'XMP ', b'GPS '}:
                return True
            length = int.from_bytes(header[4:], 'little')
            stream.seek(length + length % 2, 1)
    return False


def save_card(image, size, path):
    import numpy as np

    resized = image.resize(size, Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    if (pixels[:, :, 3] == 255).all():
        # Keep an alpha channel even on the full-frame fallback. libwebp
        # drops the channel when every pixel is fully opaque.
        pixels[0, 0, 3] = 254
    clean = Image.fromarray(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        candidate = Path(temporary) / path.name
        for _ in range(2):
            clean.save(candidate, 'WEBP', quality=80, method=4)
            if not has_metadata(candidate):
                with Image.open(candidate) as check:
                    if check.mode != 'RGBA' or check.size != size:
                        raise ValueError(f'Unexpected WebP mode or dimensions: {path}')
                    check.load()
                candidate.replace(path)
                return
            clean = Image.fromarray(np.asarray(clean))
    raise ValueError(f'Metadata remains after rewriting {path}')


def update_catalog(items, cards):
    catalog_path = ROOT / 'data/catalog.json'
    zones_path = ROOT / 'data/source/zones.json'
    catalog = json.loads(catalog_path.read_text())
    source_zones = json.loads(zones_path.read_text())
    categories = catalog['categories']
    if not any(category['id'] == 'totes' for category in categories):
        index = next(i for i, category in enumerate(categories) if category['id'] == 'bags') + 1
        categories.insert(index, dict(id='totes', label='托特包', label_en='Totes',
                                      order=index, cover_zone='totes-a', count=0))

    grouped = defaultdict(list)
    new_items = []
    for item in items:
        entry = {key: item[key] for key in ('id', 'category', 'zone', 'number', 'name', 'brand', 'type')}
        entry.update(display_name=item['name'], story='故事待补充', hotspot=None, layer=None)
        if item['photo']:
            variants = cards[item['id']]
            entry.update(image=variants[0]['src'], card_images=variants)
        new_items.append(entry)
        grouped[item['zone']].append(item['id'])

    zone_index = {zone['id']: zone for zone in catalog['zones']}
    for zone_id, members in grouped.items():
        category, letter = zone_id.rsplit('-', 1)
        zone = dict(id=zone_id, category=category, letter=letter, has_layout=False, items=members)
        if zone_id in zone_index:
            existing = zone_index[zone_id]
            if existing.get('has_layout'):
                raise ValueError(f'Unexpected existing catalog zone: {zone_id}')
            existing.update(zone)
        else:
            catalog['zones'].append(zone)
            zone_index[zone_id] = zone
        letters = source_zones.setdefault(category, {})
        # New list-only zones (no paintings) may be rewritten on re-run; never touch furnished letters.
        if letter in letters and letters[letter] != members:
            if zone_id in zone_index and zone_index[zone_id].get('has_layout'):
                raise ValueError(f'Unexpected existing membership: {zone_id}')
        letters[letter] = members

    ids = {item['id'] for item in new_items}
    catalog['items'] = [item for item in catalog['items'] if item['id'] not in ids] + new_items
    counts = Counter(item['category'] for item in catalog['items'])
    for order, category in enumerate(categories):
        category.update(order=order, count=counts[category['id']])

    write_json(catalog_path, catalog, newline=catalog_path.read_bytes().endswith(b'\n'))
    write_json(zones_path, source_zones, indent=1, newline=zones_path.read_bytes().endswith(b'\n'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--catalog-only', action='store_true', help='Register items without preparing cards')
    mode.add_argument('--sample', nargs='+', metavar='ID', help='Test one or two cards in a temporary directory')
    args = parser.parse_args()
    items = json.loads((SOURCE / 'items.json').read_text())['items']
    if {item['id'] for item in items if not item['photo']} != NO_PHOTO:
        raise ValueError('Unexpected photo exclusions in source manifest')
    if args.sample:
        by_id = {item['id']: item for item in items}
        if len(args.sample) > 2 or any(i not in by_id for i in args.sample):
            parser.error('--sample requires one or two manifest IDs')
        with TemporaryDirectory() as temporary:
            for item_id in args.sample:
                item = by_id[item_id]
                destination = Path(temporary) / ('site' / CARDS if item['photo'] else Path('pending-doubao'))
                _, result = prepare(item, destination)
                location = 'site cards' if item['photo'] else 'pending-doubao'
                print(f'{item_id} ({location}): {json.dumps(result)}')
        return

    cards, log = {}, {}
    for item in items:
        item_id = item['id']
        if args.catalog_only:
            if item['photo']:
                with Image.open(original(item)) as source:
                    size = source.size
                    if source.getexif().get(274) in (5, 6, 7, 8):
                        size = size[::-1]
                    cards[item_id] = card_specs(item, size)
        else:
            destination = ROOT / CARDS if item['photo'] else PENDING
            cards[item_id], log[item_id] = prepare(item, destination)
    if not args.catalog_only:
        write_json(SOURCE / 'cutout-log.json', log, indent=1)
    update_catalog(items, cards)
    for (category, subtype), count in sorted(Counter((i['category'], i['type']) for i in items).items()):
        print(f'{category} / {subtype}: {count}')
    for category, count in sorted(Counter(i['category'] for i in items).items()):
        print(f'{category}: {count}')
    print(f'Total batch items: {len(items)}')


def prepare(item, destination):
    with Image.open(original(item)) as source:
        image, result = cutout(ImageOps.exif_transpose(source))
    specs = card_specs(item, image.size)
    for spec in specs:
        save_card(image, (spec['width'], spec['height']), destination / Path(spec['src']).name)
    return specs, result


if __name__ == '__main__':
    main()
