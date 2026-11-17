# Collection Room

Collection Room is a buildless static website that presents a personal collection as an illustrated room. Select objects in the hub to open item cards, or browse the illustrated category rooms. The hub uses responsive WebP images and transparent illustrated layers for hover and motion effects.

The site uses HTML, CSS, vanilla JavaScript, and JSON. This repository contains the code and a runnable sample with eight category rooms. Hub cards show painted cutouts, and category cards show room illustrations. The full experience requires the complete catalog and asset set, including the original item photos. See the [live demo](https://the-collection-room.surge.sh) for the full collection.

## Run locally

From the repository root, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/ in a browser. The site loads its JSON over HTTP.

## Optional illustration tool

`tools/cutout.py` generates transparent object layers from a furnished illustration and a pixel-aligned empty illustration. It requires Python 3, Pillow, NumPy, and OpenCV. Run `python3 tools/cutout.py --help` for input and output options. Serving the sample requires only the static files.
