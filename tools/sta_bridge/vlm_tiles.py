"""Hand the sheet to a VLM tile by tile, instead of to RF-DETR and PaddleOCR.

Same shape as the detector path -- slice, run each tile, map back, merge -- but
the tile is answered by a vision model reading the picture rather than by a
network trained on 32 classes. What that buys is a reader that can say *why* a
symbol is what it is, tie a tag to it from the drawing's own conventions, and
name something outside the 32 classes. What it costs is a call per tile.

This adds a path; it changes nothing. ``PID_pipeline_`` and ``export`` are
untouched, and the output is the same ``text_detection.json`` the rest of the
pipeline already reads, so line detection and the graph stages cannot tell
which reader produced it.

Two steps, because the model is you:

    # 1. cut the sheet up and write the prompt
    python -m tools.sta_bridge.vlm_tiles slice \
        --image out/page75/diagram.png --output-dir out/p75-vlm

    # ... read out/p75-vlm/tiles/tile_r0c0.png and friends, and write each
    #     answer to out/p75-vlm/answers/tile_r0c0.json ...

    # 2. put the answers back together
    python -m tools.sta_bridge.vlm_tiles collect --output-dir out/p75-vlm

The overlap is what makes a symbol on a tile edge recoverable: it is whole on
the neighbour. Merging then has to undo the duplication that creates, which is
the same class-agnostic NMS the detector path uses.
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

TILE = 1280
OVERLAP = 256
NMS_IOU = 0.5
TEXT_NMS_IOU = 0.4


PROMPT = '''You are reading one tile of a piping and instrumentation diagram
(P&ID). Report EVERYTHING in THIS tile, in tile pixel coordinates with the
origin at the tile's top-left.

Return one JSON object, nothing else:

{
  "symbols": [
    {"cls": "<class>", "bbox": [x1, y1, x2, y2], "score": 0.0-1.0,
     "tag": "<the tag written beside it, or null>"}
  ],
  "texts": [
    {"text": "<exactly as written>", "bbox": [x1, y1, x2, y2]}
  ]
}

## Completeness comes first

Report every symbol you can see. **Most symbols on a P&ID carry no tag at
all** -- flanges, reducers, flow arrows, blinds, couplings -- and they count
exactly as much as a tagged valve. A symbol with "tag": null is a correct and
expected answer; leaving it out is not.

Work through the tile deliberately rather than reporting what catches the eye:

1. Follow every pipe run from one end of the tile to the other. On each run,
   note every mark sitting *on* the line: valves, flanges, reducers,
   couplings, blinds, flow arrows, insulation marks. These are small, they
   repeat, and they are what gets skipped.
2. Then the circles and rectangles off to the side: instrument bubbles, panel
   boxes, tables, note boxes.
3. Then the equipment outlines: tanks, pumps, vessels.
4. Before you answer, count the valves you have listed and sweep the tile once
   more for ones you did not.

A dense tile of a P&ID like this holds on the order of 40-80 symbols. If your
list is much shorter than that and the tile is not mostly blank, you have
missed some -- go back.

## Rules

* "cls" must be one of the classes listed below. If a symbol is clearly present
  but fits none of them, use "Other" and put what you would call it in "tag".
* A symbol cut off by the tile edge: report it anyway, with the box clipped to
  the tile. It is whole on the neighbouring tile and the merge will prefer that
  one.
* "tag" is the association step: the identifier written next to the symbol that
  names it (V147, F131A, PI 104). Give a tag only where you can actually read
  one next to that symbol; null otherwise -- never guess, and never borrow a
  tag from a neighbouring symbol. A room name, an elevation, a pipe label
  ("2.5" RCF L147") and a table cell are not tags.
* Every piece of text goes in "texts" as well, tagged or not, including the
  ones you used as tags.
* Do not report the pipe lines themselves. Line detection is a separate stage.
* Boxes are tight around the symbol, excluding its tag.
* Where two symbols sit side by side on one run -- a pair of flanges around a
  valve, say -- report each separately rather than one box over both.

Classes:
{CLASSES}
'''


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ios(a, b):
    """Intersection over the smaller box -- what SAHI uses to rejoin a box the
    slicing cut in half, where IoU alone would keep both."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    smaller = min(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
                  max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def tiles_for(width, height, tile, overlap):
    """Top-left corners covering the image, the last row and column pulled back
    inside the edge rather than left short."""
    step = tile - overlap
    xs = list(range(0, max(width - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(height - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(c, r, max(0, x), max(0, y))
            for r, y in enumerate(ys) for c, x in enumerate(xs)]


def do_slice(args):
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f'ERROR: cannot read {args.image}')
    height, width = image.shape[:2]

    # dataset.yaml is read by hand rather than with PyYAML: this runs in the
    # pipeline's venv, which does not carry it, and the file is a flat
    # "  <id>: <name>" list under names:.
    here = os.path.dirname(os.path.abspath(__file__))
    classes, in_names = set(), False
    for line in open(os.path.join(here, 'dataset.yaml')):
        if line.startswith('names:'):
            in_names = True
            continue
        if in_names:
            if not line.startswith((' ', '\t')):
                break
            if ':' in line:
                name = line.split(':', 1)[1].strip()
                if name and name != 'Not_used':
                    classes.add(name)
    classes = sorted(classes)

    tile_dir = os.path.join(args.output_dir, 'tiles')
    os.makedirs(tile_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'answers'), exist_ok=True)

    records = []
    for c, r, x, y in tiles_for(width, height, args.tile, args.overlap):
        piece = image[y:y + args.tile, x:x + args.tile]
        name = f'tile_r{r}c{c}.png'
        cv2.imwrite(os.path.join(tile_dir, name), piece)
        records.append({'file': name, 'x': int(x), 'y': int(y),
                        'w': int(piece.shape[1]), 'h': int(piece.shape[0])})

    with open(os.path.join(args.output_dir, 'tiles.json'), 'w') as f:
        json.dump({'image': os.path.abspath(args.image),
                   'image_details': {'format': 'png', 'width': width,
                                     'height': height},
                   'tile': args.tile, 'overlap': args.overlap,
                   'tiles': records}, f, indent=1)
    with open(os.path.join(args.output_dir, 'PROMPT.md'), 'w') as f:
        f.write(PROMPT.replace('{CLASSES}',
                               '\n'.join('  - ' + c for c in classes)))

    print(f'{width}x{height} -> {len(records)} tiles of {args.tile}px '
          f'(overlap {args.overlap})')
    print(f'  tiles   {tile_dir}/')
    print(f'  prompt  {os.path.join(args.output_dir, "PROMPT.md")}')
    print(f'  answers go in {os.path.join(args.output_dir, "answers")}/'
          f'<tile name>.json')


def do_collect(args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from tools.sta_bridge.label_map import map_label

    with open(os.path.join(args.output_dir, 'tiles.json')) as f:
        meta = json.load(f)
    width = meta['image_details']['width']
    height = meta['image_details']['height']

    symbols, texts, answered = [], [], 0
    for rec in meta['tiles']:
        path = os.path.join(args.output_dir, 'answers',
                            os.path.splitext(rec['file'])[0] + '.json')
        if not os.path.exists(path):
            continue
        answered += 1
        with open(path) as f:
            ans = json.load(f)
        ox, oy = rec['x'], rec['y']
        for s in ans.get('symbols', []):
            x1, y1, x2, y2 = s['bbox']
            symbols.append({'cls': s.get('cls', 'Other'),
                            'score': float(s.get('score', 0.5)),
                            'tag': s.get('tag'),
                            'bbox': [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                            'from_tile': rec['file']})
        for t in ans.get('texts', []):
            x1, y1, x2, y2 = t['bbox']
            texts.append({'text': t['text'],
                          'bbox': [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                          'from_tile': rec['file']})

    print(f'{answered}/{len(meta["tiles"])} tiles answered: '
          f'{len(symbols)} symbols, {len(texts)} texts')
    if not answered:
        raise SystemExit('ERROR: no answers found')

    # The overlap that saves an edge symbol reports it twice. Class-agnostic,
    # for the reason the detector path is: one symbol per position, and two
    # boxes on one valve compete for its single tag.
    def suppress(items, threshold, metric):
        kept = []
        for it in sorted(items, key=lambda s: -s.get('score', 0.5)):
            if any(metric(it['bbox'], k['bbox']) > threshold for k in kept):
                continue
            kept.append(it)
        return kept

    n_sym, n_txt = len(symbols), len(texts)
    symbols = suppress(symbols, args.nms_iou, _ios)
    texts = suppress(texts, args.text_nms_iou, _iou)
    print(f'  after merging across tiles: {len(symbols)} symbols '
          f'(-{n_sym - len(symbols)}), {len(texts)} texts (-{n_txt - len(texts)})')

    out_symbols, unmapped = [], []
    for i, s in enumerate(symbols):
        label = map_label(s['cls'])
        if label is None:
            unmapped.append(s['cls'])
            continue
        b = s['bbox']
        out_symbols.append({
            'id': len(out_symbols), 'label': label, 'score': s['score'],
            'text_associated': s.get('tag'),
            'topX': b[0] / width, 'topY': b[1] / height,
            'bottomX': b[2] / width, 'bottomY': b[3] / height})
    if unmapped:
        from collections import Counter
        print('  no label mapping, dropped: '
              + ', '.join(f'{k}x{v}' for k, v in Counter(unmapped).items()))

    payload = {
        'image_url': 'file://' + os.path.abspath(meta['image']),
        'image_details': meta['image_details'],
        'all_text_list': [{'text': t['text'],
                           'topX': t['bbox'][0] / width,
                           'topY': t['bbox'][1] / height,
                           'bottomX': t['bbox'][2] / width,
                           'bottomY': t['bbox'][3] / height} for t in texts],
        'text_and_symbols_associated_list': out_symbols,
    }
    out_path = os.path.join(args.output_dir, 'text_detection.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=1)

    image = cv2.imread(meta['image'])
    overlay = cv2.addWeighted(image, 0.25, np.full_like(image, 255), 0.75, 0)
    for t in texts:
        b = [int(v) for v in t['bbox']]
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (60, 160, 60), 1)
    for s in out_symbols:
        b = (int(s['topX'] * width), int(s['topY'] * height),
             int(s['bottomX'] * width), int(s['bottomY'] * height))
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (40, 40, 220), 2)
        if s['text_associated']:
            cv2.putText(overlay, str(s['text_associated']), (b[0], b[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 220), 1,
                        cv2.LINE_AA)
    png = os.path.join(args.output_dir, '05_vlm_detections.png')
    cv2.imwrite(png, overlay)

    tagged = sum(1 for s in out_symbols if s['text_associated'])
    print(f'wrote {out_path} ({len(out_symbols)} symbols, {tagged} with a tag, '
          f'{len(texts)} texts) and {png}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('slice', help='cut the sheet into tiles and write the prompt')
    s.add_argument('--image', required=True)
    s.add_argument('--output-dir', required=True)
    s.add_argument('--tile', type=int, default=TILE)
    s.add_argument('--overlap', type=int, default=OVERLAP)
    s.set_defaults(func=do_slice)

    c = sub.add_parser('collect', help='merge the per-tile answers')
    c.add_argument('--output-dir', required=True)
    c.add_argument('--nms-iou', type=float, default=NMS_IOU,
                   help='intersection-over-smaller above which two symbol boxes '
                        'are the same symbol seen from two tiles')
    c.add_argument('--text-nms-iou', type=float, default=TEXT_NMS_IOU)
    c.set_defaults(func=do_collect)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
