"""Classify OCR text and cut a crop around each label, for a VLM to read.

Two jobs, both driven off a finished run's ``text_detection.json`` and
``line_detection.json``:

**Prose filter.** A drawing sheet carries two kinds of text. Tags and line
numbers are short and sit on the drawing; the notes paragraph is sentences,
set away from any symbol. Measured on sheet 18 the two barely overlap -- tags
run 6 characters and 1 word, notes 39 characters and 5 words -- so word count
separates them, and distance to the nearest symbol settles the few that word
count alone gets wrong. Three drawing labels there have four or more words
(``1 BBB L101D (V101D``, ``BBA CBC 1" CBC L152A``, ``3201 RCS PZR SPRA``) and
all three touch a symbol, while the nearest note sits 45px clear of one.

**Crops.** For each label, the nearest horizontal line above and below and the
nearest vertical line left and right are found, and the crop is the smallest
rectangle containing those four lines' midpoints. The label can fall outside
its own crop when a neighbouring line is long and offset -- that is inherent to
using midpoints, and the crop is padded and clamped rather than re-centred.

Usage:
    python -m tools.sta_bridge.text_crops --output-dir out/18 --crops
"""
import argparse
import json
import math
import os

import cv2

PROSE_MIN_WORDS = 4
PROSE_MIN_SYMBOL_DISTANCE = 40.0     # px
CROP_PADDING = 8                     # px
MIN_CROP_SIDE = 24                   # px


def _px_box(item, width, height):
    return (item['topX'] * width, item['topY'] * height,
            item['bottomX'] * width, item['bottomY'] * height)


def _box_gap(a, b):
    """Distance between two pixel boxes; 0 when they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def classify_prose(texts, symbols, width, height,
                   min_words: int = PROSE_MIN_WORDS,
                   min_symbol_distance: float = PROSE_MIN_SYMBOL_DISTANCE):
    """Mark each text as prose (notes) or a drawing label.

    :return: list of dicts with text, box, is_prose, words, symbol_distance
    """
    symbol_boxes = [_px_box(s, width, height) for s in symbols]
    out = []
    for t in texts:
        s = (t.get('text') or '').strip()
        box = _px_box(t, width, height)
        distance = min((_box_gap(box, sb) for sb in symbol_boxes), default=1e9)
        words = len(s.split())
        # A leading "12." is how the notes block numbers its entries; it settles
        # a line that wrapped and so carries fewer words than the first line.
        numbered = bool(s[:3].strip().rstrip('.').isdigit() and '.' in s[:4])
        is_prose = (words >= min_words or numbered) and distance > min_symbol_distance
        out.append({'text': s, 'box': [round(v, 1) for v in box],
                    'words': words, 'symbol_distance': round(distance, 1),
                    'is_prose': is_prose})
    return out


def _line_midpoints(segments, width, height, axis_tolerance=2.0,
                    skip_inside_box=True):
    """Split segments into axis-aligned horizontals and verticals with midpoints.

    Segments flagged ``inside_box`` are dropped: they lie entirely within a
    symbol or a text box, so they are that box's own drawing -- a glyph stroke
    or a bubble outline -- and taking one as the neighbouring pipe puts the crop
    boundary inside the label it is supposed to frame.
    """
    horizontals, verticals = [], []
    for s in segments:
        if skip_inside_box and s.get('inside_box'):
            continue
        x1, y1 = s['startX'] * width, s['startY'] * height
        x2, y2 = s['endX'] * width, s['endY'] * height
        mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if abs(y2 - y1) <= axis_tolerance and abs(x2 - x1) > axis_tolerance:
            horizontals.append({'level': (y1 + y2) / 2.0, 'mid': mid,
                                'lo': min(x1, x2), 'hi': max(x1, x2)})
        elif abs(x2 - x1) <= axis_tolerance and abs(y2 - y1) > axis_tolerance:
            verticals.append({'level': (x1 + x2) / 2.0, 'mid': mid,
                              'lo': min(y1, y2), 'hi': max(y1, y2)})
    return horizontals, verticals


def crop_box_for_text(box, horizontals, verticals, mode='midpoints'):
    """The rectangle spanned by four neighbouring lines' midpoints.

    Nearest horizontal above and below, nearest vertical left and right, each
    measured from the text's centre. A side with no line on it is dropped, so a
    label at the edge of the sheet still gets a crop from whatever it has.
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    def gap(line, axis):
        """Actual distance from the text box to the segment, not just its level.

        Measuring only the perpendicular offset picks a horizontal sitting a few
        pixels above the label but two thousand pixels off to the side, which is
        not the line anyone would call its neighbour.
        """
        if axis == 'h':
            dy = max(box[1] - line['level'], line['level'] - box[3], 0.0)
            dx = max(box[0] - line['hi'], line['lo'] - box[2], 0.0)
        else:
            dx = max(box[0] - line['level'], line['level'] - box[2], 0.0)
            dy = max(box[1] - line['lo'], line['lo'] - box[3], 0.0)
            dy = max(box[1] - line['hi'], line['lo'] - box[3], 0.0)
        return math.hypot(dx, dy)

    def nearest(lines, above):
        # Beyond the box edge, not merely beyond its centre: a line running
        # through the label is not its neighbour, and taking one as a boundary
        # cuts the text in half.
        best, best_d = None, None
        for line in lines:
            if above and line['level'] >= box[1]:
                continue
            if not above and line['level'] <= box[3]:
                continue
            d = gap(line, 'h')
            if best_d is None or d < best_d:
                best, best_d = line, d
        return best

    def nearest_v(lines, left):
        best, best_d = None, None
        for line in lines:
            if left and line['level'] >= box[0]:
                continue
            if not left and line['level'] <= box[2]:
                continue
            d = gap(line, 'v')
            if best_d is None or d < best_d:
                best, best_d = line, d
        return best

    above, below = nearest(horizontals, True), nearest(horizontals, False)
    left, right = nearest_v(verticals, True), nearest_v(verticals, False)
    picks = [above, below, left, right]
    found = sum(1 for p in picks if p)
    if not found:
        return None, 0, []

    if mode == 'bounds':
        # The lines' own positions bound the crop, so the label is inside it.
        # A line can be near the text yet have its midpoint far away -- one
        # horizontal 57px above a tag stretched to x=3754, which under the
        # midpoint rule dragged the crop out to 2727px wide with the tag
        # outside it.
        x1 = left['level'] if left else box[0]
        x2 = right['level'] if right else box[2]
        y1 = above['level'] if above else box[1]
        y2 = below['level'] if below else box[3]
        # Never smaller than the text itself.
        marks = [(p['mid'][0], p['mid'][1]) for p in picks if p]
        return (min(x1, box[0]), min(y1, box[1]),
                max(x2, box[2]), max(y2, box[3])), found, marks

    # Each axis takes its bounds from the lines that run across it: the
    # verticals set left and right, the horizontals set top and bottom. Pooling
    # all four midpoints instead lets a short horizontal sitting off to one side
    # drag the crop's width out with its own x -- which is what widened the FPP
    # crop by 100px for a line 11px long.
    xs = [p['mid'][0] for p in (left, right) if p] or \
         [p['mid'][0] for p in picks if p]
    ys = [p['mid'][1] for p in (above, below) if p] or \
         [p['mid'][1] for p in picks if p]
    mids = [p['mid'] for p in picks if p]
    return (min(xs), min(ys), max(xs), max(ys)), found, mids


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--crops', action='store_true', help='write the crop images')
    p.add_argument('--include-prose', action='store_true',
                   help='crop the notes text too; skipped by default')
    p.add_argument('--min-words', type=int, default=PROSE_MIN_WORDS)
    p.add_argument('--min-symbol-distance', type=float,
                   default=PROSE_MIN_SYMBOL_DISTANCE)
    p.add_argument('--padding', type=int, default=CROP_PADDING)
    p.add_argument('--crop-mode', choices=('midpoints', 'bounds'),
                   default='midpoints',
                   help="'midpoints' spans the four neighbouring lines' "
                        "midpoints; 'bounds' uses the lines' own positions, which "
                        'always keeps the label inside its crop')
    p.add_argument('--annotate', action='store_true',
                   help='draw the text box on the crop')
    p.add_argument('--limit', type=int, help='stop after this many crops')
    p.add_argument('--start', type=int, default=0,
                   help='skip this many crops before writing any')
    p.add_argument('--crop-dir-name', default='text_crops')
    p.add_argument('--keep-boxed-lines', action='store_true',
                   help='keep segments that lie entirely inside a symbol or text '
                        'box as crop boundaries; they are dropped by default')
    args = p.parse_args()
    d = args.output_dir

    image = cv2.imread(os.path.join(d, 'diagram.png'))
    if image is None:
        raise SystemExit(f'ERROR: no diagram.png in {d}')
    height, width = image.shape[:2]

    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)
    with open(os.path.join(d, 'line_detection.json')) as f:
        segments = json.load(f)['line_segments']

    texts = td['all_text_list']
    symbols = td['text_and_symbols_associated_list']

    classified = classify_prose(texts, symbols, width, height,
                                args.min_words, args.min_symbol_distance)
    n_prose = sum(1 for c in classified if c['is_prose'])
    print(f'text {len(classified)}: {n_prose} prose, '
          f'{len(classified) - n_prose} drawing labels')

    horizontals, verticals = _line_midpoints(segments, width, height,
                                             skip_inside_box=not args.keep_boxed_lines)
    n_boxed = sum(1 for s in segments if s.get('inside_box'))
    print(f'lines: {len(horizontals)} horizontal, {len(verticals)} vertical '
          f'({n_boxed} inside a symbol or text box '
          f'{"kept" if args.keep_boxed_lines else "dropped"})')

    crop_dir = os.path.join(d, args.crop_dir_name)
    if args.crops:
        os.makedirs(crop_dir, exist_ok=True)

    records, written, skipped, skipped_by_start = [], 0, 0, 0
    for i, c in enumerate(classified):
        if c['is_prose'] and not args.include_prose:
            continue
        crop, n_sides, marks = crop_box_for_text(
            c['box'], horizontals, verticals, args.crop_mode)
        if crop is None:
            skipped += 1
            continue

        x1 = max(0, int(crop[0]) - args.padding)
        y1 = max(0, int(crop[1]) - args.padding)
        x2 = min(width, int(crop[2]) + args.padding)
        y2 = min(height, int(crop[3]) + args.padding)
        if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
            skipped += 1
            continue

        records.append({
            'id': i, 'file': '', 'text': c['text'],
            'text_box': c['box'], 'crop_box': [x1, y1, x2, y2],
            'sides_found': n_sides,
            'contains_text': (x1 <= c['box'][0] and c['box'][2] <= x2 and
                              y1 <= c['box'][1] and c['box'][3] <= y2),
        })
        records[-1]['file'] = f'{i:04d}.png'

        if args.crops:
            piece = image[y1:y2, x1:x2].copy()
            if args.annotate:
                # Red: the text box. Blue: the four line midpoints the crop was
                # built from -- when the label sits outside its crop, these show
                # which neighbour pulled it away.
                cv2.rectangle(piece,
                              (int(c['box'][0]) - x1, int(c['box'][1]) - y1),
                              (int(c['box'][2]) - x1, int(c['box'][3]) - y1),
                              (40, 40, 220), 2)
                for mx, my in marks:
                    cv2.circle(piece, (int(mx) - x1, int(my) - y1), 6,
                               (220, 120, 20), -1, cv2.LINE_AA)
                    cv2.circle(piece, (int(mx) - x1, int(my) - y1), 6,
                               (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(crop_dir, records[-1]['file']), piece)
            written += 1

    with open(os.path.join(d, 'text_classification.json'), 'w') as f:
        json.dump(classified, f, indent=2, ensure_ascii=False)
    with open(os.path.join(d, 'text_crops.json'), 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    inside = sum(1 for r in records if r['contains_text'])
    print(f'crops {len(records)} ({inside} contain their own text), '
          f'{skipped} skipped for having no lines around them')
    if args.crops:
        print(f'wrote {written} images to {crop_dir}')
    print(f'wrote {d}/text_classification.json and {d}/text_crops.json')


if __name__ == '__main__':
    main()
