"""Split a drawing into its red layer and everything else.

Red on a P&ID is a second drawing laid over the first: a revision cloud, an
as-built markup, a highlighted subsystem. It is not part of the piping the
black line work describes, and left in place it is found as pipe, so the graph
gains runs that belong to the annotation rather than the plant.

This lifts the red out before anything else runs:

* every reddish pixel is collected, saved as its own layer, and its connected
  components recorded with their boxes, so the annotation survives as data;
* those pixels are erased from the working image, painted with the background;
* what remains is flattened to black on white, so the line work is one colour
  whatever it was drawn in.

The red test is deliberately loose -- orange, pink and magenta all count. In a
scan the red of a marker pen lands anywhere in that range, and a run that
should be red but reads as orange is worse than an orange run counted as red.

    .venv-pid/bin/python -m tools.sta_bridge.split_red \
        --image drawing.png --output-dir out/drawing
"""
import argparse
import json
import os

import cv2
import numpy as np


# Hue is 0-179 in OpenCV. Red wraps around the ends, and this window is wide on
# purpose: it takes in orange at one end and magenta at the other.
HUE_LOW_MAX = 25
HUE_HIGH_MIN = 150
MIN_SATURATION = 55
MIN_VALUE = 40

# A second, colour-space-free test, for pixels a scanner left pale or dark
# enough that hue is unreliable: red simply has to dominate the other channels.
MIN_DOMINANCE = 32

BACKGROUND = 255
INK_THRESHOLD = 200


def red_mask(image,
             hue_low_max=HUE_LOW_MAX,
             hue_high_min=HUE_HIGH_MIN,
             min_saturation=MIN_SATURATION,
             min_value=MIN_VALUE,
             min_dominance=MIN_DOMINANCE):
    """Boolean mask of the reddish pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    coloured = (sat >= min_saturation) & (val >= min_value)
    by_hue = coloured & ((hue <= hue_low_max) | (hue >= hue_high_min))

    b, g, r = cv2.split(image.astype(np.int16))
    by_dominance = (r - np.maximum(g, b)) >= min_dominance

    return by_hue | by_dominance


def split(image, mask, dilate=1, ink_threshold=INK_THRESHOLD):
    """Return (black-only image, red layer on white).

    *dilate* grows the mask a little before erasing. Red strokes are
    antialiased against the paper, so their edge pixels are half red and fail
    the test; without the grow they stay behind as a grey ghost of the stroke
    and line detection finds it.
    """
    grown = mask
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    red_layer = np.full_like(image, BACKGROUND)
    red_layer[mask] = image[mask]

    without_red = image.copy()
    without_red[grown] = BACKGROUND

    grey = cv2.cvtColor(without_red, cv2.COLOR_BGR2GRAY)
    black = np.full_like(grey, BACKGROUND)
    black[grey < ink_threshold] = 0
    return cv2.cvtColor(black, cv2.COLOR_GRAY2BGR), red_layer


def components(mask, min_area=12):
    """Connected components of the red, as boxes -- the annotation as data."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'area': int(area),
                    'centroid': [round(float(centroids[i][0]), 1),
                                 round(float(centroids[i][1]), 1)]})
    out.sort(key=lambda c: -c['area'])
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--name', default='diagram',
                   help='stem for the written images; default "diagram"')
    p.add_argument('--dilate', type=int, default=1,
                   help='pixels to grow the red mask before erasing, so the '
                        "stroke's antialiased edge goes with it")
    p.add_argument('--min-area', type=int, default=12,
                   help='red blobs smaller than this are not recorded as '
                        'components (they are still erased)')
    p.add_argument('--ink-threshold', type=int, default=INK_THRESHOLD,
                   help='grey level below which a pixel becomes black')
    p.add_argument('--min-saturation', type=int, default=MIN_SATURATION)
    p.add_argument('--min-dominance', type=int, default=MIN_DOMINANCE)
    p.add_argument('--hue-low-max', type=int, default=HUE_LOW_MAX)
    p.add_argument('--hue-high-min', type=int, default=HUE_HIGH_MIN)
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f'ERROR: cannot read {args.image}')
    height, width = image.shape[:2]

    mask = red_mask(image,
                    hue_low_max=args.hue_low_max,
                    hue_high_min=args.hue_high_min,
                    min_saturation=args.min_saturation,
                    min_dominance=args.min_dominance)
    black, red_layer = split(image, mask, args.dilate, args.ink_threshold)
    blobs = components(mask, args.min_area)

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda n: os.path.join(args.output_dir, n)  # noqa: E731
    cv2.imwrite(out(f'{args.name}.png'), black)
    cv2.imwrite(out(f'{args.name}_red.png'), red_layer)
    cv2.imwrite(out(f'{args.name}_red_mask.png'),
                (mask.astype(np.uint8) * 255))
    with open(out(f'{args.name}_red.json'), 'w') as f:
        json.dump({'image_details': {'width': width, 'height': height},
                   'source': os.path.abspath(args.image),
                   'red_pixels': int(mask.sum()),
                   'components': blobs}, f, indent=1)

    ink = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < args.ink_threshold).sum()
    print(f'{width}x{height}: {int(mask.sum())} red pixels '
          f'({mask.sum() / max(ink, 1) * 100:.1f}% of the ink), '
          f'{len(blobs)} components >= {args.min_area}px')
    print(f'wrote {out(args.name + ".png")} (red removed, rest black)')
    print(f'      {out(args.name + "_red.png")} and _red_mask.png, _red.json')


if __name__ == '__main__':
    main()
