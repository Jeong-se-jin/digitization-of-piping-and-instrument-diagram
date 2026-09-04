"""Convert an STA-main export into the request payload this pipeline expects.

STA gives pixel coordinates against its *cropped* diagram; the models here use
coordinates normalized to [0, 1] against the image they are handed.  So the
image the downstream stages read must be STA's ``diagram.png``, not the
original scan -- ``run_local.py`` enforces that by reading the path recorded in
the export.

Usage:
    python -m tools.sta_bridge.adapt --export STA-main/results/18/sta_export.json \
                                     --output out/18/text_detection.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.sta_bridge.label_map import map_label  # noqa: E402


def _normalize(bbox, width, height):
    """[x1, y1, x2, y2] pixels -> the normalized topX/topY/bottomX/bottomY dict."""
    x1, y1, x2, y2 = bbox
    return {
        'topX': max(0.0, min(1.0, x1 / width)),
        'topY': max(0.0, min(1.0, y1 / height)),
        'bottomX': max(0.0, min(1.0, x2 / width)),
        'bottomY': max(0.0, min(1.0, y2 / height)),
    }


def adapt(export: dict, hough: dict | None = None) -> dict:
    width = export['image_width']
    height = export['image_height']

    symbols = []
    dropped = {}
    next_id = 0
    for sym in export['symbols']:
        label = map_label(sym['class'])
        if label is None:
            dropped[sym['class']] = dropped.get(sym['class'], 0) + 1
            continue
        # Re-number ids densely: graph construction uses the id as a node key
        # and assumes it indexes into this list.
        symbols.append({
            'id': next_id,
            'label': label,
            'score': sym['score'],
            'text_associated': sym.get('tag'),
            **_normalize(sym['bbox'], width, height),
        })
        next_id += 1

    texts = [
        {'text': t['text'], **_normalize(t['bbox'], width, height)}
        for t in export['texts']
    ]

    payload = {
        'image_url': export['image'],
        'image_details': {'format': 'png', 'width': width, 'height': height},
        'bounding_box_inclusive': None,
        'all_text_list': texts,
        'text_and_symbols_associated_list': symbols,
        # Hough params: None means "fall back to config defaults"
        'hough_threshold': None,
        'hough_min_line_length': None,
        'hough_max_line_gap': None,
        'hough_rho': None,
        'hough_theta': None,
        'thinning_enabled': None,
        'propagation_pass_exhaustive_search': False,
    }
    if hough:
        payload.update(hough)

    return payload, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True,
                        help='sta_export.json written by PID_pipeline_.py --export-pipeline')
    parser.add_argument('--output', required=True,
                        help='where to write the text-detection payload')
    args = parser.parse_args()

    with open(args.export) as f:
        export = json.load(f)

    payload, dropped = adapt(export)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)

    n_sym = len(payload['text_and_symbols_associated_list'])
    n_tagged = sum(1 for s in payload['text_and_symbols_associated_list']
                   if s['text_associated'])
    print(f'symbols kept : {n_sym} ({n_tagged} with a tag)')
    print(f'texts        : {len(payload["all_text_list"])}')
    if dropped:
        summary = ', '.join(f'{k}={v}' for k, v in sorted(dropped.items()))
        print(f'symbols dropped by label map: {summary}')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
