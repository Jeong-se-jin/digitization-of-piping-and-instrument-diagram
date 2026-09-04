"""Rescue the valves the detector called flow arrows.

On the fire-protection sheet the 45-degree filled bow-tie valve is detected
under two names: ``Gate_Valve_NC`` on 19 instances and ``Flow_Arrow`` on 9
more.  The two hypotheses come out of the same head and the class-agnostic NMS
in ``PID_pipeline_.deduplicate_symbols`` keeps whichever scored higher on that
instance, so the class flips from valve to valve.  The ``Flow_Arrow`` winners
then vanish from the graph, because an arrow is a direction marker and
``pre_find_symbol_connectivities`` skips it before it can become an asset.

Two things separate the misfires from the real arrows:

* **A tag.**  A flow arrow is never labelled; a valve carries ``V147``,
  ``F121`` and so on.  OCR association had already attached one to 13 of the
  22 arrows on that sheet.
* **Size.**  Long side 21px median for the tagged ones against 12px for the
  untagged -- and 21px is exactly the median of the ``Gate_Valve_NC``
  detections on the same sheet.

Either signal on its own is enough, so ``--require`` chooses.  Everything
promoted keeps its box and score; only the label changes.

    .venv-pid/bin/python -m tools.sta_bridge.promote_arrows \
        --text-detection out/page75/text_detection_merged.json
"""
import argparse
import json
import os
import re
import sys

ARROW_LABEL = 'Piping/Fittings/Mid arrow flow direction'
DEFAULT_TARGET = 'Instrument/Valve/Gate valve NC'

# A valve or instrument tag: a letter or two then digits, e.g. V147, F131A.
TAG_RE = re.compile(r'^[A-Z]{1,3}[-\s]?\d{2,4}[A-Z]?$')


def _looks_like_a_tag(text):
    return bool(text) and bool(TAG_RE.match(str(text).strip().upper()))


def promote(symbols, width, height, min_long_side, require,
            use_tag=True, use_size=True,
            arrow_label=ARROW_LABEL, target_label=DEFAULT_TARGET):
    """Relabel the arrows that are really valves. Returns the promoted ones."""
    promoted = []
    for s in symbols:
        if s.get('label') != arrow_label:
            continue
        long_side = max((s['bottomX'] - s['topX']) * width,
                        (s['bottomY'] - s['topY']) * height)
        by_tag = use_tag and _looks_like_a_tag(s.get('text_associated'))
        by_size = use_size and long_side >= min_long_side
        hit = (by_tag and by_size) if require == 'both' else (by_tag or by_size)
        if not hit:
            continue
        s['label'] = target_label
        s['promoted_from'] = arrow_label
        promoted.append({
            'id': s.get('id'), 'text': s.get('text_associated'),
            'score': s.get('score'), 'long_side': round(long_side),
            'by_tag': by_tag, 'by_size': by_size,
        })
    return promoted


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text-detection', required=True)
    p.add_argument('--output', help='defaults to overwriting --text-detection')
    p.add_argument('--min-long-side', type=float, default=18.0,
                   help='pixels; real arrows sit near 12, the valves near 21')
    p.add_argument('--require', choices=('tag', 'size', 'both', 'either'),
                   default='both',
                   help="'both' is deliberate: a tag alone promotes the small "
                        "instrument bubbles too, and size alone promotes the "
                        "one oversized genuine arrow")
    p.add_argument('--target-label', default=DEFAULT_TARGET)
    args = p.parse_args()

    with open(args.text_detection) as f:
        td = json.load(f)
    width = td['image_details']['width']
    height = td['image_details']['height']
    symbols = td['text_and_symbols_associated_list']

    before = sum(1 for s in symbols if s['label'] == ARROW_LABEL)
    promoted = promote(symbols, width, height, args.min_long_side,
                       'both' if args.require == 'both' else 'either',
                       use_tag=args.require != 'size',
                       use_size=args.require != 'tag',
                       target_label=args.target_label)

    out = args.output or args.text_detection
    with open(out, 'w') as f:
        json.dump(td, f, indent=2, ensure_ascii=False)

    print(f'arrows in: {before}')
    for r in sorted(promoted, key=lambda r: -(r['score'] or 0)):
        why = ','.join(k for k, v in (('tag', r['by_tag']), ('size', r['by_size'])) if v)
        print(f"  promoted {str(r['text'] or '-'):<14} {r['score']:.2f} "
              f"{r['long_side']:>3}px  ({why})")
    print(f'promoted {len(promoted)} to {args.target_label}, '
          f'{before - len(promoted)} arrows left')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
