"""Clean up the symbols found by the crop pass.

Two steps, in this order:

1. **Resolve overlaps.** Two crops that share ground report the same place
   twice, sometimes under different classes. Where two new symbols overlap, the
   more confident one stands.

2. **Re-judge the ones that swallowed something.** A new symbol whose box
   contains a symbol the full-sheet pass already found is almost always
   spurious -- on the fire-protection sheet a pipe run, a branch and a label
   together made a shape the detector called a panel box, 107x45px around a
   valve. Those get their own crop, generously padded, and are detected again;
   whatever comes back replaces the original claim.

The re-detection pads to 1280 for the same reason the crop pass does: the
detector resizes its input, so a small image arrives magnified far past the
scale the model was trained at.

    /home/rx/project/STA-main/.venv-detect/bin/python \
        -m tools.sta_bridge.refine_crop_symbols --output-dir out/bin-p75
"""
import argparse
import json
import os
import sys

DEFAULT_STA_ROOT = '/home/rx/project/STA-main'
CHECKPOINT = os.path.join('model', 'checkpoint_best_total.pth')


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


def _contains(outer, inner, slack=2.0):
    return (outer[0] - slack <= inner[0] and inner[2] <= outer[2] + slack and
            outer[1] - slack <= inner[1] and inner[3] <= outer[3] + slack)


def resolve_overlaps(symbols, min_iou=0.0):
    """Keep the most confident symbol wherever two of them overlap."""
    ordered = sorted(symbols, key=lambda s: -s['score'])
    kept = []
    for s in ordered:
        if any(_iou(s['bbox'], k['bbox']) > min_iou for k in kept):
            continue
        kept.append(s)
    return kept


def _find_weights(here, sta_root):
    for root in (here, sta_root):
        path = os.path.join(root, CHECKPOINT)
        if os.path.exists(path):
            return path
    raise SystemExit(f'ERROR: no {CHECKPOINT} under {here} or {sta_root}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sta-root', default=DEFAULT_STA_ROOT)
    p.add_argument('--overlap-iou', type=float, default=0.0,
                   help='IoU above which two new symbols count as the same place; '
                        '0 means any overlap at all')
    p.add_argument('--repad', type=int, default=40,
                   help='pixels of context added around a box being re-judged')
    p.add_argument('--pad-to', type=int, default=1280)
    p.add_argument('--symbol-conf', type=float)
    p.add_argument('--symbol-scale', type=float, default=1.8)
    p.add_argument('--iou-existing', type=float, default=0.3)
    p.add_argument('--rejudge-overlap', type=float, default=0.0,
                   help='a re-judged detection is only accepted where it overlaps '
                        'the box being re-judged. Without this the widened crop '
                        'returns the neighbours too: at 300px of context one '
                        'valve came back with eleven detections, ten of them '
                        'other symbols that happened to be nearby.')
    p.add_argument('--min-score', type=float, default=0.0,
                   help='drop anything below this confidence from the result')
    p.add_argument('--exclude-class', action='append', default=[],
                   help='drop this class from the result; repeatable')
    p.add_argument('--device')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import cv2
    import numpy as np
    import PID_pipeline_ as sta

    d = args.output_dir
    with open(os.path.join(d, 'crop_symbols.json')) as f:
        new_symbols = json.load(f)
    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)

    image = cv2.imread(os.path.join(d, 'diagram.png'))
    height, width = image.shape[:2]
    existing = [[s['topX'] * width, s['topY'] * height,
                 s['bottomX'] * width, s['bottomY'] * height]
                for s in td['text_and_symbols_associated_list']]

    print(f'new symbols in: {len(new_symbols)}')
    if args.exclude_class:
        before = len(new_symbols)
        new_symbols = [s for s in new_symbols
                       if s['class'] not in args.exclude_class]
        print(f'dropped {before - len(new_symbols)} of '
              f'{", ".join(args.exclude_class)}')
    survivors = resolve_overlaps(new_symbols, args.overlap_iou)
    print(f'after resolving overlaps: {len(survivors)}')

    suspect = [s for s in survivors
               if any(_contains(s['bbox'], e) for e in existing)]
    clean = [s for s in survivors if s not in suspect]
    print(f'contain an existing symbol, so re-judged: {len(suspect)}')

    replacements = []
    if suspect:
        id_to_name = sta.load_class_mapping(os.path.join(here, 'dataset.yaml'))
        device = sta.resolve_device(args.device)
        conf = args.symbol_conf if args.symbol_conf is not None \
            else sta.SYMBOL_DETECTION_CONFIDENCE
        weights = _find_weights(here, os.path.abspath(args.sta_root))
        model = sta.build_detection_model(weights, id_to_name, device, conf)

        for s in suspect:
            b = s['bbox']
            x1 = max(0, int(b[0]) - args.repad)
            y1 = max(0, int(b[1]) - args.repad)
            x2 = min(width, int(b[2]) + args.repad)
            y2 = min(height, int(b[3]) + args.repad)
            piece = image[y1:y2, x1:x2]
            if piece.size == 0:
                continue

            pad_x = pad_y = 0
            ph, pw = piece.shape[:2]
            if ph <= args.pad_to and pw <= args.pad_to:
                canvas = np.full((args.pad_to, args.pad_to, 3), 255, np.uint8)
                pad_x = (args.pad_to - pw) // 2
                pad_y = (args.pad_to - ph) // 2
                canvas[pad_y:pad_y + ph, pad_x:pad_x + pw] = piece
                piece = canvas

            try:
                found = sta.detect_symbols(piece, model, args.symbol_scale)
            except Exception as e:
                print(f'  re-judge failed at {b}: {e}')
                continue

            ox, oy = x1 - pad_x, y1 - pad_y
            for det in found:
                dx1, dy1, dx2, dy2 = det['bbox']
                box = [dx1 + ox, dy1 + oy, dx2 + ox, dy2 + oy]
                if _iou(box, b) <= args.rejudge_overlap:
                    continue                      # a neighbour, not this claim
                if max((_iou(box, e) for e in existing), default=0.0) \
                        >= args.iou_existing:
                    continue                      # already on the sheet
                replacements.append({
                    'class': det['class'], 'score': float(det['score']),
                    'bbox': box, 'from_crop': s.get('from_crop'),
                    'crop_text': s.get('crop_text'), 'rejudged': True,
                    'replaced': {'class': s['class'], 'score': s['score'],
                                 'bbox': s['bbox']},
                })
            print(f"  {s['class']:<22} {b[2]-b[0]:>4.0f}x{b[3]-b[1]:>3.0f} "
                  f"-> {len([r for r in replacements if r['replaced']['bbox'] == s['bbox']])} "
                  f"detection(s)")

    final = resolve_overlaps(clean + replacements, args.overlap_iou)
    if args.min_score:
        before = len(final)
        final = [f for f in final if f['score'] >= args.min_score]
        print(f'dropped {before - len(final)} below score {args.min_score}')
    print(f'final: {len(final)} '
          f'({len(clean)} kept, {len(replacements)} from re-judging, '
          f'overlaps resolved)')

    out_path = os.path.join(d, 'crop_symbols_refined.json')
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    overlay = np.full_like(image, 255)
    overlay = cv2.addWeighted(image, 0.25, overlay, 0.75, 0)
    for e in existing:
        cv2.rectangle(overlay, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])),
                      (170, 170, 170), 1)
    for m in final:
        b = m['bbox']
        colour = (30, 160, 240) if m.get('rejudged') else (40, 40, 220)
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      colour, 2)
        cv2.putText(overlay, f"{m['class'][:16]} {m['score']:.2f}",
                    (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    colour, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(d, '24_crop_symbols_refined.png'), overlay)

    from collections import Counter
    for cls, count in Counter(m['class'] for m in final).most_common():
        print(f'   {count:>3}  {cls}')
    print(f'wrote {out_path} and {d}/24_crop_symbols_refined.png')


if __name__ == '__main__':
    main()
