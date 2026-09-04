"""Re-run symbol detection on the text crops and keep only what was missed.

The full-sheet pass runs the detector once over a 4000px drawing, so a symbol
drawn 40px across is a handful of pixels after the network's own downscaling.
Each text crop is a few hundred pixels holding one piece of the drawing, and the
same detector sees it far larger there.

Everything found is mapped back to sheet coordinates using the crop's offset,
then filtered twice:

* against the symbols the full-sheet pass already has, by IoU -- these are
  re-findings, not new information;
* against each other, since crops overlap and one symbol can be detected from
  several of them. The class-agnostic NMS from ``PID_pipeline_`` does that job.

Run with STA's interpreter -- this loads RF-DETR:

    /home/rx/project/STA-main/.venv-detect/bin/python -m tools.sta_bridge.crop_symbols \
        --output-dir out/bin-p75 --crop-dir-name crops_clean
"""
import argparse
import json
import os
import sys

DEFAULT_STA_ROOT = '/home/rx/project/STA-main'
CHECKPOINT = os.path.join('model', 'checkpoint_best_total.pth')


def _iou(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
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
    p.add_argument('--crop-dir-name', default='crops_clean',
                   help='crops to scan; must be the unannotated set')
    p.add_argument('--sta-root', default=DEFAULT_STA_ROOT)
    p.add_argument('--symbol-conf', type=float,
                   help="detection cut-off; defaults to PID_pipeline_'s constant")
    p.add_argument('--symbol-scale', type=float, default=1.8,
                   help='upscale applied to each crop before detection')
    p.add_argument('--pad-to', type=int, default=0,
                   help='paste the crop onto a white square of this size before '
                        'detecting. The detector resizes whatever it is given to a '
                        'fixed input, so a 200px crop arrives blown up far past the '
                        'scale it was trained at and it starts calling line '
                        'crossings symbols. Padding to the slice size the '
                        'full-sheet pass uses (1280) keeps symbols at that scale.')
    p.add_argument('--iou-existing', type=float, default=0.3,
                   help='IoU above which a detection counts as one the '
                        'full-sheet pass already found')
    p.add_argument('--device')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import cv2
    import numpy as np
    import PID_pipeline_ as sta

    d = args.output_dir
    crop_dir = os.path.join(d, args.crop_dir_name)
    if not os.path.isdir(crop_dir):
        raise SystemExit(f'ERROR: no {crop_dir}. Generate crops without '
                         f'--annotate first.')

    with open(os.path.join(d, 'text_crops.json')) as f:
        crops = json.load(f)
    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)

    image = cv2.imread(os.path.join(d, 'diagram.png'))
    height, width = image.shape[:2]
    existing = [[s['topX'] * width, s['topY'] * height,
                 s['bottomX'] * width, s['bottomY'] * height]
                for s in td['text_and_symbols_associated_list']]

    id_to_name = sta.load_class_mapping(os.path.join(here, 'dataset.yaml'))
    device = sta.resolve_device(args.device)
    conf = args.symbol_conf if args.symbol_conf is not None \
        else sta.SYMBOL_DETECTION_CONFIDENCE
    weights = _find_weights(here, os.path.abspath(args.sta_root))
    model = sta.build_detection_model(weights, id_to_name, device, conf)
    print(f'{len(crops)} crops, device={device}, conf={conf}, '
          f'scale=x{args.symbol_scale}')

    found = []
    for n, rec in enumerate(crops, 1):
        path = os.path.join(crop_dir, rec['file'])
        piece = cv2.imread(path)
        if piece is None:
            continue
        pad_x = pad_y = 0
        if args.pad_to:
            ph, pw = piece.shape[:2]
            if ph <= args.pad_to and pw <= args.pad_to:
                canvas = np.full((args.pad_to, args.pad_to, 3), 255, np.uint8)
                pad_x = (args.pad_to - pw) // 2
                pad_y = (args.pad_to - ph) // 2
                canvas[pad_y:pad_y + ph, pad_x:pad_x + pw] = piece
                piece = canvas

        try:
            detections = sta.detect_symbols(piece, model, args.symbol_scale)
        except Exception as e:                       # a crop can be tiny
            print(f'  {rec["file"]}: {e}')
            continue

        ox = rec['crop_box'][0] - pad_x
        oy = rec['crop_box'][1] - pad_y
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            found.append({
                'class': det['class'], 'score': float(det['score']),
                'bbox': [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                'from_crop': rec['file'], 'crop_text': rec['text'],
            })
        if n % 50 == 0:
            print(f'  {n}/{len(crops)} crops, {len(found)} raw detections')

    print(f'raw detections from crops: {len(found)}')

    # One symbol seen from several overlapping crops collapses here.
    for i, f in enumerate(found):
        f['id'] = i
        f['center'] = [(f['bbox'][0] + f['bbox'][2]) / 2,
                       (f['bbox'][1] + f['bbox'][3]) / 2]
        f['width'] = f['bbox'][2] - f['bbox'][0]
        f['height'] = f['bbox'][3] - f['bbox'][1]
    merged = sta.deduplicate_symbols(found)
    print(f'after merging across crops: {len(merged)}')

    new = [m for m in merged
           if max((_iou(m['bbox'], e) for e in existing), default=0.0)
           < args.iou_existing]
    print(f'not already on the sheet: {len(new)}')

    out_path = os.path.join(d, 'crop_symbols.json')
    with open(out_path, 'w') as f:
        json.dump(new, f, indent=2, ensure_ascii=False)

    overlay = cv2.addWeighted(image, 0.25,
                              cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                           cv2.COLOR_GRAY2BGR) * 0 + 255, 0.75, 0)
    for e in existing:
        cv2.rectangle(overlay, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])),
                      (170, 170, 170), 1)
    for m in new:
        b = m['bbox']
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (40, 40, 220), 2)
        cv2.putText(overlay, m['class'][:18], (int(b[0]), int(b[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 220), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(d, '23_crop_symbols.png'), overlay)

    from collections import Counter
    for cls, count in Counter(m['class'] for m in new).most_common(10):
        print(f'   {count:>4}  {cls}')
    print(f'wrote {out_path} and {d}/23_crop_symbols.png')


if __name__ == '__main__':
    main()
