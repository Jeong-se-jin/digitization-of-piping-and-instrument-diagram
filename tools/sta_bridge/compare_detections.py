"""Score two readers of the same sheet against each other.

Symbols, text and association only -- the stages before line detection. Takes
two ``text_detection.json`` payloads and matches them by geometry, so the
question becomes: where do they agree, and where does each find something the
other missed?

Neither side is ground truth, so nothing here is called correct. What it gives
is the disagreement, small enough to look at by eye and adjudicate.

Restrict the comparison to where both readers actually ran with ``--region``:
a VLM path answered for three tiles cannot be scored over the whole sheet.

    python -m tools.sta_bridge.compare_detections \
        --a out/page75/text_detection.json --a-name detector \
        --b out/p75-vlm/text_detection.json --b-name vlm \
        --region 1024,1024,3328,2304 --image out/page75/diagram.png \
        --output-dir out/p75-vlm
"""
import argparse
import collections
import json
import os
import re

import cv2
import numpy as np

MATCH_IOU = 0.3
TEXT_MATCH_IOU = 0.3


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load(path, region):
    with open(path) as f:
        td = json.load(f)
    width = td['image_details']['width']
    height = td['image_details']['height']

    def boxes(items):
        out = []
        for it in items:
            b = [it['topX'] * width, it['topY'] * height,
                 it['bottomX'] * width, it['bottomY'] * height]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if region and not (region[0] <= cx < region[2] and
                               region[1] <= cy < region[3]):
                continue
            out.append({**it, 'box': b})
        return out

    return (boxes(td['text_and_symbols_associated_list']),
            boxes(td.get('all_text_list') or []),
            width, height)


def match(a_items, b_items, threshold):
    """Greedy best-first pairing by IoU. Returns pairs and the two leftovers."""
    scored = sorted(((_iou(x['box'], y['box']), i, j)
                     for i, x in enumerate(a_items)
                     for j, y in enumerate(b_items)),
                    reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in scored:
        if iou <= threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, iou))
    return (pairs,
            [i for i in range(len(a_items)) if i not in used_a],
            [j for j in range(len(b_items)) if j not in used_b])


def _norm_tag(tag):
    if not tag:
        return None
    return re.sub(r'[^A-Z0-9]', '', str(tag).upper()) or None


def _norm_text(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().upper())


def leaf(label):
    return str(label).split('/')[-1]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--a-name', default='A')
    p.add_argument('--b-name', default='B')
    p.add_argument('--region', help='x1,y1,x2,y2 in pixels; compare only here')
    p.add_argument('--match-iou', type=float, default=MATCH_IOU)
    p.add_argument('--image', help='background for the disagreement overlay')
    p.add_argument('--output-dir')
    p.add_argument('--list', type=int, default=25,
                   help='how many of each disagreement to print')
    args = p.parse_args()

    region = [float(v) for v in args.region.split(',')] if args.region else None
    a_sym, a_txt, width, height = load(args.a, region)
    b_sym, b_txt, _, _ = load(args.b, region)

    A, B = args.a_name, args.b_name
    if region:
        print(f'region x {region[0]:.0f}-{region[2]:.0f}, '
              f'y {region[1]:.0f}-{region[3]:.0f}')

    # --- symbols ---------------------------------------------------------
    pairs, only_a, only_b = match(a_sym, b_sym, args.match_iou)
    print(f'\nSYMBOLS   {A}: {len(a_sym)}   {B}: {len(b_sym)}')
    print(f'  matched (IoU > {args.match_iou}): {len(pairs)}')
    print(f'  only {A}: {len(only_a)}     only {B}: {len(only_b)}')

    same_label = sum(1 for i, j, _ in pairs
                     if leaf(a_sym[i]['label']) == leaf(b_sym[j]['label']))
    print(f'  of the matched, same label: {same_label}/{len(pairs)} '
          f'({same_label / max(len(pairs), 1) * 100:.0f}%)')

    # --- association -----------------------------------------------------
    both_tagged = agree = 0
    a_only_tag = b_only_tag = neither = 0
    disagreed = []
    for i, j, _ in pairs:
        ta = _norm_tag(a_sym[i].get('text_associated'))
        tb = _norm_tag(b_sym[j].get('text_associated'))
        if ta and tb:
            both_tagged += 1
            if ta == tb:
                agree += 1
            else:
                disagreed.append((a_sym[i], b_sym[j]))
        elif ta:
            a_only_tag += 1
        elif tb:
            b_only_tag += 1
        else:
            neither += 1
    print(f'\nASSOCIATION  (on the {len(pairs)} matched symbols)')
    print(f'  both gave a tag: {both_tagged}, of which the same tag: {agree} '
          f'({agree / max(both_tagged, 1) * 100:.0f}%)')
    print(f'  only {A} gave a tag: {a_only_tag}     '
          f'only {B}: {b_only_tag}     neither: {neither}')
    ta_all = sum(1 for s in a_sym if s.get('text_associated'))
    tb_all = sum(1 for s in b_sym if s.get('text_associated'))
    print(f'  tagged overall  {A}: {ta_all}/{len(a_sym)}   '
          f'{B}: {tb_all}/{len(b_sym)}')

    # --- text ------------------------------------------------------------
    tpairs, t_only_a, t_only_b = match(a_txt, b_txt, TEXT_MATCH_IOU)
    exact = sum(1 for i, j, _ in tpairs
                if _norm_text(a_txt[i]['text']) == _norm_text(b_txt[j]['text']))
    print(f'\nTEXT      {A}: {len(a_txt)}   {B}: {len(b_txt)}')
    print(f'  matched: {len(tpairs)}   only {A}: {len(t_only_a)}   '
          f'only {B}: {len(t_only_b)}')
    print(f'  of the matched, identical string: {exact}/{len(tpairs)} '
          f'({exact / max(len(tpairs), 1) * 100:.0f}%)')

    # --- what to look at -------------------------------------------------
    if disagreed:
        print(f'\ntag disagreements ({len(disagreed)}):')
        for sa, sb in disagreed[:args.list]:
            print(f"  ({sa['box'][0]:>6.0f},{sa['box'][1]:>6.0f})  "
                  f"{A} {str(sa.get('text_associated')):<14} "
                  f"{B} {str(sb.get('text_associated'))}")
    if only_a:
        print(f'\nsymbols only {A} found ({len(only_a)}):')
        for i in only_a[:args.list]:
            s = a_sym[i]
            print(f"  ({s['box'][0]:>6.0f},{s['box'][1]:>6.0f})  "
                  f"{leaf(s['label']):<34} tag={s.get('text_associated')}")
    if only_b:
        print(f'\nsymbols only {B} found ({len(only_b)}):')
        for j in only_b[:args.list]:
            s = b_sym[j]
            print(f"  ({s['box'][0]:>6.0f},{s['box'][1]:>6.0f})  "
                  f"{leaf(s['label']):<34} tag={s.get('text_associated')}")

    mismatched = [(i, j) for i, j, _ in pairs
                  if leaf(a_sym[i]['label']) != leaf(b_sym[j]['label'])]
    if mismatched:
        counts = collections.Counter(
            (leaf(a_sym[i]['label']), leaf(b_sym[j]['label']))
            for i, j in mismatched)
        print(f'\nlabel disagreements, {A} -> {B} ({len(mismatched)}):')
        for (la, lb), n in counts.most_common(args.list):
            print(f'  {n:>3}  {la:<32} -> {lb}')

    # --- overlay ---------------------------------------------------------
    if args.image and args.output_dir:
        image = cv2.imread(args.image)
        if image is not None:
            out = cv2.addWeighted(image, 0.18, np.full_like(image, 255), 0.82, 0)
            for i, j, _ in pairs:
                b = [int(v) for v in a_sym[i]['box']]
                cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (150, 150, 150), 1)
            for i in only_a:
                b = [int(v) for v in a_sym[i]['box']]
                cv2.rectangle(out, (b[0] - 2, b[1] - 2), (b[2] + 2, b[3] + 2),
                              (200, 90, 30), 2)
            for j in only_b:
                b = [int(v) for v in b_sym[j]['box']]
                cv2.rectangle(out, (b[0] - 2, b[1] - 2), (b[2] + 2, b[3] + 2),
                              (40, 40, 220), 2)
            for sa, sb in disagreed:
                b = [int(v) for v in sa['box']]
                cv2.rectangle(out, (b[0] - 4, b[1] - 4), (b[2] + 4, b[3] + 4),
                              (30, 170, 30), 2)
            if region:
                cv2.rectangle(out, (int(region[0]), int(region[1])),
                              (int(region[2]), int(region[3])), (0, 0, 0), 1)

            # A legend on the image itself: the four colours are the whole
            # point of the picture and are not guessable from it.
            rows = [((150, 150, 150), 1, f'both agree  ({len(pairs)})'),
                    ((200, 90, 30), 2, f'only {A}  ({len(only_a)})'),
                    ((40, 40, 220), 2, f'only {B}  ({len(only_b)})'),
                    ((30, 170, 30), 2, f'tag disagreement  ({len(disagreed)})')]
            pad, line_h, sw = 18, 34, 30
            box_w, box_h = 430, pad * 2 + line_h * len(rows)
            x0, y0 = 24, 24
            cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h),
                          (255, 255, 255), -1)
            cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h),
                          (120, 120, 120), 1)
            for i, (colour, thickness, label) in enumerate(rows):
                y = y0 + pad + line_h * i + line_h // 2
                cv2.rectangle(out, (x0 + pad, y - 9),
                              (x0 + pad + sw, y + 9), colour, thickness)
                cv2.putText(out, label, (x0 + pad + sw + 14, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1,
                            cv2.LINE_AA)

            path = os.path.join(args.output_dir, '06_detection_diff.png')
            cv2.imwrite(path, out)
            print(f'\nwrote {path}  '
                  f'(grey = both, blue = only {A}, red = only {B}, '
                  f'green = tag disagreement)')


if __name__ == '__main__':
    main()
