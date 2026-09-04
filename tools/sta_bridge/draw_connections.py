"""Draw what the graph actually connected, and what it left behind.

``14_graph_connections.png`` cannot answer "is A connected to B": it gives each
asset a random colour and paints that asset's own path segments, so two
connected assets come out in unrelated colours and nothing is drawn between
them. This draws the graph itself.

Two views, written side by side:

``20_connections.png``
    A straight line between the centres of every connected pair, over a faded
    drawing. Asset dots are sized by degree, and an asset the graph could not
    connect to anything is boxed in red and named. A healthy P&ID graph looks
    like the piping; a blob of criss-crossing lines means it over-connected.

``21_orphan_lines.png``
    Every detected segment, coloured by what became of it: blue on a connection
    path, orange for one lying entirely inside a symbol or text box, purple for
    an unused dashed run, red for unused pipe. The drawing underneath is faded
    hard, so anything still pale grey is ink no segment covers at all -- Hough
    did not detect it. Red means the opposite: detected, but the graph could not
    attach it to an asset.

Usage:
    python -m tools.sta_bridge.draw_connections --output-dir out/18-final
"""
import argparse
import json
import os

import cv2
import numpy as np

USED = (150, 60, 20)        # BGR, deep blue: segment on some connection path
ORPHAN = (60, 60, 230)      # BGR, red: detected but unused
EDGE = (150, 60, 20)
DOT = (40, 40, 40)
ISOLATED = (40, 40, 220)    # BGR, red: asset the graph could not connect
DASHED = (200, 60, 200)     # BGR, purple: orphaned dashed
BOXED = (30, 160, 240)      # BGR, orange: entirely inside a symbol or text box


def _fade(img, keep=0.12):
    return cv2.addWeighted(img, keep, np.full_like(img, 255), 1 - keep, 0)


def _centre(box, w, h):
    return (int((box['topX'] + box['bottomX']) / 2 * w),
            int((box['topY'] + box['bottomY']) / 2 * h))


def draw_connections(graph, img, w, h, path):
    """Straight line per connected pair; dot size follows degree."""
    out = _fade(img)
    by_id = {a['id']: a for a in graph}
    seen = set()
    for a in graph:
        p1 = _centre(a['bounding_box'], w, h)
        for c in a['connections']:
            pair = tuple(sorted((a['id'], c['id'])))
            if pair in seen or c['id'] not in by_id:
                continue
            seen.add(pair)
            cv2.line(out, p1, _centre(by_id[c['id']]['bounding_box'], w, h),
                     EDGE, 2, cv2.LINE_AA)
    n_iso = 0
    for a in graph:
        deg = len(a['connections'])
        c = _centre(a['bounding_box'], w, h)
        b = a['bounding_box']
        if deg:
            cv2.circle(out, c, int(4 + min(deg, 40) * 0.5), DOT, -1, cv2.LINE_AA)
            continue
        # Unconnected: box it and name it, so the gaps are findable on the sheet.
        n_iso += 1
        cv2.rectangle(out, (int(b['topX'] * w) - 3, int(b['topY'] * h) - 3),
                      (int(b['bottomX'] * w) + 3, int(b['bottomY'] * h) + 3),
                      ISOLATED, 3)
        cv2.putText(out, a['text_associated'] or f"id {a['id']}",
                    (int(b['topX'] * w) - 3, int(b['topY'] * h) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ISOLATED, 2, cv2.LINE_AA)
    cv2.imwrite(path, out)
    return len(seen), n_iso


def draw_orphans(graph, segments, symbols, img, w, h, path):
    """Colour each segment by whether a connection path used it."""
    used = set()
    for a in graph:
        for c in a['connections']:
            for s in c['segments']:
                used.add((round(s['topX'], 4), round(s['topY'], 4),
                          round(s['bottomX'], 4), round(s['bottomY'], 4)))

    out = _fade(img, 0.10)
    n_used = 0
    for s in segments:
        k = (round(s['startX'], 4), round(s['startY'], 4),
             round(s['endX'], 4), round(s['endY'], 4))
        is_used = k in used
        n_used += is_used
        if is_used:
            colour = USED
        elif s.get('inside_box'):
            colour = BOXED
        elif s.get('line_type') == 'dashed':
            colour = DASHED
        else:
            colour = ORPHAN
        cv2.line(out,
                 (int(s['startX'] * w), int(s['startY'] * h)),
                 (int(s['endX'] * w), int(s['endY'] * h)),
                 colour, 3 if is_used else 2, cv2.LINE_AA)

    # Every detected symbol, not just the ones that became assets: a symbol
    # without a usable tag is still masked out of the image, so it has to show
    # up here or the gap it leaves looks unexplained.
    asset_boxes = {(round(a['bounding_box']['topX'], 4),
                    round(a['bounding_box']['topY'], 4)) for a in graph}
    for s in symbols:
        is_asset = (round(s['topX'], 4), round(s['topY'], 4)) in asset_boxes
        cv2.rectangle(out, (int(s['topX'] * w), int(s['topY'] * h)),
                      (int(s['bottomX'] * w), int(s['bottomY'] * h)),
                      (110, 110, 110) if is_asset else (150, 190, 150),
                      1 if is_asset else 2)
    cv2.imwrite(path, out)
    return n_used


def draw_leftover_text(attachments, img, w, h, path):
    """Show what each unclaimed label was attached to.

    Green box: the text. Blue: a line it went to. Orange: a symbol. A hairline
    joins the pair, so a wrong attachment is visible as a leader running
    somewhere it should not.
    """
    out = _fade(img, 0.12)
    for a in attachments:
        tx1, ty1, tx2, ty2 = a['text_box']
        p1 = (int((tx1 + tx2) / 2 * w), int((ty1 + ty2) / 2 * h))
        cv2.rectangle(out, (int(tx1 * w), int(ty1 * h)), (int(tx2 * w), int(ty2 * h)),
                      (60, 160, 60), 2)

        bx1, by1, bx2, by2 = a['target_box']
        if a['target'].startswith('line'):
            cv2.line(out, (int(bx1 * w), int(by1 * h)), (int(bx2 * w), int(by2 * h)),
                     (200, 90, 30), 3, cv2.LINE_AA)
            p2 = (int((bx1 + bx2) / 2 * w), int((by1 + by2) / 2 * h))
        else:
            cv2.rectangle(out, (int(bx1 * w), int(by1 * h)),
                          (int(bx2 * w), int(by2 * h)), (30, 160, 240), 2)
            p2 = (int((bx1 + bx2) / 2 * w), int((by1 + by2) / 2 * h))
        cv2.line(out, p1, p2, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imwrite(path, out)
    return len(attachments)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    d = args.output_dir

    img = cv2.imread(os.path.join(d, 'diagram.png'))
    if img is None:
        raise SystemExit(f'ERROR: no diagram.png in {d}')
    h, w = img.shape[:2]

    with open(os.path.join(d, 'graph_connectivity.json')) as f:
        graph = json.load(f)
    with open(os.path.join(d, 'line_detection.json')) as f:
        segments = json.load(f)['line_segments']
    td = os.path.join(d, 'text_detection.json')
    if not os.path.exists(td):
        raise SystemExit(f'ERROR: no text_detection.json in {d} (needed to draw '
                         f'every detected symbol, not only the assets)')
    with open(td) as f:
        symbols = json.load(f)['text_and_symbols_associated_list']

    n_edges, n_iso = draw_connections(graph, img, w, h,
                                      os.path.join(d, '20_connections.png'))
    n_used = draw_orphans(graph, segments, symbols, img, w, h,
                          os.path.join(d, '21_orphan_lines.png'))

    leftover_path = os.path.join(d, 'leftover_text.json')
    if os.path.exists(leftover_path):
        with open(leftover_path) as f:
            attachments = json.load(f)
        n = draw_leftover_text(attachments, img, w, h,
                               os.path.join(d, '22_leftover_text.png'))
        print(f'leftover text attachments drawn: {n}')

    linked = sum(1 for a in graph if a['connections'])
    print(f'assets {len(graph)} ({linked} linked, {n_iso} isolated in red), '
          f'undirected edges {n_edges}')
    print(f'symbols {len(symbols)} drawn ({len(graph)} of them assets)')
    print(f'segments {len(segments)}: {n_used} on a path, '
          f'{len(segments) - n_used} orphaned '
          f'({(len(segments) - n_used) / max(1, len(segments)) * 100:.0f}%)')
    print(f'wrote {d}/20_connections.png and {d}/21_orphan_lines.png')


if __name__ == '__main__':
    main()
