"""Join the black graph and the red graph, and say what crosses between them.

``split_red`` leaves two drawings where there was one: the black line work and
the red overlay. Run each through the pipeline and each has its own runs,
symbols and connection points -- but nothing yet says where the overlay meets
the plant it was drawn on.

This pools both layers and runs the crossing search over the pool. Because the
two images are the same size and were never moved, their coordinates already
line up, so a red run crossing a black run is simply a junction whose two runs
carry different layer tags. The same holds for a red run reaching a black
symbol.

What comes out is one graph plus the list of links that span the layers -- the
part neither layer knew on its own.

    .venv-pid/bin/python -m tools.sta_bridge.merge_layers \
        --black out/ri2-x --red out/ri2-red-x --output-dir out/ri2-merged
"""
import argparse
import collections
import json
import os

import cv2
import numpy as np

from tools.sta_bridge.intersection_graph import (
    chain_points, find_points, _merge_points, _symbols_at, symbol_edges,
    ARROW_LABEL)


def load_layer(d, tag):
    """Runs and symbols from one output directory, tagged with its layer."""
    with open(os.path.join(d, 'intersection_graph.json')) as f:
        graph = json.load(f)
    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)
    width = graph['image_details']['width']
    height = graph['image_details']['height']

    runs = {'h': [], 'v': []}
    for r in graph['runs']:
        runs[r['axis']].append({'axis': r['axis'], 'pos': r['pos'],
                                'lo': r['lo'], 'hi': r['hi'],
                                'dashed': r.get('dashed', False),
                                'layer': tag})

    symbols = []
    for s in td['text_and_symbols_associated_list']:
        if s['label'] == ARROW_LABEL:
            continue
        symbols.append({'id': f"{tag}:{s['id']}", 'layer': tag,
                        'label': s['label'], 'text': s.get('text_associated'),
                        'box': [s['topX'] * width, s['topY'] * height,
                                s['bottomX'] * width, s['bottomY'] * height]})
    return runs, symbols, width, height


def stitch_collinear(runs, axis_tol, gap_tol):
    """Rejoin runs that are two halves of one pipe, one per layer.

    Where the overlay was drawn over part of a pipe, the black remainder and
    the red stretch are collinear neighbours -- not a crossing. The crossing
    search only ever pairs a horizontal with a vertical, so on its own it never
    puts the pipe back together, and everything past the painted stretch stays
    unreachable. This is the same stitching that built the runs in the first
    place, applied once more across the layers.
    """
    joined = 0
    for axis in ('h', 'v'):
        items = sorted(runs[axis], key=lambda r: (r['pos'], r['lo']))
        parent = list(range(len(items)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[j]['pos'] - items[i]['pos'] > axis_tol:
                    break                          # sorted by pos
                if items[j]['lo'] - items[i]['hi'] > gap_tol:
                    continue
                if items[i]['lo'] - items[j]['hi'] > gap_tol:
                    continue
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
                    joined += 1

        groups = collections.defaultdict(list)
        for i in range(len(items)):
            groups[find(i)].append(items[i])
        out = []
        for members in groups.values():
            span = sum(m['hi'] - m['lo'] + 1 for m in members)
            out.append({
                'axis': axis,
                'lo': min(m['lo'] for m in members),
                'hi': max(m['hi'] for m in members),
                'pos': sum(m['pos'] * (m['hi'] - m['lo'] + 1) for m in members) / span,
                'dashed': any(m['dashed'] for m in members),
                # A stitched run belongs to both layers, which is what makes it
                # the join between them.
                'layer': ('both' if len({m['layer'] for m in members}) > 1
                          else members[0]['layer']),
            })
        runs[axis] = out
    return joined


def _neighbours_along_runs(points, on_run):
    """For each point, the nearest symbol either way along each of its runs.

    Bounded where the symbol-to-symbol walk is not: a crossing has two runs and
    at most two symbols along each, so this says what the crossing sits between
    rather than everything the pooled network can reach.
    """
    ordered = {run_id: sorted(entries, key=lambda e: e[0])
               for run_id, entries in on_run.items()}
    out = collections.defaultdict(lambda: {'red': [], 'black': []})
    for run_id, entries in ordered.items():
        for i, (_, pt) in enumerate(entries):
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(entries):
                    other = entries[j][1]
                    found = _symbols_at(other)
                    if found:
                        for sym in found:
                            layer = 'red' if str(sym).startswith('red') else 'black'
                            if sym not in out[id(pt)][layer]:
                                out[id(pt)][layer].append(sym)
                        break
                    j += step
    return out


def draw(image, runs, points, symbols, path):
    out = cv2.addWeighted(image, 0.12, np.full_like(image, 255), 0.88, 0)
    for run in runs:
        colour = {'red': (40, 40, 220), 'both': (30, 160, 40)}.get(
            run['layer'], (190, 120, 40))
        if run['axis'] == 'h':
            p1, p2 = (int(run['lo']), int(run['pos'])), (int(run['hi']), int(run['pos']))
        else:
            p1, p2 = (int(run['pos']), int(run['lo'])), (int(run['pos']), int(run['hi']))
        cv2.line(out, p1, p2, colour, 2, cv2.LINE_AA)
    for s in symbols:
        x1, y1, x2, y2 = s['box']
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      (150, 150, 220) if s['layer'] == 'red' else (170, 170, 170), 1)
    for p in points:
        layers = {r['layer'] for r in p['_runs']}
        if len(layers) > 1:                       # the crossing that matters
            cv2.circle(out, (int(p['x']), int(p['y'])), 7, (20, 170, 20), -1,
                       cv2.LINE_AA)
            cv2.circle(out, (int(p['x']), int(p['y'])), 7, (255, 255, 255), 1,
                       cv2.LINE_AA)
        else:
            colour = (60, 60, 230) if 'red' in layers else (150, 150, 150)
            cv2.circle(out, (int(p['x']), int(p['y'])), 3, colour, -1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--black', required=True, help='the black layer output dir')
    p.add_argument('--red', required=True, help='the red layer output dir')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--image', help='background for the overlay; default the '
                                   "black layer's diagram.png")
    p.add_argument('--cross-tolerance', type=float, default=4.0)
    p.add_argument('--corner-reach', type=float, default=20.0)
    p.add_argument('--touch-tolerance', type=float, default=3.0)
    p.add_argument('--endpoint-reach', type=float, default=9.0)
    p.add_argument('--min-run-length', type=float, default=6.0)
    p.add_argument('--merge-tolerance', type=float, default=3.0)
    p.add_argument('--no-stitch-collinear', action='store_true',
                   help='do not rejoin runs that are two halves of one pipe, '
                        'one per layer. Without the stitch the crossing search '
                        'never puts a part-painted pipe back together, since it '
                        'only ever pairs a horizontal with a vertical.')
    p.add_argument('--stitch-axis-tolerance', type=float, default=2.5)
    p.add_argument('--stitch-gap', type=float, default=25.0)
    args = p.parse_args()

    black_runs, black_symbols, width, height = load_layer(args.black, 'black')
    red_runs, red_symbols, rw, rh = load_layer(args.red, 'red')
    if (rw, rh) != (width, height):
        raise SystemExit(f'ERROR: the layers are different sizes, '
                         f'{width}x{height} vs {rw}x{rh}')

    runs = {'h': black_runs['h'] + red_runs['h'],
            'v': black_runs['v'] + red_runs['v']}
    symbols = black_symbols + red_symbols
    print(f"black: {len(black_runs['h']) + len(black_runs['v'])} runs, "
          f'{len(black_symbols)} symbols')
    print(f"red  : {len(red_runs['h']) + len(red_runs['v'])} runs, "
          f'{len(red_symbols)} symbols')
    if not args.no_stitch_collinear:
        before = len(runs['h']) + len(runs['v'])
        stitch_collinear(runs, args.stitch_axis_tolerance, args.stitch_gap)
        after = len(runs['h']) + len(runs['v'])
        spanning = sum(1 for r in runs['h'] + runs['v'] if r['layer'] == 'both')
        print(f'collinear stitch: {before} -> {after} runs, '
              f'{spanning} of them spanning both layers')

    hs, vs, points, on_run = find_points(
        runs, symbols, args.cross_tolerance, args.touch_tolerance,
        args.min_run_length, args.endpoint_reach, args.corner_reach)
    points = _merge_points(points, on_run, args.merge_tolerance)
    by_id = {r['id']: r for r in hs + vs}
    for pt in points:
        pt['_runs'] = [by_id[r] for r in pt['runs'] if r in by_id]
    edges = chain_points(by_id, on_run)

    layer_of = {s['id']: s['layer'] for s in symbols}
    name_of = {s['id']: (s['text'] or s['label'].split('/')[-1] + ' ' + str(s['id']))
               for s in symbols}
    label_of = {s['id']: s['label'] for s in symbols}

    links = symbol_edges(points, edges)
    undirected = {tuple(sorted((a, b))) for a, bs in links.items() for b in bs}
    cross = sorted(pair for pair in undirected
                   if layer_of[pair[0]] != layer_of[pair[1]])
    crossings = [pt for pt in points
                 if len({r['layer'] for r in pt['_runs']}) > 1]

    print(f'\npooled: {len(hs) + len(vs)} runs, {len(points)} points, '
          f'{len(undirected)} symbol links')
    print(f'points where a red run meets a black one: {len(crossings)}')

    # The symbol-to-symbol walk fans out once the layers are pooled -- a red
    # contact far from any black symbol reaches dozens of them before stopping.
    # What is actually new is each crossing, so report those: the nearest
    # symbol either way along each of the two runs that meet there. That is
    # bounded, and it says where the overlay lands on the plant.
    neighbours = _neighbours_along_runs(points, on_run)
    print(f'\ncrossings (red run x black run):')
    for pt in sorted(crossings, key=lambda p: (round(p['y']), round(p['x']))):
        near = neighbours[id(pt)]
        red = ', '.join(name_of[s] for s in near['red']) or '-'
        black = ', '.join(name_of[s] for s in near['black']) or '-'
        print(f"  ({pt['x']:>6.0f},{pt['y']:>6.0f})  red: {red:<26} black: {black}")

    os.makedirs(args.output_dir, exist_ok=True)
    index = {id(pt): i for i, pt in enumerate(points)}
    payload = {
        'image_details': {'width': width, 'height': height},
        'layers': {'black': args.black, 'red': args.red},
        'runs': [{'id': r['id'], 'axis': r['axis'], 'layer': r['layer'],
                  'pos': round(r['pos'], 1), 'lo': round(r['lo'], 1),
                  'hi': round(r['hi'], 1), 'dashed': r['dashed']}
                 for r in hs + vs],
        'points': [{'id': i, 'kind': pt['kind'],
                    'x': round(pt['x'], 1), 'y': round(pt['y'], 1),
                    'runs': pt['runs'], 'symbol': pt['symbol'],
                    'layers': sorted({r['layer'] for r in pt['_runs']}),
                    **({'also': pt['also']} if pt.get('also') else {})}
                   for i, pt in enumerate(points)],
        'edges': [{'a': index[id(e['a'])], 'b': index[id(e['b'])]} for e in edges],
        'symbols': [{'id': s['id'], 'layer': s['layer'], 'label': s['label'],
                     'text': s['text'],
                     'box': [round(v, 1) for v in s['box']]} for s in symbols],
        'symbol_links': sorted([list(pair) for pair in undirected]),
        'cross_layer_links': [list(pair) for pair in cross],
    }
    out_path = os.path.join(args.output_dir, 'merged_graph.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=1)

    image_path = args.image or os.path.join(args.black, 'diagram.png')
    image = cv2.imread(image_path)
    png = os.path.join(args.output_dir, '40_merged_layers.png')
    draw(image, hs + vs, points, symbols, png)
    print(f'\nwrote {out_path} and {png}')


if __name__ == '__main__':
    main()
