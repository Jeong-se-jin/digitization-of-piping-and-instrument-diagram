"""Recolour part of a drawing red, to make a two-layer test case.

``split_red`` pulls a red overlay out of a drawing; this puts one in. Give it a
finished ``intersection_graph.json`` and it can paint what the pipeline already
found -- named symbols, the pipes touching them, a connected subsystem, a
quadrant -- so a round trip can be measured against the graph the same sheet
gave before the paint.

Selecting by name is the point of ``--symbol``: name a tag and the symbol goes
red, and ``--hops`` carries the red outward along the piping from there.

    # one valve and the pipes it sits on
    paint_red.py --graph out/p75-notext-keep/intersection_graph.json \
        --image out/page75/diagram.png \
        --text-detection out/p75-notext-keep/text_detection.json \
        --symbol V147 --hops 1 --out sample.png

    # a whole subsystem, two symbols out from a tag
    ... --symbol V147 --symbol V145 --hops 4 --out sample.png

    # no names: a connected chunk, a scattering, or one quadrant
    ... --mode component --fraction 0.3 --out sample.png

``--hops`` counts steps through the point graph: 1 reaches the runs touching
the chosen symbols, 2 the symbols at the far end of those runs, 3 their runs,
and so on. Odd numbers stop on piping, even numbers on symbols.
"""
import argparse
import collections
import json
import random

import cv2
import numpy as np

RED = (40, 40, 230)          # BGR
INK_THRESHOLD = 200


def load(graph_path, td_path, image_path):
    graph = json.load(open(graph_path))
    td = json.load(open(td_path))
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f'ERROR: cannot read {image_path}')
    return graph, td, image


def resolve_symbols(td, wanted, width, height):
    """Turn --symbol arguments into symbol records.

    A bare word matches the associated tag, ``#12`` an id, and
    ``label:gate valve`` any symbol whose label contains that text.
    """
    symbols = td['text_and_symbols_associated_list']
    by_tag = collections.defaultdict(list)
    for s in symbols:
        if s.get('text_associated'):
            by_tag[str(s['text_associated']).strip().upper()].append(s)

    picked, missing = {}, []
    for want in wanted:
        want = want.strip()
        found = []
        if want.startswith('#'):
            found = [s for s in symbols if str(s['id']) == want[1:]]
        elif want.lower().startswith('label:'):
            needle = want.split(':', 1)[1].strip().lower()
            found = [s for s in symbols if needle in s['label'].lower()]
        else:
            found = by_tag.get(want.upper(), [])
        if not found:
            missing.append(want)
        for s in found:
            picked[s['id']] = s
    if missing:
        print(f'  no symbol matched: {", ".join(missing)}')
    return list(picked.values())


def grow(graph, symbol_ids, hops):
    """Walk out from the chosen symbols, returning (symbol ids, run ids).

    A hop is one symbol deep: at 1 the red reaches the pipes leaving the chosen
    symbols and the symbols on the far end of them, at 2 the pipes past those,
    and so on. At 0 only the named symbols are painted, with no piping.
    """
    points = graph['points']
    adjacency = collections.defaultdict(set)
    for e in graph['edges']:
        adjacency[e['a']].add(e['b'])
        adjacency[e['b']].add(e['a'])

    def symbols_at(p):
        if p['symbol'] is None:
            return []
        return [p['symbol']] + list(p.get('also', []))

    points_of_symbol = collections.defaultdict(list)
    for i, p in enumerate(points):
        for sym in symbols_at(p):
            points_of_symbol[sym].append(i)

    symbols = set(symbol_ids)
    runs = set()
    if hops <= 0:
        return symbols, runs

    start = [i for s in symbols for i in points_of_symbol[s]]
    depth = {i: 0 for i in start}
    queue = collections.deque(start)
    while queue:
        node = queue.popleft()
        d = depth[node]
        runs.update(points[node]['runs'])
        for sym in symbols_at(points[node]):
            symbols.add(sym)
        for nxt in adjacency[node]:
            # Crossing a symbol is what costs a hop; running along piping is
            # free, so one hop reaches a whole pipe rather than one segment of
            # it.
            step = 1 if symbols_at(points[nxt]) else 0
            if d + step > hops:
                continue
            if nxt in depth and depth[nxt] <= d + step:
                continue
            depth[nxt] = d + step
            queue.append(nxt)
    return symbols, runs


def pick_runs_without_names(graph, mode, fraction, seed):
    runs = graph['runs']
    rng = random.Random(seed)
    if mode == 'scatter':
        return {r['id'] for r in rng.sample(runs, int(len(runs) * fraction))}
    if mode == 'region':
        xs = [r['pos'] if r['axis'] == 'v' else (r['lo'] + r['hi']) / 2 for r in runs]
        ys = [r['pos'] if r['axis'] == 'h' else (r['lo'] + r['hi']) / 2 for r in runs]
        cx, cy = np.median(xs), np.median(ys)
        return {r['id'] for r, x, y in zip(runs, xs, ys) if x > cx and y > cy}

    adjacency = collections.defaultdict(set)
    for p in graph['points']:
        for i in p['runs']:
            for j in p['runs']:
                if i != j:
                    adjacency[i].add(j)
    # The longest run of all is the sheet border, which crosses nothing.
    seed_run = max((r for r in runs if adjacency[r['id']]),
                   key=lambda r: r['hi'] - r['lo'])['id']
    chosen, queue = {seed_run}, [seed_run]
    target = int(len(runs) * fraction)
    while queue and len(chosen) < target:
        for v in sorted(adjacency[queue.pop(0)]):
            if v not in chosen:
                chosen.add(v)
                queue.append(v)
                if len(chosen) >= target:
                    break
    return chosen


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph', required=True, help='an intersection_graph.json')
    p.add_argument('--image', required=True)
    p.add_argument('--text-detection', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--symbol', action='append', default=[],
                   help='a tag (V147), an id (#12) or label:<text>; repeatable')
    p.add_argument('--hops', type=int, default=1,
                   help='steps outward through the point graph from the chosen '
                        'symbols; 0 paints the symbols alone')
    p.add_argument('--mode', choices=('component', 'scatter', 'region'),
                   help='pick runs without naming anything, when --symbol is '
                        'not given')
    p.add_argument('--fraction', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--halfwidth', type=int, default=3,
                   help='pixels either side of a run to recolour')
    p.add_argument('--paint-symbols', action='store_true', default=True,
                   help='recolour the chosen symbols themselves (default on)')
    p.add_argument('--no-paint-symbols', dest='paint_symbols',
                   action='store_false')
    p.add_argument('--protect-text', action='store_true', default=True,
                   help='leave text boxes black, so OCR sees what it saw '
                        'before and a comparison stays about the graph')
    p.add_argument('--no-protect-text', dest='protect_text', action='store_false')
    args = p.parse_args()

    if not args.symbol and not args.mode:
        raise SystemExit('ERROR: give --symbol or --mode')

    graph, td, image = load(args.graph, args.text_detection, args.image)
    height, width = image.shape[:2]
    runs_by_id = {r['id']: r for r in graph['runs']}

    symbol_ids, run_ids = set(), set()
    if args.symbol:
        chosen = resolve_symbols(td, args.symbol, width, height)
        if not chosen:
            raise SystemExit('ERROR: nothing matched --symbol')
        print(f'  matched {len(chosen)} symbols: '
              + ', '.join(str(s.get('text_associated') or f"#{s['id']}")
                          for s in chosen[:12])
              + (' ...' if len(chosen) > 12 else ''))
        symbol_ids, run_ids = grow(graph, [s['id'] for s in chosen], args.hops)
    if args.mode:
        run_ids |= pick_runs_without_names(graph, args.mode, args.fraction,
                                           args.seed)

    protect = np.zeros((height, width), bool)
    if args.protect_text:
        for t in (td.get('all_text_list') or []):
            x1, y1 = int(t['topX'] * width), int(t['topY'] * height)
            x2, y2 = int(t['bottomX'] * width), int(t['bottomY'] * height)
            protect[max(0, y1 - 2):y2 + 3, max(0, x1 - 2):x2 + 3] = True

    mask = np.zeros((height, width), bool)
    hw = args.halfwidth
    for rid in run_ids:
        r = runs_by_id.get(rid)
        if r is None:
            continue
        if r['axis'] == 'h':
            y = int(round(r['pos']))
            mask[max(0, y - hw):y + hw + 1, int(r['lo']):int(r['hi']) + 1] = True
        else:
            x = int(round(r['pos']))
            mask[int(r['lo']):int(r['hi']) + 1, max(0, x - hw):x + hw + 1] = True

    if args.paint_symbols and symbol_ids:
        for s in td['text_and_symbols_associated_list']:
            if s['id'] not in symbol_ids:
                continue
            x1, y1 = int(s['topX'] * width), int(s['topY'] * height)
            x2, y2 = int(s['bottomX'] * width), int(s['bottomY'] * height)
            mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
            # A symbol the user named is the point of the exercise, so its own
            # box wins over the text guard.
            protect[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = False

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = grey < INK_THRESHOLD
    paint = mask & ink & ~protect
    out = image.copy()
    out[paint] = RED
    cv2.imwrite(args.out, out)
    print(f'  painted {len(symbol_ids)} symbols and {len(run_ids)} runs: '
          f'{int(paint.sum())} px ({paint.sum() / max(ink.sum(), 1) * 100:.1f}% '
          f'of the ink)')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
