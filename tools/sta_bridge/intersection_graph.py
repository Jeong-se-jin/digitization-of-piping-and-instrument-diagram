"""Build connectivity from where axis-aligned runs cross, ignoring endpoints.

This is a second, independent route to the graph. It does not touch the
candidate matching in ``app.services.graph_construction`` -- run it after a
normal ``run_local`` pass and it reads that pass's output.

The existing route gives every line segment's two endpoints one candidate each,
so a pipe that crosses another pipe, or that runs through a valve on its way
somewhere, has nowhere to record the crossing. That is what leaves three
quarters of the segments in no connection at all.

Here the geometry decides instead:

1. **Keep only horizontal and vertical.** P&ID piping is on the axes.
2. **Stitch the fragments back into runs.** Collinear pieces of one pipe, split
   by dashes, by text masking or by ink breaks, become a single run. This is
   safe here in a way it is not for the existing route: a run is allowed to
   pass straight through a symbol, because step 4 records the symbol
   separately rather than relying on the pipe ending at it.
3. **Cross every horizontal run with every vertical one.** Where they meet is a
   junction point.
4. **Clip every run against every symbol box.** Where a run enters or leaves a
   symbol is a contact point.
5. **Chain the points along each run.** Consecutive points on the same run are
   connected, which turns the points into a graph.

Symbol-to-symbol edges then fall out of walking that graph from each contact
point until the next one.

    .venv-pid/bin/python -m tools.sta_bridge.intersection_graph \
        --output-dir out/p75-axis
"""
import argparse
import collections
import json
import math
import os

import cv2
import numpy as np


ARROW_LABEL = 'Piping/Fittings/Mid arrow flow direction'


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _orientation(seg, width, height, tolerance_degrees):
    """'h', 'v' or None, measured in pixels rather than normalised units."""
    dx = (seg['endX'] - seg['startX']) * width
    dy = (seg['endY'] - seg['startY']) * height
    if dx == 0 and dy == 0:
        return None
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    if angle <= tolerance_degrees:
        return 'h'
    if angle >= 90 - tolerance_degrees:
        return 'v'
    return None


def build_runs_stable(segments, width, height, tolerance_degrees,
                      axis_tol, gap_tol):
    """Stitch collinear fragments into runs.

    Fragments join when they sit within *axis_tol* of the same axis position
    and their extents are no more than *gap_tol* apart. Returns dicts keyed by
    ``pos`` (the row or column the run sits on) and ``lo``/``hi`` (its extent
    along that axis)."""
    buckets = {'h': collections.defaultdict(list), 'v': collections.defaultdict(list)}
    parsed = {'h': [], 'v': []}
    for seg in segments:
        axis = _orientation(seg, width, height, tolerance_degrees)
        if axis is None:
            continue
        x1, y1 = seg['startX'] * width, seg['startY'] * height
        x2, y2 = seg['endX'] * width, seg['endY'] * height
        if axis == 'h':
            pos, lo, hi = (y1 + y2) / 2, min(x1, x2), max(x1, x2)
        else:
            pos, lo, hi = (x1 + x2) / 2, min(y1, y2), max(y1, y2)
        idx = len(parsed[axis])
        parsed[axis].append({'pos': pos, 'lo': lo, 'hi': hi,
                             'dashed': seg.get('line_type') == 'dashed',
                             'boxed': bool(seg.get('inside_box'))})
        # Bucket by axis position so only nearby fragments are compared.
        key = int(pos // max(axis_tol, 1))
        for k in (key - 1, key, key + 1):
            buckets[axis][k].append(idx)

    runs = {}
    for axis, items in parsed.items():
        parent = list(range(len(items)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _, idxs in buckets[axis].items():
            idxs = sorted(set(idxs), key=lambda i: items[i]['lo'])
            for a in range(len(idxs)):
                ia = items[idxs[a]]
                for b in range(a + 1, len(idxs)):
                    ib = items[idxs[b]]
                    if ib['lo'] - ia['hi'] > gap_tol:
                        break                      # sorted: nothing nearer left
                    if abs(ia['pos'] - ib['pos']) > axis_tol:
                        continue
                    ra, rb = find(idxs[a]), find(idxs[b])
                    if ra != rb:
                        parent[rb] = ra

        merged = collections.defaultdict(list)
        for i in range(len(items)):
            merged[find(i)].append(i)
        out = []
        for members in merged.values():
            members.sort(key=lambda i: items[i]['lo'])
            # A piece that lay entirely inside a text or symbol box is kept only
            # where it bridges: real ink on both sides of it along this run.
            # That is the pipe running under a label -- line, text, line -- and
            # dropping it breaks one pipe into two. A boxed piece hanging off
            # either end is the label's own strokes and is trimmed away.
            first = next((n for n, i in enumerate(members)
                          if not items[i]['boxed']), None)
            if first is None:
                continue                          # nothing but box contents
            last = max(n for n, i in enumerate(members)
                       if not items[i]['boxed'])
            members = members[first:last + 1]
            lo = min(items[i]['lo'] for i in members)
            hi = max(items[i]['hi'] for i in members)
            span = sum(items[i]['hi'] - items[i]['lo'] + 1 for i in members)
            pos = sum(items[i]['pos'] * (items[i]['hi'] - items[i]['lo'] + 1)
                      for i in members) / span
            out.append({'axis': axis, 'pos': pos, 'lo': lo, 'hi': hi,
                        'parts': len(members),
                        'ink': span,
                        'bridged': sum(1 for i in members if items[i]['boxed']),
                        # By ink, not by any(): one letter stroke classified
                        # dashed used to paint a whole solid header purple.
                        'dashed': sum(items[i]['hi'] - items[i]['lo'] + 1
                                      for i in members if items[i]['dashed']
                                      ) * 2 > span,
                        'boxed': False})
        runs[axis] = out
    return runs


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def drop_symbol_outlines(runs, symbols, slack):
    """Throw away the runs that are a symbol's own drawn rectangle.

    A box symbol is four strokes on the axes, so line detection finds them and
    stitches them into runs like any other. They are not piping. They survive
    the ``inside_box`` filter because that wants both endpoints strictly inside
    the box and a border stroke sits on the boundary, and they survive masking
    because the mask is inset a couple of pixels.

    A run whose whole extent fits within one symbol box cannot be piping
    passing through it -- it is shorter than the symbol.
    """
    boxes = [s['box'] for s in symbols]
    dropped = 0
    for axis in ('h', 'v'):
        kept = []
        for run in runs[axis]:
            inside = False
            for x1, y1, x2, y2 in boxes:
                if run['axis'] == 'h':
                    along, across = (x1, x2), (y1, y2)
                else:
                    along, across = (y1, y2), (x1, x2)
                if (across[0] - slack <= run['pos'] <= across[1] + slack and
                        along[0] - slack <= run['lo'] and
                        run['hi'] <= along[1] + slack):
                    inside = True
                    break
            if inside:
                dropped += 1
            else:
                kept.append(run)
        runs[axis] = kept
    return dropped


def _merge_points(points, on_run, tolerance):
    """Fold points that landed on the same spot into one.

    A junction that falls exactly where a run meets a symbol, and two
    overlapping symbol detections clipped by the same run, both produce points
    a pixel or two apart that mean one place. The entry and exit of a single
    symbol on a single run are deliberately left alone -- that pair is what
    lets a pipe pass through a valve.
    """
    if tolerance <= 0:
        return points

    def pair_key(p):
        return (p['runs'][0], p['symbol']) if p['kind'] == 'symbol' else None

    cells = collections.defaultdict(list)
    for p in points:
        cells[(int(p['x'] // max(tolerance, 1)), int(p['y'] // max(tolerance, 1)))].append(p)

    merged_into = {}
    for (cx, cy), _ in list(cells.items()):
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near.extend(cells.get((cx + dx, cy + dy), []))
        for a in cells[(cx, cy)]:
            if id(a) in merged_into:
                continue
            for b in near:
                if b is a or id(b) in merged_into:
                    continue
                if math.dist((a['x'], a['y']), (b['x'], b['y'])) > tolerance:
                    continue
                if pair_key(a) is not None and pair_key(a) == pair_key(b):
                    continue                  # the entry/exit of one contact
                merged_into[id(b)] = a
                a['runs'] = sorted(set(a['runs']) | set(b['runs']))
                if a['symbol'] is None:
                    a['symbol'] = b['symbol']
                    a['kind'] = b['kind']
                elif b['symbol'] is not None and b['symbol'] != a['symbol']:
                    a.setdefault('also', []).append(b['symbol'])

    if not merged_into:
        return points
    for run_id, entries in on_run.items():
        on_run[run_id] = [(pos, merged_into.get(id(p), p)) for pos, p in entries]
    return [p for p in points if id(p) not in merged_into]


def find_points(runs, symbols, cross_tol, touch_tol, min_run_length,
                endpoint_reach=0.0, corner_reach=0.0):
    """Junctions between runs, and contacts between runs and symbol boxes."""
    hs = [r for r in runs['h'] if r['hi'] - r['lo'] >= min_run_length]
    vs = [r for r in runs['v'] if r['hi'] - r['lo'] >= min_run_length]
    for i, r in enumerate(hs):
        r['id'] = f'h{i}'
    for i, r in enumerate(vs):
        r['id'] = f'v{i}'

    points = []
    on_run = collections.defaultdict(list)      # run id -> [(position, point)]

    for h in hs:
        for v in vs:
            # The vertical must reach the horizontal's row and vice versa.
            # Where *both* have to stretch, the two are meeting end to end: a
            # corner. Corners get the longer reach, because a pipe turning a
            # corner leaves the two runs a dozen pixels short of each other --
            # the ink there is a staircase that the detector breaks into short
            # slanted pieces the axis filter then drops. Where only one has to
            # stretch it is a T, its partner's body is already under it, and
            # the strict tolerance stays.
            turns_h = not (h['lo'] <= v['pos'] <= h['hi'])
            turns_v = not (v['lo'] <= h['pos'] <= v['hi'])
            tol = corner_reach if (turns_h and turns_v) else cross_tol
            if not (h['lo'] - tol <= v['pos'] <= h['hi'] + tol):
                continue
            if not (v['lo'] - tol <= h['pos'] <= v['hi'] + tol):
                continue
            point = {'kind': 'junction', 'x': v['pos'], 'y': h['pos'],
                     'runs': [h['id'], v['id']], 'symbol': None}
            points.append(point)
            on_run[h['id']].append((v['pos'], point))
            on_run[v['id']].append((h['pos'], point))

    for run in hs + vs:
        horizontal = run['axis'] == 'h'
        for symbol in symbols:
            x1, y1, x2, y2 = symbol['box']
            # A run reaches a little past its own ends when looking for a
            # symbol. Dashed leaders are drawn a few pixels clear of the box
            # they serve -- the signal line into ZA109 stops 7px short of it --
            # and without the reach that gap costs the connection. The reach
            # applies along the run only; sideways it still has to line up.
            if horizontal:
                if not (y1 - touch_tol <= run['pos'] <= y2 + touch_tol):
                    continue
                lo = max(run['lo'] - endpoint_reach, x1 - touch_tol)
                hi = min(run['hi'] + endpoint_reach, x2 + touch_tol)
            else:
                if not (x1 - touch_tol <= run['pos'] <= x2 + touch_tol):
                    continue
                lo = max(run['lo'] - endpoint_reach, y1 - touch_tol)
                hi = min(run['hi'] + endpoint_reach, y2 + touch_tol)
            if lo > hi:
                continue
            # Two points -- an entry and an exit -- only when the run really
            # crosses the box, so a pipe can pass through the valve on it. A
            # run that merely grazes a corner, or that stops at the box, gets
            # one point: its overlap is a couple of pixels wide and two points
            # there are the same place twice.
            box_lo, box_hi = (x1, x2) if horizontal else (y1, y2)
            through = (run['lo'] < box_lo + touch_tol and
                       run['hi'] > box_hi - touch_tol)
            lo, hi = max(lo, run['lo'] - endpoint_reach), min(hi, run['hi'] + endpoint_reach)
            for pos in ({lo, hi} if through else {(lo + hi) / 2}):
                point = {'kind': 'symbol',
                         'x': pos if horizontal else run['pos'],
                         'y': run['pos'] if horizontal else pos,
                         'runs': [run['id']], 'symbol': symbol['id']}
                points.append(point)
                on_run[run['id']].append((pos, point))

    return hs, vs, points, on_run


def prune_dangling(runs, symbols, cross_tol, touch_tol, min_run_length,
                   endpoint_reach, merge_tol, corner_reach=0.0):
    """Drop the runs that lead nowhere, until none are left.

    With the text left in the image so a pipe under a label stays whole, the
    letters produce runs of their own -- the bar of an F, the stem of a T. A
    letter stroke crosses nothing and touches no symbol, so it collects fewer
    than two connection points, and that is what tells it apart from piping.

    Removing a run can strand its neighbour, so this repeats until the count
    settles.
    """
    total = 0
    while True:
        hs, vs, points, on_run = find_points(
            runs, symbols, cross_tol, touch_tol, min_run_length, endpoint_reach,
            corner_reach)
        _merge_points(points, on_run, merge_tol)
        supported = {run_id for run_id, entries in on_run.items()
                     if len(entries) >= 2}
        before = len(runs['h']) + len(runs['v'])
        for axis in ('h', 'v'):
            runs[axis] = [r for r in runs[axis]
                          if r.get('id') is None or r['id'] in supported]
        after = len(runs['h']) + len(runs['v'])
        # find_points only ids the runs it kept, so anything it filtered out for
        # length is gone from `supported` too and drops here on the first pass.
        removed = before - after
        total += removed
        if removed == 0:
            return total
        for axis in ('h', 'v'):
            for r in runs[axis]:
                r.pop('id', None)


def chain_points(runs_by_id, on_run):
    """Connect consecutive points along each run."""
    edges = []
    for run_id, entries in on_run.items():
        entries.sort(key=lambda e: e[0])
        for (pa, a), (pb, b) in zip(entries, entries[1:]):
            if b is a:
                continue
            edges.append({'a': a, 'b': b, 'run': run_id,
                          'length': abs(pb - pa)})
    return edges


def _symbols_at(point):
    """Every symbol a point stands for -- more than one where overlapping
    detections were clipped by the same run and then merged."""
    if point['symbol'] is None:
        return ()
    return (point['symbol'], *point.get('also', ()))


def symbol_edges(points, edges):
    """Walk the point graph from each symbol contact to the next one."""
    index = {id(p): i for i, p in enumerate(points)}
    adjacency = collections.defaultdict(list)
    for e in edges:
        adjacency[index[id(e['a'])]].append(index[id(e['b'])])
        adjacency[index[id(e['b'])]].append(index[id(e['a'])])

    found = collections.defaultdict(set)
    starts = [i for i, p in enumerate(points) if p['kind'] == 'symbol']
    for start in starts:
        here = set(_symbols_at(points[start]))
        # Symbols sharing one merged point are on the same spot, so they are
        # connected to each other whatever the walk finds.
        for a in here:
            found[a] |= here - {a}
        seen = {start}
        queue = collections.deque(adjacency[start])
        seen.update(adjacency[start])
        while queue:
            node = queue.popleft()
            point = points[node]
            there = _symbols_at(point)
            if there:
                for a in here:
                    found[a] |= set(there) - {a}
                continue                          # stop at the next symbol
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return found


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def draw(image, hs, vs, points, symbols, path):
    out = cv2.addWeighted(image, 0.12,
                          np.full_like(image, 255), 0.88, 0)
    for run in hs + vs:
        colour = (200, 60, 200) if run['dashed'] else (190, 120, 40)
        if run['axis'] == 'h':
            p1, p2 = (int(run['lo']), int(run['pos'])), (int(run['hi']), int(run['pos']))
        else:
            p1, p2 = (int(run['pos']), int(run['lo'])), (int(run['pos']), int(run['hi']))
        cv2.line(out, p1, p2, colour, 2, cv2.LINE_AA)
    for s in symbols:
        x1, y1, x2, y2 = s['box']
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      (170, 170, 170), 1)
    for p in points:
        if p['kind'] == 'junction':
            cv2.circle(out, (int(p['x']), int(p['y'])), 4, (30, 120, 30), -1,
                       cv2.LINE_AA)
        else:
            cv2.circle(out, (int(p['x']), int(p['y'])), 5, (40, 40, 220), -1,
                       cv2.LINE_AA)
    cv2.imwrite(path, out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True,
                   help='a directory a run_local pass has already written')
    p.add_argument('--axis-tolerance', type=float, default=5.0,
                   help='degrees off horizontal/vertical still counted as aligned')
    p.add_argument('--run-axis-tolerance', type=float, default=2.5,
                   help='pixels two fragments may differ on their shared axis '
                        'and still belong to one run')
    p.add_argument('--run-gap', type=float, default=25.0,
                   help='pixels of gap two fragments may leave between them')
    p.add_argument('--min-run-length', type=float, default=6.0,
                   help='runs shorter than this are dropped before crossing')
    p.add_argument('--cross-tolerance', type=float, default=4.0,
                   help='pixels a run may fall short of another and still cross it')
    p.add_argument('--corner-reach', type=float, default=20.0,
                   help='pixels two runs may fall short of each other when they '
                        'meet end to end, at a corner. Larger than '
                        '--cross-tolerance on purpose: a T-junction has one '
                        "run's body already under the other, but a corner has "
                        'nothing but the raggedness the axis filter dropped')
    p.add_argument('--touch-tolerance', type=float, default=3.0,
                   help='pixels of slack when clipping a run against a symbol box')
    p.add_argument('--keep-flow-arrows', action='store_true',
                   help='keep the flow-direction arrows among the symbols. They '
                        'are excluded by default: an arrow is a marker drawn on '
                        'a pipe, not a fitting in it, so every run it sits on '
                        'gains a contact point and stops there, splitting one '
                        'pipe into two links through something that connects '
                        'nothing.')
    p.add_argument('--exclude-label', action='append', default=[],
                   help='drop symbols whose label contains this; repeatable')
    p.add_argument('--prune-dangling', action='store_true',
                   help='drop every run that ends up with fewer than two '
                        'connection points, repeatedly until none are left. A '
                        'run that crosses nothing and touches no symbol leads '
                        'nowhere. Needed with run_local --no-text-mask, where '
                        'the letters left in the image become runs of their own.')
    p.add_argument('--merge-tolerance', type=float, default=3.0,
                   help='pixels within which two points are the same place and '
                        'are folded together; 0 disables')
    p.add_argument('--endpoint-reach', type=float, default=9.0,
                   help='pixels a run may reach past its own end to meet a '
                        'symbol box. Dashed leaders stop a few pixels clear of '
                        'the box they serve; 0 disables')
    p.add_argument('--keep-outline-runs', action='store_true',
                   help="keep the runs that are a symbol's own drawn rectangle. "
                        'They are dropped by default -- a box symbol is four '
                        'axis-aligned strokes, and they are not piping.')
    p.add_argument('--no-bridge-boxed', action='store_true',
                   help='drop every segment lying entirely inside a symbol or '
                        'text box, instead of keeping the ones that bridge. By '
                        'default such a segment survives only where the run has '
                        'real ink on both sides of it -- a pipe passing under a '
                        'label -- and is trimmed wherever it hangs off an end, '
                        "which is the label's own strokes.")
    args = p.parse_args()

    d = args.output_dir
    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)
    width = td['image_details']['width']
    height = td['image_details']['height']
    symbols = [{'id': s['id'], 'label': s['label'],
                'text': s.get('text_associated'),
                'box': [s['topX'] * width, s['topY'] * height,
                        s['bottomX'] * width, s['bottomY'] * height]}
               for s in td['text_and_symbols_associated_list']]
    excluded = list(args.exclude_label)
    if not args.keep_flow_arrows:
        excluded.append(ARROW_LABEL)
    if excluded:
        before = len(symbols)
        symbols = [s for s in symbols
                   if not any(e in s['label'] for e in excluded)]
        print(f'excluded {before - len(symbols)} symbols by label '
              f"({', '.join(e.split('/')[-1] for e in excluded)})")

    with open(os.path.join(d, 'line_detection.json')) as f:
        segments = json.load(f)['line_segments']
    n_boxed = sum(1 for s in segments if s.get('inside_box'))
    if args.no_bridge_boxed:
        segments = [s for s in segments if not s.get('inside_box')]
        print(f'dropped all {n_boxed} segments lying entirely inside a box')
    else:
        print(f'{n_boxed} segments lie entirely inside a symbol or text box; '
              f'keeping the ones that bridge real ink on both sides')

    runs = build_runs_stable(segments, width, height, args.axis_tolerance,
                             args.run_axis_tolerance, args.run_gap)
    if not args.keep_outline_runs:
        n_outline = drop_symbol_outlines(runs, symbols, args.touch_tolerance)
        print(f"dropped {n_outline} runs that were a symbol's own outline")
    n_bridged = sum(r.get('bridged', 0) for r in runs['h'] + runs['v'])
    if n_bridged:
        print(f'kept {n_bridged} boxed segments that bridge a run')
    n_runs = len(runs['h']) + len(runs['v'])
    print(f"segments {len(segments)} -> runs {n_runs} "
          f"({len(runs['h'])} horizontal, {len(runs['v'])} vertical)")

    if args.prune_dangling:
        n_pruned = prune_dangling(runs, symbols, args.cross_tolerance,
                                  args.touch_tolerance, args.min_run_length,
                                  args.endpoint_reach, args.merge_tolerance,
                                  args.corner_reach)
        print(f'pruned {n_pruned} runs that led nowhere -> '
              f"{len(runs['h']) + len(runs['v'])} left")

    hs, vs, points, on_run = find_points(
        runs, symbols, args.cross_tolerance, args.touch_tolerance,
        args.min_run_length, args.endpoint_reach, args.corner_reach)
    n_before = len(points)
    points = _merge_points(points, on_run, args.merge_tolerance)
    if n_before != len(points):
        print(f'coincident points merged: {n_before} -> {len(points)}')
    runs_by_id = {r['id']: r for r in hs + vs}
    edges = chain_points(runs_by_id, on_run)

    n_junction = sum(1 for p in points if p['kind'] == 'junction')
    n_contact = len(points) - n_junction
    touched = {s for p in points for s in _symbols_at(p)}
    print(f'runs kept (>= {args.min_run_length:.0f}px): {len(hs) + len(vs)}')
    print(f'connection points: {len(points)} '
          f'({n_junction} line-to-line, {n_contact} line-to-symbol)')
    print(f'symbols touched by a run: {len(touched)}/{len(symbols)}')
    print(f'point-to-point edges along runs: {len(edges)}')

    links = symbol_edges(points, edges)
    undirected = {tuple(sorted((a, b))) for a, bs in links.items() for b in bs}
    connected = {a for a, b in undirected} | {b for a, b in undirected}
    print(f'symbol-to-symbol edges: {len(undirected)}, '
          f'symbols connected: {len(connected)}/{len(symbols)}')

    index = {id(p): i for i, p in enumerate(points)}
    payload = {
        'image_details': {'width': width, 'height': height},
        'runs': [{'id': r['id'], 'axis': r['axis'], 'pos': r['pos'],
                  'lo': r['lo'], 'hi': r['hi'], 'parts': r['parts'],
                  'dashed': r['dashed']} for r in hs + vs],
        'points': [{'id': i, 'kind': p['kind'], 'x': p['x'], 'y': p['y'],
                    'runs': p['runs'], 'symbol': p['symbol'],
                    **({'also': p['also']} if p.get('also') else {})}
                   for i, p in enumerate(points)],
        'edges': [{'a': index[id(e['a'])], 'b': index[id(e['b'])],
                   'run': e['run'], 'length': e['length']} for e in edges],
        'symbol_links': sorted([list(pair) for pair in undirected]),
    }
    out_path = os.path.join(d, 'intersection_graph.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=1)

    image = cv2.imread(os.path.join(d, 'diagram.png'))
    png = os.path.join(d, '30_connection_points.png')
    draw(image, hs, vs, points, symbols, png)
    print(f'wrote {out_path} and {png}')


if __name__ == '__main__':
    main()
