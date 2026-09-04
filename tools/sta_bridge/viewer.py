"""Build a standalone interactive viewer from a graph-construction run.

The three PNGs the pipeline already writes are static: `13_graph.png` is a
networkx layout with no relation to the drawing, and `14_graph_connections.png`
draws all assets at once, so labels and colours collide. This renders the same
data as one HTML file where a single asset can be isolated -- click it and only
its connections and their pipe paths are drawn over the diagram.

Everything is inlined (diagram as a data URI, graph as JSON), so the output is
one file that opens anywhere with no server.

Usage:
    python -m tools.sta_bridge.viewer --output-dir out/18
"""
import argparse
import base64
import json
import os
import struct

CATEGORIES = [
    ('Equipment/', 'equipment', 'Equipment'),
    ('Instrument/Valve/', 'valve', 'Valve'),
    ('Piping/Endpoint/Pagination', 'pagination', 'Off-page'),
    ('Piping/', 'fitting', 'Fitting'),
    ('Instrument/', 'instrument', 'Instrument'),
]


def categorize(label):
    for prefix, key, name in CATEGORIES:
        if label.startswith(prefix):
            return key, name
    return 'other', 'Other'


def build_payload(graph, width, height):
    """Flatten the adjacency list, rounding coordinates to keep the file small."""
    r = lambda v: round(float(v), 4)  # noqa: E731
    box = lambda b: [r(b['topX']), r(b['topY']), r(b['bottomX']), r(b['bottomY'])]  # noqa: E731

    assets = []
    for a in graph:
        key, name = categorize(a['label'])
        assets.append({
            'id': a['id'],
            'tag': a['text_associated'],
            'label': a['label'],
            'leaf': a['label'].rsplit('/', 1)[-1],
            'cat': key,
            'catName': name,
            'box': box(a['bounding_box']),
            'conns': [
                {
                    'id': c['id'],
                    'tag': c['text_associated'],
                    'leaf': c['label'].rsplit('/', 1)[-1],
                    'cat': categorize(c['label'])[0],
                    'dir': c['flow_direction'],
                    'segs': [box(s) for s in c['segments']],
                }
                for c in a['connections']
            ],
        })
    assets.sort(key=lambda a: (a['cat'], a['tag']))
    return {'width': width, 'height': height, 'assets': assets}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True,
                        help='directory holding graph_connectivity.json and diagram.png')
    parser.add_argument('--graph', help='default: <output-dir>/graph_connectivity.json')
    parser.add_argument('--image', help='default: <output-dir>/diagram.png')
    parser.add_argument('--lines', help='line detection output; '
                                        'default: <output-dir>/line_detection.json')
    parser.add_argument('--preprocessed',
                        help='the masked/binarized image Hough actually ran on; '
                             'default: <output-dir>/10_preprocessed.png')
    parser.add_argument('--out', help='default: <output-dir>/graph_viewer.html')
    parser.add_argument('--title', default='P&ID Connectivity')
    args = parser.parse_args()

    graph_path = args.graph or os.path.join(args.output_dir, 'graph_connectivity.json')
    image_path = args.image or os.path.join(args.output_dir, 'diagram.png')
    lines_path = args.lines or os.path.join(args.output_dir, 'line_detection.json')
    pre_path = args.preprocessed or os.path.join(args.output_dir, '10_preprocessed.png')
    out_path = args.out or os.path.join(args.output_dir, 'graph_viewer.html')

    with open(graph_path) as f:
        graph = json.load(f)

    with open(image_path, 'rb') as f:
        raw = f.read()
    # PNG IHDR: width and height are big-endian uint32 at bytes 16..24
    width, height = struct.unpack('>II', raw[16:24])
    data_uri = 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')

    payload = build_payload(graph, width, height)

    # Every raw Hough segment, so a missed connection can be traced back to
    # whether the line was found at all.
    r = lambda v: round(float(v), 4)  # noqa: E731
    if os.path.exists(lines_path):
        with open(lines_path) as f:
            ld = json.load(f)
        payload['lines'] = [[r(s['startX']), r(s['startY']), r(s['endX']), r(s['endY'])]
                            for s in ld['line_segments']]
        # Label a pipe picked up by leftover-text association, by line index.
        payload['lineText'] = {str(i): s['text_associated']
                               for i, s in enumerate(ld['line_segments'])
                               if s.get('text_associated')}
    else:
        payload['lines'] = []
        payload['lineText'] = {}

    # Text that was attached to a line or symbol, so the page can show the label
    # on its target and light the pair from either side.
    payload['texts'] = []
    leftover_path = os.path.join(args.output_dir, 'leftover_text.json')
    if os.path.exists(leftover_path):
        with open(leftover_path) as f:
            for a in json.load(f):
                kind, _, idx = a['target'].partition(' ')
                payload['texts'].append({
                    't': a['text'],
                    'box': [r(v) for v in a['text_box']],
                    'kind': kind,
                    'idx': int(idx),
                })

    # The binarized, symbol/text-masked image Hough ran on. When a pipe is
    # missing from the segments, this usually shows why.
    pre_uri = ''
    if os.path.exists(pre_path):
        with open(pre_path, 'rb') as f:
            pre_uri = 'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')

    template_path = os.path.join(os.path.dirname(__file__), 'viewer_template.html')
    with open(template_path) as f:
        html = f.read()

    html = (html
            .replace('__TITLE__', args.title)
            .replace('__IMAGE_SRC__', data_uri)
            .replace('__PRE_SRC__', pre_uri)
            .replace('__DATA__', json.dumps(payload, separators=(',', ':'))))

    with open(out_path, 'w') as f:
        f.write(html)

    n_edges = sum(len(a['conns']) for a in payload['assets'])
    print(f'assets {len(payload["assets"])}, connections {n_edges} '
          f'({n_edges // 2} undirected), line segments {len(payload["lines"])}, '
          f'attached labels {len(payload["texts"])}')
    if not pre_uri:
        print(f'note: no preprocessed image at {pre_path} '
              f'(run run_local.py with DEBUG=true to get one)')
    print(f'wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
