"""Single-file viewer for the crossing-based graph.

Reads ``intersection_graph.json`` and ``text_detection.json`` from an output
directory and writes ``intersection_viewer.html`` beside them, with the drawing
embedded so the page travels on its own.

    .venv-pid/bin/python -m tools.sta_bridge.intersection_viewer \
        --output-dir out/p75-axis --title "Fire Protection P&ID"
"""
import argparse
import base64
import json
import os
import struct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--title', default='Connection points')
    p.add_argument('--out', help='defaults to intersection_viewer.html in the dir')
    args = p.parse_args()

    d = args.output_dir
    with open(os.path.join(d, 'intersection_graph.json')) as f:
        payload = json.load(f)
    with open(os.path.join(d, 'text_detection.json')) as f:
        td = json.load(f)

    width = payload['image_details']['width']
    height = payload['image_details']['height']
    payload['symbols'] = [
        {'id': s['id'], 'label': s['label'], 'text': s.get('text_associated'),
         'box': [round(s['topX'] * width, 1), round(s['topY'] * height, 1),
                 round(s['bottomX'] * width, 1), round(s['bottomY'] * height, 1)]}
        for s in td['text_and_symbols_associated_list']]

    # Trim the payload: the page never reads a coordinate past one decimal.
    for r in payload['runs']:
        for k in ('pos', 'lo', 'hi'):
            r[k] = round(r[k], 1)
        r.pop('parts', None)
    for pt in payload['points']:
        pt['x'] = round(pt['x'], 1)
        pt['y'] = round(pt['y'], 1)
    for e in payload['edges']:
        e.pop('length', None)
        e.pop('run', None)

    image_path = os.path.join(d, 'diagram.png')
    with open(image_path, 'rb') as f:
        raw = f.read()
    struct.unpack('>II', raw[16:24])          # sanity-check the PNG header
    data_uri = 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')

    template = os.path.join(os.path.dirname(__file__),
                            'intersection_viewer_template.html')
    with open(template) as f:
        html = f.read()
    html = (html
            .replace('__TITLE__', args.title)
            .replace('__IMAGE_SRC__', data_uri)
            .replace('__DATA__', json.dumps(payload, separators=(',', ':'))))

    out_path = args.out or os.path.join(d, 'intersection_viewer.html')
    with open(out_path, 'w') as f:
        f.write(html)
    n_jx = sum(1 for pt in payload['points'] if pt['kind'] == 'junction')
    print(f"wrote {out_path} ({len(html) // 1024} KB): "
          f"{len(payload['symbols'])} symbols, {len(payload['runs'])} runs, "
          f"{len(payload['points'])} points ({n_jx} junctions), "
          f"{len(payload['symbol_links'])} symbol links")


if __name__ == '__main__':
    main()
