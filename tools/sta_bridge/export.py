"""Run the symbol + OCR pipeline and dump what line detection needs.

The pipeline itself is ``PID_pipeline_.py``, now vendored beside this file and
edited here. This module drives it: it mirrors the call sequence of that file's
``main()`` rather than calling it, so each stage's result is available to write
out. If the pipeline gains or reorders a stage, this needs the same change --
the duplication buys access to the intermediates, not less coupling.

``associations.json`` alone is not enough downstream, for two reasons:

* when ``crop_diagram`` runs, every bbox is in *cropped* coordinates, so the
  image the next stage reads must be that crop, not the original scan -- hence
  ``diagram.png``; and
* it records only the tags that won a symbol, while line detection masks out
  *all* text before Hough runs.  Any text left unmasked gets picked up as line
  strokes, so the export carries the full OCR list.

Run it with STA's interpreter -- RF-DETR and PaddleOCR live there, and they need
pydantic 2, which this repo's own venv cannot have:

    /home/rx/project/STA-main/.venv-detect/bin/python -m tools.sta_bridge.export \
        --image /home/rx/project/STA-main/samples/18.png \
        --output-dir out/18 --no-crop

Or let ``run_all`` pick the interpreter for each stage.
"""
import argparse
import json
import os
import sys

# Only the RF-DETR checkpoint still lives in the STA checkout; it is ~128 MB, so
# it is referenced rather than copied.
DEFAULT_STA_ROOT = '/home/rx/project/STA-main'


CHECKPOINT = os.path.join('model', 'checkpoint_best_total.pth')


def _find_weights(here, sta_root):
    """This repo's checkpoint, falling back to the STA checkout it came from."""
    for root in (here, sta_root):
        path = os.path.join(root, CHECKPOINT)
        if os.path.exists(path):
            return path
    raise SystemExit(
        f'ERROR: no {CHECKPOINT} under {here} or {sta_root}. '
        f'Pass --weights with the checkpoint path.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sta-root', default=DEFAULT_STA_ROOT,
                   help='STA checkout. Only consulted with --pipeline-from-sta, or '
                        'as a fallback for the checkpoint. Default: '
                        f'{DEFAULT_STA_ROOT}')
    p.add_argument('--image', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pipeline-from-sta', action='store_true',
                   help="import STA's copy of PID_pipeline_ instead of this repo's, "
                        'to compare the two')
    p.add_argument('--yaml', help='class names (default: this repo\'s dataset.yaml)')
    p.add_argument('--weights', help='RF-DETR checkpoint '
                                     '(default: <sta-root>/model/checkpoint_best_total.pth)')
    # Pass-throughs. Defaults are None so STA's own constants stay the source
    # of truth -- including any you have retuned there.
    p.add_argument('--symbol-conf', type=float)
    p.add_argument('--symbol-scale', default='auto',
                   help="detection input scale, or 'auto' to infer it "
                        "(mirrors PID_pipeline_'s --symbol-scale)")
    p.add_argument('--cost-cap', type=float)
    p.add_argument('--ocr-scale', type=float)
    p.add_argument('--det-unclip-ratio', type=float)
    p.add_argument('--device')
    p.add_argument('--no-crop', action='store_true')
    p.add_argument('--no-merge', action='store_true')
    p.add_argument('--restrict-charset', action='store_true')
    p.add_argument('--ocr-cache',
                   help='JSON of consolidated OCR results (same format as '
                        "PID_pipeline_'s --ocr-cache). Reuses it instead of running "
                        'OCR, which is the slow stage. Its coordinates must match '
                        'the crop mode of this run.')
    p.add_argument('--sta-outputs', action='store_true',
                   help="also write STA's own artifacts (the OCR, symbol and "
                        'association overlays, associations.json/xlsx) into '
                        'output-dir, so one folder holds the whole run')
    return p.parse_args()


def main():
    args = parse_args()

    sta_root = os.path.abspath(args.sta_root)
    here = os.path.dirname(os.path.abspath(__file__))
    # PID_pipeline_ lives in this repo now and is edited here; STA's copy is only
    # imported on request. The checkpoint stays in the STA checkout either way.
    pipeline_dir = sta_root if args.pipeline_from_sta else here
    sys.path.insert(0, pipeline_dir)
    import cv2
    import PID_pipeline_ as sta
    print(f'pipeline {sta.__file__}')

    pick = lambda a, c: c if a is None else a  # noqa: E731
    yaml_path = args.yaml or os.path.join(pipeline_dir, 'dataset.yaml')
    weights = args.weights or _find_weights(here, sta_root)
    ocr_scale = pick(args.ocr_scale, sta.OCR_UPSCALE)
    unclip = pick(args.det_unclip_ratio, sta.OCR_DET_UNCLIP_RATIO)
    symbol_conf = pick(args.symbol_conf, sta.SYMBOL_DETECTION_CONFIDENCE)
    cost_cap = pick(args.cost_cap, sta.ASSOCIATION_COST_CAP)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. image ---------------------------------------------------------
    image = sta.load_image(args.image)
    diagram = image if args.no_crop else sta.crop_diagram(image)
    height, width = diagram.shape[:2]
    print(f'[1/4] diagram {width}x{height}px (crop={"off" if args.no_crop else "on"})')

    # --- 2. OCR (by far the slowest stage, so reuse a cache when offered) ---
    if args.ocr_cache and os.path.exists(args.ocr_cache):
        with open(args.ocr_cache) as f:
            texts = json.load(f)
        for t in texts:  # JSON turns the centre tuple into a list
            t['center'] = tuple(t['center'])
        xs = [v for t in texts for v in (t['bbox'][0], t['bbox'][2])]
        ys = [v for t in texts for v in (t['bbox'][1], t['bbox'][3])]
        print(f'[2/4] OCR cache: {len(texts)} texts from {args.ocr_cache}')
        # A cache built under the other crop mode puts every box in the wrong
        # place, and nothing downstream would flag it -- so check the extents.
        if max(xs) > width or max(ys) > height:
            raise SystemExit(
                f'ERROR: cache coordinates reach ({max(xs)}, {max(ys)}) but this '
                f'run\'s diagram is {width}x{height}. The cache was built with a '
                f'different crop mode; re-run with{"out" if args.no_crop else ""} '
                f'--no-crop, or drop --ocr-cache.')
        print(f'      extent ({max(xs)}, {max(ys)}) fits {width}x{height}')
    else:
        ocr_engine = sta.build_ocr_engine(unclip)
        if args.restrict_charset:
            print(f'      charset restricted to {sta.restrict_ocr_charset(ocr_engine)} entries')
        ocr_input = sta.upscale_for_ocr(diagram, ocr_scale)
        tiles = sta.generate_tiles(ocr_input, tile_size=sta.TILE_SIZE, overlap=sta.TILE_OVERLAP)
        raw_ocr = sta.rescale_detections(sta.run_ocr_on_tiles(tiles, ocr_engine), ocr_scale)
        texts = sta.deduplicate_ocr(raw_ocr)
        if not args.no_merge:
            texts = sta.merge_adjacent_lines(texts)
        texts, n_repaired = sta.repair_tag_text(texts)
        print(f'[2/4] OCR: {len(raw_ocr)} raw -> {len(texts)} cleaned ({n_repaired} repaired)')
        if args.ocr_cache:
            with open(args.ocr_cache, 'w') as f:
                json.dump(texts, f)
            print(f'      cache written: {args.ocr_cache}')

    # --- 3. symbols -------------------------------------------------------
    # Keep this in step with PID_pipeline_.main(): scale inference, sliced
    # detection, then the class-agnostic dedup that drops one symbol detected
    # under two class names.
    id_to_name = sta.load_class_mapping(yaml_path)
    device = sta.resolve_device(args.device)
    model = sta.build_detection_model(weights, id_to_name, device, symbol_conf)

    if str(args.symbol_scale).lower() == 'auto':
        symbol_scale = sta.infer_symbol_scale(diagram, weights, id_to_name, device)
    else:
        symbol_scale = float(args.symbol_scale)

    symbols = sta.detect_symbols(diagram, model, symbol_scale)
    n_raw = len(symbols)
    symbols = sta.deduplicate_symbols(symbols)
    print(f'[3/4] symbols: {len(symbols)} (device={device}, conf={symbol_conf}, '
          f'scale=x{symbol_scale}'
          + (f', {n_raw - len(symbols)} overlapping removed' if n_raw != len(symbols) else '')
          + ')')

    # --- 4. association + export -----------------------------------------
    associations = sta.associate_symbols_to_text(symbols, texts, cost_cap)
    tag_by_symbol = {a['symbol_id']: a['tag'] for a in associations}
    print(f'[4/4] associations: {len(associations)} (cost cap={cost_cap})')

    diagram_path = os.path.join(args.output_dir, 'diagram.png')
    cv2.imwrite(diagram_path, diagram)

    export = {
        'image': os.path.basename(diagram_path),
        'image_width': int(width),
        'image_height': int(height),
        'source_image': os.path.abspath(args.image),
        'cropped': not args.no_crop,
        'symbols': [
            {
                'id': int(s['id']),
                'class': s['class'],
                'score': float(s['score']),
                'bbox': [int(v) for v in s['bbox']],
                'tag': tag_by_symbol.get(s['id']),
            }
            for s in symbols
        ],
        'texts': [
            {'text': t['text'], 'score': float(t['score']),
             'bbox': [int(v) for v in t['bbox']]}
            for t in texts
        ],
    }

    export_path = os.path.join(args.output_dir, 'sta_export.json')
    with open(export_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f'      wrote {diagram_path}')
    print(f'      wrote {export_path}')

    if args.sta_outputs:
        # STA's own views of this run, produced by its own functions rather than
        # re-drawn here, so they stay identical to what running STA gives you.
        out = lambda n: os.path.join(args.output_dir, n)  # noqa: E731
        sta.save_associations_json(associations, out('associations.json'))
        try:
            sta.save_associations_excel(associations, out('associations.xlsx'))
        except Exception as e:                       # openpyxl is optional
            print(f'      associations.xlsx skipped: {e}')
        sta.visualise_ocr(diagram, texts, title='OCR detections',
                          save_path=out('02_cleaned_ocr.png'))
        sta.visualise_symbols(diagram, symbols, save_path=out('03_symbols.png'))
        sta.visualise_associations(diagram, associations,
                                   save_path=out('04_associations.png'))
        print(f"      wrote STA's own outputs into {args.output_dir}")


if __name__ == '__main__':
    main()
