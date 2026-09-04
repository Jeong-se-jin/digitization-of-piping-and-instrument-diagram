"""Run line detection + graph construction locally, with no Azure services.

Both stages are pure functions -- ``detect_lines`` touches blob storage not at
all, and its only file writes are cv2.imwrite calls guarded by ``config.debug``
-- so the whole thing runs offline once ``app.config`` can be imported.  That
import is the one gate: ``Config()`` executes at import time and rejects empty
values for blob_storage_account_url, form_recognizer_endpoint,
symbol_detection_api, symbol_detection_api_bearer_token and
graph_db_connection_string.  We satisfy it with placeholders below; nothing
ever dials them.

Usage:
    python -m tools.sta_bridge.run_local --text-detection out/18/text_detection.json \
                                         --image STA-main/results/18/diagram.png \
                                         --output-dir out/18
"""
import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'src'))

# Placeholders for the import-time validators in app/config.py.  Set before
# importing anything under `app`.  Real values, if present in the environment
# or src/.env, win.
for _key, _val in {
    'BLOB_STORAGE_ACCOUNT_URL': 'https://local',
    'BLOB_STORAGE_CONTAINER_NAME': 'local',
    'FORM_RECOGNIZER_ENDPOINT': 'https://local',
    'SYMBOL_DETECTION_API': 'http://local',
    'SYMBOL_DETECTION_API_BEARER_TOKEN': 'local',
    'GRAPH_DB_CONNECTION_STRING': 'local',
}.items():
    os.environ.setdefault(_key, _val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-detection', required=True,
                        help='payload from tools.sta_bridge.adapt')
    parser.add_argument('--image', required=True,
                        help="STA's cropped diagram.png -- must be the image the "
                             'bboxes were measured against')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pid-id', default='local')
    parser.add_argument('--skip-graph', action='store_true',
                        help='stop after line detection')
    # Hough overrides; None -> config default
    parser.add_argument('--hough-threshold', type=int)
    parser.add_argument('--hough-min-line-length', type=int)
    parser.add_argument('--hough-max-line-gap', type=int)
    parser.add_argument('--hough-rho', type=float)
    parser.add_argument('--hough-theta', type=int)
    parser.add_argument('--thinning', dest='thinning', action='store_true', default=None)
    parser.add_argument('--no-thinning', dest='thinning', action='store_false')
    parser.add_argument('--thin-min-stroke-width', type=float,
                        help='thin only strokes at least this many pixels wide, '
                             'leaving thinner ones (dashes) at full length')
    parser.add_argument('--binary-threshold', type=int,
                        help='grey value below which a pixel is ink (default 240); '
                             '0 uses Otsu')
    parser.add_argument('--line-backend', choices=('hough', 'fld', 'lsd'),
                        help='raw line detector; default from config (hough)')
    parser.add_argument('--fld-length', type=int,
                        help='FLD length_threshold; shorter segments are dropped '
                             'inside the detector. Defaults to --hough-min-line-length')
    parser.add_argument('--fld-distance', type=float,
                        help='FLD distance_threshold, the point-to-line tolerance '
                             'when growing a segment (default 1.414)')
    parser.add_argument('--fld-merge', action='store_true',
                        help='turn FLD segment merging back on. Off by default: it '
                             'joins collinear neighbours and swallows the dashes a '
                             'signal line is made of')
    parser.add_argument('--dedup-segments', action='store_true',
                        help='drop segments that re-detect a stroke another segment '
                             'already covers. Lets --no-thinning be used without the '
                             'double detection thinning existed to prevent.')
    parser.add_argument('--thinning-iterations', type=float,
                        help='stop thinning after N passes instead of running to a '
                             'one-pixel skeleton. Strokes here are ~2.3px, so 1 is '
                             'already a partial thin.')
    parser.add_argument('--symbol-mask-inset', type=int,
                        help='shrink each symbol mask box by N px, keeping the pipe '
                             'stubs that meet the symbol')
    parser.add_argument('--text-mask-inset-x', type=int,
                        help='horizontal inset for text boxes. This is the one that '
                             'pulls a tag box away from a pipe running beside it: a '
                             'tag set vertically gets a narrow box, so trimming its '
                             'width moves the mask edge off the pipe.')
    parser.add_argument('--box-mask-inset', type=int, default=2,
                        help='mask inset for panel boxes, used instead of '
                             '--symbol-mask-inset. They are drawn rectangles, so a '
                             'large inset would just leave their own border behind.')
    parser.add_argument('--text-mask-inset-y', type=int,
                        help='vertical inset for text boxes. Independent of -x; it '
                             'shortens the box along the pipe instead of across it, '
                             'and tends to leave the top and bottom of the glyphs '
                             'behind.')
    parser.add_argument('--associate-leftover-text', action='store_true',
                        help='attach text that won no symbol to a nearby line or '
                             'symbol (see associate_leftover_text)')
    parser.add_argument('--leftover-text-distance', type=float, default=10.0,
                        help='how near, in pixels; default 10')
    parser.add_argument('--drop-boxed-segments', action='store_true',
                        help='keep segments that lie entirely inside one symbol or '
                             'text box out of graph construction. Containment must '
                             'be total -- a segment merely crossing a box is kept, '
                             'since that is pipe passing through. A box is convex, '
                             'so both endpoints inside means the whole segment is.')
    parser.add_argument('--axis-aligned-only', action='store_true',
                        help='drop every segment that is neither horizontal nor '
                             'vertical. P&ID piping runs on the axes; what slants '
                             'is usually a leader, hatching or a scrap left by a '
                             'symbol mask, and it still competes for the one '
                             'candidate an endpoint has to give.')
    parser.add_argument('--axis-tolerance', type=float,
                        help='degrees off the axis still counted as aligned; '
                             'default 5')
    parser.add_argument('--strip-red', action='store_true',
                        help='before anything else, lift the red out of the '
                             'drawing: save it as its own layer with its blobs '
                             'boxed, erase it, and flatten what remains to black '
                             'on white. Red on a P&ID is a second drawing laid '
                             'over the first -- a revision cloud, an as-built '
                             'markup -- and left in it is found as pipe. The '
                             'original is kept as diagram_source.png.')
    parser.add_argument('--red-dilate', type=int, default=1,
                        help='pixels to grow the red mask before erasing, so the '
                             "stroke's antialiased edge goes with it")
    parser.add_argument('--no-text-mask', action='store_true',
                        help='leave the text in the image for line detection. A '
                             'pipe running under a label is normally cut in two '
                             'by the mask and stops being one run. The letters '
                             'then produce their own short segments, so pair '
                             'this with --intersection-graph, which prunes runs '
                             'that lead nowhere.')
    parser.add_argument('--prune-dangling', action='store_true',
                        help='with --intersection-graph, drop runs that end up '
                             'with fewer than two connection points. Mostly '
                             'redundant: the letter strokes this was meant to '
                             'remove sit inside their own text boxes, so '
                             '--drop-boxed-segments already has them.')
    parser.add_argument('--intersection-graph', action='store_true',
                        help='additionally build the crossing-based graph: stitch '
                             'the axis-aligned segments into runs, record every '
                             'point where two runs cross or a run meets a symbol '
                             'box, and write intersection_graph.json plus its own '
                             'overlay and viewer. Leaves the normal graph '
                             'untouched -- this is a second, independent route.')
    parser.add_argument('--all-symbols-as-assets', action='store_true',
                        help='make every detected symbol a graph node, not only the '
                             'ones whose associated text looks like a tag')
    parser.add_argument('--classify-line-types', action='store_true',
                        help='label each segment solid (pipe) or dashed (signal)')
    parser.add_argument('--exclude-dashed', action='store_true',
                        help='keep dashed segments out of graph construction. Lets '
                             'Hough be opened up enough to catch dashes without the '
                             'signal lines bridging unrelated pipe runs. Implies '
                             '--classify-line-types; the full set is still written '
                             'to line_detection.json.')
    args = parser.parse_args()

    os.environ.setdefault('DEBUG', 'true')  # enables the preprocessed-image dumps

    from app.config import config
    if args.all_symbols_as_assets:
        config.treat_all_symbols_as_assets = True
    from tools.sta_bridge.label_map import NO_MASK_INSET_LABELS
    from app.models.graph_construction.graph_construction_request import \
        GraphConstructionInferenceRequest
    from app.services.line_detection import line_detection_service
    from app.services.graph_construction import graph_construction_service

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)  # noqa: E731

    if args.strip_red:
        import cv2 as _cv2
        from tools.sta_bridge import split_red as _split_red
        _source = _cv2.imread(args.image)
        if _source is None:
            raise SystemExit(f'ERROR: cannot read {args.image}')
        _mask = _split_red.red_mask(_source)
        _black, _red = _split_red.split(_source, _mask, args.red_dilate)
        _blobs = _split_red.components(_mask)
        _cv2.imwrite(out('diagram_source.png'), _source)
        _cv2.imwrite(out('diagram_red.png'), _red)
        _cv2.imwrite(out('diagram_red_mask.png'), _mask.astype('uint8') * 255)
        _cv2.imwrite(out('diagram.png'), _black)
        with open(out('red_layer.json'), 'w') as f:
            json.dump({'image_details': {'width': _source.shape[1],
                                         'height': _source.shape[0]},
                       'source': os.path.abspath(args.image),
                       'red_pixels': int(_mask.sum()),
                       'components': _blobs}, f, indent=1)
        _ink = (_cv2.cvtColor(_source, _cv2.COLOR_BGR2GRAY) < 200).sum()
        print(f'      red split off: {int(_mask.sum())} px '
              f'({_mask.sum() / max(_ink, 1) * 100:.1f}% of the ink), '
              f'{len(_blobs)} components; the rest is black')
        # Everything downstream -- detection, overlays, the viewer -- reads the
        # image without the red.
        args.image = out('diagram.png')

    with open(args.image, 'rb') as f:
        image_bytes = f.read()
    with open(args.text_detection) as f:
        request = GraphConstructionInferenceRequest.parse_raw(f.read())

    height = request.image_details.height
    width = request.image_details.width

    # Mirror the resolution order the controller uses: request value, else config.
    pick = lambda a, c: a if a is not None else c  # noqa: E731
    thinning = pick(args.thinning, config.enable_thinning_preprocessing_line_detection)

    print(f'[1/2] Line detection  ({width}x{height}px, '
          f'{len(request.text_and_symbols_associated_list)} symbols, '
          f'{len(request.all_text_list)} texts, thinning={thinning})')

    line_results = line_detection_service.detect_lines(
        pid_id=args.pid_id,
        image_bytes=image_bytes,
        text_detection_results=request,
        enable_thinning=thinning,
        threshold=pick(args.hough_threshold, config.line_detection_hough_threshold),
        max_line_gap=pick(args.hough_max_line_gap, config.line_detection_hough_max_line_gap),
        min_line_length=pick(args.hough_min_line_length, config.line_detection_hough_min_line_length),
        rho=pick(args.hough_rho, config.line_detection_hough_rho),
        theta_param=pick(args.hough_theta, config.line_detection_hough_theta),
        bounding_box_inclusive=request.bounding_box_inclusive,
        image_height=height,
        image_width=width,
        debug_image_preprocessed_path=out('10_preprocessed.png'),
        debug_image_preprocessed_before_thinning_path=out('11_before_thinning.png'),
        output_image_line_segments_path=out('12_line_segments.png'),
        symbol_mask_inset=pick(args.symbol_mask_inset,
                               config.line_detection_symbol_mask_inset_pixels),
        text_mask_inset_x=pick(args.text_mask_inset_x,
                               config.line_detection_text_mask_inset_x_pixels),
        text_mask_inset_y=pick(args.text_mask_inset_y,
                               config.line_detection_text_mask_inset_y_pixels),
        binary_threshold=pick(args.binary_threshold,
                              config.line_detection_binary_threshold),
        backend=pick(args.line_backend, config.line_detection_backend),
        backend_params={
            # Left unset, the FLD length threshold is derived from stroke width
            # rather than borrowing the Hough minimum, which is tuned for a
            # different thing.
            'fld_length_threshold': args.fld_length,
            'min_line_length': None,
            'fld_distance_threshold': (args.fld_distance
                                       if args.fld_distance is not None else 1.414),
            'fld_do_merge': args.fld_merge or config.line_detection_fld_merge,
        },
        deduplicate_segments=(args.dedup_segments
                              or config.line_detection_deduplicate_segments),
        axis_aligned_only=(args.axis_aligned_only
                           or config.line_detection_axis_aligned_only),
        mask_text=not args.no_text_mask,
        axis_tolerance_degrees=pick(args.axis_tolerance,
                                    config.line_detection_axis_tolerance_degrees),
        thin_min_stroke_width=pick(args.thin_min_stroke_width,
                                   config.line_detection_thin_min_stroke_width),
        thinning_iterations=pick(args.thinning_iterations,
                                 config.line_detection_thinning_iterations),
        classify_line_types=(args.classify_line_types or args.exclude_dashed
                             or config.classify_line_types),
        mask_inset_exempt_labels=NO_MASK_INSET_LABELS or None,
        exempt_mask_inset=args.box_mask_inset,
    )

    if args.drop_boxed_segments:
        boxes = [(b.topX, b.topY, b.bottomX, b.bottomY) for b in
                 list(request.text_and_symbols_associated_list) + list(request.all_text_list)]

        def inside_a_box(seg):
            for x1, y1, x2, y2 in boxes:
                if (x1 <= seg.startX <= x2 and y1 <= seg.startY <= y2 and
                        x1 <= seg.endX <= x2 and y1 <= seg.endY <= y2):
                    return True
            return False

        # Marked rather than removed, so line_detection.json still records every
        # segment Hough found and the overlays can show why one was set aside.
        for seg in line_results.line_segments:
            seg.inside_box = inside_a_box(seg)
        n_inside = sum(1 for s in line_results.line_segments if s.inside_box)
        print(f'      {n_inside} segments lie entirely inside a symbol or text box')

    if args.associate_leftover_text:
        from app.services.graph_construction.associate_leftover_text import \
            associate_leftover_text
        attached = associate_leftover_text(
            request.all_text_list,
            request.text_and_symbols_associated_list,
            line_results.line_segments,
            height, width, args.leftover_text_distance)
        with open(out('leftover_text.json'), 'w') as f:
            json.dump(attached, f, indent=2, ensure_ascii=False)
        # The symbols were updated in place, so the request written out matches
        # what graph construction is about to see.
        with open(args.text_detection) as f_in:
            pass
        with open(out('text_detection.json'), 'w') as f:
            json.dump(request.dict(), f, indent=2)

    with open(out('line_detection.json'), 'w') as f:
        json.dump(line_results.dict(), f, indent=2)
    print(f'      line segments: {line_results.line_segments_count}')

    if args.skip_graph:
        return

    if args.drop_boxed_segments:
        kept = [s for s in line_results.line_segments if not s.inside_box]
        print(f'      excluded {len(line_results.line_segments) - len(kept)} boxed '
              f'segments from the graph')
        line_results = line_results.copy(
            update={'line_segments': kept, 'line_segments_count': len(kept)})

    if args.exclude_dashed:
        # Graph construction sees pipe only. line_detection.json above keeps
        # every segment, so the dashed ones are still available downstream.
        solid = [s for s in line_results.line_segments if s.line_type != 'dashed']
        n_dropped = len(line_results.line_segments) - len(solid)
        line_results = line_results.copy(
            update={'line_segments': solid, 'line_segments_count': len(solid)})
        print(f'      excluded {n_dropped} dashed segments from the graph')

    print('[2/2] Graph construction')
    asset_connectivities, arrow_nodes = graph_construction_service.construct_graph(
        pid_id=args.pid_id,
        pid_image=image_bytes,
        text_detection_results=request,
        line_detection_results=line_results,
        output_image_graph_path=out('13_graph.png'),
        debug_image_graph_connections_path=out('14_graph_connections.png'),
        debug_image_graph_with_lines_and_symbols_path=out('15_graph_lines_symbols.png'),
        symbol_label_prefixes_to_include_in_graph_image_output=(
            config.symbol_label_prefixes_to_include_in_graph_image_output),
    )

    with open(out('graph_connectivity.json'), 'w') as f:
        json.dump([a.dict() for a in asset_connectivities], f, indent=2)

    print(f'      connected assets: {len(asset_connectivities)}')
    print(f'      arrow nodes with a direction: {len(arrow_nodes)}')

    # Every run folder gets the same set of overlays. These used to need a
    # separate command, so whether a folder had them depended on what was run
    # by hand afterwards.
    import shutil
    from tools.sta_bridge import draw_connections
    import cv2

    # run_all already hands us out/<name>/diagram.png as the image, so the
    # copy would be onto itself; copyfile raises SameFileError on that.
    if not os.path.exists(out('diagram.png')) or not os.path.samefile(
            args.image, out('diagram.png')):
        shutil.copyfile(args.image, out('diagram.png'))
    if not os.path.exists(out('text_detection.json')):
        shutil.copyfile(args.text_detection, out('text_detection.json'))

    img = cv2.imread(out('diagram.png'))
    h, w = img.shape[:2]
    graph = [a.dict() for a in asset_connectivities]
    all_segments = json.load(open(out('line_detection.json')))['line_segments']
    symbols = request.dict()['text_and_symbols_associated_list']

    n_edges, n_iso = draw_connections.draw_connections(
        graph, img, w, h, out('20_connections.png'))
    n_used = draw_connections.draw_orphans(
        graph, all_segments, symbols, img, w, h, out('21_orphan_lines.png'))
    print(f'      overlays: {n_edges} edges, {n_iso} isolated assets, '
          f'{len(all_segments) - n_used} orphaned segments')

    if args.intersection_graph:
        print('[extra] Crossing-based graph')
        import subprocess
        import sys as _sys
        for module, extra in (
            ('tools.sta_bridge.intersection_graph',
             ['--prune-dangling'] if args.prune_dangling else []),
            ('tools.sta_bridge.intersection_viewer', ['--title', args.pid_id]),
        ):
            subprocess.run([_sys.executable, '-m', module,
                            '--output-dir', args.output_dir] + extra, check=True)

    print(f'      outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
