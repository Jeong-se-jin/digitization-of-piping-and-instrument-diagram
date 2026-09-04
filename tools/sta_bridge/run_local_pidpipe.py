"""Run the pidpipe line-detection backend locally, then graph construction.

A sibling of ``run_local.py``, which it leaves alone: same inputs, same output
filenames, same offline setup -- only the line-detection stage differs.  Run
both into different directories to compare backends on one sheet:

    python -m tools.sta_bridge.run_local         --output-dir out/18 ...
    python -m tools.sta_bridge.run_local_pidpipe --output-dir out/18-pidpipe ...

Usage:
    python -m tools.sta_bridge.run_local_pidpipe \
        --text-detection out/18/text_detection.json \
        --image STA-main/results/18/diagram.png \
        --output-dir out/18-pidpipe
"""
import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'src'))

# Placeholders for the import-time validators in app/config.py, same as
# run_local.py. Nothing here dials a service.
for _key, _val in {
    'BLOB_STORAGE_ACCOUNT_URL': 'https://local',
    'BLOB_STORAGE_CONTAINER_NAME': 'local',
    'FORM_RECOGNIZER_ENDPOINT': 'https://local',
    'SYMBOL_DETECTION_API': 'http://local',
    'SYMBOL_DETECTION_API_BEARER_TOKEN': 'local',
    'GRAPH_DB_CONNECTION_STRING': 'local',
}.items():
    os.environ.setdefault(_key, _val)


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument('--thinning', dest='thinning', action='store_true', default=None)
    parser.add_argument('--no-thinning', dest='thinning', action='store_false')

    tuning = parser.add_argument_group(
        'pidpipe tuning', 'omit to use the defaults in PidpipeParams')
    tuning.add_argument('--hough-threshold', type=int)
    tuning.add_argument('--hough-min-line-length', type=int)
    tuning.add_argument('--hough-max-line-gap', type=int)
    tuning.add_argument('--angle-tolerance-deg', type=float,
                       help='how far off-axis a segment may be and still be kept')
    tuning.add_argument('--coord-tolerance-px', type=int,
                       help='off-axis distance at which two segments are one line')
    tuning.add_argument('--merge-gap-px', type=int,
                       help='gap merged without an evidence check')
    tuning.add_argument('--bridge-max-gap-px', type=int,
                       help='largest gap a bridge may span')
    tuning.add_argument('--support-min-ratio', type=float,
                       help='ink coverage a bridge must find along the gap')
    tuning.add_argument('--support-band-px', type=int)
    tuning.add_argument('--crossing-guard-max', type=int,
                       help='perpendicular crossings that veto a bridge')
    tuning.add_argument('--min-length-px', type=int)
    tuning.add_argument('--no-corner-snap', dest='corner_snap',
                       action='store_false', default=None)
    return parser


_TUNING_FIELDS = (
    'hough_threshold', 'hough_min_line_length', 'hough_max_line_gap',
    'angle_tolerance_deg', 'coord_tolerance_px', 'merge_gap_px',
    'bridge_max_gap_px', 'support_min_ratio', 'support_band_px',
    'crossing_guard_max', 'min_length_px', 'corner_snap',
)


def main():
    args = _build_parser().parse_args()

    os.environ.setdefault('DEBUG', 'true')

    from app.config import config
    from app.models.graph_construction.graph_construction_request import \
        GraphConstructionInferenceRequest
    from app.services.graph_construction import graph_construction_service

    from tools.sta_bridge import pidpipe_lines

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)  # noqa: E731

    with open(args.image, 'rb') as f:
        image_bytes = f.read()
    with open(args.text_detection) as f:
        request = GraphConstructionInferenceRequest.parse_raw(f.read())

    height = request.image_details.height
    width = request.image_details.width

    thinning = (args.thinning if args.thinning is not None
                else config.enable_thinning_preprocessing_line_detection)

    # Only the flags actually given override the dataclass defaults.
    overrides = {name: getattr(args, name) for name in _TUNING_FIELDS
                 if getattr(args, name, None) is not None}
    params = pidpipe_lines.PidpipeParams(**overrides)

    print(f'[1/2] Line detection: pidpipe  ({width}x{height}px, '
          f'{len(request.text_and_symbols_associated_list)} symbols, '
          f'{len(request.all_text_list)} texts, thinning={thinning})')
    if overrides:
        print('      overrides: '
              + ', '.join(f'{k}={v}' for k, v in sorted(overrides.items())))

    line_results = pidpipe_lines.detect_lines(
        image_bytes=image_bytes,
        text_detection_results=request,
        image_height=height,
        image_width=width,
        enable_thinning=thinning,
        params=params,
        bounding_box_inclusive=request.bounding_box_inclusive,
        image_url=request.image_url,
        debug_image_preprocessed_path=out('10_preprocessed.png'),
        debug_image_preprocessed_before_thinning_path=out('11_before_thinning.png'),
        debug_image_lines_on_preprocessed_path=out('12b_lines_on_preprocessed.png'),
        output_image_line_segments_path=out('12_line_segments.png'),
    )

    with open(out('line_detection.json'), 'w') as f:
        json.dump(line_results.dict(), f, indent=2)
    print(f'      line segments: {line_results.line_segments_count}')

    if args.skip_graph:
        print(f'      outputs in {args.output_dir}')
        return

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
    print(f'      outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
