"""Run the whole chain on one sheet, from image to interactive viewer.

    image -> STA (symbols + OCR) -> adapt -> line detection -> graph -> viewer

Everything lands in one folder under this repo, STA's own overlays included, so
a sheet's entire run is in a single place.

Why this is a driver and not one process: STA needs pydantic 2 (RF-DETR,
PaddleOCR) and this repo needs pydantic 1 (``BaseSettings``, v1 validators).
They cannot be imported together, so each stage runs in the venv that suits it
and hands the next stage files on disk.

    stage 1 (STA)          STA_PYTHON, default <sta-root>/.venv-detect/bin/python
    stages 2-5 (this repo) PID_PYTHON, default <repo>/.venv-pid/bin/python

Usage:
    python -m tools.sta_bridge.run_all --image /path/to/18.png --name 18

    # skip the slow stage by reusing (or seeding) an OCR cache
    python -m tools.sta_bridge.run_all --image ... --name 18 \
        --ocr-cache STA-main/results/ocr_cache_18.json

    # already have STA results on disk? read them instead of running STA
    python -m tools.sta_bridge.run_all --image ... --name 18 \
        --from-results /home/rx/project/STA-main/results/18
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_STA_ROOT = '/home/rx/project/STA-main'


def run(label, python, module, argv, env=None):
    cmd = [python, '-m', module] + argv
    print(f'\n\033[1m== {label} ==\033[0m')
    print('  ' + ' '.join(cmd[1:]))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise SystemExit(f'\n{label} failed (exit {r.returncode}); stopping here.')
    print(f'  {time.time() - t0:.1f}s')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', required=True, help='the sheet to digitize')
    p.add_argument('--name', help='output folder name; default: the image stem')
    p.add_argument('--output-root', default=os.path.join(REPO, 'out'))
    p.add_argument('--sta-root', default=DEFAULT_STA_ROOT)
    p.add_argument('--sta-python', help='default: <sta-root>/.venv-detect/bin/python')
    p.add_argument('--pid-python', help='default: <repo>/.venv-pid/bin/python')
    p.add_argument('--ocr-cache', help='reuse if present, write if absent')
    p.add_argument('--from-results',
                   help="an STA results folder to read instead of running STA")
    p.add_argument('--crop', action='store_true',
                   help="apply STA's crop_diagram(). Off by default: its slice is "
                        'fixed to 7168x4562 sheets and cuts into smaller ones.')
    p.add_argument('--symbol-conf', type=float)
    p.add_argument('--symbol-scale')
    p.add_argument('--cost-cap', type=float)
    p.add_argument('--restrict-charset', action='store_true', default=True)
    p.add_argument('--no-restrict-charset', dest='restrict_charset',
                   action='store_false')
    p.add_argument('--title', help='viewer title; default: "<name> Connectivity"')
    # Line-detection knobs, passed straight through to run_local.
    p.add_argument('--hough-threshold', type=int)
    p.add_argument('--hough-min-line-length', type=int)
    p.add_argument('--hough-max-line-gap', type=int)
    p.add_argument('--hough-rho', type=float)
    p.add_argument('--hough-theta', type=int)
    p.add_argument('--no-thinning', action='store_true')
    p.add_argument('--thinning-iterations', type=float)
    p.add_argument('--thin-min-stroke-width', type=float)
    p.add_argument('--binary-threshold', type=int)
    p.add_argument('--line-backend', choices=('hough','fld','lsd'))
    p.add_argument('--fld-length', type=int)
    p.add_argument('--fld-distance', type=float)
    p.add_argument('--fld-merge', action='store_true')
    p.add_argument('--dedup-segments', action='store_true')
    p.add_argument('--classify-line-types', action='store_true')
    p.add_argument('--all-symbols-as-assets', action='store_true')
    p.add_argument('--drop-boxed-segments', action='store_true')
    p.add_argument('--strip-red', action='store_true',
                   help='lift the red out of the sheet before stage 1, so symbol '
                        'detection and OCR see it gone too. Writes stripped.png '
                        '(the drawing in black, red erased) and stripped_red.png '
                        'beside it, and runs everything on the former.')
    p.add_argument('--red-dilate', type=int, default=1)
    p.add_argument('--associate-leftover-text', action='store_true')
    p.add_argument('--leftover-text-distance', type=float)
    p.add_argument('--exclude-dashed', action='store_true')
    p.add_argument('--symbol-mask-inset', type=int)
    p.add_argument('--box-mask-inset', type=int)
    p.add_argument('--text-mask-inset-x', type=int)
    p.add_argument('--text-mask-inset-y', type=int)
    p.add_argument('--debug', action='store_true', default=True,
                   help='write the intermediate images (on by default)')
    p.add_argument('--no-debug', dest='debug', action='store_false')
    args = p.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(args.output_root, name)
    sta_python = args.sta_python or os.path.join(
        args.sta_root, '.venv-detect', 'bin', 'python')
    pid_python = args.pid_python or os.path.join(REPO, '.venv-pid', 'bin', 'python')

    for label, path in (('STA', sta_python), ('this repo', pid_python)):
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: no {label} interpreter at {path}')

    os.makedirs(out_dir, exist_ok=True)
    print(f'sheet  {args.image}')
    print(f'out    {out_dir}')

    # --- 0. red layer -----------------------------------------------------
    # Before stage 1, so the symbol detector and OCR never see the annotation.
    if args.strip_red:
        run('0/5  split the red layer off', pid_python,
            'tools.sta_bridge.split_red',
            ['--image', args.image, '--output-dir', out_dir,
             '--name', 'stripped', '--dilate', str(args.red_dilate)])
        args.image = os.path.join(out_dir, 'stripped.png')

    # --- 1. symbols + OCR -------------------------------------------------
    if args.from_results:
        argv = ['--results-dir', args.from_results, '--image', args.image,
                '--output-dir', out_dir]
        if args.ocr_cache:
            argv += ['--ocr-cache', args.ocr_cache]
        run('1/5  STA results -> export', pid_python,
            'tools.sta_bridge.from_results', argv)
    else:
        argv = ['--sta-root', args.sta_root, '--image', args.image,
                '--output-dir', out_dir, '--sta-outputs']
        if not args.crop:
            argv.append('--no-crop')
        if args.restrict_charset:
            argv.append('--restrict-charset')
        if args.ocr_cache:
            argv += ['--ocr-cache', args.ocr_cache]
        if args.symbol_conf is not None:
            argv += ['--symbol-conf', str(args.symbol_conf)]
        if args.cost_cap is not None:
            argv += ['--cost-cap', str(args.cost_cap)]
        if args.symbol_scale is not None:
            argv += ['--symbol-scale', str(args.symbol_scale)]
        run('1/5  STA symbols + OCR', sta_python, 'tools.sta_bridge.export', argv)

    # --- 2. translate to this repo's request schema -----------------------
    run('2/5  adapt', pid_python, 'tools.sta_bridge.adapt',
        ['--export', os.path.join(out_dir, 'sta_export.json'),
         '--output', os.path.join(out_dir, 'text_detection.json')])

    # --- 3+4. line detection and graph construction -----------------------
    argv = ['--text-detection', os.path.join(out_dir, 'text_detection.json'),
            '--image', os.path.join(out_dir, 'diagram.png'),
            '--output-dir', out_dir, '--pid-id', name]
    for flag, val in (('--hough-threshold', args.hough_threshold),
                      ('--hough-min-line-length', args.hough_min_line_length),
                      ('--hough-max-line-gap', args.hough_max_line_gap),
                      ('--hough-rho', args.hough_rho),
                      ('--hough-theta', args.hough_theta),
                      ('--thinning-iterations', args.thinning_iterations),
                      ('--thin-min-stroke-width', args.thin_min_stroke_width),
                      ('--binary-threshold', args.binary_threshold),
                      ('--line-backend', args.line_backend),
                      ('--fld-length', args.fld_length),
                      ('--fld-distance', args.fld_distance),
                      ('--symbol-mask-inset', args.symbol_mask_inset),
                      ('--box-mask-inset', args.box_mask_inset),
                      ('--text-mask-inset-x', args.text_mask_inset_x),
                      ('--text-mask-inset-y', args.text_mask_inset_y),
                      ):
        if val is not None:
            argv += [flag, str(val)]
    if args.no_thinning:
        argv.append('--no-thinning')
    if args.dedup_segments:
        argv.append('--dedup-segments')
    if args.fld_merge:
        argv.append('--fld-merge')
    if args.classify_line_types:
        argv.append('--classify-line-types')
    if args.all_symbols_as_assets:
        argv.append('--all-symbols-as-assets')
    if args.drop_boxed_segments:
        argv.append('--drop-boxed-segments')
    if args.associate_leftover_text:
        argv.append('--associate-leftover-text')
    if args.leftover_text_distance is not None:
        argv += ['--leftover-text-distance', str(args.leftover_text_distance)]
    if args.exclude_dashed:
        argv.append('--exclude-dashed')
    run('3/5  line detection + 4/5  graph construction', pid_python,
        'tools.sta_bridge.run_local', argv,
        env={'DEBUG': 'true' if args.debug else 'false'})

    # --- 5. viewer --------------------------------------------------------
    run('5/5  viewer', pid_python, 'tools.sta_bridge.viewer',
        ['--output-dir', out_dir,
         '--title', args.title or f'{name} Connectivity'])

    print(f'\n\033[1mdone\033[0m  {out_dir}')
    print(f'  open {os.path.join(out_dir, "graph_viewer.html")}')


if __name__ == '__main__':
    main()
