"""Build a pipeline export from STA's own result files, without re-running it.

``export.py`` drives STA itself, which means importing torch, RF-DETR and
PaddleOCR and paying for a detection pass. When STA has already been run and
its results are on disk, this reads them instead: no STA import, no model, no
GPU -- it runs in this repo's own venv in about a second.

    STA-main/results/18/associations.json   symbols + their tags
    STA-main/results/ocr_cache_18.json      the full OCR list
    STA-main/samples/18.png                 the sheet those coordinates describe

Output is byte-for-byte the same contract ``export.py`` writes, so ``adapt``,
``run_local`` and ``viewer`` take it unchanged.

One thing is lost by construction: ``associations.json`` records only symbols
that won a tag, so symbols STA detected but could not tag are absent. They were
never going to become graph nodes (graph construction drops untagged symbols),
but they would have been masked out before Hough -- so without them a few
symbol outlines survive into line detection as spurious segments. Use
``export.py`` when that matters.

Usage:
    python -m tools.sta_bridge.from_results \
        --results-dir /home/rx/project/STA-main/results/18 \
        --image       /home/rx/project/STA-main/samples/18.png \
        --ocr-cache   /home/rx/project/STA-main/results/ocr_cache_18.json \
        --output-dir  out/18-latest
"""
import argparse
import json
import os
import shutil
import struct


def png_size(path):
    """Width and height from the IHDR chunk, without decoding the image."""
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return struct.unpack('>II', head[16:24])


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', required=True,
                   help="an STA results folder, e.g. STA-main/results/18")
    p.add_argument('--image', required=True,
                   help='the sheet the results were produced from')
    p.add_argument('--ocr-cache',
                   help='full OCR list. Without it only the tag boxes are known, '
                        'which leaves most text unmasked before line detection.')
    p.add_argument('--associations',
                   help='default: <results-dir>/associations.json')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    assoc_path = args.associations or os.path.join(args.results_dir, 'associations.json')
    with open(assoc_path) as f:
        assoc = json.load(f)

    size = png_size(args.image)
    if size is None:
        raise SystemExit(f'ERROR: {args.image} is not a PNG; point --image at the '
                         f'sheet STA ran on.')
    width, height = size

    # STA's crop_diagram() uses a slice fixed to 7168x4562 sheets, so a cropped
    # run and a --no-crop run put the same symbol in different places. The
    # extents say which one produced these results.
    xs = [v for a in assoc for v in (a['symbol_bbox'][0], a['symbol_bbox'][2])]
    ys = [v for a in assoc for v in (a['symbol_bbox'][1], a['symbol_bbox'][3])]
    if max(xs) > width or max(ys) > height:
        raise SystemExit(
            f'ERROR: symbol coordinates reach ({max(xs)}, {max(ys)}) but {args.image} '
            f'is {width}x{height}. These results came from a different sheet, or from '
            f'a run whose crop this image does not match.')

    symbols = [
        {
            'id': int(a['symbol_id']),
            'class': a['class'],
            'score': float(a['symbol_confidence']),
            'bbox': [int(v) for v in a['symbol_bbox']],
            'tag': a['tag'],
        }
        for a in assoc
    ]

    if args.ocr_cache and os.path.exists(args.ocr_cache):
        with open(args.ocr_cache) as f:
            cache = json.load(f)
        texts = [{'text': t['text'], 'score': float(t['score']),
                  'bbox': [int(v) for v in t['bbox']]} for t in cache]
        tx = [v for t in texts for v in (t['bbox'][0], t['bbox'][2])]
        ty = [v for t in texts for v in (t['bbox'][1], t['bbox'][3])]
        if max(tx) > width or max(ty) > height:
            raise SystemExit(
                f'ERROR: OCR cache coordinates reach ({max(tx)}, {max(ty)}), past '
                f'{width}x{height}. The cache is from a run with a different crop.')
        source = os.path.basename(args.ocr_cache)
    else:
        # Fall back to the tag boxes. Line detection masks text before Hough, so
        # everything not listed here stays in the image as strokes.
        texts = [{'text': a['tag'], 'score': float(a['ocr_confidence']),
                  'bbox': [int(v) for v in a['tag_bbox']]} for a in assoc]
        source = 'tag boxes only'

    os.makedirs(args.output_dir, exist_ok=True)
    diagram_path = os.path.join(args.output_dir, 'diagram.png')
    shutil.copyfile(args.image, diagram_path)

    export = {
        'image': 'diagram.png',
        'image_width': width,
        'image_height': height,
        'source_image': os.path.abspath(args.image),
        'cropped': False,
        'symbols': symbols,
        'texts': texts,
    }
    export_path = os.path.join(args.output_dir, 'sta_export.json')
    with open(export_path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f'sheet    {width}x{height}  ({args.image})')
    print(f'symbols  {len(symbols)} from {assoc_path}')
    print(f'texts    {len(texts)} from {source}')
    if source == 'tag boxes only':
        print('         warning: no --ocr-cache, so untagged text is not masked '
              'and will be picked up as line segments')
    print(f'wrote    {diagram_path}')
    print(f'wrote    {export_path}')


if __name__ == '__main__':
    main()
