# P&ID digitization — offline pipeline

Turns a scanned piping and instrumentation diagram into a graph: symbols, the
text that names them, the pipe runs, and what connects to what.

This started as a Microsoft sample (see `LICENSE.md`), whose line detection
and graph construction are still the core of it. The Azure
web service, the queue consumer and the SQL graph store have been taken out —
everything here runs locally against files on disk. Symbol detection and OCR
come from a separate STA project's RF-DETR + PaddleOCR pipeline, vendored
under `tools/sta_bridge/` so this repo runs on its own.

```
image ──▶ symbols + text ──▶ association ──▶ line detection ──▶ graph ──▶ viewer
          RF-DETR/OCR         Hungarian       FLD + thinning     crossings
          or a VLM            or the VLM
```

## Install

Two virtualenvs, because the two halves need incompatible pydantic versions.

```bash
python -m venv .venv-pid && .venv-pid/bin/pip install -r src/requirements.txt
# and STA's, holding torch + rfdetr + paddleocr:
python -m venv /path/to/STA/.venv-detect
```

The detector checkpoint (128 MB) belongs at `tools/sta_bridge/model/checkpoint_best_total.pth`.

## Run one sheet

```bash
python -m tools.sta_bridge.run_all --image sheet.png --name sheet
```

That drives five stages, each in the venv that suits it, and leaves everything
in `out/sheet/` — STA's own overlays included. Open `graph_viewer.html`.

A PDF page first:

```bash
pdftoppm -f 61 -l 61 -r 230 -png -singlefile doc.pdf out/sheet/diagram
```

230 dpi is not arbitrary: the detector's scale probe, the dashed-line
thresholds and the slice size are all tuned around a ~3900px-wide sheet.

## The flags that matter

`run_all` passes these through to `run_local`, or use `run_local` directly when
symbols and text are already extracted.

| flag | what it does |
|---|---|
| `--strip-red` | Lift a red overlay out first, saving it as its own layer. A revision cloud left in is found as pipe. |
| `--no-text-mask` | Leave text in the image for line detection. A pipe under a label is otherwise cut in two — worth 32% more graph edges on the sheets tried here. |
| `--axis-aligned-only` | Keep only horizontal and vertical segments (±5°). |
| `--drop-boxed-segments` | Keep segments lying entirely inside a symbol or text box out of the graph. |
| `--intersection-graph` | Also build the crossing-based graph (below). |
| `--associate-leftover-text` | Attach unclaimed labels to a nearby line or symbol. |

A good default for a clean drawing:

```bash
python -m tools.sta_bridge.run_local \
    --text-detection out/sheet/text_detection.json \
    --image out/sheet/diagram.png --output-dir out/sheet \
    --box-mask-inset 2 --drop-boxed-segments --associate-leftover-text \
    --axis-aligned-only --no-text-mask --intersection-graph
```

## Two routes to the graph

**Candidate matching** — the original. Every line segment's two endpoints each
get one candidate: a symbol, a text, or another segment. The graph is then a
walk over those links. It is precise where it fires, but one candidate per
endpoint means a pipe cannot both stop at a valve and carry on past it, and
about three quarters of the segments end up in no connection at all.

**Crossings** (`--intersection-graph`) — stitches the axis-aligned segments back
into whole pipe runs, then records every point where two runs cross or a run
meets a symbol box. A run passes straight through a valve and the valve is
recorded separately, so the two things stop competing.

Neither route dominates. On one fire-protection sheet the first connected
153 of 179 symbols and the second 132 of 166, but the first used 267 of 2155
detected segments against the second's 206 runs stitched from all of them, and
the second found 388 symbol-to-symbol links against 300. Both write JSON and a
single-file HTML viewer; nothing downstream has to choose.

## Tools

Each is `python -m tools.sta_bridge.<name> --help`.

| | |
|---|---|
| `run_all` | image → viewer, the whole chain |
| `run_local` | line detection + graph, when symbols and text exist |
| `export` | STA symbol detection + OCR → `sta_export.json` |
| `adapt` | `sta_export.json` → the request payload the pipeline reads |
| `intersection_graph` | the crossing-based graph |
| `intersection_viewer` | its single-file viewer |
| `viewer` | the viewer for the candidate-matching graph |
| `draw_connections` | static overlays: edges, orphaned segments |
| `split_red` / `merge_layers` | split a red overlay off; join the two layers back |
| `paint_red` | colour part of a drawing red, to test that round trip |
| `text_crops` | crop each OCR region, bounded by the nearest pipe runs |
| `crop_symbols` / `refine_crop_symbols` | re-run detection on those crops to find missed symbols |
| `promote_arrows` | relabel flow arrows that are really valves |
| `vlm_tiles` | slice the sheet and read each tile with a VLM instead |
| `compare_detections` | score two readers against each other |

## Reading the sheet with a VLM

`vlm_tiles` replaces RF-DETR and PaddleOCR with a vision model, tile by tile.
Same shape — slice, read, map back, merge with class-agnostic NMS — and it
writes the same `text_detection.json`, so line detection cannot tell which
reader produced it.

```bash
python -m tools.sta_bridge.vlm_tiles slice --image sheet.png --output-dir out/vlm
# read out/vlm/tiles/*.png with the prompt in out/vlm/PROMPT.md,
# write each answer to out/vlm/answers/<tile>.json
python -m tools.sta_bridge.vlm_tiles collect --output-dir out/vlm
```

On one fire-protection sheet, of 98 symbols both readers tagged, the tags
agreed on 89. All eleven disagreements were detector misreadings — `ZS 180`
read as `SZ 180`, `ZS 190` as `23 10`, a room name attached as a tag. How many
symbols each reader finds depends heavily on how the prompt is worded; saying
plainly that most symbols carry no tag took one tile from 45 symbols to 74.
`compare_detections` prints the disagreement and draws it.

## Layout

```
src/app/services/line_detection/     preprocessing, FLD/LSD/Hough, dedup, dashed
src/app/services/graph_construction/ candidates, traversal, flow direction
tools/sta_bridge/                    everything above, plus the vendored STA pipeline
Data/                                PID2Graph, for training experiments
out/                                 every run, one folder each
out/_removed/                        the Azure and SQL code, moved aside
```

`tools/sta_bridge/README.md` goes into the bridge in detail.

## Known limits

- Flow direction stays `unknown`: STA's 32 classes hold no Equipment or
  Pagination symbol, and that is what the propagation keys on.
- A crossing is geometric. Two pipes crossing without joining and a real tee
  look the same to it; only the T-junctions (one run ending on another) are
  certain.
- `dataset.yaml` has duplicate class names — ids 3 and 5 are both
  `Globe_valve_NO`, and 26/27, 28/31, 30/32 collide. One trained class is
  reported under another's name, which is part of why gate and globe valves
  swap labels between instances.
- Symbol detection runs on CPU here. A sheet takes about five minutes, most of
  it OCR.

## Licence

MIT, from the upstream project. See `LICENSE.md`.
