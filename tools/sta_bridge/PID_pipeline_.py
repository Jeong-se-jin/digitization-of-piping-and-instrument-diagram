"""
P&ID Symbol Detection and OCR Pipeline
=======================================
Detects engineering symbols and associates OCR text (equipment tags)
with detected symbols on a P&ID diagram image.

Usage:
    python pid_pipeline_.py --image path/to/diagram.jpg --yaml dataset.yaml --weights model/checkpoint_best_total.pth

Dependencies:
    pip install paddleocr rfdetr sahi scipy opencv-python-headless pillow pyyaml matplotlib
"""

import argparse
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from paddleocr import PaddleOCR
from rfdetr import RFDETRBase
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from scipy.optimize import linear_sum_assignment
# ----
import os
import json
import collections
import statistics
import pandas as pd

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OCR quality thresholds
OCR_DET_THRESH = 0.75
OCR_REC_THRESH = 0.75

# Tiling parameters
TILE_SIZE = 1280
TILE_OVERLAP = 640

# Duplicate-detection thresholds (clustering step)
IOU_THRESHOLD = 0.20
CONTAINMENT_THRESHOLD = 0.60

# Line-merging thresholds (join vertically adjacent OCR boxes)
MERGE_VERTICAL_GAP_PX = 10
MERGE_X_OVERLAP_RATIO = 0.80

# Merging exists to rebuild an instrument tag split across the two lines of a
# bubble ("TE" over "204B").  Geometry alone also glues things that are not one
# label - the rows of a valve list, the lines of a note - so two content guards
# bound it: a group larger than this many lines is left unmerged, and a line
# longer than this many characters is never a tag fragment.
MERGE_MAX_GROUP_LINES = 3
MERGE_MAX_PART_CHARS = 12

# How far the detector dilates a text region before boxing it.  PP-OCR's
# default (~1.5-2.0) bridges neighbouring cells on a shared baseline, so
# "NOTE 18 | OP | PMS" arrives as one line.  Lowering it keeps cells apart at
# the cost of clipping wide-spaced text.  None keeps the model default.
OCR_DET_UNCLIP_RATIO = None

# Symbol detection.
# The checkpoint was trained on 7168 px synthetic sheets where a valve spans
# ~75 px; on these 3916 px drawings the same valve spans ~41 px, and the 1280 px
# slice is then resized to the model's 560 px input, leaving ~18 px of valve.
# That is a scale mismatch, not a style mismatch - the same weights find 62
# valves on samples/default/100.jpg.  Rescaling the input by the measured ratio
# (75/41 = 1.8) restores the training scale and the model becomes confident
# again: valves go from 2 to 24 at this very cut-off, so the threshold does not
# need to be lowered to admit low-confidence guesses.
# Measured on samples/default/100.jpg, a sheet from the training benchmark.
SYMBOL_TRAINING_VALVE_WIDTH = 75

# Guards for the automatic scale: too few valves in the probe means the estimate
# has no basis, so fall back to 1.0 rather than guess, and never enlarge beyond
# SYMBOL_SCALE_MAX (slice count grows with the square of the scale).
SYMBOL_SCALE_PROBE_CONFIDENCE = 0.20
SYMBOL_SCALE_MIN_PROBE_VALVES = 8
SYMBOL_SCALE_MAX = 4.0

SYMBOL_UPSCALE = 1.8
SYMBOL_DETECTION_CONFIDENCE = 0.30

# SAHI de-duplicates per class, so one valve detected as both Ball_Valve and
# Globe_valve_NO survives as two boxes with IoU 1.00.  A P&ID holds one symbol
# per position, so overlapping detections are merged regardless of class - and
# the duplicates matter: the Hungarian assignment is one-to-one, so two boxes on
# one valve compete for its single tag and one of them is always left unlabelled.
SYMBOL_NMS_IOU = 0.5
SYMBOL_SLICE_SIZE = 1280
SYMBOL_SLICE_OVERLAP = 0.2

# OCR input upscaling (applied to the OCR path only, never to symbol detection)
OCR_UPSCALE = 1.0

# Characters that occur on an engineering drawing.  PP-OCRv5 ships an
# 18385-character multilingual dictionary, so a line-art fragment can be
# decoded as a CJK glyph or an emoji; restricting the decoder removes that
# whole failure class.  Applied only when --restrict-charset is passed.
OCR_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,:;-_/()[]%+*=<>#&@'\"°ØΦ"
)

# Association.
# _association_cost stacks flag bonuses far larger than the distance term:
#   text inside the symbol  -1000     valid tag  -200  (invalid +300)
#   two or more characters   -600     single char +300
# so the plausible combinations land at roughly
#   inside + valid + multi   ~ -1800     inside + invalid + multi ~ -1300
#   outside + valid + multi  ~  -800     outside + invalid + multi ~ -300
# The old cap of 5000 sat above every reachable value and never rejected
# anything, so each symbol received a label whether or not it deserved one.
# -500 requires a pair to be either enclosed by the symbol or a genuine
# multi-character tag.
ASSOCIATION_COST_CAP = -500
ASSOCIATION_SEARCH_MARGIN = 20

# How much of a text box must fall inside a symbol to count as enclosed.
# Strict containment made the -1000 bonus a cliff: an instrument box 58 px tall
# whose three-line label starts 8 px above it scored -279 instead of -1800 and
# was rejected, even though the label sits on the symbol.
ASSOCIATION_CONTAINMENT_RATIO = 0.75

# Visualisation weights.  Instrument text on these drawings is about 25 px
# tall, so the label must stay well under that or it hides the drawing it is
# annotating - the original scale of 1.2 with a 3 px stroke did exactly that.
VIZ_FONT_SCALE = 0.32
VIZ_FONT_THICKNESS = 1
VIZ_BOX_THICKNESS = 1

# Valid equipment-tag pattern  (e.g. FIC-101, XV-12A, "TE 125A")
# The separator accepts a space because merge_adjacent_lines joins stacked
# boxes with one - "TE" over "125A" arrives here as "TE 125A".
TAG_REGEX = re.compile(r"^[A-Z]{1,5}[ -]?\d{1,5}[A-Z]?$")

# Glyph pairs a recogniser confuses on small text.  Font-independent prior -
# which of these actually apply is decided per drawing by infer_tag_grammar.
DIGIT_LOOKALIKE_LETTERS = {
    "0": "DOQ", "1": "ILT", "2": "Z", "5": "S", "6": "G", "8": "B",
}
_TAGLIKE_REGEX = re.compile(r"^([A-Z]{1,5})([ -]?)(\d+)([A-Z]?)$")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from disk and return as an RGB NumPy array."""
    img = Image.open(path)
    return np.array(img)


def crop_diagram(image: np.ndarray) -> np.ndarray:
    """
    Crop away the drawing border / title block.
    Adjust the slice values to match your document layout.

    NOTE: these slice values are specific to the 7168x4562 sample sheets, where
    the right-hand 1500 px hold the title block.  Images that already contain
    only the drawing (e.g. figures extracted from a PDF) must be passed through
    unchanged - use --no-crop.
    """
    return image[160:-160, 290:-1500]


def upscale_for_ocr(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Enlarge *image* by *scale* for the OCR pass only.

    Engineering drawings scanned at ~230 dpi render instrument tags about 10 px
    tall, well below the text height PP-OCRv5 recognition expects.  Upscaling
    trades tile count (which grows with scale^2) for recognition accuracy.
    """
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LANCZOS4,
    )


def rescale_detections(detections: list[dict], scale: float) -> list[dict]:
    """Map OCR detections from the upscaled image back to original coordinates."""
    if scale == 1.0:
        return detections
    for d in detections:
        d["bbox"] = [int(round(v / scale)) for v in d["bbox"]]
        x1, y1, x2, y2 = d["bbox"]
        d["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return detections


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def generate_tiles(image: np.ndarray, tile_size: int, overlap: int) -> list[dict]:
    """
    Divide *image* into overlapping tiles of *tile_size* × *tile_size* pixels.
    The last column and row of tiles are snapped to the image edge so no pixels
    are missed.

    Returns a list of dicts with keys: image, x_offset, y_offset.
    """
    h, w = image.shape[:2]
    step = tile_size - overlap

    def _anchors(dim: int) -> list[int]:
        positions = list(range(0, max(1, dim - tile_size + 1), step))
        # Ensure the last tile always reaches the edge
        last = max(0, dim - tile_size)
        if not positions or positions[-1] != last:
            positions.append(last)
        return positions

    tiles = []
    for y in _anchors(h):
        for x in _anchors(w):
            tiles.append({
                "image": image[y : y + tile_size, x : x + tile_size],
                "x_offset": x,
                "y_offset": y,
            })
    return tiles


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def build_ocr_engine(det_unclip_ratio: float = OCR_DET_UNCLIP_RATIO) -> PaddleOCR:
    """Initialise PaddleOCR with the project settings."""
    options = {}
    if det_unclip_ratio is not None:
        options["text_det_unclip_ratio"] = det_unclip_ratio
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        #text_det_thresh=OCR_DET_THRESH,
        #text_rec_score_thresh=OCR_REC_THRESH,
        **options,
    )


def restrict_ocr_charset(ocr_engine: PaddleOCR, allowed: str = OCR_CHARSET) -> int:
    """
    Constrain the recogniser's CTC decoder to *allowed* characters.

    The decoder picks the arg-max over its full dictionary, so a glyph the
    drawing never contains can win: "V172B" comes back as "V172\u65e5".  Masking
    the disallowed columns to zero probability before the arg-max keeps the
    blank symbol (index 0) and lets the best *plausible* character win instead.

    This restricts decoding only - the weights are untouched, so a character
    the model was never trained to emit here cannot be recovered by it.

    Returns the number of dictionary entries left enabled.
    """
    from paddlex.inference.models.text_recognition import processors as _proc

    decoders = []

    def _walk(obj, depth=0, seen=None):
        seen = set() if seen is None else seen
        if depth > 7 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, _proc.CTCLabelDecode):
            decoders.append(obj)
            return
        children = obj if isinstance(obj, (list, tuple)) else \
            (vars(obj).values() if hasattr(obj, "__dict__") else [])
        for child in children:
            if isinstance(child, (list, tuple)) or hasattr(child, "__dict__"):
                _walk(child, depth + 1, seen)

    _walk(ocr_engine)
    if not decoders:
        raise RuntimeError("could not reach the CTC decoder to restrict its charset")

    if not getattr(_proc.CTCLabelDecode, "_charset_patched", False):
        _original_call = _proc.CTCLabelDecode.__call__

        def _masked_call(self, pred, *args, **kwargs):
            mask = getattr(self, "_allowed_mask", None)
            if mask is not None:
                probs = np.array(pred[0], copy=True)
                probs[..., ~mask] = 0.0
                pred = [probs, *list(pred[1:])]
            return _original_call(self, pred, *args, **kwargs)

        _proc.CTCLabelDecode.__call__ = _masked_call
        _proc.CTCLabelDecode._charset_patched = True

    permitted = set(allowed)
    enabled = 0
    for decoder in decoders:
        mask = np.zeros(len(decoder.character), dtype=bool)
        mask[0] = True                              # CTC blank must survive
        for i, ch in enumerate(decoder.character):
            if ch in permitted:
                mask[i] = True
        decoder._allowed_mask = mask
        enabled = int(mask.sum())

    return enabled


def run_ocr_on_tiles(tiles: list[dict], ocr_engine: PaddleOCR) -> list[dict]:
    """
    Run OCR on every tile and return a flat list of detections, each with
    absolute coordinates in the full-image space.
    """
    detections = []
    for tile in tqdm(
            tiles,
            desc="OCR tiles",
            unit="tile"):
        result = ocr_engine.predict(tile["image"])
        x_off, y_off = tile["x_offset"], tile["y_offset"]

        for txt, score, poly in zip(
            result[0]["rec_texts"],
            result[0]["rec_scores"],
            result[0]["rec_polys"],
        ):
            poly = np.array(poly)
            x1, y1 = int(poly[:, 0].min()), int(poly[:, 1].min())
            x2, y2 = int(poly[:, 0].max()), int(poly[:, 1].max())

            detections.append({
                "text": txt,
                "score": float(score),
                "bbox": [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off],
                "center": ((x1 + x2) / 2 + x_off, (y1 + y2) / 2 + y_off),
            })

    return detections


# ---------------------------------------------------------------------------
# OCR post-processing: remove duplicates produced by tile overlap
# ---------------------------------------------------------------------------

def _bbox_iou(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB - xA) * (yB - yA)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _containment(small: list, large: list) -> float:
    """Fraction of *small* that is covered by *large*."""
    xA, yA = max(small[0], large[0]), max(small[1], large[1])
    xB, yB = min(small[2], large[2]), min(small[3], large[3])
    if xB <= xA or yB <= yA:
        return 0.0
    area_small = max(0, small[2] - small[0]) * max(0, small[3] - small[1])
    return (xB - xA) * (yB - yA) / area_small if area_small else 0.0


def _are_same_region(a: dict, b: dict) -> bool:
    iou = _bbox_iou(a["bbox"], b["bbox"])
    c1 = _containment(a["bbox"], b["bbox"])
    c2 = _containment(b["bbox"], a["bbox"])
    return iou > IOU_THRESHOLD or c1 > CONTAINMENT_THRESHOLD or c2 > CONTAINMENT_THRESHOLD


def _connected_components(n: int, neighbours: list[list[int]]) -> list[list[int]]:
    visited = [False] * n
    groups = []
    for start in range(n):
        if visited[start]:
            continue
        stack, component = [start], []
        while stack:
            k = stack.pop()
            if visited[k]:
                continue
            visited[k] = True
            component.append(k)
            stack.extend(neighbours[k])
        groups.append(component)
    return groups


def deduplicate_ocr(detections: list[dict]) -> list[dict]:
    """
    Cluster overlapping detections (artefacts of tiled inference) and keep
    the single best detection per cluster — the one with the highest
    score × text-length product.
    """
    n = len(detections)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _are_same_region(detections[i], detections[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)

    groups = _connected_components(n, neighbours)
    cleaned = []
    for group in groups:
        best = max(
            (detections[i] for i in group),
            key=lambda d: len(d["text"]) * d.get("score", 1.0),
        )
        cleaned.append(best)

    return cleaned


# ---------------------------------------------------------------------------
# OCR post-processing: merge vertically adjacent boxes into single labels
# ---------------------------------------------------------------------------

def _vertical_gap(a: dict, b: dict) -> int:
    """Pixel gap between the bottom of the upper box and the top of the lower box."""
    if a["bbox"][1] > b["bbox"][1]:
        a, b = b, a
    return b["bbox"][1] - a["bbox"][3]


def _x_overlap_ratio(a: dict, b: dict) -> float:
    ax1, ax2 = a["bbox"][0], a["bbox"][2]
    bx1, bx2 = b["bbox"][0], b["bbox"][2]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    min_width = min(ax2 - ax1, bx2 - bx1)
    return overlap / min_width if min_width else 0.0


def _should_merge(a: dict, b: dict) -> bool:
    if not (_vertical_gap(a, b) <= MERGE_VERTICAL_GAP_PX
            and _x_overlap_ratio(a, b) > MERGE_X_OVERLAP_RATIO):
        return False

    ta, tb = a["text"].strip(), b["text"].strip()

    # A string that is already a complete tag is a finished label; gluing
    # anything to it only destroys it.  "1\" BBB L102D" over "V102D" merged into
    # one 17-character cell, and the valve underneath could then no longer claim
    # its own tag because the merged string matches no tag pattern.
    if _is_valid_tag(ta) or _is_valid_tag(tb):
        return False

    # A line this long is prose or a full designation, never half a tag.
    if len(ta) > MERGE_MAX_PART_CHARS or len(tb) > MERGE_MAX_PART_CHARS:
        return False

    return True


def merge_adjacent_lines(detections: list[dict]) -> list[dict]:
    """Merge vertically adjacent OCR boxes that belong to the same label."""
    n = len(detections)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(detections[i], detections[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)

    groups = _connected_components(n, neighbours)
    merged = []
    for group in groups:
        boxes = [detections[i] for i in group]

        # A run this long is a table or a paragraph; keep its lines separate
        # rather than concatenating them into one unusable string.
        if len(boxes) > MERGE_MAX_GROUP_LINES:
            merged.extend(boxes)
            continue

        x1 = min(b["bbox"][0] for b in boxes)
        y1 = min(b["bbox"][1] for b in boxes)
        x2 = max(b["bbox"][2] for b in boxes)
        y2 = max(b["bbox"][3] for b in boxes)
        # Sort top-to-bottom, left-to-right before joining text
        boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        merged.append({
            "text": " ".join(b["text"] for b in boxes),
            "score": max(b["score"] for b in boxes),
            "bbox": [x1, y1, x2, y2],
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "count": len(boxes),
        })
    return merged


# ---------------------------------------------------------------------------
# OCR post-processing: repair known recognition confusions
# ---------------------------------------------------------------------------

def infer_tag_grammar(detections: list[dict]) -> tuple[int, dict[str, str]]:
    """
    Work out this drawing's tag numbering convention from the drawing itself.

    Tag conventions differ per drawing set; the authoritative source is the
    project's P&ID legend sheet, which is usually a separate drawing.  Absent
    that, the sheet calibrates itself:

    1. The suffix alphabet is whatever trailing letters actually occur on
       tag-shaped strings (here A-D, one per reactor coolant loop).
    2. The loop-number width is the most common digit count among tags that
       *do* carry a suffix - the subset least likely to hide a misread suffix.
    3. A digit is treated as a misread suffix only when its lookalike letters
       intersect the suffix alphabet in exactly one letter.  On a sheet whose
       suffixes are A/B only, "0" resolves to nothing and is left alone.

    Returns (numeric_width, {trailing_digit: letter}).  Width 0 disables repair.
    """
    suffixes, widths = collections.Counter(), collections.Counter()
    for d in detections:
        m = _TAGLIKE_REGEX.fullmatch(" ".join(d["text"].split()))
        if not m:
            continue
        _, _, digits, suffix = m.groups()
        if suffix:
            suffixes[suffix] += 1
            widths[len(digits)] += 1

    if not widths:
        return 0, {}

    width = widths.most_common(1)[0][0]
    alphabet = set(suffixes)
    mapping = {}
    for digit, lookalikes in DIGIT_LOOKALIKE_LETTERS.items():
        candidates = alphabet.intersection(lookalikes)
        if len(candidates) == 1:                  # ambiguous digits stay digits
            mapping[digit] = candidates.pop()

    return width, mapping


def repair_tag_text(detections: list[dict]) -> tuple[list[dict], int]:
    """
    Collapse whitespace and undo suffix letters the recogniser read as digits.

    A tag-shaped string carrying one digit more than the drawing's loop-number
    width and no suffix letter is a misread suffix: "TE 1318" is "TE 131B".
    Strings that already end in a letter are left alone, so "PIC 1400F" is
    never touched.

    The grammar comes from infer_tag_grammar, so nothing here is specific to
    one drawing.  The original string is preserved in "text_raw" so every
    rewrite stays auditable in the exported results.  Returns the detections
    and the number of suffix repairs made.
    """
    width, mapping = infer_tag_grammar(detections)
    repaired = 0
    for d in detections:
        d["text_raw"] = d["text"]
        text = " ".join(d["text"].split())

        m = _TAGLIKE_REGEX.fullmatch(text)
        if m and width:
            head, sep, digits, suffix = m.groups()
            if (not suffix
                    and len(digits) == width + 1
                    and digits[-1] in mapping):
                text = head + sep + digits[:-1] + mapping[digits[-1]]
                repaired += 1

        d["text"] = text

    return detections, repaired


# ---------------------------------------------------------------------------
# Symbol detection
# ---------------------------------------------------------------------------

def load_class_mapping(yaml_path: str) -> dict[int, str]:
    """Load the id→class-name mapping from a YOLO dataset YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["names"].items()}


def resolve_device(requested: str = None) -> str:
    """Pick a torch device: the requested one, else CUDA when it is usable."""
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def build_detection_model(
    weights_path: str,
    id_to_name: dict,
    device: str = None,
    confidence: float = SYMBOL_DETECTION_CONFIDENCE,
) -> AutoDetectionModel:
    model = RFDETRBase(pretrain_weights=weights_path)
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model=model,
        confidence_threshold=confidence,
        category_mapping=id_to_name,
        device=resolve_device(device),
    )


def infer_symbol_scale(
    image: np.ndarray,
    weights_path: str,
    id_to_name: dict,
    device: str = None,
) -> float:
    """
    Work out how much to enlarge *image* so its symbols reach training scale.

    The same drawing arrives at different sizes - a figure extracted from a PDF
    fills its canvas, the same figure rendered as a full page occupies about two
    thirds of it - so a fixed factor is right for one input and wrong for the
    other.  A cheap low-confidence probe measures the median valve width on this
    particular image and the factor follows from the training width.

    Valves are the yardstick because they are the smallest frequent symbol and
    the first to be lost; instrument bubbles survive a scale mismatch that
    already hides every valve.  Returns 1.0 when the probe finds too few to
    measure.
    """
    probe_model = build_detection_model(
        weights_path, id_to_name, device, SYMBOL_SCALE_PROBE_CONFIDENCE)
    result = get_sliced_prediction(
        image,
        probe_model,
        slice_height=SYMBOL_SLICE_SIZE,
        slice_width=SYMBOL_SLICE_SIZE,
        overlap_height_ratio=SYMBOL_SLICE_OVERLAP,
        overlap_width_ratio=SYMBOL_SLICE_OVERLAP,
        verbose=0,
    )

    widths = [
        obj.bbox.maxx - obj.bbox.minx
        for obj in result.object_prediction_list
        if "valve" in obj.category.name.lower()
    ]
    if len(widths) < SYMBOL_SCALE_MIN_PROBE_VALVES:
        print(f"      Scale probe found only {len(widths)} valves - keeping ×1.0")
        return 1.0

    measured = statistics.median(widths)
    scale = SYMBOL_TRAINING_VALVE_WIDTH / measured
    scale = round(min(max(scale, 1.0), SYMBOL_SCALE_MAX), 2)
    print(f"      Scale probe: {len(widths)} valves, median {measured:.0f} px "
          f"vs {SYMBOL_TRAINING_VALVE_WIDTH} px trained -> ×{scale}")
    return scale


def detect_symbols(
    image: np.ndarray,
    detection_model: AutoDetectionModel,
    scale: float = SYMBOL_UPSCALE,
) -> list[dict]:
    """
    Run sliced inference and return a flat list of symbol detections.

    *scale* enlarges the image so symbols reach the size the checkpoint was
    trained at (see SYMBOL_UPSCALE).  Detections are mapped back to original
    image coordinates before returning, so everything downstream - association,
    visualisation, the exports - stays in one coordinate system.
    """
    detect_input = image if scale == 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    result = get_sliced_prediction(
        detect_input,
        detection_model,
        slice_height=SYMBOL_SLICE_SIZE,
        slice_width=SYMBOL_SLICE_SIZE,
        overlap_height_ratio=SYMBOL_SLICE_OVERLAP,
        overlap_width_ratio=SYMBOL_SLICE_OVERLAP,
    )

    symbols = []
    for idx, obj in enumerate(result.object_prediction_list):
        b = obj.bbox
        x1, y1, x2, y2 = (int(round(v / scale))
                          for v in (b.minx, b.miny, b.maxx, b.maxy))
        symbols.append({
            "id": idx,
            "class": obj.category.name,
            "score": float(obj.score.value),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return symbols


def deduplicate_symbols(symbols: list[dict], iou_threshold: float = SYMBOL_NMS_IOU) -> list[dict]:
    """
    Drop overlapping symbol detections, keeping the most confident one.

    Class-agnostic on purpose: the duplicates this removes are the same symbol
    classified two different ways, which per-class NMS cannot see.  Ids are
    reassigned so they stay contiguous for the exports.
    """
    kept: list[dict] = []
    for symbol in sorted(symbols, key=lambda s: -s["score"]):
        if all(_bbox_iou(symbol["bbox"], k["bbox"]) <= iou_threshold for k in kept):
            kept.append(symbol)

    kept.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    for new_id, symbol in enumerate(kept):
        symbol["id"] = new_id
    return kept


# ---------------------------------------------------------------------------
# Symbol ↔ text association
# ---------------------------------------------------------------------------

def _expand_bbox(bbox: list, margin: int) -> list:
    x1, y1, x2, y2 = bbox
    return [x1 - margin, y1 - margin, x2 + margin, y2 + margin]


def _intersection_area(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    if xB <= xA or yB <= yA:
        return 0.0
    return float((xB - xA) * (yB - yA))


def _text_inside_symbol(sym_bbox: list, txt_bbox: list) -> bool:
    """True when most of the text area falls within the symbol - see
    ASSOCIATION_CONTAINMENT_RATIO for why this is a ratio and not containment."""
    tx1, ty1, tx2, ty2 = txt_bbox
    text_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
    if text_area <= 0:
        return False
    return _intersection_area(sym_bbox, txt_bbox) / text_area >= ASSOCIATION_CONTAINMENT_RATIO


def _is_valid_tag(text: str) -> bool:
    return TAG_REGEX.fullmatch(" ".join(text.split())) is not None


def _association_cost(symbol: dict, text_obj: dict) -> float:
    """
    Lower cost = better match.
    Returns a large sentinel (1e9) when the text is completely outside the
    symbol's search zone.
    """
    expanded = _expand_bbox(symbol["bbox"], ASSOCIATION_SEARCH_MARGIN)
    if _intersection_area(expanded, text_obj["bbox"]) == 0:
        return 1e9

    sx, sy = symbol["center"]
    tx, ty = text_obj["center"]
    dy = ty - sy
    cost = float(np.hypot(tx - sx, dy))

    if _text_inside_symbol(symbol["bbox"], text_obj["bbox"]):
        cost -= 1000           # strong bonus for enclosed text
    if _is_valid_tag(text_obj["text"]):
        cost -= 200            # bonus for tag-like strings
    else:
        cost += 300
    if len(text_obj["text"]) > 1:
        cost -= 600            # prefer multi-character labels
    else:
        cost += 300
    cost += abs(dy)            # prefer horizontally aligned text

    return cost


def associate_symbols_to_text(
    symbols: list[dict],
    texts: list[dict],
    cost_cap: float = ASSOCIATION_COST_CAP,
) -> list[dict]:
    """
    Use the Hungarian algorithm to find the globally optimal one-to-one
    assignment between detected symbols and OCR text boxes.

    Pairs whose optimal cost exceeds ASSOCIATION_COST_CAP are discarded.
    """
    n_sym, n_txt = len(symbols), len(texts)
    cost_matrix = np.zeros((n_sym, n_txt))
    for i, sym in enumerate(
            tqdm(
                symbols,
                desc="Building cost matrix",
                unit="symbol"
            )
    ):
        for j, txt in enumerate(texts):
            cost_matrix[i, j] = _association_cost(sym, txt)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    associations = []
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost > cost_cap:
            continue
        associations.append({
            "symbol_id": symbols[r]["id"],
            "class": symbols[r]["class"],
            "tag": texts[c]["text"],
            "cost": float(cost),
            # Keep full objects for visualisation
            "_symbol": symbols[r],
            "_text": texts[c],
        })

    return associations

def save_associations_excel(
    associations,
    output_file,
):

    rows = []

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        rows.append({

            "symbol_id":
                assoc["symbol_id"],

            "symbol_class":
                assoc["class"],

            "symbol_confidence":
                sym["score"],

            "symbol_bbox":
                str(sym["bbox"]),

            "tag":
                assoc["tag"],

            "tag_ocr_raw":
                txt.get("text_raw", txt["text"]),

            "ocr_confidence":
                txt["score"],

            "tag_bbox":
                str(txt["bbox"]),

            "association_cost":
                assoc["cost"],
        })

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Associations",
            index=False,
        )


def save_associations_json(
    associations,
    output_file,
):

    export = []

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        export.append({

            "symbol_id":
                assoc["symbol_id"],

            "class":
                assoc["class"],

            "symbol_confidence":
                sym["score"],

            "symbol_bbox":
                sym["bbox"],

            "tag":
                assoc["tag"],

            "tag_ocr_raw":
                txt.get("text_raw", txt["text"]),

            "ocr_confidence":
                txt["score"],

            "tag_bbox":
                txt["bbox"],

            "association_cost":
                assoc["cost"],
        })

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            export,
            f,
            indent=2,
        )
# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def visualise_symbols(
    image: np.ndarray,
    symbols: list[dict],
    save_path: str = None,
):

    viz = image.copy()

    for s in symbols:

        x1, y1, x2, y2 = s["bbox"]

        cv2.rectangle(
            viz,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            VIZ_BOX_THICKNESS + 1,
        )

        label = f"{s['class']} {s['score']:.2f}"

        cv2.putText(
            viz,
            label,
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            VIZ_FONT_SCALE,
            (255, 0, 0),
            VIZ_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(20,20))
    plt.imshow(viz)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    #plt.show()
    plt.close()


def visualise_ocr(
    image: np.ndarray,
    detections: list[dict],
    title: str = "OCR detections",
    save_path: str | None = None,
) -> None:

    viz = image.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), VIZ_BOX_THICKNESS)

        cv2.putText(
            viz,
            d["text"],
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            VIZ_FONT_SCALE,
            (0, 255, 0),
            VIZ_FONT_THICKNESS,
            cv2.LINE_AA
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(viz)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()


def visualise_associations(
    image: np.ndarray,
    associations: list[dict],
    save_path: str | None = None,
) -> None:

    viz = image.copy()

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        sx, sy = map(int, sym["center"])
        tx, ty = map(int, txt["center"])

        x1, y1, x2, y2 = sym["bbox"]
        tx1, ty1, tx2, ty2 = txt["bbox"]

        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), VIZ_BOX_THICKNESS + 1)
        cv2.rectangle(viz, (tx1, ty1), (tx2, ty2), (0, 255, 0), VIZ_BOX_THICKNESS)

        cv2.line(viz, (sx, sy), (tx, ty), (255, 200, 0), VIZ_BOX_THICKNESS)

        cv2.putText(
            viz,
            txt["text"],
            (sx, max(10, sy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            VIZ_FONT_SCALE,
            (200, 120, 0),
            VIZ_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(viz)
    plt.axis("off")
    plt.title("Final Associations")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P&ID OCR + Symbol Detection Pipeline")
    parser.add_argument("--image",   required=True, help="Path to the input P&ID image")
    parser.add_argument("--yaml",    required=True, help="Path to dataset.yaml (class names)")
    parser.add_argument("--weights", required=True, help="Path to RF-DETR checkpoint (.pth)")
    parser.add_argument("--no-viz",  action="store_true", help="Skip all visualisation windows")
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop the border/title block (use for images that already "
             "contain only the drawing, e.g. figures extracted from a PDF)",
    )
    parser.add_argument(
        "--ocr-cache",
        default=None,
        help="Path to a JSON cache of the consolidated OCR result.  Loaded when "
             "it exists (stages 2-3 are skipped), written when it does not.  "
             "Tuning symbol detection re-runs in seconds instead of minutes",
    )
    parser.add_argument(
        "--symbol-scale",
        default="auto",
        help='Enlarge the image by this factor for symbol detection only, or '
             '"auto" (default) to measure it from the drawing.  Detections are '
             'mapped back to original coordinates either way',
    )
    parser.add_argument(
        "--symbol-conf",
        type=float,
        default=SYMBOL_DETECTION_CONFIDENCE,
        help="Symbol detection confidence cut-off (default: %(default)s)",
    )
    parser.add_argument(
        "--cost-cap",
        type=float,
        default=ASSOCIATION_COST_CAP,
        help="Reject a symbol-text pair whose cost exceeds this "
             "(default: %(default)s; lower is stricter)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for symbol detection (default: cuda:0 when "
             "available, otherwise cpu)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep one detection per text line - skip merging vertically "
             "stacked boxes.  Note that a two-line instrument tag then stays "
             "split ('TE' and '204B' separately) and neither half matches the "
             "tag grammar",
    )
    parser.add_argument(
        "--det-unclip-ratio",
        type=float,
        default=OCR_DET_UNCLIP_RATIO,
        help="Detector dilation before boxing (default: model default). "
             "Lower values stop neighbouring cells on one baseline from being "
             "boxed as a single line",
    )
    parser.add_argument(
        "--restrict-charset",
        action="store_true",
        help="Constrain the OCR decoder to drawing characters (A-Z, 0-9 and "
             "punctuation), removing CJK/emoji misreads from the 18385-entry "
             "multilingual dictionary",
    )
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=OCR_UPSCALE,
        help="Upscale factor applied to the OCR pass only (default: %(default)s). "
             "Use ~2.0 on low-resolution scans; tile count grows with the square "
             "of this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading image and class mapping …")
    id_to_name = load_class_mapping(args.yaml)
    image = load_image(args.image)
    image_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    output_dir = os.path.join(
        "results",
        image_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"Output folder: {output_dir}"
    )
    if args.no_crop:
        diagram = image
        print("      Crop disabled (--no-crop): using the full image")
    else:
        diagram = crop_diagram(image)
    print(f"      Diagram size: {diagram.shape[1]}×{diagram.shape[0]} px")

    # ------------------------------------------------------------------
    # 2. OCR
    # ------------------------------------------------------------------
    if args.ocr_cache and os.path.exists(args.ocr_cache):
        print(f"[2-3/6] Reusing OCR cache: {args.ocr_cache}")
        with open(args.ocr_cache) as f:
            merged_texts = json.load(f)
        for d in merged_texts:                       # json turns tuples into lists
            d["center"] = tuple(d["center"])
        print(f"      Texts loaded: {len(merged_texts)}")
    else:
        merged_texts = None

    if merged_texts is None:
      print("[2/6] Running OCR …")
      ocr_engine = build_ocr_engine(args.det_unclip_ratio)
      if args.restrict_charset:
          n_chars = restrict_ocr_charset(ocr_engine)
          print(f"      Decoder charset restricted to {n_chars} entries")
      ocr_input = upscale_for_ocr(diagram, args.ocr_scale)
      if args.ocr_scale != 1.0:
          print(
              f"      OCR input upscaled ×{args.ocr_scale}: "
              f"{ocr_input.shape[1]}×{ocr_input.shape[0]} px"
          )
      tiles = generate_tiles(ocr_input, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
      print(f"      Tiles generated: {len(tiles)}")
      raw_ocr = run_ocr_on_tiles(tiles, ocr_engine)
      # Back to original (un-upscaled) coordinates before any post-processing
      raw_ocr = rescale_detections(raw_ocr, args.ocr_scale)
      print(f"      Raw detections:  {len(raw_ocr)}")

      if not args.no_viz:
          #visualise_ocr(diagram, raw_ocr, title="Raw OCR detections")
          visualise_ocr(
              diagram,
              raw_ocr,
              title="Raw OCR detections",
              save_path=os.path.join(
                  output_dir,
                  "01_raw_ocr.png"
              )
          )

      # ------------------------------------------------------------------
      # 3. OCR post-processing
      # ------------------------------------------------------------------
      print("[3/6] Deduplicating and merging OCR results …")
      deduped = deduplicate_ocr(raw_ocr)
      print(f"      After dedup:  {len(deduped)}")
      if args.no_merge:
          merged_texts = deduped
          print("      Merge skipped (--no-merge): one detection per text line")
      else:
          merged_texts = merge_adjacent_lines(deduped)
          print(f"      After merge:  {len(merged_texts)}")
      merged_texts, n_repaired = repair_tag_text(merged_texts)
      print(f"      Tag suffixes repaired: {n_repaired}")

      if not args.no_viz:
          #visualise_ocr(diagram, merged_texts, title="Cleaned OCR detections")
          visualise_ocr(
              diagram,
              merged_texts,
              title="Cleaned OCR detections",
              save_path=os.path.join(
                  output_dir,
                  "02_cleaned_ocr.png"
              )
          )


      if args.ocr_cache:
          with open(args.ocr_cache, "w") as f:
              json.dump(merged_texts, f)
          print(f"      OCR cache written: {args.ocr_cache}")

    # ------------------------------------------------------------------
    # 4. Symbol detection
    # ------------------------------------------------------------------
    print("[4/6] Loading detection model and running symbol detection …")
    device = resolve_device(args.device)
    print(f"      Symbol detection device: {device}")
    det_model = build_detection_model(
        args.weights, id_to_name, device, args.symbol_conf)
    print(f"      Confidence cut-off: {args.symbol_conf}")
    if str(args.symbol_scale).lower() == "auto":
        symbol_scale = infer_symbol_scale(
            diagram, args.weights, id_to_name, device)
    else:
        symbol_scale = float(args.symbol_scale)
    print(f"      Detection input scale: ×{symbol_scale}")
    symbols = detect_symbols(diagram, det_model, symbol_scale)
    if not args.no_viz:
        visualise_symbols(
            diagram,
            symbols,
            save_path=os.path.join(
                output_dir,
                "03_symbols.png"
            )
        )
    n_raw = len(symbols)
    symbols = deduplicate_symbols(symbols)
    print(f"      Symbols detected: {len(symbols)}"
          + (f"  ({n_raw - len(symbols)} overlapping removed)"
             if n_raw != len(symbols) else ""))

    # ------------------------------------------------------------------
    # 5. Association
    # ------------------------------------------------------------------
    print("[5/6] Associating symbols with text labels …")
    associations = associate_symbols_to_text(
        symbols, merged_texts, args.cost_cap)
    print(f"      Cost cap: {args.cost_cap}")
    save_associations_excel(
        associations,
        os.path.join(
            output_dir,
            "associations.xlsx"
        )
    )

    save_associations_json(
        associations,
        os.path.join(
            output_dir,
            "associations.json"
        )
    )
    print(f"      Associations found: {len(associations)}")

    # ------------------------------------------------------------------
    # 6. Output
    # ------------------------------------------------------------------
    print("[6/6] Results:")
    print(f"{'Symbol ID':>10}  {'Class':<20}  {'Tag':<20}  {'Cost':>8}")
    print("-" * 65)
    for a in associations:
        print(f"{a['symbol_id']:>10}  {a['class']:<20}  {a['tag']:<20}  {a['cost']:>8.1f}")

    if not args.no_viz:
        #visualise_associations(diagram, associations)
        visualise_associations(
            diagram,
            associations,
            save_path=os.path.join(
                output_dir,
                "04_associations.png"
            )
        )


if __name__ == "__main__":
    main()
