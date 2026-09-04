"""Line detection with the pidpipe post-processing chain.

Preprocessing is the repo's own -- ``LineDetectionImagePreprocessor`` clears the
symbol and text boxes, greyscales, binarizes with Otsu and (optionally) thins --
so this backend sees exactly the image ``line_detection_service`` sees.  What
differs is everything after ``HoughLinesP``:

    Hough -> H/V gate + snap -> 1-D interval merge
          -> evidence-checked gap bridging -> orthogonal corner snap -> re-merge

Plain Hough answers a P&ID with hundreds of short collinear fragments: one pipe
run broken at every symbol, every tag gap and every dash.  Graph construction
then has to rediscover that those fragments are one line.  The chain above does
that consolidation up front, and refuses a join it cannot see evidence for in
the binarized image, so a bridge across a 400px gap is only taken when ink
actually sits along it.

The algorithm is ported from ChatP-ID's ``pidpipe.runner.line_detect``.  Two
deliberate departures: the BlackHat/Otsu edge step is dropped, because this
repo's preprocessor already produces a thinned binary; and dash-rhythm scoring
is folded into the single ``min_ratio`` support test, which is what the rhythm
check effectively decided on the sheets we have.

Coordinates are pixel-space internally and normalized to [0, 1] on the way out,
matching ``LineSegment``.

Usage:
    from tools.sta_bridge import pidpipe_lines
    results = pidpipe_lines.detect_lines(
        image_bytes=..., text_detection_results=request,
        image_height=h, image_width=w)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.models.bounding_box import BoundingBox
from app.models.image_details import ImageDetails
from app.models.line_detection.line_detection_response import \
    LineDetectionInferenceResponse
from app.models.line_detection.line_segment import LineSegment
from app.services.line_detection.utils.line_detection_image_preprocessor import \
    LineDetectionImagePreprocessor

Segment = Tuple[int, int, int, int]


@dataclass
class PidpipeParams:
    """Tuning for the chain. Defaults are ChatP-ID's config.yaml values."""

    # Hough
    hough_threshold: int = 14
    hough_min_line_length: int = 12
    hough_max_line_gap: int = 240

    # H/V gate: a segment more than this far off an axis is discarded, not
    # rotated. NOTE: on the AP1000 sheets this drops real piping -- the reactor
    # loop lines and the containment boundary are drawn on the diagonal, about
    # a tenth of the lines the plain-Hough backend finds. Diagonal support is
    # not implemented here yet.
    angle_tolerance_deg: float = 6.0

    # 1-D merge: how far apart two segments may sit on the off-axis and still
    # count as the same line, and the gap they may span along the axis.
    coord_tolerance_px: int = 25
    merge_gap_px: int = 100

    # Evidence test for a bridge
    support_band_px: int = 11
    support_min_ratio: float = 0.004
    bridge_max_gap_px: int = 600
    crossing_guard_max: int = 2

    # Corner snap for L and T junctions left open by the gate
    corner_snap: bool = True
    corner_tolerance_px: int = 160

    # Drop anything shorter than this at the very end
    min_length_px: int = 12


def detect_lines(
    image_bytes: bytes,
    text_detection_results,
    image_height: int,
    image_width: int,
    enable_thinning: bool = True,
    params: Optional[PidpipeParams] = None,
    bounding_box_inclusive: Optional[BoundingBox] = None,
    image_url: str = '',
    debug_image_preprocessed_path: Optional[str] = None,
    debug_image_preprocessed_before_thinning_path: Optional[str] = None,
    debug_image_lines_on_preprocessed_path: Optional[str] = None,
    output_image_line_segments_path: Optional[str] = None,
) -> LineDetectionInferenceResponse:
    """Detect line segments, returning the same response type as the Hough backend.

    :param text_detection_results: the request carrying symbol and text boxes,
        normalized -- the same object ``line_detection_service.detect_lines`` takes
    :param enable_thinning: apply Zhang-Suen thinning before Hough
    :param bounding_box_inclusive: if given, keep only segments fully inside it
    """
    p = params or PidpipeParams()

    symbol_boxes = _denormalize(
        text_detection_results.text_and_symbols_associated_list,
        image_height, image_width)
    text_boxes = _denormalize(
        text_detection_results.all_text_list, image_height, image_width)

    binary = LineDetectionImagePreprocessor.preprocess(
        image_bytes, symbol_boxes, text_boxes)
    if debug_image_preprocessed_before_thinning_path:
        cv2.imwrite(debug_image_preprocessed_before_thinning_path, binary)
    if enable_thinning:
        binary = LineDetectionImagePreprocessor.apply_thinning(binary)
    if debug_image_preprocessed_path:
        cv2.imwrite(debug_image_preprocessed_path, binary)

    segments = _run_chain(binary, p)

    if debug_image_lines_on_preprocessed_path:
        # Lines over the masked binary, which is what the chain actually saw.
        # Reading them against the original drawing hides the failure mode that
        # matters: a segment that fits no stroke in *this* image.
        canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in segments:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(debug_image_lines_on_preprocessed_path, canvas)

    if bounding_box_inclusive is not None:
        keep = _denormalize([bounding_box_inclusive], image_height, image_width)[0]
        segments = [s for s in segments if _inside(s, keep)]

    if output_image_line_segments_path:
        _write_overlay(image_bytes, segments, output_image_line_segments_path)

    line_segments = [
        LineSegment(
            startX=x1 / image_width, startY=y1 / image_height,
            endX=x2 / image_width, endY=y2 / image_height,
        )
        for (x1, y1, x2, y2) in segments
    ]

    return LineDetectionInferenceResponse(
        image_url=image_url,
        image_details=ImageDetails(
            format='png', width=image_width, height=image_height),
        line_segments_count=len(line_segments),
        line_segments=line_segments,
    )


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
def _run_chain(binary: np.ndarray, p: PidpipeParams) -> List[Segment]:
    # A slightly dilated copy is the evidence map. Thinning leaves 1px strokes;
    # sampling those directly makes the support ratio depend on rounding.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    support = cv2.dilate(binary, kernel, iterations=1)

    raw = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=p.hough_threshold,
        minLineLength=p.hough_min_line_length,
        maxLineGap=p.hough_max_line_gap,
    )
    if raw is None:
        return []
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    raw = raw.reshape(-1, 4)

    horizontals, verticals = _gate_and_snap(raw, p.angle_tolerance_deg)

    horizontals = _merge_1d(horizontals, 'h', p.coord_tolerance_px, p.merge_gap_px)
    verticals = _merge_1d(verticals, 'v', p.coord_tolerance_px, p.merge_gap_px)

    horizontals = _bridge(horizontals, 'h', support, verticals, p)
    verticals = _bridge(verticals, 'v', support, horizontals, p)

    if p.corner_snap:
        added_h, added_v = _snap_corners(horizontals, verticals, support, p)
        horizontals += added_h
        verticals += added_v

    horizontals = _merge_1d(horizontals, 'h', p.coord_tolerance_px, p.merge_gap_px)
    verticals = _merge_1d(verticals, 'v', p.coord_tolerance_px, p.merge_gap_px)

    out = horizontals + verticals
    return [s for s in out if _length(s) >= p.min_length_px]


def _gate_and_snap(
    raw: np.ndarray, angle_tolerance_deg: float
) -> Tuple[List[Segment], List[Segment]]:
    """Keep near-axis segments, snapping each onto its axis exactly.

    Snapping is what makes the 1-D merge downstream possible: two fragments of
    one pipe run only share an off-axis coordinate once both are flattened.
    """
    horizontals: List[Segment] = []
    verticals: List[Segment] = []

    for x1, y1, x2, y2 in raw.astype(int):
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180.0 - angle)          # fold to [0, 90]
        if angle <= angle_tolerance_deg:
            y = (y1 + y2) // 2
            horizontals.append((min(x1, x2), y, max(x1, x2), y))
        elif angle >= 90.0 - angle_tolerance_deg:
            x = (x1 + x2) // 2
            verticals.append((x, min(y1, y2), x, max(y1, y2)))

    return horizontals, verticals


def _merge_1d(
    segments: List[Segment], axis: str, coord_tol: int, gap_tol: int
) -> List[Segment]:
    """Group segments by off-axis coordinate, then union overlapping intervals.

    Two segments join when they sit within ``coord_tol`` of each other off-axis
    and their intervals overlap or are separated by at most ``gap_tol``.
    """
    if not segments:
        return []

    def off(s: Segment) -> int:
        return s[1] if axis == 'h' else s[0]

    def span(s: Segment) -> Tuple[int, int]:
        return (s[0], s[2]) if axis == 'h' else (s[1], s[3])

    merged: List[Segment] = []
    for _, group in _cluster_by(segments, off, coord_tol):
        level = int(round(float(np.mean([off(s) for s in group]))))
        intervals = sorted(span(s) for s in group)

        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start - current_end <= gap_tol:
                current_end = max(current_end, end)
            else:
                merged.append(_build(axis, level, current_start, current_end))
                current_start, current_end = start, end
        merged.append(_build(axis, level, current_start, current_end))

    return merged


def _bridge(
    segments: List[Segment], axis: str, support: np.ndarray,
    crossing: List[Segment], p: PidpipeParams,
) -> List[Segment]:
    """Join consecutive collinear segments across a gap, when the ink agrees.

    Two guards keep this from fusing unrelated runs: the gap must carry ink
    along it (``_has_support``), and it must not be crossed by more than
    ``crossing_guard_max`` perpendicular lines -- a long gap striped by many
    crossings is whitespace between two runs, not one dashed run.
    """
    if not segments:
        return []

    def off(s: Segment) -> int:
        return s[1] if axis == 'h' else s[0]

    def span(s: Segment) -> Tuple[int, int]:
        return (s[0], s[2]) if axis == 'h' else (s[1], s[3])

    out: List[Segment] = []
    for _, group in _cluster_by(segments, off, p.coord_tolerance_px):
        level = off(group[0])
        ordered = sorted(group, key=span)

        current = ordered[0]
        for nxt in ordered[1:]:
            gap_start, gap_end = span(current)[1], span(nxt)[0]
            gap = gap_end - gap_start
            joinable = (
                0 < gap <= p.bridge_max_gap_px
                and _has_support(support, axis, level, gap_start, gap_end, p)
                and _crossings(crossing, axis, level, gap_start, gap_end)
                <= p.crossing_guard_max
            )
            if joinable or gap <= 0:
                start = min(span(current)[0], span(nxt)[0])
                end = max(span(current)[1], span(nxt)[1])
                current = _build(axis, level, start, end)
            else:
                out.append(current)
                current = nxt
        out.append(current)

    return out


def _snap_corners(
    horizontals: List[Segment], verticals: List[Segment],
    support: np.ndarray, p: PidpipeParams,
) -> Tuple[List[Segment], List[Segment]]:
    """Close L and T junctions whose two arms stop short of meeting.

    The gate discards the short diagonal Hough fragment that usually sits in a
    corner, leaving a few px of daylight. Emit the stub that closes it, so the
    junction becomes a real shared endpoint for graph construction.
    """
    added_h: List[Segment] = []
    added_v: List[Segment] = []

    for hx1, hy, hx2, _ in horizontals:
        for vx, vy1, _, vy2 in verticals:
            # Does the vertical's x fall near either end of the horizontal?
            for hx in (hx1, hx2):
                dx = abs(vx - hx)
                if dx == 0 or dx > p.corner_tolerance_px:
                    continue
                # ...and the horizontal's y near either end of the vertical?
                for vy in (vy1, vy2):
                    if abs(hy - vy) > p.corner_tolerance_px:
                        continue
                    lo, hi = min(hx, vx), max(hx, vx)
                    if _has_support(support, 'h', hy, lo, hi, p):
                        added_h.append((lo, hy, hi, hy))
                    lo, hi = min(hy, vy), max(hy, vy)
                    if _has_support(support, 'v', vx, lo, hi, p):
                        added_v.append((vx, lo, vx, hi))

    return added_h, added_v


# ---------------------------------------------------------------------------
# Evidence and geometry helpers
# ---------------------------------------------------------------------------
def _has_support(
    support: np.ndarray, axis: str, level: int, start: int, end: int,
    p: PidpipeParams,
) -> bool:
    """True when a single line of ink runs the length of the span.

    The band is ``support_band_px`` wide so a stroke that drifts a pixel or two
    off the snapped level still counts -- but each row of the band is scored
    *separately* and only the best row decides.  That distinction is the whole
    test: collapsing the band with an ``any`` would let a text paragraph pass,
    because some glyph sits in almost every column of a span crossing one.  A
    pipe puts one row near 1.0; a paragraph breaks every row at the gaps
    between letters, so its best row stays well below.
    """
    if end <= start:
        return True

    height, width = support.shape[:2]
    half = p.support_band_px // 2

    if axis == 'h':
        y1, y2 = max(0, level - half), min(height, level + half + 1)
        x1, x2 = max(0, start), min(width, end)
    else:
        x1, x2 = max(0, level - half), min(width, level + half + 1)
        y1, y2 = max(0, start), min(height, end)

    if x2 <= x1 or y2 <= y1:
        return False

    window = support[y1:y2, x1:x2] > 0
    # Coverage of each row (for a horizontal span) or column (for a vertical).
    per_row = window.mean(axis=1 if axis == 'h' else 0)
    return float(per_row.max()) >= p.support_min_ratio


def _crossings(
    crossing: List[Segment], axis: str, level: int, start: int, end: int
) -> int:
    """Count perpendicular segments cutting across the given span."""
    count = 0
    for x1, y1, x2, y2 in crossing:
        if axis == 'h':
            if start < x1 < end and min(y1, y2) <= level <= max(y1, y2):
                count += 1
        else:
            if start < y1 < end and min(x1, x2) <= level <= max(x1, x2):
                count += 1
    return count


def _cluster_by(items, key, tolerance: int):
    """Group items whose ``key`` values lie within ``tolerance`` *of each other*.

    The width of a cluster is capped at ``tolerance``, not just the step between
    neighbours.  Comparing only against the previous item lets a cluster walk:
    parallel pipe runs spaced 20px apart chain into one 400px-wide group, and
    every segment in it then gets flattened onto the group's mean level -- a
    line drawn where no pipe is.  P&ID sheets are dense with parallel runs, so
    that failure is the common case, not an edge case.
    """
    ordered = sorted(items, key=key)
    cluster = [ordered[0]]
    for item in ordered[1:]:
        if key(item) - key(cluster[0]) <= tolerance:
            cluster.append(item)
        else:
            yield key(cluster[0]), cluster
            cluster = [item]
    yield key(cluster[0]), cluster


def _build(axis: str, level: int, start: int, end: int) -> Segment:
    """Assemble a segment, start-to-end in reading order (left-right, top-down)."""
    if axis == 'h':
        return (int(start), int(level), int(end), int(level))
    return (int(level), int(start), int(level), int(end))


def _length(s: Segment) -> int:
    return max(abs(s[2] - s[0]), abs(s[3] - s[1]))


def _inside(s: Segment, box: BoundingBox) -> bool:
    xs, ys = (s[0], s[2]), (s[1], s[3])
    return (box.topX <= min(xs) and max(xs) <= box.bottomX
            and box.topY <= min(ys) and max(ys) <= box.bottomY)


def _denormalize(items, image_height: int, image_width: int) -> List[BoundingBox]:
    """Normalized boxes -> pixel-space boxes, as the preprocessor expects."""
    return [
        BoundingBox(
            topX=int(item.topX * image_width),
            topY=int(item.topY * image_height),
            bottomX=int(item.bottomX * image_width),
            bottomY=int(item.bottomY * image_height),
        )
        for item in items
    ]


def _write_overlay(image_bytes: bytes, segments: List[Segment], path: str) -> None:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    for x1, y1, x2, y2 in segments:
        cv2.line(image, (x1, y1), (x2, y2), (0, 155, 0), 2)
    cv2.imwrite(path, image)
