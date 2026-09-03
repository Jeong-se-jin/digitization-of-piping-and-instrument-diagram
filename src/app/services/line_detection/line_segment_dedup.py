# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Drop segments that re-detect a stroke another segment already covers.

Thinning exists to stop Hough reporting a stroke twice, once down each edge.
The cost is paid by short marks: a 9px dash on this sheet comes back 7px after
thinning, because Zhang-Suen erodes the ends as well as the sides. Widening the
gap between dashes is exactly what the dash rhythm cannot afford.

Removing the duplicates afterwards instead keeps every dash at full length. Two
segments are duplicates when they lie along the same line -- near-parallel, with
little perpendicular separation -- *and* their extents overlap. The overlap test
is what protects dashes: consecutive dashes are collinear and close, but they do
not overlap, so they survive. The longer segment wins.
"""
import math

from app.models.line_detection.line_segment import LineSegment
import logger_config

logger = logger_config.get_logger(__name__)

ANGLE_TOLERANCE_DEG = 10.0
# A stroke is ~2.3px wide here, so its two edges land within a few pixels.
PERPENDICULAR_TOLERANCE = 5.0
# Fraction of the shorter segment that must be covered to call it a duplicate.
#
# Loose enough for diagonals. On an axis-aligned stroke the two edges are exactly
# parallel and cover each other, so a strict test catches them; a diagonal is
# drawn as a staircase, which tilts each edge slightly differently and lets Hough
# cut them into pieces that only partly line up. Measured on the reactor vessel,
# a strict test left those duplicates in place, and they then took the single
# connection slot each endpoint has -- the region's segments went from 63% used
# down to 36%.
MIN_OVERLAP_RATIO = 0.25


def _as_pixels(segment, height, width):
    return (segment.startX * width, segment.startY * height,
            segment.endX * width, segment.endY * height)


def _angle(x1, y1, x2, y2):
    """Direction in degrees, folded to [0, 180) so opposite ends match."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _angle_gap(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def deduplicate_line_segments(
    line_segments: list[LineSegment],
    image_height: int,
    image_width: int,
    angle_tolerance: float = ANGLE_TOLERANCE_DEG,
    perpendicular_tolerance: float = PERPENDICULAR_TOLERANCE,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO
) -> list[LineSegment]:
    """Keep the longest segment among those covering the same stroke.

    :param line_segments: Segments with normalized coordinates
    :param image_height: Image height in pixels
    :param image_width: Image width in pixels
    :param angle_tolerance: Degrees within which two segments count as parallel
    :param perpendicular_tolerance: Pixels of sideways separation still allowed
    :param min_overlap_ratio: Overlap needed, as a fraction of the shorter one
    :return: The kept segments, longest first within each stroke
    """
    if not line_segments:
        return line_segments

    items = []
    for i, s in enumerate(line_segments):
        x1, y1, x2, y2 = _as_pixels(s, image_height, image_width)
        length = math.hypot(x2 - x1, y2 - y1)
        items.append({
            'i': i, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'length': length, 'angle': _angle(x1, y1, x2, y2),
            'ux': (x2 - x1) / length if length else 0.0,
            'uy': (y2 - y1) / length if length else 0.0,
        })

    # Longest first: a duplicate is judged against a segment already kept, so
    # the survivor is the longest of each stroke.
    items.sort(key=lambda it: -it['length'])

    kept = []
    dropped = 0
    for cand in items:
        duplicate = False
        for k in kept:
            if _angle_gap(cand['angle'], k['angle']) > angle_tolerance:
                continue

            # Sideways distance from the kept line, measured at both ends of the
            # candidate. Both have to be close, or it is a different stroke.
            perp = []
            for px, py in ((cand['x1'], cand['y1']), (cand['x2'], cand['y2'])):
                perp.append(abs((px - k['x1']) * k['uy'] - (py - k['y1']) * k['ux']))
            if max(perp) > perpendicular_tolerance:
                continue

            # Overlap along the kept segment's own direction. Dashes fail here:
            # they are collinear but sit end to end, never on top of each other.
            proj = []
            for px, py in ((k['x1'], k['y1']), (k['x2'], k['y2']),
                           (cand['x1'], cand['y1']), (cand['x2'], cand['y2'])):
                proj.append((px - k['x1']) * k['ux'] + (py - k['y1']) * k['uy'])
            k_lo, k_hi = sorted(proj[:2])
            c_lo, c_hi = sorted(proj[2:])
            overlap = min(k_hi, c_hi) - max(k_lo, c_lo)
            shorter = min(cand['length'], k['length'])
            if shorter > 0 and overlap / shorter >= min_overlap_ratio:
                duplicate = True
                break

        if duplicate:
            dropped += 1
        else:
            kept.append(cand)

    logger.info(f'Segment dedup: dropped {dropped} duplicates, kept {len(kept)}')
    return [line_segments[k['i']] for k in kept]
