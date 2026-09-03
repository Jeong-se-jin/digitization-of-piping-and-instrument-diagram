# Keep only the segments drawn on the drawing's axes.
"""Discard segments that run neither horizontally nor vertically.

P&ID piping is drawn on the axes. What slants is, with few exceptions, not
piping: a leader line from a label, hatching inside a symbol, the edge of an
arrowhead, a scrap left where a symbol mask cut a curve. Those slanted scraps
still compete for the one candidate each endpoint has to give, so dropping them
costs nothing on the piping and frees candidates that were going to them.

The exceptions are real -- a slanted run does occur, and check valves and
relief lines are sometimes drawn at an angle -- so this stays off by default
and is turned on per run.
"""
import math

from app.models.line_detection.line_segment import LineSegment


def keep_axis_aligned_segments(
    line_segments: list[LineSegment],
    image_height: int,
    image_width: int,
    tolerance_degrees: float
) -> list[LineSegment]:
    """Return the segments within *tolerance_degrees* of horizontal or vertical.

    The angle is measured in pixels, not in normalised coordinates: a sheet is
    not square, so a 45-degree line normalises to something else entirely.
    """
    kept = []
    for segment in line_segments:
        dx = (segment.endX - segment.startX) * image_width
        dy = (segment.endY - segment.startY) * image_height
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle <= tolerance_degrees or angle >= 90 - tolerance_degrees:
            kept.append(segment)
    return kept
