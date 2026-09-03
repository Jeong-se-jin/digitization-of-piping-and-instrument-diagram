# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Label detected segments solid (pipe) or dashed (instrument signal).

The Hough transform makes no such distinction: it reports a dashed run as a
handful of short collinear segments, indistinguishable from a solid line that
happened to break. This runs afterwards and measures the ink along each line.

Two things make the measurement work:

* It reads the binary image *before* thinning. Thinning reduces a stroke to one
  wobbling pixel wide, and a solid line then reads as broken -- measured here,
  a solid run came back with 5px gaps.
* It groups collinear segments first. A dashed run arrives as one segment per
  dash, so a single segment shows no gaps at all; only the group does.

Two gaps is enough to judge on. A signal line running between the two symbols
it links is masked at both ends, so only the middle three dashes survive -- the
FT-to-panel-box lines on this sheet measure exactly that.

Regularity is the actual discriminator. A solid pipe punched through by symbol
masking also has several gaps -- one measured run had gaps of 15/18/16px, which
on their own look like a dash rhythm. Its dashes were 339/139/86/430px, wildly
uneven, so the dash coefficient of variation separates the two. Both
coefficients have to be checked; the gap one alone would pass that line.
"""
import numpy as np
from statistics import median

from app.models.line_detection.line_segment import LineSegment
import logger_config

logger = logger_config.get_logger(__name__)

# Tuned on a 300dpi 3916x2572 sheet. Every value is a pixel distance, so they
# scale with resolution -- pass reference_width so they can be rescaled rather
# than silently mismeasuring a sheet of another size.
REFERENCE_WIDTH = 3916

GAP_MIN = 2
GAP_MAX = 25
DASH_MIN = 4
DASH_MAX = 60
GAP_CV_MAX = 0.45
DASH_CV_MAX = 0.45
MIN_GAP_COUNT = 2
BREAK_ON_BLANK = 25
RUN_BREAK = 40            # px along the axis; beyond this it is a separate line

COLLINEAR_TOLERANCE = 2   # px, for grouping segments onto one line
BAND_RADIUS = 3           # px, rows/columns searched for the inkiest one
AXIS_TOLERANCE = 2        # px, below which a segment counts as axis-aligned

SOLID = 'solid'
DASHED = 'dashed'


def _cv(values):
    """Coefficient of variation; 0 for a constant run, inf for a zero mean."""
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    if mean == 0:
        return float('inf')
    return float(np.std(values)) / mean


def _runs(profile):
    """Run-length encode a boolean profile into (is_ink, start, length)."""
    out = []
    if len(profile) == 0:
        return out
    start = 0
    for i in range(1, len(profile) + 1):
        if i == len(profile) or profile[i] != profile[start]:
            out.append((bool(profile[start]), start, i - start))
            start = i
    return out


def _classify_profile(profile, params):
    """Decide solid/dashed for one 1-D ink profile.

    Returns a list of (start, end, line_type, metrics) covering the profile.
    The profile is first cut at any blank longer than break_on_blank: those are
    symbol-mask holes or the end of the run, not part of a dash rhythm.
    """
    results = []
    runs = _runs(profile)

    # Split into chunks at long blanks.
    chunks, current = [], []
    for is_ink, start, length in runs:
        if not is_ink and length > params['break_on_blank']:
            if current:
                chunks.append(current)
            current = []
        else:
            current.append((is_ink, start, length))
    if current:
        chunks.append(current)

    for chunk in chunks:
        # Drop the leading and trailing run: both are clipped by where the
        # inspection interval happens to end, so their lengths mean nothing.
        inner = chunk[1:-1] if len(chunk) > 2 else []
        gaps = [ln for is_ink, _, ln in inner if not is_ink]
        dashes = [ln for is_ink, _, ln in inner if is_ink]

        c_start = chunk[0][1]
        c_end = chunk[-1][1] + chunk[-1][2]

        line_type, metrics = SOLID, None
        if len(gaps) >= params['min_gap_count'] and dashes:
            g_med, d_med = median(gaps), median(dashes)
            if (params['gap_min'] <= g_med <= params['gap_max'] and
                    params['dash_min'] <= d_med <= params['dash_max'] and
                    _cv(gaps) < params['gap_cv_max'] and
                    _cv(dashes) < params['dash_cv_max']):
                line_type = DASHED
                metrics = {'dash_px': float(d_med), 'gap_px': float(g_med),
                           'period_px': float(d_med + g_med)}
        results.append((c_start, c_end, line_type, metrics))

    return results


def _split_along_axis(members, max_break):
    """Cut a collinear group wherever its members are far apart along the axis.

    Members arrive as (index, level, lo, hi). Two runs on the same level that
    sit further than max_break apart are different lines and are measured
    separately.
    """
    members = sorted(members, key=lambda m: m[2])
    out, current, reach = [], [members[0]], members[0][3]
    for m in members[1:]:
        if m[2] - reach > max_break:
            out.append(current)
            current = []
        current.append(m)
        reach = max(reach, m[3])
    out.append(current)
    return out


def _scaled_params(image_width):
    """Thresholds rescaled from the reference sheet width."""
    scale = image_width / REFERENCE_WIDTH
    r = lambda v: max(1, int(round(v * scale)))  # noqa: E731
    return {
        'gap_min': r(GAP_MIN), 'gap_max': r(GAP_MAX),
        'dash_min': r(DASH_MIN), 'dash_max': r(DASH_MAX),
        'break_on_blank': r(BREAK_ON_BLANK),
        'run_break': r(RUN_BREAK),
        'collinear_tolerance': r(COLLINEAR_TOLERANCE),
        'band_radius': r(BAND_RADIUS),
        'gap_cv_max': GAP_CV_MAX, 'dash_cv_max': DASH_CV_MAX,
        'min_gap_count': MIN_GAP_COUNT,
    }


def classify_line_segments(
    binary_image,
    line_segments: list[LineSegment],
    image_height: int,
    image_width: int
) -> list[LineSegment]:
    """Annotate each segment with line_type, and dash metrics when dashed.

    :param binary_image: The masked, binarized image *before* thinning, ink=255
    :param line_segments: Segments with normalized coordinates
    :param image_height: Image height in pixels
    :param image_width: Image width in pixels
    :return: The same segments, with line_type set
    """
    params = _scaled_params(image_width)
    ink = binary_image > 0

    # Pixel-space endpoints, and which axis each segment lies on.
    px = []
    for i, s in enumerate(line_segments):
        x1, y1 = s.startX * image_width, s.startY * image_height
        x2, y2 = s.endX * image_width, s.endY * image_height
        if abs(y2 - y1) <= AXIS_TOLERANCE:
            px.append((i, 'h', (y1 + y2) / 2, min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) <= AXIS_TOLERANCE:
            px.append((i, 'v', (x1 + x2) / 2, min(y1, y2), max(y1, y2)))
        else:
            px.append((i, None, 0, 0, 0))   # diagonal: left solid, see below

    # Group collinear segments: a dashed run is one segment per dash, so the
    # gaps only exist between members of a group.
    groups = {}
    for i, axis, level, lo, hi in px:
        if axis is None:
            continue
        key = (axis, int(round(level / params['collinear_tolerance'])))
        groups.setdefault(key, []).append((i, level, lo, hi))

    types = {}
    metrics = {}
    for (axis, _), all_members in groups.items():
        # Split the group along its own axis before measuring. Sharing a level
        # does not make two segments part of one line: a column at x=965 holds
        # both a 30px signal line near the top and unrelated pipe 1400px below,
        # and spanning them makes the inspection interval mostly blank, which
        # destroys the rhythm the test looks for.
        for members in _split_along_axis(all_members, params['run_break']):
            level = int(round(np.mean([m[1] for m in members])))
            start = int(np.floor(min(m[2] for m in members)))
            end = int(np.ceil(max(m[3] for m in members)))
            if end - start < 2:
                continue

            # Pick the single inkiest row/column in the band. Folding the band
            # with an OR mixes in a neighbouring stroke and a dashed line reads
            # solid.
            radius = params['band_radius']
            best, best_ratio = None, -1.0
            for offset in range(-radius, radius + 1):
                lvl = level + offset
                if axis == 'h':
                    if not (0 <= lvl < image_height):
                        continue
                    profile = ink[lvl, start:end]
                else:
                    if not (0 <= lvl < image_width):
                        continue
                    profile = ink[start:end, lvl]
                ratio = float(profile.mean()) if profile.size else 0.0
                if ratio > best_ratio:
                    best, best_ratio = profile, ratio

            if best is None or best.size == 0:
                continue

            for c_start, c_end, line_type, m in _classify_profile(best, params):
                if line_type != DASHED:
                    continue
                # Attribute the verdict to the segments lying in that chunk.
                lo_abs, hi_abs = start + c_start, start + c_end
                for i, _, seg_lo, seg_hi in members:
                    if seg_lo < hi_abs and seg_hi > lo_abs:
                        types[i] = DASHED
                        metrics[i] = m

    n_dashed = 0
    for i, s in enumerate(line_segments):
        # Diagonals are left solid: the row/column profile this method reads is
        # only defined for axis-aligned runs.
        s.line_type = types.get(i, SOLID)
        m = metrics.get(i)
        if m:
            s.dash_px, s.gap_px, s.period_px = m['dash_px'], m['gap_px'], m['period_px']
            n_dashed += 1

    logger.info(f'Line types: {n_dashed} dashed, {len(line_segments) - n_dashed} solid')
    return line_segments
