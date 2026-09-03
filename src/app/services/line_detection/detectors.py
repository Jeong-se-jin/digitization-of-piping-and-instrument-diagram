# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Interchangeable raw line detectors.

Each backend takes the preprocessed binary image and returns raw segments as
``[(x1, y1, x2, y2), ...]`` in pixels. Everything after that -- endpoint
ordering, the inclusive-box filter, normalization, dedup, dash classification --
is shared, so a backend can be swapped without touching the rest of the
pipeline.

``fld`` is the default; ``hough`` is the original and is kept intact. ``fld`` and ``lsd`` grow line
support regions from pixel gradients instead of voting into a quantized (rho,
theta) accumulator, which is where the Hough backend loses ground on this
drawing: a diagonal's staircase pixels spread their votes across bins, and a
dense area of short marks produces a crowd of spurious fragments. Whether that
trade is worth it is an empirical question per drawing -- hence the switch.
"""
import cv2
import numpy as np

import logger_config

logger = logger_config.get_logger(__name__)

HOUGH = 'hough'
FLD = 'fld'
LSD = 'lsd'
BACKENDS = (HOUGH, FLD, LSD)


def detect_raw_segments(image: np.ndarray, backend: str = HOUGH, **params):
    """Run one backend and return raw pixel-space segments.

    :param image: Preprocessed binary image, ink = 255
    :param backend: One of BACKENDS
    :param params: Backend-specific settings; unknown keys are ignored
    :return: list of (x1, y1, x2, y2) floats
    """
    if backend == HOUGH:
        segments = _hough(image, params)
    elif backend == FLD:
        segments = _fld(image, params)
    elif backend == LSD:
        segments = _lsd(image, params)
    else:
        raise ValueError(f'Unknown line detection backend {backend!r}; '
                         f'expected one of {", ".join(BACKENDS)}')

    logger.info(f'Line backend {backend}: {len(segments)} raw segments')
    return segments


def _hough(image, p):
    """Probabilistic Hough transform -- the original behaviour, unchanged."""
    result = cv2.HoughLinesP(
        image,
        rho=p.get('rho', 0.1),
        theta=np.pi / p.get('theta_param', 1080),
        threshold=p.get('threshold', 5),
        minLineLength=p.get('min_line_length'),
        maxLineGap=p.get('max_line_gap'))
    if result is None:
        return []
    return [tuple(float(v) for v in line[0]) for line in result]


LENGTH_PER_STROKE_WIDTH = 1.7


def estimate_stroke_width(image):
    """Average stroke width in pixels: total ink divided by skeleton length.

    The median of the distance transform does not separate these drawings --
    both sit at 1.91px, because most ink is thin either way. Total ink over
    skeleton length does: it is the mean width along every stroke, and it comes
    out 2.34px on one sheet and 1.39px on the other, which is the same ratio as
    the length thresholds each of them wants.
    """
    binary = (image > 0).astype(np.uint8) * 255
    ink = int((binary > 0).sum())
    if not ink:
        return 1.0
    skeleton = cv2.ximgproc.thinning(
        binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    bones = int((skeleton > 0).sum())
    return ink / bones if bones else 1.0


def _fld_length_threshold(image, p):
    """The explicit setting, else one derived from how thick the drawing is.

    A short segment on a finely drawn sheet is a dash; on a heavier sheet the
    same length is noise. Scaling with stroke width tracks that, and saves
    retuning the threshold per drawing.
    """
    explicit = p.get('fld_length_threshold') or p.get('min_line_length')
    if explicit:
        return int(explicit)
    # Measure on the un-thinned binary when one was handed over: a thinned image
    # is one pixel wide everywhere and would report a width of 1.
    width = estimate_stroke_width(
        p.get('reference_image') if p.get('reference_image') is not None else image)
    value = max(2, int(round(LENGTH_PER_STROKE_WIDTH * width)))
    logger.info(f'FLD length threshold {value} from stroke width {width:.2f}px')
    return value


def _fld(image, p):
    """OpenCV's FastLineDetector.

    ``canny_aperture_size=0`` tells it to treat the input as an edge image,
    which is exactly what the preprocessed binary already is -- running Canny
    over a binary would find the two sides of every stroke and hand back each
    line twice.
    """
    detector = cv2.ximgproc.createFastLineDetector(
        length_threshold=_fld_length_threshold(image, p),
        distance_threshold=float(p.get('fld_distance_threshold', 1.414)),
        canny_th1=50.0,
        canny_th2=50.0,
        canny_aperture_size=0,
        do_merge=bool(p.get('fld_do_merge', False)))
    result = detector.detect(image)
    if result is None:
        return []
    return [tuple(float(v) for v in line[0]) for line in result]


def _lsd(image, p):
    """OpenCV's LineSegmentDetector.

    Kept beside FLD because the two disagree on thin strokes often enough to be
    worth comparing; LSD returns sub-pixel endpoints and no merging.
    """
    if not hasattr(cv2, 'createLineSegmentDetector'):
        raise RuntimeError('This OpenCV build has no LineSegmentDetector; '
                           'use the fld backend instead.')
    detector = cv2.createLineSegmentDetector()
    result = detector.detect(image)[0]
    if result is None:
        return []
    min_length = float(p.get('min_line_length') or 0)
    segments = []
    for line in result:
        x1, y1, x2, y2 = (float(v) for v in line[0])
        if min_length and np.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        segments.append((x1, y1, x2, y2))
    return segments
