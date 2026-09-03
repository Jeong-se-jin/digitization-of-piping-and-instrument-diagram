# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Give the text that won no symbol a second chance at a nearby line or symbol.

Symbol-text association runs before this and leaves a lot of text unclaimed:
line numbers, notes, service labels. Much of it sits right on a pipe or against
a symbol, and once it is attached that pipe or symbol gains a name.

Who may receive a leftover text:

* a line that has no text yet;
* a symbol that has no tag yet;
* a symbol whose existing tag sits *inside* its own box. A tag drawn inside a
  bubble names the instrument, and a second label beside it -- a range, a
  service, a note reference -- belongs to the same instrument, so that symbol
  may take one more. A symbol whose tag was found outside its box has already
  spent its one nearby label, and is skipped.

Everything is matched nearest-first within a small radius, and nothing is ever
reassigned once taken.
"""
import math

import logger_config

logger = logger_config.get_logger(__name__)


def _box_distance(a, b, width, height):
    """Gap in pixels between two normalized boxes; 0 when they overlap."""
    dx = max(a.topX - b.bottomX, b.topX - a.bottomX, 0.0) * width
    dy = max(a.topY - b.bottomY, b.topY - a.bottomY, 0.0) * height
    return math.hypot(dx, dy)


def _inside(inner, outer):
    """True when `inner` lies entirely within `outer`."""
    return (outer.topX <= inner.topX and inner.bottomX <= outer.bottomX and
            outer.topY <= inner.topY and inner.bottomY <= outer.bottomY)


def _segment_box(segment):
    """A box covering a line segment, for the same distance test as a symbol."""
    class _Box:
        topX = min(segment.startX, segment.endX)
        topY = min(segment.startY, segment.endY)
        bottomX = max(segment.startX, segment.endX)
        bottomY = max(segment.startY, segment.endY)
    return _Box


def find_unassociated_text(all_text, symbols, image_height, image_width):
    """Text not already spoken for by a symbol's tag.

    A symbol's tag is a plain string, so the text it came from is identified by
    matching the string and taking the nearest one -- with duplicate tags on a
    sheet, the nearest is the one that symbol actually used.
    """
    claimed = set()
    for symbol in symbols:
        tag = (symbol.text_associated or '').strip()
        if not tag:
            continue
        best, best_distance = None, None
        for i, text in enumerate(all_text):
            if i in claimed or (text.text or '').strip() != tag:
                continue
            d = _box_distance(symbol, text, image_width, image_height)
            if best_distance is None or d < best_distance:
                best, best_distance = i, d
        if best is not None:
            claimed.add(best)

    return [i for i in range(len(all_text)) if i not in claimed]


def associate_leftover_text(
    all_text,
    symbols,
    line_segments,
    image_height: int,
    image_width: int,
    max_distance_pixels: float = 10.0
):
    """Attach unclaimed text to the nearest eligible line or symbol.

    Symbols gain the text on ``text_associated`` (appended when they already
    carry an inside tag); line segments gain ``text_associated``.

    :return: list of {text, target, distance_px} describing what was attached
    """
    leftover = find_unassociated_text(all_text, symbols, image_height, image_width)

    # A symbol may receive text if it has none, or if the tag it has was found
    # inside its own box.
    symbol_open = []
    for symbol in symbols:
        tag = (symbol.text_associated or '').strip()
        if not tag:
            symbol_open.append(True)
            continue
        has_inside_tag = any(
            (t.text or '').strip() == tag and _inside(t, symbol) for t in all_text)
        symbol_open.append(has_inside_tag)

    # A segment that touches a text box at all is that text's own drawing, not
    # pipe: text is masked out before Hough, so nothing of a real pipe survives
    # inside one. Requiring full containment was not enough -- a glyph stroke
    # reaching the edge of its box stayed eligible, and the notes paragraph then
    # collected a "line" for almost every line of prose.
    def touches_text(segment):
        lo_x = min(segment.startX, segment.endX)
        hi_x = max(segment.startX, segment.endX)
        lo_y = min(segment.startY, segment.endY)
        hi_y = max(segment.startY, segment.endY)
        for t in all_text:
            if (lo_x <= t.bottomX and t.topX <= hi_x and
                    lo_y <= t.bottomY and t.topY <= hi_y):
                return True
        return False

    line_open = [getattr(s, 'text_associated', None) is None and
                 not getattr(s, 'inside_box', False) and
                 not touches_text(s)
                 for s in line_segments]

    # Nearest pair first, so a text goes to the thing it is actually touching.
    pairs = []
    for ti in leftover:
        text = all_text[ti]
        for si, symbol in enumerate(symbols):
            if not symbol_open[si]:
                continue
            d = _box_distance(text, symbol, image_width, image_height)
            if d <= max_distance_pixels:
                pairs.append((d, ti, 'symbol', si))
        for li, segment in enumerate(line_segments):
            d = _box_distance(text, _segment_box(segment), image_width, image_height)
            if d <= max_distance_pixels:
                pairs.append((d, ti, 'line', li))
    pairs.sort(key=lambda p: p[0])

    used_text = set()
    attached = []
    for distance, ti, kind, index in pairs:
        if ti in used_text:
            continue
        text = (all_text[ti].text or '').strip()
        if not text:
            continue

        if kind == 'symbol':
            if not symbol_open[index]:
                continue
            symbol = symbols[index]
            existing = (symbol.text_associated or '').strip()
            symbol.text_associated = f'{existing} {text}'.strip() if existing else text
            symbol_open[index] = False        # one extra label, not more
            target = f'symbol {symbol.id}'
            target_box = [symbol.topX, symbol.topY, symbol.bottomX, symbol.bottomY]
        else:
            if not line_open[index]:
                continue
            line_segments[index].text_associated = text
            line_open[index] = False
            target = f'line {index}'
            seg = line_segments[index]
            target_box = [seg.startX, seg.startY, seg.endX, seg.endY]

        used_text.add(ti)
        t = all_text[ti]
        attached.append({
            'text': text, 'target': target, 'distance_px': round(distance, 1),
            'text_box': [t.topX, t.topY, t.bottomX, t.bottomY],
            'target_box': target_box,
        })

    n_symbol = sum(1 for a in attached if a['target'].startswith('symbol'))
    logger.info(
        f'Leftover text: {len(leftover)} unassociated, attached {len(attached)} '
        f'({n_symbol} to symbols, {len(attached) - n_symbol} to lines) '
        f'within {max_distance_pixels}px')
    return attached
