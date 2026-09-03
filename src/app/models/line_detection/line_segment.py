# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pydantic import BaseModel
from typing import Optional


class LineSegment(BaseModel):
    """
    This class represents the line segment detected on a P&ID image.
    """
    startX: float
    startY: float
    endX: float
    endY: float
    # Set by line_type_classifier: 'solid' (pipe) or 'dashed' (signal line).
    # The dash measurements are only present on a dashed segment.
    line_type: Optional[str] = None
    dash_px: Optional[float] = None
    gap_px: Optional[float] = None
    period_px: Optional[float] = None
    # True when the segment lies entirely within one symbol or text box, so it
    # is the box's own drawing rather than pipe.
    inside_box: Optional[bool] = None
    # Set by associate_leftover_text: a nearby label that no symbol claimed.
    text_associated: Optional[str] = None
