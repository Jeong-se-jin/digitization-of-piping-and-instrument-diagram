# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import cv2


# Anything darker than this counts as ink. A drawing is ink on white paper, so
# the split is not really a two-cluster problem and Otsu handles it badly: on a
# sheet rendered from PDF the page is 98% white, which drags Otsu's threshold up
# to 154 and throws away every stroke drawn in light grey. One sheet measured
# here lost 21,106 ink pixels that way -- half as many as it kept -- including
# whole dashed runs sitting at value 191.
BINARY_THRESHOLD = 240


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_binary(image, threshold: int = BINARY_THRESHOLD):
    """Ink (dark) to 255, paper to 0.

    Pass threshold=0 to fall back to Otsu.
    """
    if threshold <= 0:
        return cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)[1]
