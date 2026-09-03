# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math

import cv2
import numpy as np
from app.services.base_image_preprocessor import to_grayscale, to_binary
from app.models.bounding_box import BoundingBox


class LineDetectionImagePreprocessor:
    '''
    Helper class to perform preprocessing on an image.
    '''
    @staticmethod
    def preprocess(image_bytes: bytes,
                   symbol_bounding_boxes: list[BoundingBox],
                   text_bounding_boxes: list[BoundingBox],
                   symbol_mask_inset: int = 0,
                   text_mask_inset: int = 0,
                   exempt_bounding_boxes: list[BoundingBox] = None,
                   exempt_mask_inset: int = 0,
                   binary_threshold: int = 240):
        '''
        Preprocesses the given image bytes. Applies the following transformations:
        1. Clears symbol bounding boxes
        2. Clears text bounding boxes
        3. Converts the image to grayscale
        4. Binarizes the image using Otsu's method for image thresholding

        :param image_bytes: The image bytes to preprocess
        :type image_bytes: bytes
        :param symbol_bounding_boxes: The symbol bounding boxes to clear
        :type symbol_bounding_boxes: list
        :param text_bounding_boxes: The text bounding boxes to clear
        :type text_bounding_boxes: list
        :param symbol_mask_inset: Pixels to shrink each symbol box by before clearing
        :type symbol_mask_inset: int
        :param text_mask_inset: Pixels to shrink each text box by before clearing
        :type text_mask_inset: int
        :param exempt_bounding_boxes: Boxes masked with exempt_mask_inset instead
            of symbol_mask_inset
        :type exempt_bounding_boxes: list
        :param exempt_mask_inset: Pixels to shrink each exempt box by
        :type exempt_mask_inset: int
        :return: The preprocessed image bytes
        :rtype: bytes
        '''
        # convert image bytes to cv2 image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Clear symbol bounding boxes
        image = LineDetectionImagePreprocessor.clear_bounding_boxes(
            image, symbol_bounding_boxes, symbol_mask_inset)

        # Clear text bounding boxes
        image = LineDetectionImagePreprocessor.clear_bounding_boxes(
            image, text_bounding_boxes, text_mask_inset)

        # Clear the boxes that use their own inset rather than the symbol one
        if exempt_bounding_boxes:
            image = LineDetectionImagePreprocessor.clear_bounding_boxes(
                image, exempt_bounding_boxes, exempt_mask_inset)

        # Convert to grayscale
        image = to_grayscale(image)

        # Binarization
        image = to_binary(image, binary_threshold)

        # return the image
        return image

    @staticmethod
    def _inset_pair(inset):
        '''Normalizes an inset to (horizontal, vertical) pixels.'''
        if isinstance(inset, (tuple, list)):
            return int(inset[0]), int(inset[1])
        return int(inset), int(inset)

    @staticmethod
    def clear_bounding_boxes(image, bounding_boxes: list[BoundingBox], inset=0):
        '''
        Clears the given bounding boxes from the image.

        With a non-zero inset each box is shrunk before being filled. A symbol
        sitting mid-run has pipe on both sides of it, and clearing the full box
        erases that pipe right up to where the line would meet the symbol;
        leaving a margin keeps those stubs, so the detected line ends next to
        the symbol instead of short of it, and graph construction's proximity
        matching has something to match.

        The inset can differ per axis, which is what text needs: a tag set
        vertically gets a tall, narrow box, and where that box runs alongside a
        pipe it erases a long stretch of it. Shrinking only the vertical extent
        shortens the erased stretch without widening the box's reach across the
        pipe.

        An axis whose side is shorter than twice its inset is left at full size
        rather than inverted.

        :param image: The image to clear the bounding boxes from
        :type image: np.ndarray
        :param bounding_boxes: The bounding boxes to clear
        :type bounding_boxes: list[BoundingBox]
        :param inset: Pixels to shrink each box by, either one value for both
            axes or (horizontal, vertical)
        :type inset: int | tuple[int, int]
        :return: The image with the bounding boxes cleared
        '''

        # Compute the histogram of the image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Find the index of the most frequent pixel value
        background_value = int(np.argmax(hist))

        inset_x, inset_y = LineDetectionImagePreprocessor._inset_pair(inset)

        for bb in bounding_boxes:
            top_x, top_y = bb.topX, bb.topY
            bottom_x, bottom_y = bb.bottomX, bb.bottomY

            if inset_x > 0 and bottom_x - top_x > 2 * inset_x:
                top_x, bottom_x = top_x + inset_x, bottom_x - inset_x
            if inset_y > 0 and bottom_y - top_y > 2 * inset_y:
                top_y, bottom_y = top_y + inset_y, bottom_y - inset_y

            points = np.array([[bottom_x, top_y],
                              [bottom_x, bottom_y],
                              [top_x, bottom_y],
                              [top_x, top_y]],
                              np.int32)
            cv2.fillPoly(image, [points], (background_value, background_value, background_value))

        return image

    @staticmethod
    def apply_thinning(image, iterations: float = 0, min_stroke_width: float = 0):
        '''
        Applies the Zhang-Suen thinning algorithm to the given image.

        OpenCV's thinning runs to convergence: every stroke ends up one pixel
        wide, whatever it started at. Strokes on this drawing measure about
        2.3px, so that is a large change in one step, and a partly thinned
        stroke sometimes reads better. Passing a positive `iterations` runs the
        same Zhang-Suen passes but stops after that many, peeling roughly one
        pixel off each side per iteration.

        :param image: The image to apply the thinning algorithm to
        :param iterations: Thinning passes to run; 0 runs to convergence
        :param min_stroke_width: When positive, thin only where the stroke is at
            least this wide and leave narrower strokes untouched; takes
            precedence over `iterations`
        '''
        if min_stroke_width > 0:
            return LineDetectionImagePreprocessor._thin_thick_strokes(
                image, min_stroke_width)

        if iterations <= 0:
            thinningType = cv2.ximgproc.THINNING_ZHANGSUEN
            return cv2.ximgproc.thinning(image, thinningType=thinningType)

        return LineDetectionImagePreprocessor._zhang_suen(
            image, int(round(iterations * 2)))

    @staticmethod
    def _thin_thick_strokes(image, min_stroke_width: float):
        """Thin only the strokes wider than min_stroke_width, keep the rest.

        Thinning is worth it on a pipe, which Hough would otherwise report twice
        -- once down each edge. It is not worth it on a dash, which is already
        one or two pixels wide and only loses length: a 9px dash comes back 7px,
        and the two pixels go straight into the gap the dash rhythm is measured
        from.

        The distance transform gives each ink pixel its distance to the nearest
        background pixel, which is half the local stroke width. Pixels above the
        cut-off are replaced by their skeleton; the rest are left as drawn. The
        thick mask is dilated first so a whole stroke is treated one way rather
        than switching along its own centre line.
        """
        binary = (image > 0).astype(np.uint8)
        half_width = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

        thick = (half_width >= (min_stroke_width / 2.0)).astype(np.uint8)
        if not thick.any():
            return image
        radius = max(1, int(math.ceil(min_stroke_width)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        thick = cv2.dilate(thick, kernel)

        thinned = cv2.ximgproc.thinning(
            image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        out = np.where(thick > 0, thinned, image)
        return out.astype(np.uint8)

    @staticmethod
    def _zhang_suen(image, half_steps: int):
        '''Zhang-Suen thinning stopped after a fixed number of sub-steps.'''
        img = (image > 0).astype(np.uint8)

        def neighbours(padded):
            # P2..P9 clockwise from north, as the algorithm names them
            return [padded[0:-2, 1:-1], padded[0:-2, 2:], padded[1:-1, 2:],
                    padded[2:, 2:], padded[2:, 1:-1], padded[2:, 0:-2],
                    padded[1:-1, 0:-2], padded[0:-2, 0:-2]]

        for half in range(half_steps):
            step = half % 2
            padded = np.pad(img, 1, mode='constant')
            p = neighbours(padded)
            total = sum(p)
            # Transitions from 0 to 1 around the ring
            ring = p + [p[0]]
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                c1 = p[0] * p[2] * p[4]
                c2 = p[2] * p[4] * p[6]
            else:
                c1 = p[0] * p[2] * p[6]
                c2 = p[0] * p[4] * p[6]
            remove = ((img == 1) & (total >= 2) & (total <= 6) &
                      (transitions == 1) & (c1 == 0) & (c2 == 0))
            if not remove.any():
                break
            img[remove] = 0

        return (img * 255).astype(np.uint8)
