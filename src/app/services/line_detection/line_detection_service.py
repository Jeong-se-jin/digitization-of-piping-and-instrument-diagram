# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from logger_config import get_logger
from app.models.text_detection.text_detection_inference_response \
    import TextDetectionInferenceResponse
from app.models.bounding_box import BoundingBox
from app.services.line_detection.line_segments_service \
    import detect_line_segments
from app.services.line_detection.line_type_classifier \
    import classify_line_segments
from app.services.line_detection.line_segment_dedup \
    import deduplicate_line_segments
from app.services.line_detection.axis_filter \
    import keep_axis_aligned_segments
from app.services.line_detection.utils.line_detection_image_preprocessor \
    import LineDetectionImagePreprocessor
from app.utils.image_utils import denormalize_coordinates
from app.models.text_detection.text_recognized import TextRecognized
from app.models.text_detection.symbol_and_text_associated \
    import SymbolAndTextAssociated
import cv2
from typing import Optional, Union
import numpy as np
from app.models.line_detection.line_detection_response \
    import LineDetectionInferenceResponse
from app.config import config
import time

logger = get_logger(__name__)


def detect_lines(
    pid_id: str,
    image_bytes: bytes,
    text_detection_results: TextDetectionInferenceResponse,
    enable_thinning: bool,
    threshold: int,
    max_line_gap: int,
    min_line_length: int,
    rho: float,
    theta_param: float,
    bounding_box_inclusive: Optional[BoundingBox],
    image_height: int,
    image_width: int,
    debug_image_preprocessed_path: Optional[str],
    debug_image_preprocessed_before_thinning_path: Optional[str],
    output_image_line_segments_path: Optional[str],
    symbol_mask_inset: int = config.line_detection_symbol_mask_inset_pixels,
    text_mask_inset_x: int = config.line_detection_text_mask_inset_x_pixels,
    text_mask_inset_y: int = config.line_detection_text_mask_inset_y_pixels,
    classify_line_types: bool = config.classify_line_types,
    thinning_iterations: float = config.line_detection_thinning_iterations,
    thin_min_stroke_width: float = config.line_detection_thin_min_stroke_width,
    deduplicate_segments: bool = config.line_detection_deduplicate_segments,
    axis_aligned_only: bool = config.line_detection_axis_aligned_only,
    mask_text: bool = True,
    axis_tolerance_degrees: float = config.line_detection_axis_tolerance_degrees,
    backend: str = config.line_detection_backend,
    backend_params: Optional[dict] = None,
    binary_threshold: int = config.line_detection_binary_threshold,
    mask_inset_exempt_labels: Optional[set[str]] = None,
    exempt_mask_inset: int = 0
) -> LineDetectionInferenceResponse:
    """
    Detects the line segments in the image using the text detection
    results and request params
    1. Denormalizes the symbol bounding box coordinates
    2. Denormalizes the text bounding box coordinates
    3. Preprocesses the image - clears the symbol and text bounding boxes,
                                converts to grayscale,
                                binarizes and thins(if chosen)
    4. Detects the line segments
    """
    start_time = time.time()
    # denormalize the symbol bounding box coordinates.
    # Symbols whose label is exempt are masked at full size: the inset is there
    # to preserve the pipe stubs meeting a symbol, and a shape with no pipe
    # running through it (an annotation box) would just keep its border ink.
    exempt_labels = mask_inset_exempt_labels or set()
    inset_symbols = [s for s in text_detection_results.text_and_symbols_associated_list
                     if s.label not in exempt_labels]
    exempt_symbols = [s for s in text_detection_results.text_and_symbols_associated_list
                      if s.label in exempt_labels]

    denormalized_symbol_coords = _get_denormalized_items(
        inset_symbols, image_height, image_width)
    denormalized_exempt_symbol_coords = _get_denormalized_items(
        exempt_symbols, image_height, image_width)

    # denormalize the text bounding box coordinates.  With mask_text off the
    # text stays in the image: a pipe that runs under a label is otherwise cut
    # in two, and the two halves stop being one run.  What the letters
    # themselves leave behind has to be filtered downstream.
    denormalized_text_coords = _get_denormalized_items(
        text_detection_results.all_text_list, image_height, image_width) \
        if mask_text else []

    # denormalize bounding_box_inclusive if it is not None
    bounding_box_inclusive_denormalized = \
        _denormalize_bounding_box(
            bounding_box_inclusive, image_height, image_width)

    # preprocess the image first for better line detection
    preprocessed_image = LineDetectionImagePreprocessor.preprocess(
        image_bytes,
        denormalized_symbol_coords,
        denormalized_text_coords,
        symbol_mask_inset,
        (text_mask_inset_x, text_mask_inset_y),
        denormalized_exempt_symbol_coords,
        exempt_mask_inset,
        binary_threshold
    )

    # Keep the un-thinned binary. The dash classifier has to read it, because
    # thinning breaks a solid stroke into something that measures as dashed; the
    # FLD backend needs it too, to measure how thick the drawing is -- on a
    # thinned image every stroke is one pixel and the measurement says nothing.
    pre_thinning_image = preprocessed_image.copy()

    # thin the image if chosen and upload the image before thinning
    # to blob storage for observing the difference
    if enable_thinning:
        # Upload preprocessed image to blob storage for debugging
        if (config.debug):
            try:
                cv2.imwrite(debug_image_preprocessed_before_thinning_path,
                            preprocessed_image)
            except Exception as e:
                logger.error(
                    'Exception while uploading pre-processed image'
                    ' before thinning to blob storage:'
                    f'{e}'
                )

        preprocessed_image = \
            LineDetectionImagePreprocessor.apply_thinning(
                preprocessed_image, thinning_iterations, thin_min_stroke_width)

    # Upload preprocessed image to blob storage for debugging
    if (config.debug):
        try:
            cv2.imwrite(debug_image_preprocessed_path, preprocessed_image)
        except Exception as e:
            logger.error(
                'Exception while uploading pre-processed image to blob storage: '
                f'{e}'
            )

    # detect the line segments using hough transform
    line_segments = detect_line_segments(
        pid_id,
        preprocessed_image,
        image_height,
        image_width,
        max_line_gap,
        threshold,
        min_line_length,
        rho,
        theta_param,
        bounding_box_inclusive_denormalized,
        backend,
        {**(backend_params or {}), 'reference_image': pre_thinning_image}
    )

    # Decode the original image from bytes
    line_segments_output_image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # draw line segments on the original image
    for line_segment in line_segments:
        # denormalize the line segment coordinates
        endX, startX, endY, startY = \
            denormalize_coordinates(
                line_segment.endX,
                line_segment.startX,
                line_segment.endY,
                line_segment.startY,
                image_height,
                image_width
            )

        # green color for each line segment
        color = (0, 155, 0)

        cv2.line(line_segments_output_image, (startX, startY),
                 (endX, endY), color, 2)

    # Upload the image with detected lines to blob storage
    try:
        cv2.imwrite(output_image_line_segments_path, line_segments_output_image)
    except Exception as e:
        logger.error(
            'Exception while uploading image with detected lines'
            ' to blob storage: '
            f'{e}'
        )
    # Runs before deduplication and classification so neither wastes work on
    # segments about to be thrown away.
    if axis_aligned_only:
        before_axis = len(line_segments)
        line_segments = keep_axis_aligned_segments(
            line_segments, image_height, image_width, axis_tolerance_degrees)
        logger.info(f'Axis-aligned filter: {before_axis} -> {len(line_segments)} '
                    f'segments (dropped {before_axis - len(line_segments)})')

    # Runs before classification so the rhythm is measured on one segment per
    # stroke rather than on a stroke's two edges.
    if deduplicate_segments:
        line_segments = deduplicate_line_segments(
            line_segments, image_height, image_width)

    if classify_line_types:
        line_segments = classify_line_segments(
            pre_thinning_image, line_segments, image_height, image_width)

    # log line segments count
    logger.info(f'Line segments detected count: {len(line_segments)}')

    line_service_response = LineDetectionInferenceResponse(
        line_segments=line_segments,
        line_segments_count=len(line_segments),
        image_details=text_detection_results.image_details,
        image_url=text_detection_results.image_url
    )
    end_time = time.time()
    logger.info(f'Time taken for line detection: {end_time - start_time} secs')

    return line_service_response


def _get_denormalized_items(
    item_list: list[Union[TextRecognized, SymbolAndTextAssociated]],
    image_height: int,
    image_width: int
) -> list[BoundingBox]:
    """
    Denormalizes the detected bounding box coordinates.
    """
    denormalized_items = []

    for item in item_list:
        denormalized_item = _denormalize_bounding_box(item, image_height,
                                                      image_width)
        denormalized_items.append(denormalized_item)

    return denormalized_items


def _denormalize_bounding_box(
    bounding_box_inclusive: BoundingBox,
    image_height: int,
    image_width: int
) -> BoundingBox:
    """
    Denormalizes a single bounding box.
    """
    if bounding_box_inclusive is not None:
        bottomX, topX, bottomY, topY = denormalize_coordinates(
            bounding_box_inclusive.bottomX, bounding_box_inclusive.topX,
            bounding_box_inclusive.bottomY, bounding_box_inclusive.topY,
            image_height, image_width
        )

        bounding_box_inclusive_denormalized = BoundingBox(
            bottomX=bottomX,
            topX=topX,
            bottomY=bottomY,
            topY=topY
        )

        return bounding_box_inclusive_denormalized
