# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pydantic import BaseSettings, root_validator, validator

from typing import Union, Optional


class Config(BaseSettings):
    arrow_symbol_label: str = 'Piping/Fittings/Mid arrow flow direction'
    blob_storage_account_url: str = str()
    blob_storage_container_name: str = str()
    centroid_distance_threshold: float = 0.5
    # Treat every detected symbol as a graph asset, instead of only those whose
    # associated text is a plausible tag. Turns the graph into "what the
    # detector found" rather than "what was cleanly tagged".
    treat_all_symbols_as_assets: bool = True
    # Also make flow arrows assets. They are normally skipped, being direction
    # markers rather than equipment, but the detector splits the 45-degree
    # filled bow-tie valve between Gate_Valve_NC and Flow_Arrow, so the arrow
    # class carries real valves. Arrows still drive flow direction as before;
    # this only adds them as nodes. Requires treat_all_symbols_as_assets.
    treat_arrows_as_assets: bool = True
    debug: bool = False
    detect_dotted_lines: bool = False
    enable_preprocessing_text_detection: bool = True
    enable_thinning_preprocessing_line_detection: bool = True
    flow_direction_asset_prefixes: Union[str, set[str]] = \
        {'Equipment/', 'Piping/Endpoint/Pagination'}
    form_recognizer_endpoint: str = str()
    graph_db_authenticate_with_azure_ad: bool = False
    graph_db_connection_string: str = str()
    graph_distance_threshold_for_lines_pixels: int = 50
    graph_distance_threshold_for_symbols_pixels: int = 12
    # Once a symbol or text has claimed a start/end candidate, do not let a line
    # segment displace it, however much closer that line is. Symbols are matched
    # before lines and under a 12px threshold against the lines' 50px, so a
    # scrap of line lying nearer than the valve at the end of the pipe would
    # otherwise take the one candidate the endpoint has to give.
    graph_prefer_symbol_over_line: bool = False
    # A symbol or text sitting this close to an endpoint is treated as touching
    # it, and a line segment may no longer displace it. Unlike
    # graph_prefer_symbol_over_line this is a contact rule, not a blanket one:
    # a symbol the pipe merely passes can still lose its candidate to the next
    # fragment of the run, which is what keeps a pipe running through a valve.
    # 0 disables. Measured on the fire-protection sheet: 329 endpoints had a
    # symbol touching them and lost it to a line, 254 of those to a line more
    # than 5px away.
    graph_symbol_contact_pixels: float = 2.0
    # Fallback for a line endpoint that found no candidate: keep going along the
    # line's own direction and attach to the first symbol it runs into, up to
    # this far. Nothing else would ever pick that endpoint up.
    graph_ray_cast_to_symbol_pixels: int = 60
    graph_distance_threshold_for_text_pixels: int = 5
    graph_line_buffer_pixels: int = 5
    graph_symbol_to_symbol_distance_threshold_pixels: int = 10
    graph_symbol_to_symbol_overlap_region_threshold: float = 0.7
    inference_score_threshold: float = 0.5
    inference_service_retry_count: int = 3
    inference_service_retry_backoff_factor: float = 0.3
    line_detection_hough_max_line_gap: Optional[int] = None  # Note conditional validation below based on detect_dotted_lines
    line_detection_hough_min_line_length: Optional[int] = 10  # Note conditional validation below based on detect_dotted_lines
    # line_detection_hough_max_line_gap value helps with returning the smaller dashed line segments
    # into single solid line segments wherever the dashed line segments are detected by Hough.
    # The default value will not work for all the images.
    # This is something that is good to start with but has to be adjusted based on the images dashed lines
    # in the graph construction post api request
    # Shrink each symbol/text mask box by this many pixels before clearing it,
    # so the pipe stubs meeting a symbol survive into Hough. 0 clears the full
    # box, which is the original behaviour.
    line_detection_symbol_mask_inset_pixels: int = 3
    line_detection_text_mask_inset_x_pixels: int = 4
    line_detection_text_mask_inset_y_pixels: int = 0
    # Label each detected segment solid (pipe) or dashed (instrument signal).
    classify_line_types: bool = True
    # Thinning passes before Hough; 0 thins to a one-pixel skeleton.
    line_detection_thinning_iterations: float = 0
    # FLD merges collinear neighbours by default, which swallows the dashes a
    # signal line is made of; off, its dash count matches Hough's and beats it.
    line_detection_fld_merge: bool = False
    # Thin only strokes at least this wide, leaving dashes at full length.
    # 0 thins everything.
    line_detection_thin_min_stroke_width: float = 0
    # Drop segments that re-detect a stroke another segment already covers. The
    # alternative to thinning, which shortens every dash by eroding its ends.
    # Keep only segments running horizontally or vertically, within
    # line_detection_axis_tolerance_degrees. P&ID piping is drawn on the axes;
    # what slants is usually a leader line, a hatch, or a scrap off a symbol.
    # Discards the diagonals rather than straightening them.
    line_detection_axis_aligned_only: bool = False
    line_detection_axis_tolerance_degrees: float = 5.0
    line_detection_deduplicate_segments: bool = True
    # Raw line detector: 'hough' (original), 'fld' or 'lsd'.
    # Grey value below which a pixel is ink. 0 falls back to Otsu, which picks
    # badly on a page that is mostly white and drops light-grey strokes.
    line_detection_binary_threshold: int = 240
    line_detection_backend: str = 'fld'
    line_detection_hough_rho: float = 0.1
    line_detection_hough_theta: int = 1080
    line_detection_hough_threshold: int = 5
    line_detection_job_timeout_seconds: int = 300
    line_segment_padding_default: float = 0.2

    port: int = 8000
    symbol_detection_api: str = str()
    symbol_detection_api_bearer_token: str = str()
    symbol_label_prefixes_to_connect_if_close: Union[str, set[str]] = \
        {'Equipment', 'Instrument/Valve/', 'Piping/Fittings/Mid arrow flow direction', 'Piping/Fittings/Flanged connection'}
    symbol_label_prefixes_to_include_in_graph_image_output: Union[str, set[str]] = \
        {'Equipment/', 'Instrument/Valve/', 'Piping/Endpoint/Pagination'}
    symbol_label_prefixes_with_text: Union[str, set[str]] = \
        {'Equipment/', 'Instrument/', 'Piping/Endpoint/Pagination'}
    symbol_overlap_threshold: float = 0.6
    text_detection_area_intersection_ratio_threshold: float = 0.8
    text_detection_distance_threshold: float = 0.01
    symbol_label_for_connectors: Union[str, set[str]] = \
        {'Piping/Endpoint/Pagination'}
    valve_symbol_prefix: str = 'Instrument/Valve/'
    workers_count_for_data_batch: int = 3

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    @validator(
        "blob_storage_account_url",
        "blob_storage_container_name",
        "form_recognizer_endpoint",
        "symbol_detection_api",
        "symbol_detection_api_bearer_token",
        "graph_db_connection_string",
        allow_reuse=True)
    def validate_string(cls, v):
        if v is None or len(v) == 0:
            raise ValueError("Value must be a non-empty string")
        return v

    @validator(
            "flow_direction_asset_prefixes",
            "symbol_label_prefixes_with_text",
            "symbol_label_prefixes_to_include_in_graph_image_output",
            "symbol_label_prefixes_to_connect_if_close",
            pre=True,
            allow_reuse=True)
    def validate_and_transform_comma_separated_list(cls, val):
        if isinstance(val, str):
            val_arr = val.split(',')
            val_arr = [x.strip() for x in val_arr]
            return set(val_arr)
        return val

    @root_validator(allow_reuse=True)
    def update_config_based_on_dotted_lines_detection(cls, values):
        if values.get('detect_dotted_lines') is True:
            values['line_detection_hough_min_line_length'] = None

            if values['line_detection_hough_max_line_gap'] is None:
                values['line_detection_hough_max_line_gap'] = 10
        elif values.get('detect_dotted_lines') is False:
            if values['line_detection_hough_min_line_length'] is None or \
                    values['line_detection_hough_min_line_length'] < 10:
                values['line_detection_hough_min_line_length'] = 10

            values['line_detection_hough_max_line_gap'] = None

        return values


config = Config()
