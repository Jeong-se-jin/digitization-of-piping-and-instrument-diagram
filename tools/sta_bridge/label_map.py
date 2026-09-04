# Maps STA-main symbol classes (dataset.yaml) onto the hierarchical label
# taxonomy that the graph-construction stage branches on.
#
# Why this file exists: graph construction does *string prefix matching* on the
# label, not class-id lookup.  See app/config.py --
#
#   arrow_symbol_label ............ 'Piping/Fittings/Mid arrow flow direction'
#                                   (exact match; drives arrow direction detection)
#   valve_symbol_prefix ........... 'Instrument/Valve/'
#   symbol_label_prefixes_with_text 'Equipment/', 'Instrument/', 'Piping/Endpoint/Pagination'
#   symbol_label_for_connectors ... 'Piping/Endpoint/Pagination'
#   symbol_label_prefixes_to_connect_if_close
#                                   'Equipment', 'Instrument/Valve/',
#                                   'Piping/Fittings/Mid arrow flow direction',
#                                   'Piping/Fittings/Flanged connection'
#
# A class mapped to None is dropped from the export entirely.  That matters
# for more than tidiness: every exported symbol bbox gets painted over with the
# background colour before Hough runs, so mapping a large non-asset region
# (a title block, an insulation span) would erase real pipe runs with it.

STA_TO_PID_LABEL: dict[str, str | None] = {
    # --- not a detection -----------------------------------------------
    'Not_used': None,

    # --- valves ---------------------------------------------------------
    'Gate_Valve':             'Instrument/Valve/Gate valve NO',
    'Gate_valve_NO':          'Instrument/Valve/Gate valve NO',
    'Gate_Valve_NC':          'Instrument/Valve/Gate valve NC',
    'Half_Filled_Gate_Valve': 'Instrument/Valve/Gate valve NO',
    'Ball_Valve':             'Instrument/Valve/Ball valve NO',
    'Ball_valve_NC':          'Instrument/Valve/Ball valve NC',
    'Globe_valve_NO':         'Instrument/Valve/Globe valve NO',
    'Globle_valve_NC':        'Instrument/Valve/Globe valve NC',
    'Butterfly_valve':        'Instrument/Valve/Butterfly Valve',
    'Plug valve':             'Instrument/Valve/Plug valve',
    'Check_valve':            'Instrument/Valve/Check valve',
    'Diaphragm_valve':        'Instrument/Valve/Diaphragm valve',
    'Needle_valve':           'Instrument/Valve/Needle Valve',
    'Control_Valve':          'Instrument/Valve/Control Valve',
    'Rotary_Valve':           'Instrument/Valve/Rotary valve NO',
    'Rupture_disk':           'Instrument/Valve/Relief valve',

    # --- piping fittings -------------------------------------------------
    # Flow_Arrow MUST land on config.arrow_symbol_label verbatim, otherwise
    # arrow-direction detection silently finds zero arrows.
    'Flow_Arrow':              'Piping/Fittings/Mid arrow flow direction',

    # The VLM path lets a reader answer "Other" for a symbol outside the 32
    # classes -- an equipment outline, a table, something it can name but not
    # classify. There is no label here that would mean anything downstream, so
    # it is dropped rather than guessed at; the reader's own words survive in
    # the tile answer.
    'Other':                   None,
    'Flange_or_Nozzle':        'Piping/Fittings/Flanged connection',
    'Reducer':                 'Piping/Fittings/Reducer',
    'Paddle_blind':            'Piping/Fittings/Paddle blind',
    'Spectacle_blind_Open':    'Piping/Fittings/Spectacle blind open',
    'Spectacle_blind_Closed':  'Piping/Fittings/Spectacle blind closed',

    # --- instruments -----------------------------------------------------
    'Instrument_Field':     'Instrument/Indicator/Field mounted discrete indicator',
    'Instrument_Panel':     'Instrument/Indicator/discrete with Pri',
    'Instrument_Aux_Panel': 'Instrument/Indicator/discrete with Aux Loc access',
    'sight_glass':          'Instrument/Indicator/Sight glass',

    # 'box' is the square panel-mounted instrument, not an annotation
    # rectangle: the class carries tags like FICA 101B, TICA 121A, TIA 204A,
    # and its boxes measure about 59x58px, the same as the other symbols. It is
    # mapped exactly as Instrument_Panel. Treating it as an annotation left 46
    # tagged instruments out of the graph, and with them the dashed signal
    # lines that run from a bubble to its panel box, which then had nothing to
    # connect to.
    # Its own leaf under Instrument/Indicator/ rather than the Instrument_Panel
    # label verbatim: everything graph construction branches on keys off the
    # 'Instrument/' prefix, so this behaves exactly like a panel instrument,
    # while staying distinguishable here so its mask can use a smaller inset.
    'box': 'Instrument/Indicator/Panel box',

    # Spans a pipe run rather than sitting on one, so its mask does eat into the
    # line underneath -- but dropping it removed a symbol the detector actually
    # found, and a detected symbol should reach the graph rather than vanish
    # here. Give it a small inset if the masking becomes a problem.
    'Pipe_Insulation_or_Tracing': 'Piping/Fittings/Insulation or tracing',
}

# Labels masked with their own inset instead of the symbol inset. A panel box is
# a drawn rectangle, so most of the symbol inset would be spent leaving its own
# border ink behind; a single pixel is enough to keep a signal line that meets
# the edge.
NO_MASK_INSET_LABELS = {'Instrument/Indicator/Panel box'}

# STA's checkpoint has no off-page connector class, so
# config.symbol_label_for_connectors ('Piping/Endpoint/Pagination') never
# matches.  Consequence: pipes that run off the sheet terminate as dangling
# line ends instead of pagination nodes, and cross-sheet stitching is
# unavailable.  Add a connector class to the detector to fix this properly.
UNMAPPED_TAXONOMY_GAPS = ('Piping/Endpoint/Pagination', 'Equipment/')


def map_label(sta_class: str) -> str | None:
    """Translate one STA class name; returns None for classes to drop."""
    if sta_class not in STA_TO_PID_LABEL:
        raise KeyError(
            f'STA class {sta_class!r} has no entry in STA_TO_PID_LABEL. '
            f'Add it (or map it to None to drop it).')
    return STA_TO_PID_LABEL[sta_class]
