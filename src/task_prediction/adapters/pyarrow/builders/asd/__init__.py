from .aware_action_interaction import AWARE_ACTION_INTERACTION_DEFINITION
from .clearance import CLEARANCE_DEFINITION
from .distance_measurement import DISTANCE_MEASUREMENT_DEFINITION
from .keyboard_shortcut import KEYBOARD_SHORTCUT_DEFINITION
from .mouse_position import MOUSE_POSITION_DEFINITION
from .pop_up import POPUP_DEFINITION
from .route_interaction import ROUTE_INTERACTION_DEFINITION
from .sep_tool import SEP_TOOL_DEFINITION
from .speed_vector import SPEED_VECTOR_DEFINITION
from .track_label_position import TRACK_LABEL_POSITION_DEFINITION
from .track_mark import TRACK_MARK_DEFINITION
from .track_screen_position import TRACK_SCREEN_POSITION_DEFINITION
from .transfer import TRANSFER_DEFINITION

from typing import Final
from ..base import TableDefinition
from .....models.asd import *

ASD_EVENT_DEFINITIONS: Final[dict[type[AsdEvent], TableDefinition]] = {
    AwareActionInteraction: AWARE_ACTION_INTERACTION_DEFINITION,
    Clearance: CLEARANCE_DEFINITION,
    DistanceMeasurementBase: DISTANCE_MEASUREMENT_DEFINITION,
    KeyboardShortcut: KEYBOARD_SHORTCUT_DEFINITION,
    MousePosition: MOUSE_POSITION_DEFINITION,
    Popup: POPUP_DEFINITION,
    RouteInteraction: ROUTE_INTERACTION_DEFINITION,
    SepToolBase: SEP_TOOL_DEFINITION,
    SpeedVectorBase: SPEED_VECTOR_DEFINITION,
    TrackLabelPosition: TRACK_LABEL_POSITION_DEFINITION,
    TrackMark: TRACK_MARK_DEFINITION,
    TrackScreenPosition: TRACK_SCREEN_POSITION_DEFINITION,
    Transfer: TRANSFER_DEFINITION
}