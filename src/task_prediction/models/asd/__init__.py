from .aware_action_interaction import AwareActionInteraction, AwareActionStatus
from .clearance import Clearance, ClearanceType
from .distance_measurement import (
    DistanceMeasurementBase,
    DistanceMeasurementAdded,
    DistanceMeasurementPositionUpdated,
    DistanceMeasurementRemoved,
    LatLon
)
from .keyboard_shortcut import KeyboardShortcut
from .mouse_position import MousePosition
from .pop_up import Popup, PopupMenu
from .route_interaction import RouteInteraction, RouteInteractionType
from .sep_tool import (
    SepToolBase,
    SepToolOpened,
    SepToolConnected,
    SepToolClosed,
    SepToolType
)
from .speed_vector import (
    SpeedVectorBase,
    SpeedVectorModeUpdated,
    SpeedVectorLength,
    SpeedVectorVisibility,
    SpeedVectorMode
)
from .track_label_position import TrackLabelPosition
from .track_mark import TrackMark, TrackMarkType, TrackMarkVariant
from .track_screen_position import TrackScreenPosition
from .transfer import Transfer, TransferType


from typing import TypeAlias

type AsdEvent = (
    AwareActionInteraction | Clearance | DistanceMeasurementBase | 
    KeyboardShortcut | MousePosition | Popup | RouteInteraction | 
    SepToolBase | SpeedVectorBase | TrackLabelPosition | 
    TrackMark | TrackScreenPosition | Transfer
)