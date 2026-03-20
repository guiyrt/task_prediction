from datetime import datetime
from dataclasses import dataclass

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class LatLon:
    lat_deg: float
    lon_deg: float

@dataclass(frozen=True, slots=True)
class DistanceMeasurementBase:
    timestamp: datetime
    measurement_id: int

@dataclass(frozen=True, slots=True)
class DistanceMeasurementAdded(DistanceMeasurementBase):
    first: LatLon | str
    second: LatLon | str

@dataclass(frozen=True, slots=True)
class DistanceMeasurementRemoved(DistanceMeasurementBase):
    pass

@dataclass(frozen=True, slots=True)
class DistanceMeasurementPositionUpdated(DistanceMeasurementBase):
    start: ScreenPosition
    end: ScreenPosition