from datetime import datetime
from dataclasses import dataclass
from typing import Any

from ..screen_position import ScreenPosition

@dataclass(frozen=True, slots=True)
class LatLon:
    lat_deg: float
    lon_deg: float

@dataclass(frozen=True, slots=True)
class DistanceMeasurementBase:
    timestamp: datetime
    measurement_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "added__first__lat_lon__lat_deg": None,
            "added__first__lat_lon__lon_deg": None,
            "added__first__callsign": None,
            "added__second__lat_lon__lat_deg": None,
            "added__second__lat_lon__lon_deg": None,
            "added__second__callsign": None,
            "position_updated__start__x": None,
            "position_updated__start__y": None,
            "position_updated__end__x": None,
            "position_updated__end__y": None,
        }

@dataclass(frozen=True, slots=True)
class DistanceMeasurementAdded(DistanceMeasurementBase):
    first: LatLon | str
    second: LatLon | str

    def _add_point_to_dict(self, prefix: str, point: LatLon | str, d: dict[str, Any]) -> None:
        if isinstance(point, LatLon):
            d[f"added__{prefix}__lat_lon__lat_deg"] = point.lat_deg
            d[f"added__{prefix}__lat_lon__lon_deg"] = point.lon_deg
            d[f"added__{prefix}__callsign"] = None
        else:
            d[f"added__{prefix}__lat_lon__lat_deg"] = None
            d[f"added__{prefix}__lat_lon__lon_deg"] = None
            d[f"added__{prefix}__callsign"] = point

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        
        self._add_point_to_dict("first", self.first, d)
        self._add_point_to_dict("second", self.second, d)
        
        return d

@dataclass(frozen=True, slots=True)
class DistanceMeasurementRemoved(DistanceMeasurementBase):
    pass

@dataclass(frozen=True, slots=True)
class DistanceMeasurementPositionUpdated(DistanceMeasurementBase):
    start: ScreenPosition
    end: ScreenPosition

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()

        d["position_updated__start__x"] = self.start.x
        d["position_updated__start__y"] = self.start.y
        d["position_updated__end__x"] = self.end.x
        d["position_updated__end__y"] = self.end.y
        
        return d