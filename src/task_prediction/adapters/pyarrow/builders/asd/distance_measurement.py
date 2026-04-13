import pyarrow as pa
from typing import Any

from ..base import TableDefinition, TIMESTAMP_COL
from .....models.asd.distance_measurement import (
    DistanceMeasurementBase,
    DistanceMeasurementAdded, 
    DistanceMeasurementPositionUpdated,
    DistanceMeasurementRemoved,
    LatLon
)

def build_distance_measurement_cols(batch: list[DistanceMeasurementBase]) -> dict[str, list[Any]]:
    size = len(batch)
    
    timestamp, m_id = [None] * size, [None] * size
    a_f_lat, a_f_lon, a_f_cs = [None] * size, [None] * size, [None] * size
    a_s_lat, a_s_lon, a_s_cs = [None] * size, [None] * size, [None] * size
    u_s_x, u_s_y, u_e_x, u_e_y = [None] * size, [None] * size, [None] * size, [None] * size
    removed = [None] * size

    for i, row in enumerate(batch):
        timestamp[i], m_id[i] = row.timestamp, row.measurement_id

        if isinstance(row, DistanceMeasurementAdded):
            if isinstance(row.first, LatLon):
                a_f_lat[i], a_f_lon[i] = row.first.lat_deg, row.first.lon_deg
            else:
                a_f_cs[i] = row.first
            
            if isinstance(row.second, LatLon):
                a_s_lat[i], a_s_lon[i] = row.second.lat_deg, row.second.lon_deg
            else:
                a_s_cs[i] = row.second

        elif isinstance(row, DistanceMeasurementPositionUpdated):
            u_s_x[i], u_s_y[i] = row.start.x, row.start.y
            u_e_x[i], u_e_y[i] = row.end.x, row.end.y

        elif isinstance(row, DistanceMeasurementRemoved):
            removed[i] = True

    return {
        "timestamp": timestamp,
        "measurement_id": m_id,
        "added__first__lat_deg": a_f_lat,
        "added__first__lon_deg": a_f_lon,
        "added__first__callsign": a_f_cs,
        "added__second__lat_deg": a_s_lat,
        "added__second__lon_deg": a_s_lon,
        "added__second__callsign": a_s_cs,
        "position_updated__start__x": u_s_x,
        "position_updated__start__y": u_s_y,
        "position_updated__end__x": u_e_x,
        "position_updated__end__y": u_e_y,
        "removed": removed
    }

DISTANCE_MEASUREMENT_DEFINITION: TableDefinition[DistanceMeasurementBase] = TableDefinition(
    name="distance_measurement",
    schema=pa.schema([
        TIMESTAMP_COL,
        pa.field("measurement_id", pa.uint32(), nullable=False),
        pa.field("added__first__lat_deg", pa.float64()),
        pa.field("added__first__lon_deg", pa.float64()),
        pa.field("added__first__callsign", pa.string()),
        pa.field("added__second__lat_deg", pa.float64()),
        pa.field("added__second__lon_deg", pa.float64()),
        pa.field("added__second__callsign", pa.string()),
        pa.field("position_updated__start__x", pa.int32()),
        pa.field("position_updated__start__y", pa.int32()),
        pa.field("position_updated__end__x", pa.int32()),
        pa.field("position_updated__end__y", pa.int32()),
        pa.field("removed", pa.bool_()),
    ]),
    extractor=build_distance_measurement_cols,
)