import logging
from typing import Final
from datetime import datetime, timezone

import zlib
import google.protobuf.json_format
from aware_protos.tern.asd.events import asd_events_pb2
from aware_protos.aware.proto import messages_pb2

from ....models.asd import *
from ....models.screen_position import ScreenPosition

logger = logging.getLogger(__name__)

LINUX_TOP_BAR_HEIGHT: Final[int] = 27

def _parse_measurement_point(mp: asd_events_pb2.MeasurementPoint) -> LatLon | str:
    return (
        LatLon(mp.lat_lon.lat_deg, mp.lat_lon.lon_deg)
        if mp.HasField("lat_lon")
        else _parse_callsign(mp.flight_id)
    )

def _parse_callsign(fid: asd_events_pb2.FlightIdentifier) -> str:
    if fid.callsign:
        return fid.callsign
    
    elif fid.track_number:
        identifier = f"TrackNumber{fid.track_number}"
        logger.debug("`callsign` not present, falling back to `track_number`: %s", identifier)
        return identifier
    
    elif fid.uuid:
        identifier = f"UUID{fid.uuid}"
        logger.debug("`callsign` and `track_number` not present, falling back to `uuid`: %s", identifier)
        return identifier
    
    else:
        logger.critical("No FlightIdentifier data available, callsign will be empty.")
        return ""

def parse_asd_proto(payload: bytes, decompress: bool = False) -> AsdEvent | None:
    if decompress:
        payload = zlib.decompress(payload)
    
    event = google.protobuf.json_format.Parse(
        payload.decode(),
        messages_pb2.Event(),
        ignore_unknown_fields=True
    )

    if event.WhichOneof("payload") != "asd_event":
            return None

    try:
        asd_payload: asd_events_pb2.Event = event.asd_event
        timestamp: datetime = event.timestamp.ToDatetime(timezone.utc)

        match event_name := asd_payload.WhichOneof("event"):
            case "aware_action_interaction":
                p = asd_payload.aware_action_interaction
                return AwareActionInteraction(
                    timestamp, _parse_callsign(p.flight_id), p.action_uuid, AwareActionStatus(p.action_status), p.action_details,
                    p.suggestion_mode_enabled
                )
            
            case "clearance":
                p = asd_payload.clearance
                return Clearance(timestamp, _parse_callsign(p.flight_id), ClearanceType(p.clearance_type), p.clearance)

            case "distance_measurement":
                if asd_payload.distance_measurement.HasField("added"):
                    p = asd_payload.distance_measurement.added

                    return DistanceMeasurementAdded(
                        timestamp, p.measurement_id, _parse_measurement_point(p.first), _parse_measurement_point(p.second)
                    )

                elif asd_payload.distance_measurement.HasField("position_updated"):
                    p = asd_payload.distance_measurement.position_updated
                    return DistanceMeasurementPositionUpdated(
                        timestamp, p.measurement_id, ScreenPosition(p.start.x, p.start.y + LINUX_TOP_BAR_HEIGHT),
                        ScreenPosition(p.end.x, p.end.y + LINUX_TOP_BAR_HEIGHT)
                    )

                elif asd_payload.distance_measurement.HasField("removed"):
                    p = asd_payload.distance_measurement.removed
                    return DistanceMeasurementRemoved(timestamp, p.measurement_id)

            case "keyboard_shortcut":
                p = asd_payload.keyboard_shortcut
                return KeyboardShortcut(timestamp, p.action_name)
        
            case "mouse_position":
                p = asd_payload.mouse_position
                return MousePosition(timestamp, ScreenPosition(p.x, p.y + LINUX_TOP_BAR_HEIGHT))

            case "popup":
                p = asd_payload.popup
                return Popup(timestamp, _parse_callsign(p.flight_id), PopupMenu(p.name), p.opened)

            case "route_interaction":
                p = asd_payload.route_interaction
                return RouteInteraction(timestamp, _parse_callsign(p.flight_id), RouteInteractionType(p.action_type), p.value)

            case "sep_tool":
                p = asd_payload.sep_tool

                if p.HasField("opened"):
                    return SepToolOpened(timestamp, SepToolType(p.type), p.measurement_id, p.opened.flight_id.callsign)
                
                elif p.HasField("connected"):
                    return SepToolConnected(
                        timestamp, SepToolType(p.type), p.measurement_id, p.connected.flight_id_1.callsign,
                        p.connected.flight_id_2.callsign
                    )

                elif p.closed:
                    return SepToolClosed(timestamp, SepToolType(p.type), p.measurement_id, p.closed)
                
            case "speed_vector":
                if asd_payload.speed_vector.HasField("mode_updated"):
                    p = asd_payload.speed_vector.mode_updated
                    return SpeedVectorModeUpdated(timestamp, SpeedVectorMode(p.mode))
                
                elif asd_payload.speed_vector.HasField("visibility"):
                    p = asd_payload.speed_vector.visibility
                    return SpeedVectorVisibility(timestamp, p.visible, _parse_callsign(p.flight_id))
                
                elif asd_payload.speed_vector.HasField("length"):
                    p = asd_payload.speed_vector.length
                    return SpeedVectorLength(timestamp, p.length_seconds)
            
            case "track_label_position":
                p = asd_payload.track_label_position
                return TrackLabelPosition(
                    timestamp, _parse_callsign(p.flight_id), ScreenPosition(p.x, p.y + LINUX_TOP_BAR_HEIGHT), p.width,
                    p.height, p.visible, p.hovered, p.selected, p.on_pip
                )
            
            case "track_mark":
                p = asd_payload.track_mark
                return TrackMark(
                    timestamp, _parse_callsign(p.flight_id), TrackMarkType(p.mark_type), TrackMarkVariant(p.mark_variant),
                    p.mark_scope, p.mark_set
                )
            
            case "track_screen_position":
                p = asd_payload.track_screen_position
                return TrackScreenPosition(
                    timestamp, _parse_callsign(p.flight_id), ScreenPosition(p.x, p.y + LINUX_TOP_BAR_HEIGHT), p.visible
                )
            
            case "transfer":
                p = asd_payload.transfer

                if p.transfer_type == 0:
                    return None

                return Transfer(timestamp, _parse_callsign(p.flight_id), TransferType(p.transfer_type - 1))
            
            case _:
                logger.critical("Event %s not recognized.", event_name)
                return None
    
    except Exception as e:
        logger.warning("Failed to parse ASD protobuf event: %s. Error: %s", event_name, e)
        return None