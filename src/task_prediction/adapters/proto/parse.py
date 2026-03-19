import logging
from datetime import datetime, timezone

from aware_protos.tern.asd.events import asd_events_pb2
from aware_protos.aware.proto import messages_pb2

from ...models.asd import *
from ...models.screen_position import ScreenPosition

logger = logging.getLogger(__name__)

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
        logger.warning("`callsign` not present, falling back to `track_number`: %s", identifier)
        return identifier
    
    elif fid.uuid:
        identifier = f"UUID{fid.uuid}"
        logger.warning("`callsign` and `track_number` not present, falling back to `uuid`: %s", identifier)
        return identifier
    
    else:
        logger.critical("No FlightIdentifier data available, callsign will be empty.")
        return ""

def parse_asd_proto(event: messages_pb2.Event) -> AsdEvent | None:
    if event.WhichOneof("payload") != "asd_event":
            return None

    try:
        payload: asd_events_pb2.Event = event.asd_event
        timestamp: datetime = event.timestamp.ToDatetime(timezone.utc)

        match event_name := payload.WhichOneof("event"):
            case "aware_action_interaction":
                p = payload.aware_action_interaction
                return AwareActionInteraction(
                    timestamp, _parse_callsign(p.flight_id), p.action_uuid, AwareActionStatus(p.action_status), p.action_details,
                    p.suggestion_mode_enabled
                )
            
            case "clearance":
                p = payload.clearance
                return Clearance(timestamp, _parse_callsign(p.flight_id), ClearanceType(p.clearance_type), p.clearance)

            case "distance_measurement":
                if payload.distance_measurement.HasField("added"):
                    p = payload.distance_measurement.added

                    return DistanceMeasurementAdded(
                        timestamp, p.measurement_id, _parse_measurement_point(p.first), _parse_measurement_point(p.second)
                    )

                elif payload.distance_measurement.HasField("position_updated"):
                    p = payload.distance_measurement.position_updated
                    return DistanceMeasurementPositionUpdated(
                        timestamp, p.measurement_id, ScreenPosition(p.start.x, p.start.y),
                        ScreenPosition(p.end.x, p.end.y)
                    )

                elif payload.distance_measurement.HasField("removed"):
                    p = payload.distance_measurement.removed
                    return DistanceMeasurementRemoved(timestamp, p.measurement_id)

            case "keyboard_shortcut":
                p = payload.keyboard_shortcut
                return KeyboardShortcut(timestamp, p.action_name)
        
            case "mouse_position":
                p = payload.mouse_position
                return MousePosition(timestamp, ScreenPosition(p.x, p.y))

            case "popup":
                p = payload.popup
                return Popup(timestamp, _parse_callsign(p.flight_id), PopupMenu(p.name), p.opened)

            case "route_interaction":
                p = payload.route_interaction
                return RouteInteraction(timestamp, _parse_callsign(p.flight_id), RouteInteractionType(p.action_type), p.value)

            case "sep_tool":
                p = payload.sep_tool

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
                if payload.speed_vector.HasField("mode_updated"):
                    p = payload.speed_vector.mode_updated
                    return SpeedVectorModeUpdated(timestamp, SpeedVectorMode(p.mode))
                
                elif payload.speed_vector.HasField("visibility"):
                    p = payload.speed_vector.visibility
                    return SpeedVectorVisibility(timestamp, p.visible, _parse_callsign(p.flight_id))
                
                elif payload.speed_vector.HasField("length"):
                    p = payload.speed_vector.length
                    return SpeedVectorLength(timestamp, p.length_seconds)
            
            case "track_label_position":
                p = payload.track_label_position
                return TrackLabelPosition(
                    timestamp, _parse_callsign(p.flight_id), ScreenPosition(p.x, p.y), p.width, p.height, p.visible, p.hovered,
                    p.selected, p.on_pip
                )
            
            case "track_mark":
                p = payload.track_mark
                return TrackMark(
                    timestamp, _parse_callsign(p.flight_id), TrackMarkType(p.mark_type), TrackMarkVariant(p.mark_variant),
                    p.mark_scope, p.mark_set
                )
            
            case "track_screen_position":
                p = payload.track_screen_position
                return TrackScreenPosition(timestamp, _parse_callsign(p.flight_id), ScreenPosition(p.x, p.y), p.visible)
            
            case "transfer":
                p = payload.transfer
                return Transfer(timestamp, _parse_callsign(p.flight_id), TransferType(p.transfer_type))
            
            case _:
                logger.critical("Event %s not recognized.", event_name)
                return None
    
    except Exception as e:
        logger.critical("Failed to parse ASD protobuf event: %s. Error: %s", event_name, e)
        return None