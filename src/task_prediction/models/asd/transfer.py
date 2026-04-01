from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum

class TransferType(IntEnum):
    TRANSFER = 0
    ASSUME = 1
    FORCE_ASSUME = 2
    RELEASE = 3
    REJECT_TRANSFER = 4
    REQUEST_TRANSFER = 5
    CANCEL_TRANSFER = 6
    ACTIVATE_NEXT_SECTOR = 7
    FORCE_ACT = 8
    DECONTROL = 9
    TRANSFER_TO_NEXT_SECTOR = 10
    FORCE_RELEASE = 11
    ENABLE_AUTO_CONTROL = 12
    TRANSFER_TO_ANY = 13
    MANUAL_OUTBOUND = 14
    MANUAL_INBOUND = 15

@dataclass(frozen=True, slots=True)
class Transfer:
    timestamp: datetime
    callsign: str

    type: TransferType