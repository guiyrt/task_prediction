from datetime import datetime
from dataclasses import dataclass
from enum import IntEnum

class TransferType(IntEnum):
    TRANSFER_TYPE_NOT_SET = 0
    TRANSFER = 1
    ASSUME = 2
    FORCE_ASSUME = 3
    RELEASE = 4
    REJECT_TRANSFER = 5
    REQUEST_TRANSFER = 6
    CANCEL_TRANSFER = 7
    ACTIVATE_NEXT_SECTOR = 8
    FORCE_ACT = 9
    DECONTROL = 10
    TRANSFER_TO_NEXT_SECTOR = 11
    FORCE_RELEASE = 12
    ENABLE_AUTO_CONTROL = 13
    TRANSFER_TO_ANY = 14
    MANUAL_OUTBOUND = 15
    MANUAL_INBOUND = 16

@dataclass(frozen=True, slots=True)
class Transfer:
    timestamp: datetime
    callsign: str

    type: TransferType