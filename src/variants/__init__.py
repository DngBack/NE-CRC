"""System variants module."""

from .unicr_filter import (
    UniCRWithFilter,
    create_unicr_with_filter,
)
from .unicr_necrc import (
    UniCRWithNECRC,
    create_unicr_with_necrc,
)
from .s_unicr import (
    ShiftAwareUniCR,
    create_s_unicr,
)

__all__ = [
    # UniCR + Filter
    "UniCRWithFilter",
    "create_unicr_with_filter",
    # UniCR + NE-CRC
    "UniCRWithNECRC",
    "create_unicr_with_necrc",
    # S-UniCR (full system)
    "ShiftAwareUniCR",
    "create_s_unicr",
]

