"""Built-in analyst specialists.

Importing this package auto-registers all four built-in analysts with the
``AnalystRegistry`` (the registry imports this module on construction).
"""
from __future__ import annotations

from .crypto import CryptoAnalyst
from .niche import NicheAnalyst
from .politics import PoliticsAnalyst
from .sports import SportsAnalyst

__all__ = [
    "CryptoAnalyst",
    "NicheAnalyst",
    "PoliticsAnalyst",
    "SportsAnalyst",
]

# Tuple of (name, class) for the registry to auto-register.
BUILTIN_ANALYSTS: tuple[tuple[str, type], ...] = (
    ("politics", PoliticsAnalyst),
    ("crypto", CryptoAnalyst),
    ("sports", SportsAnalyst),
    ("niche", NicheAnalyst),
)
