"""pythia-delphi-adapter — Gensyn Delphi ATT integration for Pythia.

This package provides a typed, async, retry-safe Python client for the
Gensyn Agentic Trading Toolkit (ATT), the entrypoint Pythia uses to
participate in Delphi information markets.

ATT docs: https://docs.gensyn.ai/tech/agentic-trading
"""

from pythia_delphi_adapter.client import DelphiClient
from pythia_delphi_adapter.models import (
    Market,
    MarketStatus,
    OrderSide,
    TradeReceipt,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DelphiClient",
    "Market",
    "MarketStatus",
    "OrderSide",
    "TradeReceipt",
]
