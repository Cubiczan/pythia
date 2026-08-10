"""pythia-delphi-adapter — Python client for the @gensyn-ai/gensyn-delphi-sdk.

This package wraps the official Gensyn Delphi TypeScript SDK
(``@gensyn-ai/gensyn-delphi-sdk`` on npm) behind a typed async Python client.
The SDK runs in a Node.js subprocess (``bridge.mjs``) and communicates via
JSON-RPC 2.0 over stdin/stdout.

SDK docs: https://docs.gensyn.ai/tech/agentic-trading
SDK on npm: https://www.npmjs.com/package/@gensyn-ai/gensyn-delphi-sdk
"""

from pythia_delphi_adapter.client import DelphiClient
from pythia_delphi_adapter.errors import (
    BridgeError,
    BridgeNotReadyError,
    DelphiAdapterError,
    DelphiAPIError,
)
from pythia_delphi_adapter.models import (
    BalanceResponse,
    BuySharesParams,
    EnsureTokenApprovalParams,
    EnsureTokenApprovalResponse,
    HealthResponse,
    LiquidateParams,
    LiquidateResponse,
    ListMarketsParams,
    ListPositionsParams,
    Market,
    MarketMetadata,
    MarketStatus,
    Network,
    Position,
    QuoteBuyParams,
    QuoteBuyResponse,
    QuoteSellParams,
    QuoteSellResponse,
    RedeemMarketParams,
    RedeemMarketResponse,
    SDK_VERSION,
    SellSharesParams,
    SignerType,
    TradeReceipt,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Client
    "DelphiClient",
    # Errors
    "DelphiAdapterError",
    "BridgeError",
    "BridgeNotReadyError",
    "DelphiAPIError",
    # Models
    "Market",
    "MarketMetadata",
    "MarketStatus",
    "Network",
    "SignerType",
    "Position",
    "ListMarketsParams",
    "ListPositionsParams",
    "QuoteBuyParams",
    "QuoteBuyResponse",
    "QuoteSellParams",
    "QuoteSellResponse",
    "BuySharesParams",
    "SellSharesParams",
    "TradeReceipt",
    "RedeemMarketParams",
    "RedeemMarketResponse",
    "LiquidateParams",
    "LiquidateResponse",
    "EnsureTokenApprovalParams",
    "EnsureTokenApprovalResponse",
    "BalanceResponse",
    "HealthResponse",
    "SDK_VERSION",
]
