"""Pydantic v2 models matching the @gensyn-ai/gensyn-delphi-sdk TypeScript types.

These models mirror the SDK's `Market`, `Position`, `BuySharesResponse`, etc.
The Python adapter parses every JSON-RPC response from the Node bridge into
these typed models so the rest of the Pythia mesh never touches raw JSON.

Key differences from the original ATT-HTTP assumption:

1. **Multi-outcome LMSR markets.** A Delphi market has `outcomes: list[str]`
   (e.g. ``["YES", "NO"]`` or ``["Bitcoin", "Ethereum", "Solana"]``) and
   `spot_prices` / `spot_implied_probabilities` arrays of the same length.
   There is no YES/NO binary assumption.

2. **No order book.** LMSR markets use a bonding curve — you quote a buy or
   sell directly via `quoteBuy` / `quoteSell` and the SDK returns the
   `tokens_in` / `tokens_out` (collateral) required for a given number of
   shares. The "price" is implicit in the curve.

3. **On-chain settlement, not HTTP.** Trades are submitted as Ethereum
   transactions; `buyShares` returns `{transaction_hash}`, not an order id.
   Settlement is read from the on-chain gateway via `getMarketStatus`.

4. **Two market identifiers.** `market_address` (the on-chain proxy contract)
   and `app_market_id` (the UUID used in the Delphi app URL). Trades use
   the address; deep-links use the UUID.

All datetimes are parsed as timezone-aware UTC. BigInt values from the SDK
(balances, allowances, share counts) arrive as decimal strings and are
preserved as strings — Python's `int` is unbounded so we could cast, but
keeping the string form avoids any precision surprise downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SDK_VERSION = "2.1.0"
SDK_DOCS_URL = "https://docs.gensyn.ai/tech/agentic-trading"
SDK_NPM_URL = "https://www.npmjs.com/package/@gensyn-ai/gensyn-delphi-sdk"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketStatus(str, Enum):
    """Lifecycle state of a Delphi market.

    Matches the SDK's ``MarketStatus`` union. ``failed`` only occurs on
    automated-settlement markets where the Truebit oracle ran but could not
    resolve the question — like ``expired``, it has no winning outcome and
    must be exited via ``liquidate()`` rather than ``redeem()``.
    """

    OPEN = "open"
    AWAITING_SETTLEMENT = "awaiting_settlement"
    SETTLED = "settled"
    EXPIRED = "expired"
    FAILED = "failed"


class Network(str, Enum):
    """Supported Delphi networks.

    ``competition-testnet`` targets the agent trading competition: same chain
    as testnet, the competition's own LMSR contracts, and competition-scoped
    market reads (the SDK sends ``X-Delphi-Mode: competition`` automatically).
    """

    TESTNET = "testnet"
    MAINNET = "mainnet"
    COMPETITION_TESTNET = "competition-testnet"


class SignerType(str, Enum):
    """How the SDK signs on-chain transactions."""

    CDP_SERVER_WALLET = "cdp_server_wallet"
    PRIVATE_KEY = "private_key"


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _coerce_utc_validator(v: object) -> object:
    if isinstance(v, datetime):
        return _ensure_utc(v)
    if isinstance(v, str):
        try:
            parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return v
        return _ensure_utc(parsed)
    return v


class MarketMetadata(BaseModel):
    """On-chain metadata attached to a market at creation.

    Matches the SDK's ``MarketMetadata`` interface. The ``outcomes`` list is
    the source of truth for how many outcomes the market has and what each
    one is labelled — the SDK uses ``outcomeIdx`` (0-based) to address them.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    question: str = Field(..., description="Human-readable market question")
    outcomes: list[str] = Field(
        default_factory=list,
        description="Outcome labels, e.g. ['YES', 'NO'] or ['Bitcoin', 'Ethereum', 'Solana']",
    )
    model_identifier: str | None = Field(
        None, alias="model_identifier", description="Identifier of the AI arbiter model that will settle this market"
    )
    prompt_context: str | None = Field(None, description="Arbiter prompt context")
    initial_liquidity: str | None = None
    initial_pool: str | None = None


class Market(BaseModel):
    """A single Delphi information market.

    Mirrors the SDK's ``Market`` interface. Key fields:

    - ``market_address`` (SDK: ``id``) — the on-chain proxy contract address.
      Used for all trading calls (buy/sell/quote/redeem/liquidate).
    - ``app_market_id`` — UUID for the Delphi app UI.
    - ``outcomes`` — list of outcome labels (from ``metadata.outcomes``).
    - ``spot_prices`` / ``spot_implied_probabilities`` — per-outcome price
      and probability arrays, only populated when the SDK call passes
      ``pricesAndImpliedProbabilities: true``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="id", description="On-chain proxy contract address (0x...)")
    app_market_id: str = Field(..., alias="appMarketId", description="UUID for the Delphi app UI")
    market_url: str = Field(..., alias="marketUrl", description="Direct link to the market on the Delphi app")
    status: MarketStatus
    category: str = Field(..., description="Market category (crypto, culture, economics, miscellaneous, politics, sports)")
    deployer: str = Field(..., description="Wallet address of the market creator")
    created_at: datetime = Field(..., alias="createdAt")
    settles_at: datetime | None = Field(None, alias="settlesAt", description="When the market is scheduled to settle")
    settled_at: datetime | None = Field(None, alias="settledAt", description="When the market was actually settled")
    winning_outcome_idx: int | None = Field(
        None,
        alias="winningOutcomeIdx",
        description="Index of the winning outcome (0-based), set after settlement",
    )
    metadata: MarketMetadata | None = None
    spot_prices: list[float] | None = Field(
        None,
        alias="spotPrices",
        description="Per-outcome spot prices (human-readable floats). Populated only when pricesAndImpliedProbabilities=true.",
    )
    spot_implied_probabilities: list[float] | None = Field(
        None,
        alias="spotImpliedProbabilities",
        description="Per-outcome implied probabilities (0..1). Populated only when pricesAndImpliedProbabilities=true.",
    )

    @field_validator("created_at", "settles_at", "settled_at", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)

    @field_validator("winning_outcome_idx", mode="before")
    @classmethod
    def _coerce_outcome_idx(cls, v: object) -> object:
        """SDK returns winningOutcomeIdx as a string; coerce to int."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return v

    @property
    def outcomes(self) -> list[str]:
        """Convenience accessor for the outcome labels."""
        if self.metadata and self.metadata.outcomes:
            return self.metadata.outcomes
        return []

    @property
    def question(self) -> str:
        """Convenience accessor for the market question."""
        if self.metadata:
            return self.metadata.question
        return ""


class ListMarketsParams(BaseModel):
    """Parameters for ``listMarkets``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)
    order_by: str = Field("liquidity", alias="orderBy", description='"liquidity" | "created" | "settles_at"')
    status: MarketStatus | None = None
    category: str | None = None
    verifiable: bool | None = None
    competition_id: str | None = Field(None, alias="competitionId", description="Competition UUID (competition networks only)")
    prices_and_implied_probabilities: bool = Field(
        False, alias="pricesAndImpliedProbabilities",
        description="Fetch on-chain spot prices + implied probabilities via multicall"
    )


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """The agent's current position in a single market outcome.

    Mirrors the SDK's ``Position`` interface. ``shares`` is a decimal string
    (18-decimal fixed-point) because the on-chain value can exceed 2^53.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    position_id: str = Field(..., alias="id")
    market_address: str = Field(..., alias="marketProxy")
    wallet: str
    outcome_idx: int = Field(..., alias="outcomeIdx", description="0-based outcome index")
    shares: str = Field(..., description="Number of shares held (18-decimal string)")
    redeemed_or_liquidated: bool = Field(False, alias="redeemedOrLiquidated")
    tokens_redeemed: str = Field("0", alias="tokensRedeemed")
    market_status: MarketStatus = Field(..., alias="marketStatus")

    @field_validator("outcome_idx", mode="before")
    @classmethod
    def _coerce_outcome_idx(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class ListPositionsParams(BaseModel):
    """Parameters for ``listPositions``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    wallet: str = Field(..., description="Wallet address to query")
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)
    redeemed_or_liquidated: bool | None = Field(None, alias="redeemedOrLiquidated")


# ---------------------------------------------------------------------------
# Trade quotes and receipts
# ---------------------------------------------------------------------------

class QuoteBuyParams(BaseModel):
    """Parameters for ``quoteBuy`` — simulates a buy without sending a tx."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress", description="Market proxy address (0x...)")
    outcome_idx: int = Field(..., alias="outcomeIdx", ge=0, description="0-based outcome index")
    shares_out: str = Field(
        ...,
        alias="sharesOut",
        description="Number of shares to buy, as a 18-decimal string (e.g. '1000000000000000000' for 1 share)",
    )


class QuoteBuyResponse(BaseModel):
    """Result of ``quoteBuy`` — the collateral required."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tokens_in: str = Field(..., alias="tokensIn", description="Collateral tokens required (18-decimal string)")


class QuoteSellParams(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    outcome_idx: int = Field(..., alias="outcomeIdx", ge=0)
    shares_in: str = Field(..., alias="sharesIn")


class QuoteSellResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tokens_out: str = Field(..., alias="tokensOut")


class BuySharesParams(BaseModel):
    """Parameters for ``buyShares`` — submits an on-chain buy transaction."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    outcome_idx: int = Field(..., alias="outcomeIdx", ge=0)
    shares_out: str = Field(..., alias="sharesOut", description="18-decimal string")
    max_tokens_in: str = Field(
        ...,
        alias="maxTokensIn",
        description="Slippage protection: max collateral willing to spend (18-decimal string)",
    )


class SellSharesParams(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    outcome_idx: int = Field(..., alias="outcomeIdx", ge=0)
    shares_in: str = Field(..., alias="sharesIn")
    min_tokens_out: str = Field(..., alias="minTokensOut")


class TradeReceipt(BaseModel):
    """Receipt returned by ``buyShares`` / ``sellShares``.

    The SDK returns just ``{transaction_hash}`` — we expand it with the input
    parameters and a timestamp so the audit log has a complete record.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str
    outcome_idx: int = Field(..., ge=0)
    side: str = Field(..., description='"buy" or "sell"')
    shares: str = Field(..., description="Number of shares traded (18-decimal string)")
    transaction_hash: str = Field(..., description="On-chain transaction hash (0x...)")
    max_tokens_in: str | None = Field(None, description="Slippage cap on buy")
    min_tokens_out: str | None = Field(None, description="Slippage floor on sell")
    timestamp: datetime

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)


# ---------------------------------------------------------------------------
# Settlement / redemption / liquidation
# ---------------------------------------------------------------------------

class RedeemMarketParams(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")


class RedeemMarketResponse(BaseModel):
    """Result of ``redeemMarket`` — burns winning shares for collateral."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    transaction_hash: str = Field(..., alias="transactionHash")
    shares_in: str = Field(..., alias="sharesIn")
    tokens_out: str = Field(..., alias="tokensOut")


class LiquidateParams(BaseModel):
    """Parameters for ``liquidate`` — exits an expired/failed market."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    outcome_indices: list[int] = Field(..., alias="outcomeIndices", description="0-based outcome indices")


class LiquidateResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    transaction_hash: str = Field(..., alias="transactionHash")
    shares_in: list[str] = Field(..., alias="sharesIn")
    total_tokens_out: str = Field(..., alias="totalTokensOut")


# ---------------------------------------------------------------------------
# Token / approval / balance
# ---------------------------------------------------------------------------

class EnsureTokenApprovalParams(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    market_address: str = Field(..., alias="marketAddress")
    minimum_amount: str = Field(..., alias="minimumAmount")
    approve_amount: str | None = Field(None, alias="approveAmount")


class EnsureTokenApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    approval_needed: bool = Field(..., alias="approvalNeeded")
    allowance: str
    transaction_hash: str | None = Field(None, alias="transactionHash")


class BalanceResponse(BaseModel):
    """ERC-20 balance + decimals for the signer wallet."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    balance: str = Field(..., description="Token balance (18-decimal string)")
    decimals: int = Field(18, ge=0)


class HealthResponse(BaseModel):
    """REST API health check response."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    status: str = Field(..., description='"ok" or error message')


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "SDK_VERSION",
    "SDK_DOCS_URL",
    "SDK_NPM_URL",
    "MarketStatus",
    "Network",
    "SignerType",
    "MarketMetadata",
    "Market",
    "ListMarketsParams",
    "Position",
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
]
