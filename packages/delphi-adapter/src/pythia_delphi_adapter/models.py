"""Pydantic v2 models for the Gensyn Delphi ATT API.

These models describe the typed shapes that flow in and out of
``DelphiClient``. The rest of the Pythia mesh depends on these types
and never touches raw ATT JSON.

Field names follow the assumed ATT response schema documented at
https://docs.gensyn.ai/tech/agentic-trading and inferred from the
``gensyn-ai/gensyn-delphi-skills`` repo. Anywhere the live ATT shape is
not yet confirmed we mark the field with a ``# VERIFY:`` comment.

All datetimes are parsed as timezone-aware UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ATT docs: https://docs.gensyn.ai/tech/agentic-trading
ATT_DOCS_URL = "https://docs.gensyn.ai/tech/agentic-trading"

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketStatus(str, Enum):
    """Lifecycle state of a Delphi market.

    Maps to the ``status`` field on ``GET /markets``. The four values below
    are the standard lifecycle: a market opens, optionally closes for new
    orders, then settles (via the AI arbiter) or is cancelled.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    SETTLED = "SETTLED"
    CANCELLED = "CANCELLED"

class OrderSide(str, Enum):
    """Side of a binary Delphi market the agent is taking.

    Delphi markets resolve to YES or NO; an order is a bet on one of those
    outcomes. ``side`` is sent in the ``POST /orders`` payload.
    """

    YES = "YES"
    NO = "NO"

class MarketCategory(str, Enum):
    """Delphi market categories.

    The first five (politics, economics, sports, crypto, subjective) match
    the categories surfaced in the ATT market list UI. ``OTHER`` is a
    fallback for markets that introduce new categories the adapter has not
    been updated for.
    """

    POLITICS = "POLITICS"
    ECONOMICS = "ECONOMICS"
    SPORTS = "SPORTS"
    CRYPTO = "CRYPTO"
    SUBJECTIVE = "SUBJECTIVE"
    OTHER = "OTHER"

class OrderStatus(str, Enum):
    """Status of a submitted order, as reported by ATT on the receipt."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class SettlementOutcome(str, Enum):
    """Outcome a market was resolved to by the AI arbiter."""

    YES = "YES"
    NO = "NO"

# ---------------------------------------------------------------------------
# Core market / orderbook models
# ---------------------------------------------------------------------------

def _ensure_utc(value: datetime) -> datetime:
    """Coerce a naive datetime to UTC.

    ATT timestamps are expected to be ISO-8601 with timezone. If a naive
    timestamp leaks through (some legacy fields), treat it as UTC rather
    than failing the parse.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value

def _coerce_utc_validator(v: object) -> object:
    """Pre-validator that accepts both datetime instances and ISO strings.

    Pydantic v2 parses ISO-8601 strings into naive ``datetime`` objects if
    the string has no timezone designator — we want to coerce those to UTC
    before the final datetime field is set, so downstream code can always
    assume timezone-aware datetimes.
    """
    if isinstance(v, datetime):
        return _ensure_utc(v)
    if isinstance(v, str):
        try:
            parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return v  # let pydantic raise its own validation error
        return _ensure_utc(parsed)
    return v

class Market(BaseModel):
    """A single Delphi information market.

    Maps to the ``GET /markets`` and ``GET /markets/{id}`` ATT responses.

    The ``arbiter_model`` field identifies the AI model that will settle the
    market — Delphi's differentiator is that settlements are AI-arbitrated
    rather than trust-based or UMA-style. Knowing the arbiter ahead of time
    lets the analyst mesh weight its confidence (e.g. markets arbitrated by
    a model the mesh has backtested should be sized more aggressively).
    """

    model_config = ConfigDict(extra="allow")

    market_id: str = Field(..., description="ATT-issued unique market identifier")
    question: str = Field(..., description="Human-readable market question")
    category: MarketCategory = Field(..., description="Market category")
    status: MarketStatus = Field(..., description="Current lifecycle state")
    yes_price: float = Field(..., ge=0.0, le=1.0, description="Current YES price (0..1)")
    no_price: float = Field(..., ge=0.0, le=1.0, description="Current NO price (0..1)")
    volume_usd: float = Field(0.0, ge=0.0, description="Lifetime volume in USD")
    liquidity_usd: float = Field(0.0, ge=0.0, description="Current on-book liquidity in USD")
    created_at: datetime
    closes_at: datetime | None = Field(None, description="When the market stops accepting orders")
    settlement_at: datetime | None = Field(None, description="When the market was / will be settled")
    arbiter_model: str | None = Field(
        None,
        description="Identifier of the AI model that will settle this market. "
        "# VERIFY: exact field name in ATT response.",
    )

    @field_validator("created_at", "closes_at", "settlement_at", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)

class OrderBookLevel(BaseModel):
    """A single price level on the ATT order book.

    ``size_usd`` is the total size available at this price; ``price`` is in
    the same 0..1 probability space as ``Market.yes_price``.
    """

    price: float = Field(..., ge=0.0, le=1.0)
    size_usd: float = Field(..., ge=0.0)

class OrderBook(BaseModel):
    """Order book for a single market.

    Maps to ``GET /markets/{id}/orderbook``. ``bids`` are buy-side levels
    (YES buyers + NO buyers aggregated); ``asks`` are sell-side. The exact
    aggregation is ATT-internal — we surface it as-is.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    # VERIFY: ATT may also surface a ``spread`` or ``mid`` field.

class TradeReceipt(BaseModel):
    """Receipt returned by ``POST /orders``.

    Includes the ATT order id (used for cancellation / status checks), the
    fill price (which may differ from the limit if partial fill), and the
    ``signed_by`` field indicating which signing key attested to the order 
    useful for the audit trail in ``pythia-observability``.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    side: OrderSide
    size_usd: float = Field(..., ge=0.0)
    fill_price: float | None = Field(None, ge=0.0, le=1.0)
    att_order_id: str = Field(..., description="ATT-issued order id")
    status: OrderStatus = OrderStatus.PENDING
    signed_by: str | None = Field(
        None,
        description="Identifier of the signing key that attested this order. "
        "# VERIFY: exact field name in ATT response.",
    )
    timestamp: datetime

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)

class Position(BaseModel):
    """The agent's current position in a single market.

    Maps to ``GET /positions``. ``unrealized_pnl_usd`` is computed by ATT
    against the current mark price; the adapter surfaces it without
    re-computing so the audit trail stays consistent with what ATT reports.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    side: OrderSide
    size_usd: float = Field(..., ge=0.0)
    avg_fill_price: float = Field(..., ge=0.0, le=1.0)
    current_value_usd: float
    unrealized_pnl_usd: float

class Settlement(BaseModel):
    """A resolved Delphi market.

    Maps to ``GET /settlements``. This is the AI-as-arbiter output: the
    market is settled to YES or NO, and ``evidence_hashes`` points to the
    public, training-reusable artifacts the arbiter produced to justify the
    outcome. Pythia uses these to (a) update P&L, (b) score analyst-mesh
    calibration against actual outcomes, and (c) feed ``pythia-forge``
    backtests.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    outcome: SettlementOutcome
    arbiter_model: str = Field(
        ...,
        description="Identifier of the AI model that arbitrated the settlement.",
    )
    settlement_price: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final settlement price (1.0 for YES, 0.0 for NO, or a "
        "fractional value if the market settled to a percentage outcome).",
    )
    resolved_at: datetime
    evidence_hashes: list[str] = Field(
        default_factory=list,
        description="Content hashes of the public arbiter artifacts. "
        "# VERIFY: exact field name + hash format in ATT response.",
    )

    @field_validator("resolved_at", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)

# ---------------------------------------------------------------------------
# Market event stream (discriminated union)
# ---------------------------------------------------------------------------

class _MarketEventBase(BaseModel):
    """Common fields across all market events.

    ATT pushes events over WebSocket with a ``type`` discriminator. We
    model each variant as a separate pydantic class and union them via
    ``MarketEvent``.
    """

    model_config = ConfigDict(extra="allow")

    market_id: str
    timestamp: datetime

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_utc(cls, v: object) -> object:
        return _coerce_utc_validator(v)

class MarketOpened(_MarketEventBase):
    """A new market was just listed on Delphi."""

    type: Literal["market_opened"] = "market_opened"
    question: str
    category: MarketCategory
    # VERIFY: ATT may use "MARKET_OPENED" / "market.opened" — confirm casing.

class PriceUpdated(_MarketEventBase):
    """A market's YES/NO prices moved."""

    type: Literal["price_updated"] = "price_updated"
    yes_price: float = Field(..., ge=0.0, le=1.0)
    no_price: float = Field(..., ge=0.0, le=1.0)

class OrderMatched(_MarketEventBase):
    """An order (ours or someone else's) was matched on this market."""

    type: Literal["order_matched"] = "order_matched"
    side: OrderSide
    size_usd: float = Field(..., ge=0.0)
    fill_price: float = Field(..., ge=0.0, le=1.0)
    # VERIFY: ATT may include ``participant`` / ``agent_id`` field.

class MarketSettled(_MarketEventBase):
    """A market was just settled by the AI arbiter."""

    type: Literal["market_settled"] = "market_settled"
    outcome: SettlementOutcome
    arbiter_model: str
    settlement_price: float = Field(..., ge=0.0, le=1.0)

# Discriminated union — pydantic dispatches on the ``type`` field.
MarketEvent = Annotated[
    Union[MarketOpened, PriceUpdated, OrderMatched, MarketSettled],
    Field(discriminator="type"),
]

__all__ = [
    "ATT_DOCS_URL",
    "MarketStatus",
    "OrderSide",
    "MarketCategory",
    "OrderStatus",
    "SettlementOutcome",
    "Market",
    "OrderBookLevel",
    "OrderBook",
    "TradeReceipt",
    "Position",
    "Settlement",
    "MarketEvent",
    "MarketOpened",
    "PriceUpdated",
    "OrderMatched",
    "MarketSettled",
]
