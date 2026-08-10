"""Pydantic v2 models for pythia-strata.

This module owns:
    - The three enrichment payload types: ``NewsArticle``, ``OnChainMetric``,
      ``SocialSignal``.
    - The ``EnrichedMarket`` aggregate that bundles them with the raw
      Delphi market data.
    - A re-export of ``MarketContext`` from ``pythia_analyst_mesh`` (with
      a local fallback if the mesh isn't installed yet, so this wrapper
      can be developed / tested in isolation).
    - A re-export of ``Market`` from ``pythia_delphi_adapter`` (same
      try/except pattern).

The fallback ``MarketContext`` is intentionally minimal — it mirrors the
contract documented in ``pythia-analyst-mesh/src/pythia_analyst_mesh/types.py``
and in the top-level ``docs/ARCHITECTURE.md`` but is *not* the source of
truth. Once ``pythia_analyst_mesh`` is vendored (it's declared as a
sibling sub-repo, not a hard dep of this one), the try/except import below
resolves to the real type.

All datetimes carried by these models are ISO 8601 strings (UTC) rather
than ``datetime`` objects — this keeps the models trivially JSON-
serialisable for the audit log and for the CLI output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Market re-export (from pythia_delphi_adapter, with local fallback).
# ---------------------------------------------------------------------------
# Preferred path: the adapter owns the canonical Market type.
try:  # pragma: no cover - exercised only when adapter is present
    from pythia_delphi_adapter.models import Market as Market  # noqa: F401

    MARKET_SOURCE = "pythia_delphi_adapter"
except Exception:  # noqa: BLE001 - we want to swallow any import failure

    class Market(BaseModel):  # type: ignore[no-redef]
        """Fallback ``Market`` — structurally identical to the adapter's.

        Source of truth lives in ``pythia_delphi_adapter.models.Market``;
        this is a faithful but minimal mirror so pythia-strata can be
        developed in isolation before the adapter is vendored.
        """

        model_config = ConfigDict(extra="allow")

        market_id: str
        question: str
        category: str = "OTHER"
        outcomes: list[str] = Field(default_factory=lambda: ["YES", "NO"])
        spot_prices: list[float] = Field(default_factory=list)
        yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
        no_price: float | None = Field(default=None, ge=0.0, le=1.0)
        volume_usd: float = Field(default=0.0, ge=0.0)
        closes_at: str | None = None

    MARKET_SOURCE = "local-fallback"

# ---------------------------------------------------------------------------
# MarketContext re-export (from pythia_analyst_mesh, with local fallback).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the mesh is present
    from pythia_analyst_mesh.types import MarketContext as MarketContext  # noqa: F401

    MARKET_CONTEXT_SOURCE = "pythia_analyst_mesh"
except Exception:  # noqa: BLE001 - we want to swallow any import failure

    class MarketContext(BaseModel):  # type: ignore[no-redef]
        """Fallback ``MarketContext`` — mirrors the analyst-mesh contract.

        Source of truth lives in
        ``pythia_analyst_mesh.types.MarketContext``; this is a faithful
        but minimal mirror so pythia-strata can be developed and tested
        in isolation before the mesh is vendored.
        """

        model_config = ConfigDict(extra="allow")

        market_id: str = Field(..., min_length=1)
        question: str = Field(..., min_length=1)
        category: str = Field(..., min_length=1)
        metadata: dict[str, Any] = Field(default_factory=dict)
        outcomes: list[str] = Field(default_factory=lambda: ["YES", "NO"])
        spot_prices: list[float] = Field(default_factory=list)
        current_yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
        current_no_price: float | None = Field(default=None, ge=0.0, le=1.0)
        volume_usd: float | None = Field(default=None, ge=0.0)
        closes_at: str | None = None

    MARKET_CONTEXT_SOURCE = "local-fallback"

# ---------------------------------------------------------------------------
# Enrichment payload types
# ---------------------------------------------------------------------------
SocialPlatform = Literal["twitter", "reddit", "farcaster"]
"""Supported social platforms for ``SocialSignal``."""

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()

class NewsArticle(BaseModel):
    """A single news article returned by ``NewsProvider.fetch``.

    Attributes
    ----------
    title:
        Article headline. Required (non-empty).
    url:
        Canonical URL of the article. Required.
    source:
        Publisher / domain (e.g. ``"reuters.com"``, ``"coindesk"``).
    published_at:
        ISO 8601 timestamp of the article's publication. May be ``None``
        if the upstream API doesn't surface it.
    summary:
        Optional 1-2 sentence summary (some APIs provide one; otherwise
        ``None`` and the analyst mesh falls back to the headline).
    sentiment_score:
        Optional sentiment in ``[-1.0, 1.0]`` (negative → negative
        coverage, positive → positive). ``None`` if the upstream doesn't
        compute sentiment.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    published_at: str | None = None
    summary: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("published_at")
    @classmethod
    def _validate_iso_or_none(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"published_at must be ISO 8601 or None, got: {v!r}") from exc
        return v

class OnChainMetric(BaseModel):
    """A single on-chain metric for a token, returned by ``OnChainProvider.fetch``.

    Attributes
    ----------
    token_symbol:
        Ticker the metric applies to (e.g. ``"ETH"``, ``"BTC"``). May be
        ``None`` for protocol-level metrics that aren't tied to a single
        token (e.g. total DEX volume).
    metric_name:
        Stable, snake_case identifier for the metric. Conventional names:
        ``"tvl_eth"``, ``"active_addresses_7d"``, ``"exchange_inflow_24h"``,
        ``"realized_cap_usd"``, ``"nvt_ratio"``. New names should be added
        to the README's on-chain section once they're produced.
    value:
        The metric's numeric value. Always a float — callers are expected
        to know the unit from ``metric_name`` (e.g. ``tvl_eth`` is in ETH).
    timestamp:
        ISO 8601 timestamp the metric was measured at (not when we
        fetched it — that's the upstream's report time).
    source:
        Upstream data provider (e.g. ``"defillama"``, ``"glassnode"``).
        Used for audit-trail provenance.
    """

    model_config = ConfigDict(extra="forbid")

    token_symbol: str | None = None
    metric_name: str = Field(..., min_length=1)
    value: float
    timestamp: str
    source: str = Field(..., min_length=1)

    @field_validator("timestamp")
    @classmethod
    def _validate_iso(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO 8601, got: {v!r}") from exc
        return v

class SocialSignal(BaseModel):
    """A single social-platform signal aggregate, returned by ``SocialProvider.fetch``.

    Each ``SocialSignal`` is one platform's view of the query (e.g. one
    record for Twitter, one for Reddit, one for Farcaster) — *not* one
    post. Aggregation happens upstream in the provider.

    Attributes
    ----------
    platform:
        Which platform this signal is from. ``Literal["twitter",
        "reddit", "farcaster"]]`` — extend this when adding new
        platforms.
    post_count_24h:
        Number of posts mentioning the query in the trailing 24h window.
        Zero is valid (e.g. a niche market with no social footprint).
    avg_sentiment:
        Mean sentiment across those posts, in ``[-1.0, 1.0]``. ``0.0``
        is neutral; the provider is expected to compute this from
        per-post sentiment upstream.
    top_keywords:
        Top N (typically 3-5) co-occurring keywords / hashtags, ordered
        by frequency. Empty list is valid.
    timestamp:
        ISO 8601 timestamp the aggregate was computed at.
    """

    model_config = ConfigDict(extra="forbid")

    platform: SocialPlatform
    post_count_24h: int = Field(..., ge=0)
    avg_sentiment: float = Field(..., ge=-1.0, le=1.0)
    top_keywords: list[str] = Field(default_factory=list)
    timestamp: str

    @field_validator("timestamp")
    @classmethod
    def _validate_iso(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO 8601, got: {v!r}") from exc
        return v

class EnrichedMarket(BaseModel):
    """A Delphi market plus its three strata of enrichment context.

    Supports multi-outcome LMSR markets: ``outcomes`` is the list of outcome
    labels and ``spot_prices`` is the per-outcome price array.
    For backward compatibility, ``current_yes_price`` / ``current_no_price"
    are kept as convenience fields for binary markets.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    outcomes: list[str] = Field(default_factory=lambda: ["YES", "NO"])
    spot_prices: list[float] = Field(default_factory=list)
    current_yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
    current_no_price: float | None = Field(default=None, ge=0.0, le=1.0)
    volume_usd: float | None = Field(default=None, ge=0.0)
    closes_at: str | None = None
    news: list[NewsArticle] = Field(default_factory=list)
    on_chain: list[OnChainMetric] = Field(default_factory=list)
    social: list[SocialSignal] = Field(default_factory=list)
    enriched_at: str = Field(default_factory=_now_iso)

    @field_validator("closes_at")
    @classmethod
    def _validate_closes_at_or_none(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"closes_at must be ISO 8601 or None, got: {v!r}") from exc
        return v

__all__ = [
    "EnrichedMarket",
    "Market",
    "MarketContext",
    "NewsArticle",
    "OnChainMetric",
    "SocialPlatform",
    "SocialSignal",
    "MARKET_SOURCE",
    "MARKET_CONTEXT_SOURCE",
]
