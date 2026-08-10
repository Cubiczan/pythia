"""pythia-strata — stratified data ingestion + enrichment for the Pythia mesh.

Wraps ``icohangar-ops/strata`` and adds:

- a thin Delphi market fetch orchestration layer (delegated to
  ``pythia_delphi_adapter``),
- three enrichment providers (news / on-chain / social) with soft-fail
  stubs,
- an ``EnrichedMarket`` schema that bundles the market + the three
  enrichment strata,
- a ``MarketEnricher`` that fetches all three strata in parallel via
  ``asyncio.gather`` and converts the result to the ``MarketContext``
  contract consumed by ``pythia_analyst_mesh``.

Public API
----------
- ``MarketEnricher``    — orchestrates the three providers + builds the EnrichedMarket.
- ``EnrichedMarket``    — pydantic v2 model: a Market + news + on_chain + social.
- ``NewsProvider``      — recent news headlines for a market question.
- ``OnChainProvider``   — on-chain metrics for crypto-category markets.
- ``SocialProvider``    — social-platform signal aggregates.

See ``pythia_strata.enricher`` for the orchestration logic and
``pythia_strata.providers`` for the per-stratum fetchers.
"""

from __future__ import annotations

from .enricher import MarketEnricher
from .providers import NewsProvider, OnChainProvider, SocialProvider
from .types import (
    EnrichedMarket,
    Market,
    MarketContext,
    NewsArticle,
    OnChainMetric,
    SocialPlatform,
    SocialSignal,
)

__version__ = "0.1.0"

__all__ = [
    "EnrichedMarket",
    "Market",
    "MarketContext",
    "MarketEnricher",
    "NewsArticle",
    "NewsProvider",
    "OnChainMetric",
    "OnChainProvider",
    "SocialPlatform",
    "SocialProvider",
    "SocialSignal",
    "__version__",
]
