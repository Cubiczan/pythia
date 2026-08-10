"""Stratified enrichment providers for pythia-strata.

Each provider is an async, soft-failing fetcher for one slice of the
enrichment context that gets layered onto a Delphi market before it is
handed to the analyst mesh:

- ``NewsProvider``     — recent news articles relevant to the market question.
- ``OnChainProvider``  — on-chain metrics for crypto-category markets.
- ``SocialProvider``   — social-platform signal aggregates (Twitter / Reddit / Farcaster).

All providers must obey the soft-fail contract: if their upstream API is
unavailable, unconfigured, or returns junk, they return an empty list rather
than raising. ``MarketEnricher.enrich`` parallelises them with
``asyncio.gather(return_exceptions=True)`` so a single flaky upstream can
never block the enrichment pipeline.
"""

from __future__ import annotations

from .news import NewsProvider
from .onchain import OnChainProvider
from .social import SocialProvider

__all__ = [
    "NewsProvider",
    "OnChainProvider",
    "SocialProvider",
]
