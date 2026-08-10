"""News enrichment provider.

Pulls recent news headlines relevant to a market question, for the
``news`` stratum of ``EnrichedMarket``.

Status: stub. Without an API key (the default), ``fetch`` returns an empty
list — the analyst mesh then falls back to its own training-data prior
plus the bare market question. With an API key configured, the provider
should call the chosen news API and parse the response into
``NewsArticle`` records.

# VERIFY: which news API to wire up. Candidates, in rough order of
preference:
  - GDELT (free, no key, broad coverage, rate-limited) — preferred default
    for development use. Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  - NewsAPI.org (free tier 100 req/day, requires key, good headline coverage)
  - Bing News Search v7 (Azure Cognitive Services, requires key)
  - AlphaVantage News & Sentiment (free, financial focus, requires key)

The stub below is structured so that wiring up GDELT (or any of the
candidates) amounts to:
  1. Implementing the ``api_key``-gated branch in ``fetch``.
  2. Calling the chosen endpoint with ``httpx.AsyncClient``.
  3. Mapping the response items to ``NewsArticle`` records.
  4. Wrapping the whole call in a try/except that logs + returns ``[]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pythia_strata.types import NewsArticle

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

class NewsProvider:
    """Fetches recent news articles relevant to a market question.

    Soft-fail contract: ``fetch`` ALWAYS returns a ``list[NewsArticle]``
    and never raises. If no API key is configured, it returns ``[]``
    immediately (the stub path). If an upstream call fails for any reason
    (timeout, 5xx, malformed JSON, parse error), it logs a warning and
    returns ``[]``.

    Parameters
    ----------
    api_key:
        Optional API key for the upstream news provider. When ``None``
        (the default), ``fetch`` returns ``[]`` without making any
        network call — this is the stub path that lets the rest of the
        mesh run end-to-end in CI without external dependencies.

        # VERIFY: per-provider auth shape. GDELT needs no key at all;
        # NewsAPI.org uses ``apiKey`` query param; Bing uses
        # ``Ocp-Apim-Subscription-Key`` header; AlphaVantage uses
        # ``apikey`` query param. Once a provider is chosen, refine this
        # constructor accordingly (e.g. accept a ``provider`` enum too).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    async def fetch(self, query: str, limit: int = 5) -> list[NewsArticle]:
        """Return up to ``limit`` recent news articles matching ``query``.

        Parameters
        ----------
        query:
            Search query — typically the market question or a keyword
            extract of it (``MarketEnricher._extract_keywords`` produces
            the latter).
        limit:
            Maximum number of articles to return. Default 5.

        Returns
        -------
        list[NewsArticle]
            Possibly empty. Never raises.
        """
        if not query or not query.strip():
            return []

        if not self.api_key:
            # Stub path — no API key configured, soft-fail to empty list.
            # The analyst mesh will fall back to the bare market question.
            logger.debug(
                "NewsProvider stub path: no api_key configured, returning [] (query=%r, limit=%d)",
                query[:80],
                limit,
            )
            return []

        # --- Wired-up path (not yet implemented) ------------------------
        # Once a news API is chosen, implement the fetch here. Skeleton:
        #
        #   try:
        #       async with httpx.AsyncClient(timeout=15.0) as client:
        #           resp = await client.get(
        #               "https://newsapi.org/v2/everything",
        #               params={
        #                   "q": query,
        #                   "pageSize": limit,
        #                   "apiKey": self.api_key,
        #               },
        #           )
        #           resp.raise_for_status()
        #           data = resp.json()
        #       return [
        #           NewsArticle(
        #               title=item["title"],
        #               url=item["url"],
        #               source=(
        #                   item["source"]["name"]
        #                   if isinstance(item.get("source"), dict)
        #                   else "unknown"
        #               ),
        #               published_at=item.get("publishedAt"),
        #               summary=item.get("description"),
        #               sentiment_score=None,  # NewsAPI doesn't compute sentiment
        #           )
        #           for item in data.get("articles", [])[:limit]
        #       ]
        #   except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        #       logger.warning(
        #           "NewsProvider upstream error for query=%r: %r",
        #           query[:80], exc,
        #       )
        #       return []
        #
        # VERIFY: which provider, endpoint shape, auth header name,
        # response envelope, sentiment source (provider-computed vs.
        # local VADER/transformer).

        logger.warning(
            "NewsProvider has api_key set but no upstream is wired up yet; "
            "returning [] (query=%r). See providers/news.py # VERIFY.",
            query[:80],
        )
        return []

__all__ = ["NewsProvider"]
