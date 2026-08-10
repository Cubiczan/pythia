"""Social-platform enrichment provider.

Pulls social-platform signal aggregates (post counts, average sentiment,
top keywords — per platform) for the ``social`` stratum of
``EnrichedMarket``.

Status: stub. ``fetch`` always returns ``[]`` (regardless of inputs)
until wired up.

# VERIFY: which social APIs to wire up. Candidates:
  - Twitter/X API v2 (paid, full search; requires Bearer token +
    optional academic tier). Endpoint: /2/tweets/search/recent.
  - Reddit API (free, OAuth, rate-limited). Endpoints:
    /search.json, /r/{subreddit}/search.json.
  - Farcaster Hub HTTP API (free, decentralised). Endpoint:
    /v1/castsByMention or /v1/search.
  - LunarCrush (paid social-sentiment aggregator — single API that
    covers Twitter + Reddit + others; preferred if budget allows).

The stub below is structured so that wiring up any of the candidates
amounts to:
  1. Implementing the body of ``fetch`` with one ``httpx.AsyncClient``
     call per platform (or a single call to an aggregator like
     LunarCrush).
  2. Mapping the response to one ``SocialSignal`` per platform.
  3. Wrapping the whole call in a try/except that logs + returns ``[]``.
"""

from __future__ import annotations

import logging

from pythia_strata.types import SocialSignal

logger = logging.getLogger(__name__)


class SocialProvider:
    """Fetches social-platform signal aggregates for a query.

    Soft-fail contract: ``fetch`` ALWAYS returns a ``list[SocialSignal]``
    and never raises. If the upstream call fails for any reason, or if
    no API credentials are configured, it logs a warning and returns
    ``[]``.

    Parameters
    ----------
    api_key:
        Optional API key for the upstream social provider. Twitter API
        v2 requires a Bearer token; Reddit requires OAuth (client_id +
        client_secret); LunarCrush requires an API key. ``None`` by
        default.

        # VERIFY: per-provider auth shape once a source is chosen. If
        # multiple platforms are wired up, this constructor likely needs
        # to accept a dict of per-platform credentials instead.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    async def fetch(self, query: str, limit: int = 5) -> list[SocialSignal]:
        """Return up to ``limit`` social-platform signal aggregates.

        Each ``SocialSignal`` represents one platform's view of the
        query (e.g. one record for Twitter, one for Reddit, one for
        Farcaster) — *not* one post. So ``limit=5`` means "up to 5
        platforms", not "up to 5 posts".

        Parameters
        ----------
        query:
            Search query — typically the market question or a keyword
            extract of it (``MarketEnricher._extract_keywords`` produces
            the latter).
        limit:
            Maximum number of platform records to return. Default 5.

        Returns
        -------
        list[SocialSignal]
            Possibly empty. Never raises.
        """
        if not query or not query.strip():
            return []

        if not self.api_key:
            # Stub path — no API key configured, soft-fail to empty list.
            logger.debug(
                "SocialProvider stub path: no api_key configured, returning [] "
                "(query=%r, limit=%d)",
                query[:80],
                limit,
            )
            return []

        # --- Wired-up path (not yet implemented) ------------------------
        # Once a social API is chosen, implement the fetch here.
        # LunarCrush skeleton (single aggregator, simplest):
        #
        #   try:
        #       async with httpx.AsyncClient(timeout=15.0) as client:
        #           resp = await client.get(
        #               "https://lunarcrush.com/api4/public/aggregate/v1",
        #               params={
        #                   "topic": query,
        #                   "limit": limit,
        #                   "api_key": self.api_key,  # VERIFY: auth shape
        #               },
        #           )
        #           resp.raise_for_status()
        #           data = resp.json()
        #       now = datetime.now(UTC).isoformat()
        #       return [
        #           SocialSignal(
        #               platform=item.get("network", "twitter"),
        #               post_count_24h=int(item.get("posts", 0)),
        #               avg_sentiment=float(item.get("sentiment", 0.0)),
        #               top_keywords=list(item.get("top_words", []))[:5],
        #               timestamp=now,
        #           )
        #           for item in data.get("data", [])[:limit]
        #       ]
        #   except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        #       logger.warning(
        #           "SocialProvider upstream error for query=%r: %r",
        #           query[:80], exc,
        #       )
        #       return []
        #
        # VERIFY: which provider(s), endpoint shape, auth header name,
        # sentiment computation (provider-computed vs. local), keyword
        # extraction source.

        logger.warning(
            "SocialProvider has api_key set but no upstream is wired up yet; "
            "returning [] (query=%r). See providers/social.py # VERIFY.",
            query[:80],
        )
        return []


__all__ = ["SocialProvider"]
