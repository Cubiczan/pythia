"""The MarketEnricher — orchestrates the three enrichment providers.

This is the heart of pythia-strata. Given a raw Delphi ``Market``, it:

1. Extracts a keyword query from ``market.question`` (simple tokenization
   that strips stopwords and punctuation).
2. Fetches the three enrichment strata in parallel via ``asyncio.gather``
   with ``return_exceptions=True`` — a flaky provider can never block
   the others, and a raised exception is converted to an empty list.
   - ``NewsProvider.fetch(query)`` — always called.
   - ``OnChainProvider.fetch(token_symbol)`` — only called for
     crypto-category markets (see ``_is_crypto_market``); for other
     categories this stratum is skipped entirely and ``on_chain=[]``.
   - ``SocialProvider.fetch(query)`` — always called.
3. Assembles the results into an ``EnrichedMarket`` (timestamped with
   the moment the enrichment pass completed).

The companion method ``to_market_context(enriched)`` converts an
``EnrichedMarket`` into the ``MarketContext`` contract consumed by
``pythia_analyst_mesh`` — embedding the three enrichment strata into the
``metadata`` dict as JSON-serialisable lists of dicts so the analyst
mesh's ``BaseAnalyst._format_market_block`` can surface them in the LLM
prompt.

Both methods are safe to call with stub providers (the default) — they'll
produce an ``EnrichedMarket`` with all-empty enrichment lists and a
``MarketContext`` with empty ``metadata["news"]`` / ``metadata["on_chain"]``
/ ``metadata["social"]``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pythia_strata.providers import NewsProvider, OnChainProvider, SocialProvider
from pythia_strata.types import (
    EnrichedMarket,
    Market,
    MarketContext,
    NewsArticle,
    OnChainMetric,
    SocialSignal,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimal English stopword list. We don't pull in NLTK / spaCy for this 
# the keyword extraction is best-effort, only used to seed the news/social
# search query, and a long tail of obscure stopwords wouldn't materially
# improve the query quality. If a market question is dominated by a
# stopword ("Will the US ..."), the news provider will still return
# relevant articles because the question itself is the primary signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "of",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "i",
        "me",
        "my",
        "yes",
        "no",
        "not",
        "than",
        "so",
        "such",
        "too",
        "very",
        "just",
        "more",
        "most",
        "some",
        "any",
        "each",
        "all",
        "both",
        "few",
    }
)

# Tokenize on any non-alphanumeric run (so "ETH/USD" → ["eth", "usd"],
# "best-of-seven" → ["best", "of", "seven"]). Lowercase everything.
_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")

class MarketEnricher:
    """Orchestrates the three enrichment providers and builds EnrichedMarket.

    Parameters
    ----------
    news:
        ``NewsProvider`` instance. Required (pass a stub-constructed
        ``NewsProvider()`` if no news upstream is configured).
    onchain:
        ``OnChainProvider`` instance. Required (same stub pattern).
    social:
        ``SocialProvider`` instance. Required (same stub pattern).

    The enricher does NOT take ownership of closing these providers 
    they're cheap, stateless objects (the underlying ``httpx.AsyncClient``
    is created per-call inside each provider's ``fetch``). If you wire
    up a stateful provider (e.g. one that holds a persistent connection
    pool), close it yourself.
    """

    def __init__(
        self,
        news: NewsProvider,
        onchain: OnChainProvider,
        social: SocialProvider,
    ) -> None:
        self.news = news
        self.onchain = onchain
        self.social = social

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def enrich(self, market: Market) -> EnrichedMarket:
        """Fetch all three enrichment strata for ``market`` in parallel.

        Returns an ``EnrichedMarket`` with the original market data plus
        the three enrichment lists (any of which may be empty if the
        corresponding provider soft-failed or returned no results).

        Never raises — provider exceptions are caught and converted to
        empty strata. The only way this can fail is if ``market`` itself
        is malformed (pydantic validation error), which is a programming
        bug in the caller, not a runtime enrichment failure.
        """
        query = " ".join(self._extract_keywords(market.question)) or market.question
        token_symbol = self._extract_token_symbol(market)

        # Decide which strata to fetch. News and social always run; on-chain
        # only runs for crypto-category markets. We express "skip" as a
        # coroutine that returns [] so the gather shape stays uniform.
        news_coro = self._safe_fetch(self.news.fetch(query=query, limit=5))
        social_coro = self._safe_fetch(self.social.fetch(query=query, limit=5))

        if self._is_crypto_market(market) and token_symbol:
            onchain_coro = self._safe_fetch(self.onchain.fetch(token_symbol=token_symbol))
        else:
            onchain_coro = _return_empty_list()

        # return_exceptions=True guarantees gather never raises — a flaky
        # provider becomes an Exception object in the result list, which
        # _safe_fetch already converted to [] before gather sees it. The
        # belt-and-braces pattern is intentional: enrichment is critical
        # enough to defend in depth.
        news_results, onchain_results, social_results = await asyncio.gather(
            news_coro,
            onchain_coro,
            social_coro,
            return_exceptions=True,
        )

        # If gather returned an Exception (it shouldn't, because _safe_fetch
        # catches everything — but be defensive), convert to [].
        news_articles = _coerce_to_list(
            news_results,
            NewsArticle,
            "news",
            market.market_id,
        )
        onchain_metrics = _coerce_to_list(
            onchain_results,
            OnChainMetric,
            "on_chain",
            market.market_id,
        )
        social_signals = _coerce_to_list(
            social_results,
            SocialSignal,
            "social",
            market.market_id,
        )

        return EnrichedMarket(
            market_id=market.market_id,
            question=market.question,
            category=_category_str(market),
            current_yes_price=getattr(market, "yes_price", None),
            current_no_price=getattr(market, "no_price", None),
            volume_usd=getattr(market, "volume_usd", None),
            closes_at=_closes_at_str(market),
            news=news_articles,
            on_chain=onchain_metrics,
            social=social_signals,
            enriched_at=datetime.now(UTC).isoformat(),
        )

    def to_market_context(self, enriched: EnrichedMarket) -> MarketContext:
        """Convert an ``EnrichedMarket`` to the ``MarketContext`` the mesh expects.

        The three enrichment strata are embedded into ``metadata`` as
        JSON-serialisable lists of dicts (via ``model_dump()``). The
        analyst mesh's ``BaseAnalyst._format_market_block`` already
        surfaces ``metadata["news_context"]`` / ``metadata["news"]`` in
        the LLM prompt; on-chain and social are available to specialised
        analyst subclasses that look them up.

        ``metadata`` also gets two provenance keys:

        - ``enriched_at``: ISO 8601 timestamp of the enrichment pass.
        - ``enrichment_source``: the string ``"pythia-strata"`` so the
          audit trail can tell enrichment-produced metadata apart from
          metadata injected by other layers.
        """
        metadata: dict[str, Any] = {
            "news": [a.model_dump() for a in enriched.news],
            "on_chain": [m.model_dump() for m in enriched.on_chain],
            "social": [s.model_dump() for s in enriched.social],
            "enriched_at": enriched.enriched_at,
            "enrichment_source": "pythia-strata",
            # Convenience: a single-string headline rollup that the
            # analyst mesh's _format_market_block will surface directly.
            # Keeps the LLM prompt compact even with 5 articles.
            "news_context": _rollup_news(enriched.news),
        }
        return MarketContext(
            market_id=enriched.market_id,
            question=enriched.question,
            category=enriched.category,
            metadata=metadata,
            current_yes_price=enriched.current_yes_price,
            current_no_price=enriched.current_no_price,
            volume_usd=enriched.volume_usd,
            closes_at=enriched.closes_at,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """Tokenize ``question`` into a list of meaningful keywords.

        Strategy: lowercase, split on non-alphanumeric runs, drop
        stopwords and single-character tokens. Returns at most 8 tokens
        — enough to seed a news search without producing an over-long
        query string.
        """
        if not question:
            return []
        tokens = _TOKEN_RE.split(question.lower())
        keywords = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
        return keywords[:8]

    @staticmethod
    def _is_crypto_market(market: Market) -> bool:
        """Heuristic: does this market reference a crypto asset?

        We look at the category string (case-insensitive) and at the
        question text for token-ticker patterns (``$ETH``, ``BTC``,
        ``$SOL``, etc.). The category check is the primary signal; the
        ticker scan is a backup for mis-categorised markets.
        """
        category = (_category_str(market) or "").upper()
        if "CRYPTO" in category or "DIGITAL" in category or "TOKEN" in category:
            return True
        # Ticker scan: $TICKER or 2-5 char all-caps word followed by
        # price/volume context. Conservative — we'd rather under-match
        # (and skip on-chain enrichment) than over-match (and waste a
        # provider call on a non-crypto market).
        question = (getattr(market, "question", "") or "").upper()
        _CRYPTO_TICKER_RE = r"\$(BTC|ETH|SOL|USDC|USDT|BNB|XRP|ADA|DOGE|AVAX|DOT|MATIC)\b"
        return bool(re.search(_CRYPTO_TICKER_RE, question))

    @staticmethod
    def _extract_token_symbol(market: Market) -> str | None:
        """Try to identify the token ticker a crypto market is about.

        Looks for ``$TICKER`` patterns first (most reliable), then for
        known tickers as bare words. Returns ``None`` if no token is
        identifiable — the on-chain provider is then skipped (its stub
        returns ``[]`` anyway, but the skip avoids a wasted call once
        the provider is wired up).
        """
        question = getattr(market, "question", "") or ""
        # $TICKER pattern — explicit, most reliable.
        m = re.search(r"\$([A-Z]{2,6})\b", question)
        if m:
            return m.group(1)
        # Bare known tickers.
        _BARE_TICKER_RE = r"\b(BTC|ETH|SOL|USDC|USDT|BNB|XRP|ADA|DOGE|AVAX|DOT|MATIC)\b"
        m = re.search(_BARE_TICKER_RE, question.upper())
        if m:
            return m.group(1)
        return None

    @staticmethod
    async def _safe_fetch(coro: Any) -> list[Any]:
        """Await ``coro`` and convert any exception to an empty list.

        This is the per-provider soft-fail wrapper. ``MarketEnricher.enrich``
        also passes ``return_exceptions=True`` to ``asyncio.gather`` as
        a second line of defense — but this wrapper is what actually
        logs the failure and converts to ``[]``.
        """
        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001 — providers may raise anything
            logger.warning(
                "enrichment provider soft-failed, returning []: %r",
                exc,
            )
            return []
        # Defensive: a misbehaving provider might return None or a non-list.
        if result is None:
            return []
        if not isinstance(result, list):
            logger.warning(
                "enrichment provider returned non-list %r, treating as []",
                type(result).__name__,
            )
            return []
        return result

# ---------------------------------------------------------------------------
# Module-level helpers (kept at module scope so they're easily testable)
# ---------------------------------------------------------------------------

async def _return_empty_list() -> list[Any]:
    """Async coroutine that returns ``[]`` — used to skip a stratum cleanly."""
    return []

def _coerce_to_list(
    value: Any,
    expected_type: type,
    stratum_name: str,
    market_id: str,
) -> list[Any]:
    """Coerce a gather() result into a list of the expected pydantic type.

    Handles three cases:
    - ``value`` is already a list of the expected type → return as-is.
    - ``value`` is an Exception (gather's ``return_exceptions=True``
      output) → log + return [].
    - ``value`` is anything else → log + return [].
    """
    if isinstance(value, Exception):
        logger.warning(
            "stratum=%s market=%s raised during gather: %r",
            stratum_name,
            market_id,
            value,
        )
        return []
    if isinstance(value, list):
        # Filter to only items of the expected type — defensive against
        # a provider returning a list with junk in it.
        return [v for v in value if isinstance(v, expected_type)]
    logger.warning(
        "stratum=%s market=%s returned non-list %r, treating as []",
        stratum_name,
        market_id,
        type(value).__name__,
    )
    return []

def _category_str(market: Market) -> str:
    """Extract a plain string category from a Market (handles enum or str)."""
    cat = getattr(market, "category", None)
    if cat is None:
        return "OTHER"
    # ``MarketCategory`` enum → use .value if available, else str().
    value = getattr(cat, "value", None)
    if isinstance(value, str):
        return value
    return str(cat)

def _closes_at_str(market: Market) -> str | None:
    """Extract ``closes_at`` as an ISO 8601 string from a Market.

    The adapter's ``Market.closes_at`` is a ``datetime | None``; the
    fallback ``Market`` is already a ``str | None``. Handle both.
    """
    closes_at = getattr(market, "closes_at", None)
    if closes_at is None:
        return None
    if isinstance(closes_at, str):
        # Already a string — pass through (validation happens in
        # EnrichedMarket's field_validator).
        return closes_at
    if isinstance(closes_at, datetime):
        # Convert datetime → ISO 8601. If naive, assume UTC.
        if closes_at.tzinfo is None:
            closes_at = closes_at.replace(tzinfo=UTC)
        return closes_at.isoformat()
    # Unknown type — best-effort string conversion.
    return str(closes_at)

def _rollup_news(articles: list[NewsArticle]) -> str:
    """Build a single-string rollup of the news headlines for the LLM prompt.

    The analyst mesh's ``_format_market_block`` looks at
    ``metadata["news_context"]`` and surfaces it as ``Recent context: ...``
    in the prompt. We produce a compact rollup so the LLM sees the gist
    of the headlines without consuming the full article bodies.

    Format: ``"source: headline; source: headline; ..."`` (max 5 items,
    each headline truncated to ~100 chars).
    """
    if not articles:
        return ""
    parts: list[str] = []
    for a in articles[:5]:
        headline = (a.title or "").strip()
        if len(headline) > 100:
            headline = headline[:97] + "..."
        source = (a.source or "?").strip()
        parts.append(f"{source}: {headline}")
    return "; ".join(parts)

__all__ = ["MarketEnricher"]
