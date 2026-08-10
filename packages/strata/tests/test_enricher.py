"""Tests for ``pythia_strata.enricher.MarketEnricher``.

These tests exercise:

1. ``enrich()`` with all-stub providers returns an ``EnrichedMarket``
   with the three enrichment lists empty but well-formed (correct
   market_id, question, category, prices, enriched_at, etc.).
2. ``to_market_context()`` conversion: the three enrichment strata land
   in ``metadata["news"]`` / ``metadata["on_chain"]`` / ``metadata["social"]``
   as JSON-serialisable lists of dicts, and the provenance keys
   (``enriched_at``, ``enrichment_source``) are set correctly.
3. Keyword extraction, token-symbol extraction, crypto-market detection.
4. The soft-fail path: a provider that raises is converted to an empty
   list rather than propagating.
5. ``_coerce_to_list`` and ``_safe_fetch`` helpers.
6. Crypto vs non-crypto markets: on-chain stratum is only attempted for
   crypto-category markets.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pythia_strata import (
    EnrichedMarket,
    MarketEnricher,
    NewsArticle,
    NewsProvider,
    OnChainMetric,
    OnChainProvider,
    SocialProvider,
    SocialSignal,
)
from pythia_strata.enricher import (
    _category_str,
    _closes_at_str,
    _coerce_to_list,
    _rollup_news,
)
from pythia_strata.types import Market

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_market(
    *,
    market_id: str = "delphi-2026-eth-above-4k",
    question: str = "Will ETH close above $4,000 on January 31, 2026?",
    category: str = "CRYPTO",
    yes_price: float = 0.42,
    no_price: float = 0.58,
    volume_usd: float = 12_345.67,
    closes_at: str | None = "2026-01-31T23:59:59+00:00",
) -> Market:
    """Build a Market using whichever definition is in scope.

    Works whether or not ``pythia_delphi_adapter`` is installed:

    - If installed, ``Market`` is the adapter's pydantic model — which
      requires ``status`` and ``created_at`` fields (we provide sensible
      defaults so the test fixture works without callers having to know).
    - If not installed, ``Market`` is the local fallback in
      ``pythia_strata.types``, which uses ``extra="allow"`` so the extra
      ``status`` / ``created_at`` keys are simply ignored.
    """
    return Market.model_validate(
        {
            "market_id": market_id,
            "question": question,
            "category": category,
            "status": "OPEN",
            "yes_price": yes_price,
            "no_price": no_price,
            "volume_usd": volume_usd,
            "liquidity_usd": 5_000.0,
            "created_at": "2026-01-15T00:00:00+00:00",
            "closes_at": closes_at,
        }
    )

def _make_politics_market() -> Market:
    return _make_market(
        market_id="delphi-2026-election",
        question="Will the incumbent win the 2026 midterm election?",
        category="POLITICS",
        yes_price=0.55,
        no_price=0.45,
        volume_usd=98_765.43,
        closes_at="2026-11-04T20:00:00+00:00",
    )

# ---------------------------------------------------------------------------
# enrich() with all-stub providers
# ---------------------------------------------------------------------------

class TestEnrichWithStubs:
    async def test_returns_enriched_market_with_empty_lists(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)

        assert isinstance(enriched, EnrichedMarket)
        assert enriched.market_id == market.market_id
        assert enriched.question == market.question
        assert enriched.category == "CRYPTO"
        assert enriched.current_yes_price == 0.42
        assert enriched.current_no_price == 0.58
        assert enriched.volume_usd == 12_345.67
        assert enriched.closes_at == "2026-01-31T23:59:59+00:00"
        # All three strata are empty (stub providers return []).
        assert enriched.news == []
        assert enriched.on_chain == []
        assert enriched.social == []
        # enriched_at is an ISO 8601 timestamp.
        assert enriched.enriched_at
        datetime.fromisoformat(enriched.enriched_at)

    async def test_enriched_at_is_recent(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        before = datetime.now(UTC)
        enriched = await enricher.enrich(market)
        after = datetime.now(UTC)

        enriched_at = datetime.fromisoformat(enriched.enriched_at)
        # enriched_at should be between `before` and `after` (allowing
        # for tiny clock skew).
        assert before - timedelta(seconds=1) <= enriched_at <= after + timedelta(seconds=1)

    async def test_politics_market_skips_onchain(self) -> None:
        """Non-crypto market: on-chain stratum should be skipped (empty)."""
        market = _make_politics_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        assert enriched.category == "POLITICS"
        assert enriched.on_chain == []
        # News and social still attempted (returned [] by stubs).
        assert enriched.news == []
        assert enriched.social == []

# ---------------------------------------------------------------------------
# to_market_context() conversion
# ---------------------------------------------------------------------------

class TestToMarketContext:
    async def test_basic_conversion(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        ctx = enricher.to_market_context(enriched)

        assert ctx.market_id == enriched.market_id
        assert ctx.question == enriched.question
        assert ctx.category == enriched.category
        assert ctx.current_yes_price == enriched.current_yes_price
        assert ctx.current_no_price == enriched.current_no_price
        assert ctx.volume_usd == enriched.volume_usd
        assert ctx.closes_at == enriched.closes_at

    async def test_metadata_contains_enrichment_strata(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        ctx = enricher.to_market_context(enriched)

        assert "news" in ctx.metadata
        assert "on_chain" in ctx.metadata
        assert "social" in ctx.metadata
        assert isinstance(ctx.metadata["news"], list)
        assert isinstance(ctx.metadata["on_chain"], list)
        assert isinstance(ctx.metadata["social"], list)
        # All empty because we used stub providers.
        assert ctx.metadata["news"] == []
        assert ctx.metadata["on_chain"] == []
        assert ctx.metadata["social"] == []

    async def test_metadata_contains_provenance_keys(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        ctx = enricher.to_market_context(enriched)

        assert ctx.metadata["enrichment_source"] == "pythia-strata"
        assert ctx.metadata["enriched_at"] == enriched.enriched_at
        assert "news_context" in ctx.metadata

    async def test_metadata_is_json_serialisable(self) -> None:
        """metadata must round-trip through json.dumps (for the audit log)."""
        import json

        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        ctx = enricher.to_market_context(enriched)

        # Should not raise.
        serialised = json.dumps(ctx.model_dump())
        assert "news" in serialised
        assert "enrichment_source" in serialised

    async def test_news_context_rollup_empty(self) -> None:
        """With no news articles, news_context should be empty string."""
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        ctx = enricher.to_market_context(enriched)
        assert ctx.metadata["news_context"] == ""

    async def test_to_market_context_with_populated_strata(self) -> None:
        """When the EnrichedMarket actually has news / on-chain / social,
        the conversion should embed them in metadata as dicts.
        """
        enriched = EnrichedMarket(
            market_id="mkt-1",
            question="Will ETH close above $4k?",
            category="CRYPTO",
            current_yes_price=0.42,
            current_no_price=0.58,
            volume_usd=1234.5,
            closes_at="2026-01-31T23:59:59+00:00",
            news=[
                NewsArticle(
                    title="ETH breaks $4k resistance",
                    url="https://example.com/eth-4k",
                    source="coindesk",
                    published_at="2026-01-30T10:00:00+00:00",
                    summary="Ethereum crossed $4k for the first time.",
                    sentiment_score=0.7,
                ),
            ],
            on_chain=[
                OnChainMetric(
                    token_symbol="ETH",
                    metric_name="tvl_eth",
                    value=15_000_000.0,
                    timestamp="2026-01-30T10:00:00+00:00",
                    source="defillama",
                ),
            ],
            social=[
                SocialSignal(
                    platform="twitter",
                    post_count_24h=1234,
                    avg_sentiment=0.3,
                    top_keywords=["eth", "merge", "4k"],
                    timestamp="2026-01-30T10:00:00+00:00",
                ),
            ],
        )
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        ctx = enricher.to_market_context(enriched)

        assert len(ctx.metadata["news"]) == 1
        assert ctx.metadata["news"][0]["title"] == "ETH breaks $4k resistance"
        assert ctx.metadata["news"][0]["url"] == "https://example.com/eth-4k"

        assert len(ctx.metadata["on_chain"]) == 1
        assert ctx.metadata["on_chain"][0]["metric_name"] == "tvl_eth"
        assert ctx.metadata["on_chain"][0]["value"] == 15_000_000.0

        assert len(ctx.metadata["social"]) == 1
        assert ctx.metadata["social"][0]["platform"] == "twitter"
        assert ctx.metadata["social"][0]["post_count_24h"] == 1234

        # news_context rollup should mention the headline.
        assert "ETH breaks $4k resistance" in ctx.metadata["news_context"]

# ---------------------------------------------------------------------------
# Keyword / token extraction helpers
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_strips_stopwords(self) -> None:
        kw = MarketEnricher._extract_keywords("Will the ETH close above $4,000?")
        # "Will", "the", "above" are stopwords → filtered out.
        assert "will" not in kw
        assert "the" not in kw
        assert "above" not in kw
        assert "eth" in kw
        assert "close" in kw
        assert "4" in kw or "000" in kw  # "$4,000" → ["4", "000"]

    def test_lowercase(self) -> None:
        kw = MarketEnricher._extract_keywords("Ethereum Bitcoin SOLANA")
        assert "ethereum" in kw
        assert "bitcoin" in kw
        assert "solana" in kw

    def test_max_8_tokens(self) -> None:
        long_q = " ".join(f"word{i}" for i in range(20))
        kw = MarketEnricher._extract_keywords(long_q)
        assert len(kw) <= 8

    def test_empty_question(self) -> None:
        assert MarketEnricher._extract_keywords("") == []

    def test_punctuation_only(self) -> None:
        assert MarketEnricher._extract_keywords("?!.,") == []

    def test_single_char_tokens_filtered(self) -> None:
        # Single-character tokens are dropped (require len >= 2).
        kw = MarketEnricher._extract_keywords("a b c real")
        assert "a" not in kw
        assert "b" not in kw
        assert "c" not in kw
        assert "real" in kw

class TestIsCryptoMarket:
    def test_crypto_category(self) -> None:
        m = _make_market(category="CRYPTO")
        assert MarketEnricher._is_crypto_market(m) is True

    def test_politics_category(self) -> None:
        m = _make_politics_market()
        assert MarketEnricher._is_crypto_market(m) is False

    def test_ticker_in_question_triggers_crypto(self) -> None:
        m = _make_market(
            question="Will $BTC halving boost miner revenue?",
            category="OTHER",
        )
        assert MarketEnricher._is_crypto_market(m) is True

    def test_no_ticker_no_crypto_category(self) -> None:
        m = _make_market(
            question="Will the Lakers win the championship?",
            category="SPORTS",
        )
        assert MarketEnricher._is_crypto_market(m) is False

class TestExtractTokenSymbol:
    def test_dollar_ticker(self) -> None:
        m = _make_market(question="Will $ETH close above $4,000?")
        assert MarketEnricher._extract_token_symbol(m) == "ETH"

    def test_bare_ticker(self) -> None:
        m = _make_market(question="Will BTC reach a new all-time high?")
        assert MarketEnricher._extract_token_symbol(m) == "BTC"

    def test_no_ticker(self) -> None:
        m = _make_politics_market()
        assert MarketEnricher._extract_token_symbol(m) is None

    def test_dollar_ticker_preferred_over_bare(self) -> None:
        # Both $ETH and "ETH" appear; $ETH should win.
        m = _make_market(question="Will $ETH outperform ETH staking yields?")
        assert MarketEnricher._extract_token_symbol(m) == "ETH"

# ---------------------------------------------------------------------------
# Soft-fail: provider that raises is converted to []
# ---------------------------------------------------------------------------

class _RaisingNewsProvider(NewsProvider):
    """A NewsProvider whose fetch() always raises — used to test soft-fail."""

    async def fetch(self, query: str, limit: int = 5) -> list[NewsArticle]:
        raise RuntimeError("simulated upstream failure")

class _RaisingOnChainProvider(OnChainProvider):
    async def fetch(self, token_symbol: str | None = None) -> list[OnChainMetric]:
        raise ConnectionError("simulated on-chain API outage")

class _RaisingSocialProvider(SocialProvider):
    async def fetch(self, query: str, limit: int = 5) -> list[SocialSignal]:
        raise ValueError("simulated social API parse failure")

class TestSoftFail:
    async def test_raising_news_provider_yields_empty_news(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=_RaisingNewsProvider(),
            onchain=OnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        assert enriched.news == []
        # Other strata still work.
        # (on_chain is [] because stub; social is [] because stub.)
        assert enriched.on_chain == []
        assert enriched.social == []

    async def test_raising_onchain_provider_yields_empty_onchain(self) -> None:
        market = _make_market()  # crypto category
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=_RaisingOnChainProvider(),
            social=SocialProvider(),
        )
        enriched = await enricher.enrich(market)
        assert enriched.on_chain == []

    async def test_raising_social_provider_yields_empty_social(self) -> None:
        market = _make_market()
        enricher = MarketEnricher(
            news=NewsProvider(),
            onchain=OnChainProvider(),
            social=_RaisingSocialProvider(),
        )
        enriched = await enricher.enrich(market)
        assert enriched.social == []

    async def test_all_providers_raising_still_returns_enriched_market(self) -> None:
        """All three providers raise — enrich() must still return a valid EnrichedMarket."""
        market = _make_market()
        enricher = MarketEnricher(
            news=_RaisingNewsProvider(),
            onchain=_RaisingOnChainProvider(),
            social=_RaisingSocialProvider(),
        )
        enriched = await enricher.enrich(market)
        assert isinstance(enriched, EnrichedMarket)
        assert enriched.market_id == market.market_id
        assert enriched.news == []
        assert enriched.on_chain == []
        assert enriched.social == []
        assert enriched.enriched_at  # timestamp still set

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCoerceToList:
    def test_returns_list_unchanged(self) -> None:
        articles = [NewsArticle(title="t", url="u", source="s")]
        result = _coerce_to_list(articles, NewsArticle, "news", "mkt-1")
        assert result == articles

    def test_filters_non_matching_types(self) -> None:
        mixed: list[Any] = [
            NewsArticle(title="t", url="u", source="s"),
            "not an article",
            42,
            None,
        ]
        result = _coerce_to_list(mixed, NewsArticle, "news", "mkt-1")
        assert len(result) == 1
        assert isinstance(result[0], NewsArticle)

    def test_exception_becomes_empty_list(self) -> None:
        exc = RuntimeError("simulated")
        result = _coerce_to_list(exc, NewsArticle, "news", "mkt-1")
        assert result == []

    def test_none_becomes_empty_list(self) -> None:
        result = _coerce_to_list(None, NewsArticle, "news", "mkt-1")
        assert result == []

    def test_non_list_non_exception_becomes_empty_list(self) -> None:
        result = _coerce_to_list("not a list", NewsArticle, "news", "mkt-1")
        assert result == []
        result = _coerce_to_list(42, NewsArticle, "news", "mkt-1")
        assert result == []

class TestCategoryStr:
    def test_plain_string(self) -> None:
        m = _make_market(category="CRYPTO")
        assert _category_str(m) == "CRYPTO"

    def test_none_category_returns_other(self) -> None:
        # Build a duck-typed Market-like object whose category is None.
        # (The real adapter's Market always has a non-None category, but
        # _category_str is defensive — it should return "OTHER" for None.)
        class FakeMarket:
            market_id = "x"
            question = "q"
            category = None

        assert _category_str(FakeMarket()) == "OTHER"  # type: ignore[arg-type]

    def test_enum_value_extracted(self) -> None:
        # When pythia_delphi_adapter is installed, Market.category is a
        # MarketCategory enum (str, Enum). _category_str should return
        # the .value, not the repr.
        m = _make_market(category="CRYPTO")
        result = _category_str(m)
        # Either "CRYPTO" (string passthrough) or the enum's .value="CRYPTO"
        # — both are acceptable as long as it's the plain string.
        assert result == "CRYPTO"

class TestClosesAtStr:
    def test_string_passthrough(self) -> None:
        m = _make_market(closes_at="2026-01-31T23:59:59+00:00")
        assert _closes_at_str(m) == "2026-01-31T23:59:59+00:00"

    def test_none_closes_at(self) -> None:
        m = _make_market(closes_at=None)
        assert _closes_at_str(m) is None

    def test_datetime_converted_to_iso(self) -> None:
        # Build a Market-like object whose closes_at is a datetime
        # (as the real pythia_delphi_adapter Market does).
        class FakeMarket:
            market_id = "x"
            question = "q"
            category = "OTHER"
            closes_at = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)

        result = _closes_at_str(FakeMarket())  # type: ignore[arg-type]
        assert isinstance(result, str)
        assert "2026-01-31" in result

    def test_naive_datetime_assumed_utc(self) -> None:
        class FakeMarket:
            market_id = "x"
            question = "q"
            category = "OTHER"
            closes_at = datetime(2026, 1, 31, 23, 59, 59)  # naive

        result = _closes_at_str(FakeMarket())  # type: ignore[arg-type]
        # Should have a tz designator (UTC).
        assert "+00:00" in result

class TestRollupNews:
    def test_empty_articles(self) -> None:
        assert _rollup_news([]) == ""

    def test_single_article(self) -> None:
        articles = [
            NewsArticle(title="ETH breaks $4k", url="u", source="coindesk"),
        ]
        rollup = _rollup_news(articles)
        assert "coindesk" in rollup
        assert "ETH breaks $4k" in rollup

    def test_multiple_articles_joined_with_semicolon(self) -> None:
        articles = [
            NewsArticle(title="Headline A", url="u1", source="src1"),
            NewsArticle(title="Headline B", url="u2", source="src2"),
        ]
        rollup = _rollup_news(articles)
        assert "src1: Headline A" in rollup
        assert "src2: Headline B" in rollup
        assert ";" in rollup

    def test_truncates_long_headlines(self) -> None:
        long_title = "A" * 200
        articles = [NewsArticle(title=long_title, url="u", source="s")]
        rollup = _rollup_news(articles)
        # The headline in the rollup should be truncated to ~100 chars.
        # We look for the ellipsis that marks the truncation.
        assert "..." in rollup
        # The full 200-char title should NOT be present.
        assert long_title not in rollup

    def test_max_five_articles(self) -> None:
        articles = [
            NewsArticle(title=f"Headline {i}", url=f"u{i}", source=f"s{i}") for i in range(10)
        ]
        rollup = _rollup_news(articles)
        # Only the first 5 should appear.
        assert "Headline 0" in rollup
        assert "Headline 4" in rollup
        assert "Headline 5" not in rollup
