"""Tests for the three enrichment providers (news / on-chain / social).

These tests exercise the stub paths only — they confirm:

1. Each provider's ``fetch`` returns ``[]`` (the soft-fail default) when
   no API key is configured.
2. Each provider's ``fetch`` returns ``[]`` for empty / whitespace-only
   inputs (defensive — the enricher should never feed these in, but the
   providers shouldn't crash if it does).
3. Each provider's ``fetch`` returns ``[]`` even when an API key IS set
   but no upstream is wired up yet (the stub-with-key path emits a
   warning and falls back to ``[]``).

Once the providers are wired up to real upstreams, replace these tests
with httpx.MockTransport-based ones (same pattern as
``pythia-delphi-adapter/tests/test_client.py``).
"""

from __future__ import annotations

import pytest

from pythia_strata.providers import NewsProvider, OnChainProvider, SocialProvider

# ---------------------------------------------------------------------------
# NewsProvider
# ---------------------------------------------------------------------------
class TestNewsProviderStub:
    async def test_returns_empty_without_api_key(self) -> None:
        provider = NewsProvider()  # no api_key
        result = await provider.fetch("ethereum merge", limit=5)
        assert result == []
        assert isinstance(result, list)

    async def test_returns_empty_for_empty_query(self) -> None:
        provider = NewsProvider(api_key="fake-key")
        assert await provider.fetch("", limit=5) == []
        assert await provider.fetch("   ", limit=5) == []

    async def test_returns_empty_for_whitespace_query(self) -> None:
        provider = NewsProvider(api_key="fake-key")
        assert await provider.fetch("\t\n", limit=5) == []

    async def test_returns_empty_when_api_key_set_but_no_upstream(self) -> None:
        # API key is set, but the stub wired-up path isn't implemented yet 
        # provider should warn + return [] rather than crash.
        provider = NewsProvider(api_key="fake-key")
        result = await provider.fetch("ethereum merge", limit=5)
        assert result == []

    async def test_default_limit_is_5(self) -> None:
        # No assertion on the actual limit being honoured (stub returns []
        # regardless), but we do confirm the call signature accepts the
        # default and doesn't raise.
        provider = NewsProvider()
        result = await provider.fetch("anything")
        assert result == []

# ---------------------------------------------------------------------------
# OnChainProvider
# ---------------------------------------------------------------------------
class TestOnChainProviderStub:
    async def test_returns_empty_without_token_symbol(self) -> None:
        provider = OnChainProvider()
        assert await provider.fetch() == []
        assert await provider.fetch(None) == []
        assert await provider.fetch("") == []
        assert await provider.fetch("   ") == []

    async def test_returns_empty_with_token_but_no_upstream(self) -> None:
        provider = OnChainProvider()  # no api_key
        assert await provider.fetch("ETH") == []

    async def test_returns_empty_with_api_key_set_but_no_upstream(self) -> None:
        provider = OnChainProvider(api_key="fake-key")
        result = await provider.fetch("BTC")
        assert result == []

    async def test_returns_list_type(self) -> None:
        provider = OnChainProvider()
        result = await provider.fetch("ETH")
        assert isinstance(result, list)

# ---------------------------------------------------------------------------
# SocialProvider
# ---------------------------------------------------------------------------
class TestSocialProviderStub:
    async def test_returns_empty_without_api_key(self) -> None:
        provider = SocialProvider()  # no api_key
        result = await provider.fetch("ethereum merge", limit=5)
        assert result == []
        assert isinstance(result, list)

    async def test_returns_empty_for_empty_query(self) -> None:
        provider = SocialProvider(api_key="fake-key")
        assert await provider.fetch("", limit=5) == []
        assert await provider.fetch("   ", limit=5) == []

    async def test_returns_empty_when_api_key_set_but_no_upstream(self) -> None:
        provider = SocialProvider(api_key="fake-key")
        result = await provider.fetch("ethereum merge", limit=5)
        assert result == []

    async def test_default_limit_is_5(self) -> None:
        provider = SocialProvider()
        result = await provider.fetch("anything")
        assert result == []

# ---------------------------------------------------------------------------
# Cross-provider: all three obey the soft-fail contract
# ---------------------------------------------------------------------------
class TestSoftFailContract:
    """All providers must return [] (never raise) for any input."""

    @pytest.mark.parametrize(
        "provider,query",
        [
            (NewsProvider(), "any query"),
            (OnChainProvider(), "ETH"),
            (SocialProvider(), "any query"),
        ],
    )
    async def test_never_raises_on_normal_input(
        self,
        provider: NewsProvider | OnChainProvider | SocialProvider,
        query: str,
    ) -> None:
        # Should never raise, regardless of upstream state.
        result = await provider.fetch(query)  # type: ignore[arg-type]
        assert isinstance(result, list)
