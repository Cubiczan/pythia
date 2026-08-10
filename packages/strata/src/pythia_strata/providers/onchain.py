"""On-chain enrichment provider.

Pulls on-chain metrics (TVL, active addresses, exchange flows, etc.) for
the token a crypto-category market references, for the ``on_chain``
stratum of ``EnrichedMarket``.

Status: stub. ``fetch`` always returns ``[]`` (regardless of inputs)
until wired up. ``MarketEnricher.enrich`` only invokes this provider for
crypto-category markets — see ``MarketEnricher._is_crypto_market``.

# VERIFY: which on-chain data source to wire up. Candidates:
  - DefiLlama (free, no key, TVL + protocol metrics + yields) — preferred
    default. Endpoints: https://api.llama.fi/v2/historicalChainTvl,
    https://api.llama.fi/protocols, https://coins.llama.fi/prices.
  - Glassnode (paid, deep on-chain metrics, requires key)
  - Dune Analytics (SQL-over-on-chain, requires API key + query IDs;
    most flexible but most setup).
  - CoinGecko (free tier, market data + community metrics; less deep
    on-chain coverage than DefiLlama but easier auth).

The stub below is structured so that wiring up DefiLlama amounts to:
  1. Implementing the body of ``fetch`` with an ``httpx.AsyncClient``
     call to the DefiLlama endpoint for the requested token.
  2. Mapping the response to ``OnChainMetric`` records (one per metric).
  3. Wrapping the whole call in a try/except that logs + returns ``[]``.
"""

from __future__ import annotations

import logging

from pythia_strata.types import OnChainMetric

logger = logging.getLogger(__name__)


class OnChainProvider:
    """Fetches on-chain metrics for a token (crypto-category markets only).

    Soft-fail contract: ``fetch`` ALWAYS returns a ``list[OnChainMetric]``
    and never raises. If the upstream call fails for any reason, or if no
    token symbol is provided, it logs a warning and returns ``[]``.

    Parameters
    ----------
    api_key:
        Optional API key for the upstream provider. DefiLlama needs no
        key (free public API); Glassnode and Dune require one. ``None``
        by default.

        # VERIFY: per-provider auth shape once a source is chosen.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    async def fetch(self, token_symbol: str | None = None) -> list[OnChainMetric]:
        """Return on-chain metrics for ``token_symbol``.

        Parameters
        ----------
        token_symbol:
            Ticker to fetch metrics for (e.g. ``"ETH"``, ``"BTC"``).
            ``None`` is valid and signals "no specific token" — the
            provider may return protocol-level metrics (e.g. total DEX
            volume) or simply ``[]`` (the stub does the latter).

        Returns
        -------
        list[OnChainMetric]
            Possibly empty. Never raises.
        """
        if not token_symbol or not token_symbol.strip():
            # No token to look up — return [] (stub doesn't surface
            # protocol-level metrics).
            logger.debug("OnChainProvider stub path: no token_symbol provided, returning []")
            return []

        # --- Wired-up path (not yet implemented) ------------------------
        # Once an on-chain source is chosen, implement the fetch here.
        # DefiLlama skeleton (no key needed):
        #
        #   try:
        #       async with httpx.AsyncClient(timeout=15.0) as client:
        #           # Total TVL for the token's chain — DefiLlama exposes
        #           # /v2/historicalChainTvl and /protocols.
        #           resp = await client.get(
        #               "https://api.llama.fi/protocols",
        #               params={"symbol": token_symbol.upper()},
        #           )
        #           resp.raise_for_status()
        #           data = resp.json()
        #       now = datetime.now(UTC).isoformat()
        #       return [
        #           OnChainMetric(
        #               token_symbol=token_symbol.upper(),
        #               metric_name="tvl_usd",
        #               value=float(p.get("tvl", 0.0)),
        #               timestamp=now,
        #               source="defillama",
        #           )
        #           for p in (data if isinstance(data, list) else [data])
        #       ]
        #   except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        #       logger.warning(
        #           "OnChainProvider upstream error for token=%r: %r",
        #           token_symbol, exc,
        #       )
        #       return []
        #
        # VERIFY: which provider, endpoint shape, response envelope,
        # metric_name conventions (see OnChainMetric docstring for the
        # canonical names: tvl_eth, active_addresses_7d, etc.).

        logger.warning(
            "OnChainProvider stub: no upstream wired up yet, returning [] "
            "(token_symbol=%r). See providers/onchain.py # VERIFY.",
            token_symbol,
        )
        return []


__all__ = ["OnChainProvider"]
