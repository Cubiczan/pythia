# pythia-strata

> Stratified data ingestion + enrichment layer that wraps
> [`icohangar-ops/strata`](https://github.com/icohangar-ops/strata) for the
> **Pythia** multi-agent trading mesh.

`pythia-strata` sits between **`pythia-delphi-adapter`** (raw ATT market
fetch) and **`pythia-analyst-mesh`** (specialist LLM analysts). It takes a
bare `Market` from Delphi and layers on three strata of enrichment
context — **news**, **on-chain metrics**, and **social signals** — before
producing the `MarketContext` the analyst mesh consumes.

The single contract the rest of the mesh relies on:

> *Enrichment is always non-blocking. A flaky upstream returns an empty
> list, never raises. Every `EnrichedMarket` is well-formed even if every
> provider is down.*

This is the wedge that makes Pythia robust on Delphi's niche, subjective
markets: even when LLMs are limited, analysts still see real-world
context (recent headlines, on-chain TVL, social momentum) instead of the
bare market question.

---

## What this adds over the upstream

| Concern | Upstream `strata` | This wrapper |
|---|---|---|
| Stratified ingest primitives | ✅ | re-exported, optional |
| **Delphi market fetch orchestration** | ❌ | ✅ (via `pythia-delphi-adapter`) |
| **News / on-chain / social enrichment** | ❌ | ✅ (stub + interface each) |
| **`EnrichedMarket` schema** (typed pydantic v2) | ❌ | ✅ |
| **`MarketContext` conversion** | ❌ | ✅ (consumed by `pythia-analyst-mesh`) |
| **Parallel enrichment** (`asyncio.gather`) + soft-fail | ❌ | ✅ |
| CLI (`enrich <id>` / `watch --interval N`) | ❌ | ✅ |

The upstream is a **black box**. If its public API changes, only the thin
re-export in `__init__.py` needs updating; the enrichment math, providers,
and `EnrichedMarket` schema are fully self-contained and tested without it.

---

## The `MarketContext` schema

The analyst mesh's input contract (defined in `pythia-analyst-mesh`):

```python
class MarketContext(BaseModel):
    market_id: str
    question: str
    category: str
    metadata: dict[str, Any]  # ← enrichment lands here
    current_yes_price: float | None  # 0..1
    current_no_price: float | None  # 0..1
    volume_usd: float | None
    closes_at: str | None  # ISO 8601
```

`pythia-strata` populates `metadata` with these keys, each JSON-serialisable:

| Key | Type | Notes |
|---|---|---|
| `news` | `list[dict]` | Recent `NewsArticle`s relevant to the question. |
| `on_chain` | `list[dict]` | `OnChainMetric`s (only for crypto-category markets). |
| `social` | `list[dict]` | `SocialSignal`s aggregated per platform. |
| `enriched_at` | `str` | ISO 8601 timestamp of the enrichment pass. |
| `enrichment_source` | `str` | `"pythia-strata"` — auditable provenance. |

The analyst mesh's `BaseAnalyst._format_market_block` already looks at
`metadata["news_context"]` / `metadata["news"]` for headline context.

---

## The three enrichment strata

### News — `NewsProvider`

`async def fetch(query: str, limit: int = 5) -> list[NewsArticle]`

Recent headlines relevant to the market question. Gives analysts current
context: court rulings, regulatory filings, earnings, breaking sports news.

**Status:** stub. Returns `[]` without an `api_key`. **# VERIFY:** which
provider to use — candidates: **GDELT** (free, no key, *preferred default*),
**NewsAPI.org** (free tier 100 req/day), **Bing News Search** (Azure),
**AlphaVantage News & Sentiment** (financial focus).

### On-chain — `OnChainProvider`

`async def fetch(token_symbol: str | None = None) -> list[OnChainMetric]`

On-chain metrics (TVL, active addresses, exchange flows) for the token a
crypto-category market references. **Only invoked for crypto-category
markets** — `MarketEnricher.enrich` checks `market.category` and skips
this stratum otherwise.

**Status:** stub. Returns `[]` always (until wired). **# VERIFY:** which
data source — candidates: **DefiLlama** (free, *preferred default*),
**Glassnode** (paid), **Dune Analytics** (SQL-over-on-chain),
**CoinGecko** (free tier).

### Social — `SocialProvider`

`async def fetch(query: str, limit: int = 5) -> list[SocialSignal]`

Social-platform signal aggregates: post counts, average sentiment, top
keywords — per platform (Twitter / Reddit / Farcaster).

**Status:** stub. Returns `[]` always (until wired). **# VERIFY:** which
APIs — candidates: **Twitter/X API v2** (paid), **Reddit API** (free,
OAuth), **Farcaster Hub HTTP API** (free, decentralised),
**LunarCrush** (paid social-sentiment aggregator — single API for all).

---

## Install

```bash
git clone https://github.com/icohangar-ops/pythia.git
cd pythia/packages/strata
cd packages/strata
pip install -e ".[dev]"
```

This installs `pythia-delphi-adapter` (declared as a normal Python dep).
`pythia-analyst-mesh` is *not* a hard dep — `MarketEnricher.to_market_context`
imports `MarketContext` from it lazily, falling back to a local minimal
definition if the mesh isn't installed, so `pythia-strata` is independently
testable.

### Vendoring the upstream `icohangar-ops/strata`

The enrichment logic works **without** the upstream. To use its
stratified-ingest primitives, vendor it:

```bash
git submodule add git@github.com:icohangar-ops/strata.git vendor/strata
```

Then pin the commit SHA in [`VENDOR_COMMIT.txt`](./VENDOR_COMMIT.txt) so
the ingestion lineage is reproducible.

---

## Quick start

```python
import asyncio
from pythia_delphi_adapter import DelphiClient, MarketStatus
from pythia_strata import (
    MarketEnricher,
    NewsProvider,
    OnChainProvider,
    SocialProvider,
)


async def main() -> None:
    async with DelphiClient(api_key="dphi_live_...") as delphi:
        markets = await delphi.list_markets(status=MarketStatus.OPEN, limit=5)

    enricher = MarketEnricher(
        news=NewsProvider(),  # stub — returns [] without an API key
        onchain=OnChainProvider(),  # stub
        social=SocialProvider(),  # stub
    )

    for market in markets:
        enriched = await enricher.enrich(market)
        print(enriched.market_id, enriched.question)
        print(
            f"  news={len(enriched.news)}  onchain={len(enriched.on_chain)}  social={len(enriched.social)}"
        )

        # Convert to the MarketContext the analyst mesh expects:
        context = enricher.to_market_context(enriched)
        # → pass `context` to pythia_analyst_mesh.run_mesh(context, analysts)


asyncio.run(main())
```

### CLI

```bash
# One-shot enrich of a single market (prints EnrichedMarket JSON to stdout):
pythia-strata enrich delphi-2026-eth-above-4k

# Poll Delphi for new markets, enrich each, write to stdout or a file:
pythia-strata watch --interval 60 --out enriched.jsonl
```

`enrich` exits non-zero only if the market can't be fetched from Delphi;
provider stubs returning `[]` are *not* errors. `watch` runs forever until
Ctrl-C; each enriched market is appended as a single JSON line to `--out`
(or printed to stdout if `--out` is omitted).

---

## Soft-fail contract

Every provider's `fetch()` **must** obey:

1. Return `list[...]` — never raise on upstream errors (timeouts, 5xx,
   malformed JSON, missing fields). Log a warning and return `[]`.
2. Be safe to call without configuration — `NewsProvider()` with no API
   key returns `[]` rather than raising.
3. Be safe to call concurrently — `MarketEnricher.enrich` parallelises
   all three providers with `asyncio.gather(return_exceptions=True)`; a
   raised exception is swallowed and converted to an empty result.

This guarantees the analyst mesh always gets a well-formed `MarketContext`,
even if every enrichment upstream is down. The worst case is "no
enrichment", never "no estimate".

---

## Module map

```
src/pythia_strata/
├── __init__.py        # public API
├── types.py           # NewsArticle, OnChainMetric, SocialSignal, EnrichedMarket + MarketContext re-export
├── enricher.py        # MarketEnricher.enrich() — parallel fetch + EnrichedMarket assembly
├── cli.py             # `pythia-strata enrich <id>` / `pythia-strata watch --interval N`
└── providers/{news,onchain,social}.py
```

## Testing

```bash
pytest -q
```

Tests run **without** the upstream `strata` vendored and **without** any
enrichment API configured — they exercise the stub paths (which return
`[]`), the `EnrichedMarket` schema, the parallel enrichment orchestration,
and the `MarketContext` conversion.

## License

MIT — see [`LICENSE`](./LICENSE). Copyright © 2026 icohangar-ops / Impactquadrant.

## Upstream attribution

This wrapper depends conceptually on
[`icohangar-ops/strata`](https://github.com/icohangar-ops/strata) for the
stratified-ingestion primitives. The Delphi market fetch, the three
enrichment providers, and the `EnrichedMarket` / `MarketContext` conversion
logic in this repo are original to Pythia.
