# Architecture: Pythia Multi-Agent Trading Mesh

This document describes the runtime architecture of Pythia. For the full design rationale, see [`pythia-design.pdf`](./pythia-design.pdf).

## Data flow

```
            ┌──────────────────────────────────────────────────────────┐
            │                    Delphi Protocol                        │
            │  (markets, order book, settlements, AI-as-arbiter)        │
            └───────────────────────┬──────────────────────────────────┘
                                    │  REST / WS via ATT
                            ┌───────▼───────┐
                            │  pythia-      │   pull markets,
                            │  delphi-      │   push orders,
                            │  adapter      │   listen settlements
                            └───────┬───────┘
                                    │  MarketEvent
                            ┌───────▼───────┐
                            │  pythia-      │   enrich with news,
                            │  strata       │   on-chain, social
                            └───────┬───────┘
                                    │  EnrichedMarket
            ┌───────────────────────┼────────────────────────┐
            │                       │                        │
    ┌───────▼───────┐       ┌───────▼───────┐        ┌───────▼───────┐
    │  Politics     │       │   Crypto      │  ...   │   Niche       │
    │  analyst      │       │   analyst     │        │   analyst     │
    └───────┬───────┘       └───────┬───────┘        └───────┬───────┘
            │ Estimate              │ Estimate               │ Estimate
            │ (prob, ci, why)       │ (prob, ci, why)        │ (prob, ci, why)
            └───────────────────────┼────────────────────────┘
                                    │
                            ┌───────▼───────┐
                            │  pythia-      │   fuse N estimates,
                            │  consensus    │   compute agreement
                            └───────┬───────┘
                                    │  ConsensusDecision
                                    │  (prob, agreement_score, gate)
                            ┌───────▼───────┐
                            │  pythia-risk  │   Kelly sizing,
                            │  (meshcfo)    │   exposure / drawdown gates
                            └───────┬───────┘
                                    │  TradePlan
                                    │  (market, side, size, rationale)
                            ┌───────▼───────┐
                            │  pythia-      │   sign + submit
                            │  executor     │   via ATT
                            └───────┬───────┘
                                    │  TradeReceipt
                            ┌───────▼───────┐
                            │  pythia-      │   signed audit trail,
                            │  observability│   replay UI, achievements
                            └───────────────┘
```

## Key invariants

1. **No trade without consensus.** The executor refuses to submit if `agreement_score < consensus.agreement_threshold` or `analysts_quorum < consensus.min_analysts`.
2. **No trade without risk approval.** The executor refuses to submit if `pythia-risk` returns `REJECT` (drawdown exceeded, exposure cap hit, market-type blocked).
3. **Every decision is signed and logged.** Each component emits a `SignedDecision` record (JSON + Ed25519 sig) to the audit log. Trades can be replayed end-to-end from the log alone.
4. **The mesh is swappable.** Analysts are plugins implementing `BaseAnalyst.estimate(market) -> Estimate`. You can add/remove analysts at runtime via config.
5. **Paper-first.** The executor's default `mode = "paper"`. Live mode requires an explicit `--mode live` flag *and* a `DELPHI_SIGNING_KEY` env var.

## Component contracts (summary)

See each sub-repo's README for full API.

### `Estimate` (analyst output)

```python
@dataclass
class Estimate:
    market_id: str
    probability: float          # 0.0 - 1.0, P(YES)
    confidence: float           # 0.0 - 1.0, analyst's own calibration
    rationale: str              # 1-3 sentence justification
    evidence: list[str]         # URLs, citations
    analyst_id: str
    timestamp: str              # ISO 8601
```

### `ConsensusDecision` (consensus output)

```python
@dataclass
class ConsensusDecision:
    market_id: str
    consensus_prob: float       # fused probability
    agreement_score: float      # 0.0 - 1.0, how aligned analysts are
    gate: Literal["trade", "skip", "wait"]
    contributor_ids: list[str]
    method: str                 # "logit-mean" | "median" | "trimmed-mean"
    timestamp: str
```

### `TradePlan` (risk output)

```python
@dataclass
class TradePlan:
    market_id: str
    side: Literal["YES", "NO"]
    size_usd: float
    limit_price: float | None   # None = market order
    rationale: str
    risk_flags: list[str]       # any warnings that did not block
    decision: Literal["APPROVE", "REJECT"]
    timestamp: str
```

### `TradeReceipt` (executor output)

```python
@dataclass
class TradeReceipt:
    market_id: str
    side: str
    size_usd: float
    fill_price: float
    att_order_id: str
    signed_by: str              # signing key fingerprint
    timestamp: str
    audit_log_path: str         # path to the full decision chain
```

## Failure modes & mitigations

| Failure | Mitigation |
|---|---|
| LLM hallucinates a market that doesn't exist | `pythia-delphi-adapter` validates every `market_id` against the live Delphi feed before passing downstream |
| All analysts agree but are all wrong | `pythia-forge` backtest harness tracks per-analyst Brier score; underperformers get auto-demoted |
| Single bad trade blows the bankroll | `pythia-risk` enforces quarter-Kelly + hard cap on per-market stake + circuit breaker on drawdown |
| ATT API goes down mid-trade | Executor catches 5xx, writes a `TradeReceipt` with `status=PENDING`, retries with idempotency key |
| Signing key leaks | All signing keys are env-only, never persisted to disk; `pythia-observability` logs the *fingerprint* not the key |
| Judge can't understand a trade | Replay UI reconstructs the full decision chain (market → estimates → consensus → risk → receipt) on one screen |

## Why ensembles beat single models on Delphi

Delphi's defining feature vs. Polymarket is **subjective and niche markets** ("best album of 2026", "most influential paper", "which meme coin flips first"). These markets:

- Have no single authoritative datasource
- Require multi-domain knowledge (cultural, technical, social)
- Are robust to one model's blind spot but vulnerable to consensus blind spots

A single LLM, no matter how strong, will have systematic biases on niche topics (training data coverage). An ensemble of *specialist* analysts — each prompted with domain context and required to cite evidence — averages out individual biases. The agreement gate filters out cases where specialists disagree (the highest-uncertainty, lowest-edge trades).

This is the same principle that won the Netflix Prize: **ensemble + diversity > single best model**.
