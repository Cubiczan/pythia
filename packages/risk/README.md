# pythia-risk

> Delphi-specific risk gating layer wrapping [`icohangar-ops/meshcfo`](https://github.com/icohangar-ops/meshcfo). Adds Kelly criterion sizing for binary-outcome markets, per-market-type exposure caps, and a drawdown circuit breaker.

Part of the [Pythia](https://github.com/icohangar-ops/pythia) hardened multi-agent trading mesh for Gensyn's Delphi information markets.

---

## What this repo does

`pythia-risk` sits between `pythia-consensus` (which fuses analyst estimates into a `ConsensusDecision`) and `pythia-executor` (which signs and submits the trade via the Gensyn ATT). Its job is **to decide whether a trade should be placed, and if so, how big**.

It answers three questions, in order:

1. **Are we allowed to trade this market at all?** Market-type rules (politics, crypto, sports, niche, subjective) gate which categories the bot is permitted to touch.
2. **Are we in shape to trade?** Drawdown circuit breaker + post-loss cool-down prevent tilt-trading.
3. **How much should we stake?** Fractional-Kelly sizing for binary-outcome markets, capped by per-market and total-exposure limits.

If all gates pass, `RiskEngine.evaluate(...)` returns a `TradePlan(decision="APPROVE", ...)` with a dollar size and side (`YES` or `NO`). Otherwise it returns `TradePlan(decision="REJECT", ...)` with a `risk_flags` entry explaining why.

The upstream `meshcfo` package provides the capital-management primitives (ledger, position tracking, audit hooks). This wrapper adds the **Delphi-specific** layer: binary-market Kelly math, market-type tagging, and a config schema tuned for the short-horizon P&L regime where *staying solvent is alpha*.

---

## Wrapping strategy

This is a **thin wrapper** repo. The upstream `meshcfo` code stays the source of truth and is consumed in one of two ways (see [`SUBMODULES.md`](../SUBMODULES.md) in the top-level pythia repo):

1. **Vendored (default):** copy the upstream code into `vendor/meshcfo/` and pin the commit SHA in [`VENDOR_COMMIT.txt`](./VENDOR_COMMIT.txt).
2. **Nested submodule:** `git submodule add git@github.com:icohangar-ops/meshcfo.git vendor/meshcfo`.

Anything that depends on a specific `meshcfo` internal API is marked with a `# VERIFY:` comment. Those calls need to be re-checked against the pinned upstream commit before going live.

---

## Kelly criterion for Delphi binary markets

Delphi markets (like Polymarket) are **binary outcome** markets: a `YES` share pays out `1.0` if the event happens, `0.0` otherwise. You can buy `YES` at the current `market_price` (a probability quote between 0 and 1).

### The math

Let:

- `p` = consensus probability that YES wins (from `pythia-consensus`)
- `q = 1 - p` = probability NO wins
- `m` = `market_price` (price of 1 YES share)
- `b` = net odds received per dollar staked = `(1 - m) / m`

  *Why:* if you spend `$1` on YES shares at price `m`, you get `1/m` shares. If YES wins, each share pays `$1`, so you receive `$1/m`. Net profit = `$(1/m - 1) = $((1-m)/m)`. If NO wins, you lose the `$1`.

The **full Kelly fraction** of bankroll to stake is:

```
f* = (b * p - q) / b
   = p - q / b
   = p - (1 - p) * m / (1 - m)
```

Equivalently:

```
f* = (p * (1 - m) - (1 - p) * m) / (1 - m)
   = (p - m) / (1 - m)
```

This is the elegant form: **the full-Kelly stake as a fraction of bankroll equals the edge `(p - m)` divided by the gross payoff odds `(1 - m)`**.

### Why fractional Kelly

Full Kelly maximizes long-run log-wealth growth *if* your probability estimate `p` is exactly correct. In practice `p` is an LLM-fused consensus with non-trivial calibration error. Full Kelly over-bets when the model is overconfident, and the variance is brutal. The standard fix is **fractional Kelly**: stake `fraction * f*` of bankroll, typically `fraction = 0.25` (quarter-Kelly).

- Quarter-Kelly gives up ~25% of expected log-growth but cuts variance by ~4x.
- It is robust to ~25% miscalibration in `p`.
- It is the default in [`configs/live-mvp.toml`](../configs/live-mvp.toml).

### Edge cases handled

- **No edge (`p == m`):** `f* = 0`, stake is 0. The engine returns `REJECT` with flag `no_edge`.
- **Negative edge (`p < m`):** `f*` is negative. We **clamp to 0** (no leverage, no shorting via negative Kelly). If `p < m` enough to flip the side to `NO`, the engine re-derives Kelly from the NO side: `b_no = m / (1 - m)`, `f*_no = ((1 - p) - m) / m`... actually simpler: from the NO buyer's perspective, the market price for NO is `1 - m`, so `f*_NO = ((1 - p) - (1 - m)) / m = (m - p) / m`.
- **Very high edge (`p → 1`, `m → 0`):** `f* → 1`, full bankroll. Capped by `max_stake_per_market_usd` and `max_total_exposure_usd` to prevent ruin.
- **`m → 0` or `m → 1`:** denominator blows up. We guard with `max(1e-6, ...)` on the payoff odds.

---

## Risk gates

`RiskEngine.evaluate(decision, market, current_bankroll)` runs the following pipeline. The first `REJECT` short-circuits the rest.

| # | Gate | Condition | Flag on reject |
|---|---|---|---|
| 1 | Market-type allowed | `not market_type_rules[category].allowed` | `market_type_not_allowed` |
| 2 | Drawdown breaker | `bankroll.drawdown_pct >= config.max_drawdown_pct` | `drawdown_breaker` |
| 3 | Post-loss cool-down | `now - last_loss_at < cool_down_min_after_loss` | `cool_down_active` |
| 4 | Exposure cap | `open_positions + proposed_stake > max_total_exposure_usd` | `exposure_cap` (size reduced, not rejected, if still > 0) |
| 5 | No-edge | `\|consensus_prob - market.yes_price\| < 0.02` | `no_edge` |
| 6 | Per-market cap | `size > market_type_rules[category].max_stake_usd` or `> config.max_stake_per_market_usd` | size reduced (silent) |

If all gates pass (or gate 4 reduces-but-doesn't-reject), the engine returns `APPROVE` with the computed `size_usd`, `side`, and a `rationale` summarizing the Kelly math.

---

## Install

```bash
cd packages/risk
pip install -e .
```

Requires Python ≥ 3.11. Dependencies: `pydantic`, `numpy`, `tomli` (for < 3.11).

## Usage

### As a library

```python
from datetime import datetime, timezone
from pythia_risk import RiskEngine, RiskConfig, MarketTypeRules, BankrollState
from pythia_risk.types import Market  # if not pulling from adapter

config = RiskConfig(
    sizing="kelly-fractional",
    kelly_fraction=0.25,
    max_stake_per_market_usd=50,
    max_total_exposure_usd=500,
    max_drawdown_pct=5.0,
    cool_down_min_after_loss=30,
    market_type_rules={
        "politics": MarketTypeRules(max_stake_usd=50, allowed=True),
        "crypto":   MarketTypeRules(max_stake_usd=40, allowed=True),
        "sports":   MarketTypeRules(max_stake_usd=20, allowed=True),
        "niche":    MarketTypeRules(max_stake_usd=30, allowed=True),
    },
)

engine = RiskEngine(config)

plan = engine.evaluate(
    decision=consensus_decision,       # from pythia_consensus
    market=market,                      # Market(market_id, yes_price, category, ...)
    current_bankroll=BankrollState(...),
)
print(plan.decision, plan.side, plan.size_usd, plan.risk_flags)
```

### As a CLI

```bash
# Quick Kelly stake sanity check
pythia-risk size --consensus 0.7 --price 0.5 --bankroll 1000

# Full evaluate against JSON inputs (decision.json, market.json)
pythia-risk evaluate decision.json market.json --bankroll 1000 --exposure 200
```

`pythia-risk size` prints just the dollar stake. `pythia-risk evaluate` prints the full `TradePlan` as JSON.

---

## Module layout

```
packages/risk/
├── README.md
├── LICENSE                          MIT
├── .gitignore
├── VENDOR_COMMIT.txt                pin upstream meshcfo commit here
├── pyproject.toml
├── src/pythia_risk/
│   ├── __init__.py                  public exports
│   ├── types.py                     RiskConfig, TradePlan, BankrollState, MarketTypeRules, Market, TradeReceipt
│   ├── sizing.py                    Kelly math (kelly_fraction, size_trade_kelly, size_trade_fixed)
│   ├── engine.py                    RiskEngine with the 6-gate pipeline
│   └── cli.py                       `pythia-risk` CLI entry point
└── tests/
    ├── __init__.py
    ├── test_sizing.py
    └── test_engine.py
```

---

## Tests

```bash
cd packages/risk && pip install -e . && pytest -v
```

Covers: Kelly positive/negative/zero edge, max-stake cap, quarter-Kelly reduction, engine approve / reject on each gate, exposure cap reducing size.

---

## Design notes

- **Bankroll state is passed in, not stored internally.** `RiskEngine.evaluate` is pure with respect to bankroll — it does not mutate. `update_state` / `record_loss` / `record_win` mutate `engine.state` only when the executor confirms a fill or settlement. This makes the engine testable and replayable.
- **No shorting.** Kelly is clamped to `[0, 1]`. If consensus is bearish on YES (i.e. `consensus_prob < market.yes_price`), we flip to buying NO rather than shorting YES.
- **No leverage.** Same clamp prevents staking > 100% of bankroll even when edge appears huge.
- **Cool-down is wall-clock.** `last_loss_at` is set by `record_loss` and cleared by `cool_down_min_after_loss` minutes of no losses. It is *not* per-market — one bad trade should pause all trading.
- **Drawdown is computed off peak bankroll**, not off starting capital. A new high-water mark resets the breaker's denominator.

---

## License

MIT, copyright icohangar-ops / Impactquadrant 2026. Wrapper code is MIT; vendored `meshcfo` retains its upstream license — see `vendor/meshcfo/LICENSE` once vendored.
