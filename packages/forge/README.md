# pythia-forge

**Backtest harness + CI deploy pipeline for the Pythia mesh.**

`pythia-forge` wraps `icohangar-ops/forge` (the build / test / deploy pipeline
primitives from the hardened multi-agent decision-workflow stack) and adds three
Delphi-specific capabilities on top:

1. **Backtest runner** — replays resolved historical Delphi markets through the
   full mesh → consensus → risk pipeline, recording every trade the bot *would
   have* placed, then settles each one against the known outcome to compute
   realised P&L.
2. **Strategy evaluation report** — a markdown report + equity-curve chart
   summarising return, Sharpe, drawdown, win rate, per-analyst Brier scores,
   and per-category breakdown, with explicit weight-tuning recommendations.
3. **CI deploy hook** — a `pythia-forge deploy` subcommand that validates a
   live config, runs a smoke-test backtest against a small fixture, and
   emits a deploy plan for a runner.

Together these let you tune the four specialist analysts
(`politics` / `crypto` / `sports` / `niche`), the consensus fusion method +
threshold, and the Kelly fraction **offline, against ground truth**, before
ever risking live capital on Delphi.

---

## Why a backtest harness?

The Pythia mesh is an LLM-driven trading system. LLMs are stochastic and
calibration is the single most important driver of long-run P&L on binary
prediction markets. You cannot tune analyst weights, consensus thresholds, or
Kelly fractions by eyeballing live trades — the sample sizes are too small and
the feedback loop is too slow (Delphi markets take days to weeks to settle).

`pythia-forge` closes that loop: feed it a JSON file of resolved Delphi markets
(market question + category + opening YES price + final outcome + volume), and
it will:

- reconstruct the `MarketContext` the mesh would have seen at open,
- run the full mesh → consensus → risk pipeline for each market,
- "place" every APPROVE'd trade and "settle" it against the known outcome,
- aggregate the results into a `BacktestResult` with calibration metrics,
- emit a markdown report + equity-curve PNG.

The whole thing runs in seconds (with the built-in `MockLLM`) and costs zero
LLM credits — so you can grid-search thousands of parameter combinations
overnight.

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │              pythia-forge                    │
                │  (wraps icohangar-ops/forge pipeline prims) │
                └─────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 ┌──────────────┐         ┌──────────────────┐        ┌──────────────┐
 │  Backtester  │         │  Report (md+png) │        │  CLI deploy  │
 │  run()       │         │  generate_report │        │  smoke + plan│
 └──────┬───────┘         └──────────────────┘        └──────────────┘
        │
        │ per market:
        ▼
 ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
 │ analyst-mesh │───▶│  pythia-consensus│───▶│  pythia-risk │
 │  (4 LLMs or  │    │  fuse → decision │    │  evaluate →  │
 │   MockLLM)   │    │                  │    │  TradePlan   │
 └──────────────┘    └──────────────────┘    └──────┬───────┘
                                                    │
                                              ┌─────▼─────┐
                                              │  settle   │
                                              │  vs known │
                                              │  outcome  │
                                              └───────────┘
```

The backtester does **not** mutate the real risk engine's `BankrollState`
between markets by default — each market is evaluated against the *starting*
bankroll so that per-market P&L is independent and additive. Pass
`track_bankroll=True` in `BacktestConfig.markets_filter` (or set
`[backtest] track_bankroll = true` in the strategy TOML) to instead thread the
running bankroll through, which exercises the drawdown breaker and cool-down
gates realistically.

---

## Install

```bash
# from the pythia monorepo root
pip install -e ./packages/forge

# or standalone (pulls the four sibling pythia-* packages from PyPI once published)
pip install pythia-forge
```

Requires Python ≥ 3.11. The sibling packages (`pythia-delphi-adapter`,
`pythia-analyst-mesh`, `pythia-consensus`, `pythia-risk`) must be importable 
they are listed as hard dependencies.

---

## Quickstart

### 1. Prepare a resolved-markets JSON file

A JSON array of objects matching `HistoricalMarket` (see `types.py`):

```json
[
  {
    "market_id": "dphi_01J",
    "question": "Will Bitcoin close above $100k on 2025-12-31?",
    "category": "crypto",
    "opened_at": "2025-10-01T00:00:00Z",
    "closed_at": "2025-12-31T23:59:59Z",
    "settled_at": "2026-01-01T00:05:00Z",
    "yes_price_at_open": 0.42,
    "final_outcome": "NO",
    "volume_usd": 125000,
    "arbiter_model": "gpt-4o"
  }
]
```

A 10-market sample fixture ships at `tests/fixtures/resolved_markets_sample.json`.

### 2. Run a backtest

```bash
pythia-forge backtest \
    --strategy configs/strategies/ensemble-v1.toml \
    --markets resolved-2025-Q4.json \
    --starting-capital 1000
```

This will:

- load the strategy TOML (mesh + consensus + risk config),
- run the mesh against each market using the **MockLLM** by default (zero LLM
  cost, deterministic),
- run consensus → risk → settle for each market,
- write `reports/backtest-<timestamp>.md` and `reports/backtest-<timestamp>.png`.

Add `--use-real-llm` to call the actual LLM provider configured in the strategy
TOML (requires `LLM_API_KEY` in the environment). This is slow and costs money
— use sparingly, e.g. for a final validation run.

### 3. Tune weights

```bash
pythia-forge tune \
    --strategy configs/strategies/ensemble-v1.toml \
    --markets resolved-2025-Q4.json \
    --iterations 20
```

Grid-searches over consensus `agreement_threshold` ∈ {0.5, 0.6, 0.65, 0.7, 0.8}
and `kelly_fraction` ∈ {0.1, 0.25, 0.5, 1.0} (20 combinations), runs a backtest
for each, and prints a table sorted by Sharpe ratio. The best config is written
to `reports/best-strategy-<timestamp>.toml`.

### 4. Deploy (CI hook)

```bash
pythia-forge deploy --config configs/live-mvp.toml
```

Validates the live config (all required sections present, risk caps sane),
runs a 5-market smoke backtest against `tests/fixtures/resolved_markets_sample.json`
to confirm the mesh produces non-degenerate output, and emits a deploy plan
as JSON. In a real CI pipeline this would be the gate before `git push` to the
runner branch.

---

## Metrics produced

`BacktestResult` (see `types.py`) carries:

| Field                     | Type                    | Notes                                                   |
| ------------------------- | ----------------------- | ------------------------------------------------------- |
| `starting_capital_usd`    | `float`                 | From config.                                            |
| `ending_capital_usd`      | `float`                 | Starting + sum of settled P&L.                          |
| `total_return_pct`        | `float`                 | `(ending - starting) / starting * 100`.                 |
| `sharpe_ratio`            | `float`                 | Annualised, assuming 1 trade/day → `sqrt(252)` factor.  |
| `max_drawdown_pct`        | `float`                 | Peak-to-trough on the equity curve.                     |
| `total_trades`            | `int`                   | Count of APPROVE'd (and thus settled) trades.           |
| `win_rate`                | `float`                 | Fraction of settled trades that were profitable.        |
| `brier_scores`            | `dict[str, float]`      | Per-analyst mean Brier score (lower = better).          |
| `per_category_stats`      | `dict[str, dict]`       | Per-category: count, win_rate, return_pct, brier.       |
| `equity_curve`            | `list[tuple[dt, float]]`| One point at start + one per settled trade.             |

### Brier score

For each analyst `a` and each market where `a` produced an estimate with
probability `p_a`:

```
outcome = 1.0 if final_outcome == "YES" else 0.0
brier_a = mean over markets of (p_a - outcome)^2
```

Brier ∈ [0, 1]; 0 = perfect, 0.25 = uninformative (always predict 0.5),
1.0 = always wrong. This is the single most useful calibration metric for
tuning analyst weights — analysts with lower Brier should get higher
`[consensus.weights]` entries.

### Sharpe ratio

```
daily_returns = diff(equity_curve) / prev_equity
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

Assumes 1 trade/day for annualisation simplicity. A real Delphi bot trades
~2-5 markets/day at peak, so this is conservative. A Sharpe > 1.0 is good;
> 2.0 is suspicious (likely overfit — widen your markets filter or add more
out-of-sample data).

### Max drawdown

Standard peak-to-trough percentage on the equity curve. The risk engine's
`max_drawdown_pct` gate trips at 5% by default — if your backtest
`max_drawdown_pct` approaches that, your Kelly fraction is too aggressive.

---

## How to tune analyst weights

1. Run a backtest with equal weights (omit `[consensus.weights]` or set all
   to 1.0).
2. Open the report. Look at the **Per-analyst Brier scores** table (sorted
   best to worst).
3. Set `[consensus.weights]` in your strategy TOML inversely proportional to
   Brier — e.g. if `politics` has Brier 0.18 and `niche` has 0.24, give
   `politics` weight 1.3 and `niche` weight 0.7 (normalise so the mean is ~1.0
   to preserve the agreement-score scale).
4. Re-run the backtest. Check that `total_return_pct` and `sharpe_ratio`
   improved and `max_drawdown_pct` did not worsen.
5. Run `pythia-forge tune` to grid-search the consensus threshold and Kelly
   fraction jointly.
6. Once you have a config that looks good, run once with `--use-real-llm` to
   confirm the MockLLM heuristic was not flattering a specific analyst.

### Weight-tuning heuristics encoded in the report

The `generate_report` function emits explicit recommendations:

- If an analyst's Brier > 0.30, recommend **down-weighting** (weight < 0.5).
- If an analyst's Brier < 0.18, recommend **up-weighting** (weight > 1.2).
- If `win_rate < 0.45`, recommend raising `agreement_threshold` (the mesh is
  trading on disagreement).
- If `max_drawdown_pct > 3.0`, recommend halving `kelly_fraction`.
- If `sharpe_ratio < 0.5`, recommend switching consensus `method` from
  `logit-mean` to `trimmed-mean` (more robust to outlier analysts).

These are starting points, not laws — always sanity-check against
out-of-sample data.

---

## The MockLLM

`MockLLM` (`pythia_forge.mock_llm`) is a deterministic LLM replacement used by
default in backtests to avoid burning credits. It maps a market question to a
probability via simple keyword heuristics:

| Keyword in question      | Mock P(YES) |
| ------------------------ | ----------- |
| "Trump"                  | 0.55        |
| "Bitcoin" / "BTC" / "ETH"| 0.50        |
| "election"               | 0.50        |
| "Fed" / "rate"           | 0.48        |
| "Super Bowl" / "NBA"     | 0.52        |
| "AI" / "GPT"             | 0.60        |
| (default)                | 0.50        |

Each specialist analyst gets a small per-specialty nudge on top (e.g. the
`politics` analyst adds +0.03 to any question containing "Trump"; the `crypto`
analyst adds +0.02 to any question containing "Bitcoin"). This produces
**realistic disagreement** between analysts — enough to exercise the consensus
fusion and agreement-score gates, but not so much that every market is a
coin flip.

> ⚠️ **MockLLM is NOT for production.** It exists solely so backtests are fast,
> free, and deterministic. Always validate final configs with `--use-real-llm`
> against a held-out market set before deploying.

---

## Strategy TOML format

See `configs/strategies/ensemble-v1.toml` in the parent monorepo for a
complete example. The relevant sections:

```toml
[strategy]
name = "ensemble-v1"
description = "4-analyst mesh, logit-mean consensus, quarter-Kelly."
version = "1.0.0"

[strategy.mesh]
analysts = ["politics", "crypto", "sports", "niche"]
llm_model = "gpt-4o-mini"
llm_temperature = 0.2

[strategy.consensus]
method = "logit-mean"            # "logit-mean" | "median" | "trimmed-mean"
agreement_threshold = 0.65       # below this → gate = "skip"
min_analysts = 2                 # below this → gate = "wait"

[strategy.consensus.weights]     # optional; omit for equal weights
politics = 1.0
crypto = 1.2
sports = 0.8
niche = 1.0

[strategy.risk]
sizing = "kelly-fractional"      # "kelly-fractional" | "fixed"
kelly_fraction = 0.25            # 0.25 = quarter-Kelly
max_stake_per_market_usd = 50
max_total_exposure_usd = 500
max_drawdown_pct = 5

[backtest]
starting_capital_usd = 1000
markets_filter = { categories = ["politics", "crypto", "sports", "subjective"], min_volume_usd = 1000 }
min_market_lifetime_sec = 3600
track_bankroll = false           # true → thread bankroll through markets (exercises gates)
```

---

## Programmatic API

```python
import asyncio
from pathlib import Path
from pythia_forge import Backtester, BacktestConfig
from pythia_forge.report import generate_report

config = BacktestConfig(
    strategy_path=Path("configs/strategies/ensemble-v1.toml"),
    markets_path=Path("resolved-2025-Q4.json"),
    starting_capital_usd=1000.0,
    markets_filter={"categories": ["politics", "crypto", "sports", "subjective"]},
)

result = asyncio.run(Backtester(config).run())
print(f"Return: {result.total_return_pct:.2f}%  Sharpe: {result.sharpe_ratio:.2f}")
print(f"Brier scores: {result.brier_scores}")

generate_report(result, output_path=Path("reports/my-backtest.md"))
```

---

## Module layout

```
packages/forge/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── VENDOR_COMMIT.txt
├── src/pythia_forge/
│   ├── __init__.py        # public exports
│   ├── types.py           # BacktestConfig, BacktestResult, HistoricalMarket
│   ├── backtester.py      # Backtester class — the core harness
│   ├── mock_llm.py        # MockLLM — deterministic LLM replacement
│   ├── report.py          # markdown report + equity-curve PNG
│   └── cli.py             # pythia-forge backtest | tune | deploy
└── tests/
    ├── __init__.py
    ├── test_backtester.py
    ├── test_report.py
    └── fixtures/
        └── resolved_markets_sample.json
```

---

## Relationship to `icohangar-ops/forge`

`forge` is the upstream pipeline-primitives repo (build / test / deploy
scaffolding for the multi-agent decision-workflow stack). `pythia-forge` wraps
it and adds the Delphi-specific backtest + report + deploy-hook layer. The
upstream commit is pinned in `VENDOR_COMMIT.txt`; until vendored, this package
is self-contained (the `Backtester`, `report`, and `cli` modules have no
hard dependency on forge's internals — only on the four sibling `pythia-*`
packages).

When forge is vendored:

- `cli.py deploy` will delegate the actual runner-deploy step to forge's
  deploy primitive (currently a stub that prints a plan).
- The backtest runner may use forge's `Pipeline` + `Stage` abstractions to
  structure the per-market loop (currently a plain `async for`).

Both call sites are marked with `# VERIFY:` comments.

---

## Testing

```bash
pytest -v
```

The test suite uses the `MockLLM` and the 10-market sample fixture — no network
access, no LLM credits, ~3 seconds end-to-end. It covers:

- `test_backtester.py` — backtest produces a well-typed `BacktestResult` with
  correct `total_trades`, non-empty `brier_scores` for every analyst, and a
  non-empty `equity_curve`.
- `test_report.py` — `generate_report` writes a markdown file and an
  equity-curve PNG; the markdown contains the executive summary, Brier table,
  and category table sections.

---

## License

MIT — see `LICENSE`. Copyright icohangar-ops / Impactquadrant 2026.
