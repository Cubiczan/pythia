# pythia-analyst-mesh

### Specialist LLM analyst agents for the Pythia Delphi trading mesh

> *No single model issues a prophecy. The mesh is the oracle.*

`pythia-analyst-mesh` is the agent layer of [Pythia](https://github.com/icohangar-ops/pythia) — a hardened, auditable, multi-agent trading mesh for Gensyn's Delphi information markets.

Each **analyst** is an LLM-powered agent that specializes in one Delphi market category. Given a `MarketContext` (the question + metadata + current order-book price), an analyst emits an `Estimate`: a probability `P(YES)`, a self-reported confidence, a short rationale, and a list of evidence URLs. Estimates from N analysts are then fused by `pythia-consensus` into a single calibrated consensus probability and an agreement score that gates whether a trade is placed at all.

---

## Why a *mesh* of specialists instead of one big prompt

Delphi is not Polymarket. Its wedge is **subjective and niche markets** — "best album of 2026", "which paper will be most cited", "which meme coin flips first". These markets:

- have no single authoritative datasource,
- require multi-domain knowledge (cultural, technical, social),
- are robust to a single model's blind spot but vulnerable to consensus blind spots.

A single LLM has systematic biases on niche topics (training-data coverage, recency, alignment-trained hedging). An ensemble of *specialist* analysts — each prompted with domain context and required to cite evidence — averages out individual biases. The agreement gate then filters out the highest-uncertainty, lowest-edge trades: when specialists disagree, we **skip**. This is the Netflix-Prize insight: ensemble + diversity > single best model.

---

## The four built-in specialists

| Analyst | `analyst_id` | Specialty | Example markets |
|---|---|---|---|
| `PoliticsAnalyst` | `politics` | Elections, legislation, geopolitical events | "Will the incumbent win the 2026 midterms?" |
| `CryptoAnalyst` | `crypto` | Token prices, on-chain metrics, DeFi events, protocol upgrades | "Will ETH/USD exceed $5,000 by Dec 31?" |
| `SportsAnalyst` | `sports` | Match outcomes, player stats, injuries, schedule factors | "Will Lakers make the playoffs?" |
| `NicheAnalyst` | `niche` | Subjective & cultural questions: awards, viral events, community outcomes | "Will 'Album X' win Best Alternative Grammy?" |

The `NicheAnalyst` is the most uniquely valuable on Delphi vs. Polymarket — that's where the ensemble approach shines.

Every analyst is required to:

1. Output a probability `P(YES)` in `[0.0, 1.0]`.
2. Output a `confidence` in `[0.0, 1.0]` — explicitly a calibration signal, not a probability. **Analysts are prompted to default to `confidence < 0.6` when uncertain.**
3. Output a 1–3 sentence rationale.
4. Output a list of evidence URLs (may be empty).

---

## Install

```bash
pip install -e .
# dev extras
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

---

## Quickstart

### Library: run a 4-analyst mesh against a market

```python
import asyncio
from pythia_analyst_mesh import (
    AnalystRegistry,
    CryptoAnalyst,
    LLMConfig,
    MarketContext,
    run_mesh,
)

# 1. Configure your LLM provider (swappable: openai | anthropic | gensyn | ollama)
llm = LLMConfig(
    provider="openai",
    model="gpt-4o-mini",
    api_key="${LLM_API_KEY}",     # or read os.environ
    temperature=0.2,
    max_tokens=800,
)

# 2. Build the mesh from names
registry = AnalystRegistry()
mesh = registry.build_mesh(["politics", "crypto", "sports", "niche"], llm)

# 3. Construct a MarketContext (in practice, fetched via pythia-delphi-adapter)
market = MarketContext(
    market_id="delphi-mkt-001",
    question="Will ETH/USD close above $5,000 on or before Dec 31, 2026?",
    category="crypto",
    metadata={"news_context": "ETF inflows accelerating; spot up 18% MoM."},
    current_yes_price=0.42,
    current_no_price=0.58,
    volume_usd=124_000.0,
    closes_at="2026-12-31T23:59:59Z",
)

# 4. Run all analysts concurrently (with per-analyst timeout)
estimates = asyncio.run(run_mesh(market, mesh, timeout_sec=30.0))

for e in estimates:
    print(f"{e.analyst_id:>8}: P(YES)={e.probability:.2f}  conf={e.confidence:.2f}")
    print(f"          {e.rationale}")
```

### CLI: estimate a market from the terminal

```bash
# List registered analysts
pythia-analyst list

# Run a subset of analysts against a live Delphi market
pythia-analyst estimate delphi-mkt-001 --analysts politics,crypto,niche
```

The CLI fetches market metadata via `pythia_delphi_adapter.DelphiClient` (the `pythia-delphi-adapter` sibling repo), runs the mesh, and prints estimates as JSON to stdout.

---

## LLM providers

The mesh is provider-agnostic. Each analyst delegates its LLM call to `BaseAnalyst._call_llm`, which dispatches on `LLMConfig.provider`:

| `provider` | SDK | Notes |
|---|---|---|
| `"openai"` | `openai>=1.0` | Tested against `gpt-4o-mini`. |
| `"anthropic"` | `anthropic>=0.20` | Tested against `claude-3-5-sonnet`. |
| `"gensyn"` | `httpx` (raw REST) | Gensyn's own inference endpoint — `# VERIFY:` exact REST shape pending Gensyn docs. |
| `"ollama"` | `httpx` → `http://localhost:11434/api/chat` | Local, no API key. Good for offline dev. |

All calls are wrapped with `tenacity` for retries on transient errors (5xx, rate limits, connection resets). See `BaseAnalyst._call_llm` for the exact retry policy.

> **Assumption:** the OpenAI / Anthropic SDKs are imported lazily so the mesh loads even if only one provider's SDK is installed. Gensyn and Ollama use plain `httpx`, no extra dependency.

---

## Pluggability: adding a new analyst

You can register a new specialist at runtime — **no edits to this repo required**. Two options:

### Option A — register a class dynamically

```python
from pythia_analyst_mesh import BaseAnalyst, AnalystRegistry, LLMConfig, MarketContext, Estimate

class MacroAnalyst(BaseAnalyst):
    analyst_id = "macro"
    specialty = "macroeconomics"

    SYSTEM_PROMPT = (
        "You are a macroeconomist specializing in interest rates, inflation, "
        "and central-bank policy. Estimate the probability that the market's "
        "YES outcome occurs. Be calibrated: prefer confidence < 0.6 when uncertain."
    )

    def _build_prompt(self, market: MarketContext) -> list[dict]:
        # ... build chat messages ...
        return [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {market.question}\n..."}]

# Register
registry = AnalystRegistry()
registry.register("macro", MacroAnalyst)

# Use
mesh = registry.build_mesh(["macro", "crypto"], llm_config)
```

### Option B — declare it in TOML config

In your `live-mvp.toml` (top-level Pythia repo):

```toml
[mesh]
analysts = ["politics", "crypto", "macro"]
# extra analysts discovered via entry-points or explicit import path
extra_analysts = ["my_pkg.macro:MacroAnalyst"]
```

(Entry-point discovery is on the roadmap — see `# TODO:` in `registry.py`.)

### Contract for a new analyst

Subclass `BaseAnalyst` and implement:

- `analyst_id: str` — short slug, used in logs / consensus weighting.
- `specialty: str` — human-readable category.
- `_build_prompt(market: MarketContext) -> list[ChatMessage]` — return the chat messages (system + user).
- `async estimate(market: MarketContext) -> Estimate` — usually: build prompt → call `_call_llm` → call `_parse_llm_response`.

The base class gives you, for free:

- `_call_llm(messages, config)` — provider-agnostic dispatch + tenacity retries.
- `_parse_llm_response(raw, market)` — robust JSON extraction with graceful fallbacks.
- A shared `_llm_config` instance.

---

## Robustness: how `_parse_llm_response` handles LLM junk

LLMs do not always return clean JSON. `_parse_llm_response` is defensive:

1. Strip Markdown code fences (` ```json ... ``` `).
2. Try `json.loads` on the whole string.
3. Try to extract the first `{...}` block via regex.
4. If still no JSON, scan for a leading probability number (e.g. `"0.42"` or `"42%"`) and build a low-confidence `Estimate` from it.
5. Always populate `analyst_id`, `market_id`, `timestamp` — never raise, never return `None`.

This means a single analyst returning garbage degrades gracefully to a low-confidence estimate rather than crashing the whole mesh.

---

## Concurrency model

`run_mesh(market, analysts, timeout_sec)` runs every analyst concurrently via `asyncio.gather`. Each analyst is wrapped in `asyncio.wait_for(timeout_sec)`. Analysts that:

- time out,
- raise an exception (LLM 5xx, network error, malformed response that even `_parse_llm_response` can't salvage — rare),

are **dropped from the result** with a logged warning. The returned list may be shorter than the input list. Downstream `pythia-consensus` checks `min_analysts` to decide whether to skip the trade entirely.

---

## Module layout

```
packages/analyst-mesh/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── src/
│   └── pythia_analyst_mesh/
│       ├── __init__.py        # public re-exports
│       ├── types.py           # Pydantic models: Estimate, MarketContext, LLMConfig
│       ├── base.py            # BaseAnalyst ABC + _call_llm + _parse_llm_response
│       ├── registry.py        # AnalystRegistry + auto-register 4 built-ins
│       ├── runner.py          # async run_mesh()
│       ├── cli.py             # pythia-analyst CLI
│       └── analysts/
│           ├── __init__.py
│           ├── politics.py    # PoliticsAnalyst
│           ├── crypto.py      # CryptoAnalyst
│           ├── sports.py      # SportsAnalyst
│           └── niche.py       # NicheAnalyst
└── tests/
    ├── __init__.py
    ├── test_registry.py
    ├── test_base.py
    └── test_analysts.py
```

---

## Testing

```bash
pytest -v
```

Tests use mocked LLM calls — no API keys required. They cover:

- `test_registry.py` — register / get / list_known / build_mesh returns instances.
- `test_base.py` — `_parse_llm_response` against valid JSON, fenced JSON, partial JSON, missing fields, plain-prose fallback.
- `test_analysts.py` — each analyst's `_build_prompt` produces the expected chat structure (system + user message, market question present, current price included).

---

## Relation to the rest of Pythia

```
            pythia-strata   (data layers)
                   │
                   ▼
        ┌─►  pythia-analyst-mesh   ◄─┐  (this repo)
        │      4 specialist LLMs     │
        │      each emits Estimate   │
        └────────────┬───────────────┘
                     ▼
              pythia-consensus   (fuse N → consensus prob + agreement gate)
                     ▼
              pythia-risk        (Kelly sizing, drawdown brake)
                     ▼
              pythia-executor    (sign + submit via ATT)
                     ▼
              pythia-observability  (signed audit trail, replay UI)
```

The mesh is consumed by `pythia-consensus`. It is fed `MarketContext` objects that come from `pythia-delphi-adapter` (raw Delphi) enriched by `pythia-strata` (news, on-chain, social).

---

## Status

Reference implementation. The 4 analysts are real (system prompts + provider dispatch + robust parsing), but:

- Gensyn provider REST shape is `# VERIFY:` — pending Gensyn's public API docs.
- Entry-point-based analyst discovery (TOML `extra_analysts`) is `# TODO:`.
- No per-analyst Brier-score tracking here — that lives in `pythia-forge` (backtest harness).

---

## License

MIT — see [`LICENSE`](./LICENSE). Copyright © 2026 icohangar-ops / Impactquadrant.
