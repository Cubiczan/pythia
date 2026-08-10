# Pythia

### A hardened multi-agent trading mesh for Gensyn's Delphi information markets

> *Pythia was the oracle at Delphi. In this system, no single model issues a prophecy — a mesh of specialized analyst agents must reach consensus before a trade is placed, and every step is human-auditable.*

**Competition:** [Delphi: Agent Arena Competition](https://dorahacks.io/hackathon/delphi-agent-competition) (Gensyn × DoraHacks) — Build an agent, point it at real Delphi information markets, let it trade for two weeks. Top 3 P&Ls share $10,000.

**Org:** [icohangar-ops / Impactquadrant](https://github.com/icohangar-ops) — *"Developer and enterprise infrastructure for building hardened, human-auditable multi-agent decision workflows."*

---

## Why Pythia

Most competitors will ship a single-LLM-with-a-prompt bot and YOLO trades. That's a losing strategy on Delphi specifically, because **Delphi prices nuance and subjective scenarios** — domains where ensemble reasoning + calibrated uncertainty beat raw news-speed. The `icohangar-ops` stack already solves the *hard* part: hardened, auditable, consensus-driven multi-agent decision workflows. Pythia is that stack with a thin Delphi adapter on top.

**The wedge:**
1. **Calibrated ensembles beat single-model punters** on subjective/niche markets — Delphi's differentiator vs. Polymarket.
2. **Agreement-gated trades** — only place a bet when the analyst mesh agrees beyond a threshold. Avoids the LLM-hallucination-blowup failure mode.
3. **Risk gates (`meshcfo`)** prevent ruin during the 2-week window — staying solvent *is* alpha in a P&L contest.
4. **Replayable audit trail** — judges can scrub through any trade and see the consensus path.
5. **Market-creation side strategy** — Delphi is permissionless; analysts can mint high-edge subjective markets the mesh is uniquely positioned to price.

---

## Repository layout

Pythia is a **single flat monorepo**. Eight Python packages live under `packages/`, each independently pip-installable but designed to work together.

| Package | Import name | Role | Wraps |
|---|---|---|---|
| [`packages/delphi-adapter`](./packages/delphi-adapter) | `pythia_delphi_adapter` | ATT client, market metadata, settlement listener, config schema | **NEW** |
| [`packages/analyst-mesh`](./packages/analyst-mesh) | `pythia_analyst_mesh` | Base analyst agent + politics/crypto/sports/niche specialists + ensemble config | NEW |
| [`packages/consensus`](./packages/consensus) | `pythia_consensus` | Estimate fusion + agreement gate | `consensus-hardening-protocol` |
| [`packages/risk`](./packages/risk) | `pythia_risk` | Kelly sizing, exposure limits, drawdown brakes | `meshcfo` |
| [`packages/executor`](./packages/executor) | `pythia_executor` | Trade orchestration CLI → ATT | `metabocommand` |
| [`packages/observability`](./packages/observability) | `pythia_observability` | Audit trail, replay UI, P&L milestones | `agent-observability` + `achievements` |
| [`packages/forge`](./packages/forge) | `pythia_forge` | Backtest harness for resolved Delphi markets | `forge` |
| [`packages/strata`](./packages/strata) | `pythia_strata` | Stratified ingestion: Delphi feed, news, on-chain, social | `strata` |

> The wrapper packages (`consensus`, `risk`, `executor`, `observability`, `forge`, `strata`) are *thin* — they pin a version of the upstream `icohangar-ops` repo and add a Delphi-specific configuration layer. The upstream code stays the source of truth.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
   Delphi markets ─▶│  packages/strata  (data layers)     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  packages/analyst-mesh              │
                    │  • Politics analyst                 │
                    │  • Macro/crypto analyst             │
                    │  • Sports analyst                   │
                    │  • Niche/subjective analyst         │
                    │  Each emits: prob + rationale + CI  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  packages/consensus                 │
                    │  Fuses N estimates → consensus prob  │
                    │  + agreement score (gates trade)     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  packages/risk  (meshcfo)           │
                    │  Kelly sizing, drawdown cap,         │
                    │  exposure limits, market-type rules  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  packages/executor  (metabocommand)  │
                    │  → Gensyn ATT / delphi-skills        │
                    │  Signs + submits trades              │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  packages/observability              │
                    │  Every decision signed, replayable. │
                    │  P&L milestones unlock achievements. │
                    └─────────────────────────────────────┘

      packages/forge  = strategy build / backtest / deploy pipeline (sideways)
```

Full design doc: [`docs/pythia-design.pdf`](./docs/pythia-design.pdf).

---

## Quickstart

### Prerequisites

- Python ≥ 3.11
- Node ≥ 20 (for the Gensyn ATT / `gensyn-delphi-skills` JS bindings)
- A Delphi account + API key (set as `DELPHI_API_KEY`)
- An LLM provider key for the analyst mesh (OpenAI / Anthropic / etc — set as `LLM_API_KEY`)

### Install (monorepo, dev mode)

```bash
git clone https://github.com/icohangar-ops/pythia.git
cd pythia
make install          # pip install -e each package in dependency order
make check            # verify env, ATT connectivity, LLM key
```

### Run a paper trade

```bash
pythia executor delphi paper-trade \
  --market <market-id> \
  --analysts politics,crypto,niche \
  --consensus-threshold 0.65 \
  --max-stake-usd 50
```

### Run live (small size)

```bash
pythia executor delphi run \
  --config configs/live-mvp.toml \
  --risk-max-drawdown-pct 5 \
  --log-level info
```

### Backtest on resolved markets

```bash
pythia forge backtest \
  --strategy configs/strategies/ensemble-v1.toml \
  --markets resolved-2025-Q4.json \
  --starting-capital 1000
```

---

## Configuration

All runtime config lives in `configs/`. The canonical MVP config is [`configs/live-mvp.toml`](./configs/live-mvp.toml):

```toml
[delphi]
api_key_env = "DELPHI_API_KEY"
endpoint = "https://api.delphi.gensyn.ai"
poll_interval_sec = 60

[mesh]
analysts = ["politics", "crypto", "sports", "niche"]
llm_provider = "openai"
llm_model = "gpt-4o-mini"
llm_api_key_env = "LLM_API_KEY"

[consensus]
method = "logit-mean"        # alternatives: "median", "trimmed-mean"
agreement_threshold = 0.65   # gating: skip trade if < threshold
min_analysts = 2             # need at least this many to even consider

[risk]
sizing = "kelly-fractional"
kelly_fraction = 0.25        # quarter-Kelly for safety
max_stake_per_market_usd = 50
max_total_exposure_usd = 500
max_drawdown_pct = 5
cool_down_min_after_loss = 30

[executor]
mode = "paper"               # "paper" | "live"
signing_key_env = "DELPHI_SIGNING_KEY"

[observability]
audit_log_path = "./logs/audit.jsonl"
replay_ui_port = 8088
achievements_enabled = true
```

---

## 2-week shipping plan

We are at day ~11 of the 24-day trading window (Jul 31 → Aug 23, 2026). Plan below assumes ~13 days remaining.

| Days | Milestone |
|---|---|
| 1–2 | Wire `packages/delphi-adapter` to ATT + `gensyn-delphi-skills`. Single market read + paper trade. |
| 3–4 | Stand up 3-analyst mesh (politics + crypto + niche) on `packages/strata` → `packages/consensus`. |
| 5–6 | Plug `packages/risk` (Kelly-capped sizing) + `packages/executor`. Go live with small size. |
| 7–10 | Iterate: tune agreement threshold, add sports analyst, run `packages/forge` backtests on resolved markets. |
| 11–12 | Polish `packages/observability` replay UI + `achievements` P&L milestones for the demo. |
| 13 | Record demo video, write README, submit. |

---

## License

MIT for the new code (`packages/delphi-adapter`, `packages/analyst-mesh`). Wrapper packages inherit the license of their upstream `icohangar-ops` repo.

---

## Status

| Component | Status |
|---|---|
| `packages/delphi-adapter` | Scaffold + ATT client stub |
| `packages/analyst-mesh` | Scaffold + 4 analyst stubs |
| `packages/consensus` | Wrapper scaffold |
| `packages/risk` | Wrapper scaffold |
| `packages/executor` | Wrapper scaffold + CLI |
| `packages/observability` | Wrapper scaffold |
| `packages/forge` | Wrapper scaffold + backtest CLI |
| `packages/strata` | Wrapper scaffold |
| Design doc PDF | [`docs/pythia-design.pdf`](./docs/pythia-design.pdf) |

This is a hackathon scaffold, not production code. Wire the upstream `icohangar-ops` repos as submodules under `packages/*/vendor/` before going live.
