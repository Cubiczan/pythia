# pythia-observability

> Replay UI + achievements layer for the Pythia multi-agent trading mesh.
> Wraps [`icohangar-ops/agent-observability`](https://github.com/icohangar-ops/agent-observability) (signed audit trail) and [`icohangar-ops/achievements`](https://github.com/icohangar-ops/achievements) (milestone tracking) and adds a polished dark-themed dashboard for hackathon judges.

[![License: MIT](https://img.shields.io/badge/License-MIT-d4a84b.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-131a2c.svg)](https://docs.astral.sh/ruff/)

---

## What this is

Pythia is a multi-agent trading mesh for the [Delphi Agent Arena Competition](https://dorahacks.io/) (Gensyn's information-market arena). Every decision the mesh makes — every analyst estimate, every consensus round, every risk gating, every trade execution — is written to an append-only JSONL **audit log** signed with Ed25519.

`pythia-observability` is the **read-side** of that audit log. It does three jobs:

1. **Parse** the audit JSONL into typed `AuditEntry` records and compute derived analytics (cumulative P&L curve, per-analyst Brier scores, win rate, current drawdown, skipped-reason breakdown).
2. **Replay** the audit log through a FastAPI server that serves a polished dark-themed dashboard at `http://127.0.0.1:8088/`. Judges can walk the full decision chain (market → estimates → consensus → risk plan → receipt → settlement) for any trade by clicking a row.
3. **Evaluate** the upstream `achievements.toml` config against the audit log — checking which P&L milestones ("Plus Fifty", "Calibrated Oracle", "Hot Hand", etc.) the mesh has unlocked.

It is intentionally **read-only** — the write side (signing decisions into the log) lives in [`pythia-consensus`](https://github.com/icohangar-ops/pythia-consensus) (audit signer) and `pythia-executor` (receipt writer). This separation lets the replay UI be served publicly (e.g. to judges) without exposing any mutation surface.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │  Audit JSONL (append-only)       │
                    │  logs/audit.jsonl                │
                    │  one entry per market cycle,     │
                    │  Ed25519-signed                  │
                    └───────────────┬──────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  AuditLogReader   │  iter_entries / read_all
                          │  (this repo)      │  compute_pnl_series
                          │                   │  compute_brier_scores
                          │                   │  compute_stats
                          └─────┬────────┬────┘
                                │        │
                ┌───────────────▼─┐    ┌──▼──────────────────┐
                │  ReplayServer   │    │ AchievementsEvaluator│
                │  (FastAPI)      │    │ load achievements.toml│
                │  / (HTML)       │    │ evaluate(log)         │
                │  /api/stats     │    └───────────┬───────────┘
                │  /api/trades    │                │
                │  /api/pnl-series│                │
                │  /api/achievements│ ◄────────────┘
                └────────┬────────┘
                         │
                  ┌──────▼──────┐
                  │ dashboard   │  vanilla JS + fetch
                  │ dark navy + │  judges click rows to
                  │ gold accent  │  expand full decision chain
                  └─────────────┘
```

### Upstream wrapping strategy

| Upstream repo | Role | Wrapper boundary |
|---|---|---|
| `icohangar-ops/agent-observability` | Defines the JSONL audit-log schema + signing format. | `AuditEntry` (pydantic model) is the contract we assume. If the upstream emits one-line-per-stage instead of one-line-per-cycle, only `audit_reader.iter_entries()` needs an adapter. Marked with `# VERIFY:` comments. |
| `icohangar-ops/achievements` | Defines the `[[achievement]]` TOML schema + the badge-emission API. | We reuse the TOML schema verbatim and add the *evaluation* layer (`AchievementsEvaluator` + per-condition functions). Badge emission is delegated to the upstream via the executor. |

Both upstreams are pinned in [`VENDOR_COMMIT.txt`](VENDOR_COMMIT.txt) once vendored. Until then, this repo runs standalone against any JSONL file matching the `AuditEntry` contract.

---

## Install

```bash
# From the repo root:
pip install -e .

# Or with dev deps (tests + httpx for TestClient):
pip install -e ".[dev]"
```

Requires Python 3.11+. The dashboard server pulls in `fastapi`, `uvicorn`, and `jinja2`; the rest is pure stdlib + pydantic.

---

## Quickstart

### 1. Inspect an audit log

```bash
pythia-replay stats --log ./logs/audit.jsonl
```

Prints aggregate stats as JSON:

```json
{
  "total_trades": 7,
  "executed_trades": 5,
  "skipped_trades": 2,
  "settled_trades": 4,
  "winning_trades": 3,
  "losing_trades": 1,
  "win_rate": 0.75,
  "avg_stake_usd": 42.5,
  "total_realized_pnl_usd": 78.20,
  "current_bankroll_usd": 1078.20,
  "peak_bankroll_usd": 1098.50,
  "current_drawdown_pct": 1.85,
  "per_analyst_brier": { "politics": 0.182, "crypto": 0.234, "niche": 0.141 },
  "skipped_reasons": { "agreement_below_threshold": 1, "drawdown_breaker": 1 },
  "signature_stub_count": 7
}
```

### 2. Launch the dashboard

```bash
pythia-replay serve \
    --log ./logs/audit.jsonl \
    --achievements-config configs/achievements.toml \
    --port 8088
```

Open <http://127.0.0.1:8088/> in your browser. You'll see:

- **Hero row** — realized P&L (big gold number), win rate, current bankroll, current drawdown.
- **P&L curve** — canvas chart of cumulative realized P&L over settled trades (gold line + area fill).
- **Calibration panel** — per-analyst Brier scores (green ≤ 0.20, amber ≤ 0.33, red > 0.33) + skipped-reason breakdown.
- **Recent trades table** — last 25 audit entries with timestamp, market, side, stake, P&L, status badge (WON / LOST / OPEN / SKIP / PAPER), and signature indicator (✓ sig vs. ∘ stub).
- **Achievements grid** — every achievement as a card. Locked = grayscale; unlocked = gold border + glow.
- **Trade detail drawer** — click any row to slide in the full decision chain: all analyst estimates (with rationale + evidence), the consensus decision JSON, the risk plan, the trade receipt, the settlement, and the signature.

The dashboard auto-refreshes every 15 seconds — judges can watch trades land live during a competition.

![Dashboard hero](docs/assets/replay-hero.png)
![Trade detail drawer](docs/assets/replay-drawer.png)
![Achievements grid](docs/assets/replay-achievements.png)

> Screenshots are placeholders — run `pythia-replay serve` against `tests/fixtures/sample_audit.jsonl` to see the real thing.

### 3. Evaluate achievements

```bash
pythia-replay achievements \
    --log ./logs/audit.jsonl \
    --config configs/achievements.toml
```

Prints the achievement list with `unlocked_at` populated for any condition met:

```json
[
  { "id": "first_trade", "name": "First Blood", "unlocked_at": "2026-01-15T10:23:11Z", "unlocked_value": 7 },
  { "id": "ten_trades",  "name": "Warming Up", "unlocked_at": null, "unlocked_value": 7 },
  ...
]
```

### 4. Export the full audit log

```bash
pythia-replay export --log ./logs/audit.jsonl --out ./logs/export.json
```

Writes the entire audit log as a single pretty-printed JSON array (useful for sharing with judges who want the raw data).

---

## Library API

```python
from pythia_observability import (
    AuditLogReader, ReplayServer, AchievementsEvaluator, AuditEntry
)
from pathlib import Path

# 1. Read + slice the audit log.
reader = AuditLogReader(Path("./logs/audit.jsonl"))
for entry in reader.iter_entries():      # lazy — streams one line at a time
    print(entry.market_id, entry.timestamp, entry.is_executed)

# 2. Compute analytics.
stats = reader.compute_stats()
print(stats["win_rate"], stats["total_realized_pnl_usd"])

pnl_curve = reader.compute_pnl_series()  # list[PnLMilestone] for charting
brier = reader.compute_brier_scores()    # dict[analyst_id, float]

# 3. Evaluate achievements.
evaluator = AchievementsEvaluator(Path("./configs/achievements.toml"))
for ach in evaluator.evaluate(reader):
    print(f"{ach.name}: {'✓' if ach.unlocked_at else '✗'}  (value={ach.unlocked_value})")

# 4. Launch the dashboard.
server = ReplayServer(
    log_path=Path("./logs/audit.jsonl"),
    achievements_config_path=Path("./configs/achievements.toml"),
)
server.run(host="127.0.0.1", port=8088)
```

---

## Audit-entry contract

The `AuditEntry` model is the single contract this wrapper assumes. Every JSONL line is one `AuditEntry` covering the **full decision chain** for one market cycle:

```python
class AuditEntry(BaseModel):
    timestamp: str                # ISO-8601 UTC
    market_id: str
    estimates: list[dict]         # analyst Estimate dicts
    decision: dict                # ConsensusDecision dict
    plan: dict                    # TradePlan dict (from pythia-risk)
    receipt: dict | None          # TradeReceipt dict, or None if skipped
    skipped_reason: str | None    # why this cycle was gated out, if applicable
    signature: str | None         # Ed25519 hex, or "stub:sha256:<hex>" fallback
```

Convenience properties: `is_executed`, `is_skipped`, `is_paper`, `realized_pnl_usd`, `won`, `category` — these are computed from the underlying dicts so consumers don't have to re-implement the field lookups.

### `# VERIFY:` items

The following upstream-API uncertainties are marked with `# VERIFY:` comments in the source (search the codebase to find them):

1. **Per-cycle vs. per-stage records.** We assume one JSONL line per market cycle (estimates + decision + plan + receipt all inline). If the upstream emits one line per pipeline stage (keyed by `market_id` + `stage`), an adapter needs to be added to `iter_entries()`.
2. **Signature shape.** We model `signature` as an opaque string. If the upstream uses a structured object (`{algorithm, key_fingerprint, sig}`), `AuditEntry.signature` becomes a `dict`.
3. **Paper-mode tagging.** We currently detect paper trades via `receipt.mode == "paper"` OR `signature.startswith("paper:")`. If the upstream uses a different convention, `AuditEntry.is_paper` needs updating.
4. **Settlement shape.** We read `receipt.settlement.outcome` ("YES"/"NO"/numeric 0..1) and `receipt.settlement.realized_pnl_usd`. If the upstream uses different field names, the `realized_pnl_usd` and `compute_brier_scores()` paths need updating.
5. **Bankroll annotation.** `compute_pnl_series()` reads `plan.bankroll_before` for the initial bankroll, falling back to $1000 if absent. Pythia's `pythia-risk` doesn't currently stamp this — we may want to add it.
6. **Achievements emission API.** `AchievementsEvaluator.evaluate()` returns the unlocked list; the executor is responsible for forwarding to the upstream's badge-emission endpoint (webhook / event bus / API call). The exact emission shape is pending upstream docs.
7. **Time-windowed drawdown.** `eval_drawdown_pct` currently checks the *current* drawdown, ignoring the `days` field on the condition. A proper time-windowed max-drawdown check needs timestamped bankroll snapshots — tracked as a `# VERIFY:` in `achievements.py`.

---

## Achievements system

Achievements are defined in a TOML file (copied from the parent monorepo's `configs/achievements.toml`):

```toml
[[achievement]]
id = "first_trade"
name = "First Blood"
description = "Place the first trade."
condition = { type = "trade_count", op = ">=", value = 1 }

[[achievement]]
id = "calibrated"
name = "Calibrated Oracle"
description = "Maintain Brier score < 0.20 across 20+ resolved trades."
condition = { type = "brier_score", op = "<=", value = 0.20, min_trades = 20 }

[[achievement]]
id = "niche_master"
name = "Niche Master"
description = "Win 3 subjective/niche markets."
condition = { type = "wins_in_category", op = ">=", value = 3, category = "subjective" }
```

### Supported condition types

| `type` | Checks | Optional fields |
|---|---|---|
| `trade_count` | Total audit entries (executed + skipped) | — |
| `win_count` | Settled winning trades | — |
| `win_streak` | Longest run of consecutive winning settled trades | — |
| `realized_pnl_usd` | Cumulative realized P&L | — |
| `brier_score` | Best (lowest) per-analyst Brier score | `min_trades` |
| `drawdown_pct` | Current drawdown from peak bankroll | `days` (advisory — see VERIFY #7) |
| `wins_in_category` | Winning trades in a specific market category | `category` |

Each condition is a standalone function (`eval_trade_count`, `eval_win_count`, …) dispatched via the `EVALUATORS` table. To add a new condition type: drop in a new function with the `(stats, condition) -> (unlocked, value)` signature and add it to the table. No other code changes needed.

---

## Project layout

```
packages/observability/
├── README.md                              ← this file
├── LICENSE                                ← MIT, icohangar-ops / Impactquadrant 2026
├── .gitignore
├── pyproject.toml                         ← hatchling, deps, entry point
├── VENDOR_COMMIT.txt                      ← pin upstream commits once vendored
├── src/
│   └── pythia_observability/
│       ├── __init__.py                    ← public API exports
│       ├── types.py                       ← AuditEntry / PnLMilestone / Achievement / AchievementCondition
│       ├── audit_reader.py                ← AuditLogReader (JSONL parse + analytics)
│       ├── achievements.py                ← AchievementsEvaluator + per-condition functions
│       ├── server.py                      ← ReplayServer (FastAPI app + endpoints + run())
│       ├── cli.py                         ← pythia-replay CLI (serve / stats / achievements / export)
│       └── templates/
│           └── dashboard.html             ← Jinja2 template, dark navy + gold, vanilla JS
└── tests/
    ├── __init__.py
    ├── test_audit_reader.py               ← JSONL parse + slicing + P&L/Brier/stats
    ├── test_achievements.py               ← each condition type + lock/unlock state
    └── fixtures/
        ├── sample_audit.jsonl             ← 8 example entries (win / loss / skip / paper / settled)
        └── sample_achievements.toml       ← copy of the parent repo's achievements.toml
```

---

## HTTP API

| Endpoint | Returns | Notes |
|---|---|---|
| `GET /` | `text/html` | Dashboard page (Jinja2 template). |
| `GET /api/stats` | `application/json` | Aggregate stats dict (see `compute_stats()`). |
| `GET /api/trades?limit=25&offset=0` | `application/json` | Paginated audit entries, newest first. |
| `GET /api/trades/{market_id}` | `application/json` | Full decision chain for one market. |
| `GET /api/pnl-series` | `application/json` | `list[PnLMilestone]` for the P&L chart. |
| `GET /api/achievements` | `application/json` | `list[Achievement]` with `unlocked_at` populated. |

All endpoints are read-only and stateless — they re-read the audit log file on every request so the dashboard picks up newly appended entries without manual cache invalidation.

---

## Testing

```bash
pytest -v
```

Tests use the fixtures in `tests/fixtures/`:

- `sample_audit.jsonl` — 8 example audit entries covering: a winning YES trade, a winning NO trade, a losing trade, a skipped trade (agreement below threshold), a paper trade, a settled-but-not-yet-resolved market, a niche-category win, and a stub-signed entry.
- `sample_achievements.toml` — copy of the parent monorepo's `configs/achievements.toml` (9 achievements across all 7 condition types).

The audit-reader tests cover: lazy iteration, eager read, market slicing, time filtering, P&L series computation, Brier score computation, and aggregate stats. The achievements tests cover: each of the 7 condition types with mock stats, lock/unlock state, and the `min_trades` gate on Brier.

---

## Design notes

- **Why read-only?** The audit log is the *evidence* that Pythia's trades were legitimate. The replay UI must not be able to mutate it — otherwise judges couldn't trust what they see. All writes happen in the executor pipeline (signed + persisted before any HTTP request returns).
- **Why JSONL and not SQLite?** Append-only JSONL is trivially auditable (judges can `tail -f` it), works on any filesystem, and is small enough for a 2-week competition run (a few hundred entries). SQLite would be overkill and harder to inspect by hand.
- **Why a fresh `AuditLogReader` per request?** So the dashboard picks up newly appended entries without manual cache invalidation. The audit log is small; the cost is negligible. (For production-scale use, an mtime-based cache would be the next step.)
- **Why vanilla JS in the dashboard?** No build step, no Node.js dependency, no framework churn. The dashboard is a single HTML file — judges can `view-source` and read every line. The chart is a hand-rolled canvas (no Chart.js dep).
- **Why dark navy + gold?** Oracle / Delphi / Pythia theme — the same palette as the parent monorepo's design PDF. Serif headings (Iowan Old Style / Palatino / Georgia) for gravitas; mono for the numbers (JetBrains Mono / Fira Code).

---

## License

MIT — © 2026 icohangar-ops / Impactquadrant. See [LICENSE](LICENSE).
