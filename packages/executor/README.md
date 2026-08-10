# pythia-executor

**Trade-orchestration CLI for the Pythia mesh.** Wraps
[`icohangar-ops/metabocommand`](https://github.com/icohangar-ops/metabocommand)
and stitches the four sibling Pythia sub-repos into a single end-to-end
pipeline:

```
   Delphi market → analyst-mesh → consensus → risk → sign → ATT order → TradeReceipt
                                                                              │
                                                                              ▼
                                                                     signed JSONL audit log
```

This is the binary a Delphi Agent Arena operator actually runs. Everything
else in the monorepo (`pythia-delphi-adapter`, `pythia-analyst-mesh`,
`pythia-consensus`, `pythia-risk`) is a library; this repo turns those
libraries into a coherent, auditable, replayable trading bot.

---

## What it does

For each open Delphi market, the executor runs a fixed 12-step pipeline:

| Step | Action                                                              | Skips on failure? |
|------|---------------------------------------------------------------------|-------------------|
| 1    | Fetch `Market` from Delphi via `delphi_client.get_market`           | no (raises)       |
| 2    | Build `MarketContext` for the analyst mesh                          | no                |
| 3    | Run the analyst mesh concurrently → `list[Estimate]`                | no                |
| 4    | `len(estimates) < consensus.min_analysts` → `skipped_reason="insufficient_analysts"` | yes |
| 5    | `consensus_engine.decide(estimates)` → `ConsensusDecision`          | no                |
| 6    | `decision.gate != "trade"` → `skipped_reason=f"gate_{gate}"`        | yes               |
| 7    | `risk_engine.evaluate(decision, market, state)` → `TradePlan`       | no                |
| 8    | `plan.decision != "APPROVE"` → `skipped_reason=f"risk_rejected:{flags}"` | yes          |
| 9    | Paper mode: synthesise a `TradeReceipt` with `status="PAPER"`, **don't submit** | n/a       |
| 10   | Live mode: `_sign_order(plan)` + `delphi_client.place_order` → real receipt | no      |
| 11   | `_write_audit(result)` — append signed JSONL to `audit_log_path`    | no                |
| 12   | `risk_engine.update_state(receipt)` — keep bankroll in sync         | no                |

Every step that runs produces a fully-typed pydantic model, all of which are
serialized into the audit log. A trade can be replayed end-to-end from a
single log line.

---

## Why this is a wrapper around `metabocommand`

`icohangar-ops/metabocommand` already provides:

- a typed command-orchestration primitive (steps, contexts, results),
- an audit-trail helper that writes signed JSONL records,
- a signal-aware runner that handles `SIGINT` / `SIGTERM` gracefully.

We wrap it for two reasons:

1. **Domain typing.** `metabocommand` is generic; we surface the
   Pythia-specific contracts (`TradePlan`, `ConsensusDecision`,
   `TradeReceipt`, `PipelineResult`) as the public types.
2. **The `delphi` subcommand.** The mesh → consensus → risk → ATT pipeline
   is unique to Pythia and lives here, not in `metabocommand`.

While `metabocommand` is not yet vendored (see `VENDOR_COMMIT.txt`), the
executor implements its own thin equivalents of the audit + signal
primitives. Once vendored, the internals swap over and the public API stays
identical.

---

## Install

```bash
# From the pythia-executor directory, in a venv that already has the
# sibling packages (pythia-delphi-adapter, pythia-analyst-mesh,
# pythia-consensus, pythia-risk) installed in editable mode:
pip install -e ".[dev]"
```

If you're working inside the Pythia monorepo, install the siblings first:

```bash
pip install -e ../packages/delphi-adapter
pip install -e ../packages/analyst-mesh
pip install -e ../packages/consensus
pip install -e ../packages/risk
pip install -e ".[dev]"
```

The four sibling deps are listed in `pyproject.toml` so a future
`pip install pythia-executor` will pull them from PyPI automatically.

---

## Quickstart (paper mode)

The fastest end-to-end run is a single-market paper trade:

```bash
pythia executor delphi paper-trade \
    --market dphi_01JZ... \
    --analysts politics,crypto \
    --consensus-threshold 0.65 \
    --max-stake-usd 50
```

This:

1. fetches the market from Delphi (using `DELPHI_API_KEY` from env),
2. runs the politics + crypto analysts against it,
3. fuses their estimates with `consensus_engine.decide`,
4. sizes the trade through `pythia-risk`,
5. **does not** submit anything — synthesises a `TradeReceipt` with
   `status="PAPER"` instead,
6. writes the full `PipelineResult` (decision chain + receipt) to the
   audit log,
7. prints the result as JSON to stdout.

Paper mode is the default in `configs/live-mvp.toml` (`[executor] mode =
"paper"`) — you have to flip it to `"live"` and set `DELPHI_SIGNING_KEY`
explicitly before any real order is submitted.

---

## CLI subcommands

```
pythia executor delphi paper-trade --market <id> [--analysts ...] [--consensus-threshold 0.65] [--max-stake-usd 50]
pythia executor delphi run --config configs/live-mvp.toml [--risk-max-drawdown-pct 5] [--log-level info]
pythia executor delphi replay <audit_log_path> [--line N]
```

### `pythia executor delphi paper-trade`

Single-market, single-shot, no submission. Used for demos, smoke tests, and
judge-facing replays. All inputs are CLI flags (no TOML required).

| Flag                      | Default       | Notes                                              |
|---------------------------|---------------|----------------------------------------------------|
| `--market`                | required      | Delphi market id.                                  |
| `--analysts`              | `politics,crypto` | Comma-separated analyst slugs.                 |
| `--consensus-threshold`   | `0.65`        | `agreement_score` cutoff for `gate="trade"`.       |
| `--max-stake-usd`         | `50.0`        | Per-market cap; overrides TOML `[risk]`.           |
| `--audit-log`             | `./logs/audit.jsonl` | Where to append the JSONL record.            |
| `--llm-provider`          | env `LLM_PROVIDER` or `openai` | Forwarded to `LLMConfig`.        |
| `--llm-model`             | env `LLM_MODEL` or `gpt-4o-mini` | Forwarded to `LLMConfig`.        |

### `pythia executor delphi run`

Continuous loop. Reads `configs/live-mvp.toml`, polls Delphi for open
markets at `[delphi].poll_interval_sec`, and runs the full pipeline for
each market that hasn't been evaluated recently. Handles `SIGINT` /
`SIGTERM` gracefully (drains in-flight tasks, writes a final audit line,
exits 0).

| Flag                       | Default                  | Notes                                            |
|----------------------------|--------------------------|--------------------------------------------------|
| `--config`                 | `configs/live-mvp.toml`  | TOML config path.                                |
| `--risk-max-drawdown-pct`  | (from config)            | Overrides `[risk].max_drawdown_pct`.             |
| `--log-level`              | `info`                   | `debug` / `info` / `warning` / `error`.          |
| `--once`                   | off                      | Run one polling iteration and exit (smoke test). |

### `pythia executor delphi replay`

Re-reads an audit log line and pretty-prints the full decision chain:
market context → per-analyst estimates → consensus decision → risk plan →
trade receipt. Used by the demo / judge replay UI.

```bash
pythia executor delphi replay logs/audit.jsonl --line -1
```

`--line` is 1-indexed; `-1` means "last line".

---

## Library API

```python
import asyncio
from pythia_executor import PythiaExecutor, ExecutorConfig, run_pipeline

# run_pipeline() builds a default executor from env + config and runs one
# market — convenience wrapper used by the CLI.
result = asyncio.run(run_pipeline(market_id="dphi_01JZ...", mode="paper"))
```

### `PythiaExecutor`

The orchestrator. Constructed with the four sibling components plus a
config and an audit-log path:

```python
from pathlib import Path
from pythia_executor import PythiaExecutor, ExecutorConfig

executor = PythiaExecutor(
    delphi_client=delphi_client,
    mesh=[politics_analyst, crypto_analyst],
    consensus_engine=consensus_engine,
    risk_engine=risk_engine,
    config=ExecutorConfig(mode="paper", signing_key_env="DELPHI_SIGNING_KEY",
                          idempotency_enabled=True, retry_max=3, retry_backoff_sec=5),
    audit_log_path=Path("./logs/audit.jsonl"),
)

result = await executor.run_for_market("dphi_01JZ...")
```

### `ExecutorConfig`

| Field                  | Type                              | Notes                                       |
|-----------------------|-----------------------------------|---------------------------------------------|
| `mode`                | `Literal["paper", "live"]`        | `paper` synthesises a receipt; `live` signs + submits. |
| `signing_key_env`     | `str`                             | Env var name holding the Ed25519 signing key. |
| `idempotency_enabled` | `bool`                            | Forward `Idempotency-Key` header on `place_order`. |
| `retry_max`           | `int`                             | Max attempts on retryable ATT errors.       |
| `retry_backoff_sec`   | `int`                             | Base for exponential backoff.               |

### `PipelineResult`

The full output of one `run_for_market` call. All fields are pydantic
models, so the entire result serialises to a single JSON line in the audit
log:

| Field            | Type                            | Notes                                            |
|------------------|---------------------------------|--------------------------------------------------|
| `market_id`      | `str`                           | The market this run was for.                     |
| `estimates`      | `list[Estimate]`                | Per-analyst estimates (may be `[]` if mesh failed). |
| `decision`       | `ConsensusDecision`             | Fused consensus output.                         |
| `plan`           | `TradePlan`                     | Risk-engine output (always set, even on REJECT).|
| `receipt`        | `TradeReceipt \| None`          | `None` if we skipped before submission.         |
| `skipped_reason` | `str \| None`                   | `None` if we submitted (paper or live).         |
| `timestamp`      | `str`                           | ISO 8601 UTC.                                   |

---

## Signing

In `live` mode, every order is signed before submission. The signing key
is loaded from the env var named by `ExecutorConfig.signing_key_env`
(default `DELPHI_SIGNING_KEY`) — never from a file, never logged.

The exact ATT signing scheme is not yet documented upstream. The current
implementation:

- prefers Ed25519 if `cryptography` is installed and the env var holds a
  base64-url Ed25519 private key,
- falls back to HMAC-SHA256 over the canonical JSON of the order body
  (key = the env var value, treated as UTF-8) when `cryptography` is not
  available or the key doesn't decode as Ed25519.

Both paths are marked `# VERIFY:` in `pipeline.py`. Once the ATT docs
publish the canonical scheme, swap the body of `_sign_order` and the
public API stays the same.

The `signed_by` field on the resulting `TradeReceipt` is a *fingerprint*
of the key (SHA-256 of the public key, first 16 hex chars) — never the key
itself.

---

## Audit log format

Each line is a single JSON object — a serialised `PipelineResult` plus the
SHA-256 signature of that JSON (computed with the signing key, or a
deterministic HMAC fallback in paper mode):

```json
{"market_id":"dphi_01JZ...","estimates":[...],"decision":{...},"plan":{...},
 "receipt":{...},"skipped_reason":null,"timestamp":"2026-02-14T12:00:00Z",
 "signature":"...","signature_algo":"ed25519"}
```

The log is append-only. The `replay` subcommand reads it line-by-line and
reconstructs the full decision chain for any historical trade.

---

## Failure modes & mitigations

| Failure                                       | Mitigation                                                              |
|-----------------------------------------------|-------------------------------------------------------------------------|
| All analysts time out / raise                 | `run_mesh` returns `[]`; pipeline skips with `insufficient_analysts`.   |
| Consensus can't reach quorum                  | `gate != "trade"`; pipeline skips with `gate_skip` / `gate_wait`.      |
| Risk engine rejects (drawdown, cool-down, ...) | Pipeline skips with `risk_rejected:<flags>` — no order submitted.      |
| ATT API 5xx mid-`place_order`                 | `DelphiClient` retries with `Idempotency-Key`; receipt has `status=PENDING`. |
| Signing key not in env                        | Pipeline raises `ValueError` before any network call.                  |
| SIGINT / SIGTERM mid-loop                     | `loop.run_loop` drains in-flight `run_for_market` calls, writes a final audit line, exits 0. |
| Audit log disk full                           | `_write_audit` raises; pipeline aborts (fail-closed — never trade without audit). |

---

## Project layout

```
packages/executor/
├── README.md                  ← this file
├── LICENSE                    ← MIT, icohangar-ops / Impactquadrant 2026
├── .gitignore
├── pyproject.toml             ← hatchling build, `pythia` entry point
├── VENDOR_COMMIT.txt          ← pin for icohangar-ops/metabocommand
├── src/pythia_executor/
│   ├── __init__.py            ← exports PythiaExecutor, ExecutorConfig, run_pipeline
│   ├── types.py               ← re-exports + ExecutorConfig + PipelineResult
│   ├── pipeline.py            ← PythiaExecutor + _sign_order + _write_audit
│   ├── loop.py                ← async run_loop() with signal handling
│   └── cli.py                 ← `pythia` argparse CLI
└── tests/
    ├── __init__.py
    ├── test_pipeline.py       ← 6 pipeline tests (paper/live/skips/audit)
    └── test_cli.py            ← CLI smoke tests
```

---

## Testing

```bash
pytest -v
```

The pipeline tests use stand-in doubles for `DelphiClient`, the mesh,
`ConsensusEngine`, and `RiskEngine` — no real LLM or ATT calls are made.
The CLI tests invoke `pythia --help` and `paper-trade --help` to confirm
the argparse tree compiles and parses.

---

## Status & next steps

- [x] 12-step pipeline implementation (paper + live).
- [x] Audit-log JSONL writer with signature.
- [x] CLI: `paper-trade`, `run`, `replay`.
- [x] Loop with `SIGINT` / `SIGTERM` handling.
- [x] Pipeline + CLI tests passing with mocked siblings.
- [ ] Vendor `icohangar-ops/metabocommand`, swap audit/signal internals.
- [ ] Confirm ATT signing scheme (resolve all `# VERIFY:` markers).
- [ ] Wire `replay` to the `pythia-observability` replay UI (port 8088).
- [ ] Add `--mode live` confirmation prompt to `paper-trade` (safety).

---

## License

MIT, icohangar-ops / Impactquadrant 2026. See [`LICENSE`](LICENSE).
