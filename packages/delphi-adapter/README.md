# pythia-delphi-adapter

### ATT client, market metadata, settlement listener, and config schema for [Pythia](https://github.com/icohangar-ops/pythia) trading on Gensyn Delphi.

> This is the **new** Pythia module — it does *not* wrap an existing `icohangar-ops` repo. It is the thin Python layer that turns Delphi's [Agentic Trading Toolkit (ATT)](https://docs.gensyn.ai/tech/agentic-trading) into typed, async, retry-safe calls the rest of the Pythia mesh can consume.

---

## What it does

`pythia-delphi-adapter` exposes a single `DelphiClient` that wraps every ATT HTTP/WebSocket endpoint an agent needs to participate in a Delphi information market:

| Method | ATT endpoint | Purpose |
|---|---|---|
| `list_markets` | `GET /markets` | Browse open / closed / settled markets, filter by category |
| `get_market` | `GET /markets/{id}` | Pull full metadata for one market (incl. AI arbiter model) |
| `get_orderbook` | `GET /markets/{id}/orderbook` | Read bids / asks before quoting |
| `place_order` | `POST /orders` | Submit a YES/NO order with an idempotency key |
| `cancel_order` | `DELETE /orders/{id}` | Pull a resting order |
| `get_positions` | `GET /positions` | Inspect what the agent currently holds |
| `get_settlements` | `GET /settlements` | Poll for AI-as-arbiter resolutions |
| `subscribe_events` | `wss://.../events` | Stream `MarketOpened` / `PriceUpdated` / `OrderMatched` / `MarketSettled` |

The adapter also ships:

- **Pydantic v2 models** for every ATT response shape (markets, orderbooks, trade receipts, positions, settlements, market events). Where the live ATT response shape is not yet confirmed from public docs, the model is annotated with `# VERIFY:` and a sensible default is assumed.
- **`SettlementListener`** — a long-running async task that polls `/settlements` on a configurable cadence and pushes resolved markets into an `asyncio.Queue`, so `pythia-risk` and `pythia-observability` can update P&L and unlock achievements in real time.
- **`DelphiConfig`** — loads `api_key`, `endpoint`, `poll_interval_sec` from env vars or a TOML file (see `configs/live-mvp.toml` in the parent repo).
- **`pythia-delphi` CLI** — for one-off inspection and paper-order previews without writing code.

---

## Why a separate adapter (and not just call ATT from the executor)?

Three reasons:

1. **Typed boundary.** Every other Pythia module depends on `pythia-delphi-adapter`'s pydantic models, never on raw ATT JSON. If Gensyn ships a v2 ATT, only this repo changes.
2. **Honest stubs.** ATT docs are public but the exact response field names for some endpoints (e.g. `arbiter_model`, `evidence_hashes`) are still being confirmed. Centralizing those assumptions here makes them easy to audit and fix.
3. **Retry + idempotency by default.** Every mutating call goes through `tenacity` retry with exponential backoff, and `place_order` always sends an `Idempotency-Key` header so the agent can safely retry a timeout without double-spending.

---

## Install

```bash
# from source (recommended for the hackathon)
git clone https://github.com/icohangar-ops/pythia.git
cd pythia/packages/delphi-adapter
cd packages/delphi-adapter
pip install -e .

# or, once published
pip install pythia-delphi-adapter
```

Requires Python ≥ 3.11.

### Runtime dependencies

- `httpx>=0.27` — async HTTP + WebSocket client
- `pydantic>=2.0` — typed response models
- `tenacity>=8.0` — retry / backoff
- `websockets>=12.0` — WebSocket event stream
- `tomli>=2.0; python_version < '3.11'` — TOML config parsing on older interpreters

---

## Configure

The adapter reads configuration from environment variables by default and optionally merges in a TOML file:

```bash
export DELPHI_API_KEY="dphi_live_..."
export DELPHI_ENDPOINT="https://api.delphi.gensyn.ai"   # optional, defaults shown
export DELPHI_POLL_INTERVAL_SEC="60"                     # optional
```

Or via TOML (path passed to `load_config`):

```toml
# delphi.toml
api_key_env = "DELPHI_API_KEY"
endpoint    = "https://api.delphi.gensyn.ai"
poll_interval_sec = 60
```

```python
from pythia_delphi_adapter.config import load_config

cfg = load_config(env_var="DELPHI_API_KEY", toml_path="delphi.toml")
```

---

## Usage

### List open crypto markets

```python
import asyncio
from pythia_delphi_adapter import DelphiClient, MarketStatus

async def main() -> None:
    async with DelphiClient(api_key="dphi_live_...") as client:
        markets = await client.list_markets(
            status=MarketStatus.OPEN,
            category="CRYPTO",
            limit=50,
        )
        for m in markets:
            print(f"{m.market_id}  yes={m.yes_price:.2f}  vol=${m.volume_usd:,.0f}")

asyncio.run(main())
```

### Place a paper-style order (with idempotency)

```python
import uuid
from pythia_delphi_adapter import DelphiClient, OrderSide

async with DelphiClient(api_key="...") as client:
    receipt = await client.place_order(
        market_id="dphi_01J...",
        side=OrderSide.YES,
        size_usd=25.0,
        limit_price=0.62,
        correlation_id=str(uuid.uuid4()),
    )
    print(receipt.att_order_id, receipt.status)
```

### Stream market events

```python
async with DelphiClient(api_key="...") as client:
    async for event in client.subscribe_events(market_id="dphi_01J..."):
        print(type(event).__name__, event)
```

### Run the settlement listener in the background

```python
import asyncio
from pythia_delphi_adapter import DelphiClient
from pythia_delphi_adapter.config import load_config
from pythia_delphi_adapter.settlement_listener import SettlementListener

async def on_settlement(s) -> None:
    print(f"SETTLED {s.market_id} → {s.outcome} via {s.arbiter_model}")

async def main() -> None:
    cfg = load_config()
    async with DelphiClient(api_key=cfg.api_key, endpoint=cfg.endpoint) as client:
        listener = SettlementListener(client=client, poll_interval_sec=cfg.poll_interval_sec)
        await listener.start(on_settlement)

asyncio.run(main())
```

---

## CLI

`pip install -e .` registers the `pythia-delphi` entry point:

```bash
# list 20 open markets
pythia-delphi markets list --limit 20

# show one market's metadata + orderbook
pythia-delphi markets get dphi_01J...

# current open positions
pythia-delphi positions

# recent AI-as-arbiter settlements
pythia-delphi settlements --since 24h

# preview what would be submitted (no POST)
pythia-delphi paper-order dphi_01J... yes 25.0 --limit-price 0.62
```

`pythia-delphi paper-order` constructs the exact payload, prints it, and exits — useful for sanity-checking before the executor submits live.

---

## Project layout

```
packages/delphi-adapter/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── VENDOR_COMMIT.txt
├── src/
│   └── pythia_delphi_adapter/
│       ├── __init__.py
│       ├── client.py              # DelphiClient — ATT HTTP/WS wrapper
│       ├── models.py              # pydantic v2 models for ATT responses
│       ├── settlement_listener.py # async polling loop → asyncio.Queue
│       ├── config.py              # env + TOML config loader
│       └── cli.py                 # `pythia-delphi` argparse entry point
└── tests/
    ├── __init__.py
    ├── test_client.py             # httpx.MockTransport smoke tests
    └── test_models.py             # pydantic validation tests
```

---

## ATT-specific assumptions (read before going live)

This is a scaffold. The ATT HTTP/WebSocket API surface is inferred from [Gensyn's public docs](https://docs.gensyn.ai/tech/agentic-trading) and the `gensyn-ai/gensyn-delphi-skills` repo. The following items are marked `# VERIFY:` in the source and should be confirmed against the live ATT before trading real size:

1. **Base URL** — assumed `https://api.delphi.gensyn.ai`. Confirm against the ATT quickstart.
2. **Auth header** — assumed `Authorization: Bearer <api_key>`. May instead be `X-Delphi-Api-Key`.
3. **Idempotency header** — assumed `Idempotency-Key`. ATT's Python SDK may already do this internally; if so, the explicit header is harmless.
4. **`Market.arbiter_model`** — Delphi settles via AI-as-arbiter; the field name identifying *which* model arbitrated is assumed. Confirm shape.
5. **`Settlement.evidence_hashes`** — assumed to be a list of content hashes (the public, training-reusable artifacts of the arbiter). Confirm shape.
6. **WebSocket URL + event schema** — assumed `wss://api.delphi.gensyn.ai/events` with a discriminated `type` field. Confirm against ATT JS bindings.
7. **Pagination** — assumed `?limit=N&offset=K` query params. May be cursor-based.

Everything else is straightforward REST and should "just work" once the field names are confirmed.

---

## Testing

```bash
pip install -e ".[test]"
pytest -q
```

Tests use `httpx.MockTransport` — they never touch the real ATT. No API key is required.

---

## Relationship to `gensyn-ai/gensyn-delphi-skills`

That repo ships TypeScript/JS *skills* for agent frameworks. `pythia-delphi-adapter` is the **Python equivalent** — a typed async client that the rest of Pythia (which is Python-native) can call directly without a Node bridge. We do not vendor the skills repo; we re-implement the same ATT calls in Python. See [`VENDOR_COMMIT.txt`](./VENDOR_COMMIT.txt).

---

## License

MIT — see [`LICENSE`](./LICENSE). Copyright icohangar-ops / Impactquadrant 2026.
