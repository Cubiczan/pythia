# pythia-consensus

> Delphi-specific consensus + audit layer that wraps
> [`icohangar-ops/consensus-hardening-protocol`](https://github.com/icohangar-ops/consensus-hardening-protocol)
> for the **Pythia** multi-agent trading mesh.

`pythia-consensus` is the *integration layer* the Pythia mesh talks to. It fuses
probability estimates from N specialist analysts into a single `ConsensusDecision`,
scores how much the analysts actually agree, gates the trade (`trade` / `skip` /
`wait`), and — when the upstream hardening protocol is vendored — produces a
signed, replayable audit record per decision.

It is **not** a reimplementation of the upstream protocol. The
`consensus-hardening-protocol` repo provides the hardened, human-auditable
multi-agent decision primitives (signature scheme, signed-decision envelope,
verifier). This wrapper:

1. Adds the **Delphi-specific fusion math** (logit-mean / median / trimmed-mean
   of `Estimate.probability`).
2. Adds the **agreement gate** used by the Pythia executor
   (`agreement_score < threshold ⇒ skip`).
3. Adds a **weight-update hook** that consumes per-analyst Brier scores from
   `pythia-forge` backtests.
4. Exposes a **clean Python API + CLI** so the rest of the Pythia mesh never
   has to know the upstream's internal types.
5. **Falls back gracefully** to a local stub signer when the upstream isn't
   vendored, so the repo is usable in CI without the (private) upstream.

---

## What this adds over the upstream

| Concern | Upstream `consensus-hardening-protocol` | This wrapper (`pythia-consensus`) |
|---|---|---|
| Signed decision envelope | ✅ | re-exported, optional |
| Verifier / replay | ✅ | re-exported, optional |
| **Fusion math** (logit-mean, median, trimmed-mean) | ❌ | ✅ |
| **Agreement score** (weighted stddev → 0..1) | ❌ | ✅ |
| **Trade / skip / wait gate** | ❌ | ✅ |
| **Brier-driven weight updates** | ❌ | ✅ |
| Delphi `Estimate` type interop | ❌ | ✅ (imports from `pythia_analyst_mesh`) |
| CLI for manual testing | ❌ | ✅ `pythia-consensus fuse <file>` |

The upstream is treated as a **black box**. If its public API changes, only
`audit.py` needs to be updated — the fusion + gate logic is fully self-contained
and tested without it.

---

## Install

### From source (recommended for development)

```bash
git clone https://github.com/icohangar-ops/pythia.git
cd pythia/packages/consensus
cd packages/consensus
pip install -e ".[dev]"
```

### Vendoring the upstream (required for signed audit records)

The fusion + gate logic works **without** the upstream. To enable signed audit
records, vendor the upstream into `vendor/`:

```bash
# Option 1 — nested submodule:
git submodule add git@github.com:icohangar-ops/consensus-hardening-protocol.git \
  vendor/consensus-hardening-protocol

# Option 2 — plain copy:
cp -r ../consensus-hardening-protocol vendor/consensus-hardening-protocol
```

Then pin the commit SHA in [`VENDOR_COMMIT.txt`](./VENDOR_COMMIT.txt) so the
audit trail is reproducible. If `vendor/` is present and importable as
`consensus_hardening_protocol`, `AuditSigner` will use it; otherwise it falls
back to a local SHA-256 stub (clearly marked in the audit log).

---

## Quick start

```python
from datetime import datetime, timezone
from pythia_consensus import ConsensusConfig, ConsensusEngine, fuse
from pythia_consensus.types import Estimate

estimates = [
    Estimate(
        market_id="delphi-2026-best-album",
        probability=0.62,
        confidence=0.7,
        rationale="Critics lean toward Artist X.",
        evidence=["https://example.com/critics-poll"],
        analyst_id="music-analyst",
        timestamp=datetime.now(timezone.utc).isoformat(),
    ),
    Estimate(
        market_id="delphi-2026-best-album",
        probability=0.58,
        confidence=0.6,
        rationale="Streaming numbers favor Artist X.",
        evidence=["https://example.com/streaming"],
        analyst_id="data-analyst",
        timestamp=datetime.now(timezone.utc).isoformat(),
    ),
]

cfg = ConsensusConfig(method="logit-mean", agreement_threshold=0.65, min_analysts=2)
decision = fuse(estimates, cfg)

print(decision.gate)             # "trade"
print(decision.consensus_prob)   # ~0.60
print(decision.agreement_score)  # ~0.998

# Or via the engine (keeps internal weights, can be updated from Brier scores):
engine = ConsensusEngine(cfg)
decision = engine.decide(estimates)
print(engine.explain(decision))
```

### CLI

```bash
# Fuse estimates from a JSON file (list of Estimate dicts):
pythia-consensus fuse estimates.json

# Print a human-readable explanation of a saved decision:
pythia-consensus explain decision.json
```

Example `estimates.json`:

```json
[
  {
    "market_id": "delphi-2026-best-album",
    "probability": 0.62,
    "confidence": 0.7,
    "rationale": "Critics lean toward Artist X.",
    "evidence": ["https://example.com/critics-poll"],
    "analyst_id": "music-analyst",
    "timestamp": "2026-01-15T12:00:00+00:00"
  },
  {
    "market_id": "delphi-2026-best-album",
    "probability": 0.58,
    "confidence": 0.6,
    "rationale": "Streaming numbers favor Artist X.",
    "evidence": ["https://example.com/streaming"],
    "analyst_id": "data-analyst",
    "timestamp": "2026-01-15T12:01:00+00:00"
  }
]
```

---

## Fusion methods

Given estimates with probabilities `p_i` and weights `w_i` (normalised to sum
to 1):

### `logit-mean` (default)

Fuse in **logit space** so extreme probabilities don't get washed out by the
mean:

```
logit(p) = ln(p / (1 - p))                   # p clamped to [0.01, 0.99]
L* = Σ w_i · logit(p_i)
p* = sigmoid(L*) = 1 / (1 + exp(-L*))
```

Use this when you care about **extreme probabilities** (e.g. a market where
three analysts say 0.97 and one says 0.5 — arithmetic mean = 0.85, logit-mean
≈ 0.93, which better reflects the confident majority).

### `median`

Weighted median of the `p_i`. Robust to a single outlier.

### `trimmed-mean`

Drop the highest and lowest probability, weighted-mean the rest. Requires at
least 3 analysts; falls back to plain weighted mean for `n < 3`.

## Agreement score

```
σ_w = sqrt( Σ w_i · (p_i - p̄_w)² )         # weighted population stddev
agreement_score = clamp(1 - σ_w / 0.5, 0, 1)
```

- All analysts identical ⇒ `σ_w = 0` ⇒ `agreement_score = 1.0` (perfect agreement).
- Analysts maximally split (some at 0, some at 1) ⇒ `σ_w ≈ 0.5` ⇒
  `agreement_score = 0.0` (maximal disagreement).

## Gate logic

```
if len(estimates) < min_analysts:                     gate = "wait"
elif agreement_score < agreement_threshold:           gate = "skip"
else:                                                 gate = "trade"
```

- **wait** — not enough analysts yet; the mesh should poll again.
- **skip** — too much disagreement; no edge, don't trade.
- **trade** — consensus reached; pass to `pythia-risk` for sizing.

## Weight updates from Brier scores

`ConsensusEngine.update_weights(brier)` accepts per-analyst Brier scores
(lower = better) from `pythia-forge` backtests and converts them to weights:

```
w_i = softmax(-α · brier_i)            # α = 4.0 by default
```

Lower Brier ⇒ exponentially higher weight. Weights are normalised and stored
on the engine; the next `decide()` call uses them as `weights_used`.

---

## Audit / signing

`audit.AuditSigner.sign(decision)` returns a signature string.

- If `consensus_hardening_protocol.SignedDecision` is importable (upstream
  vendored), the signer delegates to it. **# VERIFY:** the exact upstream
  constructor signature needs to be confirmed once the upstream is vendored.
- Otherwise, a deterministic SHA-256 over the canonical JSON of the decision
  is returned, prefixed with `stub:` so downstream consumers can tell the
  audit record is not cryptographically signed.

---

## Module map

```
src/pythia_consensus/
├── __init__.py     # public API: ConsensusEngine, ConsensusDecision, ConsensusConfig, fuse, agreement_score
├── types.py        # Estimate (re-exported), ConsensusConfig, ConsensusDecision, ConsensusMethod
├── fusion.py       # fuse() + agreement_score() — the core math
├── engine.py       # ConsensusEngine — stateful wrapper around fuse()
├── audit.py        # AuditSigner — integrates with upstream SignedDecision
└── cli.py          # `pythia-consensus fuse|explain` entrypoint
```

## Testing

```bash
pytest -q
```

All fusion math, gate logic, and weight-update behaviour is unit-tested in
`tests/test_fusion.py` and `tests/test_engine.py`. Tests run **without** the
upstream vendored (they exercise the stub signer path).

## License

MIT — see [`LICENSE`](./LICENSE). Copyright © 2026 icohangar-ops / Impactquadrant.

## Upstream attribution

This wrapper depends conceptually on
[`icohangar-ops/consensus-hardening-protocol`](https://github.com/icohangar-ops/consensus-hardening-protocol).
The fusion, agreement, and gate logic in `fusion.py` is original to Pythia; the
audit envelope is delegated to the upstream when available.
