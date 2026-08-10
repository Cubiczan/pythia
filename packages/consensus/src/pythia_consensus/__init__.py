"""pythia-consensus — Delphi-specific consensus + audit layer.

Wraps `icohangar-ops/consensus-hardening-protocol` for the Pythia mesh.

Public API
----------
- `ConsensusEngine`    — stateful wrapper around `fuse()` with weight updates.
- `ConsensusDecision`  — pydantic model, the fused output of one consensus round.
- `ConsensusConfig`    — pydantic model, config for fusion + gate.
- `fuse`               — pure function: estimates + config → decision.
- `agreement_score`    — pure function: how aligned a set of estimates is (0..1).

See `pythia_consensus.fusion` for the math and `pythia_consensus.engine` for
the stateful wrapper.
"""

from __future__ import annotations

from .audit import AuditSigner
from .engine import ConsensusEngine
from .fusion import agreement_score, fuse
from .types import ConsensusConfig, ConsensusDecision, ConsensusMethod, Estimate

__version__ = "0.1.0"

__all__ = [
    "AuditSigner",
    "ConsensusConfig",
    "ConsensusDecision",
    "ConsensusEngine",
    "ConsensusMethod",
    "Estimate",
    "agreement_score",
    "fuse",
    "__version__",
]
