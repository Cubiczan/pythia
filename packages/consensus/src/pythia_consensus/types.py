"""Shared types for the pythia-consensus wrapper.

`Estimate` is re-exported from `pythia_analyst_mesh.types` when that package is
available, so the whole Pythia mesh shares a single canonical definition of an
analyst's estimate. If `pythia_analyst_mesh` is not installed (e.g. in CI or
while this wrapper is being tested in isolation), a minimal local definition
is used instead — it is structurally identical and serialises the same way.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Estimate — re-exported from pythia_analyst_mesh if available, else local.
# ---------------------------------------------------------------------------
try:
    # Preferred path: the mesh owns the canonical Estimate type.
    from pythia_analyst_mesh.types import Estimate as Estimate  # noqa: F401
    ESTIMATE_SOURCE = "pythia_analyst_mesh"
except ImportError:  # pragma: no cover - exercised only when mesh isn't installed
    # Fallback: define a minimal Estimate so this wrapper is usable in isolation.
    from pydantic import Field as _Field

    class Estimate(BaseModel):  # type: ignore[no-redef]
        """Minimal local Estimate — structurally identical to the mesh's."""

        model_config = ConfigDict(extra="ignore")

        market_id: str = _Field(..., description="Delphi market identifier.")
        probability: float = _Field(..., ge=0.0, le=1.0, description="P(YES).")
        confidence: float = _Field(
            default=0.5, ge=0.0, le=1.0, description="Analyst's self-reported calibration."
        )
        rationale: str = _Field(default="", description="1-3 sentence justification.")
        evidence: list[str] = _Field(default_factory=list, description="Citations / URLs.")
        analyst_id: str = _Field(..., description="Which analyst produced this estimate.")
        timestamp: str = _Field(..., description="ISO-8601 timestamp.")

    ESTIMATE_SOURCE = "local-fallback"


# ---------------------------------------------------------------------------
# Consensus config + decision.
# ---------------------------------------------------------------------------
ConsensusMethod = Literal["logit-mean", "median", "trimmed-mean"]
"""Fusion method used by `fuse()`."""


Gate = Literal["trade", "skip", "wait"]
"""Executor-facing decision gate."""


class ConsensusConfig(BaseModel):
    """Configuration for the consensus fusion + gate.

    Attributes
    ----------
    method:
        Fusion algorithm. See `pythia_consensus.fusion` for the math.
    agreement_threshold:
        Minimum `agreement_score` required for `gate="trade"`. Below this, the
        gate flips to `"skip"`. Default 0.65 — empirically a good cutoff for
        Delphi's niche markets (tight enough to filter noisy disagreement, loose
        enough to allow trade when analysts converge).
    min_analysts:
        Minimum number of estimates required before a `trade` decision is
        possible. Below this, `gate="wait"`. Default 2.
    weights:
        Optional per-analyst weights. If `None` or missing an analyst, equal
        weight is used. The `ConsensusEngine` populates this from Brier scores
        via `update_weights`.
    """

    model_config = ConfigDict(extra="forbid")

    method: ConsensusMethod = "logit-mean"
    agreement_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_analysts: int = Field(default=2, ge=1)
    weights: Optional[dict[str, float]] = None


class ConsensusDecision(BaseModel):
    """The fused output of one consensus round.

    This is the contract the rest of the Pythia mesh consumes. It is a Pydantic
    v2 model so it serialises cleanly to JSON for the audit log.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str
    consensus_prob: float = Field(..., ge=0.0, le=1.0)
    agreement_score: float = Field(..., ge=0.0, le=1.0)
    gate: Gate
    contributor_ids: list[str]
    method: ConsensusMethod
    weights_used: dict[str, float]
    timestamp: str  # ISO-8601


__all__ = [
    "ConsensusConfig",
    "ConsensusDecision",
    "ConsensusMethod",
    "Estimate",
    "Gate",
    "ESTIMATE_SOURCE",
]
