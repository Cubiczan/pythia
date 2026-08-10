"""Type re-exports and Pythia-executor-specific config / result models.

This module is the single import surface for everything that flows through
the executor. The four data contracts that come from the sibling packages
(``TradePlan``, ``ConsensusDecision``, ``TradeReceipt``) are re-exported
here so callers have one place to look. ``ExecutorConfig`` and
``PipelineResult`` are local to this repo.

Re-exports use the canonical sibling-package types when those packages are
installed. They are *not* wrapped in try/except — the executor is the top
of the stack and cannot function without its siblings. If you want to test
the executor in isolation, install the siblings in editable mode (they're
all in the same monorepo).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Estimate comes from pythia-analyst-mesh (carried through consensus).
from pythia_analyst_mesh import Estimate

# ConsensusDecision comes from pythia-consensus.
from pythia_consensus import ConsensusDecision

# TradeReceipt comes from pythia-delphi-adapter (it's the POST /orders response).
from pythia_delphi_adapter import TradeReceipt

# ---------------------------------------------------------------------------
# Sibling-package re-exports (single source of truth for the executor).
# ---------------------------------------------------------------------------
# TradePlan comes from pythia-risk (it's the output of RiskEngine.evaluate).
from pythia_risk import TradePlan

# ---------------------------------------------------------------------------
# ExecutorConfig
# ---------------------------------------------------------------------------

ExecutorMode = Literal["paper", "live"]
"""Executor operating mode.

``paper``: synthesise a ``TradeReceipt`` with ``status="PAPER"`` and **do
not** submit. Safe default for development + demos.

``live``: sign the order with the signing key, submit it via
``DelphiClient.place_order``, get a real ``TradeReceipt`` back. Requires
``DELPHI_SIGNING_KEY`` (or whatever env var ``signing_key_env`` names) to
be set.
"""

class ExecutorConfig(BaseModel):
    """Configuration for the ``PythiaExecutor``.

    Mirrors the ``[executor]`` section of ``configs/live-mvp.toml``:

        [executor]
        mode = "paper"
        signing_key_env = "DELPHI_SIGNING_KEY"
        idempotency_enabled = true
        retry_max = 3
        retry_backoff_sec = 5
    """

    model_config = ConfigDict(extra="forbid")

    mode: ExecutorMode = "paper"
    signing_key_env: str = "DELPHI_SIGNING_KEY"
    idempotency_enabled: bool = True
    retry_max: int = Field(default=3, ge=1, le=20)
    retry_backoff_sec: int = Field(default=5, ge=0, le=600)

# ---------------------------------------------------------------------------
# PipelineResult — the full output of one run_for_market() call.
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """The full decision chain for one market.

    Every field is a pydantic model, so the entire result serialises to a
    single JSON line in the audit log. Fields are populated progressively
    as the pipeline advances; the ``skipped_reason`` is ``None`` iff the
    pipeline reached submission (paper or live) and produced a receipt.

    The ``decision`` and ``plan`` fields are optional in the *type system*
    (the pipeline may skip before reaching them), but the executor always
    sets them to a synthesised minimal object on skip paths so that the
    audit log is uniform.
    """

    model_config = ConfigDict(extra="forbid")

    market_id: str
    estimates: list[Estimate] = Field(default_factory=list)
    decision: ConsensusDecision | None = None
    plan: TradePlan | None = None
    receipt: TradeReceipt | None = None
    skipped_reason: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

__all__ = [
    "ConsensusDecision",
    "Estimate",
    "ExecutorConfig",
    "ExecutorMode",
    "PipelineResult",
    "TradePlan",
    "TradeReceipt",
]
